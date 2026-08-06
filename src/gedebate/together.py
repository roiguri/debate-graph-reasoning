"""Together.ai chat-completions backend, duck-typed to `gedebate.model.Model`.

Same contract the conditions already rely on -- `.generate(prompt, max_new_tokens=,
temperature=, top_p=, top_k=, seed=) -> GenResult` -- so `baseline`, `majority_vote`,
`debate` and the run loop need no changes. Selected with `provider = "together"` in a
run config.

Three things differ from the in-process HuggingFace path. All three are recorded in the
manifest rather than left implicit, because each one breaks a comparison that the
existing results rely on:

* **Token counts come from the server.** `usage.prompt_tokens` / `usage.completion_tokens`
  are the served model's own tokenizer, which is what matched compute has to be measured
  in. They are NOT comparable to the Qwen-tokenized counts in `results/`; a
  responses-and-tokens table may only pool rows produced by one provider + model.
* **Sampled draws do not replay.** `seed` is forwarded, but Together documents no
  determinism guarantee, so a majority-vote row records the seed it asked for rather than
  one that reproduces. The HF path's `torch.manual_seed` guarantee does not carry over,
  and `--verify-sample` is correspondingly weaker here: greedy still reproduces in
  practice, sampling need not.
* **The chat template is applied server-side.** `model.py` calls `apply_chat_template`
  locally; here the same single user message is posted to `/v1/chat/completions` and the
  served model's template is applied there. Both send one user turn and no system
  prompt, so the prompt text the conditions build is unchanged -- but the tokens
  wrapping it are the new model's, which is exactly why a prompt frozen against one
  model needs a compliance pilot before it is trusted on another.

Stdlib-only (`urllib`), so this installs with the core deps and needs neither the
`inference` extra nor a GPU. One connection per request, no pooling: that costs a TLS
handshake per call, and is why a run is spread across `--shard` processes rather than
made concurrent inside one.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

API_URL = "https://api.together.xyz/v1/chat/completions"
API_KEY_ENV = "TOGETHER_API_KEY"

# Cloudflare fronts the API and bans urllib's default `Python-urllib/3.x` signature with
# a 403 (Cloudflare error 1010) before the request ever reaches Together, so this is
# required, not cosmetic. Identifying the project also makes our traffic legible in
# Together's logs if a run ever needs explaining.
USER_AGENT = "gedebate/0.0.1 (+https://github.com/graphqa-debate; python-urllib)"

# Retry budget for the two failures a long sharded run actually hits: dynamic rate
# limiting (429) and transient upstream errors (5xx). Together publishes no fixed
# per-model limit -- the cap adjusts to recent usage and sudden bursts get throttled
# regardless of history -- so a shard has to back off and carry on rather than die two
# thousand calls in and leave a half-written arm.
MAX_ATTEMPTS = 6
BACKOFF_BASE = 2.0
BACKOFF_CAP = 60.0
RETRY_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


class TogetherError(RuntimeError):
    """A request that could not be completed, after retries."""


@dataclass
class GenResult:
    """One generation.

    Structurally identical to `gedebate.model.GenResult`, defined separately so the API
    path never imports torch. The conditions duck-type this (they read `.text`,
    `.n_gen_tokens`, `.n_prompt_tokens`), as the test stubs already do.
    """

    text: str
    n_gen_tokens: int
    n_prompt_tokens: int


class TogetherModel:
    def __init__(self, name: str, api_key: str, *, url: str = API_URL,
                 timeout: float = 120.0) -> None:
        self.name = name
        # The run loop prints `model.device`; say where this ran rather than "cpu".
        self.device = "together-api"
        self._api_key = api_key
        self._url = url
        self._timeout = timeout

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        seed: int | None = None,
    ) -> GenResult:
        """Generate a completion for a single user prompt.

        `temperature=None` (default) is greedy, matching the HF path's
        `do_sample=False`. Truncation knobs are sent only when sampling: argmax ignores
        them, and sending them anyway would put a sampling parameter in the manifest of
        a run that does not sample.
        """
        payload: dict = {
            "model": self.name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
        }
        if temperature is None:
            payload["temperature"] = 0.0
        else:
            payload["temperature"] = temperature
            if top_p is not None:
                payload["top_p"] = top_p
            # top_k is not honoured by every served model. It is forwarded so the
            # request matches the recorded decoding; it is not a guarantee.
            if top_k is not None:
                payload["top_k"] = top_k
            if seed is not None:
                payload["seed"] = seed  # forwarded, not a replay guarantee

        body = self._post(payload)
        try:
            message = body["choices"][0].get("message") or {}
        except (KeyError, IndexError, TypeError) as exc:
            raise TogetherError(f"response has no choices: {body!r}") from exc
        text = message.get("content") or ""

        usage = body.get("usage") or {}
        # Matched compute is measured in these counts, so a response without them cannot
        # be scored. Fail loudly instead of writing zeros into the token columns, which
        # would silently corrupt the compute table.
        try:
            n_prompt_tokens = int(usage["prompt_tokens"])
            n_gen_tokens = int(usage["completion_tokens"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TogetherError(f"response has no usage counts: {usage!r}") from exc

        return GenResult(
            text=text.strip(),
            n_gen_tokens=n_gen_tokens,
            n_prompt_tokens=n_prompt_tokens,
        )

    # --- transport ------------------------------------------------------------

    def _post(self, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,  # without this Cloudflare 403s; see USER_AGENT
        }
        last = "no attempt made"
        for attempt in range(MAX_ATTEMPTS):
            request = urllib.request.Request(
                self._url, data=data, method="POST", headers=headers
            )
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code not in RETRY_STATUS:
                    raise TogetherError(f"HTTP {exc.code}: {_error_body(exc)}") from exc
                last = f"HTTP {exc.code}"
                wait = _reset_hint(exc.headers)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last = f"{type(exc).__name__}: {exc}"
                wait = None
            if attempt == MAX_ATTEMPTS - 1:
                break
            delay = wait if wait is not None else _backoff(attempt)
            # Say so on stderr. A retry is otherwise invisible, and a throttled shard
            # looks exactly like a slow one from outside -- across ~32 shards that is
            # the difference between "still working" and "silently rate-limited for an
            # hour". Shard logs are the only place this surfaces.
            print(f"[together] {last}; retry {attempt + 1}/{MAX_ATTEMPTS - 1} "
                  f"in {delay:.1f}s", file=sys.stderr, flush=True)
            time.sleep(delay)
        raise TogetherError(f"giving up after {MAX_ATTEMPTS} attempts ({last})")


def _backoff(attempt: int) -> float:
    """Exponential backoff with jitter.

    The jitter is load-bearing, not decoration: a throttled run has ~32 shards hitting
    the same model, and without it they all retry in lockstep and re-trigger the limit.
    """
    ceiling = min(BACKOFF_CAP, BACKOFF_BASE ** attempt)
    return ceiling * (0.5 + random.random() / 2)


def _reset_hint(headers) -> float | None:
    """Seconds to wait, from the `x-ratelimit-reset` header a throttled call carries.

    Together documents the header as how long to wait but not its unit or format, so an
    unparseable or implausible value falls back to plain backoff rather than being
    trusted.
    """
    raw = headers.get("x-ratelimit-reset") if headers else None
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        return None
    return min(seconds, BACKOFF_CAP) if 0 < seconds else None


def _error_body(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace")[:500]
    except Exception:
        return exc.reason or ""


def load_model(name: str, *, api_key: str | None = None) -> TogetherModel:
    """Mirror of `gedebate.model.load_model` for the API path (nothing to load)."""
    key = api_key or os.environ.get(API_KEY_ENV, "").strip()
    if not key:
        raise RuntimeError(
            f"{API_KEY_ENV} is not set. Put it in .env (gitignored) or export it; "
            "see .env.example."
        )
    return TogetherModel(name, key)
