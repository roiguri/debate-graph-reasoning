"""Tests for `gedebate.together` -- the API backend, exercised without a network.

Every test fakes the transport. What is worth pinning down is the contract the run loop
depends on: the request carries the decoding the manifest claims, and the token counts
that matched compute is measured in come from the server rather than being guessed.
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from gedebate import together


def _reply(text="ANSWER: Yes", prompt_tokens=311, completion_tokens=12) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens,
                  "completion_tokens": completion_tokens},
    }


class _Transport:
    """Records the payloads it was posted and replays a canned script of responses."""

    def __init__(self, script):
        self.script = list(script)
        self.payloads: list[dict] = []

    def __call__(self, payload):
        self.payloads.append(payload)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def model(monkeypatch):
    def _make(script):
        m = together.TogetherModel("some/model", "k")
        transport = _Transport(script)
        monkeypatch.setattr(m, "_post", transport)
        return m, transport
    return _make


def test_generate_returns_server_token_counts(model):
    m, _ = model([_reply(prompt_tokens=311, completion_tokens=12)])
    out = m.generate("q")
    assert out.text == "ANSWER: Yes"
    # Not len(text.split()) or any local estimate: the compute table is only meaningful
    # if these are the served tokenizer's own numbers.
    assert out.n_prompt_tokens == 311 and out.n_gen_tokens == 12


def test_generate_strips_whitespace(model):
    m, _ = model([_reply(text="  ANSWER: 3\n\n")])
    assert m.generate("q").text == "ANSWER: 3"


def test_default_is_greedy_and_sends_no_sampling_knobs(model):
    """Baseline and debate are greedy; a truncation knob in that request would be
    recorded as decoding the run never used."""
    m, t = model([_reply()])
    m.generate("q", max_new_tokens=256)
    payload = t.payloads[0]
    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] == 256
    assert "top_p" not in payload and "top_k" not in payload and "seed" not in payload
    assert payload["messages"] == [{"role": "user", "content": "q"}]


def test_sampling_forwards_the_recorded_decoding(model):
    m, t = model([_reply()])
    m.generate("q", temperature=0.7, top_p=0.8, top_k=20, seed=41)
    payload = t.payloads[0]
    assert payload["temperature"] == 0.7
    assert payload["top_p"] == 0.8 and payload["top_k"] == 20
    assert payload["seed"] == 41


def test_missing_usage_raises_rather_than_writing_zeros(model):
    """A row with 0 tokens would silently corrupt the matched-compute table, so this
    has to be loud."""
    m, _ = model([{"choices": [{"message": {"content": "hi"}}]}])
    with pytest.raises(together.TogetherError, match="usage"):
        m.generate("q")


def test_missing_choices_raises(model):
    m, _ = model([{"usage": {"prompt_tokens": 1, "completion_tokens": 1}}])
    with pytest.raises(together.TogetherError, match="choices"):
        m.generate("q")


def test_empty_content_is_not_an_error(model):
    """A model that emits nothing is a parse failure for the scorer to record, not a
    transport failure to retry."""
    m, _ = model([_reply(text=None, completion_tokens=0)])
    out = m.generate("q")
    assert out.text == "" and out.n_gen_tokens == 0


# --- transport ----------------------------------------------------------------

def _http_error(code):
    return urllib.error.HTTPError("u", code, "boom", {}, None)


def test_retries_then_succeeds(monkeypatch):
    """A throttled shard must back off and carry on, not die mid-arm."""
    monkeypatch.setattr(together.time, "sleep", lambda _: None)
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(request)
        if len(calls) < 3:
            raise _http_error(429)
        return _FakeResponse(_reply())

    monkeypatch.setattr(together.urllib.request, "urlopen", fake_urlopen)
    out = together.TogetherModel("m", "k").generate("q")
    assert out.n_gen_tokens == 12
    assert len(calls) == 3


def test_sends_a_non_default_user_agent(monkeypatch):
    """Cloudflare fronts the API and 403s urllib's default `Python-urllib/3.x`
    signature (error 1010) before Together ever sees the request. Observed, not
    hypothetical -- the first live call failed exactly this way."""
    seen = []

    def fake_urlopen(request, timeout=None):
        seen.append(request)
        return _FakeResponse(_reply())

    monkeypatch.setattr(together.urllib.request, "urlopen", fake_urlopen)
    together.TogetherModel("m", "k").generate("q")
    agent = seen[0].get_header("User-agent")
    assert agent == together.USER_AGENT
    assert "Python-urllib" not in agent


def test_authorization_header_carries_the_key(monkeypatch):
    seen = []

    def fake_urlopen(request, timeout=None):
        seen.append(request)
        return _FakeResponse(_reply())

    monkeypatch.setattr(together.urllib.request, "urlopen", fake_urlopen)
    together.TogetherModel("m", "sk-abc").generate("q")
    assert seen[0].get_header("Authorization") == "Bearer sk-abc"


def test_gives_up_after_max_attempts(monkeypatch):
    monkeypatch.setattr(together.time, "sleep", lambda _: None)
    monkeypatch.setattr(together.urllib.request, "urlopen",
                        lambda *a, **k: (_ for _ in ()).throw(_http_error(503)))
    with pytest.raises(together.TogetherError, match="giving up"):
        together.TogetherModel("m", "k").generate("q")


def test_retries_are_announced_on_stderr(monkeypatch, capsys):
    """A silent retry makes a rate-limited shard indistinguishable from a slow one."""
    monkeypatch.setattr(together.time, "sleep", lambda _: None)
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(request)
        if len(calls) < 3:
            raise _http_error(429)
        return _FakeResponse(_reply())

    monkeypatch.setattr(together.urllib.request, "urlopen", fake_urlopen)
    together.TogetherModel("m", "k").generate("q")
    err = capsys.readouterr().err
    assert err.count("[together]") == 2      # one line per retry, not per attempt
    assert "429" in err and "retry 1/" in err


def test_success_is_silent(monkeypatch, capsys):
    monkeypatch.setattr(together.urllib.request, "urlopen",
                        lambda *a, **k: _FakeResponse(_reply()))
    together.TogetherModel("m", "k").generate("q")
    assert capsys.readouterr().err == ""


def test_non_retryable_status_fails_immediately(monkeypatch):
    """401 is a bad key: retrying it five times just delays the error."""
    seen = []

    def fake_urlopen(request, timeout=None):
        seen.append(request)
        raise _http_error(401)

    monkeypatch.setattr(together.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(together.TogetherError, match="401"):
        together.TogetherModel("m", "k").generate("q")
    assert len(seen) == 1


def test_backoff_is_jittered_and_capped():
    # Lockstep retries across ~32 shards would re-trigger the limit they are waiting out.
    waits = {together._backoff(3) for _ in range(20)}
    assert len(waits) > 1
    assert all(0 < w <= together.BACKOFF_CAP for w in waits)
    assert together._backoff(50) <= together.BACKOFF_CAP


@pytest.mark.parametrize("raw,expected", [
    ("2.5", 2.5),
    ("9999", together.BACKOFF_CAP),  # clamped
    ("0", None),                      # implausible -> fall back to backoff
    ("later", None),                  # unparseable -> fall back
    (None, None),                     # header absent
])
def test_reset_hint(raw, expected):
    assert together._reset_hint({"x-ratelimit-reset": raw}) == expected


def test_load_model_requires_a_key(monkeypatch):
    monkeypatch.delenv(together.API_KEY_ENV, raising=False)
    with pytest.raises(RuntimeError, match=together.API_KEY_ENV):
        together.load_model("m")


def test_load_model_reads_the_env(monkeypatch):
    monkeypatch.setenv(together.API_KEY_ENV, "secret")
    m = together.load_model("some/model")
    assert m.name == "some/model" and m.device == "together-api"


class _FakeResponse:
    def __init__(self, body):
        self._body = json.dumps(body).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
