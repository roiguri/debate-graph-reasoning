"""Minimal in-process HuggingFace model wrapper.

Deliberately small: load an instruct model, generate on a single prompt,
and report the number of generated tokens. Token counting is the one thing that
must be exact here -- the matched-compute comparison across conditions (baseline
/ majority-vote / debate) is measured in total generated tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class GenResult:
    """One generation. `n_gen_tokens` is the count that feeds matched-compute."""

    text: str
    n_gen_tokens: int
    n_prompt_tokens: int


class Model:
    def __init__(self, name: str, tokenizer, model, device: str):
        self.name = name
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

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

        `temperature=None` (default) is greedy/deterministic. A float enables
        sampling -- used by the majority-vote condition. `seed` makes a sampled draw
        reproducible. `top_p`/`top_k` are set explicitly when sampling so the
        truncation is a deliberate, recorded choice rather than the model's shipped
        generation_config defaults (Qwen ships top_p=0.8/top_k=20); pass top_p=1.0
        and top_k=0 to disable nucleus/top-k filtering.
        """
        if seed is not None:
            torch.manual_seed(seed)

        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)
        n_prompt_tokens = input_ids.shape[-1]
        # Single unpadded sequence -> all-ones mask. Passing it explicitly avoids
        # unreliable generation when pad_token == eos_token (can't be inferred).
        attention_mask = torch.ones_like(input_ids)

        do_sample = temperature is not None
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id
            or self.tokenizer.eos_token_id,
        }
        if do_sample:
            # Set truncation explicitly so we never inherit the model's shipped
            # generation_config sampling defaults. Greedy ignores these (argmax),
            # so they are only passed when sampling.
            gen_kwargs["temperature"] = temperature
            if top_p is not None:
                gen_kwargs["top_p"] = top_p
            if top_k is not None:
                gen_kwargs["top_k"] = top_k

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids, attention_mask=attention_mask, **gen_kwargs
            )

        # Only the newly generated tail -- excludes the prompt.
        new_ids = output_ids[0, n_prompt_tokens:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return GenResult(
            text=text.strip(),
            n_gen_tokens=int(new_ids.shape[-1]),
            n_prompt_tokens=int(n_prompt_tokens),
        )


def load_model(name: str) -> Model:
    """Load an instruct model in fp16 onto the GPU (CPU fallback for local dev)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return Model(name=name, tokenizer=tokenizer, model=model, device=device)
