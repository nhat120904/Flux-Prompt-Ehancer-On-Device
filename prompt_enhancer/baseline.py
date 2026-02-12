from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

DEFAULT_MODEL_ID = "imranali291/flux-prompt-enhancer"

_tokenizer: T5Tokenizer | None = None
_model: T5ForConditionalGeneration | None = None
_loaded_model_id: str | None = None


@dataclass(slots=True)
class EnhanceOptions:
    top_p: float = 0.92
    temperature: float = 0.8
    repetition_penalty: float = 1.2
    no_repeat_ngram_size: int = 3
    min_new_tokens: int = 60
    max_new_tokens: int = 180
    retries: int = 3
    seed: int | None = None
    timeout_ms: int = 8_000


@dataclass(slots=True)
class EnhanceResult:
    text: str
    latency_ms: int
    tokens_generated: int
    finish_reason: str


def _load(model_id: str = DEFAULT_MODEL_ID) -> tuple[T5Tokenizer, T5ForConditionalGeneration]:
    global _tokenizer, _model, _loaded_model_id
    if _tokenizer is None or _model is None or _loaded_model_id != model_id:
        _tokenizer = T5Tokenizer.from_pretrained(model_id)
        _model = T5ForConditionalGeneration.from_pretrained(model_id)
        _loaded_model_id = model_id
    return _tokenizer, _model


def is_complete_sentence(text: str) -> bool:
    return text.rstrip().endswith((".", "!", "?"))


def enhance_prompt(
    input_text: str,
    options: EnhanceOptions | None = None,
    model_id: str = DEFAULT_MODEL_ID,
) -> EnhanceResult:
    options = options or EnhanceOptions()
    tokenizer, model = _load(model_id)
    start = time.perf_counter()
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    best_text = ""
    best_tokens = 0
    finish_reason = "max_tokens"

    for attempt in range(options.retries):
        if (time.perf_counter() - start) * 1000 > options.timeout_ms:
            finish_reason = "timeout"
            break

        generate_kwargs: dict[str, object] = {
            "do_sample": True,
            "top_p": options.top_p,
            "temperature": options.temperature,
            "repetition_penalty": options.repetition_penalty,
            "no_repeat_ngram_size": options.no_repeat_ngram_size,
            "min_new_tokens": options.min_new_tokens,
            "max_new_tokens": options.max_new_tokens,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if options.seed is not None:
            generator = torch.Generator(device=model.device).manual_seed(options.seed + attempt)
            generate_kwargs["generator"] = generator

        output = model.generate(input_ids, **generate_kwargs)
        text = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        token_count = int(output.shape[-1])

        if len(text) > len(best_text):
            best_text = text
            best_tokens = token_count

        if is_complete_sentence(text):
            finish_reason = "eos"
            latency_ms = int((time.perf_counter() - start) * 1000)
            return EnhanceResult(
                text=text,
                latency_ms=latency_ms,
                tokens_generated=token_count,
                finish_reason=finish_reason,
            )

    if best_text and not is_complete_sentence(best_text):
        best_text = f"{best_text}."

    latency_ms = int((time.perf_counter() - start) * 1000)
    if finish_reason not in {"timeout", "cancelled", "error"} and best_text:
        finish_reason = "max_tokens"
    return EnhanceResult(
        text=best_text,
        latency_ms=latency_ms,
        tokens_generated=best_tokens,
        finish_reason=finish_reason,
    )
