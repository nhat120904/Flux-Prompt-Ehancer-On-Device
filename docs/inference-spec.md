# Inference Spec (Android LiteRT + iOS Core ML)

This document is the source of truth for output parity between Android and iOS.

## Model and Tokenization
- Base model: `imranali291/flux-prompt-enhancer` (T5 seq2seq).
- Tokenizer: SentencePiece model from the Hugging Face repo.
- Input truncation default: 128 source tokens.
- Decoder start token: model-specific `decoder_start_token_id`.
- Stop token: tokenizer/model EOS token.

## Default Decode Parameters
- `top_p = 0.92`
- `temperature = 0.8`
- `repetition_penalty = 1.2`
- `no_repeat_ngram_size = 3`
- `min_new_tokens = 60`
- `max_new_tokens = 180`
- `retries = 3`
- `timeout_ms = 8000`

Mobile runtime note:
- Effective `max_new_tokens` on current exported mobile graphs is `63` (decoder context length is 64 including decoder start token).

## Logits Post-Processing Order
1. Apply repetition penalty on previously generated tokens.
2. Apply no-repeat-ngram blocking.
3. Apply temperature scaling.
4. Apply top-p filtering.
5. Sample next token from filtered distribution.

## Retry and Completion Rule
- Generate up to `retries`.
- If candidate text ends with `.`, `!`, or `?`, return immediately (`finish_reason = eos`).
- Otherwise keep the longest candidate seen.
- If all retries are incomplete and longest candidate is non-empty, append `.`.
- If timeout is hit before a completed sentence is produced, return best candidate with `finish_reason = timeout`.

## Finish Reasons
- `eos`: completed sentence from decoding loop.
- `max_tokens`: generated without EOS-complete sentence before limits.
- `timeout`: wall-clock timeout reached.
- `cancelled`: user/app cancellation.
- `error`: runtime or model failure.

## Platform Constraints
- Android runtime is LiteRT only.
- iOS runtime is Core ML only.
- No cloud fallback in v1.
