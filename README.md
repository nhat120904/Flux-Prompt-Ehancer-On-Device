# Flux Prompt Enhancer On Device

This repository contains:
- Python baseline inference for `imranali291/flux-prompt-enhancer`
- Mobile export tooling for:
  - Android LiteRT (`.tflite`)
  - iOS Core ML (`.mlpackage`)
- Artifact packaging + `model-manifest.json` generation
- Native reference implementations:
  - `android-reference/` (Kotlin)
  - `ios-reference/` (Swift)

## Local Baseline

```bash
uv run python main.py "beautiful house with text 'hello'"
```

## Inference Spec

See `docs/inference-spec.md` for decode defaults and parity rules.
For mobile handoff + integration snippets, see `docs/mobile-inference-handoff.md`.

## Export Mobile Artifacts

### Android LiteRT

Install export dependencies first:

```bash
uv add tensorflow tf-keras "transformers<5"
```

```bash
uv run python tools/export_tflite.py \
  --model-id imranali291/flux-prompt-enhancer \
  --out-dir build/android-export \
  --quantize-fp16 \
  --allow-no-cache-decoder
```

### iOS Core ML

```bash
uv run python tools/export_coreml.py \
  --model-id imranali291/flux-prompt-enhancer \
  --out-dir build/ios-export \
  --compute-precision FLOAT16
```

## Package and Manifest

```bash
uv run python tools/package_model_bundle.py \
  --runtime android \
  --artifacts-dir build/android-export \
  --version v1 \
  --out-root artifacts

uv run python tools/build_manifest.py \
  --model-root artifacts/android/v1 \
  --version v1 \
  --base-url https://cdn.example.com/models/android \
  --output artifacts/android/model-manifest.json
```

`model-manifest.json` is validated by `schema/model-manifest.schema.json`.

## Parity Evaluation

```bash
uv run python tools/parity_eval.py \
  --prompts eval/prompts.jsonl \
  --baseline-out eval/baseline_outputs.jsonl
```

If you export mobile outputs into JSONL with `{"prompt": "...", "text": "..."}` rows, pass it as `--candidate-out`.

## Notes
- Android runtime target is LiteRT only (no ONNX fallback).
- iOS runtime target is Core ML only.
- First run downloads model artifacts and verifies SHA-256 checksums.
- v1 scripts export non-KV-cache decoder graphs by default; replace with cache-aware variants when your conversion stack supports it.
