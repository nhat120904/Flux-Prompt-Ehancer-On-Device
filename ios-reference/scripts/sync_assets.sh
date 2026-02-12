#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT_DIR="${ROOT_DIR}/PromptEnhancerApp"
MODEL_DST="${PROJECT_DIR}/AssetsData/Model"
TOKENIZER_DST="${PROJECT_DIR}/AssetsData/Tokenizer"

IOS_EXPORT_DIR="${1:-${ROOT_DIR}/../build/ios-export}"
HF_SNAPSHOT_DIR="${2:-$HOME/.cache/huggingface/hub/models--imranali291--flux-prompt-enhancer/snapshots/eaeb982805f1a755d301cb4dbe1f3773230e97c6}"

mkdir -p "${MODEL_DST}" "${TOKENIZER_DST}"

rm -rf "${MODEL_DST}/encoder.mlpackage" "${MODEL_DST}/decoder_init.mlpackage"
cp -R "${IOS_EXPORT_DIR}/encoder.mlpackage" "${MODEL_DST}/"
cp -R "${IOS_EXPORT_DIR}/decoder_init.mlpackage" "${MODEL_DST}/"
cp "${IOS_EXPORT_DIR}/conversion-metadata.json" "${MODEL_DST}/"

cp "${HF_SNAPSHOT_DIR}/spiece.model" "${TOKENIZER_DST}/"
cp "${HF_SNAPSHOT_DIR}/config.json" "${TOKENIZER_DST}/"
cp "${HF_SNAPSHOT_DIR}/tokenizer_config.json" "${TOKENIZER_DST}/"
cp "${HF_SNAPSHOT_DIR}/special_tokens_map.json" "${TOKENIZER_DST}/"

TOKENIZER_DST="${TOKENIZER_DST}" uv run python - <<'PY'
import json
import os
from pathlib import Path
import sentencepiece as spm

root = Path(os.environ["TOKENIZER_DST"])
sp = spm.SentencePieceProcessor(model_file=str(root / "spiece.model"))
out = root / "tokenizer_vocab.json"
entries = [
    {"id": i, "piece": sp.id_to_piece(i), "score": float(sp.get_score(i))}
    for i in range(sp.get_piece_size())
]
payload = {
    "model_type": "sentencepiece_unigram",
    "piece_size": sp.get_piece_size(),
    "unk_id": int(sp.unk_id()),
    "pad_id": 0,
    "eos_id": 1,
    "decoder_start_id": 0,
    "extra_ids": 100,
    "entries": entries,
}
out.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print(f"Wrote {out}")
PY

# Remove macOS extended attributes that can break iOS code signing.
xattr -cr "${MODEL_DST}" || true
xattr -cr "${TOKENIZER_DST}" || true

echo "Synced model + tokenizer assets into ${PROJECT_DIR}/AssetsData"
