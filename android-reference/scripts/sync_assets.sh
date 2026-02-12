#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_SRC="${1:-$ROOT_DIR/../build/android-export}"
TOKENIZER_SRC="${2:-$ROOT_DIR/../ios-reference/PromptEnhancerApp/AssetsData/Tokenizer}"

MODEL_DST="$ROOT_DIR/app/src/main/assets/model"
TOKENIZER_DST="$ROOT_DIR/app/src/main/assets/tokenizer"

mkdir -p "$MODEL_DST" "$TOKENIZER_DST"

required_model=(
  "decoder_init.tflite"
  "decoder_with_past.tflite"
  "encoder.tflite"
  "conversion-metadata.json"
)

for file in "${required_model[@]}"; do
  if [[ ! -f "$MODEL_SRC/$file" ]]; then
    echo "Missing model artifact: $MODEL_SRC/$file" >&2
    exit 1
  fi
done

if [[ ! -f "$TOKENIZER_SRC/tokenizer_vocab.json" ]]; then
  echo "Missing tokenizer_vocab.json in $TOKENIZER_SRC" >&2
  exit 1
fi

cp "$MODEL_SRC/decoder_init.tflite" "$MODEL_DST/"
cp "$MODEL_SRC/decoder_with_past.tflite" "$MODEL_DST/"
cp "$MODEL_SRC/encoder.tflite" "$MODEL_DST/"
cp "$MODEL_SRC/conversion-metadata.json" "$MODEL_DST/"
cp "$TOKENIZER_SRC"/* "$TOKENIZER_DST/"

echo "Synced model assets from: $MODEL_SRC"
echo "Synced tokenizer assets from: $TOKENIZER_SRC"
ls -lh "$MODEL_DST"
ls -lh "$TOKENIZER_DST"
