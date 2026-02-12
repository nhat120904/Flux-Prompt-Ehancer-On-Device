from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile


def _require_tensorflow() -> tuple[object, object]:
    try:
        import tensorflow as tf  # type: ignore
        import transformers  # type: ignore
    except Exception as exc:  # pragma: no cover - setup-time guard
        raise SystemExit(
            "TensorFlow + tf-keras + transformers<5 are required. "
            "Install with: uv add tensorflow tf-keras \"transformers<5\""
        ) from exc

    if int(transformers.__version__.split(".", maxsplit=1)[0]) >= 5:
        raise SystemExit(
            "This exporter requires TensorFlow-backed T5 classes removed in transformers>=5. "
            "Install a 4.x version: uv add \"transformers<5\""
        )

    try:
        from transformers import TFT5ForConditionalGeneration  # type: ignore
    except Exception as exc:  # pragma: no cover - setup-time guard
        raise SystemExit(
            "Failed to import TFT5ForConditionalGeneration. "
            "Use: uv add tensorflow tf-keras \"transformers<5\". "
            f"Original error: {exc}"
        ) from exc
    return tf, TFT5ForConditionalGeneration


def convert_concrete_function_to_tflite(
    tf: object,
    concrete_fn: object,
    output_file: Path,
    quantize_fp16: bool,
) -> None:
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])  # type: ignore[attr-defined]
    converter.target_spec.supported_ops = [  # type: ignore[attr-defined]
        tf.lite.OpsSet.TFLITE_BUILTINS,  # type: ignore[attr-defined]
        tf.lite.OpsSet.SELECT_TF_OPS,  # type: ignore[attr-defined]
    ]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # type: ignore[attr-defined]
    if quantize_fp16:
        converter.target_spec.supported_types = [tf.float16]  # type: ignore[attr-defined]

    model_bytes = converter.convert()
    output_file.write_bytes(model_bytes)
    print(f"Wrote {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export T5 encoder/decoder LiteRT artifacts.")
    parser.add_argument("--model-id", default="imranali291/flux-prompt-enhancer")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--quantize-fp16",
        action="store_true",
        help="Apply float16 optimization for LiteRT conversion.",
    )
    parser.add_argument(
        "--allow-no-cache-decoder",
        action="store_true",
        help=(
            "Export decoder_with_past.tflite as no-cache fallback if cache tracing "
            "is unsupported in your TF stack."
        ),
    )
    return parser.parse_args()


def load_tf_t5_model(TFT5ForConditionalGeneration: object, model_id: str) -> object:
    try:
        return TFT5ForConditionalGeneration.from_pretrained(model_id, from_pt=True)  # type: ignore[attr-defined]
    except OSError as exc:
        if "does not appear to have a file named pytorch_model.bin" not in str(exc):
            raise

    from transformers import AutoModelForSeq2SeqLM  # type: ignore

    print(
        "No pytorch_model.bin found; converting model.safetensors to PyTorch bin "
        "in a temporary directory for TF export."
    )
    with tempfile.TemporaryDirectory(prefix="t5-ptbin-") as temp_dir:
        pt_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        pt_model.save_pretrained(temp_dir, safe_serialization=False)
        return TFT5ForConditionalGeneration.from_pretrained(temp_dir, from_pt=True)  # type: ignore[attr-defined]


def main() -> None:
    args = parse_args()
    tf, TFT5ForConditionalGeneration = _require_tensorflow()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = load_tf_t5_model(TFT5ForConditionalGeneration, args.model_id)

    class EncoderModule(tf.Module):  # type: ignore[misc]
        def __init__(self, tf_model: object):
            super().__init__()
            self.tf_model = tf_model

        @tf.function(  # type: ignore[misc]
            input_signature=[
                tf.TensorSpec([1, None], tf.int32, name="input_ids"),
                tf.TensorSpec([1, None], tf.int32, name="attention_mask"),
            ]
        )
        def __call__(self, input_ids: object, attention_mask: object) -> dict[str, object]:
            outputs = self.tf_model.encoder(  # type: ignore[attr-defined]
                input_ids=input_ids,
                attention_mask=attention_mask,
                training=False,
            )
            return {"last_hidden_state": outputs.last_hidden_state}

    class DecoderNoCacheModule(tf.Module):  # type: ignore[misc]
        def __init__(self, tf_model: object):
            super().__init__()
            self.tf_model = tf_model

        @tf.function(  # type: ignore[misc]
            input_signature=[
                tf.TensorSpec([1, None], tf.int32, name="decoder_input_ids"),
                tf.TensorSpec([1, None], tf.int32, name="input_ids"),
                tf.TensorSpec([1, None], tf.int32, name="attention_mask"),
            ]
        )
        def __call__(
            self,
            decoder_input_ids: object,
            input_ids: object,
            attention_mask: object,
        ) -> dict[str, object]:
            outputs = self.tf_model(  # type: ignore[operator]
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                use_cache=False,
                training=False,
            )
            return {"logits": outputs.logits}

    encoder_module = EncoderModule(model)
    encoder_cf = encoder_module.__call__.get_concrete_function()
    convert_concrete_function_to_tflite(
        tf=tf,
        concrete_fn=encoder_cf,
        output_file=args.out_dir / "encoder.tflite",
        quantize_fp16=args.quantize_fp16,
    )

    decoder_module = DecoderNoCacheModule(model)
    decoder_cf = decoder_module.__call__.get_concrete_function()
    convert_concrete_function_to_tflite(
        tf=tf,
        concrete_fn=decoder_cf,
        output_file=args.out_dir / "decoder_init.tflite",
        quantize_fp16=args.quantize_fp16,
    )

    decoder_with_past_target = args.out_dir / "decoder_with_past.tflite"
    if args.allow_no_cache_decoder:
        decoder_with_past_target.write_bytes((args.out_dir / "decoder_init.tflite").read_bytes())
        print(
            "Cache-aware decoder export is not enabled in this script; "
            "copied decoder_init.tflite to decoder_with_past.tflite."
        )
    else:
        raise SystemExit(
            "decoder_with_past export is model-stack dependent. "
            "Re-run with --allow-no-cache-decoder for iterative decode without KV cache."
        )

    metadata = {
        "model_id": args.model_id,
        "runtime": "LiteRT",
        "uses_kv_cache": False,
        "artifacts": [
            "encoder.tflite",
            "decoder_init.tflite",
            "decoder_with_past.tflite",
        ],
    }
    (args.out_dir / "conversion-metadata.json").write_text(json.dumps(metadata, indent=2), "utf-8")


if __name__ == "__main__":
    main()
