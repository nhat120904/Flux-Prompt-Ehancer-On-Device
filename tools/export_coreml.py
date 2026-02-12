from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import T5ForConditionalGeneration


def _require_coremltools() -> object:
    try:
        import coremltools as ct  # type: ignore
    except Exception as exc:  # pragma: no cover - setup-time guard
        raise SystemExit(
            "coremltools is required for iOS export. Install with: uv add coremltools"
        ) from exc
    return ct


def _patch_coremltools_int_cast(ct: object) -> None:
    """Work around coremltools torch frontend bug for scalar ndarray int casts.

    With recent torch/coremltools combos, torch op `int()` may surface as a
    folded numpy array with shape (1,). coremltools tries `int(ndarray)` and
    crashes; scalar extraction fixes conversion.
    """
    try:
        from coremltools.converters.mil.frontend.torch import ops as torch_ops  # type: ignore
    except Exception:
        return

    if getattr(torch_ops, "_prompt_enhancer_int_cast_patch", False):
        return

    original_cast = torch_ops._cast

    def patched_cast(context: object, node: object, dtype: object, dtype_name: str) -> None:
        inputs = torch_ops._get_inputs(context, node, expected=1)
        x = inputs[0]
        if x.can_be_folded_to_const() and hasattr(x, "val"):
            val = x.val
            if isinstance(val, np.ndarray) and val.size == 1:
                scalar = val.reshape(()).item()
                res = torch_ops.mb.const(val=dtype(scalar), name=node.name)
                context.add(res, node.name)
                return
        return original_cast(context, node, dtype, dtype_name)

    torch_ops._cast = patched_cast
    torch_ops._prompt_enhancer_int_cast_patch = True


class EncoderWrapper(torch.nn.Module):
    def __init__(self, model: T5ForConditionalGeneration):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        return out.last_hidden_state


class DecoderWrapper(torch.nn.Module):
    def __init__(self, model: T5ForConditionalGeneration):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
            return_dict=True,
        )
        return out.logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export T5 encoder/decoder Core ML artifacts.")
    parser.add_argument("--model-id", default="imranali291/flux-prompt-enhancer")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-input-tokens", type=int, default=128)
    parser.add_argument("--max-decoder-tokens", type=int, default=64)
    parser.add_argument(
        "--compute-precision",
        default="FLOAT16",
        choices=["FLOAT16", "FLOAT32"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ct = _require_coremltools()
    _patch_coremltools_int_cast(ct)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = T5ForConditionalGeneration.from_pretrained(args.model_id).eval()

    sample_input_ids = torch.ones((1, args.max_input_tokens), dtype=torch.long)
    sample_attention_mask = torch.ones((1, args.max_input_tokens), dtype=torch.long)
    sample_decoder_input_ids = torch.ones((1, args.max_decoder_tokens), dtype=torch.long)

    encoder = EncoderWrapper(model).eval()
    traced_encoder = torch.jit.trace(encoder, (sample_input_ids, sample_attention_mask))
    encoder_mlmodel = ct.convert(  # type: ignore[attr-defined]
        traced_encoder,
        inputs=[
            ct.TensorType(name="input_ids", shape=sample_input_ids.shape, dtype=ct.converters.mil.input_types.types.int32),  # type: ignore[attr-defined]
            ct.TensorType(name="attention_mask", shape=sample_attention_mask.shape, dtype=ct.converters.mil.input_types.types.int32),  # type: ignore[attr-defined]
        ],
        compute_precision=getattr(ct.precision, args.compute_precision),  # type: ignore[attr-defined]
    )
    encoder_path = args.out_dir / "encoder.mlpackage"
    encoder_mlmodel.save(str(encoder_path))
    print(f"Wrote {encoder_path}")

    decoder = DecoderWrapper(model).eval()
    traced_decoder = torch.jit.trace(
        decoder,
        (sample_input_ids, sample_attention_mask, sample_decoder_input_ids),
    )
    decoder_mlmodel = ct.convert(  # type: ignore[attr-defined]
        traced_decoder,
        inputs=[
            ct.TensorType(name="input_ids", shape=sample_input_ids.shape, dtype=ct.converters.mil.input_types.types.int32),  # type: ignore[attr-defined]
            ct.TensorType(name="attention_mask", shape=sample_attention_mask.shape, dtype=ct.converters.mil.input_types.types.int32),  # type: ignore[attr-defined]
            ct.TensorType(
                name="decoder_input_ids",
                shape=sample_decoder_input_ids.shape,
                dtype=ct.converters.mil.input_types.types.int32,  # type: ignore[attr-defined]
            ),
        ],
        compute_precision=getattr(ct.precision, args.compute_precision),  # type: ignore[attr-defined]
    )
    decoder_init_path = args.out_dir / "decoder_init.mlpackage"
    decoder_mlmodel.save(str(decoder_init_path))
    print(f"Wrote {decoder_init_path}")

    # For v1 parity we keep a non-cache decoder artifact under the cache name.
    decoder_with_past_path = args.out_dir / "decoder_with_past.mlpackage"
    decoder_mlmodel.save(str(decoder_with_past_path))
    print(
        "Saved decoder_with_past.mlpackage using non-cache graph. "
        "Replace with KV-cache graph if available in your Core ML conversion pipeline."
    )

    metadata = {
        "model_id": args.model_id,
        "runtime": "CoreML",
        "uses_kv_cache": False,
        "artifacts": [
            "encoder.mlpackage",
            "decoder_init.mlpackage",
            "decoder_with_past.mlpackage",
        ],
        "max_input_tokens": args.max_input_tokens,
        "max_decoder_tokens": args.max_decoder_tokens,
    }
    (args.out_dir / "conversion-metadata.json").write_text(json.dumps(metadata, indent=2), "utf-8")


if __name__ == "__main__":
    main()
