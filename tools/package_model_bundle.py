from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


TOKENIZER_FILES = [
    "tokenizer_config.json",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "config.json",
    "generation_config.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package mobile artifacts with tokenizer files.")
    parser.add_argument("--model-id", default="imranali291/flux-prompt-enhancer")
    parser.add_argument("--runtime", choices=["android", "ios"], required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--out-root", type=Path, default=Path("artifacts"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = args.out_root / args.runtime / args.version
    target.mkdir(parents=True, exist_ok=True)

    for artifact in args.artifacts_dir.glob("*"):
        if artifact.is_file():
            shutil.copy2(artifact, target / artifact.name)

    for file_name in TOKENIZER_FILES:
        try:
            src = hf_hub_download(repo_id=args.model_id, filename=file_name)
        except Exception:
            continue
        shutil.copy2(src, target / file_name)

    print(f"Packaged bundle in {target}")


if __name__ == "__main__":
    main()
