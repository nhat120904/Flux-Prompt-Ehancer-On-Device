from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ManifestFile:
    path: str
    size_bytes: int
    sha256: str
    url: str


def sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    model_root: Path,
    version: str,
    base_url: str,
    min_app_version: str,
) -> dict[str, object]:
    files: list[ManifestFile] = []

    for path in sorted(model_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(model_root).as_posix()
        files.append(
            ManifestFile(
                path=rel,
                size_bytes=path.stat().st_size,
                sha256=sha256_file(path),
                url=f"{base_url.rstrip('/')}/{version}/{rel}",
            )
        )

    return {
        "version": version,
        "minAppVersion": min_app_version,
        "files": [asdict(file) for file in files],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build model-manifest.json for mobile artifacts.")
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--min-app-version", default="1.0.0")
    parser.add_argument("--output", default="model-manifest.json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(
        model_root=args.model_root,
        version=args.version,
        base_url=args.base_url,
        min_app_version=args.min_app_version,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest to {args.output}")


if __name__ == "__main__":
    main()
