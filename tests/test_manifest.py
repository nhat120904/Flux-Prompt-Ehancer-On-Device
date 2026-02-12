from __future__ import annotations

import json
from pathlib import Path

from tools.build_manifest import build_manifest


def test_build_manifest(tmp_path: Path) -> None:
    model_root = tmp_path / "model"
    model_root.mkdir()
    sample = model_root / "encoder.tflite"
    sample.write_bytes(b"abc")

    manifest = build_manifest(
        model_root=model_root,
        version="v1",
        base_url="https://cdn.example.com/models/android",
        min_app_version="1.0.0",
    )

    assert manifest["version"] == "v1"
    assert manifest["minAppVersion"] == "1.0.0"
    files = manifest["files"]
    assert isinstance(files, list)
    assert len(files) == 1
    assert files[0]["path"] == "encoder.tflite"
    assert files[0]["size_bytes"] == 3
    assert files[0]["url"].endswith("/v1/encoder.tflite")
    assert len(files[0]["sha256"]) == 64
    json.dumps(manifest)
