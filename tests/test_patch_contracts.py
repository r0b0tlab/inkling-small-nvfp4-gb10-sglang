from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH_DIR = ROOT / "patches" / "sglang-b7252-sm121"


def test_inkling_multimodal_import_fix_is_content_addressed() -> None:
    descriptor = json.loads((PATCH_DIR / "bundle-descriptor.json").read_text())
    entry = next(item for item in descriptor["files"] if item["path"] == "image_processing.py")
    patch = PATCH_DIR / entry["path"]
    digest = hashlib.sha256(patch.read_bytes()).hexdigest()
    assert digest == entry["sha256"]
    assert entry["target"].endswith("/sglang/srt/multimodal/inkling/image_processing.py")
    text = patch.read_text()
    assert "@njit(cache=False)" in text
    assert "@njit(cache=True)" not in text


def test_inkling_dense_mlp_has_no_unconditional_torch_compile() -> None:
    descriptor = json.loads((PATCH_DIR / "bundle-descriptor.json").read_text())
    entry = next(item for item in descriptor["files"] if item["path"] == "dense_mlp.py")
    patch = PATCH_DIR / entry["path"]
    assert hashlib.sha256(patch.read_bytes()).hexdigest() == entry["sha256"]
    assert entry["target"].endswith("/sglang/srt/models/inkling_common/dense_mlp.py")
    assert "@torch.compile" not in patch.read_text()