from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from inkling_release.launch import build_docker_argv, launch, owned_container_name
from inkling_release.preflight import PreflightError, bind_admission, run_preflight, validate_admission
from inkling_release.profiles import load_profile
from inkling_release.render import render_command
from inkling_release.stop import build_stop_argv

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "example.invalid/inkling@sha256:" + "a" * 64
DIGEST = "b" * 64


def _admission(path: Path, *, digest: str | None = DIGEST) -> Path:
    value: dict[str, object] = {"schema_version": 1, "status": "PHASE0_PASS", "descriptor_rooted": True, "non_symlink": True, "read_only": True, "read_only_mount": "ro"}
    if digest is not None:
        value["runtime_manifest_sha256"] = digest
    path.write_text(json.dumps(value))
    return path


def _model(tmp_path: Path) -> Path:
    path = tmp_path / "model"
    path.mkdir()
    path.chmod(0o555)
    return path


def _cache(tmp_path: Path) -> Path:
    path = tmp_path / "cache"
    path.mkdir(mode=0o700)
    return path


def _rendered() -> dict[str, object]:
    profile = load_profile(ROOT / "profiles" / "inkling-small-nvfp4.json")
    return render_command(profile, node_rank=0, dist_init_addr="192.0.2.1:25160", image_ref=IMAGE)


def test_admission_requires_fail_closed_assertions() -> None:
    value = {"schema_version": 1, "status": "PHASE0_PASS", "descriptor_rooted": True, "non_symlink": True, "read_only": True, "runtime_manifest_sha256": DIGEST}
    assert validate_admission(value, require_runtime_binding=True)["status"] == "PHASE0_PASS"
    for key in ("descriptor_rooted", "non_symlink", "read_only"):
        changed = dict(value)
        changed[key] = False
        with pytest.raises(PreflightError):
            validate_admission(changed)


def test_preflight_requires_read_only_root_and_digest(tmp_path: Path) -> None:
    model = _model(tmp_path)
    admission = _admission(tmp_path / "admission.json")
    result = run_preflight(admission, model, expected_runtime_manifest_sha256=DIGEST)
    assert result["status"] == "PASS" and result["descriptor_rooted"] is True
    with pytest.raises(PreflightError, match="mismatch"):
        run_preflight(admission, model, expected_runtime_manifest_sha256="c" * 64)


def test_preflight_rejects_writable_and_symlink_roots(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir(mode=0o755)
    admission = _admission(tmp_path / "admission.json")
    with pytest.raises(PreflightError, match="read-only"):
        run_preflight(admission, model, expected_runtime_manifest_sha256=DIGEST)
    model.chmod(0o555)
    link = tmp_path / "link"
    link.symlink_to(model, target_is_directory=True)
    with pytest.raises(OSError):
        run_preflight(admission, link, expected_runtime_manifest_sha256=DIGEST)


def test_bind_admission_adds_identity_and_private_mode(tmp_path: Path) -> None:
    output = tmp_path / "bound.json"
    value = bind_admission(_admission(tmp_path / "legacy.json", digest=None), output, runtime_manifest_sha256=DIGEST, model_root=_model(tmp_path))
    assert value["runtime_manifest_sha256"] == DIGEST
    assert value["binding"] == "runtime-manifest-sha256"
    assert output.stat().st_mode & 0o777 == 0o600


def test_docker_argv_is_hardened_and_weights_read_only(tmp_path: Path) -> None:
    argv = build_docker_argv(image_ref=IMAGE, rendered=_rendered(), model_root=_model(tmp_path), cache_root=_cache(tmp_path), node_rank=0, manifest_sha256_value=DIGEST)
    joined = " ".join(argv)
    for expected in ("--pull never", "--network host", "--ipc host", "--gpus all", "--cap-drop ALL", "--security-opt no-new-privileges:true", "--read-only", "target=/models/Inkling-Small-NVFP4,readonly", "--device /dev/infiniband/rdma_cm", "--device /dev/infiniband/uverbs0", "--device /dev/infiniband/uverbs1", "--device /dev/infiniband/uverbs2", "--device /dev/infiniband/uverbs3", "FLASHINFER_DISABLE_JIT=1", "FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer", "NCCL_IB_HCA==rocep1s0f0", "NCCL_NET=IB", "TVM_FFI_CACHE_DIR=/cache/user-cache/.cache/tvm-ffi", "inkling.manifest_sha256=" + DIGEST):
        assert expected in joined
    assert "--user 1001:1001" in joined
    assert "--model" not in argv


def test_docker_argv_rejects_credentials_and_unsafe_cache(tmp_path: Path) -> None:
    model, cache = _model(tmp_path), _cache(tmp_path)
    with pytest.raises(ValueError, match="credential"):
        build_docker_argv(image_ref=IMAGE, rendered=_rendered(), model_root=model, cache_root=cache, node_rank=0, env_passthrough=["HF_TOKEN"])
    argv = build_docker_argv(image_ref=IMAGE, rendered=_rendered(), model_root=model, cache_root=cache, node_rank=0, env_passthrough={"HELION_AOT_AUTOTUNE": "none"})
    assert "HELION_AOT_AUTOTUNE=none" in argv
    cache.chmod(0o777)
    with pytest.raises(ValueError, match="group/world"):
        build_docker_argv(image_ref=IMAGE, rendered=_rendered(), model_root=model, cache_root=cache, node_rank=0)


def test_container_ownership_stop_and_live_boundary(tmp_path: Path) -> None:
    assert owned_container_name(0) == "inkling-sglang-tp2-rank0"
    assert build_stop_argv("inkling-sglang-tp2-rank1")[-1] == "inkling-sglang-tp2-rank1"
    with pytest.raises(ValueError):
        build_stop_argv("foreign")
    base = {"image_kind": "base", "image_ref": IMAGE, "argv": ["docker", "run"], "container_name": "inkling-sglang-tp2-rank0"}
    with pytest.raises(ValueError, match="derivative"):
        launch(base, receipt_path=tmp_path / "r.json", execute=True)


def test_launch_dry_run_and_ready_receipt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    spec = {"image_kind": "derivative", "image_ref": IMAGE, "argv": ["docker", "run", IMAGE], "container_name": "inkling-sglang-tp2-rank0", "manifest_sha256": DIGEST, "source_commit": "d" * 40, "node_rank": 0}
    assert launch(spec, receipt_path=tmp_path / "dry.json", execute=False)["status"] == "DRY_RUN"
    process = SimpleNamespace(pid=123, returncode=None, poll=lambda: None)
    monkeypatch.setattr("inkling_release.launch.subprocess.Popen", lambda *a, **k: process)
    monkeypatch.setattr("inkling_release.launch._probe_readiness", lambda *a, **k: True)
    receipt = tmp_path / "receipt.json"
    result = launch(spec, receipt_path=receipt, execute=True, readiness_url="http://127.0.0.1:30000/health")
    assert result["status"] == "READY"
    assert json.loads(receipt.read_text())["readiness"] == "PASS"