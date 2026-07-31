from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCANNER = ROOT / "scripts" / "public_safety_scan.py"


def _git_root(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "tracked.txt").write_text("public\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.txt"], check=True)
    return tmp_path


def test_scan_includes_untracked_nonignored_files(tmp_path: Path) -> None:
    root = _git_root(tmp_path)
    private_prefix = "/" + "home" + "/"
    (root / "untracked.txt").write_text(private_prefix + "example/private\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SCANNER), str(root)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "PUBLIC_SAFETY_SCAN=FAIL" in result.stdout
    assert "untracked.txt: home-path" in result.stdout


def test_scan_accepts_clean_tracked_and_untracked_files(tmp_path: Path) -> None:
    root = _git_root(tmp_path)
    (root / "untracked.txt").write_text("also public\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SCANNER), str(root)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PUBLIC_SAFETY_SCAN=PASS roots=1" in result.stdout