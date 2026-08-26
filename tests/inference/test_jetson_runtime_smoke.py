"""Target-owned regression tests for the Jetson runtime smoke CLI."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def _make_executable(path: Path) -> None:
    """Create the executable placeholder required by dry-run validation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    path.chmod(0o755)


def test_benchmark_separator_is_not_forwarded(tmp_path: Path) -> None:
    """Keep the CLI-only `--` separator out of benchmark_model arguments."""

    build_dir = tmp_path / "build"
    _make_executable(build_dir / "src/programs/get_available_providers")
    _make_executable(build_dir / "src/programs/benchmark_model")
    script_path = Path(__file__).parents[2] / "scripts/run_jetson_runtime_smoke.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--dry-run",
            "--build-dir",
            str(build_dir),
            "--ort-model",
            "model.onnx",
            "--",
            "--role",
            "raw_tensor",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert " --role raw_tensor" in result.stdout
    assert " -- --role raw_tensor" not in result.stdout
