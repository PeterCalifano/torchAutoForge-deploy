"""Observable native manifest and interactive creation behavior."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def tool(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the native tool selected by CTest."""
    executable = os.environ.get("PTAFMODEL_EXECUTABLE")
    if not executable:
        pytest.skip("Native manifest tool was not built")
    return subprocess.run([executable, *args], capture_output=True, text=True, check=False)


def test_creation_validation_and_collision(tmp_path: Path) -> None:
    """Create a portable manifest without a model; refuse destructive replacement."""
    artifact = tmp_path / "model # one.onnx"
    manifest = tmp_path / "model.ptafmodel"
    assert (
        tool("init", str(artifact), "--task", "centroiding", "--output", str(manifest)).returncode
        == 0
    )
    data = json.loads(manifest.read_text())
    assert data["task"] == "centroiding" and "role" not in data
    assert data["artifact_path"] == artifact.name
    assert data["execution_target_priority"] == ["cpu"]
    assert data["allow_fallback"] is False
    assert tool("validate", str(manifest)).returncode == 0
    assert tool("validate", str(manifest), "--check-artifact").returncode != 0
    artifact.write_text("Deliberately not an ONNX model; validation must not load it.")
    assert tool("validate", str(manifest), "--check-artifact").returncode == 0
    before = manifest.read_bytes()
    assert (
        tool("init", str(artifact), "--task", "raw_tensor", "--output", str(manifest)).returncode
        != 0
    )
    assert manifest.read_bytes() == before


@pytest.mark.parametrize(
    "fragment",
    [
        '"role":"centroiding"',
        '"unknown":1',
        '"device_id":-1',
        '"device_id":2147483648',
        '"allow_fallback":"false"',
        '"execution_target_priority":[]',
        '"execution_target_priority":["cpu","cpu"]',
        '"backend":"imaginary"',
        '"backend":"tensorrt_engine"',
        '"artifact":"tensorrt_engine"',
        '"task":"raw_tensor"',
    ],
)
def test_invalid_fields(tmp_path: Path, fragment: str) -> None:
    """Reject invalid schema values, duplicate task keys, and incompatible artifacts."""
    manifest = tmp_path / "invalid.ptafmodel"
    manifest.write_text(
        '{"schema_version":1,"artifact_path":"absent.onnx","task":"centroiding",' + fragment + "}"
    )
    assert tool("validate", str(manifest)).returncode != 0


@pytest.mark.parametrize(
    "text",
    [
        "schema_version = 1\nartifact_path = model.onnx\nrole = centroiding\n",
        "{",
        "{}",
        '{"schema_version":2,"artifact_path":"a.onnx","task":"raw_tensor"}',
        '{"schema_version":1,"artifact_path":"a.onnx"}',
    ],
)
def test_invalid_documents(tmp_path: Path, text: str) -> None:
    """Reject legacy syntax, malformed JSON, missing keys, and unsupported versions."""
    path = tmp_path / "invalid.ptafmodel"
    path.write_text(text)
    assert tool("validate", str(path)).returncode != 0


def test_fractional_runtime_integer(tmp_path: Path) -> None:
    """Reject fractional values before conversion to native runtime integers."""
    path = tmp_path / "valid.ptafmodel"
    path.write_text(
        '{"schema_version":1,"artifact_path":"a.onnx","task":"raw_tensor","device_id":0.5}'
    )
    assert tool("validate", str(path)).returncode != 0


def wizard(tmp_path: Path, answers: str) -> subprocess.CompletedProcess[str]:
    """Drive the installed-tool interface through redirected interactive input."""
    executable = os.environ.get("PTAFMODEL_EXECUTABLE")
    if not executable:
        pytest.skip("Native manifest tool was not built")
    script = Path(__file__).resolve().parents[2] / "scripts/make_ptafmodel.py"
    return subprocess.run(
        [sys.executable, str(script), "--ptafmodel-executable", executable],
        cwd=tmp_path,
        input=answers,
        capture_output=True,
        text=True,
        check=False,
    )


def test_wizard_creation_and_cancel(tmp_path: Path) -> None:
    """Create through native defaults; cancellation and EOF leave no output."""
    assert wizard(tmp_path, "a.onnx\nraw_tensor\na.ptafmodel\n\nn\ny\n").returncode == 0
    assert json.loads((tmp_path / "a.ptafmodel").read_text())["task"] == "raw_tensor"
    assert wizard(tmp_path, "a.onnx\nraw_tensor\nb.ptafmodel\n\nn\nn\n").returncode == 0
    assert not (tmp_path / "b.ptafmodel").exists()
    assert wizard(tmp_path, "a.onnx\n").returncode == 0


def test_wizard_advanced_and_error(tmp_path: Path) -> None:
    """Pass advanced settings without shell interpretation or a second validator."""
    answers = "a # model.onnx\ncentroiding\nadvanced.ptafmodel\ncpu\ny\nonnxruntime\n0\n2\n3\nresize\nxy\ny\ny\n"
    assert wizard(tmp_path, answers).returncode == 0
    data = json.loads((tmp_path / "advanced.ptafmodel").read_text())
    assert data["intra_op_num_threads"] == 2 and data["inter_op_num_threads"] == 3
    assert data["preprocessing"] == "resize" and data["allow_fallback"] is True
    assert wizard(tmp_path, answers).returncode != 0


def test_wizard_missing_executable(tmp_path: Path) -> None:
    """Report a missing native prerequisite before prompting or writing."""
    script = Path(__file__).resolve().parents[2] / "scripts/make_ptafmodel.py"
    result = subprocess.run(
        [sys.executable, str(script), "--ptafmodel-executable", str(tmp_path / "absent")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0 and "Cannot locate" in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_creation_through_directory_symlink(tmp_path: Path) -> None:
    """Resolve generated relative paths against the physical manifest directory."""
    actual = tmp_path / "actual"
    actual.mkdir()
    link = tmp_path / "link"
    link.symlink_to(actual, target_is_directory=True)
    artifact = tmp_path / "artifact.onnx"
    artifact.touch()
    manifest = link / "model.ptafmodel"
    assert (
        tool("init", str(artifact), "--task", "raw_tensor", "--output", str(manifest)).returncode
        == 0
    )
    assert tool("validate", str(manifest), "--check-artifact").returncode == 0
