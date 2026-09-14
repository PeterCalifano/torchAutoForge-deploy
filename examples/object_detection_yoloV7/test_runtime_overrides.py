"""Check YOLO override behavior against the native manifest reader and facade."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("autoforge_deploy")
from run_object_detection import load_model, parse_arguments


@pytest.mark.parametrize(
    "target,device", [("manifest", None), ("cpu", None), ("manifest", 0), ("cpu", 0)]
)
def test_python_overrides_preserve_manifest(
    tmp_path: Path, target: str, device: int | None
) -> None:
    """Target and device overrides retain unrelated settings from a real manifest."""
    root = Path(__file__).resolve().parents[2]
    manifest = tmp_path / "model.ptafmodel"
    settings = {
        "schema_version": 1,
        "artifact_path": str(root / "tests/matlab/testSamples/tracedSampleModel.onnx"),
        "task": "object_detection",
        "backend": "onnxruntime",
        "execution_target_priority": ["cpu"],
        "allow_fallback": True,
        "device_id": 1,
        "intra_op_num_threads": 2,
        "inter_op_num_threads": 3,
        "tensorrt_optimization_profile_index": 2,
        "enable_profiling": True,
        "log_id": str(tmp_path / "preserved_profile"),
    }
    manifest.write_text(json.dumps(settings))
    args = [str(manifest), "unused.png", "--target", target]
    if device is not None:
        args += ["--device", str(device)]
    runtime = load_model(parse_arguments(args)).GetContract().runtime
    assert runtime.GetDeviceId() == (1 if device is None else device)
    assert runtime.GetIntraOpNumThreads() == 2
    assert runtime.GetInterOpNumThreads() == 3
    assert runtime.GetTensorRtOptimizationProfileIndex() == 2
    assert runtime.GetEnableProfiling()
    assert runtime.GetLogId() == settings["log_id"]
    assert runtime.GetAllowFallback() == (target == "manifest")


@pytest.mark.parametrize(
    "args,device",
    [
        ([], 1),
        (["--target", "cpu"], 1),
        (["--device", "0"], 0),
        (["--target", "cpu", "--device", "0"], 0),
    ],
)
def test_native_overrides_preserve_runtime(tmp_path: Path, args: list[str], device: int) -> None:
    """Exercise the native CLI when its external YOLO fixture is available."""
    executable = os.environ.get("YOLO_EXECUTABLE")
    root = Path(__file__).resolve().parents[2]
    model = root / "models/onnx/yolov7_640x640.onnx"
    image = Path(__file__).parent / "sample_data/horses.jpg"
    if not executable or not model.is_file() or not image.is_file():
        pytest.skip("Native YOLO executable and model/image fixtures are required")
    manifest = tmp_path / "model.ptafmodel"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_path": str(model),
                "task": "object_detection",
                "execution_target_priority": ["cpu"],
                "device_id": 1,
                "intra_op_num_threads": 2,
                "inter_op_num_threads": 3,
                "enable_profiling": True,
                "log_id": str(tmp_path / "preserved_profile"),
            }
        )
    )
    result = subprocess.run(
        [executable, str(manifest), str(image), *args], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert f"device_id={device}" in result.stdout
    assert list(tmp_path.glob("preserved_profile*.json"))
