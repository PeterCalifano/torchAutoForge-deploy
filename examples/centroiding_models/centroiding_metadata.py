"""Centroiding model metadata and preprocessing identity."""

from __future__ import annotations
from pathlib import Path
import autoforge_deploy as ptaf
import PIL
from inference_output import Json


def metadata(
    source: Path,
    count: int,
    model: ptaf.CModelFacade | None = None,
    requested_model: Path | None = None,
) -> dict[str, Json]:
    """Describe effective facade metadata and this demo's preprocessing.

    Args:
        source: Supplied input location.
        count: Number of selected frames.
        model: Loaded facade, or None before model loading.
        requested_model: Supplied manifest or artifact path.

    Returns:
        Run metadata; wrapper-unavailable target priority is explicitly null."""
    result: dict[str, Json] = {
        "schema_version": 1,
        "model": None,
        "requested_model_path": str(requested_model) if requested_model is not None else "",
        "input": {
            "path": str(source),
            "kind": "directory" if source.is_dir() else "image",
            "ordering": "natural_filename",
            "selected_frame_count": count,
        },
        "preprocessing": {
            "library": f"Pillow {PIL.__version__}",
            "grayscale": "L uint8",
            "resize": "bilinear",
            "scale": 1.0 / 255.0,
        },
    }
    if model is not None:
        c = model.GetContract()
        r = c.runtime
        result["model"] = {
            "artifact_path": c.artifact_path,
            "config_path": c.config_path,
            "role": c.role,
            "backend_detail": c.backend_detail,
            "preprocessing": c.preprocessing,
            "postprocessing": c.postprocessing,
            "runtime": {
                "device_id": r.GetDeviceId(),
                "intra_op_num_threads": r.GetIntraOpNumThreads(),
                "inter_op_num_threads": r.GetInterOpNumThreads(),
                "allow_fallback": r.GetAllowFallback(),
                "backend": int(r.GetBackend()),
                "artifact": int(r.GetArtifact()),
                "execution_target_priority": None,
                "enable_profiling": r.GetEnableProfiling(),
                "log_id": r.GetLogId(),
                "tensorrt_optimization_profile_index": r.GetTensorRtOptimizationProfileIndex(),
            },
            "inputs": [
                {"name": t.name, "dtype": t.dtype, "shape": list(t.shape)} for t in c.inputs
            ],
            "outputs": [
                {"name": t.name, "dtype": t.dtype, "shape": list(t.shape)} for t in c.outputs
            ],
        }
    return result
