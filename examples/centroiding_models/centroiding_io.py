"""Sequence selection and JSON publication private to the centroiding demo."""

from __future__ import annotations

import json
import re
from functools import cmp_to_key
from pathlib import Path

import autoforge_deploy as ptaf
import numpy as np
import PIL
from PIL import Image

type Json = None | bool | int | float | str | list[Json] | dict[str, Json]


def natural_compare(a: str, b: str) -> int:
    """Compare filenames by ASCII digit value, then original spelling for ties.

    Args:
        a: First Unicode filename.
        b: Second Unicode filename.

    Returns:
        Negative, zero, or positive for the documented total ordering."""
    left, right = re.findall(r"[0-9]+|[^0-9]", a), re.findall(r"[0-9]+|[^0-9]", b)
    for x, y in zip(left, right):
        if x[0] in "0123456789" and y[0] in "0123456789":
            x, y = x.lstrip("0"), y.lstrip("0")
            if len(x) != len(y):
                return -1 if len(x) < len(y) else 1
        if x != y:
            return -1 if x < y else 1
    if len(left) != len(right):
        return -1 if len(left) < len(right) else 1
    return (a > b) - (a < b)


def select_frames(path: Path) -> list[Path]:
    """Select supported regular images without descending into subdirectories.

    Args:
        path: Image or directory supplied by the caller.

    Returns:
        Naturally ordered paths, with no retained image data.

    Raises:
        FileNotFoundError: If the input is absent.
        ValueError: If a directory contains no supported images.
        OSError: If filesystem inspection fails."""
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Missing input: {path}")
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    frames = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in extensions]
    frames.sort(key=cmp_to_key(lambda a, b: natural_compare(a.name, b.name)))
    if not frames:
        raise ValueError(f"No supported images: {path}")
    return frames


def prepare_output(output: Path, source: Path) -> None:
    """Reserve a new or empty directory which does not contain the input.

    Args:
        output: Destination exclusively owned by this run.
        source: Input location, resolved through symlinks for comparison.

    Raises:
        ValueError: If output conflicts with input or existing results.
        OSError: If filesystem inspection or directory creation fails."""
    if source.resolve().is_relative_to(output.resolve()):
        raise ValueError("Output must not equal or contain the input location")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output must be a new or empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)


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


class Report:
    """Publish a report from an on-disk spool without retaining frame records."""

    root: Path
    metadata: dict[str, Json]
    committed_frames: int

    def __init__(self, root: Path, initial: dict[str, Json]) -> None:
        """Create the initial incomplete report in a previously reserved directory.

        Args:
            root: Empty output directory exclusively owned by this run.
            initial: Metadata returned before model loading.

        Raises:
            OSError: If the spool or initial report cannot be created.
        """
        self.root, self.metadata = root, initial
        self.committed_frames = 0
        with (root / "frames.jsonl").open("xb"):
            pass
        self.publish(False)

    def append(self, frame: dict[str, Json]) -> None:
        """Flush one successfully completed frame to the spool.

        Args:
            frame: Validated frame record with finite numerical values.

        Raises:
            ValueError: If a value is not JSON-serializable or finite.
            OSError: If a record cannot be written and closed successfully."""
        record = (json.dumps(frame, allow_nan=False) + "\n").encode("utf-8")
        with (self.root / "frames.jsonl").open("ab") as spool:
            if spool.write(record) != len(record):
                raise OSError("Incomplete frame spool write")
        self.committed_frames += 1

    def publish(self, complete: bool, error: dict[str, Json] | None = None) -> None:
        """Atomically replace the report, keeping the prior file on IO failure.

        Args:
            complete: Whether every selected frame succeeded.
            error: Failure details, or None on success.

        Raises:
            OSError: If spool reading, report writing, or replacement fails."""
        header = dict(self.metadata, status="complete" if complete else "incomplete")
        if error is not None:
            header["error"] = error
        temporary = self.root / "predictions.json.tmp"
        with temporary.open("w", encoding="utf-8") as destination:
            destination.write(json.dumps(header, allow_nan=False)[:-1] + ', "frames":[')
            with (self.root / "frames.jsonl").open(encoding="utf-8") as records:
                # A failed append may leave an uncommitted tail; never publish that tail.
                for index in range(self.committed_frames):
                    record = records.readline()
                    if not record.endswith("\n"):
                        raise OSError("Incomplete frame spool record")
                    destination.write(("," if index else "") + record.rstrip("\n"))
            destination.write("]}\n")
        temporary.replace(self.root / "predictions.json")

    def close(self, complete: bool) -> None:
        """Remove the spool after success; retain it on failure for diagnostic recovery.

        Args:
            complete: Whether a complete report was successfully published."""
        if complete:
            try:
                (self.root / "frames.jsonl").unlink()
            except OSError:
                pass  # Published results remain valid if spool cleanup is unavailable.


def save_overlay(source: Path, destination: Path, x: float, y: float) -> None:
    """Save an overlay with naturally clipped contrasting crosshair strokes.

    Args:
        source: Original image; source bytes remain unchanged.
        destination: Reserved PNG path.
        x: Finite zero-origin horizontal coordinate, without clamping.
        y: Finite zero-origin vertical coordinate, without clamping.

    Raises:
        ValueError: If image samples cannot be preserved in the supported PNG modes.
        OSError: If image decoding or writing fails."""
    with Image.open(source) as image:
        # Preserve high-bit-depth grayscale samples and RGB/RGBA appearance.
        if image.mode in ("I", "I;16", "I;16B"):
            values = np.asarray(image)
            if np.any(values < 0) or np.any(values > 65535):
                raise ValueError("PNG overlays require unsigned 8-bit or 16-bit image samples")
            pixels = values.astype(np.uint16, copy=True)
            white = 65535
        elif image.mode == "F":
            raise ValueError("PNG overlays require unsigned 8-bit or 16-bit image samples")
        else:
            pixels = np.array(
                image.convert(
                    "RGBA" if "A" in image.getbands() or "transparency" in image.info else "RGB"
                )
            )
            white = 255
    height, width = pixels.shape[:2]
    if -10 <= x < width + 10 and -10 <= y < height + 10:
        x, y = int(np.floor(x + 0.5)), int(np.floor(y + 0.5))
        for value, radius, half_width in [(0, 9, 1), (white, 8, 0)]:
            horizontal = pixels[
                max(0, y - half_width) : max(0, min(height, y + half_width + 1)),
                max(0, x - radius) : max(0, min(width, x + radius + 1)),
            ]
            vertical = pixels[
                max(0, y - radius) : max(0, min(height, y + radius + 1)),
                max(0, x - half_width) : max(0, min(width, x + half_width + 1)),
            ]
            horizontal[...] = value
            vertical[...] = value
            if pixels.ndim == 3 and pixels.shape[2] == 4:
                horizontal[..., 3] = 255
                vertical[..., 3] = 255
    Image.fromarray(pixels).save(destination)
