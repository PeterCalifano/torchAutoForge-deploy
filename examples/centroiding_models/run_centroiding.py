"""Run independent image frames through the generated centroiding Python wrapper."""

from __future__ import annotations

import argparse
import sys
import time
import image_sequence as images
import inference_output as reports
from centroiding_metadata import metadata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import autoforge_deploy as ptaf
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class DemoOptions:
    """Validated options for one wrapper-backed centroiding run."""

    model_path: Path
    image_path: Path
    target: str
    device_id: int
    output_path: Path | None = None
    overlays: bool = False


@dataclass(frozen=True)
class CentroidResult:
    """One normalized centroid mapped into model and source-image pixels."""

    normalized_x: float
    normalized_y: float
    model_input_x_px: float
    model_input_y_px: float
    original_x_px: float
    original_y_px: float


def parse_arguments(arguments: Sequence[str] | None = None) -> DemoOptions:
    """Parse and validate command-line arguments.

    Args:
        arguments: Optional arguments excluding the executable name.

    Returns:
        Validated demo options.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run an image or directory through a centroiding manifest or raw ONNX "
            "using the generated autoforge_deploy wrapper."
        )
    )
    parser.add_argument("model_path", type=Path)
    parser.add_argument("image_path", metavar="input_path", type=Path)
    parser.add_argument("--output", type=Path, dest="output_path")
    parser.add_argument("--overlays", action="store_true")
    parser.add_argument(
        "--target",
        choices=("manifest", "cpu", "cuda"),
        default="manifest",
        help="Use configured runtime settings or force one execution target.",
    )
    parser.add_argument("--device", type=int, default=0, dest="device_id")
    parsed = parser.parse_args(arguments)

    if parsed.overlays and parsed.output_path is None:
        parser.error("--overlays requires --output")
    if parsed.device_id < 0:
        parser.error("--device must be non-negative")

    return DemoOptions(
        model_path=parsed.model_path,
        image_path=parsed.image_path,
        target=parsed.target,
        device_id=parsed.device_id,
        output_path=parsed.output_path,
        overlays=parsed.overlays,
    )


def make_runtime(options: DemoOptions) -> ptaf.SRuntimeConfig:
    """Build a strict runtime override from validated options.

    Args:
        options: Validated target and device selection.

    Returns:
        Backend-neutral runtime configuration.
    """

    runtime = ptaf.SRuntimeConfig()
    runtime.SetDeviceId(options.device_id)
    runtime.ClearExecutionTargetPriority()
    target = ptaf.EExecutionTarget.cpu if options.target == "cpu" else ptaf.EExecutionTarget.cuda
    runtime.AddExecutionTarget(target)
    runtime.SetAllowFallback(False)
    return runtime


def load_model(options: DemoOptions) -> ptaf.CModelFacade:
    """Load a centroiding manifest or ONNX artifact.

    Args:
        options: Validated model and runtime selection.

    Returns:
        Loaded role-level facade.

    Raises:
        ValueError: If the artifact extension is unsupported.
        RuntimeError: If the loaded model contract is not image-only centroiding.
    """

    model = ptaf.CModelFacade()
    extension = options.model_path.suffix.lower()
    has_override = options.target != "manifest"

    if extension == ".ptafmodel":
        if has_override:
            model.LoadModelConfigWithRuntimeConfig(str(options.model_path), make_runtime(options))
        else:
            model.LoadModelConfig(str(options.model_path))
    elif extension == ".onnx":
        if has_override:
            model.LoadModelWithRoleAndRuntimeConfig(
                str(options.model_path),
                ptaf.EModelRole.centroiding,
                make_runtime(options),
            )
        else:
            model.LoadModelWithRole(str(options.model_path), ptaf.EModelRole.centroiding)
    else:
        raise ValueError("Expected a .ptafmodel manifest or .onnx artifact")

    if model.GetRole() != "centroiding":
        raise RuntimeError("Centroiding requires role=centroiding")
    if model.GetNumInputs() != 1 or model.GetNumOutputs() != 1:
        raise RuntimeError("Centroiding requires exactly one input and one output")
    return model


def prepare_image(
    model: ptaf.CModelFacade, image_path: Path
) -> tuple[ptaf.SFloatTensor, tuple[int, int]]:
    """Decode, resize, and convert one image to a wrapper-owned NCHW tensor.

    Args:
        model: Loaded single-input centroiding facade.
        image_path: Image path understood by Pillow.

    Returns:
        Prepared tensor and original ``(width, height)``.

    Raises:
        RuntimeError: If the input tensor contract is unsupported.
    """

    input_info = model.GetInputInfo(0)
    shape = list(input_info.shape)
    if (
        input_info.dtype != "float32"
        or len(shape) != 4
        or shape[0] not in (-1, 1)
        or shape[1] != 1
        or shape[2] <= 0
        or shape[3] <= 0
    ):
        raise RuntimeError("Centroiding expects float32 input [N,1,H,W] with concrete H/W")

    height, width = int(shape[2]), int(shape[3])
    with Image.open(image_path) as source_image:
        grayscale_image = source_image.convert("L")
        original_size = grayscale_image.size
        resized_image = grayscale_image.resize((width, height), Image.Resampling.BILINEAR)
        hwc_values = np.asarray(resized_image, dtype=np.float32).reshape(-1)

    tensor = ptaf.MakeNchwFloatTensorFromHwcFloat(
        input_info.name,
        hwc_values,
        height,
        width,
        1,
        1.0 / 255.0,
        False,
    )
    return tensor, original_size


def decode_centroid(
    output: ptaf.SFloatTensor,
    model_input_size: tuple[int, int],
    original_size: tuple[int, int],
) -> CentroidResult:
    """Decode and map one normalized wrapper output.

    Args:
        output: Sole model output with concrete shape ``[1,2]``.
        model_input_size: Model ``(width, height)``.
        original_size: Decoded image ``(width, height)``.

    Returns:
        Validated coordinates in all supported spaces.

    Raises:
        RuntimeError: If the selected output contract is invalid.
    """

    if list(output.shape) != [1, 2]:
        raise RuntimeError("Centroiding expects output shape [1,2]")

    schema = ptaf.SFeatureRowSchema()
    features = ptaf.DecodeFeatureRows(output, schema)
    if len(features) != 1:
        raise RuntimeError("Centroiding must produce exactly one feature")

    normalized_x = float(features[0].position.x)
    normalized_y = float(features[0].position.y)
    if not np.isfinite(normalized_x) or not np.isfinite(normalized_y):
        raise RuntimeError("Centroiding coordinates must be finite")

    model_width, model_height = model_input_size
    original_width, original_height = original_size
    return CentroidResult(
        normalized_x=normalized_x,
        normalized_y=normalized_y,
        model_input_x_px=normalized_x * model_width,
        model_input_y_px=normalized_y * model_height,
        original_x_px=normalized_x * original_width,
        original_y_px=normalized_y * original_height,
    )


def print_result(
    model: ptaf.CModelFacade,
    input_tensor: ptaf.SFloatTensor,
    output_tensor: ptaf.SFloatTensor,
    original_size: tuple[int, int],
    result: CentroidResult,
) -> None:
    """Print stable fields shared with the native and MATLAB demos.

    Args:
        model: Loaded role-level facade.
        input_tensor: Prepared model input.
        output_tensor: Raw model output.
        original_size: Decoded image ``(width, height)``.
        result: Validated centroid coordinates.
    """

    print(f"role={model.GetRole()}")
    print(f"backend={model.GetBackendDetail()}")
    print(f"input_name={input_tensor.name}")
    print(f"input_shape={list(input_tensor.shape)}")
    print(f"image_size={list(original_size)}")
    print(f"output_name={output_tensor.name}")
    print(f"output_shape={list(output_tensor.shape)}")
    print(f"centroid.normalized_x={result.normalized_x:.9g}")
    print(f"centroid.normalized_y={result.normalized_y:.9g}")
    print(f"centroid.model_input_x_px={result.model_input_x_px:.9g}")
    print(f"centroid.model_input_y_px={result.model_input_y_px:.9g}")
    print(f"centroid.original_x_px={result.original_x_px:.9g}")
    print(f"centroid.original_y_px={result.original_y_px:.9g}")


def run_demo(options: DemoOptions) -> None:
    """Run one model load and independent frame inferences with optional reports.

    Args:
        options: Validated demo options.

    Raises:
        FileNotFoundError: If the model or image does not exist.
    """

    frames = images.select_frames(options.image_path)
    report = None
    if options.output_path is not None:
        reports.prepare_output(options.output_path, options.image_path)
        report = reports.Report(
            options.output_path,
            metadata(options.image_path, len(frames), requested_model=options.model_path),
        )
    index = None
    source = None
    stage = "model_load"
    complete = False
    try:
        model = load_model(options)
        if report is not None:
            report.metadata = metadata(options.image_path, len(frames), model, options.model_path)
        if options.overlays:
            if options.output_path is None:
                raise ValueError("--overlays requires --output")
            (options.output_path / "overlays").mkdir()
        for index, source in enumerate(frames):
            stage = "preprocessing"
            input_tensor, original_size = prepare_image(model, source)
            stage = "inference"
            start = time.perf_counter()
            output_tensor = model.InferSingleFloatTensor(input_tensor)
            duration = (time.perf_counter() - start) * 1000.0
            stage = "decoding_output"
            model_size = (int(input_tensor.shape[3]), int(input_tensor.shape[2]))
            result = decode_centroid(output_tensor, model_size, original_size)
            print(f"frame.index={index}\nframe.source={source.name}\ninference_ms={duration}")
            print_result(model, input_tensor, output_tensor, original_size, result)
            overlay = None
            if options.overlays and options.output_path is not None:
                stage = "overlay"
                overlay = f"overlays/{index:06d}_{source.stem}.png"
                images.save_overlay(
                    source,
                    options.output_path / overlay,
                    result.original_x_px,
                    result.original_y_px,
                )
            stage = "report"
            if report is not None:
                report.append(
                    {
                        "index": index,
                        "source": source.name,
                        "image_size": {"width": original_size[0], "height": original_size[1]},
                        "raw_output": {
                            "name": output_tensor.name,
                            "shape": list(output_tensor.shape),
                            "values": [float(v) for v in output_tensor.values],
                        },
                        "centroid": {
                            "normalized": {"x": result.normalized_x, "y": result.normalized_y},
                            "model_pixels": {
                                "x": result.model_input_x_px,
                                "y": result.model_input_y_px,
                            },
                            "image_pixels": {"x": result.original_x_px, "y": result.original_y_px},
                            "inside_image": (
                                0 <= result.original_x_px < original_size[0]
                                and 0 <= result.original_y_px < original_size[1]
                            ),
                        },
                        "inference_ms": duration,
                        "overlay": overlay,
                    }
                )
        index = None
        source = None
        stage = "report"
        if report is not None:
            report.publish(True)
        complete = True
    except Exception as error:
        if report is not None:
            try:
                report.publish(
                    False,
                    {
                        "stage": stage,
                        "frame_index": index,
                        "source": source.name if source else None,
                        "message": str(error),
                    },
                )
            except OSError as publication:
                print(
                    f"Report publication failed: {publication}; retained files in {report.root}",
                    file=sys.stderr,
                )
        raise
    finally:
        if report is not None:
            report.close(complete)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the command-line demo and translate failures to process status.

    Args:
        arguments: Optional arguments excluding the executable name.

    Returns:
        Zero on success and one on a model, image, or runtime failure.
    """

    try:
        run_demo(parse_arguments(arguments))
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        print(f"run_centroiding.py failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
