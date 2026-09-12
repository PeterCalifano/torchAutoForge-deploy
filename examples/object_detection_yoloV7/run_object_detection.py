"""Run a YOLOv7 image through the generated PTAF Deploy Python wrapper."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
from PIL import Image

import autoforge_deploy as ptaf


@dataclass(frozen=True)
class DemoOptions:
    """Validated command-line options for one YOLO inference."""

    manifest_path: Path
    image_path: Path
    target: str
    device_id: int
    score_threshold: float
    max_detections: int


def parse_arguments(arguments: Sequence[str] | None = None) -> DemoOptions:
    """Parse command-line arguments.

    Args:
        arguments: Optional argument sequence. Defaults to ``sys.argv``.

    Returns:
        Validated demo options.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run one image through CModelFacade and a generic row decoder. "
            "Results are pre-NMS and no annotated image is rendered."
        )
    )
    parser.add_argument("manifest_path", type=Path)
    parser.add_argument("image_path", type=Path)
    parser.add_argument(
        "--target",
        choices=("manifest", "cpu", "cuda"),
        default="manifest",
        help="Use manifest runtime settings or force one execution target.",
    )
    parser.add_argument("--device", type=int, default=0, dest="device_id")
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--max-detections", type=int, default=20)
    parsed = parser.parse_args(arguments)

    if parsed.device_id < 0:
        parser.error("--device must be non-negative")
    if not np.isfinite(parsed.score_threshold) or parsed.score_threshold < 0.0:
        parser.error("--score-threshold must be finite and non-negative")
    if parsed.max_detections < 0:
        parser.error("--max-detections must be non-negative")

    return DemoOptions(
        manifest_path=parsed.manifest_path,
        image_path=parsed.image_path,
        target=parsed.target,
        device_id=parsed.device_id,
        score_threshold=parsed.score_threshold,
        max_detections=parsed.max_detections,
    )


def load_model(options: DemoOptions) -> ptaf.CModelFacade:
    """Load the model manifest with optional execution-target override.

    Args:
        options: Validated demo options.

    Returns:
        Loaded object-detection facade.

    Raises:
        RuntimeError: If the manifest is not an object-detection contract.
    """
    model = ptaf.CModelFacade()
    if options.target == "manifest":
        model.LoadModelConfig(str(options.manifest_path))
    else:
        runtime = ptaf.SRuntimeConfig()
        runtime.SetDeviceId(options.device_id)
        runtime.ClearExecutionTargetPriority()
        target = (
            ptaf.EExecutionTarget.cpu
            if options.target == "cpu"
            else ptaf.EExecutionTarget.cuda
        )
        runtime.AddExecutionTarget(target)
        runtime.SetAllowFallback(False)
        model.LoadModelConfigWithRuntimeConfig(str(options.manifest_path), runtime)

    if model.GetRole() != "object_detection":
        raise RuntimeError("YOLO demo requires role=object_detection in the manifest")
    if model.GetNumInputs() != 1 or model.GetNumOutputs() != 1:
        raise RuntimeError("YOLO demo currently requires one model input and one output")
    return model


def prepare_image(model: ptaf.CModelFacade, image_path: Path) -> ptaf.SFloatTensor:
    """Resize an RGB image and invoke shared HWC-to-NCHW preprocessing.

    Args:
        model: Loaded single-input model facade.
        image_path: Input image path.

    Returns:
        Wrapper-owned float32 NCHW tensor.

    Raises:
        RuntimeError: If the model input is not concrete float32 NCHW RGB.
    """
    input_info = model.GetInputInfo(0)
    shape = list(input_info.shape)
    if (
        input_info.dtype != "float32"
        or len(shape) != 4
        or shape[0] != 1
        or shape[1] != 3
        or shape[2] <= 0
        or shape[3] <= 0
    ):
        raise RuntimeError(
            "YOLO demo expects one float32 input with concrete shape [1,3,H,W]"
        )

    height, width = int(shape[2]), int(shape[3])
    with Image.open(image_path) as source_image:
        rgb_image = source_image.convert("RGB").resize(
            (width, height), Image.Resampling.BILINEAR
        )
        hwc_values = np.asarray(rgb_image, dtype=np.uint8).reshape(-1)

    return ptaf.MakeNchwFloatTensorFromHwcFloat(
        input_info.name,
        hwc_values,
        height,
        width,
        3,
        1.0 / 255.0,
        False,
    )


def print_detections(
    model: ptaf.CModelFacade,
    input_tensor: ptaf.SFloatTensor,
    output_tensor: ptaf.SFloatTensor,
    detections: Sequence[ptaf.SDetection2D],
) -> None:
    """Print a stable summary shared with the native and MATLAB demos.

    Args:
        model: Loaded model facade.
        input_tensor: Prepared model input.
        output_tensor: Raw model output.
        detections: Score-sorted pre-NMS detections.
    """
    print(f"role={model.GetRole()}")
    print(f"backend={model.GetBackendDetail()}")
    print(f"input_name={input_tensor.name}")
    print(f"input_shape={list(input_tensor.shape)}")
    print(f"output_name={output_tensor.name}")
    print(f"output_shape={list(output_tensor.shape)}")
    print(f"detections={len(detections)}")
    for index, detection in enumerate(detections):
        print(
            f"detection[{index}]=class={detection.classification.class_id},"
            f"score={detection.classification.score:.6f},"
            f"cx={detection.bounds.center.x:.3f},"
            f"cy={detection.bounds.center.y:.3f},"
            f"w={detection.bounds.size.width:.3f},"
            f"h={detection.bounds.size.height:.3f}"
        )


def run_demo(options: DemoOptions) -> None:
    """Run one facade-backed YOLO inference.

    Args:
        options: Validated demo options.
    """
    if not options.manifest_path.is_file():
        raise FileNotFoundError(f"Missing model manifest: {options.manifest_path}")
    if not options.image_path.is_file():
        raise FileNotFoundError(f"Missing input image: {options.image_path}")

    model = load_model(options)
    input_tensor = prepare_image(model, options.image_path)
    output_tensor = model.InferSingleFloatTensor(input_tensor)

    output_shape = list(output_tensor.shape)
    if not output_shape or output_shape[-1] < 6:
        raise RuntimeError(
            "YOLO demo expects output rows [cx,cy,w,h,objectness,class_scores...]"
        )

    schema = ptaf.SDetectionRowSchema()
    schema.box_encoding = ptaf.EBoundingBoxEncoding.center_xywh
    schema.box_coordinate_0_index = 0
    schema.box_coordinate_1_index = 1
    schema.box_coordinate_2_index = 2
    schema.box_coordinate_3_index = 3
    schema.objectness_index = 4
    schema.first_class_score_index = 5
    schema.class_score_count = output_shape[-1] - 5
    detections = ptaf.DecodeDetectionRows(
        output_tensor,
        schema,
        options.score_threshold,
        options.max_detections,
    )
    print_detections(model, input_tensor, output_tensor, detections)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the command-line demo and convert runtime errors to exit status.

    Args:
        arguments: Optional argument sequence. Defaults to ``sys.argv``.

    Returns:
        Zero on success, one on runtime failure.
    """
    try:
        run_demo(parse_arguments(arguments))
    except Exception as error:  # Boundary: provide one actionable CLI failure.
        print(f"run_object_detection.py failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
