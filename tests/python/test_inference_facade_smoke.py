"""Exercise target-owned Python inference, facade, and task-adapter bindings."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import autoforge_deploy as ptaf


@dataclass(frozen=True)
class SmokePaths:
    """Runtime fixture paths used by the Python wrapper smoke."""

    model: Path
    config: Path


def parse_arguments(argv: Sequence[str] | None = None) -> SmokePaths:
    """Parse required model and manifest paths.

    Args:
        argv: Optional argument vector excluding the program name.

    Returns:
        Validated fixture paths.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parsed = parser.parse_args(argv)
    if not parsed.model.is_file():
        parser.error(f"model does not exist: {parsed.model}")
    if not parsed.config.is_file():
        parser.error(f"config does not exist: {parsed.config}")
    return SmokePaths(model=parsed.model, config=parsed.config)


def require(condition: bool, message: str) -> None:
    """Raise an actionable smoke failure when a contract is not satisfied."""

    if not condition:
        raise RuntimeError(message)


def verify_generic_task_adapter() -> None:
    """Verify composed detection records survive the generated Python wrapper."""

    schema = ptaf.SDetectionRowSchema()
    schema.box_encoding = ptaf.EBoundingBoxEncoding.center_xywh
    schema.box_coordinate_0_index = 0
    schema.box_coordinate_1_index = 1
    schema.box_coordinate_2_index = 2
    schema.box_coordinate_3_index = 3
    schema.objectness_index = 4
    schema.first_class_score_index = 5
    schema.class_score_count = 2

    tensor = ptaf.SFloatTensor(
        "detections",
        [1, 7],
        [10.0, 20.0, 30.0, 40.0, 0.9, 0.1, 0.8],
    )
    detections = ptaf.DecodeDetectionRows(tensor, schema, 0.5, 0)
    require(len(detections) == 1, "expected one decoded detection")
    detection = detections[0]
    require(abs(detection.bounds.center.x - 10.0) < 1.0e-6, "unexpected box center")
    require(detection.classification.class_id == 1, "unexpected class id")
    require(
        abs(detection.classification.score - 0.72) < 1.0e-6,
        "unexpected detection score",
    )


def verify_generic_feature_adapter() -> None:
    """Verify generic feature rows survive the generated Python wrapper."""

    schema = ptaf.SFeatureRowSchema()
    tensor = ptaf.SFloatTensor("features", [1, 2], [0.25, 0.75])
    features = ptaf.DecodeFeatureRows(tensor, schema)

    require(len(features) == 1, "expected one decoded feature")
    require(abs(features[0].position.x - 0.25) < 1.0e-6, "unexpected feature x")
    require(abs(features[0].position.y - 0.75) < 1.0e-6, "unexpected feature y")
    require(abs(features[0].score - 1.0) < 1.0e-6, "unexpected feature score")


def run_smoke(paths: SmokePaths) -> None:
    """Exercise the generated manager, facade, and generic adapter bindings."""

    require(getattr(ptaf, "HAS_WRAPPER", False), "expected HAS_WRAPPER=True")
    require(hasattr(ptaf, "CInferenceManager"), "missing CInferenceManager wrapper")
    require(hasattr(ptaf, "CModelFacade"), "missing CModelFacade wrapper")

    input_values = [float(value) for value in range(1, 12)]
    manager = ptaf.CInferenceManager(str(paths.model))
    outputs = manager.InferSingleFloatInput(input_values, [1, 11])
    require(len(outputs) == 2, "manager fixture must return two values")

    model = ptaf.CModelFacade()
    model.LoadModelConfig(str(paths.config))
    require(model.GetRole() == "centroiding", "manifest role was not preserved")
    facade_outputs = model.InferSingleFloatInput(input_values, [1, 11])
    require(len(facade_outputs) == 2, "facade fixture must return two values")
    verify_generic_task_adapter()
    verify_generic_feature_adapter()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Python wrapper smoke and return a process status."""

    try:
        run_smoke(parse_arguments(argv))
    except (RuntimeError, ValueError) as error:
        print(f"python_facade_smoke error: {error}", file=sys.stderr)
        return 1

    print("python_facade_smoke=ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
