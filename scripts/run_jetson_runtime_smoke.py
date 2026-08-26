#!/usr/bin/env python3
"""Run target-owned ONNX Runtime and TensorRT smoke commands on Jetson."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Sequence


LOGGER = logging.getLogger("jetson_runtime_smoke")


@dataclass(frozen=True)
class SmokeArguments:
    """Validated command-line inputs for the runtime smoke sequence."""

    build_dir: Path
    ort_model: Path | None
    trt_engine: Path | None
    device: int
    iterations: int
    warmup: int
    with_tensorrt_ep: bool
    dry_run: bool
    benchmark_args: tuple[str, ...]


def build_parser() -> argparse.ArgumentParser:
    """Create the Jetson smoke command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Verify requested ORT providers and run short benchmark_model "
            "passes without model-specific code."
        )
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("build"),
        help="Build directory containing src/programs binaries (default: build)",
    )
    parser.add_argument(
        "--ort-model",
        type=Path,
        help="ONNX or .ptafmodel artifact for the ORT CUDA smoke",
    )
    parser.add_argument(
        "--trt-engine",
        type=Path,
        help="TensorRT .engine/.plan artifact for the standalone smoke",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA/TensorRT device id")
    parser.add_argument(
        "--iterations", type=int, default=5, help="Benchmark iterations (default: 5)"
    )
    parser.add_argument(
        "--warmup", type=int, default=1, help="Benchmark warmup iterations (default: 1)"
    )
    parser.add_argument(
        "--with-tensorrt-ep",
        action="store_true",
        help="Also require and smoke the ORT TensorRT Execution Provider",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing them"
    )
    parser.add_argument(
        "benchmark_args",
        nargs=argparse.REMAINDER,
        help="Arguments after -- are appended to each benchmark_model invocation",
    )
    return parser


def parse_arguments(argv: Sequence[str] | None = None) -> SmokeArguments:
    """Parse and validate command-line inputs.

    Args:
        argv: Optional argument vector excluding the program name.

    Returns:
        Validated immutable smoke configuration.
    """

    parser = build_parser()
    parsed = parser.parse_args(argv)
    if parsed.ort_model is None and parsed.trt_engine is None:
        parser.error("provide --ort-model and/or --trt-engine")
    if parsed.device < 0:
        parser.error("--device must be a non-negative integer")
    if parsed.iterations <= 0:
        parser.error("--iterations must be a positive integer")
    if parsed.warmup < 0:
        parser.error("--warmup must be a non-negative integer")

    benchmark_args = tuple(parsed.benchmark_args)
    if benchmark_args[:1] == ("--",):
        benchmark_args = benchmark_args[1:]

    return SmokeArguments(
        build_dir=parsed.build_dir,
        ort_model=parsed.ort_model,
        trt_engine=parsed.trt_engine,
        device=parsed.device,
        iterations=parsed.iterations,
        warmup=parsed.warmup,
        with_tensorrt_ep=parsed.with_tensorrt_ep,
        dry_run=parsed.dry_run,
        benchmark_args=benchmark_args,
    )


def require_executable(path: Path) -> None:
    """Raise an actionable error when a required program is unavailable."""

    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"Missing executable: {path}")


def require_artifact(path: Path, artifact_kind: str) -> None:
    """Raise an actionable error when a requested runtime artifact is missing."""

    if not path.is_file():
        raise ValueError(f"Missing {artifact_kind}: {path}")


def run_command(command: Sequence[str], dry_run: bool) -> None:
    """Report and optionally execute one subprocess without shell interpolation."""

    rendered_command = shlex.join(command)
    if dry_run:
        print(f"DRY RUN: {rendered_command}")
        return

    print(rendered_command)
    subprocess.run(command, check=True)


def make_benchmark_command(
    benchmark_program: Path,
    artifact: Path,
    arguments: SmokeArguments,
) -> list[str]:
    """Create the common portion of a benchmark_model command."""

    return [
        str(benchmark_program),
        str(artifact),
        "--device",
        str(arguments.device),
        "--no-fallback",
        "--iterations",
        str(arguments.iterations),
        "--warmup",
        str(arguments.warmup),
    ]


def run_smoke(arguments: SmokeArguments) -> None:
    """Execute the requested provider checks and runtime benchmark passes."""

    providers_program = arguments.build_dir / "src/programs/get_available_providers"
    benchmark_program = arguments.build_dir / "src/programs/benchmark_model"
    require_executable(providers_program)
    require_executable(benchmark_program)

    if not arguments.dry_run:
        if arguments.ort_model is not None:
            require_artifact(arguments.ort_model, "ORT model/config")
        if arguments.trt_engine is not None:
            require_artifact(arguments.trt_engine, "TensorRT engine")

    if arguments.ort_model is not None:
        run_command(
            [str(providers_program), "--require-targets", "cuda"],
            arguments.dry_run,
        )
        ort_command = make_benchmark_command(
            benchmark_program, arguments.ort_model, arguments
        )
        ort_command[2:2] = ["--targets", "cuda"]
        ort_command.extend(arguments.benchmark_args)
        run_command(ort_command, arguments.dry_run)

        if arguments.with_tensorrt_ep:
            run_command(
                [str(providers_program), "--require-targets", "tensorrt"],
                arguments.dry_run,
            )
            trt_ep_command = make_benchmark_command(
                benchmark_program, arguments.ort_model, arguments
            )
            trt_ep_command[2:2] = ["--targets", "tensorrt,cuda"]
            trt_ep_command.extend(arguments.benchmark_args)
            run_command(trt_ep_command, arguments.dry_run)

    if arguments.trt_engine is not None:
        trt_command = make_benchmark_command(
            benchmark_program, arguments.trt_engine, arguments
        )
        trt_command[2:2] = [
            "--backend",
            "tensorrt_engine",
            "--artifact",
            "tensorrt_engine",
            "--targets",
            "tensorrt",
        ]
        trt_command.extend(arguments.benchmark_args)
        run_command(trt_command, arguments.dry_run)

    if arguments.dry_run:
        print("jetson_runtime_smoke_dry_run=ok")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Jetson smoke command and return a process status."""

    logging.basicConfig(
        level=logging.INFO,
        format="[jetson_runtime_smoke][%(levelname)s] %(message)s",
    )
    try:
        run_smoke(parse_arguments(argv))
    except ValueError as error:
        LOGGER.error("%s", error)
        return 2
    except subprocess.CalledProcessError as error:
        LOGGER.error("Command failed with status %d", error.returncode)
        return error.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
