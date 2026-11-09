"""Verify the wrapped backend build report matches the configured backends."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass

import autoforge_deploy as ptaf


@dataclass(frozen=True)
class ExpectedBackends:
    """Backend options the wrapper was configured with."""

    onnxruntime: bool
    tensorrt_engine: bool


def parse_bool(value: str) -> bool:
    """Parse a CMake-style boolean value."""

    normalized = value.strip().upper()
    if normalized in {"ON", "TRUE", "1", "YES"}:
        return True
    if normalized in {"OFF", "FALSE", "0", "NO", ""}:
        return False
    raise argparse.ArgumentTypeError(f"expected a CMake boolean, got {value!r}")


def parse_arguments(argv: Sequence[str] | None = None) -> ExpectedBackends:
    """Parse the expected backend configuration.

    Args:
        argv: Optional argument vector excluding the program name.

    Returns:
        Expected compiled-in state for each backend.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-ort", required=True, type=parse_bool)
    parser.add_argument("--expect-tensorrt", required=True, type=parse_bool)
    parsed = parser.parse_args(argv)
    return ExpectedBackends(onnxruntime=parsed.expect_ort, tensorrt_engine=parsed.expect_tensorrt)


def require(condition: bool, message: str) -> None:
    """Raise an actionable smoke failure when a contract is not satisfied."""

    if not condition:
        raise RuntimeError(message)


def verify_report(expected: ExpectedBackends) -> None:
    """Check every wrapped report entry against the configured backends."""

    report = {info.name: info for info in ptaf.GetBackendBuildInfo()}
    require(
        sorted(report) == ["onnxruntime", "tensorrt_engine"],
        f"unexpected backend names: {sorted(report)}",
    )

    for name, compiled in (
        ("onnxruntime", expected.onnxruntime),
        ("tensorrt_engine", expected.tensorrt_engine),
    ):
        info = report[name]
        require(info.compiled_in == compiled, f"{name}: compiled_in={info.compiled_in}")
        require(bool(info.version) == compiled, f"{name}: unexpected version {info.version!r}")
        require(bool(info.note) != compiled, f"{name}: unexpected note {info.note!r}")
        if not compiled:
            require(not info.runtime_available, f"{name}: available but not compiled")

    ort = report["onnxruntime"]
    require(ort.backend == ptaf.EInferenceBackend.onnxruntime, "unexpected ORT backend enum")
    require(bool(list(ort.providers)) == expected.onnxruntime, "unexpected ORT providers")
    require(
        ptaf.IsOrtExecutionTargetAvailable(ptaf.EExecutionTarget.cpu) == expected.onnxruntime,
        "unexpected ORT CPU target availability",
    )
    require(
        "onnxruntime: compiled=" in ptaf.FormatBackendBuildInfo(),
        "formatted report is missing the onnxruntime line",
    )
    require(
        ptaf.FormatBackendBuildInfo(list(report.values())) == ptaf.FormatBackendBuildInfo(),
        "formatting an explicit report differs from the current build report",
    )


def verify_disabled_ort_error(expected: ExpectedBackends) -> None:
    """Check that loading an ONNX artifact without ORT raises the not-built error."""

    if expected.onnxruntime:
        return

    try:
        ptaf.CInferenceManager().LoadModel("missing.onnx")
    except RuntimeError as error:
        require("not built" in str(error), f"unexpected ORT-off error: {error}")
        return
    raise RuntimeError("loading an ONNX artifact without ORT did not fail")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the backend build-info smoke and return a process status."""

    expected = parse_arguments(argv)
    try:
        verify_report(expected)
        verify_disabled_ort_error(expected)
    except RuntimeError as error:
        print(f"python_backend_build_info error: {error}", file=sys.stderr)
        return 1

    print("python_backend_build_info=ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
