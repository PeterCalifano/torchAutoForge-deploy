"""Incremental JSON publication independent of centroid interpretation."""

from __future__ import annotations
import json
from pathlib import Path

type Json = None | bool | int | float | str | list[Json] | dict[str, Json]


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
