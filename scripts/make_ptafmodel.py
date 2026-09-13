#!/usr/bin/env python3
"""Interactively create a manifest through the native make-ptafmodel tool.

Requires Python 3.12 and an installed make-ptafmodel executable. Serialization,
validation, and omitted-setting defaults belong to the native implementation.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from typing import Any


def prompt(text: str, *, required: bool = False) -> str:
    """Read a value, repeating only when a required response is empty."""
    while True:
        value = input(f"{text}: ").strip()
        if value or not required:
            return value


def choose(text: str, choices: list[str], *, required: bool = False) -> str:
    """Offer schema-provided enum values without maintaining a second list."""
    while True:
        value = prompt(f"{text} ({', '.join(choices)})", required=required)
        if not value or value in choices:
            return value
        print("Choose one of the listed values.")


def main() -> int:
    """Collect settings and delegate creation; return a process exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ptafmodel-executable", default="make-ptafmodel")
    args = parser.parse_args()
    executable = shutil.which(args.ptafmodel_executable)
    if executable is None:
        print(
            "Cannot locate make-ptafmodel; install it or use --ptafmodel-executable PATH.",
            file=sys.stderr,
        )
        return 1

    try:
        result = subprocess.run([executable, "schema"], check=True, capture_output=True, text=True)
        # JSON is an external boundary. The native schema supplies these enum lists.
        schema: dict[str, Any] = json.loads(result.stdout)
        properties = schema["properties"]
        artifact = prompt("Artifact path", required=True)
        task = choose("Task", properties["task"]["enum"], required=True)
        output = prompt("New manifest path", required=True)
        target_choices = ", ".join(properties["execution_target_priority"]["items"]["enum"])
        targets = prompt(f"Execution targets ({target_choices}; blank uses native default)")
        command = [executable, "init", artifact, "--task", task, "--output", output]
        if targets:
            command += ["--targets", targets]

        if prompt("Configure advanced settings? [y/N]").lower() == "y":
            backend = choose("Backend (blank leaves it automatic)", properties["backend"]["enum"])
            if backend:
                command += ["--backend", backend]
            for flag, label in (
                ("--device", "Device index"),
                ("--intra-op-threads", "Intra-operation threads"),
                ("--inter-op-threads", "Inter-operation threads"),
                ("--preprocessing", "Preprocessing label"),
                ("--postprocessing", "Postprocessing label"),
            ):
                value = prompt(f"{label} (blank uses native default)")
                if value:
                    command += [flag, value]
            if prompt("Allow automatic fallback? [y/N]").lower() == "y":
                command.append("--allow-fallback")

        print("\nSelected arguments (omitted settings use native defaults):")
        print(json.dumps(command[1:], indent=2))
        if prompt("Create this manifest? [y/N]").lower() != "y":
            print("Cancelled; no manifest written.")
            return 0
        subprocess.run(command, check=True)
        return 0
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        return 0
    except (subprocess.CalledProcessError, OSError, ValueError, KeyError) as error:
        print(f"Manifest creation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
