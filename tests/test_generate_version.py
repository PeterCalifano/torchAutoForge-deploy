"""Exercise explicit version synchronization without modifying the checkout."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("source", ["tag", "version", "none"])
def test_explicit_sync_requires_authoritative_version(tmp_path: Path, source: str) -> None:
    """Synchronize known versions and reject default-only exports before writing."""
    script = Path(__file__).resolve().parents[1] / "generate_version.sh"
    shutil.copy2(script, tmp_path / script.name)
    helper = tmp_path / "ros2/tools/sync_package_metadata.py"
    helper.parent.mkdir(parents=True)
    helper.write_text("from pathlib import Path\nPath('sync-called').write_text('yes')\n")
    version = tmp_path / "VERSION"
    if source == "version":
        version.write_text("Project version core: 1.2.3\n")
    elif source == "tag":
        for args in (
            ["init", "--quiet"],
            [
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "--quiet",
                "--allow-empty",
                "-m",
                "Test fixture",
            ],
            ["tag", "v1.2.3"],
        ):
            subprocess.run(
                ["git", *args],
                cwd=tmp_path,
                check=True,
                capture_output=True,
                env=os.environ | {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"},
            )
    result = subprocess.run(
        ["bash", script.name, "--sync-ros2"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    if source == "none":
        assert result.returncode != 0
        assert not version.exists()
        assert not (tmp_path / "sync-called").exists()
        ordinary = subprocess.run(
            ["bash", script.name, "--no-sync-ros2"],
            cwd=tmp_path,
            capture_output=True,
            check=False,
        )
        assert ordinary.returncode == 0 and version.exists()
    else:
        assert result.returncode == 0, result.stderr
        assert "Project version core: 1.2.3" in version.read_text()
        assert (tmp_path / "sync-called").exists()


@pytest.mark.parametrize("missing", ["overlay", "helper", "python", "broken_python"])
def test_missing_sync_prerequisites_preserve_version(tmp_path: Path, missing: str) -> None:
    """An explicit request must not publish VERSION before prerequisite checks."""
    script = Path(__file__).resolve().parents[1] / "generate_version.sh"
    shutil.copy2(script, tmp_path / script.name)
    version = tmp_path / "VERSION"
    original = "Project version core: 1.2.3\n"
    version.write_text(original)
    if missing != "overlay":
        (tmp_path / "ros2/tools").mkdir(parents=True)
    if missing in {"python", "broken_python"}:
        (tmp_path / "ros2/tools/sync_package_metadata.py").write_text("raise AssertionError\n")
    # Constrain PATH to the required shell utilities, excluding Python when requested.
    executable_dir = tmp_path / "bin"
    executable_dir.mkdir()
    for name in ("bash", "dirname", "awk"):
        (executable_dir / name).symlink_to(shutil.which(name))
    if missing == "broken_python":
        interpreter = executable_dir / "python3"
        interpreter.write_text("#!/bin/sh\nexit 1\n")
        interpreter.chmod(0o755)
    env = os.environ | {"PATH": str(executable_dir)}
    result = subprocess.run(
        [str(executable_dir / "bash"), script.name, "--sync-ros2"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert version.read_text() == original
