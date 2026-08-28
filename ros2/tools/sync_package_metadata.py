#!/usr/bin/env python3
"""Synchronize the four ROS 2 manifests from root CMake metadata."""

from __future__ import annotations

import argparse
import os
import re
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr


STRICT_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
PACKAGE_ROLES = {
    "ptafdeploy": "ROS 2 colcon shim package.",
    "ptafdeploy_interfaces": "ROS 2 message and service interfaces.",
    "ptafdeploy_ros": "ROS 2 inference bridge package.",
    "ptafdeploy_spinup": "ROS 2 launch and runtime assets.",
}


@dataclass(frozen=True)
class ProjectMetadata:
    """Root project fields that are shared by ROS package manifests."""

    version: str
    description: str
    homepage_url: str
    maintainer_name: str
    maintainer_email: str
    license_name: str


def _cache_value(cache_text: str, key: str) -> str:
    """Return one nonempty CMake cache value."""

    prefix = f"{key}:"
    for line in cache_text.splitlines():
        if line.startswith(prefix):
            _, separator, value = line.partition("=")
            if separator and value:
                return value
    raise ValueError(f"Missing or empty CMake metadata field: {key}")


def _read_project_metadata(
    project_root: Path, requested_version: str
) -> ProjectMetadata:
    """Resolve authoritative project metadata through metadata-only CMake."""

    with tempfile.TemporaryDirectory(
        prefix="ptafdeploy_ros_metadata_"
    ) as build_dir:
        result = subprocess.run(
            [
                "cmake",
                "-S",
                str(project_root),
                "-B",
                build_dir,
                "-DPROJECT_METADATA_ONLY=ON",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Metadata-only CMake configure failed:\n"
                f"{result.stdout}\n{result.stderr}"
            )
        cache_text = (
            Path(build_dir) / "CMakeCache.txt"
        ).read_text(encoding="utf-8")

    cmake_version = _cache_value(cache_text, "CMAKE_PROJECT_VERSION")
    if cmake_version != requested_version:
        raise ValueError(
            f"CMake resolved {cmake_version}, requested {requested_version}"
        )

    return ProjectMetadata(
        version=requested_version,
        description=_cache_value(
            cache_text, "CMAKE_PROJECT_DESCRIPTION"
        ),
        homepage_url=_cache_value(
            cache_text, "CMAKE_PROJECT_HOMEPAGE_URL"
        ),
        maintainer_name=_cache_value(
            cache_text, "PROJECT_MAINTAINER_NAME"
        ),
        maintainer_email=_cache_value(
            cache_text, "PROJECT_MAINTAINER_EMAIL"
        ),
        license_name=_cache_value(cache_text, "PROJECT_LICENSE"),
    )


def _replace_tag(text: str, tag: str, value: str) -> str:
    """Replace the first required XML element while preserving its attributes."""

    pattern = re.compile(
        rf"(<{tag}(?:\s[^>]*)?>).*?(</{tag}>)", re.DOTALL
    )
    updated, count = pattern.subn(
        lambda match: f"{match.group(1)}{escape(value)}{match.group(2)}",
        text,
        count=1,
    )
    if count != 1:
        raise ValueError(f"Expected exactly one <{tag}> element")
    return updated


def _replace_maintainer(text: str, metadata: ProjectMetadata) -> str:
    """Replace the manifest maintainer name and email atomically."""

    pattern = re.compile(
        r"<maintainer(?:\s[^>]*)?>.*?</maintainer>", re.DOTALL
    )
    replacement = (
        f"<maintainer email={quoteattr(metadata.maintainer_email)}>"
        f"{escape(metadata.maintainer_name)}</maintainer>"
    )
    updated, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise ValueError("Expected exactly one <maintainer> element")
    return updated


def _replace_website(text: str, homepage_url: str) -> str:
    """Replace the required website URL in a package manifest."""

    pattern = re.compile(
        r'(<url\s+type=["\']website["\']>).*?(</url>)', re.DOTALL
    )
    updated, count = pattern.subn(
        lambda match: (
            f"{match.group(1)}{escape(homepage_url)}{match.group(2)}"
        ),
        text,
        count=1,
    )
    if count != 1:
        raise ValueError("Expected exactly one website <url> element")
    return updated


def _updated_manifest(
    path: Path, metadata: ProjectMetadata, role_description: str
) -> str:
    """Render one package manifest from root metadata and its package role."""

    text = path.read_text(encoding="utf-8")
    text = _replace_tag(text, "version", metadata.version)
    description = (
        f"{metadata.description.rstrip().removesuffix('.')}: "
        f"{role_description}"
    )
    text = _replace_tag(text, "description", description)
    text = _replace_maintainer(text, metadata)
    text = _replace_tag(text, "license", metadata.license_name)
    return _replace_website(text, metadata.homepage_url)


def _write_atomic(path: Path, content: str) -> None:
    """Replace a manifest without changing its permissions or exposing partial data."""

    mode = stat.S_IMODE(path.stat().st_mode)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
        temporary_path.chmod(mode)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> int:
    """Synchronize or check all four target ROS package manifests."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--ros2-dir", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report stale manifests without modifying them.",
    )
    arguments = parser.parse_args()

    if not STRICT_VERSION.fullmatch(arguments.version):
        raise ValueError(
            f"ROS package version must be X.Y.Z: {arguments.version}"
        )

    project_root = arguments.project_root.resolve()
    ros2_dir = arguments.ros2_dir.resolve()
    metadata = _read_project_metadata(project_root, arguments.version)

    manifest_paths = sorted(ros2_dir.glob("*/package.xml"))
    package_names = {path.parent.name for path in manifest_paths}
    if package_names != set(PACKAGE_ROLES):
        raise ValueError(
            "Expected exactly these ROS packages: "
            + ", ".join(sorted(PACKAGE_ROLES))
            + "; found: "
            + ", ".join(sorted(package_names))
        )

    changed_paths: list[Path] = []
    for path in manifest_paths:
        updated = _updated_manifest(
            path, metadata, PACKAGE_ROLES[path.parent.name]
        )
        if updated == path.read_text(encoding="utf-8"):
            continue
        changed_paths.append(path)
        if not arguments.check:
            _write_atomic(path, updated)

    if arguments.check and changed_paths:
        for path in changed_paths:
            print(f"Stale ROS manifest: {path}")
        return 1

    action = "Would synchronize" if arguments.check else "Synchronized"
    print(f"{action} {len(manifest_paths)} ROS 2 package manifests.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
