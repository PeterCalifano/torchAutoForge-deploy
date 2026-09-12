"""Image selection and annotation for Python demo applications."""

from __future__ import annotations
from functools import cmp_to_key
from pathlib import Path
import re
import numpy as np
from PIL import Image


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
