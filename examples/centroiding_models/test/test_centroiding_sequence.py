"""Observable sequence and publication checks; run with the generated Python wrapper."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import image_sequence as images
import inference_output as reports
from centroiding_metadata import metadata
import run_centroiding as demo


def test_selection_and_collision(tmp_path: Path) -> None:
    """Order filenames naturally and exclude nested outputs from the fixed selection."""
    source = tmp_path / "input"
    source.mkdir()
    for name in ["frame10.PNG", "frame2.png", "frame02.png"]:
        Image.new("L", (8, 6)).save(source / name)
    (source / "nested").mkdir()
    Image.new("L", (8, 6)).save(source / "nested" / "frame1.png")
    assert [p.name for p in images.select_frames(source)] == [
        "frame02.png",
        "frame2.png",
        "frame10.PNG",
    ]
    output = source / "results"
    reports.prepare_output(output, source)
    (output / "existing.txt").write_text("keep")
    with pytest.raises(ValueError):
        reports.prepare_output(output, source)
    with pytest.raises(ValueError):
        reports.prepare_output(source, source)


def test_partial_report(tmp_path: Path) -> None:
    """A failed run keeps completed records and an explicit incomplete status."""
    report = reports.Report(tmp_path, metadata(tmp_path, 2))
    report.append({"index": 0, "source": "frame1.png"})
    with (tmp_path / "frames.jsonl").open("a") as spool:
        spool.write("{uncommitted partial write")
    report.publish(False, {"stage": "decode", "frame_index": 1, "message": "unreadable"})
    report.close(False)
    data = json.loads((tmp_path / "predictions.json").read_text())
    assert data["status"] == "incomplete"
    assert data["frames"] == [{"index": 0, "source": "frame1.png"}]
    assert data["error"]["frame_index"] == 1


def test_overlay_preserves_surrounding_pixels(tmp_path: Path) -> None:
    """Drawing retains dimensions and original 16-bit pixels outside the crosshair."""
    pixels = np.full((64, 64), 12345, dtype=np.uint16)
    source, destination = tmp_path / "source.png", tmp_path / "overlay.png"
    Image.fromarray(pixels).save(source)
    images.save_overlay(source, destination, 32.0, 32.0)
    with Image.open(destination) as image:
        actual = np.asarray(image)
    assert actual.shape == pixels.shape
    assert np.array_equal(actual[:20], pixels[:20])
    assert actual[32, 32] == 65535


def test_single_load_and_partial_decode_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Load once and retain the first frame when the next selected image cannot decode."""
    source, output = tmp_path / "input", tmp_path / "output"
    source.mkdir()
    Image.new("L", (8, 6)).save(source / "frame1.png")
    (source / "frame2.png").write_bytes(b"not an image")
    loads: list[bool] = []

    class FakeModel:
        """Expose only the facade calls needed by the sequence loop."""

        def InferSingleFloatTensor(self, tensor: demo.ptaf.SFloatTensor) -> demo.ptaf.SFloatTensor:
            return demo.ptaf.SFloatTensor("prediction", [1, 2], [0.5, 0.5])

    def load_model(options: demo.DemoOptions) -> FakeModel:
        loads.append(True)
        return FakeModel()

    def prepare(model: FakeModel, path: Path) -> tuple[demo.ptaf.SFloatTensor, tuple[int, int]]:
        with Image.open(path) as image:
            size = image.size
        return demo.ptaf.SFloatTensor("image", [1, 1, 6, 8], [0.0] * 48), size

    monkeypatch.setattr(demo, "load_model", load_model)
    monkeypatch.setattr(demo, "prepare_image", prepare)
    monkeypatch.setattr(demo, "print_result", lambda *args: None)
    original_metadata = metadata
    monkeypatch.setattr(
        demo,
        "metadata",
        lambda path, count, model=None, requested_model=None: original_metadata(path, count),
    )
    with pytest.raises(OSError):
        demo.run_demo(demo.DemoOptions(tmp_path / "model.onnx", source, "manifest", 0, output))
    assert len(loads) == 1
    data = json.loads((output / "predictions.json").read_text())
    assert data["status"] == "incomplete"
    assert len(data["frames"]) == 1
    assert data["frames"][0]["centroid"]["image_pixels"] == {"x": 4.0, "y": 3.0}
