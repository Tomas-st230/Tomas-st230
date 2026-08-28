"""Export correctness: the right frame, at the right place, under the right name."""

from __future__ import annotations

import os

import pytest

from framepicker import export
from framepicker.decode import Normalisation

from conftest import CLIP_FPS, MARKER_STEP, read_marker_x, requires_ffmpeg


@requires_ffmpeg
@pytest.mark.parametrize("requested_t", [0.5, 2.0, 3.7, 5.0])
def test_export_timestamp_accuracy(counter_clip, counter_markers, requested_t, tmp_path):
    """The exported still is the frame that was scored, within one frame.

    The fixture carries a burnt-in counter: a white block that moves
    ``MARKER_STEP`` pixels per frame. Reading it back out of the JPEG gives
    the frame number with no OCR and no fixed magic offset - the mapping is
    measured from the fixture itself in ``conftest``.
    """
    import cv2

    out_path = str(tmp_path / f"frame_{requested_t}.jpg")
    result = export.export_frame(counter_clip, requested_t, out_path)
    assert result.ok, result.detail
    assert os.path.isfile(out_path)

    exported = cv2.imread(out_path, cv2.IMREAD_COLOR)
    assert exported is not None
    got = read_marker_x(exported)
    assert got is not None, "no burnt-in marker found in the exported frame"

    expected_index = min(int(round(requested_t * CLIP_FPS)), len(counter_markers) - 1)
    expected = counter_markers[expected_index]
    assert expected >= 0, "the fixture frame itself carried no marker"

    drift_frames = abs(got - expected) / MARKER_STEP
    assert drift_frames <= 1.0, (requested_t, got, expected, drift_frames)


@requires_ffmpeg
def test_export_applies_the_analysis_transform(counter_clip, tmp_path):
    """What you look at is what was scored."""
    import cv2

    plain = str(tmp_path / "plain.jpg")
    converted = str(tmp_path / "converted.jpg")
    normalisation = Normalisation(lo=40.0, hi=180.0, saturation_gain=1.8)

    assert export.export_frame(counter_clip, 2.0, plain).ok
    assert export.export_frame(counter_clip, 2.0, converted, normalisation=normalisation).ok

    a = cv2.imread(plain)
    b = cv2.imread(converted)
    assert a is not None and b is not None
    assert a.shape == b.shape
    assert float(abs(a.astype("float32") - b.astype("float32")).mean()) > 1.0


@requires_ffmpeg
def test_export_of_a_missing_file_fails_without_raising(tmp_path):
    result = export.export_frame(str(tmp_path / "nope.mp4"), 1.0, str(tmp_path / "out.jpg"))
    assert result.ok is False
    assert result.detail


def test_output_names_sort_by_rank():
    names = [export.output_name("DJI_0042.MP4", rank, 120.0 - rank, 0.5) for rank in range(1, 7)]
    assert names == sorted(names)
    assert names[0].startswith("DJI_0042_01_")
    assert names[0].endswith(".jpg")


def test_output_name_sanitises_the_clip_name():
    name = export.output_name("a b/c:d*e.mov", 1, 1.0, 0.5)
    assert "/" not in name and ":" not in name and " " not in name


@requires_ffmpeg
def test_export_height_scales_down_and_never_up(counter_clip, tmp_path):
    """4K stays 4K by default; --export-height only ever scales down."""
    import cv2

    from framepicker.probe import probe

    source = probe(counter_clip)

    native = str(tmp_path / "native.jpg")
    smaller = str(tmp_path / "small.jpg")
    bigger = str(tmp_path / "big.jpg")
    assert export.export_frame(counter_clip, 1.0, native).ok
    assert export.export_frame(counter_clip, 1.0, smaller, height=120).ok
    assert export.export_frame(counter_clip, 1.0, bigger, height=source.height * 4).ok

    assert cv2.imread(native).shape[0] == source.height
    assert cv2.imread(smaller).shape[0] == 120
    assert cv2.imread(bigger).shape[0] == source.height, "an upscale must never happen"


def test_export_filter_is_empty_when_nothing_is_asked_for():
    assert export.build_export_filter(None, 0) is None
    assert "scale" in export.build_export_filter(None, 1080)
    assert "lut3d" in export.build_export_filter("look.cube", 0)
    chain = export.build_export_filter("look.cube", 1080)
    assert chain.index("lut3d") < chain.index("scale"), "grade first, then resize"
