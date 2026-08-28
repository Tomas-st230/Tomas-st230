"""The pure measurements. No pipeline, no files."""

from __future__ import annotations

import numpy as np
import pytest

from framepicker import features


def _frame(value: int, size: int = 64) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def _noise(seed: int = 0, size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)


def test_sharpness_ranks_detail_above_flat():
    flat = _frame(128)
    detailed = _noise(1)
    assert features.sharpness(detailed) > features.sharpness(flat)
    assert features.sharpness(flat) == pytest.approx(0.0, abs=1e-6)


def test_sharpness_drops_when_a_frame_is_blurred():
    import cv2

    detailed = _noise(2)
    blurred = cv2.GaussianBlur(detailed, (0, 0), 4.0)
    assert features.sharpness(blurred) < features.sharpness(detailed)


def test_exposure_clipping_counts_pixels_not_opinions():
    black = _frame(0)
    white = _frame(255)
    assert features.exposure_clip_low(black) == 1.0
    assert features.exposure_clip_high(black) == 0.0
    assert features.exposure_clip_high(white) == 1.0
    assert features.exposure_clip_low(white) == 0.0

    half = np.concatenate([_frame(0, 64)[:32], _frame(255, 64)[:32]], axis=0)
    assert features.exposure_clip_low(half) == pytest.approx(0.5)
    assert features.exposure_clip_high(half) == pytest.approx(0.5)


def test_dynamic_range_is_zero_on_a_flat_frame():
    assert features.dynamic_range(_frame(90)) == pytest.approx(0.0, abs=1e-6)
    ramp = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
    frame = np.dstack([ramp, ramp, ramp])
    assert features.dynamic_range(frame) > 0.9


def test_colorfulness_is_zero_on_grey_and_high_on_colour():
    grey = _frame(120)
    colourful = np.zeros((64, 64, 3), dtype=np.uint8)
    colourful[:, :32, 0] = 255
    colourful[:, 32:, 2] = 255
    assert features.colorfulness(grey) == pytest.approx(0.0, abs=1e-6)
    assert features.colorfulness(colourful) > 50


def test_saturation_mean_is_zero_on_grey():
    assert features.saturation_mean(_frame(120)) == pytest.approx(0.0, abs=1e-6)


def test_thirds_distance_is_zero_on_a_thirds_point():
    assert features.thirds_distance(1 / 3, 1 / 3) == pytest.approx(0.0, abs=1e-9)
    assert features.thirds_distance(2 / 3, 2 / 3) == pytest.approx(0.0, abs=1e-9)
    centre = features.thirds_distance(0.5, 0.5)
    corner = features.thirds_distance(0.0, 0.0)
    assert 0.0 < centre < corner <= 1.0


def test_percentile_ranks_span_zero_to_one_and_tie_together():
    values = [5.0, 1.0, 3.0, 9.0]
    ranks = features.percentile_ranks(values)
    assert min(ranks) == 0.0 and max(ranks) == 1.0
    by_value = [rank for _, rank in sorted(zip(values, ranks))]
    assert by_value == sorted(by_value), "a larger value must never get a smaller rank"
    tied = features.percentile_ranks([2.0, 2.0, 2.0])
    assert tied == [0.5, 0.5, 0.5]
    assert features.percentile_ranks([]) == []
    assert features.percentile_ranks([7.0]) == [1.0]


def test_dhash_is_stable_and_separates_different_frames():
    frame = _noise(3)
    assert features.dhash(frame) == features.dhash(frame.copy())
    assert features.hamming(features.dhash(frame), features.dhash(_noise(4))) > 5
    assert features.hamming(features.dhash(frame), features.dhash(frame)) == 0


def test_histogram_distance_is_zero_for_identical_frames():
    frame = _noise(5)
    a = features.color_histogram(frame)
    b = features.color_histogram(frame.copy())
    assert features.histogram_distance(a, b) == pytest.approx(0.0, abs=1e-9)
    assert features.histogram_distance(a, features.color_histogram(_frame(0))) > 0.5


def test_extract_leaves_unavailable_detectors_as_none():
    result = features.extract(_noise(6))
    for key in ("face_count", "face_max_rel", "subject_rel", "thirds_distance"):
        assert result[key] is None, key
    for key in ("sharpness", "dynamic_range", "colorfulness", "saturation_mean"):
        assert isinstance(result[key], float)


def test_missing_face_model_gives_none_not_a_detector():
    detector, status = features.load_face_detector("/no/such/model.onnx")
    assert detector is None
    assert status.available is False
    assert status.detail == "model-missing"


def test_yunet_filenames_are_the_ones_published_by_opencv_zoo():
    """Checked against the repository, not guessed (task section 12)."""
    assert "face_detection_yunet_2023mar.onnx" in features.YUNET_FILENAMES
    assert all(name.endswith(".onnx") for name in features.YUNET_FILENAMES)


# --------------------------------------------------------------------------
# Motion, horizon tilt, subject separation
# --------------------------------------------------------------------------


def _horizon(degrees: float, width: int = 640, height: int = 360) -> np.ndarray:
    """A bright sky over dark land, tilted by *degrees*."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = (60, 130, 210)
    for x in range(width):
        y = int(height / 2 + np.tan(np.radians(degrees)) * (x - width / 2))
        image[max(0, min(height - 1, y)):, x] = (25, 60, 30)
    return image


def test_motion_is_none_for_the_first_frame_not_zero():
    frame = _noise(10)
    assert features.motion(None, frame) is None
    assert features.motion(frame, frame) == 0.0


def test_motion_grows_with_the_change_between_samples():
    still = _horizon(0.0)
    nudged = _horizon(1.5)
    moved = _horizon(6.0)
    assert 0.0 < features.motion(still, nudged) < features.motion(still, moved)


def test_horizon_tilt_measures_the_angle():
    assert features.horizon_tilt(_horizon(0.0)) == pytest.approx(0.0, abs=0.02)
    two = features.horizon_tilt(_horizon(2.0))
    four = features.horizon_tilt(_horizon(4.0))
    assert two is not None and four is not None
    assert two < four
    assert features.horizon_tilt(_horizon(2.0)) == pytest.approx(2.0 / features.TILT_FULL_DEGREES, abs=0.06)
    assert features.horizon_tilt(_horizon(20.0)) == 1.0


def test_a_frame_with_no_horizon_withholds_the_measurement():
    """None means "cannot tell", and must never be scored as "level"."""
    assert features.horizon_tilt(np.full((360, 640, 3), 120, np.uint8)) is None
    assert features.horizon_tilt(_noise(11, size=256)) is None
