"""A look must adapt to the frame, not be pasted onto it."""

from __future__ import annotations

import numpy as np
import pytest

from framepicker import grading


def _frame(mean: int, spread: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(mean, spread, (64, 64, 3)), 0, 255).astype(np.uint8)


def test_strength_zero_returns_the_frame_untouched():
    frame = _frame(120, 30)
    for name in ("nature", "city"):
        out, info = grading.apply_look(frame, grading.get(name), strength=0.0)
        assert np.array_equal(out, frame)
        assert info["strength"] == 0.0


def test_none_is_not_a_look():
    assert grading.get(None) is None
    assert grading.get("none") is None
    assert grading.get("nature") is grading.LOOK_NATURE
    assert "none" in grading.available()


def test_a_flat_frame_is_lifted_and_a_punchy_one_is_left_alone():
    """The whole point: the correction is measured per frame."""
    flat = _frame(120, 10)
    punchy = _frame(128, 70, seed=1)
    look = grading.get("nature")

    flat_out, flat_info = grading.apply_look(flat, look, 0.6)
    punchy_out, punchy_info = grading.apply_look(punchy, look, 0.6)

    flat_change = flat_info["after"]["span"] - flat_info["before"]["span"]
    punchy_change = abs(punchy_info["after"]["span"] - punchy_info["before"]["span"])
    assert flat_change > 3 * punchy_change, (flat_change, punchy_change)
    assert punchy_change < 0.05, "one that already has contrast must be left alone"
    # The flat frame asks for far more than it is allowed to get, and the cap
    # is what stops a foggy frame turning neon. The punchy one asks for almost
    # nothing, and gets almost nothing.
    assert flat_info["applied"]["contrast_scale"] == pytest.approx(grading.CONTRAST_SCALE_MAX)
    assert punchy_info["applied"]["contrast_scale"] == pytest.approx(1.0, abs=0.1)
    assert not np.array_equal(flat_out, flat)
    assert punchy_out.shape == punchy.shape


def test_saturation_is_moved_toward_the_target_not_multiplied_blindly():
    look = grading.get("nature")
    dull = _frame(120, 20)
    for strength in (0.3, 0.6, 1.0):
        _out, info = grading.apply_look(dull, look, strength)
        gain = info["applied"]["saturation_gain"]
        assert grading.SATURATION_GAIN_MIN <= gain <= grading.SATURATION_GAIN_MAX


def test_a_nearly_grey_frame_is_never_dragged_into_neon():
    """The 2.5x multiplier that wrecked four sunsets must not come back."""
    grey = np.full((64, 64, 3), 110, dtype=np.uint8)
    grey[:, :2, 0] = 115                       # a whisper of colour
    for name in ("nature", "city"):
        _out, info = grading.apply_look(grey, grading.get(name), 1.0)
        assert info["after"]["saturation"] < 0.20, info["after"]


def test_nature_is_warmer_than_city_on_the_same_frame():
    frame = _frame(120, 25)
    warm, _ = grading.apply_look(frame, grading.get("nature"), 1.0)
    cool, _ = grading.apply_look(frame, grading.get("city"), 1.0)

    def tilt(image):
        return float(image[..., 0].mean()) - float(image[..., 2].mean())

    assert tilt(warm) > tilt(cool)


def test_highlights_keep_their_gradation_instead_of_becoming_one_white_shape():
    """A rolloff is only worth having if bright levels stay distinguishable."""
    frame = np.full((32, 96, 3), 120, dtype=np.uint8)
    frame[:, 32:64] = 244
    frame[:, 64:] = 252
    out, _ = grading.apply_look(frame, grading.get("nature"), 1.0)
    bright_a = float(out[:, 32:64].mean())
    bright_b = float(out[:, 64:].mean())
    assert bright_b > bright_a, "the two bright levels were mashed together"
    assert bright_a > float(out[:, :32].mean())


def test_a_nearly_flat_frame_is_not_stretched_into_neon():
    """Fog, a grey sky, a night shot: the target span would ask for 100x."""
    fog = np.full((64, 64, 3), 118, dtype=np.uint8)
    fog[:, :8, 0] = 123
    out, info = grading.apply_look(fog, grading.get("nature"), 1.0)
    assert info["applied"]["contrast_scale"] <= grading.CONTRAST_SCALE_MAX
    assert info["after"]["saturation"] < 0.10, info["after"]
    assert int(out.max()) < 255


def test_measure_reports_what_it_reads():
    grey = np.full((16, 16, 3), 100, dtype=np.uint8)
    reading = grading.measure(grey)
    assert reading["saturation"] == pytest.approx(0.0, abs=1e-6)
    assert reading["span"] == pytest.approx(0.0, abs=1e-6)


def test_every_preset_is_reportable():
    for name in grading.available():
        look = grading.get(name)
        if look is None:
            continue
        assert set(look.as_dict()) >= {"name", "target_saturation", "target_span", "warmth"}


# --------------------------------------------------------------------------
# --look auto
# --------------------------------------------------------------------------


def _flat_frame(height: int = 240, width: int = 320) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_auto_picks_nature_for_a_green_and_blue_frame():
    from framepicker import features

    frame = _flat_frame()
    frame[:120, :] = (40, 150, 60)     # vegetation
    frame[120:, :] = (60, 120, 200)    # sky
    result = grading.classify_frames([features.scene_signature(frame)])
    assert result["choice"] == grading.NATURE
    assert result["decided"] is True
    assert result["nature_score"] > result["city_score"]


def test_auto_picks_city_for_grey_with_vertical_lines():
    from framepicker import features

    frame = _flat_frame()
    frame[:, :] = (128, 130, 132)      # concrete
    for x in range(0, 320, 24):        # window columns and building edges
        frame[:, x:x + 3] = (210, 212, 214)
    result = grading.classify_frames([features.scene_signature(frame)])
    assert result["choice"] == grading.CITY
    assert result["decided"] is True


def test_auto_refuses_to_decide_when_the_evidence_is_close():
    """An undecided answer is a real answer: no look is applied."""
    from framepicker import features

    frame = _flat_frame()
    frame[:120, :] = (128, 130, 132)   # grey: half the frame
    frame[120:168, :] = (40, 150, 60)  # colour: a fifth of it
    result = grading.classify_frames([features.scene_signature(frame)])
    assert result["margin"] < grading.AUTO_MARGIN
    assert result["choice"] == grading.NONE
    assert result["decided"] is False
    assert grading.get(result["choice"]) is None


def test_auto_with_nothing_to_measure_is_not_a_crash():
    result = grading.classify_frames([])
    assert result["choice"] == grading.NONE
    assert result["frames_measured"] == 0
    assert result["decided"] is False


def test_auto_is_offered_but_is_not_itself_a_look():
    assert grading.AUTO in grading.available()
    assert grading.get(grading.AUTO) is None
