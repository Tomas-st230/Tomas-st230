"""Scoring is tested on hand-built feature dicts.

That is the point: if the test fed the pipeline's own output back in, it
would mirror whatever the pipeline does and could never catch an inversion.
"""

from __future__ import annotations

from framepicker import scoring
from framepicker import strings_lt as S

PERSON = {
    "sharpness_rank": 0.45,          # mediocre sharpness
    "colorfulness_rank": 0.35,
    "dynamic_range_rank": 0.35,
    "exposure_clip_low": 0.0,
    "exposure_clip_high": 0.0,
    "face_count": 1,
    "face_max_rel": 0.18,            # a person, not a portrait
    "subject_rel": 0.12,
    "thirds_distance": 0.30,
}

EMPTY_AERIAL = {
    "sharpness_rank": 0.99,          # razor sharp
    "colorfulness_rank": 0.95,
    "dynamic_range_rank": 0.95,
    "exposure_clip_low": 0.0,
    "exposure_clip_high": 0.0,
    "face_count": 0,
    "face_max_rel": 0.0,
    "subject_rel": 0.05,
    "thirds_distance": 0.30,         # same composition, so it cannot be the reason
}


def test_score_prefers_person_over_empty_aerial():
    """TRAP-08: texture statistics must not out-rank a person in frame."""
    person_score, person_reasons = scoring.score_frame_explained(PERSON)
    aerial_score, _ = scoring.score_frame_explained(EMPTY_AERIAL)
    assert person_score > aerial_score, (person_score, aerial_score)
    assert any("18" in reason for reason in person_reasons)


def test_landscape_footage_is_not_punished_for_having_no_people():
    """R-VIS-006: B-roll without people still scores on its own merits."""
    score, _ = scoring.score_frame_explained(EMPTY_AERIAL)
    assert score > 0.5


def test_face_component_saturates_around_half_frame():
    small = scoring.face_component(0.05)
    medium = scoring.face_component(0.18)
    half = scoring.face_component(0.50)
    full = scoring.face_component(0.90)
    assert small < medium < half
    assert half > 0.95
    assert full - half < 0.02          # saturated: nothing left to gain


def test_face_below_minimum_size_is_not_a_subject():
    assert scoring.face_component(scoring.FACE_MIN_REL / 2) == 0.0


def test_none_face_is_not_zero_face():
    assert scoring.face_component(None) is None
    assert scoring.face_component(0.0) == 0.0


def test_missing_composition_drops_the_term_instead_of_scoring_zero():
    with_composition = dict(PERSON)
    without = dict(PERSON, thirds_distance=None, subject_rel=None)
    bad_composition = dict(PERSON, thirds_distance=1.0)
    dropped, reasons = scoring.score_frame_explained(without)
    scored, _ = scoring.score_frame_explained(with_composition)
    worst, _ = scoring.score_frame_explained(bad_composition)
    assert dropped > worst, "a missing term must not score like the worst possible value"
    assert S.reason_composition_unknown() in reasons
    assert 0.0 < dropped <= 1.0 and 0.0 < scored <= 1.0


def test_clipping_is_penalised():
    clean, _ = scoring.score_frame_explained(EMPTY_AERIAL)
    blown, reasons = scoring.score_frame_explained(dict(EMPTY_AERIAL, exposure_clip_high=0.12))
    assert blown < clean
    assert any("12" in reason for reason in reasons)


def test_confidence_flags_a_flat_ranking():
    flat = [0.700 + i * 0.0005 for i in range(20)]
    spread_out = [0.9 - i * 0.02 for i in range(20)]
    flat_verdict = scoring.confidence(flat)
    real_verdict = scoring.confidence(spread_out)
    assert flat_verdict["informative"] is False
    assert real_verdict["informative"] is True
    assert flat_verdict["message"] != real_verdict["message"]


def test_reasons_are_never_empty():
    _, reasons = scoring.score_frame_explained(PERSON)
    assert len(reasons) >= 4
