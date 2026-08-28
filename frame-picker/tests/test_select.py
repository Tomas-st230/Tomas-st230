from __future__ import annotations

import numpy as np

from framepicker.select import Candidate, effective_gap, select


def _candidate(index: int, t: float, score: float, dhash: int, hist=None) -> Candidate:
    if hist is None:
        hist = np.zeros(8 ** 3)
        hist[index % (8 ** 3)] = 1.0
    return Candidate(index=index, t=t, score=score, dhash=dhash, histogram=hist)


def test_diversity_rejects_near_duplicates():
    """Two identical frames must never both be selected."""
    shared_hist = np.zeros(8 ** 3)
    shared_hist[5] = 1.0
    twin_a = Candidate(index=0, t=0.0, score=0.90, dhash=0xDEADBEEFCAFEF00D, histogram=shared_hist)
    twin_b = Candidate(index=1, t=10.0, score=0.89, dhash=0xDEADBEEFCAFEF00D, histogram=shared_hist)
    different = _candidate(2, 20.0, 0.50, 0x0123456789ABCDEF)

    result = select([twin_a, twin_b, different], per_clip=3, min_gap=2.0, clip_duration=30.0)

    kept = {c.index for c in result.selected}
    assert not {0, 1} <= kept, "identical frames were both selected"
    assert 2 in kept
    assert result.rejected_duplicate == 1


def test_time_gap_is_enforced():
    a = _candidate(0, 10.0, 0.9, 0x1111111111111111)
    b = _candidate(1, 10.5, 0.8, 0x8888888888888888)
    result = select([a, b], per_clip=2, min_gap=2.0, clip_duration=60.0)
    assert [c.index for c in result.selected] == [0]
    assert result.rejected_time_gap == 1


def test_effective_gap_uses_three_percent_of_a_long_clip():
    assert effective_gap(2.0, 30.0) == 2.0
    assert effective_gap(2.0, 600.0) == 18.0


def test_shortfall_reports_a_reason():
    a = _candidate(0, 1.0, 0.9, 0x1111111111111111)
    b = _candidate(1, 3.5, 0.8, 0x2222222222222222)
    result = select([a, b], per_clip=6, min_gap=2.0, clip_duration=5.0)
    assert result.shortfall > 0
    assert result.reasons, "a shortfall with no stated reason is the bug this test exists for"
    assert result.as_dict()["shortfall_reasons"]


def test_selection_is_returned_in_rank_order():
    items = [_candidate(i, i * 5.0, 0.5 + i * 0.05, 1 << (i * 7)) for i in range(4)]
    result = select(items, per_clip=4, min_gap=2.0, clip_duration=60.0)
    scores = [c.score for c in result.selected]
    assert scores == sorted(scores, reverse=True)


# --------------------------------------------------------------------------
# The threshold decides how many, not a cap
# --------------------------------------------------------------------------


def _spread(count: int, score: float = 0.8, step: float = 10.0) -> list[Candidate]:
    """Candidates far apart in time and visually unlike each other."""
    return [
        _candidate(i, i * step, score, dhash=(0x0F0F0F0F0F0F0F0F * (i + 1)) & ((1 << 64) - 1))
        for i in range(count)
    ]


def test_by_default_nothing_caps_a_strong_clip():
    """A clip full of good frames returns all of them, not the first twelve."""
    from framepicker.select import DEFAULT_MAX_PER_CLIP, MODE_THRESHOLD

    assert DEFAULT_MAX_PER_CLIP == 0, "a cap must not be the default"
    candidates = _spread(30)
    result = select(candidates, per_clip=6, mode=MODE_THRESHOLD, min_score=0.6,
                    min_gap=2.0, clip_duration=60.0)
    assert len(result.selected) == 30
    assert result.capped is False


def test_one_good_frame_out_of_many_is_a_complete_answer():
    """Most of the clip below the bar, one frame above it: one frame is right."""
    from framepicker.select import MODE_THRESHOLD

    candidates = _spread(200, score=0.42)
    candidates[137] = _candidate(137, 1370.0, 0.71, dhash=0xAAAA5555AAAA5555)
    result = select(candidates, per_clip=6, mode=MODE_THRESHOLD, min_score=0.6,
                    min_gap=2.0, clip_duration=2000.0)

    assert [c.index for c in result.selected] == [137]
    assert result.passed_threshold == 1
    assert result.rejected_below_threshold == 199
    assert result.shortfall == 0, "threshold mode has no target to fall short of"


def test_nothing_good_enough_returns_nothing_and_names_the_best_score():
    from framepicker import strings_lt as S
    from framepicker.select import MODE_THRESHOLD

    result = select(_spread(12, score=0.41), per_clip=6, mode=MODE_THRESHOLD,
                    min_score=0.6, min_gap=2.0, clip_duration=120.0)
    assert result.selected == []
    assert result.best_score == 0.41
    assert S.threshold_none_passed(0.41, 0.6) in result.notes


def test_a_cap_is_available_and_says_it_was_the_cap():
    from framepicker import strings_lt as S
    from framepicker.select import MODE_THRESHOLD

    result = select(_spread(30), per_clip=6, mode=MODE_THRESHOLD, min_score=0.6,
                    max_per_clip=4, min_gap=2.0, clip_duration=60.0)
    assert len(result.selected) == 4
    assert result.capped is True
    assert S.threshold_capped(4) in result.notes


def test_the_gap_and_the_duplicate_test_still_apply_without_a_cap():
    """"No bound" is not "everything": the same picture twice is still rejected."""
    from framepicker.select import MODE_THRESHOLD

    shared = np.zeros(8 ** 3)
    shared[9] = 1.0
    twins = [
        Candidate(index=i, t=i * 30.0, score=0.9, dhash=0x1234123412341234, histogram=shared)
        for i in range(5)
    ]
    result = select(twins, per_clip=6, mode=MODE_THRESHOLD, min_score=0.6,
                    min_gap=2.0, clip_duration=300.0)
    assert len(result.selected) == 1
    assert result.rejected_duplicate == 4
    assert result.capped is False
