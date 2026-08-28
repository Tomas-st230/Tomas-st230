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
