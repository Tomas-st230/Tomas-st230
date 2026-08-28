"""Greedy selection with diversity constraints.

Ranking alone gives six near-identical frames from the same second. This
module walks the ranking from the top and keeps a candidate only if it is far
enough away in time *and* visually different from everything already kept.

Rule 9.1: when the constraints cannot yield the requested count, the
shortfall is returned with a stated reason. Silently returning three frames
when six were asked for is the failure mode this project keeps paying for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from . import strings_lt as S
from .features import hamming, histogram_distance

#: Minimum dHash Hamming distance (64-bit hash) for two frames to count as
#: visually different.
DHASH_MIN_DISTANCE = 10
#: Minimum total-variation distance between colour histograms, 0..1.
HISTOGRAM_MIN_DISTANCE = 0.25
#: The time gap is at least this fraction of the clip length.
GAP_CLIP_FRACTION = 0.03


@dataclass
class Candidate:
    index: int
    t: float
    score: float
    dhash: int
    histogram: object
    features: dict = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


@dataclass
class SelectionResult:
    selected: list[Candidate]
    requested: int
    effective_gap: float
    rejected_time_gap: int = 0
    rejected_duplicate: int = 0
    reasons: list[str] = field(default_factory=list)

    @property
    def shortfall(self) -> int:
        return max(0, self.requested - len(self.selected))

    def as_dict(self) -> dict:
        return {
            "requested": self.requested,
            "delivered": len(self.selected),
            "shortfall": self.shortfall,
            "effective_min_gap_s": self.effective_gap,
            "rejected_time_gap": self.rejected_time_gap,
            "rejected_duplicate": self.rejected_duplicate,
            "shortfall_reasons": list(self.reasons),
        }


def effective_gap(min_gap: float, clip_duration: float | None) -> float:
    """``--min-gap`` seconds, or 3 % of the clip, whichever is larger."""
    by_length = (clip_duration or 0.0) * GAP_CLIP_FRACTION
    return float(max(min_gap, by_length))


def is_visually_different(
    a: Candidate,
    b: Candidate,
    dhash_min: int = DHASH_MIN_DISTANCE,
    hist_min: float = HISTOGRAM_MIN_DISTANCE,
) -> bool:
    """True when *a* and *b* are far enough apart to both be worth keeping."""
    if hamming(a.dhash, b.dhash) >= dhash_min:
        return True
    if a.histogram is not None and b.histogram is not None:
        if histogram_distance(a.histogram, b.histogram) >= hist_min:
            return True
    return False


def select(
    candidates: Sequence[Candidate],
    per_clip: int,
    *,
    min_gap: float = 2.0,
    clip_duration: float | None = None,
    dhash_min: int = DHASH_MIN_DISTANCE,
    hist_min: float = HISTOGRAM_MIN_DISTANCE,
) -> SelectionResult:
    """Pick up to *per_clip* frames, and explain any shortfall."""
    gap = effective_gap(min_gap, clip_duration)
    result = SelectionResult(selected=[], requested=per_clip, effective_gap=gap)

    ordered = sorted(candidates, key=lambda c: (-c.score, c.t))
    for candidate in ordered:
        if len(result.selected) >= per_clip:
            break
        too_close = any(abs(candidate.t - kept.t) < gap for kept in result.selected)
        if too_close:
            result.rejected_time_gap += 1
            continue
        duplicate = any(
            not is_visually_different(candidate, kept, dhash_min, hist_min)
            for kept in result.selected
        )
        if duplicate:
            result.rejected_duplicate += 1
            continue
        result.selected.append(candidate)

    result.selected.sort(key=lambda c: (-c.score, c.t))
    if result.shortfall:
        result.reasons = _shortfall_reasons(result, candidates, clip_duration, gap, per_clip)
    return result


def _shortfall_reasons(
    result: SelectionResult,
    candidates: Sequence[Candidate],
    clip_duration: float | None,
    gap: float,
    per_clip: int,
) -> list[str]:
    reasons: list[str] = []
    clip_limited = False
    if clip_duration and gap > 0:
        possible = int(clip_duration // gap) + 1
        if possible < per_clip:
            reasons.append(S.shortfall_clip_too_short(clip_duration, gap, possible))
            clip_limited = True
    if len(candidates) < per_clip:
        reasons.append(S.shortfall_not_enough_candidates(len(candidates)))
    if result.rejected_duplicate:
        reasons.append(S.shortfall_near_duplicates(result.rejected_duplicate))
    if result.rejected_time_gap and not clip_limited:
        reasons.append(S.shortfall_time_gap(result.rejected_time_gap))
    if not reasons:
        reasons.append(S.shortfall_not_enough_candidates(len(candidates)))
    return reasons
