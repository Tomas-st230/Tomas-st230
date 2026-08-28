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

MODE_THRESHOLD = "threshold"
MODE_COUNT = "count"

#: Score a frame has to reach in threshold mode. Like WEIGHTS, this is a
#: starting value chosen by argument, not by measurement: it means nothing
#: until the weights are calibrated on real footage, and everything that
#: prints it says so.
DEFAULT_MIN_SCORE = 0.65

#: Safety bound in threshold mode, so a long clip cannot quietly export a
#: hundred stills. 0 means no bound.
DEFAULT_MAX_PER_CLIP = 12


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
    mode: str = MODE_COUNT
    min_score: float | None = None
    passed_threshold: int = 0
    rejected_below_threshold: int = 0
    rejected_time_gap: int = 0
    rejected_duplicate: int = 0
    capped: bool = False
    best_score: float | None = None
    reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def shortfall(self) -> int:
        """Only meaningful in count mode: a fixed target that was not met."""
        if self.mode != MODE_COUNT:
            return 0
        return max(0, self.requested - len(self.selected))

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "min_score": self.min_score,
            "requested": self.requested if self.mode == MODE_COUNT else None,
            "delivered": len(self.selected),
            "shortfall": self.shortfall,
            "passed_threshold": self.passed_threshold,
            "rejected_below_threshold": self.rejected_below_threshold,
            "effective_min_gap_s": self.effective_gap,
            "rejected_time_gap": self.rejected_time_gap,
            "rejected_duplicate": self.rejected_duplicate,
            "capped": self.capped,
            "best_score": self.best_score,
            "shortfall_reasons": list(self.reasons),
            "notes": list(self.notes),
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
    per_clip: int = 6,
    *,
    mode: str = MODE_COUNT,
    min_score: float = DEFAULT_MIN_SCORE,
    max_per_clip: int = DEFAULT_MAX_PER_CLIP,
    min_gap: float = 2.0,
    clip_duration: float | None = None,
    dhash_min: int = DHASH_MIN_DISTANCE,
    hist_min: float = HISTOGRAM_MIN_DISTANCE,
) -> SelectionResult:
    """Pick frames, and explain both what was taken and what was not.

    ``MODE_COUNT`` aims at *per_clip* frames and reports any shortfall.
    ``MODE_THRESHOLD`` takes every frame scoring at least *min_score* - two
    from a weak clip, twelve from a strong one - bounded by *max_per_clip*
    so a long file cannot silently export a hundred stills.
    """
    gap = effective_gap(min_gap, clip_duration)
    ordered = sorted(candidates, key=lambda c: (-c.score, c.t))
    best = ordered[0].score if ordered else None

    if mode == MODE_THRESHOLD:
        eligible = [c for c in ordered if c.score >= min_score]
        limit = max_per_clip if max_per_clip and max_per_clip > 0 else len(eligible)
    else:
        eligible = ordered
        limit = per_clip

    result = SelectionResult(
        selected=[],
        requested=per_clip,
        effective_gap=gap,
        mode=mode,
        min_score=min_score if mode == MODE_THRESHOLD else None,
        passed_threshold=len(eligible) if mode == MODE_THRESHOLD else 0,
        rejected_below_threshold=(len(ordered) - len(eligible)) if mode == MODE_THRESHOLD else 0,
        best_score=best,
    )

    for candidate in eligible:
        if len(result.selected) >= limit:
            result.capped = mode == MODE_THRESHOLD
            break
        if any(abs(candidate.t - kept.t) < gap for kept in result.selected):
            result.rejected_time_gap += 1
            continue
        if any(not is_visually_different(candidate, kept, dhash_min, hist_min) for kept in result.selected):
            result.rejected_duplicate += 1
            continue
        result.selected.append(candidate)

    result.selected.sort(key=lambda c: (-c.score, c.t))

    if mode == MODE_THRESHOLD:
        result.notes.append(S.selection_mode_threshold(min_score))
        if not eligible:
            result.notes.append(S.threshold_none_passed(best if best is not None else 0.0, min_score))
        else:
            result.notes.append(S.threshold_passed(len(eligible), min_score))
        if result.capped:
            result.notes.append(S.threshold_capped(len(result.selected)))
        if eligible and len(result.selected) < len(eligible) and not result.capped:
            # Fewer frames than passed the threshold: say which constraint ate them.
            if result.rejected_duplicate:
                result.notes.append(S.shortfall_near_duplicates(result.rejected_duplicate))
            if result.rejected_time_gap:
                result.notes.append(S.shortfall_time_gap(result.rejected_time_gap))
    else:
        result.notes.append(S.selection_mode_count(per_clip))
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
