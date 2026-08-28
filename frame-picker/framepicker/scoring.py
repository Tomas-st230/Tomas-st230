"""Turning one feature dict into one score, plus the reasons for it.

Pure functions. No I/O, no OpenCV, no pipeline. That is deliberate: the
acceptance tests build feature dicts by hand, so a test cannot accidentally
mirror whatever the pipeline happens to produce.

Design, stated so it can be argued with
--------------------------------------
1. **Technical quality is a gate, not a ladder.** Past "sharp enough", more
   sharpness buys almost nothing; below it, the frame is punished. A razor
   sharp frame of nothing is still a frame of nothing (TRAP-08).
2. **Percentile ranks are relative.** ``colorfulness_rank == 1.0`` means
   "the most colourful frame in *this* clip", not "beautiful". So the
   landscape term is capped below what a real subject can earn.
3. **A recognisable face is content.** Any face above a minimum size gets a
   high content floor, because a person in frame is the thing a human is
   usually looking for (R-VIS-006), while texture statistics are only a
   proxy for content.
4. **A missing detector is not a zero.** ``None`` drops the term and
   renormalises the weights; it never scores as "bad".
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from . import strings_lt as S

WEIGHTS = {          # starting values - NOT measured, to be calibrated on
    "content":     0.50,   # Tomas's own footage before anyone trusts them
    "technical":   0.25,
    "composition": 0.15,
    "moment":      0.10,   # is this a moving shot, and is anyone in it
}

# -- content ---------------------------------------------------------------
#: Faces smaller than this fraction of the frame are not treated as subject.
FACE_MIN_REL = 0.01
#: Exponential scale of the face term; a half-frame face is fully saturated.
FACE_SATURATION_REL = 0.12
#: Any qualifying face starts here (principle 3).
FACE_FLOOR = 0.70
#: Ceiling of the statistics-only landscape term (principle 2).
LANDSCAPE_CEILING = 0.75

# -- technical -------------------------------------------------------------
#: Score of a frame that is in focus but not the sharpest in the clip.
TECHNICAL_FLOOR = 0.35
#: Sharpness percentile at which "sharp enough" is reached (principle 1).
SHARPNESS_SATURATION = 0.70
#: Clipping fractions at which the penalty is fully applied.
CLIP_HIGH_FULL_PENALTY = 0.10
CLIP_LOW_FULL_PENALTY = 0.25
#: Blown highlights are unrecoverable, crushed blacks are merely ugly.
CLIP_HIGH_SHARE = 0.6
CLIP_LOW_SHARE = 0.4
#: How much of the technical term clipping can eat.
CLIP_MAX_PENALTY = 0.8

# -- moment ----------------------------------------------------------------
#: A locked-off shot and an orbit are different kinds of frame. Motion is
#: measured between consecutive samples, ranked within the clip, and only
#: counts as a "moment" together with what is in the frame: a moving shot with
#: a person in it is the thing being looked for, a moving shot of nothing is
#: worth a little, and a static shot is worth none of this term.
MOTION_WITHOUT_PEOPLE = 0.45
MOTION_WITH_PEOPLE = 1.0

# -- horizon ---------------------------------------------------------------
#: How much of the technical term a fully crooked horizon can eat. A tilted
#: horizon is the one composition defect in drone footage that is unambiguous.
TILT_MAX_PENALTY = 0.25

# -- composition -----------------------------------------------------------
#: Share of the composition term carried by how cleanly the subject separates
#: from its surroundings; the rest is the rule-of-thirds placement.
SEPARATION_SHARE = 0.25

# -- confidence (TRAP-11) --------------------------------------------------
#: Number of top candidates whose spread decides whether the ranking is
#: informative at all.
CONFIDENCE_SAMPLE = 20
#: Starting threshold - to be tuned on real footage, like the weights.
CONFIDENCE_MIN_SPREAD = 0.08


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else float(value))


def face_component(face_max_rel: float | None) -> float | None:
    """Content contribution of the largest face, or ``None`` if unknown."""
    if face_max_rel is None:
        return None
    if face_max_rel < FACE_MIN_REL:
        return 0.0
    saturating = 1.0 - math.exp(-float(face_max_rel) / FACE_SATURATION_REL)
    return _clamp01(FACE_FLOOR + (1.0 - FACE_FLOOR) * saturating)


def landscape_component(colorfulness_rank: float, dynamic_range_rank: float) -> float:
    """Content contribution of a frame with no people in it."""
    mixed = 0.5 * _clamp01(colorfulness_rank) + 0.5 * _clamp01(dynamic_range_rank)
    return LANDSCAPE_CEILING * mixed


def clipping_penalty(clip_low: float, clip_high: float) -> float:
    """0 = nothing clipped, 1 = fully clipped in both directions."""
    high = _clamp01(float(clip_high) / CLIP_HIGH_FULL_PENALTY)
    low = _clamp01(float(clip_low) / CLIP_LOW_FULL_PENALTY)
    return _clamp01(CLIP_HIGH_SHARE * high + CLIP_LOW_SHARE * low)


def technical_component(
    sharpness_rank: float, clip_low: float, clip_high: float, horizon_tilt: float | None = None
) -> float:
    """Sharpness as a gate, reduced by clipping and by a crooked horizon."""
    sharp = TECHNICAL_FLOOR + (1.0 - TECHNICAL_FLOOR) * _clamp01(
        _clamp01(sharpness_rank) / SHARPNESS_SATURATION
    )
    value = sharp * (1.0 - CLIP_MAX_PENALTY * clipping_penalty(clip_low, clip_high))
    if horizon_tilt is not None:
        value *= 1.0 - TILT_MAX_PENALTY * _clamp01(horizon_tilt)
    return _clamp01(value)


def moment_component(motion_rank: float | None, face_present: bool | None) -> float | None:
    """Is this a moving shot, and is anyone in it? ``None`` if motion is unknown."""
    if motion_rank is None:
        return None
    weight = MOTION_WITH_PEOPLE if face_present else MOTION_WITHOUT_PEOPLE
    return _clamp01(_clamp01(motion_rank) * weight)


def composition_from(thirds_dist: float | None, separation: float | None) -> float | None:
    """Thirds placement, plus how cleanly the subject stands out."""
    if thirds_dist is None:
        return None
    placement = _clamp01(1.0 - _clamp01(thirds_dist))
    if separation is None:
        return placement
    return _clamp01((1.0 - SEPARATION_SHARE) * placement + SEPARATION_SHARE * _clamp01(separation))


def composition_component(thirds_dist: float | None) -> float | None:
    """Rule-of-thirds bonus, or ``None`` when there is no subject to place."""
    if thirds_dist is None:
        return None
    return _clamp01(1.0 - _clamp01(thirds_dist))


def score_frame_explained(features: dict) -> tuple[float, list[str]]:
    """Score one frame in 0..1 and say, in Lithuanian, why.

    Expected keys: ``sharpness_rank``, ``colorfulness_rank``,
    ``dynamic_range_rank`` (within-clip percentile ranks, 0..1),
    ``exposure_clip_low``, ``exposure_clip_high``. Optional and possibly
    ``None``: ``face_count``, ``face_max_rel``, ``subject_rel``,
    ``thirds_distance``.
    """
    sharp_rank = float(features.get("sharpness_rank") or 0.0)
    color_rank = float(features.get("colorfulness_rank") or 0.0)
    range_rank = float(features.get("dynamic_range_rank") or 0.0)
    clip_low = float(features.get("exposure_clip_low") or 0.0)
    clip_high = float(features.get("exposure_clip_high") or 0.0)

    faces_known = features.get("face_max_rel") is not None
    face_term = face_component(features.get("face_max_rel"))
    land_term = landscape_component(color_rank, range_rank)
    content = land_term if face_term is None else max(face_term, land_term)

    technical = technical_component(sharp_rank, clip_low, clip_high, features.get("horizon_tilt"))
    composition = composition_from(features.get("thirds_distance"), features.get("subject_separation"))
    moment = moment_component(
        features.get("motion_rank"),
        bool(features.get("face_max_rel") or 0.0) if faces_known else None,
    )

    parts = {"content": content, "technical": technical}
    if composition is not None:
        parts["composition"] = composition
    if moment is not None:
        parts["moment"] = moment
    total_weight = sum(WEIGHTS[k] for k in parts)
    score = sum(WEIGHTS[k] * v for k, v in parts.items()) / total_weight

    reasons: list[str] = []
    if not faces_known:
        reasons.append(S.reason_faces_unknown())
    elif features.get("face_max_rel", 0.0) >= FACE_MIN_REL:
        reasons.append(S.reason_face_area(float(features["face_max_rel"])))
        count = features.get("face_count")
        if count:
            reasons.append(S.reason_faces_count(int(count)))
    else:
        reasons.append(S.reason_no_faces())

    reasons.append(S.reason_sharpness_rank(sharp_rank * 100.0, features.get("sharpness")))
    if sharp_rank >= SHARPNESS_SATURATION:
        # Without this line the report says "sharpness: 74th percentile" and
        # "technical: 1.00" next to each other and looks like it is lying.
        reasons.append(S.reason_technical_gate(SHARPNESS_SATURATION * 100.0))
    reasons.append(S.reason_colorfulness_rank(color_rank * 100.0))
    reasons.append(S.reason_dynamic_range_rank(range_rank * 100.0))
    if clip_high > 0.005:
        reasons.append(S.reason_clipped_high(clip_high))
    if clip_low > 0.02:
        reasons.append(S.reason_clipped_low(clip_low))
    tilt = features.get("horizon_tilt")
    if tilt is not None and tilt > 0.15:
        reasons.append(S.reason_horizon_tilt(float(tilt) * S.TILT_FULL_DEGREES_FOR_TEXT))
    if composition is None:
        reasons.append(S.reason_composition_unknown())
    else:
        reasons.append(S.reason_thirds(float(features["thirds_distance"])))
        if features.get("subject_rel") is not None:
            reasons.append(S.reason_subject_size(float(features["subject_rel"])))
        if features.get("subject_separation") is not None:
            reasons.append(S.reason_subject_separation(float(features["subject_separation"])))
    if moment is not None:
        reasons.append(S.reason_motion(float(features.get("motion_rank") or 0.0) * 100.0))
        if not (features.get("face_max_rel") or 0.0):
            reasons.append(S.reason_moment_capped(MOTION_WITHOUT_PEOPLE))

    reasons.append(S.reason_component(S.COMPONENT_CONTENT, content))
    reasons.append(S.reason_component(S.COMPONENT_TECHNICAL, technical))
    if composition is not None:
        reasons.append(S.reason_component(S.COMPONENT_COMPOSITION, composition))
    if moment is not None:
        reasons.append(S.reason_component(S.COMPONENT_MOMENT, moment))

    return _clamp01(score), reasons


def confidence(scores: Sequence[float], sample: int = CONFIDENCE_SAMPLE,
               threshold: float = CONFIDENCE_MIN_SPREAD) -> dict:
    """Is this ranking worth reading? (TRAP-11)

    A flat distribution must never be presented in the same confident format
    as a real one.
    """
    ordered = sorted((float(s) for s in scores), reverse=True)[:sample]
    if len(ordered) < 2:
        spread = 0.0
        informative = False
    else:
        spread = ordered[0] - ordered[-1]
        informative = spread >= threshold
    message = S.confidence_ok(spread, len(ordered)) if informative else S.confidence_low(spread, len(ordered))
    return {
        "informative": informative,
        "spread": spread,
        "sample": len(ordered),
        "threshold": threshold,
        "message": message,
    }


def attach_ranks(feature_dicts: Iterable[dict]) -> list[dict]:
    """Add within-clip percentile ranks to a list of raw feature dicts."""
    from .features import percentile_ranks

    items = list(feature_dicts)
    if not items:
        return items
    for key in ("sharpness", "dynamic_range", "colorfulness", "motion"):
        if all(item.get(key) is None for item in items):
            continue
        ranks = percentile_ranks([float(item.get(key) or 0.0) for item in items])
        for item, rank in zip(items, ranks):
            # A frame with no measurement keeps None: the first frame of a clip
            # has no motion to speak of, and that is not "no motion".
            item[f"{key}_rank"] = None if item.get(key) is None else rank
    return items
