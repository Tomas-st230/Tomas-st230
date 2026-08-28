"""Named looks for the exported stills.

The point of this module is what a look is *not*: it is not a fixed curve
slapped on top of whatever came out of the camera. Every preset is expressed
as a set of **targets**, and each frame is measured and then moved partway
toward those targets. A frame that is already saturated is barely touched; a
flat one is lifted. Nothing is multiplied blindly, and ``strength`` scales the
whole move, so 0.0 is the untouched frame and 1.0 lands on the targets.

Applied to the **exported image only**, never to the analysis frames: the
scores have to stay comparable between clips, and a look is a taste decision
while the score is meant to be a measurement. The report says which look was
used on every clip.

The preset numbers below are chosen, not measured. Like the weights, they are
a starting point to be argued with.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NONE = "none"
NATURE = "nature"
CITY = "city"
#: Not a look: "measure the clip and pick one of the above, or none".
AUTO = "auto"


@dataclass(frozen=True)
class Look:
    """A look expressed as targets, not as a curve."""

    name: str
    #: Mean HSV saturation the frame is moved toward, 0..1.
    target_saturation: float
    #: Luma p1-p99 span the frame is moved toward, 0..1.
    target_span: float
    #: Warm/cool shift. Positive warms (more red, less blue), -1..1.
    warmth: float
    #: Shadow lift, fraction of the range; keeps crushed blacks off the floor.
    lift: float
    #: Highlight rolloff, 0..1; compresses a bright sky instead of clipping it.
    highlight_rolloff: float

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "target_saturation": self.target_saturation,
            "target_span": self.target_span,
            "warmth": self.warmth,
            "lift": self.lift,
            "highlight_rolloff": self.highlight_rolloff,
        }


#: Landscape / golden hour: a little warmer and a little richer, highlights
#: protected because the sky is usually the brightest thing in the frame.
LOOK_NATURE = Look(NATURE, target_saturation=0.42, target_span=0.82,
                   warmth=0.10, lift=0.02, highlight_rolloff=0.35)

#: Architecture / streets: cooler and crisper, blacks kept honest, saturation
#: pulled back so concrete and glass do not go candy-coloured.
LOOK_CITY = Look(CITY, target_saturation=0.30, target_span=0.90,
                 warmth=-0.07, lift=0.0, highlight_rolloff=0.20)

LOOKS = {NATURE: LOOK_NATURE, CITY: LOOK_CITY}

#: Fraction of the way to the targets travelled by default. Not 1.0 on
#: purpose: a look you can see is a look you can argue with.
DEFAULT_STRENGTH = 0.6

#: Bounds on the saturation multiplier, so an already-rich frame is left alone
#: and a nearly grey one is never dragged into neon.
SATURATION_GAIN_MIN = 0.7
SATURATION_GAIN_MAX = 1.45

#: Hard cap on the contrast stretch. Without it, a nearly flat frame - fog,
#: a grey sky, a night shot - asks for a scale of 100x or more to reach the
#: target span, which explodes its last trace of colour into neon. This is the
#: same failure that a 2.5x saturation multiplier already caused once.
CONTRAST_SCALE_MAX = 1.6


def available() -> tuple[str, ...]:
    return (NONE, AUTO, NATURE, CITY)


# --------------------------------------------------------------------------
# --look auto
# --------------------------------------------------------------------------
#
# The evidence is deliberately coarse and scale-free. Two things make a naive
# version wrong on this footage:
#
# 1. A D-Log frame is desaturated everywhere, so counting grey pixels alone
#    labels a sunset over water as a city. So the two masses are compared as
#    *shares of each other*, never against a fixed threshold.
# 2. Grey without structure is haze, cloud or water, not architecture. So the
#    city side has to be carried by vertical lines as well as by grey.
#
# The decision is made once per clip on the mean of the sampled frames, not
# per frame: a look that changes shot to shot inside one clip would be worse
# than no look at all.

#: Minimum lead one side needs over the other. Below it, neither look is
#: applied and the report says the evidence was too close to call.
AUTO_MARGIN = 0.08
#: Frames measured per clip. More than this buys nothing; the evidence is a
#: whole-clip average of wide colour bands.
AUTO_SAMPLE = 8
#: Weight of the "there are vertical lines here" evidence in the city score.
#: The floor is not zero: a street at night has few clean lines and is still
#: a street.
STRUCTURE_FLOOR = 0.4


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def classify_frames(signatures: list[dict]) -> dict:
    """Pick a look for one clip from :func:`features.scene_signature` output.

    Returns the choice, both scores, the averaged evidence and the number of
    frames it was measured on. ``choice`` is :data:`NONE` when the two scores
    are within :data:`AUTO_MARGIN` of each other - an undecided answer is a
    real answer here, and better than grading a clip the wrong way.
    """
    usable = [s for s in signatures if s]
    if not usable:
        return {"choice": NONE, "nature_score": None, "city_score": None,
                "margin": None, "evidence": None, "frames_measured": 0,
                "decided": False}

    evidence = {
        "vegetation": _mean([float(s.get("vegetation") or 0.0) for s in usable]),
        "sky": _mean([float(s.get("sky") or 0.0) for s in usable]),
        "warm": _mean([float(s.get("warm") or 0.0) for s in usable]),
        "grey": _mean([float(s.get("grey") or 0.0) for s in usable]),
        # None means "too few lines to divide by", which is evidence against
        # architecture, not a missing measurement to be filled in with a mean.
        "vertical_share": _mean([float(s.get("vertical_share") or 0.0) for s in usable]),
        "vertical_measured": sum(1 for s in usable if s.get("vertical_share") is not None),
    }

    colour_mass = evidence["vegetation"] + evidence["sky"] + evidence["warm"]
    grey_mass = evidence["grey"]
    total = colour_mass + grey_mass
    if total <= 1e-6:
        # Nothing to go on: an almost black clip, or one that is all blown out.
        return {"choice": NONE, "nature_score": 0.0, "city_score": 0.0,
                "margin": 0.0, "evidence": evidence,
                "frames_measured": len(usable), "decided": False}

    nature_score = colour_mass / total
    structure = STRUCTURE_FLOOR + (1.0 - STRUCTURE_FLOOR) * min(1.0, evidence["vertical_share"])
    city_score = (grey_mass / total) * structure

    margin = abs(nature_score - city_score)
    if margin < AUTO_MARGIN:
        choice = NONE
        decided = False
    else:
        choice = NATURE if nature_score > city_score else CITY
        decided = True
    return {"choice": choice, "nature_score": nature_score, "city_score": city_score,
            "margin": margin, "evidence": evidence, "frames_measured": len(usable),
            "decided": decided}


def get(name: str | None) -> Look | None:
    if not name or name == NONE:
        return None
    return LOOKS.get(name)


def measure(frame: np.ndarray) -> dict:
    """What this frame already is, before anything is decided about it."""
    import cv2

    arr = np.ascontiguousarray(np.asarray(frame, dtype=np.uint8))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    luma = arr.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    lo, hi = np.percentile(luma, (1.0, 99.0))
    return {
        "saturation": float(hsv[..., 1].mean() / 255.0),
        "luma_low": float(lo),
        "luma_high": float(hi),
        "span": float(max(0.0, hi - lo) / 255.0),
    }


def _toward(current: float, target: float, strength: float) -> float:
    """Move *current* a *strength* fraction of the way to *target*."""
    return current + (target - current) * strength


def apply_look(frame: np.ndarray, look: Look, strength: float = DEFAULT_STRENGTH) -> tuple[np.ndarray, dict]:
    """Grade *frame* toward *look*. Returns the image and what was actually done."""
    import cv2

    strength = float(min(1.0, max(0.0, strength)))
    before = measure(frame)
    original = np.asarray(frame, dtype=np.uint8)
    if strength <= 0.0:
        return original, {"look": look.name, "strength": 0.0, "before": before,
                          "after": before, "applied": {}}

    work = original.astype(np.float32) / 255.0

    # --- contrast: stretch the measured span toward the target -------------
    wanted_span = _toward(before["span"], look.target_span, strength)
    scale = 1.0
    if before["span"] > 1e-3 and wanted_span > before["span"]:
        scale = min(CONTRAST_SCALE_MAX, wanted_span / before["span"])
        centre = (before["luma_low"] + before["luma_high"]) / 510.0
        work = (work - centre) * scale + centre

    # --- highlight rolloff: compress the top instead of clipping it --------
    # Asymptotic on purpose: whatever comes in above the knee lands strictly
    # below 1.0, so a bright sky keeps its gradation instead of becoming one
    # flat white shape. A linear divisor could still exceed 1.0 and clip.
    rolloff = look.highlight_rolloff * strength
    if rolloff > 1e-3:
        knee = 1.0 - 0.4 * rolloff
        headroom = max(1.0 - knee, 1e-6)
        high = work > knee
        if high.any():
            work[high] = knee + headroom * (1.0 - np.exp(-(work[high] - knee) / headroom))

    # --- shadow lift ------------------------------------------------------
    lift = look.lift * strength
    if lift > 1e-4:
        work = work * (1.0 - lift) + lift

    # --- warmth: a channel tilt ------------------------------------------
    warmth = look.warmth * strength
    if abs(warmth) > 1e-4:
        work[..., 0] *= 1.0 + 0.16 * warmth
        work[..., 2] *= 1.0 - 0.16 * warmth

    np.clip(work, 0.0, 1.0, out=work)
    graded = (work * 255.0).astype(np.uint8)

    # --- saturation: measured, then moved toward the target ---------------
    hsv = cv2.cvtColor(graded, cv2.COLOR_RGB2HSV).astype(np.float32)
    current_sat = float(hsv[..., 1].mean() / 255.0)
    wanted_sat = _toward(current_sat, look.target_saturation, strength)
    gain = 1.0
    if current_sat > 0.01:
        gain = float(min(SATURATION_GAIN_MAX, max(SATURATION_GAIN_MIN, wanted_sat / current_sat)))
        if abs(gain - 1.0) > 1e-3:
            hsv[..., 1] *= gain
            np.clip(hsv[..., 1], 0, 255, out=hsv[..., 1])
            graded = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    return graded, {
        "look": look.name,
        "strength": strength,
        "before": before,
        "after": measure(graded),
        "applied": {
            "span_target": wanted_span,
            "contrast_scale": scale,
            "saturation_target": wanted_sat,
            "saturation_gain": gain,
            "warmth": warmth,
            "lift": lift,
            "highlight_rolloff": rolloff,
        },
    }
