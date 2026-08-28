"""Calibration: what Tomas actually keeps, compared with what the tool picked.

The weights in :mod:`framepicker.scoring` are chosen numbers, and the module
says so. This is how they stop being chosen numbers: run a batch, keep the
frames you would actually use, and point this tool at both.

    python -m framepicker.learn OUT/run-.../results.json --picks D:/keepers

It reads which exported frames survived, measures how the kept ones differ
from the discarded ones component by component, and proposes weights. Then it
checks its own proposal: it re-ranks every frame with them and reports how
many of the kept frames the ranking would have put on top, before and after.

Three rules it does not break:

* Nothing is fitted below :data:`MIN_PICKS` kept and :data:`MIN_REJECTS`
  discarded frames. A weight vector from six examples is noise with a decimal
  point.
* A component that could not be measured is not evidence. It drops out of the
  averages instead of counting as zero.
* The proposal is printed, never applied. Weights change when a person edits
  them, so a bad calibration can never quietly rewrite the tool's judgement.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Iterable, Sequence

from . import strings_lt as S
from .scoring import WEIGHTS, score_from_components

#: Below these counts nothing is proposed, only described.
MIN_PICKS = 20
MIN_REJECTS = 20
#: Cohen's d below this is not treated as a difference worth weighing.
MIN_EFFECT = 0.20
#: Cap on the reported effect size. Two groups that never overlap have an
#: undefined d (the within-group spread is zero), which is the strongest
#: separation there is, not a missing measurement - so it is reported as this.
MAX_EFFECT = 3.0
#: No component's weight is moved outside this band, whatever the numbers say:
#: one enthusiastic evening of picking must not be able to delete a component.
WEIGHT_MIN = 0.05
WEIGHT_MAX = 0.60
#: Frames counted as "on top" when checking a proposal.
TOP_N = 20

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff")


def read_picks(target: str) -> set[str]:
    """Filenames of the frames that were kept.

    *target* is either a folder (the kept images are in it, under whatever
    names they now have) or a text file with one name per line. Matching is by
    basename without extension, so re-saving a JPEG as PNG or copying it
    somewhere else does not break the link.
    """
    names: set[str] = set()
    if os.path.isdir(target):
        for entry in os.listdir(target):
            if entry.lower().endswith(IMAGE_SUFFIXES):
                names.add(_key(entry))
        return names
    with open(target, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                names.add(_key(line))
    return names


def _key(name: str) -> str:
    return os.path.splitext(os.path.basename(name.strip()))[0].lower()


def collect_frames(results: dict) -> list[dict]:
    """Every exported frame in a results.json, with its components."""
    frames: list[dict] = []
    for clip in results.get("clips", []):
        for frame in clip.get("frames", []):
            if not frame.get("file"):
                continue
            frames.append({
                "file": frame["file"],
                "key": _key(frame["file"]),
                "clip": clip.get("probe", {}).get("name"),
                "score": float(frame.get("score") or 0.0),
                "components": dict(frame.get("features", {}).get("components") or {}),
            })
    return frames


def _mean(values: Sequence[float]) -> float | None:
    usable = [float(v) for v in values if v is not None]
    return sum(usable) / len(usable) if usable else None


def _std(values: Sequence[float]) -> float | None:
    usable = [float(v) for v in values if v is not None]
    if len(usable) < 2:
        return None
    mean = sum(usable) / len(usable)
    return math.sqrt(sum((v - mean) ** 2 for v in usable) / (len(usable) - 1))


def effect_size(kept: Sequence[float], dropped: Sequence[float]) -> float | None:
    """Cohen's d. ``None`` when either side has nothing measurable in it."""
    kept_mean, dropped_mean = _mean(kept), _mean(dropped)
    kept_std, dropped_std = _std(kept), _std(dropped)
    if kept_mean is None or dropped_mean is None or kept_std is None or dropped_std is None:
        return None
    difference = kept_mean - dropped_mean
    pooled = math.sqrt((kept_std ** 2 + dropped_std ** 2) / 2.0)
    if pooled < 1e-9:
        # No spread inside either group. If the means differ, the two groups
        # are perfectly separated; if they do not, there is nothing here.
        if abs(difference) < 1e-9:
            return None
        return math.copysign(MAX_EFFECT, difference)
    return max(-MAX_EFFECT, min(MAX_EFFECT, difference / pooled))


def propose_weights(effects: dict, current: dict | None = None) -> dict:
    """Weights in proportion to how well each component separates the picks.

    A component that does not separate them keeps a floor, not a zero: "this
    batch says nothing about composition" is not the same as "composition does
    not matter".
    """
    current = dict(current or WEIGHTS)
    raw: dict[str, float] = {}
    for name, weight in current.items():
        effect = effects.get(name)
        if effect is None or effect <= MIN_EFFECT:
            raw[name] = weight * 0.5      # halved, not deleted
        else:
            raw[name] = weight * (1.0 + min(2.0, effect))
    total = sum(raw.values()) or 1.0
    scaled = {name: value / total for name, value in raw.items()}
    # Clamp, renormalise, repeat: renormalising after a clamp can push another
    # component back out of the band, so it is done until it stops moving.
    for _ in range(20):
        clamped = {name: min(WEIGHT_MAX, max(WEIGHT_MIN, value)) for name, value in scaled.items()}
        total = sum(clamped.values()) or 1.0
        renormalised = {name: value / total for name, value in clamped.items()}
        if all(
            WEIGHT_MIN - 1e-9 <= value <= WEIGHT_MAX + 1e-9
            for value in renormalised.values()
        ):
            scaled = renormalised
            break
        scaled = renormalised
    return {name: round(value, 3) for name, value in scaled.items()}


def hit_rate(frames: Iterable[dict], picks: set[str], weights: dict, top_n: int = TOP_N) -> dict:
    """How many kept frames land in the top *top_n* under *weights*."""
    ranked = sorted(frames, key=lambda f: -score_from_components(f["components"], weights))
    top = ranked[:top_n]
    hits = sum(1 for frame in top if frame["key"] in picks)
    return {"top_n": len(top), "hits": hits,
            "rate": hits / len(top) if top else 0.0}


def analyse(results: dict, picks: set[str]) -> dict:
    """Compare kept against discarded, component by component."""
    frames = collect_frames(results)
    kept = [f for f in frames if f["key"] in picks]
    dropped = [f for f in frames if f["key"] not in picks]

    components = sorted({name for f in frames for name in f["components"]})
    per_component: dict[str, dict] = {}
    effects: dict[str, float] = {}
    for name in components:
        kept_values = [f["components"].get(name) for f in kept]
        dropped_values = [f["components"].get(name) for f in dropped]
        effect = effect_size(kept_values, dropped_values)
        per_component[name] = {
            "kept_mean": _mean(kept_values),
            "dropped_mean": _mean(dropped_values),
            "kept_measured": sum(1 for v in kept_values if v is not None),
            "dropped_measured": sum(1 for v in dropped_values if v is not None),
            "effect_size": effect,
        }
        if effect is not None:
            effects[name] = effect

    enough = len(kept) >= MIN_PICKS and len(dropped) >= MIN_REJECTS
    report = {
        "frames_in_report": len(frames),
        "picks_matched": len(kept),
        "picks_given": len(picks),
        "picks_not_found": sorted(picks - {f["key"] for f in frames})[:20],
        "dropped": len(dropped),
        "enough_data": enough,
        "minimum": {"picks": MIN_PICKS, "rejects": MIN_REJECTS},
        "components": per_component,
        "current_weights": dict(WEIGHTS),
        "proposed_weights": None,
        "check": None,
        "messages": [],
    }
    if not enough:
        report["messages"].append(S.learn_not_enough(len(kept), len(dropped), MIN_PICKS, MIN_REJECTS))
        return report

    proposed = propose_weights(effects)
    report["proposed_weights"] = proposed
    before = hit_rate(frames, picks, WEIGHTS)
    after = hit_rate(frames, picks, proposed)
    report["check"] = {"before": before, "after": after,
                       "improved": after["hits"] > before["hits"]}
    report["messages"].append(S.learn_summary(len(kept), len(dropped)))
    report["messages"].append(S.learn_hit_rate(before["hits"], after["hits"], before["top_n"]))
    if after["hits"] <= before["hits"]:
        report["messages"].append(S.learn_no_improvement())
    report["messages"].append(S.learn_not_applied())
    return report


def format_report(report: dict) -> str:
    lines: list[str] = [S.LEARN_TITLE, ""]
    lines.append(S.learn_counts(report["frames_in_report"], report["picks_matched"],
                               report["picks_given"], report["dropped"]))
    if report["picks_not_found"]:
        lines.append(S.learn_unmatched(", ".join(report["picks_not_found"])))
    lines.append("")
    lines.append(S.LEARN_COMPONENT_HEADER)
    for name, data in report["components"].items():
        lines.append(S.learn_component_line(
            name, data["kept_mean"], data["dropped_mean"], data["effect_size"],
            data["kept_measured"], data["dropped_measured"]))
    lines.append("")
    if report["proposed_weights"]:
        lines.append(S.LEARN_WEIGHTS_HEADER)
        for name, value in report["proposed_weights"].items():
            lines.append(S.learn_weight_line(name, WEIGHTS.get(name, 0.0), value))
        lines.append("")
    lines += list(report["messages"])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m framepicker.learn",
        description="Compare the frames you kept with the frames the tool picked.",
    )
    parser.add_argument("results", help="path to a run's results.json")
    parser.add_argument("--picks", required=True,
                        help="folder of the frames you kept, or a text file of their names")
    parser.add_argument("--out", default=None, help="write the analysis to this JSON file too")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        with open(args.results, "r", encoding="utf-8") as handle:
            results = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(S.learn_cannot_read(args.results, str(exc)), flush=True)
        return 2
    try:
        picks = read_picks(args.picks)
    except OSError as exc:
        print(S.learn_cannot_read(args.picks, str(exc)), flush=True)
        return 2

    report = analyse(results, picks)
    print(format_report(report), flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
    return 0 if report["enough_data"] else 1


if __name__ == "__main__":
    sys.exit(main())
