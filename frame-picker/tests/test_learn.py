"""Calibration must refuse to pretend.

The weights are chosen numbers until somebody's own picks say otherwise, and
this module is the only path from picks to weights. Its dangerous failure mode
is confidence: proposing a weight vector from a handful of examples, or
proposing one that does not actually rank the picks any higher.
"""

from __future__ import annotations

import json
import os

from framepicker import learn
from framepicker import strings_lt as S
from framepicker.scoring import WEIGHTS


def _results(frames: list[dict]) -> dict:
    return {
        "clips": [{
            "probe": {"name": "clip.mp4"},
            "frames": [
                {"file": f"frame_{i:03d}.jpg", "score": frame.get("score", 0.5),
                 "features": {"components": frame["components"]}}
                for i, frame in enumerate(frames)
            ],
        }],
    }


def _frame(content: float, technical: float = 0.8, composition: float = 0.5,
           moment: float = 0.3) -> dict:
    return {"components": {"content": content, "technical": technical,
                            "composition": composition, "moment": moment}}


def test_a_handful_of_picks_proposes_nothing():
    results = _results([_frame(0.9)] * 5 + [_frame(0.2)] * 5)
    picks = {learn._key(f"frame_{i:03d}.jpg") for i in range(5)}
    report = learn.analyse(results, picks)
    assert report["enough_data"] is False
    assert report["proposed_weights"] is None
    assert S.learn_not_enough(5, 5, learn.MIN_PICKS, learn.MIN_REJECTS) in report["messages"]


def test_the_component_that_separates_the_picks_gains_weight():
    kept = [_frame(0.95, technical=0.8) for _ in range(25)]
    dropped = [_frame(0.15, technical=0.8) for _ in range(25)]
    results = _results(kept + dropped)
    picks = {learn._key(f"frame_{i:03d}.jpg") for i in range(25)}

    report = learn.analyse(results, picks)
    assert report["enough_data"] is True
    assert report["components"]["content"]["effect_size"] > 1.0
    assert report["components"]["technical"]["effect_size"] is None, \
        "a component with the same value on both sides cannot be evidence"
    proposed = report["proposed_weights"]
    assert proposed["content"] > WEIGHTS["content"]
    assert abs(sum(proposed.values()) - 1.0) < 0.01


def test_no_weight_is_ever_deleted_or_allowed_to_take_over():
    kept = [_frame(0.99, technical=0.1, composition=0.1, moment=0.1) for _ in range(30)]
    dropped = [_frame(0.01, technical=0.9, composition=0.9, moment=0.9) for _ in range(30)]
    results = _results(kept + dropped)
    picks = {learn._key(f"frame_{i:03d}.jpg") for i in range(30)}
    proposed = learn.analyse(results, picks)["proposed_weights"]
    for name, value in proposed.items():
        assert learn.WEIGHT_MIN - 0.002 <= value <= learn.WEIGHT_MAX + 0.002, (name, value)


def test_the_proposal_is_checked_against_the_picks():
    kept = [_frame(0.9) for _ in range(25)]
    dropped = [_frame(0.2) for _ in range(25)]
    results = _results(kept + dropped)
    picks = {learn._key(f"frame_{i:03d}.jpg") for i in range(25)}
    report = learn.analyse(results, picks)
    check = report["check"]
    assert check["before"]["top_n"] == learn.TOP_N
    assert check["after"]["hits"] >= check["before"]["hits"]


def test_a_proposal_that_does_not_help_says_so():
    """Picks that the components do not explain must not produce a confident answer."""
    frames = [_frame(0.5) for _ in range(60)]
    results = _results(frames)
    picks = {learn._key(f"frame_{i:03d}.jpg") for i in range(0, 60, 2)}
    report = learn.analyse(results, picks)
    assert report["enough_data"] is True
    assert S.learn_no_improvement() in report["messages"] or report["check"]["improved"] is False


def test_the_weights_are_never_applied_automatically():
    results = _results([_frame(0.9)] * 25 + [_frame(0.2)] * 25)
    picks = {learn._key(f"frame_{i:03d}.jpg") for i in range(25)}
    report = learn.analyse(results, picks)
    assert S.learn_not_applied() in report["messages"]
    assert dict(WEIGHTS) == report["current_weights"], "scoring.WEIGHTS must be untouched"


def test_picks_are_matched_by_name_not_by_extension(tmp_path):
    """A kept frame re-saved as PNG, or copied elsewhere, still counts."""
    folder = tmp_path / "keepers"
    folder.mkdir()
    (folder / "frame_001.png").write_bytes(b"x")
    (folder / "FRAME_002.JPG").write_bytes(b"x")
    (folder / "notes.txt").write_bytes(b"x")
    picks = learn.read_picks(str(folder))
    assert picks == {"frame_001", "frame_002"}


def test_picks_can_come_from_a_text_file(tmp_path):
    listing = tmp_path / "picks.txt"
    listing.write_text("# my keepers\nframe_003.jpg\nD:/somewhere/frame_004.jpg\n\n",
                       encoding="utf-8")
    assert learn.read_picks(str(listing)) == {"frame_003", "frame_004"}


def test_names_that_are_not_in_the_report_are_named(tmp_path):
    results = _results([_frame(0.9)] * 3)
    report = learn.analyse(results, {"frame_000", "who_is_this"})
    assert "who_is_this" in report["picks_not_found"]


def test_the_command_line_runs_end_to_end(tmp_path, capsys):
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(_results([_frame(0.9)] * 25 + [_frame(0.2)] * 25)),
                            encoding="utf-8")
    picks_dir = tmp_path / "keepers"
    picks_dir.mkdir()
    for i in range(25):
        (picks_dir / f"frame_{i:03d}.jpg").write_bytes(b"x")
    out = tmp_path / "analysis.json"

    code = learn.main([str(results_path), "--picks", str(picks_dir), "--out", str(out)])
    assert code == 0
    printed = capsys.readouterr().out
    assert S.LEARN_TITLE in printed
    assert S.LEARN_WEIGHTS_HEADER in printed
    assert os.path.isfile(out)
    assert json.loads(out.read_text(encoding="utf-8"))["proposed_weights"]
