"""End-to-end behaviour on generated fixtures."""

from __future__ import annotations

import json
import os

import numpy as np

from framepicker import features, report, scoring
from framepicker import strings_lt as S
from framepicker.cli import Options, run_batch
from framepicker.decode import detect_log
from framepicker.probe import probe

from conftest import requires_ffmpeg

MISSING_MODEL = os.path.join(os.sep, "definitely", "not", "a", "yunet", "model.onnx")


def _run(paths, tmp_path, **kwargs):
    options = Options(paths=list(paths), out_dir=str(tmp_path / "out"), **kwargs)
    return run_batch(options)


@requires_ffmpeg
def test_missing_detector_is_none_not_zero(normal_clip, tmp_path):
    """No YuNet model: face features stay None, the run continues, the report says so."""
    result = _run([normal_clip], tmp_path, face_model=MISSING_MODEL, per_clip=3, min_gap=1.0)

    clip = result.results["clips"][0]
    assert clip["detectors"]["faces"]["available"] is False
    assert clip["detectors"]["statistics_only"] is True
    assert clip["frames"], "the run must continue without a face detector"
    for frame in clip["frames"]:
        assert frame["features"]["face_count"] is None
        assert frame["features"]["face_max_rel"] is None
        assert S.reason_faces_unknown() in frame["reasons"]

    assert S.STATISTICS_ONLY_MODE in clip["notes"]
    html = open(os.path.join(result.out_dir, report.REPORT_HTML), encoding="utf-8").read()
    assert S.STATISTICS_ONLY_MODE in html


@requires_ffmpeg
def test_shortfall_is_reported(short_clip, tmp_path):
    """A five second clip asked for six frames returns fewer, with a reason."""
    result = _run([short_clip], tmp_path, per_clip=6, min_gap=2.0)

    clip = result.results["clips"][0]
    selection = clip["selection"]
    assert selection["requested"] == 6
    assert selection["delivered"] < 6
    assert len(clip["frames"]) == selection["delivered"]
    assert selection["shortfall_reasons"], "a silent shortfall is the bug this test exists for"
    assert any(S.shortfall_header(selection["delivered"], 6) in m for m in result.messages)

    summary = result.results["summary"]
    assert summary["frames_requested"] == 6
    assert summary["frames_delivered"] == selection["delivered"]


@requires_ffmpeg
def test_unreadable_file_does_not_abort_batch(normal_clip, unreadable_file, blurred_clip, tmp_path):
    """One bad file is named and skipped; the rest of the batch still runs."""
    result = _run([unreadable_file, normal_clip, blurred_clip], tmp_path, per_clip=2, min_gap=1.0, jobs=1)

    assert len(result.results["clips"]) == 2
    assert len(result.results["skipped"]) == 1
    assert result.results["skipped"][0]["path"] == unreadable_file
    assert result.results["summary"]["files_given"] == 3
    assert result.results["summary"]["files_processed"] == 2
    assert result.results["summary"]["files_skipped"] == 1
    assert any(os.path.basename(unreadable_file) in m for m in result.messages)

    saved = json.load(open(os.path.join(result.out_dir, report.RESULTS_JSON), encoding="utf-8"))
    assert saved["skipped"][0]["path"] == unreadable_file


def test_percentile_ranks_survive_a_flattening_transform():
    """The reason a log clip is not punished: ranks are invariant to a flat curve."""
    graded = np.array([12.0, 40.0, 55.0, 71.0, 200.0])
    flattened = 0.18 * graded + 30.0
    assert features.percentile_ranks(graded) == features.percentile_ranks(flattened)


@requires_ffmpeg
def test_log_clip_is_not_penalised(flat_clip, normal_clip, tmp_path):
    """A flattened copy must not collapse to near-zero landscape scores."""
    clip_info = probe(flat_clip)
    verdict = detect_log(clip_info, "auto")
    assert verdict.is_log and verdict.source == "filename"
    assert verdict.is_a_guess is True

    flat = _run([flat_clip], tmp_path / "flat", per_clip=3, min_gap=1.0).results["clips"][0]
    normal = _run([normal_clip], tmp_path / "normal", per_clip=3, min_gap=1.0).results["clips"][0]

    assert flat["color"]["mode"] == "normalise"
    assert S.normalisation_applied() in flat["notes"]

    raw_saturation = max(f["features"]["saturation_mean"] for f in flat["frames"])
    assert raw_saturation > 0.0

    flat_scores = [f["score"] for f in flat["frames"]]
    normal_scores = [f["score"] for f in normal["frames"]]
    assert flat_scores and normal_scores
    assert max(flat_scores) > 0.4, flat_scores
    # Within-clip ranking is what is compared, so the flattened copy must land
    # in the same neighbourhood as the graded one, not an order of magnitude below.
    assert max(flat_scores) > 0.6 * max(normal_scores), (flat_scores, normal_scores)

    landscape = [
        scoring.landscape_component(f["features"]["colorfulness_rank"], f["features"]["dynamic_range_rank"])
        for f in flat["frames"]
    ]
    assert max(landscape) > 0.4, landscape


@requires_ffmpeg
def test_sampling_count_is_measured_not_assumed(normal_clip, tmp_path):
    result = _run([normal_clip], tmp_path, per_clip=2, fps=2.0, min_gap=1.0)
    decode_report = result.results["clips"][0]["decode"]
    assert decode_report["frames_expected"] > 0
    assert decode_report["frames_yielded"] > 0
    assert abs(decode_report["frames_yielded"] - decode_report["frames_expected"]) <= 2
    assert decode_report["path_used"] in ("hw", "cpu")


@requires_ffmpeg
def test_report_and_json_are_written(normal_clip, tmp_path):
    result = _run([normal_clip], tmp_path, per_clip=2, min_gap=1.0)
    assert os.path.isfile(os.path.join(result.out_dir, report.RESULTS_JSON))
    html_path = os.path.join(result.out_dir, report.REPORT_HTML)
    assert os.path.isfile(html_path)
    html = open(html_path, encoding="utf-8").read()
    assert "data:image/jpeg;base64," in html, "the report must be self-contained"
    assert S.REPORT_WEIGHTS_NOTE in html


@requires_ffmpeg
def test_cancellation_deletes_partial_output(normal_clip, tmp_path):
    import threading

    cancel = threading.Event()
    cancel.set()
    out_dir = str(tmp_path / "cancelled")
    result = run_batch(Options(paths=[normal_clip], out_dir=out_dir, per_clip=3), cancel=cancel)
    assert result.cancelled is True
    leftovers = [n for n in os.listdir(out_dir)] if os.path.isdir(out_dir) else []
    assert leftovers == [], leftovers


@requires_ffmpeg
def test_cancelling_mid_batch_removes_the_frames_already_written(normal_clip, blurred_clip, tmp_path):
    """Rule 9.3: no half-finished output survives a cancel."""
    import threading

    cancel = threading.Event()
    out_dir = str(tmp_path / "midway")
    written: list[list[str]] = []

    def on_message(text: str) -> None:
        # Cancel as soon as the first clip has finished exporting, so there
        # really are JPEGs on disk when the cleanup runs.
        if text.startswith(S.clip_done("", 0, 0, 0.0).split(":")[0]) and not cancel.is_set():
            written.append(sorted(os.listdir(out_dir)))
            cancel.set()

    result = run_batch(
        Options(paths=[normal_clip, blurred_clip], out_dir=out_dir, per_clip=2, min_gap=1.0, jobs=1),
        on_message=on_message,
        cancel=cancel,
    )

    assert result.cancelled is True
    assert written and any(name.endswith(".jpg") for name in written[0]), written
    assert sorted(os.listdir(out_dir)) == []


def test_no_input_files_is_reported(tmp_path):
    result = run_batch(Options(paths=[], out_dir=str(tmp_path / "empty")))
    assert result.results == {}
    assert S.no_input_files() in result.messages
