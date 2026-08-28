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
from framepicker.select import MODE_COUNT, MODE_THRESHOLD

from conftest import requires_ffmpeg

MISSING_MODEL = os.path.join(os.sep, "definitely", "not", "a", "yunet", "model.onnx")


def _run(paths, tmp_path, **kwargs):
    options = Options(paths=list(paths), out_dir=str(tmp_path / "out"), **kwargs)
    return run_batch(options)


@requires_ffmpeg
def test_missing_detector_is_none_not_zero(normal_clip, tmp_path):
    """No YuNet model: face features stay None, the run continues, the report says so."""
    result = _run([normal_clip], tmp_path, face_model=MISSING_MODEL, per_clip=3, min_gap=1.0,
                  select_mode=MODE_COUNT)

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
    result = _run([short_clip], tmp_path, per_clip=6, min_gap=2.0, select_mode=MODE_COUNT)

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
    result = _run([unreadable_file, normal_clip, blurred_clip], tmp_path, per_clip=2, min_gap=1.0,
                  jobs=1, select_mode=MODE_COUNT)

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

    flat = _run([flat_clip], tmp_path / "flat", per_clip=3, min_gap=1.0,
                select_mode=MODE_COUNT).results["clips"][0]
    normal = _run([normal_clip], tmp_path / "normal", per_clip=3, min_gap=1.0,
                  select_mode=MODE_COUNT).results["clips"][0]

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
    result = _run([normal_clip], tmp_path, per_clip=2, fps=2.0, min_gap=1.0, select_mode=MODE_COUNT)
    decode_report = result.results["clips"][0]["decode"]
    assert decode_report["frames_expected"] > 0
    assert decode_report["frames_yielded"] > 0
    assert abs(decode_report["frames_yielded"] - decode_report["frames_expected"]) <= 2
    assert decode_report["path_used"] in ("hw", "cpu")


@requires_ffmpeg
def test_report_and_json_are_written(normal_clip, tmp_path):
    result = _run([normal_clip], tmp_path, per_clip=2, min_gap=1.0, select_mode=MODE_COUNT)
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
    result = run_batch(
        Options(paths=[normal_clip], out_dir=out_dir, per_clip=3, select_mode=MODE_COUNT), cancel=cancel
    )
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
        Options(paths=[normal_clip, blurred_clip], out_dir=out_dir, per_clip=2, min_gap=1.0,
                jobs=1, select_mode=MODE_COUNT),
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


# --------------------------------------------------------------------------
# Threshold selection (the default mode)
# --------------------------------------------------------------------------


def test_threshold_mode_is_the_default():
    options = Options()
    assert options.select_mode == MODE_THRESHOLD
    assert options.global_top > 0, "the batch page is on by default"
    assert options.export_height == 0, "exports keep the source resolution by default"


@requires_ffmpeg
def test_threshold_mode_returns_a_count_that_follows_the_footage(normal_clip, tmp_path):
    """No fixed target: a low bar returns frames, a high bar returns none."""
    generous = _run([normal_clip], tmp_path / "low", min_score=0.0, min_gap=0.5,
                    max_per_clip=0).results["clips"][0]
    strict = _run([normal_clip], tmp_path / "high", min_score=0.99, min_gap=0.5).results["clips"][0]

    assert generous["selection"]["mode"] == MODE_THRESHOLD
    assert len(generous["frames"]) > len(strict["frames"])
    assert strict["frames"] == []
    assert generous["selection"]["requested"] is None, "threshold mode promises no count"
    assert generous["selection"]["shortfall"] == 0


@requires_ffmpeg
def test_threshold_mode_says_so_when_nothing_passes(normal_clip, tmp_path):
    """Zero frames is a result that has to be explained, not an empty folder."""
    result = _run([normal_clip], tmp_path, min_score=0.99, min_gap=0.5)
    clip = result.results["clips"][0]

    assert clip["frames"] == []
    assert clip["selection"]["passed_threshold"] == 0
    assert clip["selection"]["best_score"] is not None
    expected = S.threshold_none_passed(clip["selection"]["best_score"], 0.99)
    assert expected in clip["notes"]
    assert expected in result.messages

    html = open(os.path.join(result.out_dir, report.REPORT_HTML), encoding="utf-8").read()
    assert S.REPORT_NO_FRAMES in html


@requires_ffmpeg
def test_threshold_mode_is_bounded(normal_clip, tmp_path):
    """A long clip cannot quietly export a hundred stills."""
    clip = _run([normal_clip], tmp_path, min_score=0.0, min_gap=0.5, max_per_clip=2).results["clips"][0]
    assert len(clip["frames"]) <= 2
    if clip["selection"]["capped"]:
        assert S.threshold_capped(len(clip["frames"])) in clip["notes"]


@requires_ffmpeg
def test_threshold_is_declared_uncalibrated(normal_clip, tmp_path):
    result = _run([normal_clip], tmp_path, min_score=0.2, min_gap=0.5)
    note = S.selection_mode_threshold(0.2)
    assert note in result.results["clips"][0]["notes"]
    assert note in result.messages


@requires_ffmpeg
def test_global_batch_page_is_produced_by_default(normal_clip, blurred_clip, tmp_path):
    result = _run([normal_clip, blurred_clip], tmp_path, min_score=0.0, min_gap=1.0, jobs=1)
    assert result.results["global_top"], "the batch page is on by default"
    html = open(os.path.join(result.out_dir, report.REPORT_HTML), encoding="utf-8").read()
    assert S.REPORT_GLOBAL in html
    assert S.REPORT_GLOBAL_NOTE in html, "the cross-clip page must be labelled less reliable"


@requires_ffmpeg
def test_global_batch_page_can_be_turned_off(normal_clip, tmp_path):
    result = _run([normal_clip], tmp_path, min_score=0.0, min_gap=1.0, global_top=0)
    assert "global_top" not in result.results
    html = open(os.path.join(result.out_dir, report.REPORT_HTML), encoding="utf-8").read()
    assert S.REPORT_GLOBAL not in html


@requires_ffmpeg
def test_export_height_is_recorded_and_native_by_default(normal_clip, tmp_path):
    native = _run([normal_clip], tmp_path / "native", min_score=0.0, min_gap=1.0, max_per_clip=1)
    scaled = _run([normal_clip], tmp_path / "scaled", min_score=0.0, min_gap=1.0, max_per_clip=1,
                  export_height=120)
    assert S.export_resolution_native() in native.results["clips"][0]["notes"]
    assert S.export_resolution_scaled(120) in scaled.results["clips"][0]["notes"]


# --------------------------------------------------------------------------
# Input expansion
# --------------------------------------------------------------------------


def test_expand_inputs_takes_a_folder(tmp_path):
    from framepicker.cli import expand_inputs

    folder = tmp_path / "clips"
    folder.mkdir()
    (folder / "a.MP4").write_bytes(b"x")
    (folder / "b.mov").write_bytes(b"x")
    (folder / "notes.txt").write_bytes(b"x")

    files, unmatched = expand_inputs([str(folder)])
    assert [os.path.basename(f) for f in files] == ["a.MP4", "b.mov"]
    assert unmatched == []


def test_expand_inputs_globs_because_the_shell_did_not(tmp_path):
    """cmd and PowerShell hand `*.MP4` to Python verbatim."""
    from framepicker.cli import expand_inputs

    (tmp_path / "one.MP4").write_bytes(b"x")
    (tmp_path / "two.MP4").write_bytes(b"x")
    (tmp_path / "three.mov").write_bytes(b"x")

    files, unmatched = expand_inputs([str(tmp_path / "*.MP4")])
    assert sorted(os.path.basename(f) for f in files) == ["one.MP4", "two.MP4"]
    assert unmatched == []


def test_expand_inputs_deduplicates_and_names_what_it_could_not_find(tmp_path):
    from framepicker.cli import expand_inputs

    clip = tmp_path / "a.mp4"
    clip.write_bytes(b"x")
    files, unmatched = expand_inputs([str(clip), str(clip), str(tmp_path / "ghost.mp4")])
    assert len(files) == 1
    assert unmatched == [str(tmp_path / "ghost.mp4")]


@requires_ffmpeg
def test_a_folder_of_clips_is_processed(normal_clip, blurred_clip, tmp_path):
    folder = os.path.dirname(normal_clip)
    result = _run([folder], tmp_path, min_score=0.0, min_gap=1.0, jobs=1)
    processed = {c["probe"]["name"] for c in result.results["clips"]}
    assert os.path.basename(normal_clip) in processed
    assert os.path.basename(blurred_clip) in processed


def test_an_input_that_matches_nothing_is_reported_not_ignored(tmp_path):
    ghost = str(tmp_path / "does-not-exist.mp4")
    result = run_batch(Options(paths=[ghost], out_dir=str(tmp_path / "out")))
    assert S.input_not_found(ghost) in result.messages
    assert S.no_input_files() in result.messages


@requires_ffmpeg
def test_container_tags_are_kept_so_the_camera_can_be_identified(normal_clip):
    """A camera writes its make and model into the container, not the stream."""
    clip = probe(normal_clip)
    assert "format_tags" in clip.extra
    assert isinstance(clip.extra["format_tags"], dict)


def test_expand_inputs_orders_oldest_first(tmp_path):
    """A card copied off a camera is processed in recording order."""
    import time

    from framepicker.cli import ORDER_DATE, ORDER_NAME, ORDER_NONE, expand_inputs

    # Deliberately reverse-alphabetical against the timestamps, so a name sort
    # and a date sort cannot accidentally agree.
    for name, mtime in (("c_first.mp4", 1000), ("b_second.mp4", 2000), ("a_third.mp4", 3000)):
        path = tmp_path / name
        path.write_bytes(b"x")
        os.utime(path, (mtime, mtime))
    del time

    by_date, _ = expand_inputs([str(tmp_path)], ORDER_DATE)
    assert [os.path.basename(p) for p in by_date] == ["c_first.mp4", "b_second.mp4", "a_third.mp4"]

    by_name, _ = expand_inputs([str(tmp_path)], ORDER_NAME)
    assert [os.path.basename(p) for p in by_name] == ["a_third.mp4", "b_second.mp4", "c_first.mp4"]

    given = [str(tmp_path / "b_second.mp4"), str(tmp_path / "a_third.mp4")]
    as_given, _ = expand_inputs(given, ORDER_NONE)
    assert as_given == given


def test_date_order_is_stable_for_identical_timestamps(tmp_path):
    from framepicker.cli import ORDER_DATE, expand_inputs

    for name in ("b.mp4", "a.mp4", "c.mp4"):
        path = tmp_path / name
        path.write_bytes(b"x")
        os.utime(path, (5000, 5000))
    files, _ = expand_inputs([str(tmp_path)], ORDER_DATE)
    assert [os.path.basename(p) for p in files] == ["a.mp4", "b.mp4", "c.mp4"]


@requires_ffmpeg
def test_clips_are_reported_in_processing_order(normal_clip, blurred_clip, tmp_path):
    import os as _os

    _os.utime(blurred_clip, (1_600_000_000, 1_600_000_000))
    _os.utime(normal_clip, (1_700_000_000, 1_700_000_000))
    result = _run([blurred_clip, normal_clip], tmp_path, min_score=0.0, min_gap=1.0, jobs=1)
    names = [c["probe"]["name"] for c in result.results["clips"]]
    assert names == [os.path.basename(blurred_clip), os.path.basename(normal_clip)]
    assert result.results["options"]["order"] == "date"
