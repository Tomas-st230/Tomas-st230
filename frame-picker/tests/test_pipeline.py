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
    normalisation = flat["color"]["normalisation"]
    assert S.normalisation_applied(normalisation["strength"], normalisation["saturation_gain"]) in flat["notes"]

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
        # really are JPEGs on disk when the cleanup runs. The run writes into
        # its own dated subfolder, which is where those JPEGs are.
        if text.startswith(S.clip_done("", 0, 0, 0.0).split(":")[0]) and not cancel.is_set():
            for entry in sorted(os.listdir(out_dir)):
                folder = os.path.join(out_dir, entry)
                if os.path.isdir(folder):
                    written.append(sorted(os.listdir(folder)))
            cancel.set()

    result = run_batch(
        Options(paths=[normal_clip, blurred_clip], out_dir=out_dir, per_clip=2, min_gap=1.0,
                jobs=1, select_mode=MODE_COUNT),
        on_message=on_message,
        cancel=cancel,
    )

    assert result.cancelled is True
    assert written and any(name.endswith(".jpg") for name in written[0]), written
    # Nothing survives: neither the frames nor the folder the run created.
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


# --------------------------------------------------------------------------
# Measured flatness must not trigger a colour transform
# --------------------------------------------------------------------------


def test_measured_flatness_only_suspects_it_never_acts(tmp_path):
    """A dark sunset measures as flat as a log profile.

    On 77 real DJI clips from one card - all the same picture profile -
    luma_span ran 0.155 to 0.925 and saturation 0.074 to 0.765 in one
    continuous distribution. Four sunset clips fell under the limits, had
    their saturation multiplied by 2.5, and came out neon. So the
    measurement may report a suspicion and may not act on it.
    """
    from framepicker.decode import LOG_STATS_MAX_LUMA_SPAN, LOG_STATS_MAX_SATURATION, detect_log
    from framepicker.probe import ClipInfo

    clip = ClipInfo(
        path="/x/DJI_0001.MP4", name="DJI_0001.MP4", duration=10.0, fps=30.0,
        width=3840, height=2160, codec="hevc", pix_fmt="yuv420p10le",
        color_transfer=None, color_primaries=None, color_space=None, color_range=None,
        nb_frames=300, size_bytes=1,
    )
    flat_looking = {"luma_span": 0.385, "saturation": 0.135, "frames_measured": 12}

    verdict = detect_log(clip, "auto", flat_looking)
    assert verdict.suspected is True, "the measurement should still be reported"
    assert verdict.is_log is False, "and must not trigger a transform on its own"
    assert verdict.source == "statistics"
    assert flat_looking["luma_span"] < LOG_STATS_MAX_LUMA_SPAN
    assert flat_looking["saturation"] < LOG_STATS_MAX_SATURATION

    # Only an explicit instruction acts.
    forced = detect_log(clip, "on", flat_looking)
    assert forced.is_log is True and forced.is_a_guess is False


def test_a_filename_hint_still_acts(tmp_path):
    from framepicker.decode import detect_log
    from framepicker.probe import ClipInfo

    clip = ClipInfo(
        path="/x/DJI_0001_DLOG.MP4", name="DJI_0001_DLOG.MP4", duration=10.0, fps=30.0,
        width=3840, height=2160, codec="hevc", pix_fmt="yuv420p10le",
        color_transfer=None, color_primaries=None, color_space=None, color_range=None,
        nb_frames=300, size_bytes=1,
    )
    verdict = detect_log(clip, "auto", {"luma_span": 0.9, "saturation": 0.5, "frames_measured": 12})
    assert verdict.is_log is True and verdict.source == "filename"


def test_normalisation_saturation_gain_is_capped_low():
    """The 2.5x gain that wrecked four sunsets is gone."""
    import numpy as np

    from framepicker.decode import NORMALISE_MAX_GAIN, estimate_normalisation

    assert NORMALISE_MAX_GAIN <= 1.8, "a timid fallback, not a look"
    dull = np.full((32, 32, 3), 90, dtype=np.uint8)
    dull[:, :16, 0] = 110
    normalisation = estimate_normalisation([dull])
    assert normalisation is not None
    assert normalisation.saturation_gain <= NORMALISE_MAX_GAIN


def test_normalisation_strength_dials_it_down():
    import numpy as np

    from framepicker.decode import Normalisation

    frame = np.random.default_rng(0).integers(60, 140, (32, 32, 3), dtype=np.uint8)
    full = Normalisation(lo=60.0, hi=140.0, saturation_gain=1.6, strength=1.0).apply(frame)
    half = Normalisation(lo=60.0, hi=140.0, saturation_gain=1.6, strength=0.5).apply(frame)
    off = Normalisation(lo=60.0, hi=140.0, saturation_gain=1.6, strength=0.0).apply(frame)

    def drift(a):
        return float(np.abs(a.astype("float32") - frame.astype("float32")).mean())

    assert drift(off) == 0.0
    assert 0.0 < drift(half) < drift(full)


# --------------------------------------------------------------------------
# Its own folder per run, and the links in the page
# --------------------------------------------------------------------------


@requires_ffmpeg
def test_each_run_writes_into_its_own_folder(normal_clip, tmp_path):
    """Two runs into the same output directory must not mix.

    The 163-file run reported 12 "files nothing refers to". They were the
    previous run's stills, sitting in the same folder.
    """
    out = str(tmp_path / "out")
    first = run_batch(Options(paths=[normal_clip], out_dir=out, per_clip=2, min_gap=1.0,
                             select_mode=MODE_COUNT))
    second = run_batch(Options(paths=[normal_clip], out_dir=out, per_clip=2, min_gap=1.0,
                              select_mode=MODE_COUNT))

    assert first.out_dir != second.out_dir
    assert os.path.dirname(first.out_dir) == os.path.abspath(out)
    for result in (first, second):
        assert os.path.isfile(os.path.join(result.out_dir, report.REPORT_HTML))
        assert result.results["integrity"]["unreferenced_files"] == []
        assert result.results["output_dir"] == os.path.abspath(result.out_dir)


@requires_ffmpeg
def test_the_run_folder_can_be_turned_off(normal_clip, tmp_path):
    out = str(tmp_path / "flat-out")
    result = run_batch(Options(paths=[normal_clip], out_dir=out, per_clip=2, min_gap=1.0,
                              select_mode=MODE_COUNT, run_folder=False))
    assert os.path.abspath(result.out_dir) == os.path.abspath(out)


@requires_ffmpeg
def test_every_link_in_the_report_resolves(normal_clip, tmp_path):
    result = _run([normal_clip], tmp_path, per_clip=3, min_gap=1.0, select_mode=MODE_COUNT)
    links = result.results["integrity"]["links"]
    assert links["checked"] > 0
    assert links["broken"] == 0, links["details"]
    assert result.results["integrity"]["ok"] is True


@requires_ffmpeg
def test_a_broken_link_in_the_report_is_reported(normal_clip, tmp_path):
    """The check has to be able to fail, or it is not a check."""
    result = _run([normal_clip], tmp_path, per_clip=2, min_gap=1.0, select_mode=MODE_COUNT)
    html_path = os.path.join(result.out_dir, report.REPORT_HTML)
    with open(html_path, "a", encoding="utf-8") as handle:
        handle.write('<img src="does-not-exist.jpg">')
    links = report.check_report_links(html_path, result.out_dir)
    assert links["broken"] == 1
    assert "does-not-exist.jpg" in links["details"][0]


# --------------------------------------------------------------------------
# The .LRF proxy, end to end
# --------------------------------------------------------------------------


@requires_ffmpeg
def test_the_proxy_is_analysed_and_the_master_is_exported(tmp_path):
    """Analysis on the 720p proxy; the exported still is full resolution."""
    import cv2
    from conftest import _make_clip

    master = _make_clip(str(tmp_path / "DJI_0200_D.MP4"), 4, "scale=1280:720", fps=10)
    built = _make_clip(str(tmp_path / "proxy-source.mp4"), 4, "scale=640:360", fps=10)
    proxy = str(tmp_path / "DJI_0200_D.LRF")
    os.replace(built, proxy)

    result = run_batch(Options(paths=[master], out_dir=str(tmp_path / "out"), per_clip=2,
                              min_gap=1.0, select_mode=MODE_COUNT))
    clip = result.results["clips"][0]
    assert clip["proxy"]["usable"] is True, clip["proxy"]["detail"]
    assert clip["decode"]["proxy_file"] == "DJI_0200_D.LRF"
    assert any("DJI_0200_D.LRF" in note for note in clip["notes"])

    assert clip["frames"], "the proxy run still has to deliver frames"
    exported = os.path.join(result.out_dir, clip["frames"][0]["file"])
    image = cv2.imread(exported)
    assert image is not None
    assert image.shape[1] == 1280, "the export must come from the master, not the proxy"


@requires_ffmpeg
def test_the_proxy_can_be_turned_off(tmp_path):
    from conftest import _make_clip

    master = _make_clip(str(tmp_path / "DJI_0201_D.MP4"), 3, "scale=1280:720", fps=10)
    built = _make_clip(str(tmp_path / "p.mp4"), 3, "scale=640:360", fps=10)
    os.replace(built, str(tmp_path / "DJI_0201_D.LRF"))

    result = run_batch(Options(paths=[master], out_dir=str(tmp_path / "out"), per_clip=1,
                              min_gap=1.0, select_mode=MODE_COUNT, proxy="off"))
    clip = result.results["clips"][0]
    assert clip["proxy"] is None
    assert clip["decode"]["proxy_file"] == ""


# --------------------------------------------------------------------------
# The camera's own word about the profile
# --------------------------------------------------------------------------


CAPTION_LINE = (
    "1\n00:00:00,000 --> 00:00:00,100\n"
    "<font size=\"28\">FrameCnt: 1, DiffTime: 100ms\n2026-08-22 19:13:45.123\n"
    "[iso: 100] [shutter: 1/500.0] [color_md: {mode}] [focal_len: 24.00]</font>\n\n"
)


@requires_ffmpeg
def test_a_normal_colour_clip_keeps_the_lut_off(tmp_path, monkeypatch):
    """The mixed dump: the sidecar says Normal, so the LUT must not be applied."""
    from conftest import _make_clip

    clip_path = _make_clip(str(tmp_path / "DJI_0300_D.MP4"), 3, fps=10)
    (tmp_path / "DJI_0300_D.SRT").write_text(CAPTION_LINE.format(mode="default"), encoding="utf-8")
    lut = tmp_path / "identity.cube"
    lut.write_text("LUT_3D_SIZE 2\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n",
                   encoding="utf-8")

    result = run_batch(Options(paths=[clip_path], out_dir=str(tmp_path / "out"), per_clip=1,
                              min_gap=1.0, select_mode=MODE_COUNT, lut=str(lut)))
    clip = result.results["clips"][0]
    assert clip["log"]["source"] == "sidecar"
    assert clip["log"]["is_log"] is False
    assert clip["log"]["is_a_guess"] is False
    assert clip["color"]["mode"] == "none"
    assert clip["color"]["lut"] is None


@requires_ffmpeg
def test_a_dlog_sidecar_turns_the_lut_on(tmp_path):
    from conftest import _make_clip

    clip_path = _make_clip(str(tmp_path / "DJI_0301_D.MP4"), 3, fps=10)
    (tmp_path / "DJI_0301_D.SRT").write_text(CAPTION_LINE.format(mode="dlog_m"), encoding="utf-8")
    lut = tmp_path / "identity.cube"
    lut.write_text("LUT_3D_SIZE 2\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n",
                   encoding="utf-8")

    result = run_batch(Options(paths=[clip_path], out_dir=str(tmp_path / "out"), per_clip=1,
                              min_gap=1.0, select_mode=MODE_COUNT, lut=str(lut)))
    clip = result.results["clips"][0]
    assert clip["log"]["is_log"] is True
    assert clip["log"]["profile"] == "dlog"
    assert clip["color"]["mode"] == "lut"


# --------------------------------------------------------------------------
# --look auto, end to end
# --------------------------------------------------------------------------


@requires_ffmpeg
def test_auto_look_decides_per_clip_and_records_the_evidence(normal_clip, tmp_path):
    from framepicker import grading

    result = _run([normal_clip], tmp_path, per_clip=2, min_gap=1.0, select_mode=MODE_COUNT,
                  look=grading.AUTO)
    clip = result.results["clips"][0]
    look = clip["look"]
    assert look["requested"] == grading.AUTO
    assert look["name"] in grading.available()
    assert look["name"] != grading.AUTO, "auto has to resolve to a real answer"
    assert look["auto"] is not None
    assert look["auto"]["frames_measured"] > 0
    assert set(look["auto"]["evidence"]) >= {"vegetation", "sky", "warm", "grey", "vertical_share"}


# --------------------------------------------------------------------------
# The run's own log
# --------------------------------------------------------------------------


@requires_ffmpeg
def test_a_run_writes_both_logs_and_the_report_links_to_them(normal_clip, tmp_path):
    from framepicker import runlog

    result = _run([normal_clip], tmp_path, per_clip=2, min_gap=1.0, select_mode=MODE_COUNT)
    txt = os.path.join(result.out_dir, runlog.LOG_TXT)
    jsonl = os.path.join(result.out_dir, runlog.LOG_JSONL)
    assert result.log_paths == [txt, jsonl]
    assert os.path.getsize(txt) > 0 and os.path.getsize(jsonl) > 0

    events = [json.loads(line) for line in open(jsonl, encoding="utf-8") if line.strip()]
    kinds = [event["event"] for event in events]
    assert kinds[0] == "run_started" and kinds[-1] == "run_finished"
    clip_events = [e for e in events if e["event"] == "clip"]
    frame_events = [e for e in events if e["event"] == "frame"]
    assert len(clip_events) == 1
    assert clip_events[0]["candidates"] > 0
    assert frame_events and frame_events[0]["components"]

    # The page links to them, and nothing calls them leftovers.
    html = open(os.path.join(result.out_dir, report.REPORT_HTML), encoding="utf-8").read()
    assert f'href="{runlog.LOG_TXT}"' in html and f'href="{runlog.LOG_JSONL}"' in html
    integrity = result.results["integrity"]
    assert integrity["unreferenced_files"] == []
    assert integrity["side_files"][runlog.LOG_JSONL] > 0
    assert integrity["links"]["broken"] == 0


@requires_ffmpeg
def test_logging_can_be_turned_off_for_a_run(normal_clip, tmp_path):
    from framepicker import runlog

    result = _run([normal_clip], tmp_path, per_clip=1, min_gap=1.0, select_mode=MODE_COUNT,
                  write_log=False)
    assert result.log_paths == []
    assert not os.path.isfile(os.path.join(result.out_dir, runlog.LOG_TXT))
    assert result.results["integrity"]["links"]["broken"] == 0


@requires_ffmpeg
def test_the_summary_the_window_gets_carries_the_measured_values(normal_clip, tmp_path):
    """The table, the log file and the console must agree, so they share one dict."""
    seen = []
    options = Options(paths=[normal_clip], out_dir=str(tmp_path / "out"), per_clip=1,
                      min_gap=1.0, select_mode=MODE_COUNT)
    run_batch(options, on_clip_done=seen.append)
    assert len(seen) == 1
    values = seen[0]["values"]
    for key in ("file", "candidates", "confidence_spread", "look_applied", "elapsed_s",
                "decode_path", "frames_delivered"):
        assert key in values, key


@requires_ffmpeg
def test_the_side_file_sizes_are_measured_after_everything_is_written(normal_clip, tmp_path):
    """results.json is written after the check runs, so its size is taken again."""
    from framepicker import runlog

    result = _run([normal_clip], tmp_path, per_clip=1, min_gap=1.0, select_mode=MODE_COUNT)
    side = result.results["integrity"]["side_files"]
    for name in (report.RESULTS_JSON, runlog.LOG_TXT, runlog.LOG_JSONL):
        assert side[name] and side[name] > 0, (name, side)
    on_disk = json.load(open(os.path.join(result.out_dir, report.RESULTS_JSON), encoding="utf-8"))
    assert on_disk["integrity"]["side_files"][report.RESULTS_JSON] > 0
