"""Command line entry point and the batch pipeline itself.

``python -m framepicker VIDEO...``

The GUI imports :func:`run_batch` from here and calls it with the same
options object argparse builds. There is no second code path.
"""

from __future__ import annotations

import argparse
import glob
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Sequence

from . import decode, export, features, proc, report, scoring
from . import strings_lt as S
from .probe import ProbeError, probe
from .select import (
    DEFAULT_MAX_PER_CLIP,
    DEFAULT_MIN_SCORE,
    MODE_COUNT,
    MODE_THRESHOLD,
    Candidate,
    select,
)

VERSION = "0.1.0"

DEFAULT_OUT_DIR = "frame-picker-out"
DEFAULT_PER_CLIP = 6
#: Frames of the whole batch on the "best of the batch" page. On by default;
#: the page is labelled less reliable than the per-clip ranking, because it
#: compares raw values across different cameras and picture profiles.
DEFAULT_GLOBAL_TOP = 20
DEFAULT_FPS = 2.0
DEFAULT_MIN_GAP = 2.0
#: Upper bound on buffered analysis frames per clip. Above this the sampling
#: rate is lowered and the report says so, instead of the tool quietly eating
#: all the memory on a long file.
DEFAULT_MAX_CANDIDATES = 3000
#: Extensions treated as video when a folder or a wildcard is given. Lives
#: here, not in the GUI: expanding an input into a file list is logic.
VIDEO_SUFFIXES = (
    ".mp4", ".mov", ".mxf", ".mkv", ".avi", ".m4v", ".mts", ".m2ts", ".insv", ".webm",
)

#: A frame this uniformly black or white is thrown away before any real work.
CHEAP_REJECT_CLIP_FRACTION = 0.90
#: Sharpness percentile below which a frame is rejected as soft, for this clip.
CHEAP_REJECT_SHARPNESS_QUANTILE = 0.10


@dataclass
class Options:
    paths: list[str] = field(default_factory=list)
    out_dir: str = DEFAULT_OUT_DIR
    per_clip: int = DEFAULT_PER_CLIP
    fps: float = DEFAULT_FPS
    min_gap: float = DEFAULT_MIN_GAP
    convert_log: str = "auto"
    lut: str | None = None
    lut_all: bool = False
    jobs: int = 0
    no_faces: bool = False
    face_model: str | None = None
    image_format: str = "jpg"
    jpeg_quality: int = 2
    global_top: int = DEFAULT_GLOBAL_TOP
    max_candidates: int = DEFAULT_MAX_CANDIDATES
    hwaccel: str = "auto"
    select_mode: str = MODE_THRESHOLD
    min_score: float = DEFAULT_MIN_SCORE
    max_per_clip: int = DEFAULT_MAX_PER_CLIP
    export_height: int = 0

    def as_dict(self) -> dict:
        return {
            "out_dir": self.out_dir,
            "per_clip": self.per_clip,
            "fps": self.fps,
            "min_gap": self.min_gap,
            "convert_log": self.convert_log,
            "lut": self.lut,
            "lut_all": self.lut_all,
            "jobs": self.jobs,
            "no_faces": self.no_faces,
            "image_format": self.image_format,
            "jpeg_quality": self.jpeg_quality,
            "global_top": self.global_top,
            "max_candidates": self.max_candidates,
            "hwaccel": self.hwaccel,
            "select_mode": self.select_mode,
            "min_score": self.min_score,
            "max_per_clip": self.max_per_clip,
            "export_height": self.export_height,
        }


class Messenger:
    """Thread-safe message sink that emits some messages only once per run."""

    def __init__(self, sink: Callable[[str], None] | None = None) -> None:
        self._sink = sink or (lambda text: print(text, flush=True))
        self._lock = threading.Lock()
        self._seen: set[str] = set()
        self.log: list[str] = []
        self.notes: list[str] = []

    def say(self, text: str) -> None:
        with self._lock:
            self.log.append(text)
            self._sink(text)

    def once(self, key: str, text: str) -> bool:
        with self._lock:
            if key in self._seen:
                return False
            self._seen.add(key)
            self.log.append(text)
            self.notes.append(text)
        self._sink(text)
        return True


@dataclass
class BatchResult:
    results: dict
    out_dir: str
    json_path: str | None = None
    html_path: str | None = None
    cancelled: bool = False
    messages: list[str] = field(default_factory=list)


class _Created:
    """Every file this run wrote, so cancellation can delete exactly those."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.paths: list[str] = []

    def add(self, path: str) -> None:
        with self._lock:
            self.paths.append(path)

    def remove_all(self) -> None:
        with self._lock:
            paths = list(self.paths)
            self.paths.clear()
        for path in paths:
            try:
                os.remove(path)
            except OSError:
                pass


# --------------------------------------------------------------------------
# One clip
# --------------------------------------------------------------------------


def _encode_buffer(frame):
    import cv2
    import numpy as np

    bgr = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return buf.tobytes() if ok else None


def _decode_buffer(blob: bytes):
    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def analyse_clip(
    path: str,
    options: Options,
    messenger: Messenger,
    created: _Created,
    cancel: threading.Event | None = None,
) -> dict:
    """Full pipeline for one file. Never raises for content reasons."""
    import numpy as np

    started = time.perf_counter()
    clip = probe(path)
    record: dict = {
        "probe": clip.as_dict(),
        "notes": [],
        "frames": [],
        "candidates_evaluated": 0,
        "rejects": {"total": 0, "dark": 0, "bright": 0, "blurry": 0},
    }

    # -- colour ------------------------------------------------------------
    # Sample a handful of frames first: the flatness measurement is the only
    # log signal left when the camera tags nothing and the filename says
    # nothing, which is the normal case for DJI files.
    samples: list = []
    stats = None
    if options.convert_log != "off":
        samples = decode.sample_for_normalisation(clip)
        stats = decode.flatness(samples)

    verdict = decode.detect_log(clip, options.convert_log, stats)
    record["log"] = verdict.as_dict()
    log_line = (
        S.log_detected(_log_source_text(verdict.source)) if verdict.is_log
        else S.log_not_detected(_log_source_text(verdict.source))
    )
    record["notes"].append(log_line)
    messenger.say(log_line)
    if stats is not None:
        measured = S.log_statistics(stats["luma_span"], stats["saturation"])
        record["notes"].append(measured)
        messenger.say(measured)
        thresholds = S.log_thresholds(
            decode.LOG_STATS_MAX_LUMA_SPAN, decode.LOG_STATS_MAX_SATURATION
        )
        record["notes"].append(thresholds)
        messenger.once("log-thresholds", thresholds)
    if verdict.is_a_guess:
        record["notes"].append(S.log_is_a_guess())
        messenger.once("log-guess", S.log_is_a_guess())

    lut_path = None
    normalisation = None
    if options.lut:
        readable, why = decode.cube_is_readable(options.lut)
        if not readable:
            _note(record, messenger, S.lut_unreadable(options.lut, why))
        elif verdict.is_log or options.lut_all:
            # A LUT is a log-to-display conversion. Putting one on footage
            # that is already Rec.709 wrecks it, and a mixed batch of D-Log
            # and normal clips is the normal case - so the LUT follows the
            # log verdict unless the user forces it onto everything.
            lut_path = options.lut
            _note(record, messenger, S.lut_applied(options.lut))
            if options.lut_all and not verdict.is_log:
                _note(record, messenger, S.lut_forced_on_all())
        else:
            _note(record, messenger, S.lut_skipped_not_log())
    if lut_path is None and verdict.is_log:
        normalisation = decode.estimate_normalisation(samples) if samples else None
        if normalisation is not None:
            _note(record, messenger, S.normalisation_applied())
    if lut_path is None and normalisation is None:
        record["notes"].append(S.no_conversion_applied())
    record["color"] = {
        "mode": "lut" if lut_path else ("normalise" if normalisation else "none"),
        "lut": lut_path,
        "normalisation": normalisation.as_dict() if normalisation else None,
    }

    # -- detectors (one instance per clip; YuNet is not thread safe) --------
    face_detector, face_status = features.load_face_detector(options.face_model, enabled=not options.no_faces)
    if face_detector is None:
        detail = S.FACES_DISABLED_BY_FLAG if options.no_faces else (
            S.FACE_MODEL_MISSING if face_status.detail == "model-missing" else face_status.detail
        )
        messenger.once("faces", S.faces_unavailable(detail))
        messenger.once("faces-mode", S.STATISTICS_ONLY_MODE)
        record["notes"].append(S.faces_unavailable(detail))
        record["notes"].append(S.STATISTICS_ONLY_MODE)
    saliency, saliency_status = features.load_saliency()
    if saliency is None:
        detail = S.SALIENCY_MISSING if "module" in saliency_status.detail.lower() else saliency_status.detail
        messenger.once("saliency", S.saliency_unavailable(detail))
        record["notes"].append(S.saliency_unavailable(detail))
    record["detectors"] = {
        "faces": {"available": face_detector is not None, "detail": face_status.detail},
        "saliency": {"available": saliency is not None, "detail": saliency_status.detail},
        "statistics_only": face_detector is None,
    }

    # -- sample ------------------------------------------------------------
    decode_report = decode.DecodeReport()
    buffers: list[bytes] = []
    timestamps: list[float] = []
    cheap: list[dict] = []
    for timestamp, frame in decode.sample_frames(
        clip,
        options.fps,
        lut_path=lut_path,
        hwaccel=options.hwaccel,
        max_candidates=options.max_candidates,
        cancel=cancel,
        report=decode_report,
    ):
        if cancel is not None and cancel.is_set():
            break
        work = normalisation.apply(frame) if normalisation is not None else frame
        blob = _encode_buffer(work)
        if blob is None:
            continue
        buffers.append(blob)
        timestamps.append(timestamp)
        cheap.append({
            "sharpness": features.sharpness(work),
            "exposure_clip_low": features.exposure_clip_low(work),
            "exposure_clip_high": features.exposure_clip_high(work),
            "dynamic_range": features.dynamic_range(work),
            "colorfulness": features.colorfulness(work),
            "saturation_mean": features.saturation_mean(work),
        })
    record["decode"] = decode_report.as_dict()
    messenger.say(S.decode_path_hw() if decode_report.path_used == "hw"
                  else S.decode_path_cpu(decode_report.hw_error))
    messenger.say(S.sampled_frames(len(cheap), decode_report.frames_expected, decode_report.effective_fps))
    if decode_report.frames_expected and len(cheap) < decode_report.frames_expected * 0.9:
        messenger.say(S.sampling_shortfall(len(cheap), decode_report.frames_expected))
        record["notes"].append(S.sampling_shortfall(len(cheap), decode_report.frames_expected))

    if cancel is not None and cancel.is_set():
        record["elapsed_s"] = time.perf_counter() - started
        return record

    # -- percentile ranks over the whole clip, then the cheap reject pass ---
    scoring.attach_ranks(cheap)
    if cheap:
        sharp_values = np.asarray([c["sharpness"] for c in cheap], dtype=np.float64)
        blur_cut = float(np.quantile(sharp_values, CHEAP_REJECT_SHARPNESS_QUANTILE))
    else:
        blur_cut = 0.0

    survivors: list[int] = []
    for i, item in enumerate(cheap):
        if item["exposure_clip_low"] >= CHEAP_REJECT_CLIP_FRACTION:
            record["rejects"]["dark"] += 1
        elif item["exposure_clip_high"] >= CHEAP_REJECT_CLIP_FRACTION:
            record["rejects"]["bright"] += 1
        elif len(cheap) >= 10 and item["sharpness"] < blur_cut:
            record["rejects"]["blurry"] += 1
        else:
            survivors.append(i)
    record["rejects"]["total"] = len(cheap) - len(survivors)
    messenger.say(S.rejected_frames(
        record["rejects"]["total"], record["rejects"]["dark"],
        record["rejects"]["bright"], record["rejects"]["blurry"],
    ))

    # -- expensive features, scoring ---------------------------------------
    candidates: list[Candidate] = []
    for i in survivors:
        if cancel is not None and cancel.is_set():
            break
        frame = _decode_buffer(buffers[i])
        if frame is None:
            continue
        item = dict(cheap[i])
        if face_detector is not None:
            count, max_rel = face_detector.detect(frame)
            item["face_count"] = count
            item["face_max_rel"] = max_rel
        else:
            item["face_count"] = None
            item["face_max_rel"] = None
        item["subject_rel"] = item["subject_cx"] = item["subject_cy"] = item["thirds_distance"] = None
        if saliency is not None:
            subject = saliency.subject(frame)
            if subject is not None:
                rel, cx, cy = subject
                item["subject_rel"], item["subject_cx"], item["subject_cy"] = rel, cx, cy
                item["thirds_distance"] = features.thirds_distance(cx, cy)
        score, reasons = scoring.score_frame_explained(item)
        candidates.append(Candidate(
            index=i,
            t=timestamps[i],
            score=score,
            dhash=features.dhash(frame),
            histogram=features.color_histogram(frame),
            features=item,
            reasons=reasons,
        ))

    record["candidates_evaluated"] = len(candidates)
    record["confidence"] = scoring.confidence([c.score for c in candidates])
    messenger.say(record["confidence"]["message"])

    # -- select ------------------------------------------------------------
    selection = select(
        candidates,
        options.per_clip,
        mode=options.select_mode,
        min_score=options.min_score,
        max_per_clip=options.max_per_clip,
        min_gap=options.min_gap,
        clip_duration=clip.duration,
    )
    record["selection"] = selection.as_dict()
    for note in selection.notes:
        messenger.say(note)
    record["notes"].extend(selection.notes)
    if selection.shortfall:
        messenger.say(S.shortfall_header(len(selection.selected), options.per_clip))
        for reason in selection.reasons:
            messenger.say(reason)

    # -- export ------------------------------------------------------------
    if cancel is None or not cancel.is_set():
        results, errors = export.export_selection(
            clip.path,
            clip.name,
            selection.selected,
            options.out_dir,
            lut_path=lut_path,
            normalisation=normalisation,
            quality=options.jpeg_quality,
            image_format=options.image_format,
            height=options.export_height,
            cancel=cancel,
        )
        record["notes"].append(
            S.export_resolution_scaled(options.export_height) if options.export_height
            else S.export_resolution_native()
        )
        for text in errors:
            messenger.say(text)
        for rank, (candidate, result) in enumerate(zip(selection.selected, results), start=1):
            # Track failed exports too: a cancelled ffmpeg can leave a stub
            # behind, and cleanup has to remove that as well.
            created.add(result.path)
            record["frames"].append({
                "rank": rank,
                "timestamp": candidate.t,
                "score": candidate.score,
                "file": os.path.basename(result.path) if result.ok else None,
                "exported": result.ok,
                "export_error": result.detail or None,
                "features": candidate.features,
                "reasons": candidate.reasons,
            })

    record["elapsed_s"] = time.perf_counter() - started
    if options.select_mode == MODE_THRESHOLD:
        messenger.say(S.clip_done_threshold(clip.name, len(record["frames"]), record["elapsed_s"]))
    else:
        messenger.say(S.clip_done(clip.name, len(record["frames"]), options.per_clip, record["elapsed_s"]))
    return record


def _note(record: dict, messenger: Messenger, text: str) -> None:
    """A decision the user has to be able to see: report it and print it."""
    record["notes"].append(text)
    messenger.say(text)


def _log_source_text(source: str) -> str:
    return {
        "flag": S.LOG_SOURCE_FLAG,
        "metadata": S.LOG_SOURCE_METADATA,
        "filename": S.LOG_SOURCE_FILENAME,
        "statistics": S.LOG_SOURCE_STATISTICS,
        "default": S.LOG_SOURCE_DEFAULT,
    }.get(source, source)


# --------------------------------------------------------------------------
# Batch
# --------------------------------------------------------------------------


def expand_inputs(paths: Sequence[str]) -> tuple[list[str], list[str]]:
    """Turn CLI arguments into a concrete file list.

    A folder becomes every video file in it, and a wildcard is expanded here
    rather than by the shell - ``cmd`` and PowerShell do not glob arguments
    for a Python program, so ``*.MP4`` would otherwise arrive verbatim and
    match nothing.

    Returns ``(files, unmatched)``; nothing is dropped silently.
    """
    files: list[str] = []
    unmatched: list[str] = []
    for raw in paths:
        if os.path.isdir(raw):
            found = [
                os.path.join(raw, name)
                for name in sorted(os.listdir(raw))
                if name.lower().endswith(VIDEO_SUFFIXES) and os.path.isfile(os.path.join(raw, name))
            ]
        elif any(char in raw for char in "*?["):
            found = sorted(match for match in glob.glob(raw) if os.path.isfile(match))
        elif os.path.isfile(raw):
            found = [raw]
        else:
            found = []
        if found:
            files += found
        else:
            unmatched.append(raw)

    seen: set[str] = set()
    unique: list[str] = []
    for path in files:
        key = os.path.normcase(os.path.abspath(path))
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique, unmatched


def clip_summary(index: int, path: str, record: dict | None, reason: str = "") -> dict:
    """Flat, display-ready summary of one file. Built here so the GUI can
    render a table without knowing anything about the pipeline."""
    if record is None:
        return {
            "index": index, "path": path, "name": os.path.basename(path), "ok": False,
            "is_log": None, "log_source": "", "color_mode": "", "decode_path": "",
            "frames": 0, "elapsed_s": 0.0, "reason": reason,
        }
    log = record.get("log", {})
    return {
        "index": index,
        "path": path,
        "name": record.get("probe", {}).get("name") or os.path.basename(path),
        "ok": True,
        "is_log": log.get("is_log"),
        "log_source": log.get("source", ""),
        "color_mode": record.get("color", {}).get("mode", ""),
        "decode_path": record.get("decode", {}).get("path_used", ""),
        "frames": len(record.get("frames", [])),
        "elapsed_s": float(record.get("elapsed_s") or 0.0),
        "reason": "",
    }


def run_batch(
    options: Options,
    on_message: Callable[[str], None] | None = None,
    cancel: threading.Event | None = None,
    on_clip_done: Callable[[dict], None] | None = None,
) -> BatchResult:
    """Process every file in ``options.paths``. One bad file never aborts it."""
    messenger = Messenger(on_message)
    created = _Created()
    started = time.perf_counter()

    for tool in ("ffmpeg", "ffprobe"):
        if proc.executable(tool) is None:
            messenger.say(S.ffmpeg_missing(tool))
            return BatchResult({}, options.out_dir, messages=messenger.log)

    paths, unmatched = expand_inputs(options.paths)
    for missing in unmatched:
        messenger.say(S.input_not_found(missing))
    if len(paths) != len(options.paths):
        messenger.say(S.inputs_expanded(len(options.paths), len(paths)))
    if not paths:
        messenger.say(S.no_input_files())
        return BatchResult({}, options.out_dir, messages=messenger.log)

    os.makedirs(options.out_dir, exist_ok=True)

    clips: list[dict | None] = [None] * len(paths)
    skipped: list[dict] = [{"path": m, "reason": S.input_not_found(m)} for m in unmatched]
    lock = threading.Lock()

    def work(index: int, path: str) -> None:
        if cancel is not None and cancel.is_set():
            return
        messenger.say(S.processing_file(index + 1, len(paths), os.path.basename(path)))
        try:
            clips[index] = analyse_clip(path, options, messenger, created, cancel)
            if on_clip_done is not None:
                on_clip_done(clip_summary(index, path, clips[index]))
        except ProbeError as exc:
            messenger.say(S.probe_failed(os.path.basename(path), str(exc)))
            with lock:
                skipped.append({"path": path, "reason": str(exc)})
            if on_clip_done is not None:
                on_clip_done(clip_summary(index, path, None, str(exc)))
        except Exception as exc:  # noqa: BLE001 - one bad file must not kill the batch
            detail = f"{type(exc).__name__}: {exc}"
            messenger.say(S.probe_failed(os.path.basename(path), detail))
            with lock:
                skipped.append({"path": path, "reason": detail})
            if on_clip_done is not None:
                on_clip_done(clip_summary(index, path, None, detail))

    jobs = options.jobs if options.jobs and options.jobs > 0 else min(4, os.cpu_count() or 1)
    if jobs == 1:
        for index, path in enumerate(paths):
            work(index, path)
    else:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            list(pool.map(lambda pair: work(*pair), list(enumerate(paths))))

    done = [c for c in clips if c is not None]
    elapsed = time.perf_counter() - started
    footage_seconds = sum(float(c["probe"].get("duration") or 0.0) for c in done)
    frames_delivered = sum(len(c["frames"]) for c in done)

    results = {
        "tool": "frame-picker",
        "version": VERSION,
        "generated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "options": options.as_dict(),
        "weights": dict(scoring.WEIGHTS),
        "summary": {
            "files_given": len(paths) + len(unmatched),
            "files_processed": len(done),
            "files_skipped": len(skipped),
            "select_mode": options.select_mode,
            "min_score": options.min_score if options.select_mode == MODE_THRESHOLD else None,
            # A fixed target only exists in count mode; in threshold mode the
            # count is an outcome, not a promise, so there is nothing to fall
            # short of and the field stays null instead of pretending.
            "frames_requested": len(done) * options.per_clip if options.select_mode == MODE_COUNT else None,
            "frames_delivered": frames_delivered,
            "footage_seconds": footage_seconds,
            "wall_clock_seconds": elapsed,
            "throughput_s_per_footage_minute": (elapsed / (footage_seconds / 60.0)) if footage_seconds > 0 else None,
        },
        "notes": list(messenger.notes),
        "clips": done,
        "skipped": skipped,
    }
    if options.global_top:
        results["global_top"] = _global_top(done, options.global_top)

    if cancel is not None and cancel.is_set():
        created.remove_all()
        messenger.say(S.cancelled_cleanup(options.out_dir))
        return BatchResult(results, options.out_dir, cancelled=True, messages=messenger.log)

    json_path = report.write_results_json(results, options.out_dir)
    created.add(json_path)
    html_path = report.write_report_html(results, options.out_dir)
    created.add(html_path)

    if options.select_mode == MODE_THRESHOLD:
        messenger.say(S.batch_summary_threshold(
            len(paths) + len(unmatched), len(done), len(skipped), frames_delivered))
    else:
        messenger.say(S.batch_summary(
            len(paths) + len(unmatched), len(done), len(skipped),
            len(done) * options.per_clip, frames_delivered))
    if skipped:
        messenger.say(S.skipped_files_header())
        for item in skipped:
            messenger.say(S.probe_failed(os.path.basename(item["path"]), item["reason"]))
    if footage_seconds > 0:
        messenger.say(S.throughput(elapsed / (footage_seconds / 60.0)))
    messenger.say(S.output_written(os.path.abspath(options.out_dir)))

    return BatchResult(results, options.out_dir, json_path, html_path, messages=messenger.log)


def _global_top(clips: Sequence[dict], count: int) -> list[dict]:
    """Best frames of the whole batch, on raw values. Less reliable, and said so."""
    pool: list[dict] = []
    for clip in clips:
        for frame in clip.get("frames", []):
            item = dict(frame)
            item["clip"] = clip["probe"].get("name")
            pool.append(item)
    pool.sort(key=lambda f: -float(f.get("score") or 0.0))
    for rank, item in enumerate(pool[:count], start=1):
        item["rank"] = rank
    return pool[:count]


# --------------------------------------------------------------------------
# argparse
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m framepicker",
        description="Pick the best frames out of video files and say why.",
    )
    parser.add_argument("videos", nargs="*",
                        help="video files, folders, or wildcards such as D:/clips/*.MP4")
    parser.add_argument("--out", dest="out_dir", default=DEFAULT_OUT_DIR, help="output directory")
    parser.add_argument("--per-clip", type=int, default=DEFAULT_PER_CLIP,
                        help="frames to pick per clip (count mode only)")
    parser.add_argument("--select", dest="select_mode", choices=(MODE_THRESHOLD, MODE_COUNT),
                        default=MODE_THRESHOLD,
                        help="'threshold' keeps every frame above --min-score; "
                             "'count' aims at --per-clip and reports any shortfall")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE,
                        help="score a frame must reach in threshold mode "
                             "(starting value, not calibrated)")
    parser.add_argument("--max-per-clip", type=int, default=DEFAULT_MAX_PER_CLIP,
                        help="upper bound on frames per clip in threshold mode (0 = no bound)")
    parser.add_argument("--export-height", type=int, default=0,
                        help="scale exported stills down to this height "
                             "(e.g. 1080); 0 keeps the source resolution")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="analysis sampling rate")
    parser.add_argument("--min-gap", type=float, default=DEFAULT_MIN_GAP,
                        help="minimum seconds between two picked frames")
    parser.add_argument("--convert-log", choices=("auto", "on", "off"), default="auto",
                        help="treat the footage as flat/log")
    parser.add_argument("--lut", default=None,
                        help="path to a .cube LUT, applied only to clips detected as log")
    parser.add_argument("--lut-all", action="store_true",
                        help="apply --lut to every clip, log or not (wrecks Rec.709 footage)")
    parser.add_argument("--jobs", type=int, default=0, help="files processed in parallel (0 = auto)")
    parser.add_argument("--no-faces", action="store_true", help="skip face detection entirely")
    parser.add_argument("--face-model", default=None, help="explicit path to a YuNet .onnx model")
    parser.add_argument("--format", dest="image_format", choices=("jpg", "png"), default="jpg",
                        help="exported image format")
    parser.add_argument("--jpeg-quality", type=int, default=2, help="ffmpeg -q:v for JPEG (2 = best)")
    parser.add_argument("--global-top", type=int, default=DEFAULT_GLOBAL_TOP,
                        help="frames on the 'best of the whole batch' section (0 turns it off)")
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES,
                        help="upper bound on analysis frames buffered per clip")
    parser.add_argument("--hwaccel", default="auto", help="hardware decoder to try ('auto', 'cuda', 'none')")
    parser.add_argument("--version", action="version", version=f"frame-picker {VERSION}")
    return parser


def options_from_args(args: argparse.Namespace) -> Options:
    return Options(
        paths=list(args.videos),
        out_dir=args.out_dir,
        per_clip=args.per_clip,
        fps=args.fps,
        min_gap=args.min_gap,
        convert_log=args.convert_log,
        lut=args.lut,
        lut_all=args.lut_all,
        jobs=args.jobs,
        no_faces=args.no_faces,
        face_model=args.face_model,
        image_format=args.image_format,
        jpeg_quality=args.jpeg_quality,
        global_top=args.global_top,
        max_candidates=args.max_candidates,
        hwaccel=args.hwaccel,
        select_mode=args.select_mode,
        min_score=args.min_score,
        max_per_clip=args.max_per_clip,
        export_height=args.export_height,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    options = options_from_args(args)
    if not options.paths:
        print(S.no_input_files(), flush=True)
        return 2

    cancel = threading.Event()
    previous = None
    try:
        previous = signal.signal(signal.SIGINT, lambda *_: cancel.set())
    except (ValueError, AttributeError):  # not the main thread, or no SIGINT
        previous = None
    try:
        result = run_batch(options, cancel=cancel)
    except KeyboardInterrupt:
        cancel.set()
        print(S.cancelled_cleanup(os.path.abspath(options.out_dir)), flush=True)
        return 130
    finally:
        if previous is not None:
            try:
                signal.signal(signal.SIGINT, previous)
            except (ValueError, AttributeError):
                pass
    if result.cancelled:
        return 130
    if not result.results:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
