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
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Callable, Sequence

from . import (decode, export, features, grading, keepawake, proc, report, runlog,
               scoring, sidecar)
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
#: Where the footage normally lives on Tomas's machine. Used only as a
#: starting point: if the folder is not there, nothing about it is assumed.
DEFAULT_SOURCE_DIR = r"D:\tomas\Videos\DJI Drone foot"
#: Every run gets its own folder under the output directory, named after the
#: moment it started. Without this, a second run drops its stills next to the
#: first run's and the check at the end reports files nothing refers to -
#: which is exactly what happened on the 163-file run: 12 stale frames.
RUN_DIR_PREFIX = "run-"
RUN_DIR_TIME_FORMAT = "%Y%m%d-%H%M%S"
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
#: Processing order of the expanded file list.
ORDER_DATE = "date"      # oldest first, by the file's modification time
ORDER_NAME = "name"
ORDER_NONE = "none"      # exactly as given on the command line

#: Extensions treated as video when a folder or a wildcard is given. Lives
#: here, not in the GUI: expanding an input into a file list is logic.
VIDEO_SUFFIXES = (
    ".mp4", ".mov", ".mxf", ".mkv", ".avi", ".m4v", ".mts", ".m2ts", ".insv", ".webm",
)

#: Strengths tried for a LUT, strongest first, when none was asked for.
#:
#: A cube file has no dial inside it, so "convert, but less" means blending the
#: converted image back over the original. The ladder exists because a LUT can
#: be too strong for the footage it is pointed at: measured on Tomas's D-Log
#: clips, his D-Log M cube at full strength took saturation 0.210 -> 0.384 and
#: pushed 12 % of every frame to black, while 0.5 gave 0.278 and 2 %.
LUT_STRENGTH_LADDER = (1.0, 0.75, 0.5, 0.25)

#: Analysis may run on DJI's ``.LRF`` proxy instead of the master file.
PROXY_AUTO = "auto"
PROXY_OFF = "off"


def default_source_dir() -> str | None:
    """The usual footage folder, or ``None`` when it does not exist here."""
    return DEFAULT_SOURCE_DIR if os.path.isdir(DEFAULT_SOURCE_DIR) else None


def default_out_dir() -> str:
    """Output next to the footage when the usual folder exists, else here."""
    source = default_source_dir()
    return os.path.join(source, DEFAULT_OUT_DIR) if source else DEFAULT_OUT_DIR


def run_directory(out_dir: str, enabled: bool = True, when: datetime | None = None) -> str:
    """Path of the folder this run writes into.

    A run never shares a folder with another run unless that is asked for:
    mixing them makes the integrity check report the previous run's frames as
    unreferenced files, and makes the folder unusable as a deliverable.
    """
    if not enabled:
        return out_dir
    stamp = (when or datetime.now()).strftime(RUN_DIR_TIME_FORMAT)
    base = os.path.join(out_dir, f"{RUN_DIR_PREFIX}{stamp}")
    candidate = base
    suffix = 2
    while os.path.exists(candidate):
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate


#: A frame this uniformly black or white is thrown away before any real work.
CHEAP_REJECT_CLIP_FRACTION = 0.90
#: Sharpness percentile below which a frame is rejected as soft, for this clip.
CHEAP_REJECT_SHARPNESS_QUANTILE = 0.10


@dataclass
class Options:
    paths: list[str] = field(default_factory=list)
    out_dir: str = field(default_factory=default_out_dir)
    per_clip: int = DEFAULT_PER_CLIP
    fps: float = DEFAULT_FPS
    min_gap: float = DEFAULT_MIN_GAP
    convert_log: str = "auto"
    lut: str | None = None
    lut_all: bool = False
    normalise_strength: float = 1.0
    #: ``None`` = measure and choose; a number = use exactly that.
    lut_strength: float | None = None
    look: str = grading.NONE
    look_strength: float = grading.DEFAULT_STRENGTH
    jobs: int = 0
    no_faces: bool = False
    face_model: str | None = None
    image_format: str = "jpg"
    jpeg_quality: int = 2
    global_top: int = DEFAULT_GLOBAL_TOP
    max_candidates: int = DEFAULT_MAX_CANDIDATES
    hwaccel: str = "auto"
    order: str = ORDER_DATE
    select_mode: str = MODE_THRESHOLD
    min_score: float = DEFAULT_MIN_SCORE
    max_per_clip: int = DEFAULT_MAX_PER_CLIP
    export_height: int = 0
    proxy: str = PROXY_AUTO
    keyframes: bool = False
    gpu_scale: bool = True
    run_folder: bool = True
    write_log: bool = True

    def as_dict(self) -> dict:
        return {
            "out_dir": self.out_dir,
            "per_clip": self.per_clip,
            "fps": self.fps,
            "min_gap": self.min_gap,
            "convert_log": self.convert_log,
            "lut": self.lut,
            "lut_all": self.lut_all,
            "normalise_strength": self.normalise_strength,
            "lut_strength": self.lut_strength,
            "look": self.look,
            "look_strength": self.look_strength,
            "jobs": self.jobs,
            "no_faces": self.no_faces,
            "image_format": self.image_format,
            "jpeg_quality": self.jpeg_quality,
            "global_top": self.global_top,
            "max_candidates": self.max_candidates,
            "hwaccel": self.hwaccel,
            "order": self.order,
            "select_mode": self.select_mode,
            "min_score": self.min_score,
            "max_per_clip": self.max_per_clip,
            "export_height": self.export_height,
            "proxy": self.proxy,
            "keyframes": self.keyframes,
            "gpu_scale": self.gpu_scale,
            "run_folder": self.run_folder,
            "write_log": self.write_log,
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
    #: log.txt and log.jsonl, when the run wrote them.
    log_paths: list[str] = field(default_factory=list)


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

    # -- proxy -------------------------------------------------------------
    # DJI writes a 720p .LRF next to every take. Analysis runs at 640 px, so
    # the proxy carries the same information for a fraction of the decoding
    # work; the exported stills always come from the master file.
    analysis_clip = clip
    proxy = None
    if options.proxy != PROXY_OFF:
        found = sidecar.find_proxy(path)
        if found:
            # A proxy is only useful if it is at least as big as the frames
            # the analysis would have used anyway.
            needed = min(decode.ANALYSIS_LONG_EDGE, max(int(clip.width or 0), int(clip.height or 0)))
            proxy = sidecar.check_proxy(clip, found, needed)
            if proxy.usable:
                try:
                    analysis_clip = probe(proxy.path)
                    _note(record, messenger, S.proxy_used(
                        os.path.basename(proxy.path), proxy.width or 0, proxy.height or 0))
                except ProbeError as exc:
                    proxy = sidecar.Proxy(found, False, str(exc))
            if not proxy.usable:
                _note(record, messenger, S.proxy_rejected(
                    os.path.basename(proxy.path), proxy.detail))
    record["proxy"] = proxy.as_dict() if proxy is not None else None

    # -- colour ------------------------------------------------------------
    # What the camera itself said, if it said anything: the caption sidecar is
    # the only place the picture profile appears in words.
    color_mode = sidecar.read_color_mode(path)
    if color_mode is not None:
        _note(record, messenger, S.color_mode_found(
            color_mode.value, os.path.basename(color_mode.source)))
    else:
        messenger.once("color-mode-missing", S.color_mode_missing())

    # Then sample a handful of frames: measured flatness is reported as
    # evidence, and is the last thing left when nothing else says anything.
    samples: list = []
    stats = None
    if options.convert_log != "off":
        samples = decode.sample_for_normalisation(analysis_clip)
        stats = decode.flatness(samples)

    verdict = decode.detect_log(clip, options.convert_log, stats, color_mode)
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
        if verdict.suspected and not verdict.is_log:
            warning = S.log_suspected_not_applied(stats["luma_span"], stats["saturation"])
            record["notes"].append(warning)
            messenger.say(warning)
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
            if verdict.profile == "hlg":
                # A D-Log LUT on HLG footage is the wrong conversion. Said out
                # loud rather than silently applied.
                _note(record, messenger, S.lut_profile_mismatch(verdict.profile))
            if options.lut_all and not verdict.is_log:
                _note(record, messenger, S.lut_forced_on_all())
        else:
            _note(record, messenger, S.lut_skipped_not_log())
    # -- does the conversion actually improve the picture? ------------------
    # Applying a LUT is a claim about the source, and the claim is now tested
    # rather than trusted: the same frames are decoded again through the LUT
    # and compared with the untouched ones. Tomas's card is the reason - 162 of
    # its 163 files are 10-bit, but their measured saturation and contrast are
    # those of ordinary footage, so the LUT was being applied to pictures that
    # were already right and made them gaudy.
    lut_strength = 1.0
    if lut_path is not None:
        forced = options.lut_all or options.convert_log == "on"
        if not samples:
            _note(record, messenger, S.conversion_not_checked())
        else:
            lut_strength, check, tried = choose_lut_strength(
                analysis_clip, lut_path, samples, options.lut_strength)
            record["color_check"] = check.as_dict()
            record["color_check"]["strength"] = lut_strength
            record["color_check"]["tried"] = [
                {"strength": s, "ok": c.ok, "reason": c.reason} for s, c in tried
            ]
            if check.ok and lut_strength >= 1.0:
                _note(record, messenger, S.conversion_ok(
                    check.before["saturation"], check.after["saturation"],
                    check.before["luma_span"], check.after["luma_span"]))
            elif check.ok:
                _note(record, messenger, S.conversion_softened(
                    lut_strength, check.before["saturation"], check.after["saturation"]))
            elif forced:
                _note(record, messenger, S.conversion_forced(
                    S.conversion_reason(check.reason)))
            else:
                lut_path = None
                _note(record, messenger, S.conversion_rejected(
                    S.conversion_reason(check.reason),
                    (check.before or {}).get("saturation", 0.0),
                    (check.after or {}).get("saturation", 0.0)))

    if (
        lut_path is None
        and verdict.is_log
        and options.normalise_strength > 0
        and record.get("color_check") is None      # not a LUT the check just refused
    ):
        normalisation = (
            decode.estimate_normalisation(samples, options.normalise_strength) if samples else None
        )
        if normalisation is not None:
            _note(record, messenger, S.normalisation_applied(
                normalisation.strength, normalisation.saturation_gain))
    if lut_path is None and normalisation is None:
        record["notes"].append(S.no_conversion_applied())
    record["color"] = {
        "mode": "lut" if lut_path else ("normalise" if normalisation else "none"),
        "check": record.get("color_check"),
        "lut_strength": lut_strength if lut_path else None,
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
    previous_frame = None
    if proxy is not None and proxy.usable:
        decode_report.proxy_file = os.path.basename(proxy.path)
    for timestamp, frame in decode.sample_frames(
        analysis_clip,
        options.fps,
        lut_path=lut_path,
        lut_strength=lut_strength,
        hwaccel=options.hwaccel,
        max_candidates=options.max_candidates,
        keyframes_only=options.keyframes,
        gpu_scale=options.gpu_scale,
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
            "motion": features.motion(previous_frame, work),
            "sharpness": features.sharpness(work),
            "exposure_clip_low": features.exposure_clip_low(work),
            "exposure_clip_high": features.exposure_clip_high(work),
            "dynamic_range": features.dynamic_range(work),
            "colorfulness": features.colorfulness(work),
            "saturation_mean": features.saturation_mean(work),
        })
        previous_frame = work
    record["decode"] = decode_report.as_dict()
    messenger.say(S.decode_path_hw() if decode_report.path_used == "hw"
                  else S.decode_path_cpu(decode_report.hw_error))
    if decode_report.gpu_scaler:
        messenger.once("gpu-scale", S.gpu_scale_on(decode_report.gpu_scaler))
    elif decode_report.path_used == "hw":
        messenger.once("gpu-scale-off", S.gpu_scale_off(decode_report.gpu_scale_error))
    if decode_report.keyframes_only:
        _note(record, messenger, S.keyframes_only(
            decode_report.frames_yielded, decode_report.effective_fps))
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
        item["subject_separation"] = None
        if saliency is not None:
            subject = saliency.subject(frame)
            if subject is not None:
                rel, cx, cy, separation = subject
                item["subject_rel"], item["subject_cx"], item["subject_cy"] = rel, cx, cy
                item["subject_separation"] = separation
                item["thirds_distance"] = features.thirds_distance(cx, cy)
        item["horizon_tilt"] = features.horizon_tilt(frame)
        # What aerial work is actually judged on: symmetry, repetition, and a
        # small subject in a lot of space. Measured here, weighed in scoring.
        item["symmetry"] = features.symmetry(frame)
        item["pattern_repetition"] = features.pattern_repetition(frame)
        item["negative_space"] = features.negative_space(
            item["subject_rel"], item["subject_separation"])
        score, reasons = scoring.score_frame_explained(item)
        # The component values, kept as numbers as well as sentences: the
        # calibration tool needs to read what the score was built from.
        item["components"] = scoring.component_values(item)
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

    # -- look --------------------------------------------------------------
    # Decided once per clip, on frames spread across it: a look that changed
    # from shot to shot inside one take would be worse than none. The decision
    # is made on the analysis frames *after* the conversion, so it sees the
    # same colours the export will.
    look_name = options.look
    auto_choice = None
    if options.look == grading.AUTO:
        signatures: list[dict] = []
        if survivors:
            step = max(1, len(survivors) // grading.AUTO_SAMPLE)
            for i in survivors[::step][:grading.AUTO_SAMPLE]:
                frame = _decode_buffer(buffers[i])
                if frame is not None:
                    signatures.append(features.scene_signature(frame))
        auto_choice = grading.classify_frames(signatures)
        look_name = auto_choice["choice"]
        if auto_choice["decided"]:
            _note(record, messenger, S.look_auto_decided(
                look_name, auto_choice["nature_score"], auto_choice["city_score"],
                auto_choice["frames_measured"]))
        else:
            _note(record, messenger, S.look_auto_undecided(
                auto_choice["nature_score"], auto_choice["city_score"],
                grading.AUTO_MARGIN))
    look = grading.get(look_name)
    record["look"] = {
        "requested": options.look,
        "name": look_name,
        "strength": options.look_strength,
        "definition": look.as_dict() if look else None,
        "auto": auto_choice,
    }

    # -- export ------------------------------------------------------------
    if cancel is None or not cancel.is_set():
        results, errors = export.export_selection(
            clip.path,
            clip.name,
            selection.selected,
            options.out_dir,
            lut_path=lut_path,
            lut_strength=lut_strength,
            normalisation=normalisation,
            quality=options.jpeg_quality,
            image_format=options.image_format,
            height=options.export_height,
            look=look,
            look_strength=options.look_strength,
            cancel=cancel,
        )
        record["notes"].append(
            S.export_resolution_scaled(options.export_height) if options.export_height
            else S.export_resolution_native()
        )
        _note(record, messenger, S.look_applied(look_name, options.look_strength)
              if look else S.look_none())
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


def choose_lut_strength(
    clip,
    lut_path: str,
    raw_samples: list,
    requested: float | None,
) -> tuple[float, "decode.ConversionCheck", list]:
    """How much of the LUT this clip can take, measured rather than assumed.

    One extra decode: the same frames through the LUT at full strength. Every
    intermediate strength is then a linear blend of the two, so the whole
    ladder costs nothing more. Returns the chosen strength, the check that
    settled it, and every rung tried.
    """
    converted = decode.sample_for_normalisation(clip, lut_path=lut_path)
    if requested is not None:
        blended = decode.blend_frames(raw_samples, converted, requested)
        return float(requested), decode.check_conversion(raw_samples, blended), []

    tried: list[tuple[float, decode.ConversionCheck]] = []
    for strength in LUT_STRENGTH_LADDER:
        blended = (
            converted if strength >= 1.0
            else decode.blend_frames(raw_samples, converted, strength)
        )
        check = decode.check_conversion(raw_samples, blended)
        tried.append((strength, check))
        if check.ok:
            return strength, check, tried
    return 0.0, tried[-1][1], tried


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


def _mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def expand_inputs(paths: Sequence[str], order: str = ORDER_DATE) -> tuple[list[str], list[str]]:
    """Turn CLI arguments into a concrete, ordered file list.

    A folder becomes every video file in it, and a wildcard is expanded here
    rather than by the shell - ``cmd`` and PowerShell do not glob arguments
    for a Python program, so ``*.MP4`` would otherwise arrive verbatim and
    match nothing.

    The whole result is then ordered: ``ORDER_DATE`` (default) puts the oldest
    file first by its modification time, which is the recording order for a
    card copied off a camera; ``ORDER_NAME`` sorts by filename; ``ORDER_NONE``
    keeps the order the arguments were given in.

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

    if order == ORDER_DATE:
        # Name as the tie-break, so two files written in the same second keep
        # a stable order instead of shuffling between runs.
        unique.sort(key=lambda p: (_mtime(p), os.path.basename(p).lower()))
    elif order == ORDER_NAME:
        unique.sort(key=lambda p: os.path.basename(p).lower())
    return unique, unmatched


def clip_summary(index: int, path: str, record: dict | None, reason: str = "",
                 values: dict | None = None) -> dict:
    """Flat, display-ready summary of one file. Built here so the GUI can
    render a table without knowing anything about the pipeline."""
    if record is None:
        return {
            "index": index, "path": path, "name": os.path.basename(path), "ok": False,
            "is_log": None, "log_source": "", "color_mode": "", "decode_path": "",
            "look": "", "proxy": "", "frames": 0, "elapsed_s": 0.0, "reason": reason,
            "values": {},
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
        "look": record.get("look", {}).get("name", ""),
        "proxy": (record.get("proxy") or {}).get("file", "") if record.get("proxy") else "",
        "decode_path": record.get("decode", {}).get("path_used", ""),
        "frames": len(record.get("frames", [])),
        "elapsed_s": float(record.get("elapsed_s") or 0.0),
        "reason": "",
        # Everything measured about this file, flat, for a window that wants to
        # show values without knowing what any of them mean.
        "values": values if values is not None else runlog.clip_event(index, path, record),
    }


def run_batch(
    options: Options,
    on_message: Callable[[str], None] | None = None,
    cancel: threading.Event | None = None,
    on_clip_done: Callable[[dict], None] | None = None,
    _print_messages: bool = True,
) -> BatchResult:
    """Process every file in ``options.paths``. One bad file never aborts it."""
    log = runlog.RunLog(enabled=options.write_log)

    def sink(text: str) -> None:
        # Everything said goes to the log file as well as to the caller, so
        # the console, the window and log.txt can never disagree.
        log.message(text)
        if on_message is not None:
            on_message(text)
        elif _print_messages:
            print(text, flush=True)

    messenger = Messenger(sink)
    created = _Created()
    started = time.perf_counter()

    for tool in ("ffmpeg", "ffprobe"):
        if proc.executable(tool) is None:
            messenger.say(S.ffmpeg_missing(tool))
            return BatchResult({}, options.out_dir, messages=messenger.log)

    paths, unmatched = expand_inputs(options.paths, options.order)
    for missing in unmatched:
        messenger.say(S.input_not_found(missing))
    if len(paths) != len(options.paths):
        messenger.say(S.inputs_expanded(len(options.paths), len(paths)))
    if len(paths) > 1:
        if options.order == ORDER_DATE:
            messenger.say(S.ordered_by_date())
        elif options.order == ORDER_NAME:
            messenger.say(S.ordered_by_name())
    if not paths:
        messenger.say(S.no_input_files())
        return BatchResult({}, options.out_dir, messages=messenger.log)

    # Its own folder for this run, created here and reported. Every path below
    # is inside it, so a second run can never overwrite or pollute the first.
    run_dir = run_directory(options.out_dir, options.run_folder)
    os.makedirs(run_dir, exist_ok=True)
    if options.run_folder:
        messenger.say(S.run_folder_created(os.path.abspath(run_dir)))
    log.open(run_dir)
    for path in log.paths:
        created.add(path)
    if log.paths:
        messenger.say(S.log_written(os.path.basename(log.paths[0]),
                                    os.path.basename(log.paths[1])))
    log.event("run_started", options=options.as_dict(), files=len(paths),
              weights=dict(scoring.WEIGHTS), version=VERSION)
    clip_options = replace(options, out_dir=run_dir)

    clips: list[dict | None] = [None] * len(paths)
    skipped: list[dict] = [{"path": m, "reason": S.input_not_found(m)} for m in unmatched]
    lock = threading.Lock()

    def work(index: int, path: str) -> None:
        if cancel is not None and cancel.is_set():
            return
        messenger.say(S.processing_file(index + 1, len(paths), os.path.basename(path)))
        try:
            clips[index] = analyse_clip(path, clip_options, messenger, created, cancel)
            values = runlog.clip_event(index, path, clips[index])
            log.event("clip", **values)
            for frame in runlog.frame_events(values["file"], clips[index]):
                log.event("frame", **frame)
            if on_clip_done is not None:
                on_clip_done(clip_summary(index, path, clips[index], values=values))
        except ProbeError as exc:
            messenger.say(S.probe_failed(os.path.basename(path), str(exc)))
            log.event("clip_failed", index=index, file=os.path.basename(path), reason=str(exc))
            with lock:
                skipped.append({"path": path, "reason": str(exc)})
            if on_clip_done is not None:
                on_clip_done(clip_summary(index, path, None, str(exc)))
        except Exception as exc:  # noqa: BLE001 - one bad file must not kill the batch
            detail = f"{type(exc).__name__}: {exc}"
            messenger.say(S.probe_failed(os.path.basename(path), detail))
            log.event("clip_failed", index=index, file=os.path.basename(path), reason=detail)
            with lock:
                skipped.append({"path": path, "reason": detail})
            if on_clip_done is not None:
                on_clip_done(clip_summary(index, path, None, detail))

    jobs = options.jobs if options.jobs and options.jobs > 0 else min(4, os.cpu_count() or 1)
    with keepawake.keep_awake() as awake:
        if awake.active:
            messenger.say(S.keep_awake_on())
        elif awake.detail and sys.platform == "win32":
            messenger.say(S.keep_awake_unavailable(awake.detail))
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
        "output_dir": os.path.abspath(run_dir),
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
        "keep_awake": awake.as_dict(),
        "notes": list(messenger.notes),
        "clips": done,
        "skipped": skipped,
    }
    if options.global_top:
        results["global_top"] = _global_top(done, options.global_top)

    if cancel is not None and cancel.is_set():
        log.event("cancelled", files_done=len(done))
        log.close()
        created.remove_all()
        # The run's own folder goes too, when this run is what created it and
        # nothing else ended up in it. Cancelling has to leave no trace, so
        # that loading a new set of files starts from a clean state.
        if options.run_folder:
            try:
                if not os.listdir(run_dir):
                    os.rmdir(run_dir)
            except OSError:
                pass
        messenger.say(S.cancelled_cleanup(run_dir))
        return BatchResult(results, run_dir, cancelled=True, messages=messenger.log)

    # Build every preview first, so a preview that cannot be produced becomes a
    # named finding instead of a silent gap in the page, then check the whole
    # output before writing anything final.
    previews = report.build_previews(results, run_dir)
    results["integrity"] = report.verify(results, run_dir, previews)
    for text in results["integrity"]["messages"]:
        messenger.say(text)

    json_path = report.write_results_json(results, run_dir)
    created.add(json_path)
    html_path = report.write_report_html(results, run_dir, previews)
    created.add(html_path)
    results["integrity"]["report_files"] = S.integrity_report_files(
        os.path.isfile(json_path) and os.path.getsize(json_path) > 0,
        os.path.isfile(html_path) and os.path.getsize(html_path) > 0,
    )
    messenger.say(results["integrity"]["report_files"])
    # The side-file sizes were measured before results.json existed, so they
    # are taken again now that everything is on disk.
    results["integrity"]["side_files"] = {
        name: (os.path.getsize(os.path.join(run_dir, name))
               if os.path.isfile(os.path.join(run_dir, name)) else None)
        for name in report.SIDE_FILES
    }
    # And the page's own links, read back off disk rather than assumed.
    links = report.check_report_links(html_path, run_dir)
    results["integrity"]["links"] = links
    results["integrity"]["ok"] = results["integrity"]["ok"] and links["broken"] == 0
    messenger.say(S.integrity_links(links["checked"], links["broken"]))
    for detail in links["details"][:10]:
        messenger.say(detail)
    # results.json is rewritten so it carries the check on the report files too.
    report.write_results_json(results, run_dir)

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
    messenger.say(S.output_written(os.path.abspath(run_dir)))
    log.event("run_finished", summary=results["summary"], integrity=results["integrity"])
    log.close()
    if log.errors:
        messenger.say(S.log_write_failed(log.errors[0]))

    return BatchResult(results, run_dir, json_path, html_path, messages=messenger.log,
                       log_paths=list(log.paths))


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
    parser.add_argument("--out", dest="out_dir", default=default_out_dir(),
                        help="output directory; every run gets its own subfolder inside it")
    parser.add_argument("--no-run-folder", dest="run_folder", action="store_false",
                        help="write straight into --out instead of a per-run subfolder")
    parser.add_argument("--per-clip", type=int, default=DEFAULT_PER_CLIP,
                        help="frames to pick per clip (count mode only)")
    parser.add_argument("--select", dest="select_mode", choices=(MODE_THRESHOLD, MODE_COUNT),
                        default=MODE_THRESHOLD,
                        help="'threshold' keeps every frame above --min-score; "
                             "'count' aims at --per-clip and reports any shortfall")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE,
                        help="score a frame must reach in threshold mode "
                             "(0.60: chosen after a 163-file run, not a measured constant)")
    parser.add_argument("--max-per-clip", type=int, default=DEFAULT_MAX_PER_CLIP,
                        help="upper bound on frames per clip in threshold mode "
                             "(0 = no bound, the default: the threshold decides how many)")
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
    parser.add_argument("--lut-strength", default="auto",
                        help="how much of the LUT to apply: 'auto' measures each clip and "
                             "picks the strongest setting that does not overshoot, or give "
                             "a number 0..1")
    parser.add_argument("--normalise-strength", type=float, default=1.0,
                        help="strength of the no-LUT log fallback, 0 = off, 1 = full")
    parser.add_argument("--look", choices=grading.available(), default=grading.NONE,
                        help="look for the exported stills: 'auto' measures each clip and "
                             "picks one, or name it yourself: 'nature', 'city', 'none'")
    parser.add_argument("--look-strength", type=float, default=grading.DEFAULT_STRENGTH,
                        help="how far toward the look's targets to travel, 0 = off, 1 = full")
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
    parser.add_argument("--no-gpu-scale", dest="gpu_scale", action="store_false",
                        help="do not scale on the GPU even when the build supports it")
    parser.add_argument("--proxy", choices=(PROXY_AUTO, PROXY_OFF), default=PROXY_AUTO,
                        help="use a DJI .LRF proxy for analysis when one sits next to the file")
    parser.add_argument("--keyframes", action="store_true",
                        help="decode only keyframes: much faster, and the sampling grid "
                             "becomes the camera's keyframe interval")
    parser.add_argument("--order", choices=(ORDER_DATE, ORDER_NAME, ORDER_NONE), default=ORDER_DATE,
                        help="processing order: 'date' = oldest file first (default), "
                             "'name' = by filename, 'none' = as given")
    parser.add_argument("--version", action="version", version=f"frame-picker {VERSION}")
    return parser


def _lut_strength_arg(value: str) -> float | None:
    """``auto`` -> None (measure it), otherwise a number between 0 and 1."""
    text = str(value).strip().lower()
    if text in ("", "auto"):
        return None
    return min(1.0, max(0.0, float(text)))


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
        normalise_strength=args.normalise_strength,
        lut_strength=_lut_strength_arg(args.lut_strength),
        look=args.look,
        look_strength=args.look_strength,
        jobs=args.jobs,
        no_faces=args.no_faces,
        face_model=args.face_model,
        image_format=args.image_format,
        jpeg_quality=args.jpeg_quality,
        global_top=args.global_top,
        max_candidates=args.max_candidates,
        hwaccel=args.hwaccel,
        order=args.order,
        select_mode=args.select_mode,
        min_score=args.min_score,
        max_per_clip=args.max_per_clip,
        export_height=args.export_height,
        proxy=args.proxy,
        keyframes=args.keyframes,
        gpu_scale=args.gpu_scale,
        run_folder=args.run_folder,
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
