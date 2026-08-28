"""The written record of a run: one readable log, one machine-readable log.

Two files, both inside the run's own folder:

``log.txt``
    Every line the console printed, with a timestamp in front of it. This is
    what to send when something looks wrong.

``log.jsonl``
    One JSON object per line - the *values* behind those lines. Each clip
    writes what was measured about it (profile verdict and its evidence,
    decode path, candidate counts, confidence spread, the look decision and
    its scores, elapsed time) and each exported frame writes its score and the
    components the score was built from. A log you can only read is not a log
    you can check, so the numbers are kept as numbers.

Nothing here decides anything. It records, and it never raises: a run must not
fail because a log line could not be written.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime

LOG_TXT = "log.txt"
LOG_JSONL = "log.jsonl"

#: Timestamp on every human-readable line. Local time on purpose - it is read
#: next to the clock on the same machine.
TIME_FORMAT = "%H:%M:%S"


class RunLog:
    """Both log files for one run. Safe to call from several threads."""

    def __init__(self, folder: str | None = None, enabled: bool = True) -> None:
        self._lock = threading.Lock()
        self._folder = folder
        self._enabled = enabled
        self._text = None
        self._data = None
        self._pending: list[str] = []
        self.errors: list[str] = []
        if folder and enabled:
            self.open(folder)

    # -- lifecycle ---------------------------------------------------------

    def open(self, folder: str) -> None:
        """Start writing into *folder*, flushing anything said before now."""
        if not self._enabled:
            return
        with self._lock:
            self._folder = folder
            try:
                os.makedirs(folder, exist_ok=True)
                self._text = open(os.path.join(folder, LOG_TXT), "a", encoding="utf-8")
                self._data = open(os.path.join(folder, LOG_JSONL), "a", encoding="utf-8")
            except OSError as exc:
                self.errors.append(str(exc))
                self._text = self._data = None
                return
            pending, self._pending = self._pending, []
        for text in pending:
            self.message(text)

    def close(self) -> None:
        with self._lock:
            for handle in (self._text, self._data):
                try:
                    if handle is not None:
                        handle.close()
                except OSError:
                    pass
            self._text = self._data = None

    def __enter__(self) -> "RunLog":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # -- writing -----------------------------------------------------------

    @property
    def paths(self) -> list[str]:
        if not self._folder or not self._enabled:
            return []
        return [os.path.join(self._folder, LOG_TXT), os.path.join(self._folder, LOG_JSONL)]

    def message(self, text: str) -> None:
        """One console line, timestamped."""
        if not self._enabled:
            return
        with self._lock:
            if self._text is None:
                self._pending.append(text)
                return
            try:
                self._text.write(f"{datetime.now().strftime(TIME_FORMAT)}  {text}\n")
                self._text.flush()
            except OSError as exc:      # a full disk must not stop the run
                self.errors.append(str(exc))

    def event(self, kind: str, **values) -> None:
        """One record of measured values."""
        if not self._enabled:
            return
        record = {"time": datetime.now().isoformat(timespec="seconds"), "event": kind}
        record.update(values)
        with self._lock:
            if self._data is None:
                return
            try:
                self._data.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                self._data.flush()
            except (OSError, TypeError, ValueError) as exc:
                self.errors.append(str(exc))


def clip_event(index: int, path: str, record: dict) -> dict:
    """Flatten one clip's record into the values worth tracking.

    Built here rather than in the GUI or the log itself, so the console, the
    log file and the window all show the same numbers.
    """
    probe = record.get("probe", {})
    log = record.get("log", {})
    decode = record.get("decode", {})
    look = record.get("look", {})
    auto = look.get("auto") or {}
    selection = record.get("selection", {})
    confidence = record.get("confidence", {})
    rejects = record.get("rejects", {})
    proxy = record.get("proxy") or {}
    tags = (probe.get("extra") or {}).get("format_tags") or {}
    color_mode = log.get("color_mode") or {}

    return {
        "index": index,
        "file": os.path.basename(path),
        "duration_s": probe.get("duration"),
        "resolution": f'{probe.get("width")}x{probe.get("height")}',
        "fps": probe.get("fps"),
        "codec": probe.get("codec"),
        "pix_fmt": probe.get("pix_fmt"),
        "encoder_tag": tags.get("encoder"),
        "is_log": log.get("is_log"),
        "log_source": log.get("source"),
        "log_profile": log.get("profile"),
        "color_md": color_mode.get("value"),
        "log_is_a_guess": log.get("is_a_guess"),
        "luma_span": (log.get("statistics") or {}).get("luma_span"),
        "saturation": (log.get("statistics") or {}).get("saturation"),
        "color_mode": record.get("color", {}).get("mode"),
        "lut": record.get("color", {}).get("lut"),
        "proxy_file": proxy.get("file"),
        "proxy_used": proxy.get("usable"),
        "decode_path": decode.get("path_used"),
        "gpu_scaler": decode.get("gpu_scaler"),
        "keyframes_only": decode.get("keyframes_only"),
        "frames_sampled": decode.get("frames_yielded"),
        "effective_fps": decode.get("effective_fps"),
        "candidates": record.get("candidates_evaluated"),
        "rejected": rejects.get("total"),
        "rejected_dark": rejects.get("dark"),
        "rejected_bright": rejects.get("bright"),
        "rejected_blurry": rejects.get("blurry"),
        "confidence_spread": confidence.get("spread"),
        "confidence_informative": confidence.get("informative"),
        "look_requested": look.get("requested"),
        "look_applied": look.get("name"),
        "look_nature_score": auto.get("nature_score"),
        "look_city_score": auto.get("city_score"),
        "look_decided": auto.get("decided"),
        "best_score": selection.get("best_score"),
        "frames_delivered": len(record.get("frames", [])),
        "elapsed_s": record.get("elapsed_s"),
    }


def frame_events(clip_name: str, record: dict) -> list[dict]:
    """One record per exported frame: what it scored and what from."""
    out: list[dict] = []
    for frame in record.get("frames", []):
        features = frame.get("features", {})
        components = features.get("components") or {}
        out.append({
            "clip": clip_name,
            "rank": frame.get("rank"),
            "file": frame.get("file"),
            "timestamp_s": frame.get("timestamp"),
            "score": frame.get("score"),
            "exported": frame.get("exported"),
            "components": components,
            "sharpness": features.get("sharpness"),
            "sharpness_rank": features.get("sharpness_rank"),
            "motion_rank": features.get("motion_rank"),
            "horizon_tilt": features.get("horizon_tilt"),
            "face_count": features.get("face_count"),
            "subject_rel": features.get("subject_rel"),
            "thirds_distance": features.get("thirds_distance"),
            "symmetry": features.get("symmetry"),
            "pattern_repetition": features.get("pattern_repetition"),
            "negative_space": features.get("negative_space"),
        })
    return out
