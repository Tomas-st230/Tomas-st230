"""Decoding video into downscaled RGB frames, and the colour handling that
has to happen before any number is measured.

Everything here goes through :mod:`framepicker.proc`; this module never
touches ``subprocess`` itself.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np

from . import proc
from .probe import ClipInfo

FFMPEG = "ffmpeg"

#: Long edge of every analysis frame. Fixed for the whole batch so that
#: sharpness numbers from different clips are at least measured on the same
#: pixel count.
ANALYSIS_LONG_EDGE = 640

#: Filename fragments that hint at a flat/log picture profile.
LOG_FILENAME_HINTS = ("dlog", "d-log", "dlogm", "d-log-m", "hlg", "slog", "s-log", "vlog", "v-log", "logc")

#: Colour transfer characteristics that mean "not a normal display gamma".
LOG_TRANSFER_TAGS = ("arib-std-b67", "smpte2084", "log100", "log316", "bt1361e")


class DecodeError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Log detection
# --------------------------------------------------------------------------


@dataclass
class LogVerdict:
    is_log: bool
    source: str          # machine-readable key: flag | metadata | filename | default
    detail: str          # what exactly was seen
    is_a_guess: bool

    def as_dict(self) -> dict:
        return {
            "is_log": self.is_log,
            "source": self.source,
            "detail": self.detail,
            "is_a_guess": self.is_a_guess,
        }


def detect_log(clip: ClipInfo, flag: str = "auto") -> LogVerdict:
    """Decide whether *clip* is flat/log footage.

    Order: explicit flag -> colour metadata -> filename hint -> off.
    The result is a guess unless the user forced it, and it says so.
    """
    if flag == "on":
        return LogVerdict(True, "flag", "--convert-log on", is_a_guess=False)
    if flag == "off":
        return LogVerdict(False, "flag", "--convert-log off", is_a_guess=False)

    transfer = (clip.color_transfer or "").lower()
    primaries = (clip.color_primaries or "").lower()
    if transfer in LOG_TRANSFER_TAGS:
        return LogVerdict(True, "metadata", f"color_transfer={transfer}", is_a_guess=True)
    if primaries == "bt2020" and transfer not in ("bt709", "bt470bg", "smpte170m", "iec61966-2-1"):
        return LogVerdict(True, "metadata", f"color_primaries={primaries}, color_transfer={transfer or '?'}", is_a_guess=True)

    lowered = clip.name.lower()
    for hint in LOG_FILENAME_HINTS:
        if hint in lowered:
            return LogVerdict(True, "filename", f"'{hint}' in {clip.name}", is_a_guess=True)

    return LogVerdict(False, "default", "no log evidence", is_a_guess=True)


# --------------------------------------------------------------------------
# Colour transform used for analysis
# --------------------------------------------------------------------------


@dataclass
class Normalisation:
    """Percentile contrast + saturation stretch, fixed for a whole clip.

    This is an approximation of a display conversion, not a colour-managed
    one. It exists so that a log clip is ranked on its own content instead of
    being flattened into noise; it is never claimed to be correct colour.
    """

    lo: float
    hi: float
    saturation_gain: float

    def as_dict(self) -> dict:
        return {"luma_low": self.lo, "luma_high": self.hi, "saturation_gain": self.saturation_gain}

    def apply(self, frame: np.ndarray) -> np.ndarray:
        import cv2

        span = max(self.hi - self.lo, 1e-3)
        work = frame.astype(np.float32)
        work = (work - self.lo) * (255.0 / span)
        np.clip(work, 0, 255, out=work)
        if abs(self.saturation_gain - 1.0) > 1e-3:
            hsv = cv2.cvtColor(work.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] *= self.saturation_gain
            np.clip(hsv[:, :, 1], 0, 255, out=hsv[:, :, 1])
            return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return work.astype(np.uint8)


#: Target mean saturation after normalising a log clip (0-255 scale).
NORMALISE_TARGET_SATURATION = 90.0
NORMALISE_MAX_GAIN = 2.5
NORMALISE_LOW_PERCENTILE = 1.0
NORMALISE_HIGH_PERCENTILE = 99.0


def estimate_normalisation(samples: Sequence[np.ndarray]) -> Normalisation | None:
    """Derive one contrast/saturation stretch from a handful of sample frames."""
    import cv2

    usable = [s for s in samples if s is not None and s.size]
    if not usable:
        return None
    stacked = np.concatenate([s.reshape(-1, 3)[::7] for s in usable], axis=0)
    luma = stacked.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    lo = float(np.percentile(luma, NORMALISE_LOW_PERCENTILE))
    hi = float(np.percentile(luma, NORMALISE_HIGH_PERCENTILE))
    if hi - lo < 1.0:
        lo, hi = 0.0, 255.0

    sat_means = []
    for frame in usable:
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        sat_means.append(float(hsv[:, :, 1].mean()))
    mean_sat = max(sum(sat_means) / len(sat_means), 1.0)
    gain = min(NORMALISE_MAX_GAIN, max(1.0, NORMALISE_TARGET_SATURATION / mean_sat))
    return Normalisation(lo=lo, hi=hi, saturation_gain=gain)


# --------------------------------------------------------------------------
# ffmpeg plumbing
# --------------------------------------------------------------------------


def analysis_size(width: int, height: int, long_edge: int = ANALYSIS_LONG_EDGE) -> tuple[int, int]:
    """Downscale *width* x *height* so the long edge is *long_edge*, even dims."""
    if width <= 0 or height <= 0:
        raise DecodeError("invalid source dimensions")
    scale = long_edge / float(max(width, height))
    if scale >= 1.0:
        scale = 1.0
    out_w = max(2, int(round(width * scale)) // 2 * 2)
    out_h = max(2, int(round(height * scale)) // 2 * 2)
    return out_w, out_h


def escape_filter_path(path: str) -> str:
    """Escape a filesystem path for use inside an ffmpeg filtergraph."""
    text = str(path).replace("\\", "/")
    for char in (":", ",", ";", "[", "]", "'"):
        text = text.replace(char, "\\" + char)
    return text


def build_video_filter(
    out_w: int,
    out_h: int,
    fps: float | None,
    lut_path: str | None,
) -> str:
    parts: list[str] = []
    if fps:
        parts.append(f"fps={fps}")
    parts.append(f"scale={out_w}:{out_h}:flags=area")
    if lut_path:
        parts.append(f"lut3d=file='{escape_filter_path(lut_path)}'")
    parts.append("format=rgb24")
    return ",".join(parts)


@dataclass
class DecodeReport:
    path_used: str = "cpu"            # "hw" or "cpu"
    hw_error: str = ""
    frames_yielded: int = 0
    frames_expected: int = 0
    effective_fps: float = 0.0
    requested_fps: float = 0.0
    capped: bool = False
    stderr: str = ""
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "path_used": self.path_used,
            "hw_error": self.hw_error,
            "frames_yielded": self.frames_yielded,
            "frames_expected": self.frames_expected,
            "effective_fps": self.effective_fps,
            "requested_fps": self.requested_fps,
            "capped": self.capped,
            "stderr": self.stderr,
            **self.extra,
        }


def hwaccel_available(path: str, hwaccel: str = "cuda") -> tuple[bool, str]:
    """Try to decode a single frame with *hwaccel*.

    Premise-checking rule (TRAP-12): we do not assume what this machine can or
    cannot decode, we run it once and look at the exit code.
    """
    if proc.executable(FFMPEG) is None:
        return False, f"{FFMPEG} not on PATH"
    argv = [
        FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error",
        "-hwaccel", hwaccel,
        "-i", path,
        "-frames:v", "1",
        "-f", "null", "-",
    ]
    try:
        result = proc.run(argv, timeout=60)
    except proc.ProcessError as exc:
        return False, str(exc)
    if result.ok:
        return True, ""
    return False, result.stderr_text(200) or f"exit {result.returncode}"


def sample_frames(
    clip: ClipInfo,
    fps: float,
    *,
    long_edge: int = ANALYSIS_LONG_EDGE,
    lut_path: str | None = None,
    hwaccel: str = "auto",
    max_candidates: int | None = None,
    cancel=None,
    report: DecodeReport | None = None,
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield ``(timestamp_seconds, rgb_frame)`` sampled at *fps*.

    Timestamps come from the sampling grid, not from container PTS: the
    ``fps`` filter emits frame *i* for source time ``i / fps``, which is what
    :mod:`framepicker.export` seeks back to.
    """
    if proc.executable(FFMPEG) is None:
        raise DecodeError(f"{FFMPEG} not on PATH")
    report = report if report is not None else DecodeReport()
    report.requested_fps = fps

    effective_fps = fps
    if max_candidates and clip.duration:
        expected = clip.duration * fps
        if expected > max_candidates:
            effective_fps = max(max_candidates / clip.duration, 1e-3)
            report.capped = True
    report.effective_fps = effective_fps
    if clip.duration:
        report.frames_expected = max(1, int(clip.duration * effective_fps))

    out_w, out_h = analysis_size(int(clip.width or 0), int(clip.height or 0), long_edge)
    vf = build_video_filter(out_w, out_h, effective_fps, lut_path)

    use_hw = False
    if hwaccel and hwaccel != "none":
        name = "cuda" if hwaccel == "auto" else hwaccel
        use_hw, why = hwaccel_available(clip.path, name)
        if not use_hw:
            report.hw_error = why
    report.path_used = "hw" if use_hw else "cpu"

    argv = [FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-nostats"]
    if use_hw:
        argv += ["-hwaccel", "cuda" if hwaccel == "auto" else hwaccel]
    argv += [
        "-i", clip.path,
        "-an", "-sn", "-dn",
        "-vf", vf,
        "-fps_mode", "passthrough",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-",
    ]

    frame_bytes = out_w * out_h * 3
    index = 0
    with proc.stream(argv) as handle:
        while True:
            if cancel is not None and cancel.is_set():
                break
            chunk = handle.read_exactly(frame_bytes)
            if chunk is None:
                break
            frame = np.frombuffer(chunk, dtype=np.uint8).reshape(out_h, out_w, 3)
            yield index / effective_fps, frame
            index += 1
        report.frames_yielded = index
        report.stderr = handle.stderr_text()


def sample_for_normalisation(
    clip: ClipInfo,
    *,
    count: int = 12,
    long_edge: int = 256,
    lut_path: str | None = None,
) -> list[np.ndarray]:
    """Decode a few frames spread over the clip, to derive one fixed transform."""
    if not clip.duration or clip.duration <= 0:
        sample_fps = 1.0
    else:
        sample_fps = max(count / clip.duration, 1e-3)
    report = DecodeReport()
    frames: list[np.ndarray] = []
    for _, frame in sample_frames(
        clip,
        sample_fps,
        long_edge=long_edge,
        lut_path=lut_path,
        hwaccel="none",
        report=report,
    ):
        frames.append(frame.copy())
        if len(frames) >= count * 2:
            break
    return frames


def cube_is_readable(path: str) -> tuple[bool, str]:
    """Cheap sanity check of a .cube file, so a bad LUT fails early and loudly."""
    if not os.path.isfile(path):
        return False, "file not found"
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            head = handle.read(8192)
    except OSError as exc:
        return False, str(exc)
    if "LUT_3D_SIZE" not in head and "LUT_1D_SIZE" not in head:
        return False, "no LUT_3D_SIZE / LUT_1D_SIZE header"
    return True, ""
