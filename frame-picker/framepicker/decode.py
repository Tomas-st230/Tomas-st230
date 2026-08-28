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
LOG_FILENAME_HINTS = (
    "dlog", "d-log", "dlogm", "d-log-m",      # DJI
    "hlog", "h-log", "hovercine",             # HOVERAir
    "hlg", "slog", "s-log", "vlog", "v-log", "logc", "flat",
)

#: Colour transfer characteristics that mean "not a normal display gamma".
LOG_TRANSFER_TAGS = ("arib-std-b67", "smpte2084", "log100", "log316", "bt1361e")

#: Flatness limits below which a clip is *suspected* of being log.
#:
#: These do NOT trigger a colour transform, and that is deliberate. Measured
#: on 77 real DJI clips from one card, luma_span ran 0.155-0.925 and mean
#: saturation 0.074-0.765 in one continuous distribution with no bimodal gap:
#: every file was the same picture profile, so the measurement was reading the
#: *scene*, not the profile. Four dark sunset clips fell under these limits and
#: had their saturation multiplied by 2.5, which wrecked the exported stills.
#:
#: So flatness is now evidence to report, not a reason to act. A transform
#: happens only on the --convert-log flag, a log colour-transfer tag, or a
#: filename hint.
LOG_STATS_MAX_LUMA_SPAN = 0.55
LOG_STATS_MAX_SATURATION = 0.22

#: Container ``encoder`` tags that identify a DJI camera. Measured, not
#: guessed: every file on Tomas's card - MP4 and LRF alike - carries
#: ``encoder="DJI Lito X1"``, while its colour tags read ``bt709`` whatever
#: picture profile was used.
DJI_ENCODER_HINTS = ("dji",)

#: The inference that finally makes a LUT land on the right clips.
#:
#: Premise: on DJI drones the 10-bit recording modes are D-Log, D-Log M and
#: HLG; the Normal profile records 8-bit. So a DJI file in a 10-bit pixel
#: format was not shot in Normal colour, even though its colour tags claim
#: Rec.709. HLG is excluded first by its own transfer tag when the camera
#: writes one - and when it does not, the caption sidecar settles it, which is
#: why :mod:`framepicker.sidecar` is consulted before this rule.
#:
#: This is an inference about one manufacturer, so it is gated on the encoder
#: tag and it is reported as a guess. ``--convert-log off`` overrides it.
LOG_TENBIT_PIX_FMT_MARK = "10"


class DecodeError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Is the conversion an improvement, or is it wrecking the picture?
# --------------------------------------------------------------------------
#
# Applying a D-Log LUT rests on one premise: the source is flat and the LUT
# makes it normal. When that premise is wrong the LUT does the opposite - it
# stretches contrast that is already there and pushes colour that is already
# there, which is exactly what "too much of everything" looks like.
#
# The premise is now measured instead of assumed. Measured on Tomas's card:
# 162 of 163 files are 10-bit, yet their saturation runs 0.065-0.765 and their
# luma span 0.155-0.939 - one continuous distribution covering everything from
# genuinely flat to fully graded. Bit depth alone therefore cannot say what a
# file is, so the conversion is tried on a handful of frames and compared with
# the same frames untouched. If the result is *less* normal than what went in,
# the LUT is dropped for that clip and the report says which measurement
# refused it.

#: Mean HSV saturation the converted frames may not exceed - but only when the
#: conversion is what put them there. Footage that arrives colourful and comes
#: out equally colourful is not the LUT's doing, and refusing it would punish
#: the source for what it already was.
CONVERSION_MAX_SATURATION = 0.55
#: Increase, relative to the source, above which the ceiling starts to apply.
CONVERSION_SATURATION_NOTICEABLE = 1.05
#: ...nor may the conversion multiply the saturation by more than this. Same
#: measurement: 1.85x at full strength, 1.57x at 0.75, 1.34x at 0.5.
CONVERSION_MAX_SATURATION_GAIN = 1.45
#: Newly blown highlights the conversion may add (fraction of the frame).
CONVERSION_MAX_NEW_HIGH_CLIP = 0.05
#: Newly crushed blacks it may add. Measured on Tomas's own D-Log clips with
#: his own cube: at full strength the LUT pushed 12 % of every frame to black
#: (0.005 -> 0.122), at 0.75 6 %, at 0.5 2 %. A conversion that buries a
#: twentieth of the picture is doing more than converting.
CONVERSION_MAX_NEW_LOW_CLIP = 0.04
#: A log-to-display conversion opens the range up. If it closes it instead,
#: the source was not log: the converted span must keep at least this share.
CONVERSION_MIN_SPAN_KEPT = 0.90


@dataclass
class ConversionCheck:
    """What a conversion did to a few sample frames, and whether to keep it."""

    ok: bool
    reason: str                 # "" | saturation | gain | highlights | shadows | flattened | no-frames
    before: dict | None = None
    after: dict | None = None

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "before": self.before,
            "after": self.after,
            "limits": {
                "max_saturation": CONVERSION_MAX_SATURATION,
                "max_saturation_gain": CONVERSION_MAX_SATURATION_GAIN,
                "max_new_high_clip": CONVERSION_MAX_NEW_HIGH_CLIP,
                "max_new_low_clip": CONVERSION_MAX_NEW_LOW_CLIP,
                "min_span_kept": CONVERSION_MIN_SPAN_KEPT,
            },
        }


def _picture_stats(samples: Sequence[np.ndarray]) -> dict | None:
    """Mean saturation, luma span and clipping over a few frames."""
    from .features import dynamic_range, exposure_clip_high, exposure_clip_low, saturation_mean

    usable = [s for s in samples if s is not None and getattr(s, "size", 0)]
    if not usable:
        return None
    return {
        "saturation": float(sum(saturation_mean(f) for f in usable) / len(usable)),
        "luma_span": float(sum(dynamic_range(f) for f in usable) / len(usable)),
        "clip_high": float(sum(exposure_clip_high(f) for f in usable) / len(usable)),
        "clip_low": float(sum(exposure_clip_low(f) for f in usable) / len(usable)),
        "frames_measured": len(usable),
    }


def blend_frames(
    raw: Sequence[np.ndarray], converted: Sequence[np.ndarray], strength: float
) -> list[np.ndarray]:
    """``strength`` of *converted* over *raw*, per pixel.

    The same linear mix ffmpeg's ``blend`` performs, done here so that every
    candidate strength can be measured from **one** extra decode instead of
    one decode each. What finally gets exported is still ffmpeg's own blend.
    """
    strength = float(min(1.0, max(0.0, strength)))
    out: list[np.ndarray] = []
    for a, b in zip(raw, converted):
        if a is None or b is None or a.shape != b.shape:
            continue
        mixed = a.astype(np.float32) * (1.0 - strength) + b.astype(np.float32) * strength
        out.append(np.clip(mixed, 0, 255).astype(np.uint8))
    return out


def check_conversion(
    before: Sequence[np.ndarray], after: Sequence[np.ndarray]
) -> ConversionCheck:
    """Compare the same frames with and without the conversion.

    Returns ``ok=False`` and the name of the measurement that refused it. A
    conversion is never rejected for being *subtle* - only for overshooting.
    """
    stats_before = _picture_stats(before)
    stats_after = _picture_stats(after)
    if stats_before is None or stats_after is None:
        return ConversionCheck(False, "no-frames", stats_before, stats_after)

    gain = (
        stats_after["saturation"] / stats_before["saturation"]
        if stats_before["saturation"] > 1e-3 else 1.0
    )
    if (
        stats_after["saturation"] > CONVERSION_MAX_SATURATION
        and gain > CONVERSION_SATURATION_NOTICEABLE
    ):
        return ConversionCheck(False, "saturation", stats_before, stats_after)
    if gain > CONVERSION_MAX_SATURATION_GAIN:
        return ConversionCheck(False, "gain", stats_before, stats_after)
    if stats_after["clip_high"] - stats_before["clip_high"] > CONVERSION_MAX_NEW_HIGH_CLIP:
        return ConversionCheck(False, "highlights", stats_before, stats_after)
    if stats_after["clip_low"] - stats_before["clip_low"] > CONVERSION_MAX_NEW_LOW_CLIP:
        return ConversionCheck(False, "shadows", stats_before, stats_after)
    if stats_after["luma_span"] < CONVERSION_MIN_SPAN_KEPT * stats_before["luma_span"]:
        return ConversionCheck(False, "flattened", stats_before, stats_after)
    return ConversionCheck(True, "", stats_before, stats_after)


# --------------------------------------------------------------------------
# Log detection
# --------------------------------------------------------------------------


@dataclass
class LogVerdict:
    is_log: bool         # true only on evidence strong enough to act on
    source: str          # flag | sidecar | metadata | filename | bitdepth | statistics | default
    detail: str          # what exactly was seen
    is_a_guess: bool
    statistics: dict | None = None
    suspected: bool = False   # measured as flat, but not acted upon
    #: Which flat profile, when something actually said: dlog | dcinelike |
    #: hlg | None. It matters because a D-Log LUT is wrong for HLG footage.
    profile: str | None = None
    color_mode: dict | None = None   # what the caption sidecar said, verbatim

    def as_dict(self) -> dict:
        return {
            "is_log": self.is_log,
            "profile": self.profile,
            "color_mode": self.color_mode,
            "suspected_flat": self.suspected,
            "source": self.source,
            "detail": self.detail,
            "is_a_guess": self.is_a_guess,
            "statistics": self.statistics,
            "thresholds": {
                "max_luma_span": LOG_STATS_MAX_LUMA_SPAN,
                "max_saturation": LOG_STATS_MAX_SATURATION,
            },
        }


def flatness(samples: Sequence[np.ndarray]) -> dict | None:
    """Mean luma span and mean saturation over a few sample frames.

    These are the two things a log profile does to an ungraded frame: it
    compresses the luma range and desaturates. Measuring them is more
    reliable than hoping the camera wrote a colour tag or the profile name
    into the filename - DJI usually writes neither.
    """
    from .features import dynamic_range, saturation_mean

    usable = [s for s in samples if s is not None and s.size]
    if not usable:
        return None
    spans = [dynamic_range(frame) for frame in usable]
    sats = [saturation_mean(frame) for frame in usable]
    return {
        "luma_span": float(sum(spans) / len(spans)),
        "saturation": float(sum(sats) / len(sats)),
        "frames_measured": len(usable),
    }


def is_dji(clip: ClipInfo) -> bool:
    """Was this written by a DJI camera? Read from the container tags."""
    tags = (clip.extra or {}).get("format_tags") or {}
    encoder = str(tags.get("encoder") or "").lower()
    make = str(tags.get("make") or tags.get("com.apple.quicktime.make") or "").lower()
    return any(hint in encoder or hint in make for hint in DJI_ENCODER_HINTS)


def is_ten_bit(clip: ClipInfo) -> bool:
    return LOG_TENBIT_PIX_FMT_MARK in str(clip.pix_fmt or "")


def detect_log(
    clip: ClipInfo,
    flag: str = "auto",
    stats: dict | None = None,
    color_mode=None,
) -> LogVerdict:
    """Decide whether *clip* needs converting before it looks right.

    Order, strongest evidence first:

    1. ``--convert-log on|off`` - the user said so.
    2. The caption sidecar's ``color_md`` - the camera said so, in words.
    3. A log colour-transfer or wide-primaries tag - rare on DJI, decisive
       when present.
    4. A filename hint.
    5. DJI + a 10-bit pixel format - see :data:`LOG_TENBIT_PIX_FMT_MARK`.
    6. Measured flatness - recorded as *suspicion only*, never acted on.

    Everything below the flag is a guess and says so. *color_mode* comes from
    :func:`framepicker.sidecar.read_color_mode`.
    """
    mode_dict = color_mode.as_dict() if color_mode is not None else None

    if flag == "on":
        return LogVerdict(True, "flag", "--convert-log on", is_a_guess=False,
                          statistics=stats, color_mode=mode_dict,
                          profile=color_mode.kind if color_mode is not None else None)
    if flag == "off":
        return LogVerdict(False, "flag", "--convert-log off", is_a_guess=False,
                          statistics=stats, color_mode=mode_dict)

    # The camera's own word for the profile. Decisive in both directions: this
    # is the one signal that says "this clip is Normal colour, leave it alone"
    # in a card that mixes profiles.
    if color_mode is not None and color_mode.kind != "other":
        return LogVerdict(
            color_mode.is_log, "sidecar",
            f"color_md={color_mode.value} ({os.path.basename(color_mode.source)})",
            is_a_guess=False, statistics=stats,
            profile=color_mode.kind if color_mode.is_log else None,
            color_mode=mode_dict,
        )

    transfer = (clip.color_transfer or "").lower()
    primaries = (clip.color_primaries or "").lower()
    if transfer in LOG_TRANSFER_TAGS:
        profile = "hlg" if transfer == "arib-std-b67" else None
        return LogVerdict(True, "metadata", f"color_transfer={transfer}", is_a_guess=True,
                          statistics=stats, profile=profile, color_mode=mode_dict)
    if primaries == "bt2020" and transfer not in ("bt709", "bt470bg", "smpte170m", "iec61966-2-1"):
        return LogVerdict(
            True, "metadata",
            f"color_primaries={primaries}, color_transfer={transfer or '?'}",
            is_a_guess=True, statistics=stats, color_mode=mode_dict,
        )

    lowered = clip.name.lower()
    for hint in LOG_FILENAME_HINTS:
        if hint in lowered:
            return LogVerdict(True, "filename", f"'{hint}' in {clip.name}", is_a_guess=True,
                              statistics=stats, profile="dlog", color_mode=mode_dict)

    if is_dji(clip) and is_ten_bit(clip):
        tags = (clip.extra or {}).get("format_tags") or {}
        return LogVerdict(
            True, "bitdepth",
            f"encoder={tags.get('encoder') or '?'}, pix_fmt={clip.pix_fmt}",
            is_a_guess=True, statistics=stats, profile="dlog", color_mode=mode_dict,
        )

    if stats is not None:
        flat = (
            stats["luma_span"] < LOG_STATS_MAX_LUMA_SPAN
            and stats["saturation"] < LOG_STATS_MAX_SATURATION
        )
        detail = f"luma_span={stats['luma_span']:.3f}, saturation={stats['saturation']:.3f}"
        # is_log stays False: a flat *scene* is not a flat *profile*, and acting
        # on the difference destroyed real exports. See the constants above.
        return LogVerdict(False, "statistics", detail, is_a_guess=True, statistics=stats,
                          suspected=flat, color_mode=mode_dict)

    return LogVerdict(False, "default", "no log evidence", is_a_guess=True,
                      statistics=stats, color_mode=mode_dict)


# --------------------------------------------------------------------------
# Colour transform used for analysis
# --------------------------------------------------------------------------


#: How much of the normalisation to apply, 0 = none, 1 = full.
NORMALISE_DEFAULT_STRENGTH = 1.0


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
    strength: float = NORMALISE_DEFAULT_STRENGTH

    def as_dict(self) -> dict:
        return {
            "luma_low": self.lo,
            "luma_high": self.hi,
            "saturation_gain": self.saturation_gain,
            "strength": self.strength,
        }

    def apply(self, frame: np.ndarray) -> np.ndarray:
        import cv2

        strength = _clamp(self.strength, 0.0, 1.0)
        if strength <= 0.0:
            return frame
        span = max(self.hi - self.lo, 1e-3)
        work = frame.astype(np.float32)
        work = (work - self.lo) * (255.0 / span)
        np.clip(work, 0, 255, out=work)
        gain = 1.0 + (self.saturation_gain - 1.0) * strength
        if abs(gain - 1.0) > 1e-3:
            hsv = cv2.cvtColor(work.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] *= gain
            np.clip(hsv[:, :, 1], 0, 255, out=hsv[:, :, 1])
            work = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
        # Blend back toward the original so the transform can be dialled down
        # instead of being all-or-nothing.
        if strength < 1.0:
            work = frame.astype(np.float32) * (1.0 - strength) + work * strength
        return np.clip(work, 0, 255).astype(np.uint8)


#: Target mean saturation after normalising a log clip (0-255 scale).
NORMALISE_TARGET_SATURATION = 90.0
#: Hard cap on the saturation multiplier. Was 2.5, which on a dark sunset
#: produced the neon output that made this limit necessary. A real log-to-
#: display conversion is a LUT; this is a fallback and must stay timid.
NORMALISE_MAX_GAIN = 1.6
NORMALISE_LOW_PERCENTILE = 1.0
NORMALISE_HIGH_PERCENTILE = 99.0


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else (high if value > high else float(value))


def estimate_normalisation(
    samples: Sequence[np.ndarray], strength: float = NORMALISE_DEFAULT_STRENGTH
) -> Normalisation | None:
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
    return Normalisation(lo=lo, hi=hi, saturation_gain=gain, strength=strength)


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


#: Strength below which a LUT is not worth applying at all.
LUT_STRENGTH_MIN = 0.05


def lut_graph(lut_path: str, strength: float = 1.0) -> str:
    """The LUT as a filtergraph fragment, at *strength* between 0 and 1.

    At full strength this is just ``lut3d``. Below it, the converted image is
    blended back over the original - which is what "convert it, but less" has
    to mean for a 3D LUT: there is no dial inside a cube file.

    Partial strength exists because a LUT can be too strong for the footage it
    is pointed at, and measurably was: on Tomas's D-Log clips his D-Log M cube
    took mean saturation 0.210 -> 0.384 and pushed 7 % of every frame to black,
    while the same cube at 0.5 gave 0.280 and 0.7 %.
    """
    lut = f"lut3d=file='{escape_filter_path(lut_path)}'"
    strength = float(min(1.0, max(0.0, strength)))
    if strength >= 1.0:
        return lut
    # [top]=converted over [bottom]=original; all_opacity is the top layer's.
    return (
        f"split[fp_a][fp_b];[fp_b]{lut}[fp_l];"
        f"[fp_l][fp_a]blend=all_mode=normal:all_opacity={strength:.4f}"
    )


def build_video_filter(
    out_w: int,
    out_h: int,
    fps: float | None,
    lut_path: str | None,
    lut_strength: float = 1.0,
) -> str:
    parts: list[str] = []
    if fps:
        parts.append(f"fps={fps}")
    parts.append(f"scale={out_w}:{out_h}:flags=area")
    if lut_path and lut_strength > LUT_STRENGTH_MIN:
        # format first: blend needs both branches in the same pixel format.
        parts.append("format=rgb24")
        parts.append(lut_graph(lut_path, lut_strength))
    parts.append("format=rgb24")
    return ",".join(parts)


#: GPU scalers, tried in this order. Both ship with common CUDA builds;
#: neither is assumed to exist.
GPU_SCALERS = ("scale_cuda", "scale_npp")


def build_gpu_video_filter(
    out_w: int,
    out_h: int,
    fps: float | None,
    lut_path: str | None,
    scaler: str = "scale_cuda",
    lut_strength: float = 1.0,
) -> str:
    """The same chain, but the frame stays on the GPU until it is small.

    ``-hwaccel cuda`` on its own decodes on the GPU and then copies every
    full-size frame back to system memory - for 4K 10-bit at 60 fps that copy
    is the bulk of the work, and 59 of every 60 frames are thrown away
    immediately afterwards. Dropping frames (``fps``) and scaling
    (``scale_cuda``) before ``hwdownload`` moves both in front of the copy.

    Whether this build supports it is *tested*, never assumed - see
    :func:`gpu_scale_available`.
    """
    parts: list[str] = []
    if fps:
        parts.append(f"fps={fps}")
    # Named options on purpose: scale_npp rejects positional w:h ("No option
    # name near '640:360'" in a real run), while scale_cuda accepts both.
    parts.append(f"{scaler}=w={out_w}:h={out_h}")
    parts.append("hwdownload")
    parts.append("format=nv12")
    if lut_path and lut_strength > LUT_STRENGTH_MIN:
        parts.append("format=rgb24")
        parts.append(lut_graph(lut_path, lut_strength))
    parts.append("format=rgb24")
    return ",".join(parts)


def gpu_scale_available(
    path: str, hwaccel: str, out_w: int, out_h: int, scaler: str = "scale_cuda"
) -> tuple[bool, str]:
    """Decode one frame through the GPU-scaling chain and look at the result.

    Same premise-checking rule as :func:`hwaccel_available` (TRAP-12): the
    chain that will be used is the chain that is tested, and a failure falls
    back to the plain path instead of breaking the run.
    """
    if proc.executable(FFMPEG) is None:
        return False, f"{FFMPEG} not on PATH"
    argv = [
        FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error",
        "-hwaccel", hwaccel,
        "-hwaccel_output_format", "cuda",
        "-i", path,
        "-frames:v", "1",
        "-vf", build_gpu_video_filter(out_w, out_h, None, None, scaler),
        "-f", "null", "-",
    ]
    try:
        result = proc.run(argv, timeout=60)
    except proc.ProcessError as exc:
        return False, str(exc)
    if result.ok:
        return True, ""
    return False, result.stderr_text(200) or f"exit {result.returncode}"


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
    #: Name of the GPU scaler actually used, "" when the frames came back to
    #: system memory at full size (which is what -hwaccel alone does).
    gpu_scaler: str = ""
    gpu_scale_error: str = ""
    keyframes_only: bool = False
    #: Set when the analysis ran on a proxy file instead of the master.
    proxy_file: str = ""
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "path_used": self.path_used,
            "hw_error": self.hw_error,
            "gpu_scaler": self.gpu_scaler,
            "gpu_scale_error": self.gpu_scale_error,
            "keyframes_only": self.keyframes_only,
            "proxy_file": self.proxy_file,
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
    lut_strength: float = 1.0,
    hwaccel: str = "auto",
    max_candidates: int | None = None,
    keyframes_only: bool = False,
    gpu_scale: bool = True,
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
    vf = build_video_filter(out_w, out_h, effective_fps, lut_path, lut_strength)

    use_hw = False
    name = "cuda" if hwaccel == "auto" else hwaccel
    if hwaccel and hwaccel != "none":
        use_hw, why = hwaccel_available(clip.path, name)
        if not use_hw:
            report.hw_error = why
    report.path_used = "hw" if use_hw else "cpu"

    scaler = ""
    if use_hw and gpu_scale:
        for candidate in GPU_SCALERS:
            ok, why = gpu_scale_available(clip.path, name, out_w, out_h, candidate)
            if ok:
                scaler = candidate
                break
            report.gpu_scale_error = why
    report.gpu_scaler = scaler
    if scaler:
        vf = build_gpu_video_filter(out_w, out_h, effective_fps, lut_path, scaler, lut_strength)

    argv = [FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-nostats"]
    if use_hw:
        argv += ["-hwaccel", name]
        if scaler:
            argv += ["-hwaccel_output_format", "cuda"]
    if keyframes_only:
        # Only intra frames are decoded. Far less work, and the frames that
        # survive are the highest-quality ones in the file - but the sampling
        # grid becomes the camera's keyframe interval, so the report has to
        # say what was actually sampled instead of what was asked for.
        argv += ["-skip_frame", "nokey"]
    argv += [
        "-i", clip.path,
        "-an", "-sn", "-dn",
        "-vf", vf,
        "-fps_mode", "passthrough",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-",
    ]
    report.keyframes_only = keyframes_only

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
    lut_strength: float = 1.0,
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
        lut_strength=lut_strength,
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
