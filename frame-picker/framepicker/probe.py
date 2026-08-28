"""``ffprobe`` wrapper.

Returns everything the rest of the pipeline needs to know about a file before
a single frame is decoded. A file that cannot be probed raises
:class:`ProbeError`; the caller skips it and keeps going (rule 9.1: one bad
file never aborts a batch).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from fractions import Fraction

from . import proc

FFPROBE = "ffprobe"


class ProbeError(RuntimeError):
    """The file could not be probed, or carries no usable video stream."""


@dataclass
class ClipInfo:
    path: str
    name: str
    duration: float | None
    fps: float | None
    width: int | None
    height: int | None
    codec: str | None
    pix_fmt: str | None
    color_transfer: str | None
    color_primaries: str | None
    color_space: str | None
    color_range: str | None
    nb_frames: int | None
    size_bytes: int | None
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def _to_float(value) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def _rate_to_float(value) -> float | None:
    if not value or value in ("0/0", "N/A"):
        return None
    try:
        fraction = Fraction(str(value))
    except (ValueError, ZeroDivisionError):
        return _to_float(value)
    if fraction.denominator == 0:
        return None
    return float(fraction)


def probe(path: str | os.PathLike[str]) -> ClipInfo:
    """Probe *path* with ffprobe. Raises :class:`ProbeError` on any problem."""
    path = os.fspath(path)
    if not os.path.isfile(path):
        raise ProbeError("file not found")
    if proc.executable(FFPROBE) is None:
        raise ProbeError(f"{FFPROBE} not on PATH")

    argv = [
        FFPROBE,
        "-v", "error",
        "-show_streams",
        "-show_format",
        "-of", "json",
        path,
    ]
    try:
        result = proc.run(argv, timeout=120)
    except proc.ProcessError as exc:
        raise ProbeError(str(exc)) from exc
    if not result.ok:
        raise ProbeError(result.stderr_text() or f"ffprobe exit {result.returncode}")
    try:
        payload = json.loads(result.stdout.decode("utf-8", "replace"))
    except json.JSONDecodeError as exc:
        raise ProbeError(f"ffprobe output is not JSON: {exc}") from exc

    streams = [s for s in payload.get("streams", []) if s.get("codec_type") == "video"]
    if not streams:
        raise ProbeError("no video stream")
    video = streams[0]
    fmt = payload.get("format", {})

    duration = _to_float(video.get("duration")) or _to_float(fmt.get("duration"))
    fps = _rate_to_float(video.get("avg_frame_rate")) or _rate_to_float(video.get("r_frame_rate"))
    nb_frames = None
    try:
        nb_frames = int(video["nb_frames"])
    except (KeyError, TypeError, ValueError):
        if duration is not None and fps:
            nb_frames = int(round(duration * fps))

    if duration is None and nb_frames and fps:
        duration = nb_frames / fps

    width = video.get("width")
    height = video.get("height")
    if not width or not height:
        raise ProbeError("video stream has no dimensions")

    size_bytes = None
    try:
        size_bytes = int(fmt["size"])
    except (KeyError, TypeError, ValueError):
        try:
            size_bytes = os.path.getsize(path)
        except OSError:
            size_bytes = None

    return ClipInfo(
        path=path,
        name=os.path.basename(path),
        duration=duration,
        fps=fps,
        width=int(width),
        height=int(height),
        codec=video.get("codec_name"),
        pix_fmt=video.get("pix_fmt"),
        color_transfer=video.get("color_transfer"),
        color_primaries=video.get("color_primaries"),
        color_space=video.get("color_space"),
        color_range=video.get("color_range"),
        nb_frames=nb_frames,
        size_bytes=size_bytes,
        extra={
            "format_name": fmt.get("format_name"),
            "bit_rate": fmt.get("bit_rate"),
            "profile": video.get("profile"),
            "field_order": video.get("field_order"),
            "tags": video.get("tags", {}),
        },
    )
