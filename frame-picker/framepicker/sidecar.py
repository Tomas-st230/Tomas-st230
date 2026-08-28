"""The files a camera writes *next to* the video, and what can be read from them.

Two of them matter for DJI footage, and both were found by opening real files
from Tomas's card rather than by guessing:

``NAME.SRT``
    DJI's "Video Captions". One caption per frame, carrying the camera
    settings - including ``[color_md: ...]``, which is the **only** place the
    picture profile is stated in plain text. The video stream itself is tagged
    ``bt709`` whatever profile was used, which is exactly why D-Log went
    unnoticed.

``NAME.LRF``
    A 720p proxy of the same take, recorded at the same time. Measured on
    ``DJI_20260822191538_0243_D.LRF``: 1280x720 H.264, 8.0 Mbit/s, 29.97 fps,
    20.354 s, plus a ``djmd`` data stream (a binary ``dvtm_Lito_X1.proto``
    protobuf - telemetry, no plain text) and a 960x540 MJPEG cover thumbnail.
    The proxy is worth having because analysis runs at 640 px anyway, so
    decoding 8 Mbit of 720p instead of 4K HEVC 10-bit costs far less for the
    same frames. Exports are always taken from the master file.

Nothing here calls ffmpeg directly; the proxy is checked with
:func:`framepicker.probe.probe`.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from .probe import ClipInfo, ProbeError, probe

#: Extension DJI uses for the low-resolution proxy.
PROXY_SUFFIX = ".lrf"
#: Extension for the caption sidecar.
CAPTION_SUFFIX = ".srt"

#: ``[color_md: dlog_m]``, ``[color_md : default]``, both seen in the wild.
_COLOR_MD = re.compile(r"\[\s*color_md\s*[:=]\s*([^\]]+)\]", re.IGNORECASE)
#: Bytes of the caption file read before giving up. The field repeats every
#: frame, so the answer is in the first caption or it is not there at all.
CAPTION_READ_BYTES = 64 * 1024

#: What a ``color_md`` value means for the conversion. Compared after every
#: non-alphanumeric character is stripped, because DJI writes ``d-log``,
#: ``dlog_m`` and ``d_log_m`` for the same thing depending on the model.
COLOR_MODE_KINDS = (
    ("dlog", "dlog"),
    ("dcinelike", "dcinelike"),
    ("dcinlike", "dcinelike"),
    ("hlg", "hlg"),
    ("default", "default"),
    ("normal", "default"),
    ("none", "default"),
)


@dataclass
class ColorMode:
    """What the caption file says the picture profile was."""

    value: str          # raw text, exactly as written
    kind: str           # dlog | dcinelike | hlg | default | other
    source: str         # the file it was read from
    values_seen: int    # distinct color_md values in the part of the file read

    @property
    def is_log(self) -> bool:
        """Does this profile need converting before it looks right?"""
        return self.kind in ("dlog", "dcinelike", "hlg")

    def as_dict(self) -> dict:
        return {
            "value": self.value,
            "kind": self.kind,
            "is_log": self.is_log,
            "source": os.path.basename(self.source),
            "values_seen": self.values_seen,
        }


def _sibling(video_path: str, suffix: str) -> str | None:
    """A file with the same stem and *suffix*, matched case-insensitively.

    A card copied off a drone has ``DJI_0001.MP4`` next to ``DJI_0001.LRF``,
    but Windows, macOS and Linux disagree about case, so the directory is
    listed instead of a name being constructed.
    """
    folder = os.path.dirname(os.path.abspath(video_path))
    stem = os.path.splitext(os.path.basename(video_path))[0].lower()
    try:
        entries = os.listdir(folder)
    except OSError:
        return None
    for entry in entries:
        base, ext = os.path.splitext(entry)
        if base.lower() == stem and ext.lower() == suffix:
            candidate = os.path.join(folder, entry)
            if os.path.isfile(candidate):
                return candidate
    return None


def find_captions(video_path: str) -> str | None:
    return _sibling(video_path, CAPTION_SUFFIX)


def find_proxy(video_path: str) -> str | None:
    return _sibling(video_path, PROXY_SUFFIX)


def classify_color_mode(value: str) -> str:
    """Map a raw ``color_md`` value onto what has to be done about it."""
    flat = re.sub(r"[^a-z0-9]", "", str(value).lower())
    for needle, kind in COLOR_MODE_KINDS:
        if needle in flat:
            return kind
    return "other"


def read_color_mode(video_path: str) -> ColorMode | None:
    """Read ``color_md`` out of the caption sidecar, or ``None``.

    ``None`` covers every way this can come up empty - no sidecar, captions
    turned off on the camera, an older model that does not write the field -
    and the caller must treat it as "unknown", never as "normal colour".
    """
    path = find_captions(video_path)
    if path is None:
        return None
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            head = handle.read(CAPTION_READ_BYTES)
    except OSError:
        return None
    matches = [m.group(1).strip() for m in _COLOR_MD.finditer(head)]
    if not matches:
        return None
    distinct = list(dict.fromkeys(matches))
    return ColorMode(
        value=distinct[0],
        kind=classify_color_mode(distinct[0]),
        source=path,
        values_seen=len(distinct),
    )


@dataclass
class Proxy:
    """A proxy file and whether it may stand in for the master."""

    path: str
    usable: bool
    detail: str
    width: int | None = None
    height: int | None = None
    duration: float | None = None

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "file": os.path.basename(self.path),
            "usable": self.usable,
            "detail": self.detail,
            "width": self.width,
            "height": self.height,
            "duration": self.duration,
        }


#: Seconds the proxy may differ from the master before it is refused. A proxy
#: that is not the same take, or that stops early, would move every timestamp.
PROXY_MAX_DURATION_DRIFT_S = 0.5
#: ...or this share of the master's duration, whichever is larger.
PROXY_MAX_DURATION_DRIFT_SHARE = 0.02
#: Aspect-ratio difference allowed between proxy and master.
PROXY_MAX_ASPECT_DRIFT = 0.02


def check_proxy(clip: ClipInfo, path: str, min_long_edge: int = 640) -> Proxy:
    """Is *path* the same take as *clip*, and big enough to analyse?

    Every refusal names its reason: a proxy silently ignored, or silently
    used, would make the report say something that is not true.
    """
    try:
        proxy_info = probe(path)
    except ProbeError as exc:
        return Proxy(path, False, f"probe failed: {exc}")

    long_edge = max(int(proxy_info.width or 0), int(proxy_info.height or 0))
    if long_edge < min_long_edge:
        return Proxy(path, False, f"long edge {long_edge} px < {min_long_edge} px",
                     proxy_info.width, proxy_info.height, proxy_info.duration)

    if clip.duration and proxy_info.duration:
        allowed = max(PROXY_MAX_DURATION_DRIFT_S,
                      PROXY_MAX_DURATION_DRIFT_SHARE * float(clip.duration))
        drift = abs(float(proxy_info.duration) - float(clip.duration))
        if drift > allowed:
            return Proxy(path, False, f"duration differs by {drift:.3f} s (max {allowed:.3f} s)",
                         proxy_info.width, proxy_info.height, proxy_info.duration)

    if clip.width and clip.height and proxy_info.width and proxy_info.height:
        master_aspect = float(clip.width) / float(clip.height)
        proxy_aspect = float(proxy_info.width) / float(proxy_info.height)
        if abs(master_aspect - proxy_aspect) / master_aspect > PROXY_MAX_ASPECT_DRIFT:
            return Proxy(path, False,
                         f"aspect {proxy_aspect:.4f} vs {master_aspect:.4f}",
                         proxy_info.width, proxy_info.height, proxy_info.duration)

    return Proxy(path, True, f"{proxy_info.width}x{proxy_info.height}",
                 proxy_info.width, proxy_info.height, proxy_info.duration)
