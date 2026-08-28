"""Full-resolution extraction of the chosen timestamps.

The proxy frames used for analysis are 640 px wide; nothing is exported from
them. Every output image is cut out of the source file at the timestamp that
was scored.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from . import proc
from . import strings_lt as S
from .decode import Normalisation, escape_filter_path

FFMPEG = "ffmpeg"

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


class ExportError(RuntimeError):
    pass


@dataclass
class ExportResult:
    path: str
    timestamp: float
    ok: bool
    detail: str = ""


def safe_stem(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name))[0]
    stem = _UNSAFE.sub("_", stem).strip("_")
    return stem or "clip"


def output_name(clip_name: str, rank: int, timestamp: float, score: float, suffix: str = "jpg") -> str:
    """``<clipname>_<index>_<t_seconds>_<score>.<ext>`` - sorts by rank."""
    return f"{safe_stem(clip_name)}_{rank:02d}_{timestamp:08.3f}s_{score:.3f}.{suffix}"


def export_frame(
    src: str,
    timestamp: float,
    out_path: str,
    *,
    lut_path: str | None = None,
    normalisation: Normalisation | None = None,
    quality: int = 2,
    image_format: str = "jpg",
) -> ExportResult:
    """Cut one frame out of *src* at *timestamp* and write it to *out_path*.

    If a LUT or a normalisation was used for analysis, the same transform is
    applied here, so what you look at is what was scored. The untouched
    timestamp stays in ``results.json`` for a full-quality re-grab later.
    """
    if proc.executable(FFMPEG) is None:
        return ExportResult(out_path, timestamp, False, f"{FFMPEG} not on PATH")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    argv = [
        FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-nostats", "-y",
        "-ss", f"{max(0.0, float(timestamp)):.6f}",
        "-i", src,
        "-map", "0:v:0",
        "-frames:v", "1",
    ]
    if lut_path:
        argv += ["-vf", f"lut3d=file='{escape_filter_path(lut_path)}'"]
    if image_format.lower() in ("jpg", "jpeg"):
        argv += ["-q:v", str(int(quality))]
    argv += [out_path]

    try:
        result = proc.run(argv, timeout=180)
    except proc.ProcessError as exc:
        return ExportResult(out_path, timestamp, False, str(exc))
    if not result.ok or not os.path.isfile(out_path):
        return ExportResult(out_path, timestamp, False, result.stderr_text() or f"exit {result.returncode}")

    if normalisation is not None:
        applied = _apply_normalisation_in_place(out_path, normalisation, quality, image_format)
        if applied is not None:
            return ExportResult(out_path, timestamp, False, applied)
    return ExportResult(out_path, timestamp, True)


def _apply_normalisation_in_place(
    path: str, normalisation: Normalisation, quality: int, image_format: str
) -> str | None:
    """Re-write *path* with the analysis transform applied. Returns an error or None."""
    import cv2
    import numpy as np

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        return "written file could not be re-read"
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    converted = normalisation.apply(rgb)
    bgr = cv2.cvtColor(np.ascontiguousarray(converted), cv2.COLOR_RGB2BGR)
    if image_format.lower() in ("jpg", "jpeg"):
        # ffmpeg -q:v 2 is roughly JPEG quality 90-95; keep the rewrite close.
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    else:
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    if not cv2.imwrite(path, bgr, params):
        return "re-encode failed"
    return None


def export_selection(
    src: str,
    clip_name: str,
    chosen,
    out_dir: str,
    *,
    lut_path: str | None = None,
    normalisation: Normalisation | None = None,
    quality: int = 2,
    image_format: str = "jpg",
    cancel=None,
) -> tuple[list[ExportResult], list[str]]:
    """Export every chosen candidate. Returns ``(results, error_messages)``."""
    results: list[ExportResult] = []
    errors: list[str] = []
    for rank, candidate in enumerate(chosen, start=1):
        if cancel is not None and cancel.is_set():
            break
        name = output_name(clip_name, rank, candidate.t, candidate.score, image_format)
        result = export_frame(
            src,
            candidate.t,
            os.path.join(out_dir, name),
            lut_path=lut_path,
            normalisation=normalisation,
            quality=quality,
            image_format=image_format,
        )
        results.append(result)
        if not result.ok:
            errors.append(S.export_failed(candidate.t, result.detail))
    return results, errors
