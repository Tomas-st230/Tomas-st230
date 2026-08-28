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
from . import grading
from . import decode
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


def build_export_filter(lut_path: str | None, height: int = 0,
                        lut_strength: float = 1.0) -> str | None:
    """Filter chain for the exported still, or ``None`` when nothing is needed.

    *height* of 0 keeps the source resolution - 4K stays 4K, 2.7K stays 2.7K.
    A non-zero *height* only ever scales **down**: a 1080p source asked for
    1080p is left alone rather than upscaled.
    """
    parts: list[str] = []
    if lut_path and lut_strength > decode.LUT_STRENGTH_MIN:
        # The same fragment the analysis uses, so what was measured is what
        # gets written.
        parts.append(decode.lut_graph(lut_path, lut_strength))
    if height and height > 0:
        parts.append(f"scale=-2:'min(ih,{int(height)})':flags=lanczos")
    return ",".join(parts) if parts else None


def export_frame(
    src: str,
    timestamp: float,
    out_path: str,
    *,
    lut_path: str | None = None,
    normalisation: Normalisation | None = None,
    quality: int = 2,
    image_format: str = "jpg",
    height: int = 0,
    lut_strength: float = 1.0,
    look: grading.Look | None = None,
    look_strength: float = grading.DEFAULT_STRENGTH,
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
    video_filter = build_export_filter(lut_path, height, lut_strength)
    if video_filter:
        argv += ["-vf", video_filter]
    if image_format.lower() in ("jpg", "jpeg"):
        argv += ["-q:v", str(int(quality))]
    argv += [out_path]

    try:
        result = proc.run(argv, timeout=180)
    except proc.ProcessError as exc:
        return ExportResult(out_path, timestamp, False, str(exc))
    if not result.ok or not os.path.isfile(out_path):
        return ExportResult(out_path, timestamp, False, result.stderr_text() or f"exit {result.returncode}")

    if normalisation is not None or look is not None:
        applied = _post_process_in_place(
            out_path, normalisation, look, look_strength, quality, image_format
        )
        if applied is not None:
            return ExportResult(out_path, timestamp, False, applied)
    return ExportResult(out_path, timestamp, True)


def _post_process_in_place(
    path: str,
    normalisation: Normalisation | None,
    look: grading.Look | None,
    look_strength: float,
    quality: int,
    image_format: str,
) -> str | None:
    """Re-write *path* with the colour conversion and/or the look applied.

    Order matters: the conversion (LUT fallback) is a correction and comes
    first; the look is a taste decision and comes last, on top of a corrected
    image.
    """
    import cv2
    import numpy as np

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        return "written file could not be re-read"
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if normalisation is not None:
        rgb = normalisation.apply(rgb)
    if look is not None:
        rgb, _info = grading.apply_look(rgb, look, look_strength)
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
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
    height: int = 0,
    lut_strength: float = 1.0,
    look: grading.Look | None = None,
    look_strength: float = grading.DEFAULT_STRENGTH,
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
            lut_strength=lut_strength,
            normalisation=normalisation,
            quality=quality,
            image_format=image_format,
            height=height,
            look=look,
            look_strength=look_strength,
        )
        results.append(result)
        if not result.ok:
            errors.append(S.export_failed(candidate.t, result.detail))
    return results, errors
