"""Per-frame measurements.

The feature functions are pure: ``ndarray -> float``. No I/O, no globals, no
hidden state — that is what makes the scoring testable without a pipeline.

The two detector loaders at the bottom are the exception; they touch the
filesystem and the optional opencv-contrib package. They return ``None`` when
a detector is unavailable, and the caller then keeps the corresponding
features as ``None``. ``None`` and ``0.0`` mean different things: a frame with
no detector is not a frame with no faces.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np

#: Rule-of-thirds intersections in normalised frame coordinates.
THIRDS_POINTS = ((1 / 3, 1 / 3), (2 / 3, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 2 / 3))

#: Distance from a frame corner to the nearest thirds point; used to
#: normalise ``thirds_distance`` into 0..1.
MAX_THIRDS_DISTANCE = float(np.hypot(1 / 3, 1 / 3))

CLIP_LOW_LEVEL = 5
CLIP_HIGH_LEVEL = 250

#: Where a YuNet model is looked for when the user gives no explicit path.
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

#: Filenames published by opencv/opencv_zoo, newest first. Verified against
#: the repository, not guessed (see README).
YUNET_FILENAMES = (
    "face_detection_yunet_2026may.onnx",
    "face_detection_yunet_2023mar.onnx",
    "face_detection_yunet_2023mar_int8bq.onnx",
)

FEATURE_KEYS = (
    "sharpness",
    "exposure_clip_low",
    "exposure_clip_high",
    "dynamic_range",
    "colorfulness",
    "saturation_mean",
    "face_count",
    "face_max_rel",
    "subject_rel",
    "subject_cx",
    "subject_cy",
    "thirds_distance",
)


# --------------------------------------------------------------------------
# Pure measurements
# --------------------------------------------------------------------------


def to_luma(frame: np.ndarray) -> np.ndarray:
    """Rec.601 luma plane of an RGB uint8 frame, as float32 0..255."""
    arr = np.asarray(frame)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("expected an RGB frame")
    return arr[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def sharpness(frame: np.ndarray) -> float:
    """Variance of the Laplacian on the luma plane. Higher = sharper."""
    import cv2

    luma = to_luma(frame)
    return float(cv2.Laplacian(luma, cv2.CV_32F).var())


def exposure_clip_low(frame: np.ndarray) -> float:
    """Fraction of luma pixels below 5/255."""
    luma = to_luma(frame)
    return float(np.count_nonzero(luma < CLIP_LOW_LEVEL) / luma.size)


def exposure_clip_high(frame: np.ndarray) -> float:
    """Fraction of luma pixels above 250/255."""
    luma = to_luma(frame)
    return float(np.count_nonzero(luma > CLIP_HIGH_LEVEL) / luma.size)


def dynamic_range(frame: np.ndarray) -> float:
    """Luma p99 - p01, normalised to 0..1."""
    luma = to_luma(frame)
    lo, hi = np.percentile(luma, (1.0, 99.0))
    return float(max(0.0, (hi - lo)) / 255.0)


def colorfulness(frame: np.ndarray) -> float:
    """Hasler & Suesstrunk (2003) colourfulness metric.

    ``M = sqrt(sd_rg^2 + sd_yb^2) + 0.3 * sqrt(mean_rg^2 + mean_yb^2)``
    on ``rg = R - G`` and ``yb = 0.5*(R + G) - B``. Not normalised; the
    pipeline converts it to a within-clip percentile rank before scoring.
    """
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("expected an RGB frame")
    red = arr[..., 0].astype(np.float32)
    green = arr[..., 1].astype(np.float32)
    blue = arr[..., 2].astype(np.float32)
    rg = red - green
    yb = 0.5 * (red + green) - blue
    std = float(np.hypot(rg.std(), yb.std()))
    mean = float(np.hypot(rg.mean(), yb.mean()))
    return std + 0.3 * mean


def saturation_mean(frame: np.ndarray) -> float:
    """Mean HSV saturation, normalised to 0..1."""
    import cv2

    arr = np.ascontiguousarray(np.asarray(frame, dtype=np.uint8))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    return float(hsv[..., 1].mean() / 255.0)


def thirds_distance(cx: float, cy: float) -> float:
    """Normalised distance from ``(cx, cy)`` to the nearest thirds point.

    ``cx``/``cy`` are in 0..1 frame coordinates. 0 means dead on a thirds
    intersection, 1 means as far from any of them as a frame corner is.
    """
    best = min(float(np.hypot(cx - px, cy - py)) for px, py in THIRDS_POINTS)
    return float(min(1.0, best / MAX_THIRDS_DISTANCE))


def percentile_ranks(values: Sequence[float]) -> list[float]:
    """Within-clip percentile rank of every value, in 0..1.

    Absolute thresholds are wrong across cameras, lenses and picture
    profiles, so sharpness, dynamic range and colourfulness are ranked inside
    their own clip before scoring. Ties get the same rank.
    """
    array = np.asarray(list(values), dtype=np.float64)
    n = array.size
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    # Average the ranks of tied values so identical frames rank identically.
    unique, inverse, counts = np.unique(array, return_inverse=True, return_counts=True)
    sums = np.zeros(unique.size, dtype=np.float64)
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    return [float(r / (n - 1)) for r in ranks]


def dhash(frame: np.ndarray, size: int = 8) -> int:
    """Difference hash of a frame, as an integer of ``size*size`` bits.

    Implemented here on purpose: it is fifteen lines and does not justify a
    dependency.
    """
    import cv2

    luma = to_luma(frame)
    small = cv2.resize(luma, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return int(bin(a ^ b).count("1"))


def color_histogram(frame: np.ndarray, bins: int = 8) -> np.ndarray:
    """Normalised 3D RGB histogram, flattened. Used for diversity checks."""
    arr = np.asarray(frame, dtype=np.uint8).reshape(-1, 3)
    idx = (arr >> (8 - int(np.log2(bins)))).astype(np.int32)
    flat = (idx[:, 0] * bins + idx[:, 1]) * bins + idx[:, 2]
    hist = np.bincount(flat, minlength=bins ** 3).astype(np.float64)
    total = hist.sum()
    return hist / total if total else hist


def histogram_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Total-variation distance between two normalised histograms, 0..1."""
    return float(0.5 * np.abs(np.asarray(a) - np.asarray(b)).sum())


# --------------------------------------------------------------------------
# Optional detectors
# --------------------------------------------------------------------------


@dataclass
class DetectorStatus:
    available: bool
    detail: str = ""


def find_face_model(explicit: str | None = None) -> str | None:
    """Locate a YuNet ONNX model, or return ``None``."""
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    for name in YUNET_FILENAMES:
        candidate = os.path.join(MODEL_DIR, name)
        if os.path.isfile(candidate):
            return candidate
    if os.path.isdir(MODEL_DIR):
        for entry in sorted(os.listdir(MODEL_DIR)):
            if entry.lower().endswith(".onnx") and "yunet" in entry.lower():
                return os.path.join(MODEL_DIR, entry)
    return None


class FaceDetector:
    """Thin wrapper over ``cv2.FaceDetectorYN``."""

    def __init__(self, model_path: str, score_threshold: float = 0.6) -> None:
        import cv2

        self.model_path = model_path
        self._detector = cv2.FaceDetectorYN.create(model_path, "", (320, 320), score_threshold, 0.3, 5000)
        self._size: tuple[int, int] | None = None

    def detect(self, frame: np.ndarray) -> tuple[int, float]:
        """Return ``(face_count, largest_face_area / frame_area)``."""
        import cv2

        arr = np.ascontiguousarray(np.asarray(frame, dtype=np.uint8))
        height, width = arr.shape[:2]
        if self._size != (width, height):
            self._detector.setInputSize((width, height))
            self._size = (width, height)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        _, faces = self._detector.detect(bgr)
        if faces is None or len(faces) == 0:
            return 0, 0.0
        frame_area = float(width * height)
        areas = [float(max(0.0, f[2]) * max(0.0, f[3])) / frame_area for f in faces]
        return len(faces), float(max(areas))


def load_face_detector(model_path: str | None = None, enabled: bool = True) -> tuple[FaceDetector | None, DetectorStatus]:
    """Build a face detector, or explain why there is none."""
    if not enabled:
        return None, DetectorStatus(False, "disabled")
    path = find_face_model(model_path)
    if path is None:
        return None, DetectorStatus(False, "model-missing")
    try:
        return FaceDetector(path), DetectorStatus(True, path)
    except Exception as exc:  # noqa: BLE001 - any OpenCV failure means no faces
        return None, DetectorStatus(False, f"{type(exc).__name__}: {exc}")


class SaliencyBackend:
    """Static spectral-residual saliency from opencv-contrib."""

    def __init__(self) -> None:
        import cv2

        self._impl = cv2.saliency.StaticSaliencySpectralResidual_create()

    def subject(self, frame: np.ndarray) -> tuple[float, float, float] | None:
        """Return ``(area_fraction, cx, cy)`` of the largest salient blob."""
        import cv2

        arr = np.ascontiguousarray(np.asarray(frame, dtype=np.uint8))
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        ok, saliency_map = self._impl.computeSaliency(bgr)
        if not ok or saliency_map is None:
            return None
        scaled = np.clip(saliency_map * 255.0, 0, 255).astype(np.uint8)
        _, mask = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count <= 1:
            return None
        areas = stats[1:, cv2.CC_STAT_AREA]
        biggest = int(np.argmax(areas)) + 1
        height, width = mask.shape[:2]
        area_fraction = float(stats[biggest, cv2.CC_STAT_AREA]) / float(width * height)
        cx = float(centroids[biggest][0]) / float(width)
        cy = float(centroids[biggest][1]) / float(height)
        return area_fraction, cx, cy


def load_saliency(enabled: bool = True) -> tuple[SaliencyBackend | None, DetectorStatus]:
    if not enabled:
        return None, DetectorStatus(False, "disabled")
    try:
        return SaliencyBackend(), DetectorStatus(True, "opencv-contrib")
    except Exception as exc:  # noqa: BLE001 - contrib absent or build without saliency
        return None, DetectorStatus(False, f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------
# One frame -> one feature dict
# --------------------------------------------------------------------------


def extract(
    frame: np.ndarray,
    *,
    face_detector: FaceDetector | None = None,
    saliency: SaliencyBackend | None = None,
) -> dict:
    """Measure one frame. Unavailable detectors leave their keys at ``None``."""
    result: dict = {
        "sharpness": sharpness(frame),
        "exposure_clip_low": exposure_clip_low(frame),
        "exposure_clip_high": exposure_clip_high(frame),
        "dynamic_range": dynamic_range(frame),
        "colorfulness": colorfulness(frame),
        "saturation_mean": saturation_mean(frame),
        "face_count": None,
        "face_max_rel": None,
        "subject_rel": None,
        "subject_cx": None,
        "subject_cy": None,
        "thirds_distance": None,
    }
    if face_detector is not None:
        count, max_rel = face_detector.detect(frame)
        result["face_count"] = count
        result["face_max_rel"] = max_rel
    if saliency is not None:
        subject = saliency.subject(frame)
        if subject is not None:
            rel, cx, cy = subject
            result["subject_rel"] = rel
            result["subject_cx"] = cx
            result["subject_cy"] = cy
            result["thirds_distance"] = thirds_distance(cx, cy)
    return result
