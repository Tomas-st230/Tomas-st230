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
    "subject_separation",
    "thirds_distance",
    "horizon_tilt",
    "motion",
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


def motion(previous: np.ndarray | None, current: np.ndarray) -> float | None:
    """How much the frame moved since the previous sample, 0..1.

    Mean absolute luma difference between two consecutive sampled frames. At
    the default 2 samples per second this reads the *shot*: a locked-off shot
    sits near zero, an orbit or a pull-back reads high. ``None`` for the first
    frame of a clip, which has nothing to compare against - and ``None`` is
    not zero.
    """
    if previous is None:
        return None
    a = to_luma(previous)
    b = to_luma(current)
    if a.shape != b.shape:
        return None
    return float(np.abs(a - b).mean() / 255.0)


#: Angle, in degrees from horizontal, at which a tilted horizon counts as
#: fully wrong. Beyond this the penalty stops growing.
TILT_FULL_DEGREES = 8.0
#: Only lines within this many degrees of horizontal are candidate horizons.
TILT_SEARCH_DEGREES = 30.0
#: If the strongest candidate lines disagree by more than this, the frame has
#: no horizon and the measurement is withheld rather than invented.
TILT_MAX_SPREAD_DEGREES = 3.0
#: ...and they must sit at the same height, within this fraction of the frame.
TILT_MAX_SPREAD_ROWS = 0.06
#: ...and the two sides of the line must differ in brightness by at least this
#: much (0..1). A horizon separates sky from ground; random texture does not,
#: and without this check uniform noise reports a perfectly level horizon.
TILT_MIN_CONTRAST = 0.05


def horizon_tilt(frame: np.ndarray) -> float | None:
    """How far off level the dominant near-horizontal line is, 0..1.

    A crooked horizon is the one composition defect in drone footage that is
    both unambiguous and cheap to measure, so it is measured rather than
    guessed. ``None`` when the frame has no near-horizontal line long enough
    to judge - a top-down shot, for instance - because "no horizon" is not the
    same as "a level one", and a missing measurement must never score as a
    good one.
    """
    import cv2

    luma = to_luma(frame).astype(np.uint8)
    blurred = cv2.GaussianBlur(luma, (5, 5), 0)
    # Thresholds from the frame's own brightness: a fixed pair misses the
    # horizon in hazy or backlit footage, which is most of a sunset clip.
    median = float(np.median(blurred))
    low = int(max(10.0, 0.66 * median))
    high = int(min(255.0, max(low + 10.0, 1.33 * median)))
    edges = cv2.Canny(blurred, low, high)

    height, width = edges.shape[:2]
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 360.0,
        threshold=50,
        minLineLength=int(width * 0.35),
        maxLineGap=max(2, int(width * 0.03)),
    )
    if lines is None or len(lines) == 0:
        return None

    angles: list[float] = []
    weights: list[float] = []
    mid_y: list[float] = []
    for entry in lines:
        # OpenCV has returned both (N, 1, 4) and (N, 4) over the years.
        values = np.asarray(entry, dtype=np.float64).ravel()
        if values.size < 4:
            continue
        x1, y1, x2, y2 = values[:4]
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length <= 0:
            continue
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180
        if abs(angle) <= TILT_SEARCH_DEGREES:
            angles.append(abs(angle))
            weights.append(length)
            mid_y.append((y1 + y2) / 2.0)
    if not angles:
        return None
    order = np.argsort(np.asarray(weights))[::-1][:8]   # the longest lines decide
    chosen = np.asarray(angles)[order]
    chosen_y = np.asarray(mid_y)[order]
    dominant = float(np.median(chosen))

    # A textured frame - forest canopy, choppy water, sensor noise - throws up
    # plenty of long "lines". Two things separate those from a horizon: the
    # candidates must agree on an angle, and they must sit at the same height
    # in the frame, because a horizon is one line and not a field of them.
    if float(np.median(np.abs(chosen - dominant))) > TILT_MAX_SPREAD_DEGREES:
        return None
    row = float(np.median(chosen_y))
    if float(np.median(np.abs(chosen_y - row))) > TILT_MAX_SPREAD_ROWS * height:
        return None

    # A horizon has something different on each side of it. Texture does not.
    split = int(round(min(max(row, 1.0), height - 2.0)))
    above = float(luma[:split].mean())
    below = float(luma[split:].mean())
    if abs(above - below) / 255.0 < TILT_MIN_CONTRAST:
        return None

    return float(min(1.0, dominant / TILT_FULL_DEGREES))


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

    def subject(self, frame: np.ndarray) -> tuple[float, float, float, float] | None:
        """Return ``(area_fraction, cx, cy, separation)`` of the largest salient blob.

        ``separation`` is how much more salient the blob is than the rest of
        the frame, 0..1 - a clean subject against calm surroundings scores
        high, a busy frame with nothing dominant scores low.
        """
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

        inside = labels == biggest
        outside = ~inside
        if inside.any() and outside.any():
            gap = float(scaled[inside].mean() - scaled[outside].mean()) / 255.0
            separation = float(min(1.0, max(0.0, gap * 2.0)))
        else:
            separation = 0.0
        return area_fraction, cx, cy, separation


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
        "subject_separation": None,
        "thirds_distance": None,
        "horizon_tilt": None,
        "motion": None,
    }
    if face_detector is not None:
        count, max_rel = face_detector.detect(frame)
        result["face_count"] = count
        result["face_max_rel"] = max_rel
    if saliency is not None:
        subject = saliency.subject(frame)
        if subject is not None:
            rel, cx, cy, separation = subject
            result["subject_rel"] = rel
            result["subject_cx"] = cx
            result["subject_cy"] = cy
            result["subject_separation"] = separation
            result["thirds_distance"] = thirds_distance(cx, cy)
    result["horizon_tilt"] = horizon_tilt(frame)
    return result
