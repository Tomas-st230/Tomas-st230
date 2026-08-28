"""What DJI writes next to the video, and what the tool does with it.

The interesting cases are the two that made the LUT land on the wrong files:
a D-Log clip whose colour tags all say ``bt709``, and a Normal-colour clip in
the same folder that must be left alone.
"""

from __future__ import annotations

import dataclasses
import os

import pytest

from framepicker import decode, sidecar
from framepicker.probe import ClipInfo

from conftest import _make_clip, requires_ffmpeg  # noqa: F401 - fixture helpers


def _clip(name: str = "DJI_20260822191345_0239_D.MP4", **kwargs) -> ClipInfo:
    base = ClipInfo(
        path=os.path.join("card", name),
        name=name,
        duration=20.0,
        fps=59.94,
        width=3840,
        height=2160,
        codec="hevc",
        pix_fmt="yuv420p10le",
        color_transfer="bt709",
        color_primaries="bt709",
        color_space="bt709",
        color_range="tv",
        nb_frames=1200,
        size_bytes=1,
        extra={"format_tags": {"encoder": "DJI Lito X1"}},
    )
    return dataclasses.replace(base, **kwargs)


CAPTION = (
    "1\n"
    "00:00:00,000 --> 00:00:00,016\n"
    "<font size=\"28\">FrameCnt: 1, DiffTime: 16ms\n"
    "2026-08-22 19:13:45.123\n"
    "[iso: 100] [shutter: 1/500.0] [fnum: 1.7] [ev: 0] [color_md: {mode}] "
    "[focal_len: 24.00] [latitude: 54.6] [longitude: 25.2]</font>\n\n"
)


def _make_proxy(folder, stem: str, seconds: float, filters: str) -> str:
    """A .LRF next to the master. ffmpeg cannot guess a muxer from ``.LRF``,
    so the file is written as MP4 and then given the name DJI uses."""
    built = _make_clip(os.path.join(str(folder), stem + "_proxy.mp4"), seconds, filters, fps=10)
    path = os.path.join(str(folder), stem + ".LRF")
    os.replace(built, path)
    return path


def _write_captions(folder, video_name: str, mode: str) -> str:
    stem = os.path.splitext(video_name)[0]
    path = os.path.join(str(folder), stem + ".SRT")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(CAPTION.format(mode=mode))
    return path


# --------------------------------------------------------------------------
# color_md
# --------------------------------------------------------------------------


@pytest.mark.parametrize("raw,kind", [
    ("default", "default"), ("dlog_m", "dlog"), ("d-log", "dlog"),
    ("D_Log_M", "dlog"), ("hlg", "hlg"), ("d_cinlike", "dcinelike"),
    ("something_new", "other"),
])
def test_color_mode_values_are_classified(raw, kind):
    assert sidecar.classify_color_mode(raw) == kind


def test_color_md_is_read_from_the_caption_sidecar(tmp_path):
    name = "DJI_0001_D.MP4"
    (tmp_path / name).write_bytes(b"x")
    _write_captions(tmp_path, name, "dlog_m")
    mode = sidecar.read_color_mode(str(tmp_path / name))
    assert mode is not None
    assert mode.value == "dlog_m"
    assert mode.kind == "dlog"
    assert mode.is_log is True


def test_no_sidecar_means_unknown_not_normal(tmp_path):
    name = "DJI_0002_D.MP4"
    (tmp_path / name).write_bytes(b"x")
    assert sidecar.read_color_mode(str(tmp_path / name)) is None


def test_the_sidecar_beats_every_guess_in_both_directions(tmp_path):
    """A Normal-colour clip in a D-Log folder must not be converted."""
    normal = sidecar.ColorMode("default", "default", str(tmp_path / "a.SRT"), 1)
    verdict = decode.detect_log(_clip(), "auto", None, normal)
    assert verdict.is_log is False
    assert verdict.source == "sidecar"
    assert verdict.is_a_guess is False

    logged = sidecar.ColorMode("dlog_m", "dlog", str(tmp_path / "b.SRT"), 1)
    verdict = decode.detect_log(_clip(), "auto", None, logged)
    assert verdict.is_log is True
    assert verdict.profile == "dlog"
    assert verdict.is_a_guess is False


def test_the_flag_still_beats_the_sidecar():
    normal = sidecar.ColorMode("default", "default", "a.SRT", 1)
    assert decode.detect_log(_clip(), "on", None, normal).is_log is True
    logged = sidecar.ColorMode("dlog_m", "dlog", "b.SRT", 1)
    assert decode.detect_log(_clip(), "off", None, logged).is_log is False


# --------------------------------------------------------------------------
# The 10-bit inference
# --------------------------------------------------------------------------


def test_ten_bit_dji_footage_is_treated_as_log():
    """Tomas's files: bt709 tags, 10-bit, D-Log M. Nothing else catches this."""
    verdict = decode.detect_log(_clip(), "auto", None, None)
    assert verdict.is_log is True
    assert verdict.source == "bitdepth"
    assert verdict.is_a_guess is True
    assert "DJI Lito X1" in verdict.detail


def test_eight_bit_dji_footage_is_left_alone():
    verdict = decode.detect_log(_clip(pix_fmt="yuv420p"), "auto", None, None)
    assert verdict.is_log is False
    assert verdict.source == "default"


def test_a_ten_bit_file_from_another_camera_is_not_touched():
    """The premise is about DJI's recording modes, so the rule is gated on DJI."""
    other = _clip(name="A001_C001.mov", extra={"format_tags": {"encoder": "Sony"}})
    assert decode.detect_log(other, "auto", None, None).is_log is False


# --------------------------------------------------------------------------
# The .LRF proxy
# --------------------------------------------------------------------------


@requires_ffmpeg
def test_a_matching_proxy_is_accepted(tmp_path):
    master = _make_clip(str(tmp_path / "DJI_0100_D.MP4"), 4, fps=10)
    proxy_path = _make_proxy(tmp_path, "DJI_0100_D", 4, "scale=160:120")
    from framepicker.probe import probe

    assert sidecar.find_proxy(master) == proxy_path
    result = sidecar.check_proxy(probe(master), proxy_path, min_long_edge=160)
    assert result.usable is True, result.detail


@requires_ffmpeg
def test_a_proxy_of_a_different_length_is_refused(tmp_path):
    master = _make_clip(str(tmp_path / "DJI_0101_D.MP4"), 6, fps=10)
    proxy_path = _make_proxy(tmp_path, "DJI_0101_D", 2, "scale=160:120")
    from framepicker.probe import probe

    result = sidecar.check_proxy(probe(master), proxy_path, min_long_edge=160)
    assert result.usable is False
    assert "duration" in result.detail


@requires_ffmpeg
def test_a_proxy_smaller_than_the_analysis_size_is_refused(tmp_path):
    master = _make_clip(str(tmp_path / "DJI_0102_D.MP4"), 3, fps=10)
    proxy_path = _make_proxy(tmp_path, "DJI_0102_D", 3, "scale=64:48")
    from framepicker.probe import probe

    result = sidecar.check_proxy(probe(master), proxy_path, min_long_edge=320)
    assert result.usable is False
    assert "long edge" in result.detail


def test_a_missing_proxy_is_simply_absent(tmp_path):
    (tmp_path / "alone.mp4").write_bytes(b"x")
    assert sidecar.find_proxy(str(tmp_path / "alone.mp4")) is None
