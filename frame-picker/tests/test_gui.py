"""The GUI is allowed to pass options through. It is not allowed to decide anything.

Skipped when PySide6 or the Qt shared libraries are not installed; the layering
rules themselves are enforced in ``test_layering.py``, which needs no Qt.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def qt_app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:  # Qt shared libraries missing
        pytest.skip(f"Qt not usable here: {exc}")
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def window(qt_app, tmp_path):
    from gui.drop_window import DropWindow

    return DropWindow(out_dir=str(tmp_path / "results"))


def test_the_output_folder_argument_is_honoured(window, tmp_path):
    assert window.options().out_dir == str(tmp_path / "results")


def test_settings_reach_the_options_object(window, tmp_path):
    from framepicker.select import MODE_THRESHOLD

    lut = tmp_path / "look.cube"
    lut.write_text("LUT_3D_SIZE 2\n")
    window._lut.setText(str(lut))
    window._lut_all.setChecked(True)
    window._min_score.setValue(0.42)
    window._max_per_clip.setValue(7)
    window._profile.setCurrentIndex(2)          # "off"

    options = window.options()
    assert options.lut == str(lut)
    assert options.lut_all is True
    assert options.min_score == pytest.approx(0.42)
    assert options.max_per_clip == 7
    assert options.convert_log == "off"
    assert options.select_mode == MODE_THRESHOLD, "the GUI must not override the default mode"


def test_the_lut_defaults_to_log_clips_only(window):
    assert window.options().lut_all is False, "a LUT must not be forced onto Rec.709 footage"


def test_dropped_paths_are_deduplicated_and_listed(window, tmp_path):
    clip = tmp_path / "a.mp4"
    clip.write_bytes(b"x")
    window._set_paths([str(clip), str(clip)])
    assert window.options().paths == [str(clip)]
    assert window._table.rowCount() == 1
    assert window._table.item(0, 0).text() == "a.mp4"


def test_a_failed_clip_is_marked_not_silently_blank(window, tmp_path):
    from framepicker import strings_lt as S

    clip = tmp_path / "broken.mp4"
    clip.write_bytes(b"x")
    window._set_paths([str(clip)])
    window._on_clip_done({"index": 0, "ok": False, "reason": "no video stream"})
    assert window._table.item(0, 5).text() == S.GUI_FAILED
    assert "no video stream" in window._table.item(0, 5).toolTip()


def test_a_log_clip_and_a_normal_clip_are_labelled_differently(window, tmp_path):
    from framepicker import strings_lt as S

    for name in ("a.mp4", "b.mp4"):
        (tmp_path / name).write_bytes(b"x")
    window._set_paths([str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4")])
    window._on_clip_done({"index": 0, "ok": True, "is_log": True, "color_mode": "lut",
                          "decode_path": "hw", "frames": 5})
    window._on_clip_done({"index": 1, "ok": True, "is_log": False, "color_mode": "none",
                          "decode_path": "cpu", "frames": 6})
    assert window._table.item(0, 1).text() == S.GUI_LOG_YES
    assert window._table.item(0, 2).text() == S.GUI_COLOR_LUT
    assert window._table.item(1, 1).text() == S.GUI_LOG_NO
    assert window._table.item(1, 2).text() == S.GUI_COLOR_NONE


def test_a_dropped_folder_becomes_one_row_per_file(window, tmp_path):
    """The table rows must be exactly the files the pipeline will process.

    A folder shown as a single row while run_batch processed twenty files left
    nineteen results with nowhere to land.
    """
    import os

    folder = tmp_path / "card"
    folder.mkdir()
    for name, mtime in (("DJI_0003.MP4", 3000), ("DJI_0001.MP4", 1000), ("DJI_0002.MP4", 2000)):
        path = folder / name
        path.write_bytes(b"x")
        os.utime(path, (mtime, mtime))
    (folder / "readme.txt").write_bytes(b"x")

    window._set_paths([str(folder)])
    assert window._table.rowCount() == 3
    listed = [window._table.item(row, 0).text() for row in range(3)]
    assert listed == ["DJI_0001.MP4", "DJI_0002.MP4", "DJI_0003.MP4"], "oldest first"
    assert len(window.options().paths) == 3
    assert window._progress.maximum() == 3
