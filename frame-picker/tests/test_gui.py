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
    status = window._table.columnCount() - 1
    assert window._table.item(0, status).text() == S.GUI_FAILED
    assert "no video stream" in window._table.item(0, status).toolTip()


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


# --------------------------------------------------------------------------
# Cancelling, and starting over with new files
# --------------------------------------------------------------------------


def test_the_look_column_shows_what_was_actually_applied(window, tmp_path):
    from framepicker import strings_lt as S

    (tmp_path / "a.mp4").write_bytes(b"x")
    window._set_paths([str(tmp_path / "a.mp4")])
    window._on_clip_done({"index": 0, "ok": True, "is_log": True, "color_mode": "lut",
                          "look": "nature", "decode_path": "hw", "frames": 4})
    assert window._table.item(0, 3).text() == S.LOOK_NAMES["nature"]


def test_auto_is_offered_in_the_look_box(window):
    from framepicker import grading, strings_lt as S

    labels = [window._look.itemText(i) for i in range(window._look.count())]
    assert S.LOOK_NAMES[grading.AUTO] in labels
    window._look.setCurrentIndex(labels.index(S.LOOK_NAMES[grading.AUTO]))
    assert window.options().look == grading.AUTO


def test_the_file_list_is_frozen_while_a_run_is_going(window, tmp_path):
    from framepicker import strings_lt as S

    (tmp_path / "a.mp4").write_bytes(b"x")
    window._set_paths([str(tmp_path / "a.mp4")])

    class _FakeThread:
        def quit(self):
            pass

        def wait(self, _ms=0):
            return True

    window._thread = _FakeThread()          # pretend a run is in progress
    (tmp_path / "b.mp4").write_bytes(b"x")
    window._set_paths([str(tmp_path / "b.mp4")])
    assert window.options().paths == [str(tmp_path / "a.mp4")]
    assert window._status.text() == S.GUI_BUSY


def test_cancelling_stops_and_leaves_the_window_ready_for_new_files(window, tmp_path):
    from framepicker import strings_lt as S

    (tmp_path / "a.mp4").write_bytes(b"x")
    window._set_paths([str(tmp_path / "a.mp4")])
    window._on_cancel()
    assert window._cancel.is_set() is True
    assert window._status.text() == S.GUI_STOPPING

    # The pipeline reports back, then the worker thread ends.
    window._on_finished(False, "", str(tmp_path / "out" / "run-1"))
    window._on_thread_finished()

    assert window._thread is None
    assert window.options().paths == [], "a cancelled run clears the list"
    assert window._table.rowCount() == 0
    assert window._status.text() == S.GUI_CANCELLED
    assert window._clear.isEnabled() is True

    # ...and a new set of files can be loaded straight away.
    (tmp_path / "b.mp4").write_bytes(b"x")
    window._set_paths([str(tmp_path / "b.mp4")])
    assert window.options().paths == [str(tmp_path / "b.mp4")]
    assert window._start.isEnabled() is True


def test_closing_the_window_is_never_blocked(window, tmp_path):
    """Requirement: this program must never stop a window closing or a shutdown."""
    from PySide6.QtGui import QCloseEvent

    stopped = {"quit": False, "waited": False}

    class _FakeThread:
        def quit(self):
            stopped["quit"] = True

        def wait(self, _ms=0):
            stopped["waited"] = True
            return True

    window._thread = _FakeThread()
    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted() is True
    assert stopped["quit"] is True and stopped["waited"] is True
    assert window._cancel.is_set() is True


def test_the_open_folder_button_points_at_the_run_folder(window, tmp_path):
    run_dir = str(tmp_path / "results" / "run-20260828-120000")
    window._on_finished(True, "", run_dir)
    assert window._run_dir == run_dir
    assert window._open_folder.isEnabled() is True
