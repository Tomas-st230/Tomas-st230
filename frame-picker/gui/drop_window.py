"""Drag-and-drop window.

Deliberately thin. It collects file paths and settings, hands them to
``framepicker.cli.run_batch``, and displays what that reports back. No
analysis, no scoring, no colour decisions, no file naming, and no Lithuanian
text of its own — every string comes from ``framepicker.strings_lt``.

``python -m gui.drop_window``
"""

from __future__ import annotations

import os
import sys
import threading

from PySide6.QtCore import QObject, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from framepicker import strings_lt as S
from framepicker.cli import (
    DEFAULT_MAX_PER_CLIP,
    DEFAULT_MIN_SCORE,
    DEFAULT_OUT_DIR,
    VIDEO_SUFFIXES,
    Options,
    run_batch,
)

#: Table columns, in order.
COLUMNS = (
    S.GUI_COL_FILE,
    S.GUI_COL_PROFILE,
    S.GUI_COL_COLOR,
    S.GUI_COL_DECODE,
    S.GUI_COL_FRAMES,
    S.GUI_COL_STATUS,
)

#: Machine-readable value from framepicker -> label to show. The mapping is
#: presentation, which is what this module is for.
COLOR_LABELS = {
    "lut": S.GUI_COLOR_LUT,
    "normalise": S.GUI_COLOR_NORMALISE,
    "none": S.GUI_COLOR_NONE,
}
DECODE_LABELS = {"hw": S.GUI_DECODE_HW, "cpu": S.GUI_DECODE_CPU}
PROFILE_CHOICES = (
    (S.GUI_PROFILE_AUTO, "auto"),
    (S.GUI_PROFILE_ON, "on"),
    (S.GUI_PROFILE_OFF, "off"),
)


class _Worker(QObject):
    message = Signal(str)
    clip_done = Signal(dict)
    finished = Signal(bool, str)

    def __init__(self, options: Options, cancel: threading.Event) -> None:
        super().__init__()
        self._options = options
        self._cancel = cancel

    def run(self) -> None:
        result = run_batch(
            self._options,
            on_message=self.message.emit,
            cancel=self._cancel,
            on_clip_done=self.clip_done.emit,
        )
        self.finished.emit(not result.cancelled, result.html_path or "")


class DropWindow(QWidget):
    def __init__(self, out_dir: str = DEFAULT_OUT_DIR) -> None:
        super().__init__()
        self.setWindowTitle(S.APP_TITLE)
        self.setAcceptDrops(True)
        self.resize(920, 620)

        self._initial_out_dir = os.path.abspath(out_dir)
        self._paths: list[str] = []
        self._cancel = threading.Event()
        self._thread: QThread | None = None
        self._worker: _Worker | None = None
        self._report_path = ""

        self._drop = QLabel(S.GUI_DROP_HERE)
        self._drop.setAlignment(Qt.AlignCenter)
        self._drop.setMinimumHeight(76)
        self._drop.setStyleSheet("border: 2px dashed palette(mid); border-radius: 8px; padding: 12px;")
        self._drop.setWordWrap(True)

        self._table = QTableWidget(0, len(COLUMNS))
        self._table.setHorizontalHeaderLabels(list(COLUMNS))
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for column in range(1, len(COLUMNS)):
            self._table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeToContents)

        self._progress = QProgressBar()
        self._progress.setTextVisible(False)
        self._status = QLabel(S.GUI_IDLE)
        self._status.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self._drop)
        layout.addWidget(self._build_settings())
        layout.addWidget(self._table, stretch=1)
        layout.addWidget(self._progress)
        layout.addWidget(self._status)
        layout.addLayout(self._build_buttons())

    # -- construction ------------------------------------------------------

    def _build_settings(self) -> QGroupBox:
        box = QGroupBox(S.GUI_SETTINGS)
        form = QFormLayout(box)

        self._lut = QLineEdit()
        lut_browse = QPushButton(S.GUI_BROWSE)
        lut_browse.clicked.connect(self._on_pick_lut)
        self._lut_all = QCheckBox(S.GUI_LUT_ALL)
        lut_row = QHBoxLayout()
        lut_row.addWidget(self._lut, stretch=1)
        lut_row.addWidget(lut_browse)
        lut_row.addWidget(self._lut_all)
        form.addRow(S.GUI_LUT, lut_row)

        hint = QLabel(S.GUI_LUT_HINT)
        hint.setWordWrap(True)
        hint.setStyleSheet("color: palette(mid);")
        form.addRow("", hint)

        self._profile = QComboBox()
        for label, _value in PROFILE_CHOICES:
            self._profile.addItem(label)
        form.addRow(S.GUI_PROFILE, self._profile)

        self._min_score = QDoubleSpinBox()
        self._min_score.setRange(0.0, 1.0)
        self._min_score.setSingleStep(0.05)
        self._min_score.setDecimals(2)
        self._min_score.setValue(DEFAULT_MIN_SCORE)
        form.addRow(S.GUI_MIN_SCORE, self._min_score)

        self._max_per_clip = QSpinBox()
        self._max_per_clip.setRange(0, 200)
        self._max_per_clip.setValue(DEFAULT_MAX_PER_CLIP)
        form.addRow(S.GUI_MAX_PER_CLIP, self._max_per_clip)

        self._out_dir = QLineEdit(self._initial_out_dir)
        out_browse = QPushButton(S.GUI_BROWSE)
        out_browse.clicked.connect(self._on_pick_out_dir)
        out_row = QHBoxLayout()
        out_row.addWidget(self._out_dir, stretch=1)
        out_row.addWidget(out_browse)
        form.addRow(S.GUI_OUT_DIR, out_row)
        return box

    def _build_buttons(self) -> QHBoxLayout:
        self._start = QPushButton(S.GUI_START)
        self._start.clicked.connect(self._on_start)
        self._start.setEnabled(False)
        self._cancel_button = QPushButton(S.GUI_CANCEL)
        self._cancel_button.clicked.connect(self._on_cancel)
        self._cancel_button.setEnabled(False)
        self._clear = QPushButton(S.GUI_CLEAR)
        self._clear.clicked.connect(lambda: self._set_paths([]))
        self._open_folder = QPushButton(S.GUI_OPEN_FOLDER)
        self._open_folder.clicked.connect(self._on_open_folder)
        self._open_folder.setEnabled(False)
        self._open_report = QPushButton(S.GUI_OPEN_REPORT)
        self._open_report.clicked.connect(self._on_open_report)
        self._open_report.setEnabled(False)

        row = QHBoxLayout()
        row.addWidget(self._start)
        row.addWidget(self._cancel_button)
        row.addWidget(self._clear)
        row.addStretch(1)
        row.addWidget(self._open_folder)
        row.addWidget(self._open_report)
        return row

    # -- drag and drop -----------------------------------------------------

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt naming
        dropped = [url.toLocalFile() for url in event.mimeData().urls()]
        # Folders are passed straight through: framepicker.cli expands them,
        # and deciding what counts as a video file is not the GUI's job.
        self._set_paths(self._paths + [p for p in dropped if os.path.isdir(p) or self._is_video(p)])
        event.acceptProposedAction()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt naming
        chosen, _ = QFileDialog.getOpenFileNames(self, S.APP_TITLE)
        if chosen:
            self._set_paths(self._paths + list(chosen))

    @staticmethod
    def _is_video(path: str) -> bool:
        return path.lower().endswith(VIDEO_SUFFIXES)

    def _set_paths(self, paths: list[str]) -> None:
        seen: set[str] = set()
        unique = []
        for path in paths:
            key = os.path.normcase(os.path.abspath(path))
            if key not in seen:
                seen.add(key)
                unique.append(path)
        self._paths = unique
        self._table.setRowCount(len(unique))
        for row, path in enumerate(unique):
            self._set_cell(row, 0, os.path.basename(path))
            for column in range(1, len(COLUMNS) - 1):
                self._set_cell(row, column, S.GUI_UNKNOWN)
            self._set_cell(row, len(COLUMNS) - 1, S.GUI_WAITING)
        self._start.setEnabled(bool(unique))
        self._progress.setMaximum(max(1, len(unique)))
        self._progress.setValue(0)

    def _set_cell(self, row: int, column: int, text: str) -> None:
        self._table.setItem(row, column, QTableWidgetItem(text))

    # -- settings pickers --------------------------------------------------

    def _on_pick_lut(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(self, S.GUI_LUT, "", "*.cube")
        if chosen:
            self._lut.setText(chosen)

    def _on_pick_out_dir(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, S.GUI_OUT_DIR, self._out_dir.text())
        if chosen:
            self._out_dir.setText(chosen)

    def options(self) -> Options:
        """Whatever the widgets currently say, as a plain Options object."""
        return Options(
            paths=list(self._paths),
            out_dir=self._out_dir.text() or os.path.abspath(DEFAULT_OUT_DIR),
            lut=self._lut.text() or None,
            lut_all=self._lut_all.isChecked(),
            convert_log=PROFILE_CHOICES[self._profile.currentIndex()][1],
            min_score=self._min_score.value(),
            max_per_clip=self._max_per_clip.value(),
        )

    # -- run ---------------------------------------------------------------

    def _on_start(self) -> None:
        if not self._paths or self._thread is not None:
            return
        self._cancel = threading.Event()
        self._report_path = ""

        thread = QThread(self)
        worker = _Worker(self.options(), self._cancel)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.message.connect(self._status.setText)
        worker.clip_done.connect(self._on_clip_done)
        worker.finished.connect(thread.quit)
        worker.finished.connect(self._on_finished)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        self._thread = thread
        self._worker = worker

        self._start.setEnabled(False)
        self._clear.setEnabled(False)
        self._open_report.setEnabled(False)
        self._cancel_button.setEnabled(True)
        thread.start()

    def _on_clip_done(self, summary: dict) -> None:
        row = int(summary.get("index", 0))
        if row >= self._table.rowCount():
            return
        if summary.get("ok"):
            is_log = summary.get("is_log")
            profile = S.GUI_UNKNOWN if is_log is None else (S.GUI_LOG_YES if is_log else S.GUI_LOG_NO)
            self._set_cell(row, 1, profile)
            self._set_cell(row, 2, COLOR_LABELS.get(summary.get("color_mode", ""), S.GUI_UNKNOWN))
            self._set_cell(row, 3, DECODE_LABELS.get(summary.get("decode_path", ""), S.GUI_UNKNOWN))
            self._set_cell(row, 4, str(summary.get("frames", 0)))
            self._set_cell(row, 5, S.GUI_DONE)
        else:
            for column in range(1, len(COLUMNS) - 1):
                self._set_cell(row, column, S.GUI_UNKNOWN)
            self._set_cell(row, 5, S.GUI_FAILED)
            self._table.item(row, 5).setToolTip(str(summary.get("reason", "")))
        self._progress.setValue(self._progress.value() + 1)

    def _on_cancel(self) -> None:
        self._cancel.set()
        self._cancel_button.setEnabled(False)

    def _on_finished(self, completed: bool, report_path: str) -> None:
        self._report_path = report_path
        self._status.setText(S.GUI_DONE if completed else S.GUI_CANCELLED)
        self._start.setEnabled(bool(self._paths))
        self._clear.setEnabled(True)
        self._cancel_button.setEnabled(False)
        self._open_folder.setEnabled(completed)
        self._open_report.setEnabled(bool(report_path) and os.path.isfile(report_path))

    def _on_thread_finished(self) -> None:
        # Never wait() on the worker thread from a slot that thread triggered:
        # the quit() that would let it exit is queued behind this very call.
        self._thread = None
        self._worker = None

    def _on_open_folder(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(self._out_dir.text())))

    def _on_open_report(self) -> None:
        if self._report_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(self._report_path)))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    app = QApplication(argv)
    window = DropWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
