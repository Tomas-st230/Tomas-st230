"""Drag-and-drop window.

Deliberately thin. It collects file paths and settings, hands them to
``framepicker.cli.run_batch``, and displays what that reports back. No
analysis, no scoring, no colour decisions, no file naming, and no Lithuanian
text of its own — every string comes from ``framepicker.strings_lt``.

Everything the command line can do is reachable here: :data:`OPTION_CONTROLS`
maps every field of ``Options`` to the widget that sets it, and a test fails if
a new option is added without one. What the run reports back is shown twice —
as the running log, and as the measured values per file, the same numbers that
land in ``log.jsonl``.

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
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from framepicker import grading, learn, report as report_module
from framepicker import strings_lt as S
from framepicker.cli import (
    DEFAULT_FPS,
    DEFAULT_GLOBAL_TOP,
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_MAX_PER_CLIP,
    DEFAULT_MIN_GAP,
    DEFAULT_MIN_SCORE,
    DEFAULT_OUT_DIR,
    DEFAULT_PER_CLIP,
    ORDER_DATE,
    ORDER_NAME,
    ORDER_NONE,
    PROXY_AUTO,
    PROXY_OFF,
    VIDEO_SUFFIXES,
    Options,
    default_out_dir,
    default_source_dir,
    expand_inputs,
    run_batch,
)
from framepicker.select import MODE_COUNT, MODE_THRESHOLD

#: Table columns, in order.
COLUMNS = (
    S.GUI_COL_FILE,
    S.GUI_COL_PROFILE,
    S.GUI_COL_COLOR,
    S.GUI_COL_LOOK,
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
#: Look names as offered by framepicker, paired with their Lithuanian labels.
LOOK_CHOICES = tuple((S.LOOK_NAMES.get(name, name), name) for name in grading.available())
HWACCEL_CHOICES = (
    (S.GUI_HWACCEL_AUTO, "auto"),
    (S.GUI_HWACCEL_CUDA, "cuda"),
    (S.GUI_HWACCEL_NONE, "none"),
)
SELECT_CHOICES = (
    (S.GUI_SELECT_THRESHOLD, MODE_THRESHOLD),
    (S.GUI_SELECT_COUNT, MODE_COUNT),
)
ORDER_CHOICES = (
    (S.GUI_ORDER_DATE, ORDER_DATE),
    (S.GUI_ORDER_NAME, ORDER_NAME),
    (S.GUI_ORDER_NONE, ORDER_NONE),
)
FORMAT_CHOICES = (("JPG", "jpg"), ("PNG", "png"))

#: Every field of ``Options`` and the widget that sets it. ``None`` marks a
#: field that is deliberately not a control: the paths come from the drop area,
#: and the face model is a developer switch with no place in a window. A test
#: walks this against the dataclass, so an option added to the pipeline cannot
#: quietly go missing from the window.
OPTION_CONTROLS = {
    "paths": None,                  # the drop area
    "face_model": None,             # --face-model, for debugging a model file
    "out_dir": "_out_dir",
    "per_clip": "_per_clip",
    "fps": "_fps",
    "min_gap": "_min_gap",
    "convert_log": "_profile",
    "lut": "_lut",
    "lut_all": "_lut_all",
    "normalise_strength": "_normalise",
    "lut_strength": "_lut_strength",
    "look": "_look",
    "look_strength": "_look_strength",
    "jobs": "_jobs",
    "no_faces": "_no_faces",
    "image_format": "_format",
    "jpeg_quality": "_jpeg_quality",
    "global_top": "_global_top",
    "max_candidates": "_max_candidates",
    "hwaccel": "_hwaccel",
    "order": "_order_box",
    "select_mode": "_select_mode",
    "min_score": "_min_score",
    "max_per_clip": "_max_per_clip",
    "export_height": "_export_height",
    "proxy": "_proxy",
    "keyframes": "_keyframes",
    "gpu_scale": "_gpu_scale",
    "run_folder": "_run_folder",
    "write_log": "_write_log",
}

#: Lines kept in the progress view. The whole log is on disk; this is a window.
LOG_MAX_LINES = 5000
#: Milliseconds the window waits for a cancelled run on close. Bounded on
#: purpose: closing or shutting down must never be blocked by this program.
CLOSE_WAIT_MS = 3000


class _Worker(QObject):
    message = Signal(str)
    clip_done = Signal(dict)
    #: completed, report path, the folder this run wrote into, log.txt path
    finished = Signal(bool, str, str, str)

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
        self.finished.emit(
            not result.cancelled,
            result.html_path or "",
            result.out_dir or "",
            result.log_paths[0] if result.log_paths else "",
        )


class DropWindow(QWidget):
    def __init__(self, out_dir: str | None = None) -> None:
        super().__init__()
        self.setWindowTitle(S.APP_TITLE)
        self.setAcceptDrops(True)
        self.resize(1080, 820)

        self._initial_out_dir = os.path.abspath(out_dir or default_out_dir())
        #: Where the file dialogs open. The footage folder when it exists.
        self._browse_dir = default_source_dir() or ""
        self._paths: list[str] = []
        self._cancel = threading.Event()
        self._thread: QThread | None = None
        self._worker: _Worker | None = None
        self._report_path = ""
        self._run_dir = ""
        self._log_path = ""
        self._stopping = False
        #: Measured values per table row, exactly as they went into log.jsonl.
        self._values: dict[int, dict] = {}

        self._drop = QLabel(S.GUI_DROP_HERE)
        self._drop.setAlignment(Qt.AlignCenter)
        self._drop.setMinimumHeight(64)
        self._drop.setStyleSheet("border: 2px dashed palette(mid); border-radius: 8px; padding: 10px;")
        self._drop.setWordWrap(True)

        self._table = QTableWidget(0, len(COLUMNS))
        self._table.setHorizontalHeaderLabels(list(COLUMNS))
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for column in range(1, len(COLUMNS)):
            self._table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeToContents)
        self._table.itemSelectionChanged.connect(self._on_row_selected)

        self._progress = QProgressBar()
        self._progress.setTextVisible(False)
        self._status = QLabel(S.GUI_IDLE)
        self._status.setWordWrap(True)

        split = QSplitter(Qt.Vertical)
        split.addWidget(self._table)
        split.addWidget(self._build_output_tabs())
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)

        layout = QVBoxLayout(self)
        layout.addWidget(self._drop)
        layout.addWidget(self._build_settings())
        layout.addWidget(split, stretch=1)
        layout.addWidget(self._progress)
        layout.addWidget(self._status)
        layout.addLayout(self._build_buttons())

    # -- construction ------------------------------------------------------

    def _build_settings(self) -> QWidget:
        box = QGroupBox(S.GUI_SETTINGS)
        tabs = QTabWidget()
        tabs.addTab(self._build_main_tab(), S.GUI_TAB_MAIN)
        tabs.addTab(self._build_speed_tab(), S.GUI_TAB_SPEED)
        tabs.addTab(self._build_export_tab(), S.GUI_TAB_EXPORT)
        outer = QVBoxLayout(box)
        outer.addWidget(tabs)
        return box

    def _build_main_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)

        self._lut = QLineEdit()
        lut_browse = QPushButton(S.GUI_BROWSE)
        lut_browse.clicked.connect(self._on_pick_lut)
        self._lut_all = QCheckBox(S.GUI_LUT_ALL)
        lut_row = QHBoxLayout()
        lut_row.addWidget(self._lut, stretch=1)
        lut_row.addWidget(lut_browse)
        lut_row.addWidget(self._lut_all)
        form.addRow(S.GUI_LUT, lut_row)
        form.addRow("", self._hint(S.GUI_LUT_HINT))

        self._lut_auto = QCheckBox(S.GUI_LUT_STRENGTH_AUTO)
        self._lut_auto.setChecked(True)
        self._lut_strength = self._spin(0.0, 1.0, 0.05, 1.0)
        self._lut_strength.setEnabled(False)
        self._lut_auto.toggled.connect(lambda on: self._lut_strength.setEnabled(not on))
        strength_row = QHBoxLayout()
        strength_row.addWidget(self._lut_auto)
        strength_row.addWidget(self._lut_strength)
        strength_row.addStretch(1)
        form.addRow(S.GUI_LUT_STRENGTH, strength_row)
        form.addRow("", self._hint(S.GUI_LUT_STRENGTH_HINT))

        self._profile = self._combo(PROFILE_CHOICES)
        form.addRow(S.GUI_PROFILE, self._profile)

        self._look = self._combo(LOOK_CHOICES)
        self._look_strength = self._spin(0.0, 1.0, 0.1, grading.DEFAULT_STRENGTH)
        look_row = QHBoxLayout()
        look_row.addWidget(self._look, stretch=1)
        look_row.addWidget(QLabel(S.GUI_LOOK_STRENGTH))
        look_row.addWidget(self._look_strength)
        form.addRow(S.GUI_LOOK, look_row)
        form.addRow("", self._hint(S.GUI_LOOK_AUTO_HINT))

        self._min_score = self._spin(0.0, 1.0, 0.05, DEFAULT_MIN_SCORE)
        form.addRow(S.GUI_MIN_SCORE, self._min_score)

        self._max_per_clip = self._int_spin(0, 200, DEFAULT_MAX_PER_CLIP)
        form.addRow(S.GUI_MAX_PER_CLIP, self._max_per_clip)

        self._out_dir = QLineEdit(self._initial_out_dir)
        out_browse = QPushButton(S.GUI_BROWSE)
        out_browse.clicked.connect(self._on_pick_out_dir)
        out_row = QHBoxLayout()
        out_row.addWidget(self._out_dir, stretch=1)
        out_row.addWidget(out_browse)
        form.addRow(S.GUI_OUT_DIR, out_row)

        self._run_folder = QCheckBox(S.GUI_RUN_FOLDER)
        self._run_folder.setChecked(True)
        self._write_log = QCheckBox(S.GUI_WRITE_LOG)
        self._write_log.setChecked(True)
        files_row = QHBoxLayout()
        files_row.addWidget(self._run_folder)
        files_row.addWidget(self._write_log)
        files_row.addStretch(1)
        form.addRow("", files_row)
        return page

    def _build_speed_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)

        self._proxy = QCheckBox(S.GUI_PROXY)
        self._proxy.setChecked(True)
        form.addRow("", self._proxy)
        form.addRow("", self._hint(S.GUI_PROXY_HINT))

        self._keyframes = QCheckBox(S.GUI_KEYFRAMES)
        form.addRow("", self._keyframes)
        form.addRow("", self._hint(S.GUI_KEYFRAMES_HINT))

        self._gpu_scale = QCheckBox(S.GUI_GPU_SCALE)
        self._gpu_scale.setChecked(True)
        form.addRow("", self._gpu_scale)

        self._hwaccel = self._combo(HWACCEL_CHOICES)
        form.addRow(S.GUI_HWACCEL, self._hwaccel)

        self._jobs = self._int_spin(0, 32, 0)
        form.addRow(S.GUI_JOBS, self._jobs)

        self._fps = self._spin(0.1, 30.0, 0.5, DEFAULT_FPS)
        form.addRow(S.GUI_FPS, self._fps)

        self._max_candidates = self._int_spin(50, 100000, DEFAULT_MAX_CANDIDATES)
        form.addRow(S.GUI_MAX_CANDIDATES, self._max_candidates)

        self._no_faces = QCheckBox(S.GUI_NO_FACES)
        form.addRow("", self._no_faces)
        return page

    def _build_export_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)

        self._select_mode = self._combo(SELECT_CHOICES)
        form.addRow(S.GUI_SELECT_MODE, self._select_mode)

        self._per_clip = self._int_spin(1, 100, DEFAULT_PER_CLIP)
        form.addRow(S.GUI_PER_CLIP, self._per_clip)

        self._min_gap = self._spin(0.0, 60.0, 0.5, DEFAULT_MIN_GAP)
        form.addRow(S.GUI_MIN_GAP, self._min_gap)

        self._export_height = self._int_spin(0, 8000, 0)
        form.addRow(S.GUI_EXPORT_HEIGHT, self._export_height)

        self._format = self._combo(FORMAT_CHOICES)
        form.addRow(S.GUI_FORMAT, self._format)

        self._jpeg_quality = self._int_spin(2, 31, 2)
        form.addRow(S.GUI_JPEG_QUALITY, self._jpeg_quality)

        self._global_top = self._int_spin(0, 200, DEFAULT_GLOBAL_TOP)
        form.addRow(S.GUI_GLOBAL_TOP, self._global_top)

        self._order_box = self._combo(ORDER_CHOICES)
        form.addRow(S.GUI_ORDER, self._order_box)

        self._normalise = self._spin(0.0, 1.0, 0.1, 1.0)
        form.addRow(S.GUI_NORMALISE, self._normalise)
        return page

    def _build_output_tabs(self) -> QWidget:
        tabs = QTabWidget()

        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumBlockCount(LOG_MAX_LINES)
        self._log_view.setPlaceholderText(S.GUI_LOG_LINES)
        tabs.addTab(self._log_view, S.GUI_TAB_PROGRESS)

        self._values_table = QTableWidget(0, 2)
        self._values_table.setHorizontalHeaderLabels([S.GUI_VALUES_COL_NAME, S.GUI_VALUES_COL_VALUE])
        self._values_table.verticalHeader().setVisible(False)
        self._values_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._values_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._values_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        wrapper = QWidget()
        inner = QVBoxLayout(wrapper)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.addWidget(self._hint(S.GUI_VALUES_HINT))
        inner.addWidget(self._values_table)
        tabs.addTab(wrapper, S.GUI_TAB_VALUES)
        return tabs

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
        self._open_log = QPushButton(S.GUI_OPEN_LOG)
        self._open_log.clicked.connect(self._on_open_log)
        self._open_log.setEnabled(False)
        self._calibrate = QPushButton(S.GUI_CALIBRATE)
        self._calibrate.clicked.connect(self._on_calibrate)
        self._calibrate.setEnabled(False)

        row = QHBoxLayout()
        row.addWidget(self._start)
        row.addWidget(self._cancel_button)
        row.addWidget(self._clear)
        row.addStretch(1)
        row.addWidget(self._calibrate)
        row.addWidget(self._open_log)
        row.addWidget(self._open_folder)
        row.addWidget(self._open_report)
        return row

    # -- small widget helpers (presentation only) --------------------------

    @staticmethod
    def _hint(text: str) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet("color: palette(mid);")
        return label

    @staticmethod
    def _combo(choices) -> QComboBox:
        box = QComboBox()
        for label, _value in choices:
            box.addItem(label)
        return box

    @staticmethod
    def _spin(low: float, high: float, step: float, value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setSingleStep(step)
        spin.setDecimals(2)
        spin.setValue(value)
        return spin

    @staticmethod
    def _int_spin(low: int, high: int, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(low, high)
        spin.setValue(value)
        return spin

    # -- drag and drop -----------------------------------------------------

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._busy():
            return
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._busy():
            self._status.setText(S.GUI_BUSY)
            return
        dropped = [url.toLocalFile() for url in event.mimeData().urls()]
        self._set_paths(self._paths + [p for p in dropped if os.path.isdir(p) or self._is_video(p)])
        event.acceptProposedAction()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._busy():
            self._status.setText(S.GUI_BUSY)
            return
        chosen, _ = QFileDialog.getOpenFileNames(self, S.APP_TITLE, self._browse_dir)
        if chosen:
            self._set_paths(self._paths + list(chosen))

    def _busy(self) -> bool:
        """Is a run in progress? While it is, the file list is frozen."""
        return self._thread is not None

    @staticmethod
    def _is_video(path: str) -> bool:
        return path.lower().endswith(VIDEO_SUFFIXES)

    def _set_paths(self, paths: list[str]) -> None:
        if self._busy():
            self._status.setText(S.GUI_BUSY)
            return
        # Expand here, with framepicker's own function, so the table rows are
        # exactly the files run_batch will process in exactly its order. A
        # dropped folder that showed as one row while the pipeline processed
        # twenty files meant nineteen results had nowhere to land.
        unique, unmatched = expand_inputs(paths, ORDER_CHOICES[self._order_box.currentIndex()][1])
        self._paths = unique
        self._values = {}
        self._values_table.setRowCount(0)
        if unmatched:
            self._status.setText(S.input_not_found(unmatched[0]))
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
        chosen, _ = QFileDialog.getOpenFileName(self, S.GUI_LUT, self._browse_dir, "*.cube")
        if chosen:
            self._lut.setText(chosen)

    def _on_pick_out_dir(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, S.GUI_OUT_DIR, self._out_dir.text() or self._browse_dir)
        if chosen:
            self._out_dir.setText(chosen)

    def options(self) -> Options:
        """Whatever the widgets currently say, as a plain Options object."""
        return Options(
            paths=list(self._paths),
            out_dir=self._out_dir.text() or os.path.abspath(DEFAULT_OUT_DIR),
            per_clip=self._per_clip.value(),
            fps=self._fps.value(),
            min_gap=self._min_gap.value(),
            convert_log=PROFILE_CHOICES[self._profile.currentIndex()][1],
            lut=self._lut.text() or None,
            lut_all=self._lut_all.isChecked(),
            normalise_strength=self._normalise.value(),
            lut_strength=None if self._lut_auto.isChecked() else self._lut_strength.value(),
            look=LOOK_CHOICES[self._look.currentIndex()][1],
            look_strength=self._look_strength.value(),
            jobs=self._jobs.value(),
            no_faces=self._no_faces.isChecked(),
            image_format=FORMAT_CHOICES[self._format.currentIndex()][1],
            jpeg_quality=self._jpeg_quality.value(),
            global_top=self._global_top.value(),
            max_candidates=self._max_candidates.value(),
            hwaccel=HWACCEL_CHOICES[self._hwaccel.currentIndex()][1],
            order=ORDER_CHOICES[self._order_box.currentIndex()][1],
            select_mode=SELECT_CHOICES[self._select_mode.currentIndex()][1],
            min_score=self._min_score.value(),
            max_per_clip=self._max_per_clip.value(),
            export_height=self._export_height.value(),
            proxy=PROXY_AUTO if self._proxy.isChecked() else PROXY_OFF,
            keyframes=self._keyframes.isChecked(),
            gpu_scale=self._gpu_scale.isChecked(),
            run_folder=self._run_folder.isChecked(),
            write_log=self._write_log.isChecked(),
        )

    # -- run ---------------------------------------------------------------

    def _on_start(self) -> None:
        if not self._paths or self._thread is not None:
            return
        self._cancel = threading.Event()
        self._report_path = ""
        self._log_path = ""
        self._log_view.clear()

        thread = QThread(self)
        worker = _Worker(self.options(), self._cancel)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.message.connect(self._on_message)
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
        self._open_log.setEnabled(False)
        self._cancel_button.setEnabled(True)
        thread.start()

    def _on_message(self, text: str) -> None:
        """Every line the pipeline said: on the status line and in the log view."""
        self._status.setText(text)
        self._log_view.appendPlainText(text)

    def _on_clip_done(self, summary: dict) -> None:
        row = int(summary.get("index", 0))
        last = len(COLUMNS) - 1
        if row >= self._table.rowCount():
            return
        self._values[row] = dict(summary.get("values") or {})
        if summary.get("ok"):
            is_log = summary.get("is_log")
            profile = S.GUI_UNKNOWN if is_log is None else (S.GUI_LOG_YES if is_log else S.GUI_LOG_NO)
            self._set_cell(row, 1, profile)
            self._set_cell(row, 2, COLOR_LABELS.get(summary.get("color_mode", ""), S.GUI_UNKNOWN))
            self._set_cell(row, 3, S.LOOK_NAMES.get(summary.get("look", ""), S.GUI_UNKNOWN))
            self._set_cell(row, 4, DECODE_LABELS.get(summary.get("decode_path", ""), S.GUI_UNKNOWN))
            self._set_cell(row, 5, str(summary.get("frames", 0)))
            self._set_cell(row, last, S.GUI_DONE)
        else:
            for column in range(1, last):
                self._set_cell(row, column, S.GUI_UNKNOWN)
            self._set_cell(row, last, S.GUI_FAILED)
            self._table.item(row, last).setToolTip(str(summary.get("reason", "")))
        self._progress.setValue(self._progress.value() + 1)
        if self._table.currentRow() == row:
            self._show_values(row)

    def _on_row_selected(self) -> None:
        self._show_values(self._table.currentRow())

    def _show_values(self, row: int) -> None:
        """The measured values of one file — the same fields log.jsonl carries."""
        values = self._values.get(row, {})
        rows = [(key, value) for key, value in values.items() if key in S.VALUE_LABELS]
        self._values_table.setRowCount(len(rows))
        for index, (key, value) in enumerate(rows):
            self._values_table.setItem(index, 0, QTableWidgetItem(S.VALUE_LABELS[key]))
            self._values_table.setItem(index, 1, QTableWidgetItem(self._format_value(value)))

    @staticmethod
    def _format_value(value) -> str:
        if value is None or value == "":
            return S.GUI_UNKNOWN
        if isinstance(value, bool):
            return S.GUI_YES if value else S.GUI_NO
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    def _on_cancel(self) -> None:
        # Stopping means stopping: the pipeline deletes what this run wrote,
        # and the window goes back to a state where new files can be loaded.
        self._stopping = True
        self._cancel.set()
        self._cancel_button.setEnabled(False)
        self._status.setText(S.GUI_STOPPING)

    def _on_finished(self, completed: bool, report_path: str, out_dir: str,
                     log_path: str = "") -> None:
        self._report_path = report_path
        self._run_dir = out_dir
        self._log_path = log_path
        self._status.setText(S.GUI_DONE if completed else S.GUI_CANCELLED)
        self._cancel_button.setEnabled(False)
        self._open_folder.setEnabled(completed and bool(out_dir))
        self._open_report.setEnabled(bool(report_path) and os.path.isfile(report_path))
        self._open_log.setEnabled(bool(log_path) and os.path.isfile(log_path))
        self._calibrate.setEnabled(completed and bool(out_dir))

    def _on_thread_finished(self) -> None:
        # Never wait() on the worker thread from a slot that thread triggered:
        # the quit() that would let it exit is queued behind this very call.
        self._thread = None
        self._worker = None
        if self._stopping:
            # A cancelled run leaves nothing behind, so neither does the table.
            self._stopping = False
            self._set_paths([])
            self._status.setText(S.GUI_CANCELLED)
        else:
            self._start.setEnabled(bool(self._paths))
        self._clear.setEnabled(True)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Closing the window must always work.

        The run is asked to stop and given a short moment to notice; then the
        window closes whatever happened. Nothing here can refuse the close -
        that was the explicit requirement.
        """
        if self._thread is not None:
            self._cancel.set()
            self._thread.quit()
            self._thread.wait(CLOSE_WAIT_MS)
        event.accept()

    # -- results -----------------------------------------------------------

    def _on_open_folder(self) -> None:
        # The run's own folder, not the parent: that is where this run's
        # stills and report actually are.
        folder = self._run_dir or self._out_dir.text()
        QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(folder)))

    def _on_open_report(self) -> None:
        if self._report_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(self._report_path)))

    def _on_open_log(self) -> None:
        if self._log_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(self._log_path)))

    def _on_calibrate(self) -> None:
        """Compare the frames Tomas kept with what the last run picked.

        The comparison itself is ``framepicker.learn``; this only asks for the
        folder and shows what came back.
        """
        if not self._run_dir:
            self._status.setText(S.GUI_CALIBRATE_NEEDS_RUN)
            return
        folder = QFileDialog.getExistingDirectory(self, S.GUI_CALIBRATE_PICK, self._browse_dir)
        if not folder:
            return
        results_path = os.path.join(self._run_dir, report_module.RESULTS_JSON)
        try:
            analysis = learn.analyse_paths(results_path, folder)
        except (OSError, ValueError) as exc:
            self._on_message(S.learn_cannot_read(results_path, str(exc)))
            return
        for line in learn.format_report(analysis).splitlines():
            self._log_view.appendPlainText(line)
        self._status.setText(analysis["messages"][-1] if analysis["messages"] else S.GUI_DONE)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    app = QApplication(argv)
    window = DropWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
