"""Drag-and-drop window.

Deliberately thin: it collects file paths, hands them to
``framepicker.cli.run_batch`` and prints whatever that reports. No analysis,
no scoring, no file naming, no Lithuanian text of its own - every string comes
from ``framepicker.strings_lt``.

``python -m gui.drop_window``
"""

from __future__ import annotations

import os
import sys
import threading

from PySide6.QtCore import QObject, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from framepicker import strings_lt as S
from framepicker.cli import DEFAULT_OUT_DIR, VIDEO_SUFFIXES, Options, run_batch

class _Worker(QObject):
    message = Signal(str)
    finished = Signal(bool)

    def __init__(self, options: Options, cancel: threading.Event) -> None:
        super().__init__()
        self._options = options
        self._cancel = cancel

    def run(self) -> None:
        result = run_batch(self._options, on_message=self.message.emit, cancel=self._cancel)
        self.finished.emit(not result.cancelled)


class DropWindow(QWidget):
    def __init__(self, out_dir: str = DEFAULT_OUT_DIR) -> None:
        super().__init__()
        self.setWindowTitle(S.APP_TITLE)
        self.setAcceptDrops(True)
        self.resize(560, 260)

        self._out_dir = os.path.abspath(out_dir)
        self._paths: list[str] = []
        self._cancel = threading.Event()
        self._thread: QThread | None = None

        self._drop = QLabel(S.GUI_DROP_HERE)
        self._drop.setMinimumHeight(120)
        self._drop.setStyleSheet(
            "border: 2px dashed palette(mid); border-radius: 8px; padding: 24px;"
        )
        self._drop.setWordWrap(True)

        self._status = QLabel(S.GUI_IDLE)
        self._status.setWordWrap(True)

        self._start = QPushButton(S.GUI_START)
        self._start.clicked.connect(self._on_start)
        self._start.setEnabled(False)
        self._cancel_button = QPushButton(S.GUI_CANCEL)
        self._cancel_button.clicked.connect(self._on_cancel)
        self._cancel_button.setEnabled(False)
        self._open = QPushButton(S.GUI_OPEN_FOLDER)
        self._open.clicked.connect(self._on_open)
        self._open.setEnabled(False)

        buttons = QHBoxLayout()
        buttons.addWidget(self._start)
        buttons.addWidget(self._cancel_button)
        buttons.addWidget(self._open)

        layout = QVBoxLayout(self)
        layout.addWidget(self._drop)
        layout.addWidget(self._status)
        layout.addLayout(buttons)

    # -- drag and drop -----------------------------------------------------

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt naming
        paths = []
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isdir(path):
                paths += [
                    os.path.join(path, name)
                    for name in sorted(os.listdir(path))
                    if name.lower().endswith(VIDEO_SUFFIXES)
                ]
            elif path.lower().endswith(VIDEO_SUFFIXES):
                paths.append(path)
        self._set_paths(paths)
        event.acceptProposedAction()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt naming
        chosen, _ = QFileDialog.getOpenFileNames(self, S.GUI_DROP_HERE)
        if chosen:
            self._set_paths(list(chosen))

    def _set_paths(self, paths: list[str]) -> None:
        self._paths = paths
        self._drop.setText("\n".join(os.path.basename(p) for p in paths) or S.GUI_DROP_HERE)
        self._start.setEnabled(bool(paths))

    # -- run ---------------------------------------------------------------

    def _on_start(self) -> None:
        if not self._paths or self._thread is not None:
            return
        self._cancel = threading.Event()
        options = Options(paths=list(self._paths), out_dir=self._out_dir)

        thread = QThread(self)
        worker = _Worker(options, self._cancel)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.message.connect(self._status.setText)
        worker.finished.connect(thread.quit)
        worker.finished.connect(self._on_finished)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        self._thread = thread
        self._worker = worker

        self._start.setEnabled(False)
        self._cancel_button.setEnabled(True)
        thread.start()

    def _on_cancel(self) -> None:
        self._cancel.set()
        self._cancel_button.setEnabled(False)

    def _on_finished(self, completed: bool) -> None:
        self._status.setText(S.GUI_DONE if completed else S.GUI_CANCELLED)
        self._start.setEnabled(bool(self._paths))
        self._cancel_button.setEnabled(False)
        self._open.setEnabled(completed)

    def _on_thread_finished(self) -> None:
        # Never wait() on the worker thread from a slot that thread triggered:
        # the quit() that would let it exit is queued behind this very call.
        self._thread = None

    def _on_open(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(self._out_dir))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    app = QApplication(argv)
    window = DropWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
