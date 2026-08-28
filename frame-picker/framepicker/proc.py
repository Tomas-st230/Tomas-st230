"""The only module in this project that is allowed to touch ``subprocess``.

Rule 9.2 of the task: every external process goes through here, and on Windows
every process is started with ``CREATE_NO_WINDOW`` + a hidden ``STARTUPINFO``
so no console window flashes when the GUI is running.

``tests/test_no_direct_subprocess.py`` asserts that no other module imports
``subprocess``, ``os.system`` or ``os.popen``.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import IO, Iterator, Sequence

_CREATE_NO_WINDOW = 0x08000000


class ProcessError(RuntimeError):
    """An external program could not be started or failed."""


@dataclass(frozen=True)
class Result:
    argv: tuple[str, ...]
    returncode: int
    stdout: bytes
    stderr: bytes

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def stderr_text(self, limit: int = 400) -> str:
        text = self.stderr.decode("utf-8", "replace").strip()
        text = " ".join(text.split())
        return text[:limit]


def _hidden_kwargs() -> dict:
    """Keyword arguments that keep a child process from opening a console."""
    if sys.platform != "win32":
        return {}
    startupinfo = subprocess.STARTUPINFO()  # type: ignore[attr-defined]
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW  # type: ignore[attr-defined]
    startupinfo.wShowWindow = subprocess.SW_HIDE  # type: ignore[attr-defined]
    return {"startupinfo": startupinfo, "creationflags": _CREATE_NO_WINDOW}


def executable(name: str) -> str | None:
    """Absolute path of *name* on PATH, or ``None``."""
    return shutil.which(name)


def require(name: str) -> str:
    path = executable(name)
    if path is None:
        raise ProcessError(name)
    return path


def run(argv: Sequence[str], *, timeout: float | None = None, stdin_bytes: bytes | None = None) -> Result:
    """Run *argv* to completion and capture both streams."""
    argv = [str(a) for a in argv]
    try:
        completed = subprocess.run(  # noqa: S603 - argv is always a list built by us
            argv,
            input=stdin_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            **_hidden_kwargs(),
        )
    except FileNotFoundError as exc:
        raise ProcessError(f"{argv[0]}: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ProcessError(f"{argv[0]}: timeout after {timeout} s") from exc
    return Result(tuple(argv), completed.returncode, completed.stdout or b"", completed.stderr or b"")


class Stream:
    """A running child process whose stdout is being read incrementally."""

    def __init__(self, popen: "subprocess.Popen[bytes]", argv: Sequence[str]) -> None:
        self._popen = popen
        self.argv = tuple(str(a) for a in argv)

    @property
    def stdout(self) -> IO[bytes]:
        assert self._popen.stdout is not None
        return self._popen.stdout

    def read_exactly(self, size: int) -> bytes | None:
        """Read exactly *size* bytes, or ``None`` if the stream ended first."""
        buf = self.stdout.read(size)
        if buf is None or len(buf) < size:
            return None
        return buf

    def terminate(self) -> None:
        if self._popen.poll() is None:
            self._popen.terminate()

    def wait(self, timeout: float | None = 10.0) -> int:
        try:
            return self._popen.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._popen.kill()
            return self._popen.wait(timeout=timeout)

    def stderr_text(self, limit: int = 400) -> str:
        if self._popen.stderr is None:
            return ""
        data = self._popen.stderr.read() or b""
        text = " ".join(data.decode("utf-8", "replace").split())
        return text[:limit]


@contextmanager
def stream(argv: Sequence[str], *, bufsize: int = 10 ** 7) -> Iterator[Stream]:
    """Start *argv* and yield a :class:`Stream` reading its stdout.

    The process is always terminated and reaped when the block exits, so a
    cancelled run never leaves an ffmpeg behind.
    """
    argv = [str(a) for a in argv]
    try:
        popen = subprocess.Popen(  # noqa: S603 - argv is always a list built by us
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=bufsize,
            **_hidden_kwargs(),
        )
    except FileNotFoundError as exc:
        raise ProcessError(f"{argv[0]}: {exc}") from exc
    handle = Stream(popen, argv)
    try:
        yield handle
    finally:
        try:
            handle.terminate()
        finally:
            try:
                if popen.stdout is not None:
                    popen.stdout.close()
            except OSError:
                pass
            handle.wait()
