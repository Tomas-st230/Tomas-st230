"""Keep the machine awake while a long batch runs.

Deliberately narrow. On Windows this calls ``SetThreadExecutionState`` with
``ES_CONTINUOUS | ES_SYSTEM_REQUIRED``, which only resets the system idle
timer. It does **not**:

* block shutdown, restart, log-off or sleep that *you* ask for;
* keep the display on (``ES_DISPLAY_REQUIRED`` is not set on purpose, so the
  screen still goes dark and the machine still locks);
* survive the process - the request is released when the block exits, and
  Windows drops it automatically if the process dies.

Everything is best-effort: if the call fails, the run continues and says so
rather than refusing to work.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Iterator

# winbase.h
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


class KeepAwakeStatus:
    """What actually happened, so the report never claims more than it did."""

    def __init__(self, active: bool, detail: str = "") -> None:
        self.active = active
        self.detail = detail

    def as_dict(self) -> dict:
        # The platform is recorded because "not active" means something
        # different on Windows (the call failed) than on Linux (there is
        # nothing here to call).
        return {"active": self.active, "detail": self.detail, "platform": sys.platform}


def _windows_set(flags: int) -> bool:
    import ctypes

    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    kernel32.SetThreadExecutionState.restype = ctypes.c_uint
    # Returns the previous state, or 0 on failure.
    return kernel32.SetThreadExecutionState(ctypes.c_uint(flags)) != 0


@contextmanager
def keep_awake() -> Iterator[KeepAwakeStatus]:
    """Ask the OS not to fall asleep on its own for the duration of the block."""
    if sys.platform != "win32":
        yield KeepAwakeStatus(False, f"unsupported platform: {sys.platform}")
        return

    try:
        ok = _windows_set(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    except Exception as exc:  # noqa: BLE001 - a failed request must not stop the run
        yield KeepAwakeStatus(False, f"{type(exc).__name__}: {exc}")
        return

    status = KeepAwakeStatus(bool(ok), "" if ok else "SetThreadExecutionState returned 0")
    try:
        yield status
    finally:
        if ok:
            try:
                _windows_set(ES_CONTINUOUS)   # release; sleep timers resume immediately
            except Exception:  # noqa: BLE001 - Windows also clears this when we exit
                pass
