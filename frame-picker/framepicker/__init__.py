"""frame-picker - pick the best still frames out of video files.

``framepicker`` runs with no Qt and no display; ``gui`` may import it, never
the other way round (``tests/test_headless.py`` asserts this).
"""

from __future__ import annotations

__all__ = ["__version__", "Options", "run_batch", "main"]

__version__ = "0.1.0"


def __getattr__(name: str):
    # Lazy so that ``import framepicker`` stays cheap and import-safe.
    if name in ("Options", "run_batch", "main"):
        from . import cli

        return getattr(cli, name)
    raise AttributeError(name)
