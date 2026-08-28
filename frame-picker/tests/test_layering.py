"""Rules from section 9 of the task, enforced instead of trusted."""

from __future__ import annotations

import ast
import os
import sys
import textwrap

import pytest

from framepicker import proc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE_DIRS = (os.path.join(ROOT, "framepicker"), os.path.join(ROOT, "gui"), os.path.join(ROOT, "tests"))
STRINGS_MODULE = os.path.join(ROOT, "framepicker", "strings_lt.py")
PROC_MODULE = os.path.join(ROOT, "framepicker", "proc.py")

FRAMEPICKER_MODULES = (
    "framepicker",
    "framepicker.cli",
    "framepicker.decode",
    "framepicker.export",
    "framepicker.features",
    "framepicker.probe",
    "framepicker.proc",
    "framepicker.report",
    "framepicker.scoring",
    "framepicker.select",
    "framepicker.strings_lt",
)

#: Letters that exist in Lithuanian and in no language this codebase writes in.
LT_LETTERS = set("ąčęėįšųūžĄČĘĖĮŠŲŪŽ")
#: Lithuanian words that carry no diacritics, so the letter check would miss them.
LT_WORDS = (
    "kadras", "kadru", "kadrai", "failas", "failai", "failu", "atmesta", "irasas",
    "vaizdo", "nepavyko", "praleista", "rezultatai", "veidas", "veidu", "ivertis",
    "priezastis", "trukme", "spalvu", "apdorojama", "ataskaita",
)


def _python_files() -> list[str]:
    found = []
    for directory in PACKAGE_DIRS:
        for base, _, names in os.walk(directory):
            if "__pycache__" in base:
                continue
            found += [os.path.join(base, n) for n in sorted(names) if n.endswith(".py")]
    return found


def _imported_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def _attribute_calls(tree: ast.AST) -> set[str]:
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            value = node.func.value
            if isinstance(value, ast.Name):
                calls.add(f"{value.id}.{node.func.attr}")
    return calls


def test_no_direct_subprocess():
    """proc.py is the only module allowed to start a process."""
    offenders = []
    for path in _python_files():
        if os.path.abspath(path) == PROC_MODULE:
            continue
        tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
        if "subprocess" in _imported_names(tree):
            offenders.append((path, "imports subprocess"))
        banned = {"os.system", "os.popen", "os.spawnv", "os.execv", "pty.spawn"}
        for call in _attribute_calls(tree) & banned:
            offenders.append((path, call))
    assert offenders == [], offenders


def test_proc_is_the_module_that_does_import_subprocess():
    """A guard that would otherwise pass by deleting all the code."""
    tree = ast.parse(open(PROC_MODULE, encoding="utf-8").read(), filename=PROC_MODULE)
    assert "subprocess" in _imported_names(tree)


def test_no_hardcoded_lt_strings():
    """Every Lithuanian string lives in strings_lt.py."""
    offenders = []
    for path in _python_files():
        # strings_lt.py is the sanctioned home; this file is the checker and has
        # to spell out the alphabet and the word list it looks for.
        if os.path.abspath(path) in (STRINGS_MODULE, os.path.abspath(__file__)):
            continue
        text = open(path, encoding="utf-8").read()
        for number, line in enumerate(text.splitlines(), start=1):
            if LT_LETTERS & set(line):
                offenders.append((os.path.relpath(path, ROOT), number, line.strip()[:60]))
                continue
            lowered = line.lower()
            for word in LT_WORDS:
                if f" {word} " in lowered or f'"{word}' in lowered or f"'{word}" in lowered:
                    offenders.append((os.path.relpath(path, ROOT), number, word))
                    break
    assert offenders == [], offenders


def test_framepicker_never_imports_the_gui():
    """The layering only holds in one direction."""
    for path in _python_files():
        if not path.startswith(os.path.join(ROOT, "framepicker")):
            continue
        tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
        imported = _imported_names(tree)
        assert "gui" not in imported, path
        assert not {"PySide6", "PySide2", "PyQt5", "PyQt6"} & imported, path


def test_gui_contains_no_analysis_logic():
    """gui/ may wire widgets to framepicker; it may not compute anything itself."""
    gui_dir = os.path.join(ROOT, "gui")
    banned = {"numpy", "cv2", "subprocess", "json"}
    for path in _python_files():
        if not path.startswith(gui_dir):
            continue
        imported = _imported_names(ast.parse(open(path, encoding="utf-8").read(), filename=path))
        assert not banned & imported, (path, banned & imported)


HEADLESS_CHILD = textwrap.dedent(
    """
    import importlib
    import os
    import sys

    sys.path.insert(0, {root!r})
    os.environ.pop("DISPLAY", None)
    os.environ.pop("WAYLAND_DISPLAY", None)
    os.environ["QT_QPA_PLATFORM"] = "definitely-not-a-platform"

    BLOCKED = ("PySide6", "PySide2", "PyQt5", "PyQt6", "tkinter")

    class NoQt:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in BLOCKED:
                raise ImportError("framepicker imported a GUI toolkit: " + name)
            return None

    sys.meta_path.insert(0, NoQt())

    for name in {modules!r}:
        importlib.import_module(name)

    leaked = [m for m in sys.modules if m.split(".")[0] in BLOCKED]
    if leaked:
        raise SystemExit("GUI modules leaked into framepicker: " + repr(leaked))
    print("headless-ok")
    """
).format(root=ROOT, modules=FRAMEPICKER_MODULES)


def test_headless():
    """`import framepicker` works with no Qt and no display."""
    result = proc.run([sys.executable, "-c", HEADLESS_CHILD], timeout=120)
    assert result.ok, result.stderr_text(2000)
    assert b"headless-ok" in result.stdout


@pytest.mark.skipif(
    os.environ.get("FRAMEPICKER_SKIP_GUI_IMPORT") == "1", reason="GUI import check disabled"
)
def test_gui_imports_only_in_one_direction():
    """The GUI is allowed to import framepicker (it is skipped when PySide6 is absent)."""
    pytest.importorskip("PySide6")
    child = textwrap.dedent(
        """
        import sys
        sys.path.insert(0, {root!r})
        import gui.drop_window as window
        assert window.run_batch.__module__ == "framepicker.cli"
        print("gui-ok")
        """
    ).format(root=ROOT)
    result = proc.run([sys.executable, "-c", child], timeout=180)
    if not result.ok and "cannot open shared object file" in result.stderr_text(2000):
        pytest.skip("Qt shared libraries are not installed on this machine")
    assert result.ok, result.stderr_text(2000)
    assert b"gui-ok" in result.stdout
