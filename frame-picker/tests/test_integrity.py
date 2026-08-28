"""The end-of-run check: does everything the report claims actually exist?

Asked for directly after a 77-file run whose report rendered several frames
with no image and said nothing about it.
"""

from __future__ import annotations

import os

from framepicker import report
from framepicker import strings_lt as S


def _results(files):
    return {
        "clips": [{
            "probe": {"name": "clip.mp4"},
            "frames": [
                {"rank": i + 1, "file": name, "exported": True, "score": 0.7,
                 "timestamp": float(i), "reasons": []}
                for i, name in enumerate(files)
            ],
        }],
    }


def test_a_complete_run_reports_ok(tmp_path):
    for name in ("a.jpg", "b.jpg"):
        (tmp_path / name).write_bytes(b"x" * 10)
    results = _results(["a.jpg", "b.jpg"])
    verdict = report.verify(results, str(tmp_path), {"a.jpg": "data:...", "b.jpg": "data:..."})
    assert verdict["ok"] is True
    assert verdict["files_present"] == 2
    assert verdict["messages"] == [S.integrity_ok(2, 2, 2)]


def test_a_frame_whose_file_vanished_is_named(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x" * 10)
    results = _results(["a.jpg", "gone.jpg"])
    verdict = report.verify(results, str(tmp_path), {"a.jpg": "data:..."})
    assert verdict["ok"] is False
    assert verdict["files_missing"] == ["gone.jpg"]
    assert any("gone.jpg" in m for m in verdict["messages"])


def test_an_empty_file_is_not_counted_as_success(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"")
    verdict = report.verify(_results(["a.jpg"]), str(tmp_path), {})
    assert verdict["ok"] is False
    assert verdict["files_empty"] == ["a.jpg"]


def test_a_missing_preview_is_a_finding_not_a_blank_gap(tmp_path):
    """The exact defect seen in the real run: JPEG on disk, no image in the page."""
    (tmp_path / "a.jpg").write_bytes(b"x" * 10)
    verdict = report.verify(_results(["a.jpg"]), str(tmp_path), previews={})
    assert verdict["ok"] is False
    assert verdict["previews_missing"] == ["a.jpg"]
    assert any("a.jpg" in m for m in verdict["messages"])


def test_a_failed_export_is_named(tmp_path):
    results = _results(["a.jpg"])
    results["clips"][0]["frames"][0]["exported"] = False
    verdict = report.verify(results, str(tmp_path), {})
    assert verdict["ok"] is False
    assert verdict["failed_exports"]


def test_unreferenced_leftovers_are_reported_but_not_a_failure(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x" * 10)
    (tmp_path / "stale.jpg").write_bytes(b"x" * 10)
    verdict = report.verify(_results(["a.jpg"]), str(tmp_path), {"a.jpg": "data:..."})
    assert verdict["unreferenced_files"] == ["stale.jpg"]
    assert verdict["ok"] is True, "a leftover file is worth saying, not a broken run"


def test_a_frame_with_no_preview_still_renders_a_visible_marker(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x" * 10)
    results = _results(["a.jpg"])
    results["summary"] = {}
    results["integrity"] = report.verify(results, str(tmp_path), {})
    path = report.write_report_html(results, str(tmp_path), previews={})
    html = open(path, encoding="utf-8").read()
    assert "a.jpg" in html
    assert S.REPORT_INTEGRITY in html
    assert "<img" not in html, "no broken image tag; a stated finding instead"


def test_previews_are_built_from_real_exported_files(tmp_path):
    import cv2
    import numpy as np

    path = tmp_path / "a.jpg"
    cv2.imwrite(str(path), np.full((40, 60, 3), 120, dtype=np.uint8))
    previews = report.build_previews(_results(["a.jpg"]), str(tmp_path))
    assert previews["a.jpg"].startswith("data:image/jpeg;base64,")
    assert not report.build_previews(_results(["missing.jpg"]), str(tmp_path))
