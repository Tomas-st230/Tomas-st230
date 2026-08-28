"""The written record of a run.

Two rules matter here. A log must never be able to stop a run - a full disk is
not a reason to lose the work. And the values in it have to be values: a log
you can only read is a log nobody can check.
"""

from __future__ import annotations

import json
import os

from framepicker import runlog


def _lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as handle:
        return [line for line in handle.read().splitlines() if line.strip()]


def test_messages_said_before_the_folder_exists_are_not_lost(tmp_path):
    log = runlog.RunLog()
    log.message("before")
    log.open(str(tmp_path))
    log.message("after")
    log.close()
    text = "\n".join(_lines(os.path.join(str(tmp_path), runlog.LOG_TXT)))
    assert "before" in text and "after" in text


def test_values_are_written_as_json_one_record_per_line(tmp_path):
    log = runlog.RunLog(str(tmp_path))
    log.event("clip", file="a.mp4", score=0.5, nested={"look": "nature"})
    log.event("frame", file="a_01.jpg", score=0.61)
    log.close()
    records = [json.loads(line) for line in _lines(os.path.join(str(tmp_path), runlog.LOG_JSONL))]
    assert [r["event"] for r in records] == ["clip", "frame"]
    assert records[0]["score"] == 0.5, "numbers must stay numbers"
    assert records[0]["nested"]["look"] == "nature"
    assert all("time" in r for r in records)


def test_a_log_that_cannot_be_written_does_not_raise(tmp_path):
    blocker = tmp_path / "blocked"
    blocker.write_text("I am a file, not a folder", encoding="utf-8")
    log = runlog.RunLog(str(blocker))
    log.message("this has nowhere to go")
    log.event("clip", file="a.mp4")
    log.close()
    assert log.errors, "the failure has to be recorded, not swallowed"


def test_logging_can_be_turned_off(tmp_path):
    log = runlog.RunLog(str(tmp_path), enabled=False)
    log.message("nothing")
    log.event("clip")
    log.close()
    assert log.paths == []
    assert os.listdir(str(tmp_path)) == []


def test_clip_values_are_flattened_from_the_record():
    record = {
        "probe": {"name": "a.mp4", "duration": 20.3, "width": 3840, "height": 2160,
                   "extra": {"format_tags": {"encoder": "DJI Lito X1"}}},
        "log": {"is_log": True, "source": "sidecar", "profile": "dlog",
                 "color_mode": {"value": "dlog_m"}, "statistics": {"luma_span": 0.6}},
        "color": {"mode": "lut", "lut": "x.cube"},
        "decode": {"path_used": "hw", "keyframes_only": True, "frames_yielded": 40},
        "look": {"requested": "auto", "name": "nature",
                  "auto": {"nature_score": 0.71, "city_score": 0.12, "decided": True}},
        "confidence": {"spread": 0.11, "informative": True},
        "rejects": {"total": 4, "blurry": 4},
        "selection": {"best_score": 0.57},
        "frames": [{"rank": 1, "file": "a_01.jpg", "score": 0.57, "timestamp": 9.0,
                     "features": {"components": {"content": 0.45}, "symmetry": 0.7}}],
        "elapsed_s": 6.8,
    }
    values = runlog.clip_event(0, "/card/a.mp4", record)
    assert values["file"] == "a.mp4"
    assert values["encoder_tag"] == "DJI Lito X1"
    assert values["color_md"] == "dlog_m"
    assert values["log_source"] == "sidecar"
    assert values["look_applied"] == "nature"
    assert values["look_nature_score"] == 0.71
    assert values["keyframes_only"] is True
    assert values["frames_delivered"] == 1

    frames = runlog.frame_events("a.mp4", record)
    assert len(frames) == 1
    assert frames[0]["components"]["content"] == 0.45
    assert frames[0]["symmetry"] == 0.7
