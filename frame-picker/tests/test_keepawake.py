"""Keeping the machine awake must never take away the user's own controls."""

from __future__ import annotations

import sys

import pytest

from framepicker import keepawake


def test_the_context_manager_always_yields_a_status():
    with keepawake.keep_awake() as status:
        assert isinstance(status.active, bool)
        assert isinstance(status.as_dict(), dict)


@pytest.mark.skipif(sys.platform == "win32", reason="this asserts the non-Windows path")
def test_elsewhere_it_is_a_no_op_that_says_so():
    with keepawake.keep_awake() as status:
        assert status.active is False
        assert sys.platform in status.detail


def test_the_display_is_never_forced_on():
    """ES_DISPLAY_REQUIRED is deliberately not set: the screen must still go
    dark and the machine must still lock."""
    source = open(keepawake.__file__, encoding="utf-8").read()
    assert "ES_DISPLAY_REQUIRED" not in source.replace("``ES_DISPLAY_REQUIRED``", "")
    assert keepawake.ES_SYSTEM_REQUIRED == 0x00000001
    assert keepawake.ES_CONTINUOUS == 0x80000000


def test_a_failure_to_request_it_does_not_raise(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(keepawake, "_windows_set", lambda flags: (_ for _ in ()).throw(OSError("nope")))
    with keepawake.keep_awake() as status:
        assert status.active is False
        assert "nope" in status.detail


def test_the_request_is_released_on_the_way_out(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(keepawake, "_windows_set", lambda flags: calls.append(flags) or True)
    with keepawake.keep_awake() as status:
        assert status.active is True
    assert calls == [
        keepawake.ES_CONTINUOUS | keepawake.ES_SYSTEM_REQUIRED,
        keepawake.ES_CONTINUOUS,
    ], "the second call is what lets the machine sleep again"


def test_the_request_is_released_even_when_the_run_blows_up(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(keepawake, "_windows_set", lambda flags: calls.append(flags) or True)
    with pytest.raises(RuntimeError):
        with keepawake.keep_awake():
            raise RuntimeError("boom")
    assert calls[-1] == keepawake.ES_CONTINUOUS
