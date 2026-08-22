"""Offline checks for the player-card renderer.

Only the renderer is tested. `facts` is pure network I/O against the MLB Stats
API, and a test that mocks it would assert that our mock matches our parser,
which is worth nothing. What IS worth guarding is the invariant the whole skill
rests on: a missing field must stop the render, not produce a blank panel that
a reader mistakes for "no ceiling".
"""

import importlib.util
import pathlib

import pytest

_CARD = pathlib.Path(__file__).parent.parent / ".claude/skills/player-card/card.py"
_spec = importlib.util.spec_from_file_location("card", _CARD)
card = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(card)


def _report() -> dict:
    return {
        "identity": {"name": "Test Player", "position": "OF", "org": "Org",
                     "age": 24, "level": "MLB", "headshot": "x.png"},
        "verdict": "TARGET",
        "ceiling": "35 HR",
        "median": "worth zero",
        "star_probability": "~15%",
        "argument_for": "bat speed",
        "argument_against": "squared-up rate",
        "unverified": ["PECOTA not configured"],
    }


def test_render_includes_every_narrative_field():
    html = card.render_html(_report())
    for value in ("Test Player", "TARGET", "35 HR", "worth zero", "~15%",
                  "bat speed", "squared-up rate", "PECOTA not configured"):
        assert value in html, f"renderer dropped {value!r}"


def test_missing_narrative_field_fails_loudly():
    for field in card.REPORT_NARRATIVE_FIELDS:
        broken = _report()
        del broken[field]
        with pytest.raises(AssertionError, match=field):
            card.render_html(broken)


def test_gap_renders_instead_of_empty_table():
    report = _report()
    report["gaps"] = {"tier2_table": "no measured tools — Savant is MLB-only"}
    html = card.render_html(report)
    assert "Savant is MLB-only" in html
    assert "<table>" not in html.split("measured tools")[0][-400:]
