"""
Offline tests for the Wayback projection harvester.
No network access. Per AGENTS.md: no classes, no fixtures, no mocking.
"""

import json

import pandas as pd

from data_prep.wayback import extract_next_data, validate_ros_capture


def _page(records: list[dict]) -> str:
    payload = {
        "props": {
            "pageProps": {
                "dehydratedState": {
                    "queries": [{"state": {"data": records}}]
                }
            }
        }
    }
    return (
        "<html><body>"
        f'<script id="__NEXT_DATA__" type="application/json">{json.dumps(payload)}</script>'
        "</body></html>"
    )


def test_extract_next_data_reads_embedded_records():
    records = [{"PlayerName": "Bobby Witt Jr.", "PA": 199, "HR": 8}]
    assert extract_next_data(_page(records)) == records, (
        "Failed to extract the embedded dataset from __NEXT_DATA__"
    )


def test_extract_returns_none_for_503_body():
    """A Wayback 503 has no __NEXT_DATA__ and must not look like an empty page."""
    body = "<html><body><h1>503 Service Temporarily Unavailable</h1></body></html>"
    assert extract_next_data(body) is None, (
        "A body with no __NEXT_DATA__ must return None so the caller can "
        "retry. Returning [] would silently record a false empty."
    )


def test_validate_rejects_non_decaying_pa():
    """RoS PA must fall as the season progresses; a rise means full-season data.

    Needs >= 10 overlapping players: validate_ros_capture short-circuits to
    "ok" whenever the merged overlap is smaller than that (real captures
    overlap on hundreds of players; below 10 there isn't enough signal to
    trust a ratio). Two players — as a naive version of this test would use —
    never clears that guard, so the median-ratio rejection path never runs
    and the "later_bad" case below would wrongly read as valid.
    """
    names = [f"P{i}" for i in range(12)]
    earlier_pa = [650.0, 600.0, 550.0, 500.0, 450.0, 400.0, 350.0, 300.0, 250.0, 200.0, 150.0, 100.0]
    earlier = pd.DataFrame({"PlayerName": names, "PA": earlier_pa})
    later_ok = pd.DataFrame({"PlayerName": names, "PA": [pa * 0.61 for pa in earlier_pa]})
    later_bad = pd.DataFrame({"PlayerName": names, "PA": [pa * 1.06 for pa in earlier_pa]})

    ok, reason = validate_ros_capture(later_ok, earlier)
    assert ok, f"A decaying capture was rejected: {reason}"

    ok, reason = validate_ros_capture(later_bad, earlier)
    assert not ok, (
        "A capture whose PA rose against an earlier one is mislabelled "
        "full-season data and must be rejected."
    )
    assert "pa" in reason.lower(), f"Reason should mention PA, got: {reason}"


def test_validate_rejects_empty_capture():
    ok, reason = validate_ros_capture(pd.DataFrame({"PlayerName": [], "PA": []}), None)
    assert not ok, "An empty capture must be rejected"
    assert "empty" in reason.lower(), f"Reason should mention emptiness, got: {reason}"


def test_validate_accepts_first_capture_with_no_predecessor():
    frame = pd.DataFrame({"PlayerName": ["A"], "PA": [650.0]})
    ok, reason = validate_ros_capture(frame, None)
    assert ok, f"The first capture of a season has nothing to compare to: {reason}"


def test_validate_tolerates_partial_row_counts():
    """Captures are partial pages, so row counts differ capture to capture.

    A capture covering only a subset of players — with a different total row
    count and some players missing entirely — must still validate on the
    overlap alone. Nothing here should compare len(frame) to len(previous):
    that would reject the common case of a smaller, later-page capture, not
    just the mislabelled-full-season case this validator exists to catch.
    """
    names = [f"Player{i}" for i in range(15)]
    earlier = pd.DataFrame({"PlayerName": names, "PA": [600.0] * 15})
    # Only 10 of the 15 players survive into this partial capture, all decayed.
    later = pd.DataFrame({"PlayerName": names[:10], "PA": [400.0] * 10})

    assert len(later) != len(earlier), (
        "Test setup error: this case only matters when row counts differ."
    )
    ok, reason = validate_ros_capture(later, earlier)
    assert ok, (
        f"A smaller, partial capture with decaying PA on the overlap was "
        f"rejected: {reason}. Coverage differences between captures are "
        f"expected and must not by themselves fail validation."
    )
