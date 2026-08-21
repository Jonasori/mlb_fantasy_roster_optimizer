"""Harvest dated rest-of-season projections from the Wayback Machine.

FanGraphs' /projections page is server-rendered and embeds its full dataset
in <script id="__NEXT_DATA__">. The JSON API is paywalled; the page is not.

Every failure mode here is silent, so each is guarded explicitly:
  - A 503 body has no __NEXT_DATA__ and parses as "no data", not an error.
  - Captures are partial pages (587 rows observed against 1327 live).
  - Some captures labelled RoS contain full-season values.
  - CDX stores some URLs with a literal '&amp;'; substituting '&' returns a
    different capture.
"""

import json
import re
import time

import pandas as pd
import requests

_BROWSER_UA = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}
_NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.DOTALL
)
_CDX = "https://web.archive.org/cdx/search/cdx"
_PROJECTIONS_URL = "https://www.fangraphs.com/projections"

# Politeness policy. The Archive answers 503 when it wants a client to back
# off, and we treat that as a stop signal rather than something to retry
# through: one attempt per capture, a fixed delay between requests, and a hard
# ceiling on how many we make in a run. Fewer captures survive this way, and
# that is the intended trade.
_MAX_ATTEMPTS = 1
_DELAY_SECONDS = 2.0
_MAX_REQUESTS_PER_RUN = 40

# Rest-of-season endpoint codes, matching scrape_fangraphs.PROJECTION_TYPES.
ROS_TYPES: tuple[str, ...] = ("ratcdc", "steamerr", "rthebatx", "rzipsdc")


def extract_next_data(html: str) -> list[dict] | None:
    """Pull the embedded dataset out of a captured page.

    Returns None when the page carries no __NEXT_DATA__ at all — which is
    what a Wayback 503 looks like. Returning an empty list here would record
    a false empty and silently shrink the training set.
    """
    match = _NEXT_DATA_RE.search(html)
    if match is None:
        return None
    payload = json.loads(match.group(1))
    queries = payload["props"]["pageProps"]["dehydratedState"]["queries"]
    return queries[0]["state"]["data"]


def validate_ros_capture(
    frame: pd.DataFrame, previous: pd.DataFrame | None
) -> tuple[bool, str]:
    """Check that a capture is genuinely rest-of-season.

    RoS playing time decays as the season progresses. A capture whose PA rose
    against an earlier one from the same season is mislabelled full-season
    data — observed in the wild on a `type=steamerr` capture.

    Args:
        frame: The capture under test.
        previous: An earlier accepted capture from the same season, or None
            for the first one.

    Returns:
        (is_valid, reason). Reason is "ok" when valid.
    """
    if len(frame) == 0:
        return False, "empty capture — no rows"
    if "PA" not in frame.columns:
        return False, f"no PA column; got {list(frame.columns)[:8]}"
    if previous is None or len(previous) == 0:
        return True, "ok"

    key = "PlayerName" if "PlayerName" in frame.columns else "Name"
    if key not in frame.columns or key not in previous.columns:
        return True, "ok"

    merged = frame[[key, "PA"]].merge(
        previous[[key, "PA"]], on=key, suffixes=("_now", "_prev")
    )
    if len(merged) < 10:
        return True, "ok"

    median_ratio = float((merged["PA_now"] / merged["PA_prev"]).median())
    if median_ratio > 1.05:
        return False, (
            f"PA rose against the earlier capture (median ratio "
            f"{median_ratio:.2f}); this looks like full-season data "
            f"mislabelled as rest-of-season"
        )
    return True, "ok"


def list_captures(
    proj_type: str, stats: str = "bat", min_length: int = 150_000
) -> list[str]:
    """List Wayback timestamps holding a usable capture of one projection feed.

    Args:
        proj_type: A RoS endpoint code, e.g. "ratcdc".
        stats: "bat" or "pit".
        min_length: Skip captures smaller than this; small ones are error
            pages or empty shells.

    Returns:
        Timestamps (YYYYMMDDhhmmss), oldest first, one per distinct day.
    """
    response = requests.get(
        _CDX,
        params={
            # No trailing '*': matchType=prefix already requests prefix
            # matching. Adding a literal '*' on top makes the CDX server
            # search for that literal character in captured URLs and return
            # zero rows — confirmed empirically (0 rows with '*', 77k+
            # without, for the same prefix and matchType).
            "url": "www.fangraphs.com/projections",
            "matchType": "prefix",
            "filter": "statuscode:200",
            "output": "json",
            "collapse": "timestamp:8",
            "fl": "timestamp,original,length",
        },
        headers=_BROWSER_UA,
        timeout=120,
    )
    rows = response.json()
    assert rows and rows[0][0] == "timestamp", (
        f"list_captures: unexpected CDX response shape: {rows[:2]}"
    )

    wanted_type = f"type={proj_type}"
    wanted_stats = f"stats={stats}"
    timestamps = []
    for timestamp, original, length in rows[1:]:
        # CDX stores some URLs with a literal '&amp;'. Normalise for MATCHING
        # only — never for fetching, where the exact stored string matters.
        normalised = original.replace("&amp;", "&")
        if wanted_type not in normalised or wanted_stats not in normalised:
            continue
        if not length.isdigit() or int(length) < min_length:
            continue
        timestamps.append(timestamp)

    print(f"CDX: {len(timestamps)} usable captures for {proj_type}/{stats}")
    return sorted(timestamps)


def harvest_capture(
    timestamp: str, proj_type: str, stats: str = "bat"
) -> pd.DataFrame | None:
    """Fetch and parse one capture. One attempt, then give up.

    A 503 from the Archive means it wants fewer requests, so we take it at
    its word rather than retrying through it. Returns None when a capture
    does not come back — the caller decides whether a gap is tolerable.
    Never returns an empty frame for a failed fetch, so "no data" and
    "not fetched" stay distinguishable.
    """
    url = f"{_PROJECTIONS_URL}?type={proj_type}&stats={stats}&pos=all"
    response = requests.get(
        f"https://web.archive.org/web/{timestamp}id_/{url}",
        headers=_BROWSER_UA,
        timeout=120,
    )
    records = extract_next_data(response.text)
    if records:
        frame = pd.DataFrame(records)
        print(f"  {timestamp} {proj_type}/{stats}: {len(frame)} rows")
        return frame

    print(
        f"  {timestamp} {proj_type}/{stats}: no data (HTTP {response.status_code}) — skipping"
    )
    return None


def harvest_dates(
    timestamps: list[str], proj_type: str, stats: str = "bat"
) -> dict[str, pd.DataFrame]:
    """Harvest a list of captures serially, within the per-run request cap.

    Serial by design: one request at a time, a fixed pause between them, and
    at most `_MAX_REQUESTS_PER_RUN` requests. Anything beyond the cap is
    reported as skipped rather than silently dropped.
    """
    assert timestamps, "harvest_dates: no timestamps given."
    attempted = timestamps[:_MAX_REQUESTS_PER_RUN]
    skipped = len(timestamps) - len(attempted)
    if skipped:
        print(
            f"harvest_dates: attempting {len(attempted)} of {len(timestamps)} "
            f"captures ({skipped} beyond the {_MAX_REQUESTS_PER_RUN}-request "
            f"per-run cap; re-run to continue)"
        )

    harvested: dict[str, pd.DataFrame] = {}
    for index, timestamp in enumerate(attempted):
        frame = harvest_capture(timestamp, proj_type, stats)
        if frame is not None:
            harvested[timestamp] = frame
        if index + 1 < len(attempted):
            time.sleep(_DELAY_SECONDS)

    print(
        f"harvest_dates: {len(harvested)} of {len(attempted)} captures returned data"
    )
    return harvested
