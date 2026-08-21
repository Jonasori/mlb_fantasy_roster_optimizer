"""Decompose raw counting stats into skill rates.

The spec's §2.5 finding is that two players with near-identical OPS can
decompose completely differently — one a real skill collapse, one batted-ball
luck — and that the stabilization constants differ by an order of magnitude
between components (K% M~49 PA vs BABIP M~433). Every consumer downstream of
here must see components, never composites.

GB_pct and HRFB are approximations: StatsAPI exposes groundOuts and airOuts,
not true batted-ball classifications, so these are out-rate proxies rather
than the FanGraphs definitions. They are directionally right and internally
consistent, which is what shrinkage needs; do not compare them to published
GB%/HR-FB values.
"""

import numpy as np
import pandas as pd

HITTING_SKILLS: tuple[str, ...] = ("K_pct", "BB_pct", "ISO", "BABIP", "SBA_rate")
PITCHING_SKILLS: tuple[str, ...] = (
    "K_pct", "BB_pct", "GB_pct", "HRFB", "BABIP_against",
)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Elementwise ratio, NaN where the denominator is zero or missing.

    A zero denominator must NOT yield zero: a 0-PA player with K_pct=0 reads
    as elite contact to any shrinkage that weights by reliability.
    """
    denom = denominator.where(denominator > 0)
    return (numerator / denom).astype(float)


def add_skill_rates(stats: pd.DataFrame) -> pd.DataFrame:
    """Add per-skill rate columns to a parsed stats frame.

    Requires columns: group, and for hitting rows PA/AB/H/HR/SB/CS/BB/SO/slg/avg/babip;
    for pitching rows BF/SOA/BBA/HRA/groundOuts/airOuts/babip.

    Adds columns: K_pct, BB_pct, ISO, BABIP, SBA_rate, n_PA (hitting rows);
    K_pct, BB_pct, GB_pct, HRFB, BABIP_against, n_BF (pitching rows).
    Rows of the other group get NaN in that group's columns.
    """
    stats = stats.copy()
    assert "group" in stats.columns, (
        "add_skill_rates: frame has no 'group' column. Pass the output of "
        "parse_stat_splits, which tags every row 'hitting' or 'pitching'."
    )

    is_hit = stats["group"] == "hitting"
    is_pit = stats["group"] == "pitching"
    assert (is_hit | is_pit).all(), (
        f"add_skill_rates: unexpected group values "
        f"{sorted(set(stats.loc[~(is_hit | is_pit), 'group']))}."
    )

    for col in (*HITTING_SKILLS, *PITCHING_SKILLS, "n_PA", "n_BF"):
        stats[col] = np.nan

    if is_hit.any():
        h = stats.loc[is_hit]
        stats.loc[is_hit, "K_pct"] = _safe_ratio(h["SO"], h["PA"])
        stats.loc[is_hit, "BB_pct"] = _safe_ratio(h["BB"], h["PA"])
        stats.loc[is_hit, "ISO"] = (h["slg"] - h["avg"]).astype(float)
        stats.loc[is_hit, "BABIP"] = h["babip"].astype(float)
        stats.loc[is_hit, "SBA_rate"] = _safe_ratio(h["SB"] + h["CS"], h["PA"])
        stats.loc[is_hit, "n_PA"] = h["PA"].astype(float)

    if is_pit.any():
        p = stats.loc[is_pit]
        stats.loc[is_pit, "K_pct"] = _safe_ratio(p["SOA"], p["BF"])
        stats.loc[is_pit, "BB_pct"] = _safe_ratio(p["BBA"], p["BF"])
        stats.loc[is_pit, "GB_pct"] = _safe_ratio(
            p["groundOuts"], p["groundOuts"] + p["airOuts"]
        )
        stats.loc[is_pit, "HRFB"] = _safe_ratio(p["HRA"], p["airOuts"])
        stats.loc[is_pit, "BABIP_against"] = p["babip"].astype(float)
        stats.loc[is_pit, "n_BF"] = p["BF"].astype(float)

    n_hit = int(is_hit.sum())
    n_pit = int(is_pit.sum())
    print(f"skill rates: {n_hit} hitting rows, {n_pit} pitching rows")
    return stats
