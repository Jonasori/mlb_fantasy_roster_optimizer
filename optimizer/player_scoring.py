"""
Per-player scoring: FV, PV, MEW.

All functions are DataFrame enrichment: players in → players with new column(s) out.
Depends on config.
"""

import numpy as np
import pandas as pd

from .config import (
    MIN_STAT_STANDARD_DEVIATION,
)

# ============================================================================
# FANTASY VALUE (FV)
# ============================================================================


def add_fantasy_value(players: pd.DataFrame) -> pd.DataFrame:
    """Add 'FV' column: sum of z-scores across 5 relevant scoring categories.

    Hitters: z(R) + z(HR) + z(RBI) + z(SB) + z(OPS)
    Pitchers: z(W) + z(SV) + z(K) + z(−ERA) + z(−WHIP)

    z-scores are computed within each player_type population (all hitters
    in the DataFrame, all pitchers in the DataFrame). This includes FAs,
    so FV is comparable across rostered players and free agents.

    For negative stats (ERA, WHIP), negate BEFORE computing z-score so
    that lower ERA → higher z-score → higher FV.

    Requires: player_type, R, HR, RBI, SB, OPS, W, SV, K, ERA, WHIP.
    Adds: FV.
    """
    players = players.copy()
    players["FV"] = 0.0

    h_mask = players["player_type"] == "hitter"
    hitters = players.loc[h_mask]

    for stat in ("R", "HR", "RBI", "SB", "OPS"):
        std = hitters[stat].std()
        assert std > MIN_STAT_STANDARD_DEVIATION, (
            f"FV: Standard deviation of hitter {stat} is {std:.6f}, "
            f"below minimum {MIN_STAT_STANDARD_DEVIATION}. "
            f"Check that hitter projections have meaningful variance."
        )
        players.loc[h_mask, "FV"] += (hitters[stat] - hitters[stat].mean()) / std

    p_mask = players["player_type"] == "pitcher"
    pitchers = players.loc[p_mask]

    for stat in ("W", "SV", "K"):
        std = pitchers[stat].std()
        assert std > MIN_STAT_STANDARD_DEVIATION, (
            f"FV: Standard deviation of pitcher {stat} is {std:.6f}, "
            f"below minimum {MIN_STAT_STANDARD_DEVIATION}."
        )
        players.loc[p_mask, "FV"] += (pitchers[stat] - pitchers[stat].mean()) / std

    for stat in ("ERA", "WHIP"):
        std = pitchers[stat].std()
        assert std > MIN_STAT_STANDARD_DEVIATION, (
            f"FV: Standard deviation of pitcher {stat} is {std:.6f}, "
            f"below minimum {MIN_STAT_STANDARD_DEVIATION}."
        )
        players.loc[p_mask, "FV"] -= (pitchers[stat] - pitchers[stat].mean()) / std

    print(
        f"FV computed for {len(players)} players "
        f"(hitters: {h_mask.sum()}, pitchers: {p_mask.sum()})"
    )
    return players


# ============================================================================
# PERCEIVED VALUE (PV)
# ============================================================================

# WAR threshold above which a player gets a recognition premium.
# 3.0 WAR ≈ "above-average starter" — the floor for name recognition in trades.
_FAME_WAR_THRESHOLD: float = 3.0

# PV points per WAR above the threshold.  Calibrated so that elite SPs
# (5–6 WAR, FV ≈ 20) end up with PV above good hitters (4 WAR, FV ≈ 23).
# Without this correction FV systematically undervalues SPs because the
# z-score pools all pitchers (closers dominate the SV category, dragging
# SP z-score totals below hitter totals for the same real quality).
_FAME_WAR_SLOPE: float = 3.0


def add_perceived_value(
    players: pd.DataFrame,
    season_fraction_remaining: float = 1.0,
) -> pd.DataFrame:
    """Add 'PV' column: how opponents likely value a player in trade talks.

    PV = max(FV, 0) + max(WAR − threshold, 0) × slope

    FV is the base: smart opponents evaluate players by projected fantasy
    production, which is exactly what FV measures (z-score sum across the
    5 scoring categories for the player's type).

    WAR is now a REST-OF-SEASON projection, so it scales down as the season
    progresses (RoS WAR ≈ full-season WAR × fraction remaining). To keep the
    fame premium calibrated to the same full-season scale, the threshold is
    scaled by `season_fraction_remaining` and the slope by its inverse:

        threshold = 3.0 × f
        slope     = 3.0 / f          (f = season_fraction_remaining, floored)

    This is exact: a player whose full-season WAR is W has RoS WAR ≈ W·f, and
    max(W·f − 3f, 0) × (3/f) = max(W − 3, 0) × 3 — identical to the preseason
    premium. With f = 1.0 (default) behavior is unchanged.

    Args:
        players: DataFrame with FV and WAR columns.
        season_fraction_remaining: Fraction of the season still to be played
            (1.0 at season start, →0 at the end). Floored at 0.1 to keep the
            slope finite near season's end.

    The fame premium corrects two biases in raw FV:
      1. FV undervalues elite SPs relative to closers (closers dominate the
         SV z-score while SPs get zero) and relative to hitters (hitters
         contribute positively across all 5 categories).
      2. Trade markets overweight general quality / name recognition beyond
         pure category production.

    The per-player max constraint in trade_finder.py separately prevents
    aggregating mid-tier players to acquire a superstar ("you need to
    send a star to get a star").

    Requires: FV, WAR.
    Adds: PV.
    """
    players = players.copy()
    assert "FV" in players.columns, (
        "add_perceived_value: FV column required. Call add_fantasy_value first."
    )

    frac = max(season_fraction_remaining, 0.1)
    threshold = _FAME_WAR_THRESHOLD * frac
    slope = _FAME_WAR_SLOPE / frac

    fv = players["FV"].clip(lower=0)
    war = players["WAR"].fillna(0).clip(lower=0)

    fame = (war - threshold).clip(lower=0) * slope
    players["PV"] = fv + fame

    rostered_mask = players["owner"].notna()
    n_rostered = rostered_mask.sum()
    if n_rostered > 0:
        median_fame_pct = (
            fame[rostered_mask] / players.loc[rostered_mask, "PV"].clip(lower=0.1)
        ).median() * 100
    else:
        median_fame_pct = 0.0

    print(
        f"PV computed for {len(players)} players "
        f"(fame premium: median {median_fame_pct:.0f}% of PV for rostered)"
    )
    return players


# ============================================================================
# MARGINAL EXPECTED WINS (MEW)
# ============================================================================


def add_mew(
    players: pd.DataFrame,
    my_totals: dict[str, float],
    gradient: dict[str, float],
) -> pd.DataFrame:
    """Add 'MEW' column: first-order marginal EW contribution per player.

    MEW is the central player-evaluation metric. It uses the EW gradient
    to score every player — hitters and pitchers alike — in one unified
    formula with no conditional logic (MATHEMATICAL_FRAMEWORK §4):

        MEW(p) = Σ_{c ∈ C_count} g_c × stat_c(p)
               + g_OPS  × PA(p) × (OPS(p)  − my_OPS)  / total_PA
               + g_ERA  × IP(p) × (ERA(p)  − my_ERA)  / total_IP
               + g_WHIP × IP(p) × (WHIP(p) − my_WHIP) / total_IP

    where C_count = {R, HR, RBI, SB, W, SV, K}.

    No hitter/pitcher branching needed: for hitters, IP = 0 so all
    pitching terms vanish; for pitchers, PA = 0 so all hitting terms
    vanish. The data encodes the player type; the formula is universal.

    The gradient is a pre-computed input (from compute_league_state),
    NOT recomputed here.

    SIGN VERIFICATION:
        g_ERA < 0. Good pitcher: (ERA − my_ERA) < 0. Product: positive. ✓

    Args:
        players: DataFrame with stat columns.
        my_totals: Converged team totals dict. Must contain all 10 category
            keys plus 'PA' and 'IP'.
        gradient: Pre-computed ∂EW/∂(my_c) from compute_ew_gradient.

    Requires: PA, IP, R, HR, RBI, SB, OPS, W, SV, K, ERA, WHIP.
    Adds: MEW.
    """
    players = players.copy()

    total_pa = my_totals["PA"]
    total_ip = my_totals["IP"]
    my_ops = my_totals["OPS"]
    my_era = my_totals["ERA"]
    my_whip = my_totals["WHIP"]

    assert total_pa > 0, (
        f"add_mew: total_PA is {total_pa}. "
        f"my_totals must come from compute_totals_for_starters (includes PA)."
    )
    assert total_ip > 0, (
        f"add_mew: total_IP is {total_ip}. "
        f"my_totals must come from compute_totals_for_starters (includes IP)."
    )

    mew = pd.Series(0.0, index=players.index)

    for cat in ("R", "HR", "RBI", "SB", "W", "SV", "K"):
        mew += gradient[cat] * players[cat]

    mew += gradient["OPS"] * players["PA"] * (players["OPS"] - my_ops) / total_pa
    mew += gradient["ERA"] * players["IP"] * (players["ERA"] - my_era) / total_ip
    mew += gradient["WHIP"] * players["IP"] * (players["WHIP"] - my_whip) / total_ip

    players["MEW"] = mew

    # print(
    #     f"MEW computed for {len(players)} players "
    #     f"(range: {mew.min():.3f} to {mew.max():.3f})"
    # )
    return players
