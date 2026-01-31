# Position Sensitivity Analysis

## Overview

The Position Sensitivity Analysis answers the question: **"Where should I focus my roster improvements to maximize expected wins?"**

Unlike SGP-based rankings (which are context-free), this analysis uses **Expected Wins Added (EWA)** — a team-specific metric that accounts for your roster's category strengths and weaknesses.

**Module:** `optimizer/roster_optimizer.py` (functions added at end of file)

**Key Insight:** SGP is a poor proxy for value at certain positions, especially relief pitchers. See [Why SGP Fails for Relief Pitchers](#why-sgp-fails-for-relief-pitchers) below.

**Note:** EWA uses lineup-aware totals (see [01a_config.md](01a_config.md#key-design-concept-lineup-aware-totals)).

---

## Cross-References

**Depends on:**
- [00_agent_guidelines.md](00_agent_guidelines.md) — code style
- [01a_config.md](01a_config.md) — `SLOT_ELIGIBILITY`, category constants
- [01b_fangraphs_loading.md](01b_fangraphs_loading.md) — `compute_team_totals()` (lineup-aware)
- [02_free_agent_optimizer.md](02_free_agent_optimizer.md) — optimizer infrastructure
- [03_trade_engine.md](03_trade_engine.md) — `compute_win_probability()` for EWA calculation

**Used by:**
- [04_visualizations.md](04_visualizations.md) — `plot_position_sensitivity_dashboard()`, related charts
- [05_notebook_integration.md](05_notebook_integration.md) — position analysis workflow
- [06_streamlit_dashboard.md](06_streamlit_dashboard.md) — My Team page analysis

---

## Imports

```python
# These are added to the existing roster_optimizer.py imports
from .trade_engine import compute_win_probability
```

---

## Core Functions

### compute_position_sensitivity

```python
def compute_position_sensitivity(
    my_roster_names: set[str],
    opponent_roster_names: set[str],
    projections: pd.DataFrame,
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
) -> dict:
    """
    Compute position-by-position sensitivity analysis using Expected Wins Added (EWA).

    This answers: "What's the EWA from upgrading at each position?"

    For each position slot:
    1. Identify all eligible players and mark status (my_roster, opponent, available)
    2. Compute baseline expected wins using current roster
    3. For sample of available candidates, compute EWA if they replaced my worst player

    Args:
        my_roster_names: Set of player names on my roster
        opponent_roster_names: Set of ALL player names on ANY opponent roster
        projections: Full projections DataFrame with SGP column
        opponent_totals: Dict mapping team_id to category totals
        category_sigmas: Standard deviations per category (from estimate_projection_uncertainty)

    Returns:
        Dict with keys:
            - 'slot_data': Dict[str, pd.DataFrame] mapping slot name to eligible players
                           Each DataFrame has columns: Name, SGP, percentile, status
                           where status ∈ {'my_roster', 'opponent', 'available'}
            - 'ewa_df': DataFrame with swap scenarios and EWA values
            - 'sensitivity_df': DataFrame summarizing each position's upgrade potential
            - 'baseline_expected_wins': float
            - 'baseline_roto_points': int

    Implementation Steps:

    1. Compute baseline:
        my_totals = compute_team_totals(my_roster_names, projections)
        _, baseline_diag = compute_win_probability(my_totals, opponent_totals, category_sigmas)
        baseline_ew = baseline_diag["expected_wins"]

    2. Build slot_data for each position:
        For each slot in SLOT_ELIGIBILITY:
            - Filter projections to eligible players (handle multi-position via set intersection)
            - Mark status: 'available' by default, then:
                - Mark players in opponent_roster_names as 'opponent'
                - Mark players in my_roster_names as 'my_roster'
            - **Filter out low-PA/IP free agents** (keep all rostered players):
                - Hitters: PA >= 50 (MIN_PA_FOR_FA)
                - Pitchers: IP >= 20 (MIN_IP_FOR_FA)
                - This reduces UTIL from 4000+ to ~500, RP from 4000+ to ~400
            - Sort by SGP descending
            - Compute rank and percentile

    3. Compute EWA for sample upgrade scenarios:
        For each position slot:
            - Get my worst player (lowest SGP among my_roster)
            - Sample available players at ranks [1, 5, 10, 20, 50]
            - For each candidate:
                - new_roster = (my_roster_names - {my_worst}) | {candidate}
                - new_totals = compute_team_totals(new_roster, projections)
                - ewa = new_expected_wins - baseline_expected_wins

    4. Compute sensitivity summary:
        For each position:
            - ewa_per_sgp = mean(EWA) / mean(SGP delta) for positive upgrades
            - better_fas_count = number of available players with SGP > my_worst_sgp
            - best_fa_ewa = EWA from best available upgrade
            - best_fa_sgp_gap = SGP difference to best available

    Print:
        "Baseline expected wins: {X:.1f} / 60"
        "Computing position sensitivities..."

    Use tqdm for progress: "Positions: {slot}"
    """
```

### sensitivity_df Columns

The `sensitivity_df` DataFrame has these columns:

| Column | Type | Description |
|--------|------|-------------|
| `slot` | str | Position slot (C, 1B, 2B, SS, 3B, OF, UTIL, SP, RP) |
| `ewa_per_sgp` | float | Sensitivity: EWA gained per 1 SGP improvement |
| `my_worst_name` | str | Name of worst rostered player at this position |
| `my_worst_sgp` | float | SGP of worst rostered player |
| `better_fas_count` | int | **Number of available FAs better than my worst** |
| `best_fa_sgp_gap` | float | SGP difference to best available FA |
| `best_fa_ewa` | float | **EWA from upgrading to best available FA** |

**Key insight:** `better_fas_count` and `best_fa_ewa` directly answer "Is there an upgrade available?" without relying on misleading percentile metrics.

---

### compute_percentile_sensitivity

```python
def compute_percentile_sensitivity(
    my_roster_names: set[str],
    projections: pd.DataFrame,
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
    slot_data: dict[str, pd.DataFrame],
    baseline_expected_wins: float,
    include_all_active: bool = True,
) -> pd.DataFrame:
    """
    Compute EWA at different percentile levels for each position.

    Players are ranked by EWA (not SGP), so x=EWA percentile, y=EWA is monotonic.
    This answers: "What's the slope of the EWA curve at position X?"

    Args:
        my_roster_names: Set of player names on my roster
        projections: Full projections DataFrame
        opponent_totals: Dict mapping team_id to category totals
        category_sigmas: Standard deviations per category
        slot_data: Output from compute_position_sensitivity
        baseline_expected_wins: Baseline expected wins (avoids recomputation)
        include_all_active: If True, include opponent roster players (for trade context).
                           If False, only include available (FA) players.
                           Default: True

    Returns:
        DataFrame with columns:
            - slot: Position slot
            - ewa_pctl: Actual EWA percentile (ranked by EWA, not SGP)
            - target_pctl: Target percentile sampled
            - candidate: Player name at this percentile
            - candidate_sgp: Player's SGP (for reference)
            - ewa: Expected Wins Added from this upgrade
            - status: 'available' (FA) or 'opponent' (trade target)
            - my_worst_ewa_pctl: My worst player's percentile RANKED BY EWA

    Implementation:

    For each slot:
        1. Get my worst player (lowest SGP)
        2. Get active players: if include_all_active, use both 'available' and 'opponent';
           otherwise use only 'available'
        3. Compute EWA for EVERY active player (swap my worst → candidate)
        4. Rank active players by EWA descending (highest EWA = rank 1)
        5. Compute ewa_pctl = 100 * (1 - rank / n_active)
        6. Compute my worst player's EWA percentile:
            my_worst_ewa = 0.0  # (swapping for yourself = no change)
            better_ewa_count = (active["ewa"] > 0).sum()
            my_worst_ewa_pctl = 100 * (1 - better_ewa_count / n_active)
        7. Sample at target percentiles [95, 90, 85, ..., 10]:
            - Find player closest to target_pctl in EWA-ranked pool
            - Record ewa, status, candidate info

    ⚠️ KEY DESIGN CHOICE: Ranking by EWA (not SGP) ensures monotonic curves.
    The slope of the curve directly answers "how valuable is this position to upgrade?"

    ⚠️ TRADE CONTEXT: By default, opponent roster players are included.
    Red points on the plot indicate trade targets (players on opponent rosters).
    To see only FA opportunities, set include_all_active=False.

    Use tqdm for progress: "EWA percentile analysis"
    """
```

---

## Why SGP Fails for Relief Pitchers

### The Problem

SGP (Standing Gain Points) ranks players by overall value across all categories. For hitters, SGP correlates highly with EWA (0.997) because all hitting stats are correlated — better hitters contribute to all categories.

For relief pitchers, SGP correlates poorly with EWA (0.591) because:

1. **Saves are binary:** Only closers get saves; setup men get ~0
2. **Saves don't correlate with other stats:** K, ERA, WHIP are independent of closer role
3. **SGP treats saves like any other stat:** A high-K/low-ERA reliever gets high SGP even with 0 saves

### Evidence

| Metric | Correlation with RP SGP |
|--------|------------------------|
| K (strikeouts) | 0.947 |
| SV (saves) | 0.686 |

This means Reid Detmers (0 SV, 157 K, 6.70 SGP) ranks higher by SGP than many actual closers.

### Correlation: Ranking Method vs Actual Value (EWA)

| Position | SGP→EWA Correlation | Best Proxy |
|----------|---------------------|------------|
| Hitters (OF, 1B, etc.) | 0.997 | SGP works great |
| Relief Pitchers | 0.591 | **Saves ranking (0.925)** |

### Example: Team Weak in Saves

For a team with only 31 projected saves (losing SV category against 4 of 6 opponents):

| Player | Saves | SGP | EWA |
|--------|-------|-----|-----|
| Reid Detmers | 0 | 6.70 | **-0.50** (hurts!) |
| Cade Smith | 29 | 8.00 | **+1.88** |
| David Bednar | 32 | 7.80 | **+1.74** |

EWA correctly penalizes Reid Detmers for not addressing the team's save weakness, even though his SGP is high.

### Why Cade Smith Beats Bednar (Despite Fewer Saves)

Even though Bednar has 3 more saves, Cade Smith has higher EWA because:

| Category | Cade Δ | Bednar Δ |
|----------|--------|----------|
| SV | +25.9% win prob | +29.1% |
| ERA | +1.1% | **-1.8%** (hurts!) |
| WHIP | +3.7% | +1.6% |
| K | +0.6% | +0.2% |

Bednar's ERA (3.30) is worse than both the player being replaced (3.04) and Cade Smith (2.98). The ratio stat degradation costs more than the extra 3 saves provide.

**This is correct behavior.** EWA accounts for all category impacts, not just the one you're targeting.

---

## Integration with Notebook

### Required Imports

Add to the notebook's import cell:

```python
# Free agent optimizer
from optimizer.roster_optimizer import (
    # ... existing imports ...
    compute_position_sensitivity,
    compute_percentile_sensitivity,
)

# Visualizations
from optimizer.visualizations import (
    # ... existing imports ...
    plot_position_sensitivity_dashboard,
    plot_percentile_ewa_curves,
    plot_position_distributions,
    plot_upgrade_opportunities,
)
```

### Compute category_sigmas Early

Update the `compute_totals` cell to also compute `category_sigmas`:

```python
@app.cell
def compute_totals(..., estimate_projection_uncertainty):
    my_totals = compute_team_totals(my_roster_names, projections)
    opponent_rosters_indexed = {i+1: names for i, (_, names) in enumerate(opponent_rosters.items())}
    opponent_totals = compute_all_opponent_totals(opponent_rosters_indexed, projections)
    
    # Compute early so it's available for position sensitivity
    category_sigmas = estimate_projection_uncertainty(my_totals, opponent_totals)
    
    return category_sigmas, my_totals, opponent_rosters_indexed, opponent_totals
```

### Position Sensitivity Cells

```python
@app.cell
def position_sensitivity_compute(
    category_sigmas, compute_percentile_sensitivity, compute_position_sensitivity,
    mo, my_roster_names, opponent_roster_names, opponent_totals, projections,
):
    """Compute position-by-position sensitivity analysis."""
    mo.md("Computing position sensitivities...")

    sensitivity_results = compute_position_sensitivity(
        my_roster_names=my_roster_names,
        opponent_roster_names=opponent_roster_names,
        projections=projections,
        opponent_totals=opponent_totals,
        category_sigmas=category_sigmas,
    )

    slot_data = sensitivity_results["slot_data"]
    ewa_df = sensitivity_results["ewa_df"]
    sensitivity_df = sensitivity_results["sensitivity_df"]
    baseline_ew = sensitivity_results["baseline_expected_wins"]

    pctl_ewa_df = compute_percentile_sensitivity(
        my_roster_names=my_roster_names,
        projections=projections,
        opponent_totals=opponent_totals,
        category_sigmas=category_sigmas,
        slot_data=slot_data,
        baseline_expected_wins=baseline_ew,
    )

    print(f"\nBaseline expected wins: {baseline_ew:.1f} / 60")
    print("\nPosition Upgrade Opportunities:")
    print(sensitivity_df[["slot", "my_worst_name", "better_fas_count", "best_fa_sgp_gap", "best_fa_ewa"]].to_string(index=False))

    return baseline_ew, ewa_df, pctl_ewa_df, sensitivity_df, slot_data
```

---

## Validation Checklist

- [ ] `compute_position_sensitivity` marks players with correct status (my_roster, opponent, available)
- [ ] Opponent players are correctly excluded from "available" pool
- [ ] `my_worst` player identified using `SGP.idxmin()` (not relying on sort order)
- [ ] EWA computed via `compute_win_probability` from trade_engine
- [ ] `sensitivity_df` includes `better_fas_count` (not misleading percentile)
- [ ] `compute_percentile_sensitivity` computes percentiles among AVAILABLE players only
- [ ] `my_worst_avail_pctl` computed within available pool, not all players
- [ ] `baseline_expected_wins` passed as parameter (not recomputed)
- [ ] tqdm used for progress reporting on both functions
- [ ] Notebook cells pass `category_sigmas` from `compute_totals`

---

## References

- Rosenof, Z. (2025). "Optimizing for Rotisserie Fantasy Basketball." arXiv:2501.00933.
- Smart Fantasy Baseball. "How to Analyze SGP Denominators."
