# Trade Engine

## Overview

The Trade Engine answers the question: **"Which trades improve my chances of winning the rotisserie league while being fair enough for opponents to accept?"**

Unlike the Free Agent Optimizer which uses MILP for global optimization, the Trade Engine uses a **probabilistic win model** to evaluate marginal player value and identify beneficial trades.

**Module:** `optimizer/trade_engine.py`

**Key insight:** In rotisserie scoring, a player's value is *context-dependent*. Production in a category where you're safely ahead has near-zero marginal value. Production in a category where you're in a close race is extremely valuable. The trade engine exploits this asymmetry.

---

## Cross-References

**Depends on:**
- [00_agent_guidelines.md](00_agent_guidelines.md) — code style, fail-fast philosophy
- [01a_config.md](01a_config.md) — `compute_team_totals()`, `estimate_projection_uncertainty()`, category constants
- [01d_database.md](01d_database.md) — `get_projections()`, `get_roster_names()`
- [01e_dynasty_valuation.md](01e_dynasty_valuation.md) — `dynasty_SGP` for trade fairness

**Used by:**
- [02b_position_sensitivity.md](02b_position_sensitivity.md) — `compute_win_probability()` for EWA
- [04_visualizations.md](04_visualizations.md) — trade impact visualizations
- [05_notebook_integration.md](05_notebook_integration.md) — trade analysis workflow
- [06_streamlit_dashboard.md](06_streamlit_dashboard.md) — Trades page

---

## Theoretical Foundation

This implementation is based on the probabilistic rotisserie optimization framework developed by Rosenof (2025):

> Rosenof, Z. (2025). "Optimizing for Rotisserie Fantasy Basketball." arXiv:2501.00933.

The paper establishes that:
1. **Direct computation of win probability is intractable** (~10^77 scenarios for typical leagues)
2. **A tractable approximation exists** based on modeling team totals as normal distributions
3. **Variance matters** — teams with higher variance in fantasy point totals have better chances of achieving the exceptional performance needed to win
4. **The approximation is differentiable** — allowing analytical computation of marginal player value

This is fundamentally superior to heuristic approaches (like counting "contested races") because it:
- Uses continuous probabilities instead of arbitrary thresholds
- Properly accounts for uncertainty and variance
- Is mathematically principled rather than ad-hoc

---

## Imports and Configuration

```python
import numpy as np
import pandas as pd
from scipy import stats
from tqdm.auto import tqdm

from .config import (
    SGP_METRIC,
    TRADE_FAIRNESS_THRESHOLD_PERCENT,
    TRADE_LOSE_COST_SCALE,
    TRADE_MAX_SIZE,
    TRADE_MIN_MEANINGFUL_IMPROVEMENT,
    MIN_STAT_STANDARD_DEVIATION,
)
from .data_loader import (
    HITTING_CATEGORIES,
    PITCHING_CATEGORIES,
    ALL_CATEGORIES,
    NEGATIVE_CATEGORIES,
    RATIO_STATS,
    MIN_HITTERS,
    MAX_HITTERS,
    MIN_PITCHERS,
    MAX_PITCHERS,
    NUM_OPPONENTS,
    compute_team_totals,
    estimate_projection_uncertainty,
)

# === TRADE ENGINE CONFIGURATION ===

# Expected value and variance of maximum of N standard normals
# From Teichroew (1956), used by Rosenof (2025)
# MEV[n] = E[max(X_1, ..., X_n)] where X_i ~ N(0,1)
# MVAR[n] = Var[max(X_1, ..., X_n)]
# We only need n=6 for our 6-opponent league, but include extras for flexibility
MEV_TABLE = {1: 0.0, 2: 0.564, 3: 0.846, 4: 1.029, 5: 1.163, 6: 1.267}
MVAR_TABLE = {1: 1.0, 2: 0.682, 3: 0.559, 4: 0.492, 5: 0.448, 6: 0.416}

# Trade engine configuration loaded from config.json
# These values are defined in the "trade_engine" section of config.json
FAIRNESS_THRESHOLD_PERCENT = TRADE_FAIRNESS_THRESHOLD_PERCENT
MAX_TRADE_SIZE = TRADE_MAX_SIZE
MIN_MEANINGFUL_IMPROVEMENT = TRADE_MIN_MEANINGFUL_IMPROVEMENT
MIN_STD = MIN_STAT_STANDARD_DEVIATION
LOSE_COST_SCALE = TRADE_LOSE_COST_SCALE

# SGP metric selection: loaded from config.json
# The trade engine uses SGP_METRIC to select which column to use:
# - SGP_METRIC == "raw": use projections["SGP"] (single-season, default)
# - SGP_METRIC == "dynasty": use projections["dynasty_SGP"] (age-adjusted)
# 
# Raw SGP is simpler and more predictable. Dynasty SGP accounts for aging
# curves and is useful for dynasty leagues that value future production.
```

---

## Mathematical Framework

### Notation

```
|C| = 10        (number of categories)
|O| = 6         (number of opponents)
μ_c,o           (normalized gap in category c against opponent o)
σ_c             (standard deviation of team strengths in category c)
Φ(x)            (standard normal CDF)
φ(x)            (standard normal PDF)
```

### Win Probability Model

The probability of winning the rotisserie league is approximated as:

```
V = Φ(μ_D / σ_D)
```

Where:
- `μ_D` = expected differential between my fantasy points and the best opponent
- `σ_D` = standard deviation of that differential

### Expected Fantasy Points (μ_T)

The expected number of matchup wins:

```
μ_T = Σ_c Σ_o Φ(μ_c,o)
```

Where `Φ(μ_c,o)` is the probability of beating opponent o in category c.

### Variance of Fantasy Points (σ_T²)

```
σ_T² = Σ_c Σ_o Φ(μ_c,o)(1 - Φ(μ_c,o)) + correlation_adjustment
```

**Key insight:** The term `Φ(1-Φ)` is maximized when `Φ = 0.5` (50-50 matchups). This means contested races contribute the most variance, which is valuable for winning.

### Target to Beat (μ_L, σ_L)

The target is the maximum of opponent totals:

```
μ_L = MEV(|O|) * √(E[σ_M²])
σ_L² = E[σ_M²] * MVAR(|O|)
```

### Differential Distribution

```
μ_D = μ_T * (|O|+1)/|O| - |C|*(|O|+1)/2 - μ_L
σ_D² = ((|O|+1)/|O|) * σ_T² + σ_L²
```

### Marginal Value via Numerical Differentiation

**Simplified approach (recommended for implementation):**

Rather than implementing the complex analytical gradient, use numerical differentiation with expected wins:

```python
def compute_marginal_value_numerical(player_name, my_totals, opponent_totals, category_sigmas, projections):
    """Compute EWA by actually computing expected_wins before and after adding player."""
    _, diag_before = compute_win_probability(my_totals, opponent_totals, category_sigmas)
    ew_before = diag_before["expected_wins"]
    
    new_totals = compute_totals_with_player(my_totals, player_name, projections)
    _, diag_after = compute_win_probability(new_totals, opponent_totals, category_sigmas)
    ew_after = diag_after["expected_wins"]
    
    return ew_after - ew_before  # EWA: Expected Wins Added
```

This is O(1) per player (just two expected wins calculations) and avoids gradient implementation complexity.

**For reference, the analytical gradient from Rosenof (2025) is:**

```
∇_{c,o}(V) ≈ φ(μ_D/σ_D) / σ_D * ∇_{c,o}(μ_D)
```

Where `∇_{c,o}(μ_D) = (|O|+1)/|O| * φ(μ_c,o)` — the PDF evaluated at the normalized gap.

The full gradient also includes variance terms but the mean term dominates. Use numerical differentiation unless performance requires analytical gradients.

---

## Core Functions

### Gap and Probability Computation

```python
def compute_normalized_gaps(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
) -> dict[str, dict[int, float]]:
    """
    Compute normalized gaps for each (category, opponent) pair.
    
    The normalized gap μ_c,o is:
        (my_total[c] - opp_total[o][c]) / (σ_c * √2)
    
    For NEGATIVE_CATEGORIES (ERA, WHIP), flip the sign so positive = winning.
    
    The √2 factor ensures the difference of two team totals has unit variance
    when each team's total has variance σ_c².
    
    Args:
        my_totals: My category totals
        opponent_totals: Dict of opponent totals
        category_sigmas: Standard deviation per category (from estimate_projection_uncertainty)
    
    Returns:
        Dict[category, Dict[opponent_id, normalized_gap]]
        
    Example:
        {'R': {1: 0.85, 2: -0.23, ...}, 'ERA': {1: 0.42, ...}, ...}
    """


def compute_matchup_probabilities(
    normalized_gaps: dict[str, dict[int, float]],
) -> dict[str, dict[int, float]]:
    """
    Compute probability of winning each matchup.
    
    P(beat opponent o in category c) = Φ(μ_c,o)
    
    Returns:
        Dict[category, Dict[opponent_id, probability]]
        All values in [0, 1].
    """
```

### Win Probability Computation

```python
def compute_win_probability(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
    category_correlations: dict[tuple[str, str], float] | None = None,
) -> tuple[float, dict]:
    """
    Compute probability of winning the rotisserie league.
    
    Implements the Rosenof (2025) tractable approximation.
    
    Args:
        my_totals: My team's category totals
        opponent_totals: Opponent totals (6 opponents)
        category_sigmas: Standard deviation per category
        category_correlations: Optional correlations between categories
                               (default: assume independence)
    
    Returns:
        V: Victory probability (0 to 1)
        diagnostics: Dict with intermediate values:
            - 'normalized_gaps': μ_c,o values
            - 'matchup_probs': Φ(μ_c,o) values  
            - 'mu_T': Expected fantasy points
            - 'sigma_T_sq': Variance of fantasy points
            - 'mu_L': Expected lead of best opponent
            - 'sigma_L_sq': Variance of that lead
            - 'mu_D': Expected differential
            - 'sigma_D': Std dev of differential
            - 'expected_wins': μ_T (out of 60)
            - 'expected_roto_points': Estimated roto points
    
    Implementation:
        1. Compute normalized gaps μ_c,o
        2. Compute matchup probabilities Φ(μ_c,o)
        3. Compute μ_T = Σ Φ(μ_c,o)
        4. Compute σ_T² = Σ Φ(1-Φ) + correlation_adjustment
        5. Compute E[σ_M²] for opponent variance
        6. Compute μ_L = MEV(6) * √(E[σ_M²])
        7. Compute σ_L² = E[σ_M²] * MVAR(6)
        8. Compute μ_D and σ_D
        9. Return V = Φ(μ_D / σ_D)
    
    Print:
        "Win probability: {V:.1%}"
        "  Expected wins: {mu_T:.1f}/60"
        "  Differential: {mu_D:.1f} ± {sigma_D:.1f}"
    """
```

### Marginal Value Computation

```python
def compute_win_probability_gradient(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
) -> dict[str, dict[int, float]]:
    """
    Compute the gradient of win probability with respect to each matchup gap.
    
    ∇_{c,o}(V) tells us how much win probability changes per unit change
    in the normalized gap for (category c, opponent o).
    
    Returns:
        Dict[category, Dict[opponent_id, gradient]]
        
    Note:
        Positive gradient means improving in this matchup increases V.
        The magnitude indicates sensitivity — large gradients are high-value targets.
    """


def compute_player_marginal_value(
    player_name: str,
    projections: pd.DataFrame,
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
    is_acquisition: bool = True,
) -> tuple[float, dict]:
    """
    Compute the marginal change in expected wins from acquiring/losing a player.
    
    Args:
        player_name: Player to evaluate (with -H/-P suffix)
        projections: Projections DataFrame
        my_totals: Current team totals
        opponent_totals: Opponent totals
        category_sigmas: Category standard deviations
        is_acquisition: True if acquiring, False if losing this player
    
    Returns:
        ewa: Expected Wins Added (change in expected category wins out of 60)
        breakdown: Dict with ew_before, ew_after
    
    Implementation:
        1. Get player's stats from projections
        2. Compute expected wins before via compute_win_probability
        3. Compute new totals with player change
        4. Compute expected wins after
        5. ewa = ew_after - ew_before
        
    Note:
        This is O(|C| × |O|) = O(60), very fast.
        For acquiring: player contributes positively
        For losing: player contribution is removed (flip sign)
    """
```

### Player Value DataFrame

```python
def compute_player_values(
    player_names: set[str],
    my_roster_names: set[str],
    projections: pd.DataFrame,
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
) -> pd.DataFrame:
    """
    Compute marginal value (EWA) for a set of players.
    
    EWA = Expected Wins Added (change in expected category matchup wins out of 60).
    
    Args:
        player_names: All players to evaluate (my roster + opponent rosters)
        my_roster_names: Players currently on my roster (subset of player_names)
        projections: Projections DataFrame (must include SGP column)
        my_totals: Current team totals
        opponent_totals: Opponent totals
        category_sigmas: Category standard deviations
    
    Returns:
        DataFrame with columns:
            Name, player_type, Position, on_my_roster,
            ewa_acquire (EWA if I acquire this player),
            ewa_lose (EWA if I lose this player, NaN if not on my roster),
            generic_value (SGP for trade fairness)
        
        Sorted by ewa_acquire descending.
    
    Implementation:
        1. Select SGP column based on config: 
           - If SGP_METRIC == "raw": use projections["SGP"]
           - If SGP_METRIC == "dynasty": use projections["dynasty_SGP"]
        2. Use selected SGP column as generic_value
        3. For each player, compute EWA for acquisition using compute_player_marginal_value
        4. For players in my_roster_names, also compute EWA for losing
        5. For players NOT in my_roster_names, ewa_lose = NaN
    
    Note:
        generic_value uses the SGP metric selected in config.json (SGP_METRIC).
        
        **Raw SGP (default):**
        - Simpler: no aging curve complexity
        - More predictable: managers understand SGP intuitively
        - Avoids bugs: age-scaling can produce unexpected results
        - Trade fairness is about this season, not 5-year projections
        
        **Dynasty SGP:**
        - Accounts for player age and future production decline
        - Useful for dynasty leagues that value long-term value
        - Requires age data to be computed (falls back to raw SGP if missing)
    """
```

---

## Trade Candidate Identification

```python
def identify_trade_targets(
    player_values: pd.DataFrame,
    my_roster_names: set[str],
    opponent_rosters: dict[int, set[str]],
    n_targets: int = 15,
) -> pd.DataFrame:
    """
    Identify players to TARGET (acquire) in trades.
    
    Targets are players on opponent rosters with POSITIVE EWA value,
    sorted by ewa_acquire (how much they help you).
    
    ⚠️ CRITICAL: Filter to players with ewa_acquire > 0.01 FIRST.
    Players with zero or negative value should never be trade targets.
    
    Sort by ewa_acquire descending (most helpful players first).
    The trade evaluation will handle fairness - targets just need to help you.
    
    Returns:
        DataFrame of targets with:
            Name, player_type, Position, ewa_acquire, generic_value (SGP),
            owner_id (which opponent owns them)
        
        Sorted by ewa_acquire descending.
    
    Print:
        "Trade targets: {N} players identified (ranked by EWA)"
        "  Best: {name} from Team {id} (EWA: +{ewa:.2f}, SGP: {sgp:.1f})"
    """


def identify_trade_pieces(
    player_values: pd.DataFrame,
    my_roster_names: set[str],
    n_pieces: int = 15,
) -> pd.DataFrame:
    """
    Identify players to OFFER (trade away) from my roster.
    
    Good trade pieces have:
        - Low ewa_lose (not critical to MY expected wins)
        - Reasonable SGP (attractive enough for opponents to accept)
    
    ⚠️ CRITICAL: The expendability formula sign convention:
    
    expendability = -SGP + ewa_lose * LOSE_COST_SCALE
    
    Since ewa_lose is NEGATIVE when losing hurts (ew_after < ew_before):
    - Low SGP player with small lose cost: -5 + (-0.5*2) = -6 (more expendable)
    - High SGP star with large lose cost: -20 + (-3.0*2) = -26 (less expendable)
    
    Higher expendability = more expendable (easier to trade away).
    
    ⚠️ DO NOT use `-(SGP + ewa_lose * scale)` - this inverts the logic
    because the double negative makes high-cost stars MORE expendable!
    
    Returns:
        DataFrame of tradeable players with:
            Name, player_type, Position, ewa_lose, generic_value (SGP),
            expendability
        
        Sorted by expendability descending.
    
    Print:
        "Trade pieces: {N} players identified"
        "  Most expendable: {name} (SGP={sgp:.1f}, lose_cost={ewa:.2f})"
    """
```

---

## Trade Evaluation

```python
def evaluate_trade(
    send_players: list[str],
    receive_players: list[str],
    player_values: pd.DataFrame,
    my_roster_names: set[str],
    projections: pd.DataFrame,
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
    opponent_rosters: dict[int, set[str]] | None = None,
) -> dict:
    """
    Evaluate a specific trade proposal.
    
    Args:
        send_players: Player names I would send
        receive_players: Player names I would receive
        player_values: Pre-computed player values
        my_roster_names: Current roster
        projections: Projections DataFrame
        my_totals: Current totals
        opponent_totals: Opponent totals
        category_sigmas: Category standard deviations
        opponent_rosters: Optional dict to identify trade partner team
    
    Returns:
        Dict with:
            'send_players': list
            'receive_players': list
            'ewa': float (Expected Wins Added, positive = good)
            'delta_generic': float (SGP change - positive means I get more SGP)
            'ew_before': float (expected wins before trade, out of 60)
            'ew_after': float (expected wins after trade, out of 60)
            'is_fair': bool (SGP differential <= 10% of total SGP)
            'is_good_for_me': bool (ewa >= MIN_MEANINGFUL_IMPROVEMENT)
            'recommendation': str ('ACCEPT', 'REJECT', 'NEUTRAL', 'UNFAIR', 'STEAL')
            'category_impact': dict (change in each category total)
            'send_generics': list of (name, SGP) tuples
            'receive_generics': list of (name, SGP) tuples
            'trade_partner_id': int or None (which opponent team)
    
    Implementation:
        1. Compute current ew_before via compute_win_probability diagnostics
        2. Compute new roster: (my_roster - send) | receive
        3. Validate roster composition (MIN/MAX hitters/pitchers)
        4. Compute new totals
        5. Compute ew_after
        6. ewa = ew_after - ew_before
        7. delta_generic = sum(receive SGP) - sum(send SGP)
        8. Fairness check (percentage-based):
           total_sgp = send_SGP + receive_SGP
           relative_diff = |delta_generic| / total_sgp
           is_fair = relative_diff <= FAIRNESS_THRESHOLD_PERCENT (10%)
        9. Determine recommendation:
           - ewa >= 0.1 and fair: 'ACCEPT'
           - ewa >= 0.1 and not fair (I'm getting more SGP): 'STEAL'
           - ewa <= -0.1 and fair: 'REJECT'
           - |ewa| < 0.1 and fair: 'NEUTRAL' (negligible impact)
           - not fair and bad for me: 'UNFAIR'
    
    Print (only for fair trades):
        "Trade with Team {id}:"
        "  Send: [{players with (SGP: X.X)}]"
        "  Receive: [{players with (SGP: X.X)}]"
        "  Net to me: {ewa:+.2f} expected wins"
        "  SGP change: {delta_generic:+.1f} ({'Fair' if is_fair else 'Unfair'})"
        "  Recommendation: {rec}"
    """


def generate_trade_candidates(
    my_roster_names: set[str],
    player_values: pd.DataFrame,
    opponent_rosters: dict[int, set[str]],
    projections: pd.DataFrame,
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
    max_send: int = 2,
    max_receive: int = 2,
    n_targets: int = 15,
    n_pieces: int = 15,
    n_candidates: int = 20,
) -> list[dict]:
    """
    Generate candidate trades to consider.
    
    Args:
        (standard inputs)
        max_send: Maximum players to send (1 to MAX_TRADE_SIZE)
        max_receive: Maximum players to receive
        n_targets: Trade targets to consider
        n_pieces: Trade pieces to consider
        n_candidates: Final candidates to return
    
    Returns:
        List of trade evaluation dicts, sorted by EWA descending.
        Only includes trades where is_fair=True and is_good_for_me=True.
        Empty list if no favorable fair trades found.
    
    Implementation:
        1. Identify trade targets (top n_targets by ewa_acquire)
        2. Identify trade pieces (top n_pieces by expendability)
        3. Generate combinations: 1-for-1, 2-for-1, 1-for-2, 2-for-2
        4. Evaluate each with evaluate_trade()
        5. Filter to fair + good trades
        6. Sort by EWA descending
        7. Return top n_candidates
        
        Use tqdm: "Evaluating trade combinations"
    
    Print:
        "Generated {K} candidate trades"
        "  Evaluated {T} combinations"
        "  Found {F} favorable fair trades"
        (if F == 0): "No favorable fair trades. Consider adjusting parameters."
    """
```

---

## Verification

```python
def verify_trade_impact(
    send_players: list[str],
    receive_players: list[str],
    my_roster_names: set[str],
    projections: pd.DataFrame,
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
) -> dict:
    """
    Verify a trade's impact by recomputing expected wins from scratch.
    
    This is the ground truth check — not using gradients, but full recomputation.
    
    Returns:
        Dict with:
            'V_before': Win probability before (for reference)
            'V_after': Win probability after (for reference)
            'ewa': Expected Wins Added (should match evaluate_trade closely)
            'old_totals': Category totals before
            'new_totals': Category totals after
            'category_changes': Dict of changes per category
            'ew_before': Expected wins before
            'ew_after': Expected wins after
            'matchup_flips': List of matchups that flip outcome
    
    Assertions:
        - All send_players in my_roster_names
        - All receive_players in projections
        - No receive_players already in my_roster_names
        - Post-trade roster meets composition bounds
    
    Print:
        "Trade verification:"
        "  Win probability: {before:.1%} → {after:.1%} ({delta:+.1%})"
        "  Expected wins: {before:.1f} → {after:.1f}"
        "  Matchups flipped: {N}"
    """
```

---

## Reporting

```python
def compute_roster_situation(
    my_roster_names: set[str],
    projections: pd.DataFrame,
    opponent_totals: dict[int, dict[str, float]],
) -> dict:
    """
    Compute full roster situation analysis. Main entry point.
    
    Implementation:
        1. my_totals = compute_team_totals(my_roster_names, projections)
        2. category_sigmas = estimate_projection_uncertainty(my_totals, opponent_totals)
        3. V, diagnostics = compute_win_probability(my_totals, opponent_totals, category_sigmas)
        4. Analyze matchup_probs to identify strengths/weaknesses
    
    Returns:
        Dict with:
            'my_totals': Category totals
            'category_sigmas': Standard deviations (computed internally)
            'win_probability': V
            'diagnostics': Full diagnostic dict from compute_win_probability
            'expected_wins': Out of 60
            'expected_roto_points': Estimated points
            'category_analysis': Per-category breakdown of matchup probabilities
            'strengths': Categories where P(win) > 0.7 for most opponents
            'weaknesses': Categories where P(win) < 0.3 for most opponents
    """


def print_trade_report(
    situation: dict,
    trade_candidates: list[dict],
    player_values: pd.DataFrame,
    top_n: int = 5,
) -> None:
    """
    Print a formatted trade recommendation report.
    
    Output format:
    
    ═══════════════════════════════════════════════════════════════════
    ROSTER SITUATION
    ═══════════════════════════════════════════════════════════════════
    Win probability: 18.3%
    Expected wins: 32.5/60 | Expected roto points: 48/70
    
    CATEGORY ANALYSIS:
    
    Category   My Value   Avg Opp   P(Win)   Status
    ────────   ────────   ───────   ──────   ──────
    R          823        795       68%      Strong
    HR         245        252       42%      Contested
    SB         95         120       23%      Weak
    ...
    
    STRENGTHS: W, SV (high win probability across opponents)
    WEAKNESSES: SB, ERA (low win probability)
    
    ═══════════════════════════════════════════════════════════════════
    TOP TRADE RECOMMENDATIONS
    ═══════════════════════════════════════════════════════════════════
    
    #1: +1.25 expected wins
        Send:    Gunnar Henderson (lose_cost: -0.72)
        Receive: Trea Turner [from opponent 3] (EWA: +1.97)
        ────────────────────────────────────────────────────────
        Net: +1.25 expected wins
        SGP change: -0.3 (Fair trade)
        Primary benefit: SB, R production
        Recommendation: ACCEPT
    
    #2: ...
    
    ═══════════════════════════════════════════════════════════════════
    
    Notes:
    - Player names displayed WITHOUT -H/-P suffix
    - If trade_candidates is empty, print suggestions for adjusting parameters
    """
```

---

## Display Name Handling

Functions that print output strip the -H/-P suffix using `strip_name_suffix()`:
- `print_trade_report`
- `evaluate_trade` (when printing)
- `identify_trade_targets` (when printing)
- `identify_trade_pieces` (when printing)
- `verify_trade_impact` (when printing)

All internal data structures preserve the suffix for uniqueness.

**Important:** `strip_name_suffix()` is defined ONLY in `data_loader.py`. Import it:
```python
from .data_loader import strip_name_suffix
```

Do NOT duplicate this function or define a local `_strip_suffix` helper.

---

## Edge Cases and Implementation Notes

1. **Zero-gradient categories:** If all matchup probabilities are ~0 or ~1, gradients are near zero. This is correct — no marginal value from changes in "solved" categories.

2. **Correlation handling:** Base implementation assumes category independence. Correlations can be added but increase complexity.

3. **Roster composition:** `evaluate_trade` validates MIN/MAX bounds for hitters (12-16) and pitchers (10-14) post-trade. Invalid trades fail with assertion.

4. **Ratio stat changes:** When computing how a player changes ratio stats, must recompute the full weighted average, not just add/subtract.

5. **Two-way players:** Ohtani-H and Ohtani-P are treated as independent players. Trading one doesn't affect the other.

6. **Self-trades:** `evaluate_trade` should assert no overlap between send and receive players.

7. **Empty results:** If no fair trades are found, `generate_trade_candidates` returns empty list. The report should suggest parameter adjustments.

8. **Numerical precision:** Use `scipy.stats.norm.cdf` and `scipy.stats.norm.pdf` for Φ and φ.

9. **Multi-player trades:** All received players must come from the SAME opponent (no 3-way trades). Filter combinations accordingly.

10. **Free agents:** If evaluating a player not on any roster, `ewa_lose` should be NaN or 0.

11. **SGP as generic_value:** Uses SGP for trade fairness. See `01e_dynasty_valuation.md` for details.

12. **Trade target ranking:** Ranked by `ewa_acquire` descending. Players most valuable to YOUR team appear first.

13. **Meaningful improvement threshold:** ACCEPT requires EWA >= 0.1 (0.1 expected wins). Trades with |EWA| < 0.1 are marked NEUTRAL.

14. **Expendability formula:** `expendability = -SGP + ewa_lose * LOSE_COST_SCALE` where `LOSE_COST_SCALE = 2`.
    ⚠️ CRITICAL: The formula `-(SGP + ewa_lose * scale)` is WRONG (double negative flips logic).

15. **Mixed player types:** Both `identify_trade_targets` and `identify_trade_pieces` ensure at least 40% of each player type for roster composition fixes.

16. **Percentage-based fairness:** `relative_diff = abs(send_dynasty_SGP - receive_dynasty_SGP) / (send_dynasty_SGP + receive_dynasty_SGP)`. Fair if <= 10%.

17. **Trade analysis uses optimized roster:** Notebook runs trade analysis on post-free-agency optimized roster, not current roster.

18. **Simplified totals computation:** `_compute_totals_with_player_change()` modifies roster set and calls `compute_team_totals()`. Do NOT reimplement weighted average logic.

19. **No arbitrary limits:** Evaluate ALL targets from each owner, not just first few.

20. **strip_name_suffix:** Import from `data_loader`, do NOT duplicate.

21. **Dynasty valuation:** 25% discount rate prioritizes current-year production (Year 0: 100%, Year 1: 80%, Year 2: 64%, Year 3: 51%).

22. **Age data missing:** Players without age get `dynasty_SGP = SGP` as fallback.

---

## Validation Checklist

- [ ] `compute_win_probability` matches paper formulation
- [ ] Normalized gaps flip sign for ERA/WHIP
- [ ] Player values use ewa_acquire for targets, ewa_lose for pieces
- [ ] Generic values use SGP
- [ ] Trade evaluation computes exact expected wins (not just gradient approximation)
- [ ] Verification matches evaluation closely
- [ ] Roster composition validated post-trade (12-16 hitters, 10-14 pitchers)
- [ ] Display functions use `strip_name_suffix()` imported from data_loader
- [ ] Empty results handled gracefully
- [ ] All print statements use descriptive messages
- [ ] tqdm for trade candidate evaluation loop
- [ ] Trade targets ranked by ewa_acquire descending
- [ ] ACCEPT requires >= 0.1 EWA; NEUTRAL for negligible impact
- [ ] Trade output shows SGP (generic value) for each player
- [ ] Trade output identifies trade partner team
- [ ] `_compute_totals_with_player_change` uses `compute_team_totals()` (no reimplementation)
- [ ] No arbitrary trade combination limits (evaluate all targets per owner)
- [ ] SGP used for fairness evaluation
