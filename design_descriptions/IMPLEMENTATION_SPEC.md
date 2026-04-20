# Implementation Specification

**Implements:** `design_descriptions/MATHEMATICAL_FRAMEWORK.md`
**Coding standards:** `AGENTS.md` (read it — every rule is enforced)

**Scope:** This is a roster optimizer for a 7-team rotisserie league. Given a pre-built "silver table" (players DataFrame with projections + roster data), it determines the best 28-man roster by computing per-player value scores and recommending transactions. The lineup assignment (who starts where) is an inner subproblem; the outer problem is roster construction: which 28 players maximize expected standing points across all plausible scenarios?

All upstream work — FanGraphs CSV loading, Fantrax API calls, name normalization, position merging — is handled separately and is NOT part of this spec. Silver table in, gold table + roster recommendations out.

---

## 1. Silver Table Input Contract

The input to the math pipeline is a pandas DataFrame called `players` with one row per player. The upstream pipeline is responsible for producing this table; the math modules assume it is correct and complete.

### Required columns

| Column | Type | Description |
|--------|------|-------------|
| Name | str | Player name with -H (hitter) or -P (pitcher) suffix. Globally unique. |
| Team | str | MLB team abbreviation (e.g., "NYY", "LAD"). "FA" for free agents without a team. |
| Position | str | Comma-separated eligible positions (e.g., "SS,2B", "SP", "OF"). |
| player_type | str | "hitter" or "pitcher". |
| PA | float | Projected plate appearances. 0 for pitchers. |
| IP | float | Projected innings pitched. 0 for hitters. |
| R | float | Projected runs. 0 for pitchers. |
| HR | float | Projected home runs. 0 for pitchers. |
| RBI | float | Projected runs batted in. 0 for pitchers. |
| SB | float | Projected stolen bases. 0 for pitchers. |
| OPS | float | Projected OPS (use directly from FanGraphs, do NOT recompute). 0 for pitchers. |
| W | float | Projected wins. 0 for hitters. |
| SV | float | Projected saves. 0 for hitters. |
| K | float | Projected strikeouts (renamed from FanGraphs "SO"). 0 for hitters. |
| ERA | float | Projected ERA. 0 for hitters. |
| WHIP | float | Projected WHIP. 0 for hitters. |
| WAR | float | Projected WAR (for PV calculation). |
| owner | str or None | Fantasy team name (e.g., "The Big Dumpers") or None/NaN for free agents. |
| roster_status | str or None | "active", "reserve", "IR", "minors", "taxi", or None for free agents. |

### Optional columns (for PV calculation)

| Column | Type | Description |
|--------|------|-------------|
| fantrax_score | float or None | Fantrax platform score (for rostered players, may need imputation). |
| pct_rostered | float or None | Percent rostered across Fantrax leagues. |
| age | int or None | Player age. |

### Invariants the upstream pipeline must guarantee

1. No duplicate Names within the same player_type.
2. Every rostered player (owner is not None) has roster_status set.
3. Every team has enough players to fill their starting lineup slots.
4. All stat columns are numeric (no NaN in projection stats).

---

## 2. Gold Table Output Contract

After the math pipeline runs, the players DataFrame gains these columns:

| Column | Type | Source | Description |
|--------|------|--------|-------------|
| FV | float | add_fantasy_value | Fantasy Value: context-free z-score sum across 5 categories per type. |
| optimal_slot | str or None | assign_optimal_slots | Starting lineup slot (e.g., "SS", "SP") or None (bench/FA). |
| PV | float | add_perceived_value | Perceived Value: how opponents subjectively value this player. |
| MEW | float | add_mew | Marginal Expected Wins: gradient-based, context-aware per-player score. |
| BV | float | add_bench_value | Bench Value: option premium for bench insurance (my bench only; 0 for others). |

Plus the pipeline produces swap and trade recommendations as separate outputs (not DataFrame columns).

---

## 3. Configuration

All league configuration is loaded from repo-root `config.json`. A thin `config.py` module reads this file and exposes constants. No v2-specific config.json is needed.

### `config.py`

```python
import json
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.json"

def load_config() -> dict:
    """Load configuration from repo-root config.json."""
    assert _CONFIG_PATH.exists(), f"Config file not found: {_CONFIG_PATH}"
    with open(_CONFIG_PATH) as f:
        return json.load(f)

_CONFIG = load_config()
LEAGUE = _CONFIG["league"]

# Categories
HITTING_CATEGORIES: list[str] = LEAGUE["hitting_categories"]      # ['R', 'HR', 'RBI', 'SB', 'OPS']
PITCHING_CATEGORIES: list[str] = LEAGUE["pitching_categories"]    # ['W', 'SV', 'K', 'ERA', 'WHIP']
ALL_CATEGORIES: list[str] = HITTING_CATEGORIES + PITCHING_CATEGORIES
NEGATIVE_CATEGORIES: set[str] = set(LEAGUE["negative_categories"])  # {'ERA', 'WHIP'}

# League structure
NUM_OPPONENTS: int = len(LEAGUE["fantrax_team_ids"]) - 1  # 6
ROSTER_SIZE: int = LEAGUE["roster_size"]                   # 28
HITTING_SLOTS: dict[str, int] = LEAGUE["hitting_slots"]    # {"C": 1, "1B": 1, ..., "UTIL": 1}
PITCHING_SLOTS: dict[str, int] = LEAGUE["pitching_slots"]  # {"SP": 5, "RP": 2}
SLOT_ELIGIBILITY: dict[str, set[str]] = {
    k: set(v) for k, v in LEAGUE["slot_eligibility"].items()
}

# Team identity
MY_TEAM_NAME: str = LEAGUE["my_team_name"]                 # "The Big Dumpers"

# Numeric safety
MIN_STAT_STANDARD_DEVIATION: float = LEAGUE["min_stat_standard_deviation"]  # 0.001

# Derived
N_STARTER_SLOTS: int = sum(HITTING_SLOTS.values()) + sum(PITCHING_SLOTS.values())  # 18

# Validation
assert len(ALL_CATEGORIES) == 10, "Must have exactly 10 scoring categories"
assert len(LEAGUE["fantrax_team_ids"]) == 7, "Must have exactly 7 teams"
```

### `players.py` — Name and position utilities

Tiny module, needed by several math modules. Port from v1 unchanged.

```python
def strip_name_suffix(name: str) -> str:
    """Strip -H or -P suffix for display. Defined ONLY here, imported everywhere else."""

def get_eligible_slots(position_str: str) -> set[str]:
    """Compute which lineup slots a player is eligible for.
    Uses SLOT_ELIGIBILITY from config."""
```

---

## 4. Module Map

```
optimizer/
├── __init__.py
├── config.py              # Configuration from repo-root config.json
├── players.py             # Name/position utilities
├── win_model.py           # EW computation, gradient, σ estimation
├── lineup_solver.py       # Lineup MILP, team totals
├── player_scoring.py      # FV, PV, MEW (per-player scoring)
├── league_state.py        # Fixed-point iteration, league-level state
├── swap_evaluator.py      # MSV, screening, exact evaluation, BV, greedy optimizer
└── trade_finder.py        # Trade evaluation (same math + PV constraint + opponent re-solve)
```

### Dependency DAG

```
config.py, players.py                   ← Layer 0 (no internal deps)
    ↓
win_model.py, lineup_solver.py          ← Layer 1 (depend on config/players)
    ↓
player_scoring.py                       ← Layer 2a (depends on Layers 0–1)
    ↓
league_state.py                         ← Layer 2b (depends on 2a: calls add_mew
                                                     during fixed-point iteration)
    ↓
swap_evaluator.py, trade_finder.py      ← Layer 3 (depend on Layers 0–2)
```

**Note on league_state → player_scoring dependency:** `compute_league_state` calls `add_mew` internally during its MEW-lineup fixed-point iteration (Section 7). This is a one-way dependency: `player_scoring` does NOT import from `league_state`. The two modules are at adjacent layers, not the same layer, because `league_state` cannot be implemented without `player_scoring`.

---

## 5. `win_model.py` — Expected Wins and Gradient

Port from `v1/optimizer/win_model.py`. This module is mathematically correct as-is.

### Constants

```python
# Projection noise floors: team-level σ estimates.
# Counting stats: σ = |league_mean| × CV
_TEAM_PROJECTION_CV: dict[str, float] = {
    "R": 0.06, "HR": 0.09, "RBI": 0.06, "SB": 0.15,
    "W": 0.12, "SV": 0.20, "K": 0.06,
}
# Ratio stats: absolute σ
_TEAM_PROJECTION_SIGMA: dict[str, float] = {
    "OPS": 0.012, "ERA": 0.30, "WHIP": 0.050,
}
```

### Functions

```python
def compute_win_probability(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
) -> tuple[float, dict]:
    """Rosenof (2025) model applied to rotisserie standings.

    EW = Σ_c Σ_o Φ(z_{c,o}): sum of pairwise beat probabilities.
    Expected standing points = 10 + EW.

    Returns:
        ew: Expected wins (float, range 0–60 for 10 categories × 6 opponents).
        diagnostics: Dict including:
            'expected_wins' (same as ew, for convenience),
            'beat_probs' (per-category per-opponent beat probabilities),
            'normalized_gaps' (z-scores).
    """

def compute_ew_gradient(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
) -> dict[str, float]:
    """∂EW/∂(my_c) for each category c.

    For c ∈ C⁺ (R, HR, RBI, SB, OPS, W, SV, K):  g_c = +Σ_o φ(z_{c,o}) / (σ_c √2) > 0
    For c ∈ C⁻ (ERA, WHIP):                        g_c = −Σ_o φ(z_{c,o}) / (σ_c √2) < 0
    """

def estimate_projection_uncertainty(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
) -> dict[str, float]:
    """Estimate σ_c: how much actual outcomes could deviate from projections.

    Counting stats: σ_c = |league_mean_c| × CV_c (fixed CV per category).
    Ratio stats: σ_c = fixed absolute value per category.

    CRITICAL: This is projection uncertainty, NOT observed cross-team
    variance. See W7 for why the distinction matters.
    """

def compute_standings(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
) -> pd.DataFrame:
    """Projected roto standings: rank and standing points per category."""
```

---

## 6. `lineup_solver.py` — Lineup MILP and Team Totals

Port from `v1/optimizer/lineup_solver.py`, with one critical change: `solve_lineup` is parameterized by objective column. My team's lineup maximizes MEW (context-aware); opponent lineups maximize FV (context-free). See MATHEMATICAL_FRAMEWORK §5, §9h.

```python
def solve_lineup(
    roster_names: Iterable[str],
    players: pd.DataFrame,
    objective_column: str = "FV",
) -> dict[str, str]:
    """Solve lineup assignment via MILP, maximizing Σ objective_column for starters.

    The objective_column parameter determines what the lineup optimizes:
    - "FV" for opponents (context-free z-score sum, no gradient needed)
    - "MEW" for my team (gradient-weighted, context-aware)

    My team uses MEW because the gradient g_c weights each category by
    its current marginal value — the lineup prioritizes players who
    contribute most to categories where improvement matters. Opponents
    use FV because we model them as optimizing context-free quality,
    not optimizing against specific category needs.

    Do NOT mix these: using FV for my lineup forfeits gradient information;
    using MEW for opponents requires a gradient we don't compute for them.
    See MATHEMATICAL_FRAMEWORK §5 and §9h.

    With 28 binary variables and ~18 slot constraints, solves in <1ms
    with HiGHS. This is a weighted bipartite matching; MILP is used for
    implementation convenience (handles multi-count slots like 5×OF).

    Returns dict mapping starter name → assigned slot. Bench players omitted.
    """

def compute_team_totals(
    roster_names: Iterable[str],
    players: pd.DataFrame,
    objective_column: str = "FV",
) -> dict[str, float]:
    """Solve lineup MILP then aggregate starters into team totals.

    This is the workhorse function: MILP + aggregation in one call.
    Passes objective_column through to solve_lineup.

    CRITICAL: ratio stats are PA/IP-weighted averages, NOT sums.
        OPS = Σ(PA × OPS) / Σ(PA)
        ERA = Σ(IP × ERA) / Σ(IP)
        WHIP = Σ(IP × WHIP) / Σ(IP)
    """

def compute_totals_for_starters(
    starters: set[str],
    players: pd.DataFrame,
) -> dict[str, float]:
    """Team totals for a known set of starters (no MILP). Fast path.

    Returns dict with keys for all 10 categories PLUS 'PA' and 'IP':
        {'R': 823, 'HR': 245, ..., 'ERA': 3.85, ..., 'PA': 5200, 'IP': 1100}

    PA and IP are simple sums of starters. These are needed by the MEW
    formula (ratio stat baselines and total-weight denominators).
    Counting stats are sums; ratio stats are weighted averages.
    """

def assign_optimal_slots(
    players: pd.DataFrame,
    my_lineup: dict[str, str],
    opponent_lineups: dict[int, dict[str, str]],
    opponent_teams: list[str],
) -> pd.DataFrame:
    """Set optimal_slot column from pre-computed lineup assignments.

    Takes lineups produced by compute_league_state (which runs the
    MEW-lineup fixed-point iteration for my team, FV for opponents).

    Args:
        players: The players DataFrame.
        my_lineup: {player_name: slot} for my team's MEW-optimal lineup.
        opponent_lineups: {opponent_id: {player_name: slot}} for each opponent.
        opponent_teams: Sorted list of opponent team names (maps to opponent IDs).

    Players not in any lineup get optimal_slot = None.
    This is a DataFrame-enrichment function: players in, players out.
    """
```

---

## 7. `league_state.py` — League State with MEW-Lineup Fixed-Point Iteration

This module computes the full league state, including the MEW-lineup fixed-point iteration for my team (MATHEMATICAL_FRAMEWORK §5). This is the central state-computation step from §8 of the math framework.

```python
_MAX_LINEUP_ITERATIONS: int = 5

def compute_league_state(players: pd.DataFrame) -> dict:
    """Compute converged league state via MEW-lineup fixed-point iteration.

    The state computation step from MATHEMATICAL_FRAMEWORK §8. Produces
    everything needed for player scoring, screening, and evaluation.

    Algorithm:
        1. Opponent lineups: solve each opponent's lineup with FV
           (one MILP each). Compute opponent_totals.

        2. My team: MEW-lineup fixed-point iteration (MATH_FRAMEWORK §5)
           a. Solve initial lineup with FV → my_starters₀, my_totals₀
           b. estimate_projection_uncertainty → category_sigmas
           c. compute_ew_gradient → gradient
           d. Compute MEW for all players via add_mew (on working copy)
           e. Re-solve my lineup with objective_column="MEW" → my_starters₁
           f. If my_starters₁ ≠ my_starters₀: update totals, go to (b)
           g. Converged when starter set stabilizes

        3. Compute current_ew from converged state

    Convergence is fast: improving in category c increases z_{c,o},
    decreasing φ(z_{c,o}), decreasing |g_c|. The concavity of Φ acts
    as a damper. A marginal lineup swap changes the aggregate gradient
    by 2–8% (MATH_FRAMEWORK §5 quantitative bound). Worst case: 2
    possible lineups — evaluate both, pick the one with higher EW.
    Total cost: at most 3 MILP solves (< 3ms).

    Requires: players has FV column (from add_fantasy_value).

    Opponent ID convention (from MATHEMATICAL_FRAMEWORK §1):
        Opponent IDs are **1-indexed**: O = {1, ..., 6}.
        opponent_teams is sorted alphabetically; the i-th name (1-indexed)
        maps to opponent ID i. So opponent_teams[0] is opponent 1,
        opponent_teams[5] is opponent 6. Do NOT use 0-indexed opponent IDs.

    Returns:
        {
            'my_totals': dict[str, float],        # 10 categories + 'PA' + 'IP'
                                                   # PA/IP are starters' sums, needed by
                                                   # add_mew as total_PA / total_IP denominators
            'opponent_totals': dict[int, dict[str, float]],   # keys are 1-indexed opponent IDs
            'category_sigmas': dict[str, float],
            'gradient': dict[str, float],         # ∂EW/∂(my_c), converged
            'my_starters': set[str],
            'my_lineup': dict[str, str],          # name → slot assignment
            'opponent_lineups': dict[int, dict[str, str]],    # keys are 1-indexed opponent IDs
            'opponent_teams': list[str],           # alphabetically sorted; index i → opponent i+1
            'current_ew': float,
        }

    Implementation sketch:
        work = players.copy()

        # 1. Opponent lineups (FV)
        for each opponent: solve_lineup(opp_roster, work, "FV") → opp_lineup
        opponent_totals = {i: compute_totals_for_starters(...)}

        # 2. My team: first solve with FV
        my_lineup = solve_lineup(my_roster, work, "FV")
        my_totals = compute_totals_for_starters(set(my_lineup.keys()), work)
        sigmas = estimate_projection_uncertainty(my_totals, opponent_totals)
        gradient = compute_ew_gradient(my_totals, opponent_totals, sigmas)
        work = add_mew(work, my_totals, gradient)

        # 3. Iterate with MEW
        for _ in range(_MAX_LINEUP_ITERATIONS):
            new_lineup = solve_lineup(my_roster, work, "MEW")
            if set(new_lineup.keys()) == set(my_lineup.keys()):
                break  # converged
            my_lineup = new_lineup
            my_totals = compute_totals_for_starters(set(my_lineup.keys()), work)
            sigmas = estimate_projection_uncertainty(my_totals, opponent_totals)
            gradient = compute_ew_gradient(my_totals, opponent_totals, sigmas)
            work = add_mew(work, my_totals, gradient)

        # 4. Compute EW from converged state
        current_ew, _ = compute_win_probability(my_totals, opponent_totals, sigmas)
        return { ... 'current_ew': current_ew ... }
    """
```

---

## 8. `player_scoring.py` — Per-Player Scoring

Merges v1's `player_scoring.py` (FV, PV) and v1's `player_valuation.add_mew` (MEW).

All functions are DataFrame enrichment: `players` in → `players` with new column(s) out.

### 8a. Fantasy Value (FV)

```python
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
```

### 8b. Perceived Value (PV)

```python
def add_perceived_value(players: pd.DataFrame) -> pd.DataFrame:
    """Add 'PV' column: how opponents likely value a player in trade talks.

    PV = max(FV, 0) + max(WAR − 3, 0) × 3

    FV is the base: smart opponents evaluate players by projected fantasy
    production. The fame premium (3 PV per WAR above 3.0) corrects FV's
    systematic undervaluation of elite SPs (whose z-score totals are
    suppressed by zero saves) and captures the general-quality / name-
    recognition premium that real trade markets exhibit.

    The per-player max constraint in trade_finder.py separately prevents
    aggregating mid-tier players to acquire a superstar.

    Requires: FV, WAR.
    Adds: PV.
    """
```

### 8c. Marginal Expected Wins (MEW)

```python
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
    NOT recomputed here. This ensures MEW uses the converged gradient
    from the MEW-lineup fixed-point iteration.

    SIGN VERIFICATION (verify this in implementation):
        g_ERA < 0. Good pitcher: (ERA − my_ERA) < 0. Product: positive. ✓

    Args:
        players: DataFrame with stat columns.
        my_totals: Converged team totals dict. Must contain all 10 category
            keys plus 'PA' and 'IP'. The ratio stat values (OPS, ERA, WHIP)
            are weighted averages. PA and IP are sums of starters' playing
            time — used as total_PA and total_IP in the denominator of the
            ratio stat terms, and my_OPS/my_ERA/my_WHIP as the baseline.
        gradient: Pre-computed ∂EW/∂(my_c) from compute_ew_gradient.

    Requires: PA, IP, R, HR, RBI, SB, OPS, W, SV, K, ERA, WHIP.
    Adds: MEW.

    Implementation:
        1. Extract my_OPS, my_ERA, my_WHIP, total_PA, total_IP from my_totals
        2. Vectorized: apply the single formula to all rows at once
    """
```

---

## 9. `swap_evaluator.py` — The Core Optimization Module (NEW)

Implements Mathematical Framework Sections 4, 5, 6, 8.

### Key design principle

A "swap" is the universal operation: drop N players from my roster, add N players
(from the FA pool, from an opponent's roster, or a mix). FA pickups, trades,
and multi-move batches are all swaps — same math, different constraints.

- **FA swap**: add comes from FA pool. No constraint beyond roster fit.
- **Trade**: add comes from an opponent's roster. PV constraint must be satisfied.
- **Batch move**: trade + FA pickups evaluated together as a single N-for-N swap.

By default, the optimizer only considers FA swaps. Trades are evaluated on
request, subject to a PV constraint (see Section 10).

### Dependencies

```python
from .config import (
    ALL_CATEGORIES, HITTING_SLOTS, PITCHING_SLOTS, NEGATIVE_CATEGORIES,
    MY_TEAM_NAME, SLOT_ELIGIBILITY, N_STARTER_SLOTS, ROSTER_SIZE,
)
from .lineup_solver import (
    solve_lineup, compute_team_totals, compute_totals_for_starters,
)
from .player_scoring import add_mew
from .players import get_eligible_slots
from .win_model import (
    compute_ew_gradient, compute_win_probability,
    estimate_projection_uncertainty,
)
```

### Constants

```python
DEFAULT_SCREEN_TOP_K: int = 50
DEFAULT_VALUE_THRESHOLD: float = 0.1  # minimum Value to execute a swap (EW units)

# Expected fraction of the remaining season a starter at this slot is absent
# (injury, rest, demotion). Dimensionless proportion, not a rate.
# In roto, EW and BV are both season-level quantities — no unit conversion needed.
POSITION_ABSENCE_RATES: dict[str, float] = {
    "C": 0.25, "1B": 0.18, "2B": 0.20, "SS": 0.22, "3B": 0.20,
    "OF": 0.20, "UTIL": 0.15, "SP": 0.35, "RP": 0.25,
}
```

### 9a. `compute_exact_msv` — Ground-truth swap value

```python
def compute_exact_msv(
    drop_names: set[str],
    add_names: set[str],
    my_roster_names: set[str],
    players: pd.DataFrame,
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
    current_ew: float,
) -> dict:
    """Compute exact Marginal Swap Value by re-solving the lineup.

    MSV = EW(new_roster) − EW(current_roster)

    This is the universal evaluator for any transaction: 1-for-1 FA swap,
    N-for-N trade, or a batch of moves. Same math regardless.

    Steps:
        1. assert len(drop_names) == len(add_names) (roster size preserved)
        2. assert drop_names ⊂ my_roster_names
        3. new_roster = (my_roster_names − drop_names) ∪ add_names
        4. assert len(new_roster) == ROSTER_SIZE
        5. new_lineup = solve_lineup(new_roster, players, objective_column="MEW")
        6. new_totals = compute_totals_for_starters(set(new_lineup.keys()), players)
        7. new_ew, _ = compute_win_probability(new_totals, opponent_totals, category_sigmas)
        8. msv = new_ew − current_ew

    Steps 5-6 use solve_lineup + compute_totals_for_starters (rather than
    the combined compute_team_totals) so the lineup assignment is available
    for downstream ΔBV computation in evaluate_top_k and evaluate_trade.

    Uses objective_column="MEW" for the lineup solve because this evaluates
    my team's roster (MATHEMATICAL_FRAMEWORK §5, §9h). The MEW values on the
    players DataFrame are from the current state's converged gradient. They are
    slightly stale relative to the post-swap gradient, but the stability
    analysis (§5) shows this introduces ≤2–8% gradient error — the lineup
    is almost certainly the same.

    The MILP's position-slot constraints are the only structural guard:
    if the new roster can't fill all slots, the MILP will fail.

    Returns: {
        'msv': float,          # EW(new) − EW(current)
        'new_ew': float,       # EW of the post-swap roster
        'new_totals': dict,    # team totals including PA/IP (from step 6)
        'new_lineup': dict,    # name → slot assignment (from step 5)
    }
    """
```

For 1-for-1 FA swaps, call as `compute_exact_msv({drop}, {add}, ...)`.
For a trade sending 2 + picking up 1 FA: `compute_exact_msv({a, b, c}, {x, y, z}, ...)`.

### 9b. `screen_swaps` — Lineup-aware screening for all FA swaps

```python
def screen_swaps(
    players: pd.DataFrame,
    my_roster_names: set[str],
    my_lineup: dict[str, str],
    top_k: int = DEFAULT_SCREEN_TOP_K,
) -> pd.DataFrame:
    """Screen all possible 1-for-1 FA swaps, ranked by approximate Value.

    Uses **lineup-aware** MSV_approx (MATHEMATICAL_FRAMEWORK §4): the EW
    change from a swap depends on who enters and leaves the *starting lineup*,
    not who enters and leaves the *roster*.

    For each (FA, droppable_roster_player) pair, compute:
        MSV_approx = lineup-aware EW change estimate (see cases below)
        ΔBV_approx = change in total bench value (gradient-based)

    Approximate Value = MSV_approx + ΔBV_approx.

    Lineup-aware MSV_approx cases:
        - Bench drop, FA displaces starter S at best eligible slot:
            MSV_approx = MEW(FA) − MEW(S)
        - Bench drop, FA doesn't start:
            MSV_approx = 0
        - Starter drop, FA eligible for vacated slot (direct replacement):
            MSV_approx = MEW(FA) − MEW(dropped_starter)
        - Starter drop, FA takes different slot (cascade):
            Approximate as (MEW(bench_fill) − MEW(drop)) + FA displacement.
            The exact evaluation handles cascades precisely.

    ΔBV_approx uses the gradient-based BV formula (Section 9d) with current
    MEW values, without re-solving the lineup. When the swap involves only
    bench players, ΔBV_approx is accurate. When lineup reshuffles occur
    (dropping a starter), ΔBV_approx is rougher — exact evaluation handles it.

    Algorithm:
        1. Pre-filter: identify "protected" roster players (sole eligible
           player for a required slot). These cannot be dropped.
        2. Precompute per-slot weakest starter (for FA displacement check)
           and per-slot best bench player (for cascade approximation).
        3. For each FA f, find the droppable roster player r that maximizes
           MSV_approx + ΔBV_approx.
        4. Rank all (f, r) pairs by approximate Value descending. Keep top K.

    The exact evaluation phase (MILP re-solve + recomputed BV) handles
    lineup cascades, position interactions, gradient changes, and EW
    surface convexity that screening cannot see.

    Args:
        my_lineup: state['my_lineup'] from compute_league_state — a dict
            mapping starter name → slot. Used to identify bench players,
            compute lineup-aware MSV, and compute ΔBV_approx. Do NOT pass
            state['my_starters'] (a set).

    Requires columns: Name, owner, optimal_slot, MEW, Position, player_type.
    Requires: add_mew() and add_bench_value() have already been called.

    Returns:
        DataFrame with columns: fa_name, drop_name, msv_approx, delta_bv_approx, value_approx.
        Sorted by value_approx descending. Length ≤ top_k.
    """
```

### 9c. `evaluate_top_k` — Exact Value for top candidates

```python
def evaluate_top_k(
    candidates: pd.DataFrame,
    my_roster_names: set[str],
    players: pd.DataFrame,
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
    current_ew: float,
    current_total_bv: float,
    include_bv: bool = True,
) -> pd.DataFrame:
    """Compute exact Value = ΔEW + ΔBV for each screened swap candidate.

    For each candidate (fa_name, drop_name):
        1. result = compute_exact_msv({drop_name}, {fa_name}, ...)
           → gives msv, new_ew, new_totals, new_lineup
        2. If include_bv:
           a. new_gradient = compute_ew_gradient(new_totals, opponent_totals, category_sigmas)
           b. Compute new_MEW for all players from new_gradient (via add_mew on copy)
           c. new_total_bv = Σ BV(b) for my bench via add_bench_value(copy, new_lineup)
           d. delta_bv = new_total_bv − current_total_bv
        3. value = msv + delta_bv

    1 MILP solve per candidate (the lineup re-solve inside compute_exact_msv).
    Steps 2a–2c are analytical — no MILP needed for BV.
    Use tqdm (calls MILP once per candidate).

    Returns: candidates with added columns msv_exact, delta_bv, value, new_ew.
    Sorted by value descending.
    """
```

### 9d. `add_bench_value` — Gradient-based bench insurance

```python
def add_bench_value(
    players: pd.DataFrame,
    my_lineup: dict[str, str],
) -> pd.DataFrame:
    """Add 'BV' column: analytical bench insurance value.

    BV is computed via the gradient-based formula from MATHEMATICAL_FRAMEWORK §6.
    No MILP solves required — it is a pure function of MEW scores and absence
    probabilities. The inherent roughness of absence estimation (estimated rates,
    independent single-absence assumption) does not justify more expensive computation.

    Formula:
        BV(b) = Σ_{k : best_bench(k) = b} P_absent(k) × max(0, MEW(b) − MEW(best_FA(k)))

    where:
        k indexes starter slots (C, 1B, 2B, SS, 3B, OF×5, UTIL, SP×5, RP×2)
        best_bench(k) = highest-MEW bench player eligible for slot k
        best_FA(k) = highest-MEW free agent eligible for slot k
        P_absent(k) = POSITION_ABSENCE_RATES[slot_type(k)]

    A bench player who is the best option for multiple slots accumulates
    BV from all of them (e.g., a 2B/SS bench player provides insurance
    for both the 2B and SS slots).

    Args:
        my_lineup: state['my_lineup'] from compute_league_state — a dict
            mapping starter name → assigned slot. Needed both to identify
            bench players (not in my_lineup.keys()) and to iterate over
            starter slots for BV computation. Do NOT pass state['my_starters']
            (which is a set without slot assignments).

    Algorithm:
        1. Identify my bench: (owner == MY_TEAM_NAME) & (Name not in my_lineup)
        2. Identify FA pool: owner is None/NaN
        3. For each starter slot k:
           a. best_bench(k) = highest-MEW bench player eligible for slot k
           b. best_FA(k) = highest-MEW free agent eligible for slot k
              (if no FA eligible: treat MEW(best_FA(k)) = 0)
        4. For each bench player b:
           BV(b) = Σ over slots k where b = best_bench(k):
                   P_absent(k) × max(0, MEW(b) − MEW(best_FA(k)))
        5. BV = 0 for starters (their value is captured by MEW in the lineup)
        6. BV = 0 for players not on my roster

    Total computation: O(slots × (bench + FAs)). Sub-millisecond.

    Key properties (from MATHEMATICAL_FRAMEWORK §6):
    - Position-aware: bench SS is worth more when starting SS has high absence rate
    - FA-pool-relative: great FAs at a position reduce bench value there
    - Identifies droppable players: BV ≈ 0 means no meaningful insurance

    Simplifying assumptions:
    - At most one starter absent at a time (first-order approximation)
    - Each bench player evaluated independently (no bench-bench interactions)
    - Lineup cascades ignored (bench player directly fills absent starter's slot)
    - Multi-absence scenarios largely cancel in ΔBV for 1-for-1 move evaluation

    Requires: Name, owner, Position, MEW, roster_status.
    Adds: BV.
    """
```

### 9e. `compute_ew_ceiling` — Diagnostic for gap-to-optimal

```python
def compute_ew_ceiling(
    players: pd.DataFrame,
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
) -> dict:
    """Compute best achievable EW from full candidate pool (diagnostic).

    Solves a larger MILP: pick ROSTER_SIZE players from all available
    (my roster + FA pool) and assign 18 to starter slots, maximizing
    Σ MEW(starters). MEW is the linear proxy for EW (EW itself is nonlinear
    due to Φ). Position slot constraints are enforced. No PV constraint.

    After solving, compute exact EW from the ceiling roster via
    compute_totals_for_starters + compute_win_probability. The gap between
    exact current EW and exact ceiling EW is the diagnostic.

    This is NOT the optimizer — it is a diagnostic (MATHEMATICAL_FRAMEWORK §5).
    If the gap between current EW and the ceiling is small, focus on bench
    depth and marginal trades. If the gap is large, major structural upgrades
    are available and aggressive moves are warranted.

    Returns: {
        'ceiling_ew': float,       # exact EW of the ceiling roster
        'ceiling_roster': set[str],
        'ceiling_lineup': dict,    # name → slot for the 18 starters
        'gap': float,              # ceiling_ew − current_ew
    }
    """
```

### 9f. `run_greedy_optimization` — The optimizer

```python
def run_greedy_optimization(
    players: pd.DataFrame,
    max_moves: int = 10,
    value_threshold: float = DEFAULT_VALUE_THRESHOLD,
    top_k: int = DEFAULT_SCREEN_TOP_K,
    include_bv: bool = True,
) -> dict:
    """Find the best reachable roster via greedy FA swaps.

    The optimizer's job: given the current 28-man roster and the FA pool,
    find the 28-man roster that maximizes Value = EW_healthy + BV.
    It does this by iteratively finding and executing the single best
    1-for-1 FA swap until no beneficial swap remains.

    Each iteration:
        1. Compute league state via compute_league_state (includes MEW-lineup
           fixed-point iteration). On the first iteration, this computes
           everything from scratch. On subsequent iterations, opponent totals
           are fixed; only my team's state is re-solved.
        2. Enrich players with MEW (from converged gradient) and BV
        3. Screen all FA swaps via screen_swaps (ranked by MSV_approx + ΔBV_approx)
        4. Exact-evaluate top K candidates (ΔEW + ΔBV, with recomputed gradient/MEW)
        5. If best Value > threshold: execute swap, update roster, loop to 1
        6. Else: stop

    MEW must be recomputed from scratch each iteration (W3). After each swap,
    the team totals change, the gradient changes, and MEW rankings shift.
    Reusing old MEW values produces incorrect rankings.

    Convergence: the greedy approach converges because EW exhibits
    approximate diminishing returns — improving in category c increases
    z_{c,o}, which decreases φ(z_{c,o}), which decreases |g_c|. The
    concavity of Φ acts as a damper (MATHEMATICAL_FRAMEWORK §8).

    Limitation: greedy can miss complementary moves — two swaps that are
    each negative individually but positive together. For strategic
    awareness, use compute_ew_ceiling (Section 9e) to gauge how far
    from optimal the greedy result is.

    Output is a batch recommendation: the diff between the starting
    roster and the final optimized roster.

    Args:
        players: Silver table with FV column. MEW recomputed each iteration.
        max_moves: Maximum swaps.
        value_threshold: Stop when best Value < this.
        top_k: Candidates to exact-evaluate per iteration.
        include_bv: Include ΔBV in Value (slower but more accurate).

    Returns:
        {
            'moves': [                      # individual swaps in order
                {'drop': str, 'add': str, 'delta_ew': float,
                 'delta_bv': float, 'value': float},
                ...
            ],
            'drops': set[str],              # batch: all players to drop
            'adds': set[str],               # batch: all players to add
            'starting_ew': float,           # EW before optimization
            'final_ew': float,              # EW after all moves
            'total_value': float,           # total Value gained
        }
    """
```

---

## 10. `trade_finder.py` — Trade Evaluation Utility

Trades are mathematically identical to FA swaps — same `compute_exact_msv`,
same Value metric. The differences:

1. The "adds" come from an opponent's roster, not the FA pool.
2. Two PV constraints must be satisfied:
   - **Aggregate**: opponent's total PV loss ≤ `pv_max_loss_frac` of what they give up.
   - **Per-player max**: the most valuable player received can't vastly exceed
     the most valuable player sent (prevents trading up by quantity).
3. Trades require opponent agreement (cannot be executed unilaterally).
4. The affected opponent's totals change post-trade, requiring one extra
   MILP solve to re-solve their lineup (MATHEMATICAL_FRAMEWORK §7).

Because of (3), trades are NOT part of the greedy optimization loop.
They are evaluated on request and presented as recommendations.

### Constants

```python
DEFAULT_PV_TOLERANCE: float = 0.10  # ε: opponent tolerates losing up to ε PV
DEFAULT_TRADE_MAX_SIZE: int = 2     # max players per side
```

### 10a. `evaluate_trade` — Score a specific trade proposal

```python
def evaluate_trade(
    send_names: set[str],
    receive_names: set[str],
    my_roster_names: set[str],
    opponent_roster_names: set[str],
    trade_opponent_id: int,
    players: pd.DataFrame,
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
    current_ew: float,
    current_total_bv: float,
    pv_tolerance: float = DEFAULT_PV_TOLERANCE,
) -> dict:
    """Evaluate a specific trade, including opponent roster change and ΔBV.

    Same math as any swap (Value = MSV + ΔBV), plus PV check and opponent
    lineup re-solve. See MATHEMATICAL_FRAMEWORK §7 for full rationale.

    Steps:
        1. PV check: pv_balance = PV(send) − PV(receive). Must be ≥ −ε.
        2. My new roster: (my_roster − send) ∪ receive
        3. my_new_lineup = solve_lineup(my_new_roster, players, "MEW")
           my_new_totals = compute_totals_for_starters(set(my_new_lineup.keys()), players)
        4. Opponent's new roster: (opponent_roster − receive_from_opp) ∪ send_to_opp
        5. Solve opponent's lineup with FV → opp_new_totals (1 MILP)
        6. updated_opponent_totals = {**opponent_totals, trade_opponent_id: opp_new_totals}
        7. new_ew, _ = compute_win_probability(my_new_totals, updated_opponent_totals, σ)
        8. msv = new_ew − current_ew
        9. ΔBV (same approach as evaluate_top_k, Section 9c):
           a. new_gradient = compute_ew_gradient(my_new_totals, updated_opponent_totals, σ)
           b. Compute new_MEW for all players via add_mew (on working copy)
           c. new_total_bv = Σ BV(b) for my bench via add_bench_value(copy, my_new_lineup)
           d. delta_bv = new_total_bv − current_total_bv
       10. value = msv + delta_bv

    This captures both effects: (a) my totals improved, and (b) the trade
    partner's totals changed. Cost: 2 MILP solves (mine + opponent's).
    Steps 9a–9c are analytical — no MILP needed for BV.

    The baseline state (computed by compute_league_state) already uses
    MILP-optimal lineups for all teams, so both pre-trade and post-trade
    comparisons use MILP-optimal lineups — internally consistent.

    Note: ΔBV uses the gradient computed from updated_opponent_totals
    (step 9a), so it correctly reflects how the trade changes the
    competitive landscape (the opponent I traded with got stronger or
    weaker, shifting the gradient).

    A trade can be combined with FA moves in a single batch:
        send_names = {traded_out_1, traded_out_2, fa_drop_1}
        receive_names = {traded_in_1, fa_pickup_1, fa_pickup_2}
    The PV constraint applies only to the trade portion; FA moves are free.

    Returns:
        {
            'msv': float,
            'new_ew': float,
            'delta_bv': float,
            'value': float,           # msv + delta_bv
            'pv_balance': float,
            'pv_feasible': bool,
            'new_totals': dict,       # my post-trade team totals
            'new_lineup': dict,       # my post-trade lineup assignment
        }
    """
```

### 10b. `search_trades` — Find good trades automatically

```python
def search_trades(
    players: pd.DataFrame,
    my_roster_names: set[str],
    my_lineup: dict[str, str],
    opponent_rosters: dict[int, set[str]],
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
    current_ew: float,
    current_total_bv: float,
    pv_tolerance: float = DEFAULT_PV_TOLERANCE,
    max_trade_size: int = DEFAULT_TRADE_MAX_SIZE,
    top_k: int = 30,
    min_value: float = 0.1,
) -> list[dict]:
    """Enumerate and rank PV-feasible trades, including imbalanced.

    This is a search utility, not part of the core optimization loop.

    For each opponent o:
        1. TARGETS: their players with high MEW (I want them)
        2. CHIPS: my players with high PV but low MEW (expendable to me)
        3. Enumerate trade shapes (MATHEMATICAL_FRAMEWORK §1):
           - 1-for-1: {chip} ↔ {target}
           - 2-for-2: {chip₁, chip₂} ↔ {target₁, target₂}
           - 2-for-1 + FA fill: send {chip₁, chip₂} to opp, receive {target},
             pick up best FA to maintain roster size
           - 1-for-2 + FA drop: send {chip} to opp, receive {target₁, target₂},
             drop lowest-value roster player to maintain roster size
        4. Filter: PV constraint on opponent-routed portion only
           (FA-routed components are unconstrained)
        5. Score: MSV_approx + ΔBV_approx (MATHEMATICAL_FRAMEWORK §8 screening
           ranks all move types by approximate Value). MSV_approx from MEW
           differences; ΔBV_approx from gradient-based BV (same as screen_swaps).
        6. Exact-evaluate top K: evaluate_trade (my re-solve + opponent re-solve
           + ΔBV). Rank by value = msv + delta_bv.

    Imbalanced trades emerge naturally from the move abstraction
    (MATHEMATICAL_FRAMEWORK §1): FA-routed components fill or vacate
    roster spots to preserve |out| = |in|.

    Returns list of:
        {
            'send': list[str],
            'receive': list[str],
            'opponent': str,
            'msv_exact': float,
            'delta_bv': float,
            'value': float,           # msv_exact + delta_bv
            'pv_balance': float,
            'new_ew': float,
        }
    """
```

---

## 11. Testing — `tests/test_core.py`

Minimal, no classes, no fixtures, no mocking. Per AGENTS.md.

```python
def test_ew_gradient_sign_convention():
    """g_c > 0 for C⁺, g_c < 0 for C⁻."""
    # Synthetic my_totals, opponent_totals (two opponents, easy numbers)
    # Assert gradient['R'] > 0, gradient['ERA'] < 0

def test_mew_era_sign_check():
    """Low-ERA pitcher must have higher MEW than high-ERA pitcher."""
    # Two pitchers: same IP, ERA 2.50 vs 4.50
    # After add_mew(), assert MEW[low_ERA] > MEW[high_ERA]

def test_mew_unified_formula():
    """MEW formula produces correct results without hitter/pitcher branching."""
    # Hitter: verify PA > 0, IP = 0 → only hitting terms contribute
    # Pitcher: verify PA = 0, IP > 0 → only pitching terms contribute
    # Both: verify single vectorized formula gives same result

def test_ratio_stat_delta_trap():
    """Replacing below-average-ERA pitcher with fewer IP can worsen ERA."""
    # Team: 1000 IP, 3.00 ERA
    # Remove: 200 IP, 2.80 ERA. Add: 50 IP, 2.50 ERA.
    # ΔERA ≈ [50×(2.50−3.00) − 200×(2.80−3.00)] / 1000 = +0.015 (worsens)

def test_team_totals_weighted_average():
    """ERA/OPS must be IP/PA-weighted averages, not sums."""
    # Two pitchers: (100 IP, 3.00 ERA), (50 IP, 4.00 ERA)
    # Team ERA = (300 + 200) / 150 = 3.333, NOT 7.0

def test_msv_identity_swap():
    """Swapping a player with themselves → MSV = 0."""

def test_pv_constraint_filters_correctly():
    """Trades where PV(send) − PV(receive) < −ε must be excluded."""

def test_gradient_based_bv_position_aware():
    """Bench player eligible for high-absence slot should have higher BV."""
    # Two bench players with identical MEW, one eligible at C (0.25 absence),
    # one eligible at UTIL (0.15 absence). Verify BV(C-eligible) > BV(UTIL-eligible).

def test_mew_lineup_differs_from_fv_lineup():
    """My lineup using MEW should differ from FV when gradient is non-uniform."""
    # Create scenario where gradient heavily weights one category.
    # MEW-optimal lineup should pick player strong in that category,
    # even if their FV is lower than an alternative.
```

---

## 12. Critical Implementation Warnings

### W1. Sign convention for ERA/WHIP

g_ERA < 0. MEW_ERA_component = g_ERA × IP × (ERA − my_ERA) / total_IP.
- Good pitcher (ERA < my_ERA): (−) × (−) = positive ✓
- Bad pitcher (ERA > my_ERA): (−) × (+) = negative ✓

**If a low-ERA pitcher has negative MEW, there is a sign error.**

### W2. Ratio stat deltas subtract team average

```
ΔOPS ≈ [PA(in) × (OPS(in) − my_OPS) − PA(out) × (OPS(out) − my_OPS)] / total_PA
```

The `− my_OPS` terms account for the denominator change in the weighted average. Omitting them is wrong.

### W3. MEW is stale after roster changes

MEW depends on the gradient, which depends on team totals, which depend on who's on the roster. After each greedy swap, MEW must be recomputed from scratch. Reusing old MEW values will produce incorrect rankings.

### W4. My lineup uses MEW, opponent lineups use FV

My team's lineup MILP maximizes Σ MEW(starters) — context-aware, reflecting current category needs via the gradient. Opponent lineups maximize Σ FV(starters) — context-free, modeling their optimization behavior.

**Do not mix these:** using FV for my lineup forfeits gradient information (the whole point of MEW is that different categories matter different amounts right now). Using MEW for opponents requires a gradient we don't compute for them.

The MEW-lineup creates a circularity (lineup → totals → gradient → MEW → lineup), resolved by the fixed-point iteration in `compute_league_state` (Section 7). See MATHEMATICAL_FRAMEWORK §5 for the convergence analysis.

### W5. BV uses MEW-ranked, position-eligible FAs

When computing BV for bench player b at starter slot k:
- `best_FA(k)` is the highest-**MEW** free agent **eligible for slot k**
- Position eligibility is required: BV measures insurance for a specific slot, not general roster depth
- MEW (not FV) is the correct ranking metric because it reflects what contributes most to my team's EW

Do NOT use FV or ignore position eligibility — both would incorrectly estimate the insurance gap.

### W6. Player names always carry -H/-P suffix

All internal operations use suffixed names. `strip_name_suffix()` is for display only.

### W7. σ_c is projection uncertainty, NOT observed cross-team variance

`category_sigmas` represents how much actual season outcomes could deviate from
projections. It is calibrated from projection noise models (CV × league_mean for
counting stats, fixed absolute σ for ratio stats).

**Do NOT use cross-team spread** (the variance of team totals across the league).
Cross-team spread reflects strategic choices (some teams punt SV, others invest
heavily), not season-to-season uncertainty. Using it would inflate σ for
categories with outlier teams, making the model insensitive to real matchups.

### W8. All quantities are season-level (roto has no weekly structure)

This is a rotisserie league: standings are determined by season-end cumulative
totals, not weekly matchups. EW, ΔEW, and BV are all season-level quantities.
`POSITION_ABSENCE_RATES` are dimensionless fractions (proportion of remaining
season absent), so BV = Σ f_absent × ΔEW is natively in the same season-EW
units as ΔEW_healthy. No per-week conversion exists or is needed.

### W9. Trade evaluation re-solves opponent lineup

`evaluate_trade` re-solves the affected opponent's lineup (FV objective) after
their roster changes from the trade, then uses the updated opponent totals when
computing the post-trade EW. This costs 1 extra MILP per trade evaluation
(MATHEMATICAL_FRAMEWORK §7). The baseline state computation already MILP-optimizes
every opponent's lineup, so pre-trade and post-trade comparisons are internally
consistent: both use MILP-optimal lineups.

For FA swaps (`compute_exact_msv`), opponent totals are unchanged — no extra solve needed.

### W10. Trade non-convexity: evaluate combined moves

Multiple trades with the same opponent interact: trade A and trade B may each
satisfy the PV constraint individually, but doing both changes both teams'
rosters, potentially making the combined package infeasible or suboptimal
(MATHEMATICAL_FRAMEWORK §7). When considering multiple trades with one opponent,
evaluate the **combined** move (all sends and receives as one batch), not
individual trades independently.

### W11. Fixed-point iteration convergence

The MEW-lineup iteration in `compute_league_state` converges in 1–2 steps because
the gradient is self-correcting: improving in category c increases z_{c,o}, which
decreases φ(z_{c,o}), which decreases |g_c|. Concavity of Φ acts as a damper.

If the iteration reaches `_MAX_LINEUP_ITERATIONS` without converging (oscillation
between two lineups), evaluate both and pick the one with higher actual EW. Log
a warning. Total cost: at most 3 MILP solves (< 3ms).

### W12. Screening MSV must be lineup-aware

The naive formula `MSV_approx = MEW(FA) − MEW(drop)` is **catastrophically wrong**
when dropping a bench player. A bench player contributes zero EW to the starting
lineup — their value is captured entirely by BV (Section 9d). The naive formula
incorrectly attributes starter-level contribution to bench players, producing
phantom MSV_approx values of 4–5 EW for swaps with zero actual MSV.

The correct approach (`_lineup_aware_msv` in `screen_swaps`): determine who actually
enters and leaves the starting lineup after the swap, and compute MSV_approx from
that lineup change. See MATHEMATICAL_FRAMEWORK §4 for the full case analysis.

### W13. EW surface convexity biases gradient-based screening

The gradient evaluates marginal value at the current team totals. In categories
where the team is deeply losing (avg |z| > 1), the EW surface is **convex**:
actual gains from improvement are **larger** than the linear prediction. In
categories where the team is deeply winning, the surface is concave: gains are
smaller than predicted.

This means MEW-based screening systematically undervalues improvements in
deeply-losing categories. For small perturbations (single FA swaps), the bias
is modest and corrected by exact evaluation. For large perturbations (trades
shifting a category by > 0.5σ), the gradient may undervalue the trade by
1.5–2×. When evaluating trades that significantly shift a deeply-losing
category, always compute exact EW. See MATHEMATICAL_FRAMEWORK §4a.

---

## 13. Implementation Order for Parallel Subagents

**Phase 1** (no dependencies, start immediately):
- config.py — load from repo-root config.json
- players.py — port from v1
- win_model.py — port from v1

**Phase 2** (depends on Phase 1):
- lineup_solver.py — port from v1, add objective_column parameter
- test_core.py — test stubs can start in Phase 1

**Phase 3** (depends on Phase 2):
- player_scoring.py — FV/PV from v1, add MEW (takes gradient as input)

**Phase 4** (depends on Phase 3):
- league_state.py — new: fixed-point iteration, calls add_mew

**Phase 5** (depends on Phase 4):
- swap_evaluator.py — new: screening, exact evaluation, gradient-based BV, greedy optimizer
- trade_finder.py — new: trade evaluation with opponent re-solve

**Critical path:** config → win_model → lineup_solver → player_scoring → league_state → swap_evaluator
