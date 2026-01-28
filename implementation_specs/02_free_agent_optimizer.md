# Free Agent Optimizer

## Overview

The Free Agent Optimizer answers the question: **"Given a pool of available players, what is the optimal 26-player roster that maximizes expected wins against known opponents?"**

This module uses Mixed-Integer Linear Programming (MILP) to find the globally optimal roster subject to positional constraints, roster composition bounds, and category win objectives.

**Module:** `optimizer/roster_optimizer.py`

**Key Design Decision:** All rostered players' stats count toward team totals (not just starters). The starting lineup slots exist only to enforce positional requirements—a valid roster must be able to field a legal starting lineup. Bench players contribute stats equally to starters.

---

## Imports and Constants

```python
import pulp
from pulp import lpSum, LpVariable, value
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from .data_loader import (
    HITTING_CATEGORIES,
    PITCHING_CATEGORIES,
    ALL_CATEGORIES,
    NEGATIVE_CATEGORIES,
    RATIO_STATS,
    ROSTER_SIZE,
    HITTING_SLOTS,
    PITCHING_SLOTS,
    SLOT_ELIGIBILITY,
    MIN_HITTERS,
    MAX_HITTERS,
    MIN_PITCHERS,
    MAX_PITCHERS,
    NUM_OPPONENTS,
    compute_team_totals,
)

# === MILP CONSTANTS ===

# Big-M values for indicator constraints
BIG_M_COUNTING = 10000  # For counting stats (R, HR, RBI, SB, W, SV, K)
BIG_M_RATIO = 5000      # For ratio stat linearized forms (OPS, ERA, WHIP)

# Epsilon values for strict inequality
EPSILON_COUNTING = 0.5   # For integer-valued counting stats
EPSILON_RATIO = 0.001    # For continuous ratio stats
```

---

## Candidate Prefiltering

Before solving the MILP, we filter to a manageable set of candidate players.

```python
def filter_candidates(
    projections: pd.DataFrame,
    quality_scores: pd.DataFrame,
    my_roster_names: set[str],
    opponent_roster_names: set[str],
    top_n_per_position: int = 30,
    top_n_per_category: int = 10,
) -> pd.DataFrame:
    """
    Filter to candidate players for optimization.
    
    Include:
        1. ALL players currently on my roster (must be candidates)
        2. Top N available players at each position by quality score
        3. Top M available players in each scoring category (specialists)
    
    Exclude:
        - All players on opponent rosters (unavailable)
    
    Args:
        projections: Combined projections DataFrame
        quality_scores: DataFrame with quality_score per player
        my_roster_names: Set of player names on my roster
        opponent_roster_names: Set of ALL names on ANY opponent roster
        top_n_per_position: Players to keep per position (default 30)
        top_n_per_category: Top players per category to ensure included (default 10)
    
    Returns:
        Filtered projections DataFrame containing only candidates.
    
    Implementation:
        1. available_pool = projections excluding opponent_roster_names
        2. Join quality_scores to available_pool
        3. For each slot type in SLOT_ELIGIBILITY:
           - Find players whose Position is in SLOT_ELIGIBILITY[slot]
           - Take top N by quality_score
           - Add to candidate set
        4. For each scoring category:
           - For R, HR, RBI, SB, W, SV, K, OPS: take top M by highest value
           - For ERA, WHIP: take top M by LOWEST value (lower is better)
           - Add to candidate set
        5. Add all my_roster_names to candidate set
        6. Return projections filtered to candidate set
    
    Print:
        "Filtered to {X} candidates from {Y} total players"
        "  - {H} hitters, {P} pitchers"
    """
```

---

## MILP Mathematical Formulation

### Index Sets

```
I     = {0, 1, ..., n-1} = candidate player indices
I_H   = subset of I where player_type == 'hitter'
I_P   = subset of I where player_type == 'pitcher'
J     = {1, 2, 3, 4, 5, 6} = opponent team_ids
C_H   = {'R', 'HR', 'RBI', 'SB', 'OPS'} = hitting categories
C_P   = {'W', 'SV', 'K', 'ERA', 'WHIP'} = pitching categories
C     = C_H ∪ C_P = all 10 categories
S_H   = {'C', '1B', '2B', 'SS', '3B', 'OF', 'UTIL'} = hitting slot types
S_P   = {'SP', 'RP'} = pitching slot types
S     = S_H ∪ S_P = all slot types
```

### Decision Variables

**Player selection:**
```
x[i] ∈ {0, 1}    for all i ∈ I
```
`x[i] = 1` means player i is on my roster.

**Slot assignment:**
```
a[i,s] ∈ {0, 1}    for all i ∈ I, s ∈ S where player i is eligible for slot s
```
`a[i,s] = 1` means player i is assigned to start in slot type s.

Only create `a[i,s]` variables for eligible (player, slot) pairs:
```python
player_position = candidates.iloc[i]['Position']
is_eligible = player_position in SLOT_ELIGIBILITY[s]
```

**Beat indicators:**
```
y[j,c] ∈ {0, 1}    for all j ∈ J, c ∈ C
```
`y[j,c] = 1` means I beat opponent j in category c.

### Objective Function

Maximize total opponent-category wins:
```
maximize  Σ_{j ∈ J} Σ_{c ∈ C} y[j,c]
```

With 6 opponents and 10 categories, maximum possible is 60.

### Constraints

#### C1: Roster Size
```
Σ_{i ∈ I} x[i] = ROSTER_SIZE (26)
```

#### C2: Slot Assignment Requires Rostering
```
a[i,s] ≤ x[i]    for all (i,s) where a[i,s] exists
```

#### C3: Each Player in At Most One Slot
```
Σ_{s : (i,s) ∈ a} a[i,s] ≤ 1    for all i ∈ I
```

#### C4: Starting Slots Must Be Filled
```
Σ_{i : (i,s) ∈ a} a[i,s] = n_s    for all s ∈ S
```
Where `n_s` is the required count for slot type s (e.g., `n_OF = 3`).

#### C5: Roster Composition Bounds
```
MIN_HITTERS ≤ Σ_{i ∈ I_H} x[i] ≤ MAX_HITTERS
MIN_PITCHERS ≤ Σ_{i ∈ I_P} x[i] ≤ MAX_PITCHERS
```

#### C6: Beat Constraints for Counting Stats

For counting stats (R, HR, RBI, SB, W, SV, K) where higher is better:

```
Σ_{i ∈ I_relevant} M[i,c] * x[i]  ≥  O[j,c] + ε - B * (1 - y[j,c])
```

Where:
- `I_relevant` = `I_H` for hitting categories, `I_P` for pitching categories
- `M[i,c]` = player i's projected value in category c
- `O[j,c]` = opponent j's total in category c
- `ε = EPSILON_COUNTING = 0.5`
- `B = BIG_M_COUNTING = 10000`

**How it works:**
- If `y[j,c] = 1`: constraint becomes `my_total ≥ O[j,c] + 0.5` (must beat them)
- If `y[j,c] = 0`: constraint becomes `my_total ≥ O[j,c] - 9999.5` (always satisfied)

#### C7: Beat Constraints for OPS (Higher is Better)

OPS requires linearization since it's a weighted average.

**Goal:** I beat opponent j if `my_OPS > opponent_OPS[j]`

**Linearization:**
```
my_OPS > O[j, OPS]
⟺  Σ(PA[i] * OPS[i] * x[i]) / Σ(PA[i] * x[i]) > O[j, OPS]
⟺  Σ(PA[i] * OPS[i] * x[i]) > O[j, OPS] * Σ(PA[i] * x[i])
⟺  Σ PA[i] * (OPS[i] - O[j, OPS]) * x[i] > 0
```

**Constraint:**
```
Σ_{i ∈ I_H} PA[i] * (OPS[i] - O[j, OPS]) * x[i]  ≥  ε - B * (1 - y[j, OPS])
```

#### C8: Beat Constraints for ERA and WHIP (Lower is Better)

For ERA, I beat opponent j if `my_ERA < opponent_ERA[j]`.

**Linearization (note the sign flip!):**
```
my_ERA < O[j, ERA]
⟺  Σ(IP[i] * ERA[i] * x[i]) < O[j, ERA] * Σ(IP[i] * x[i])
⟺  Σ IP[i] * (O[j, ERA] - ERA[i]) * x[i] > 0
```

**Constraint:**
```
Σ_{i ∈ I_P} IP[i] * (O[j, ERA] - ERA[i]) * x[i]  ≥  ε - B * (1 - y[j, ERA])
```

Same pattern for WHIP.

**Critical:** The coefficient is `(opponent_value - player_value)` for ERA/WHIP, but `(player_value - opponent_value)` for OPS.

---

## MILP Implementation

```python
def build_and_solve_milp(
    candidates: pd.DataFrame,
    opponent_totals: dict[int, dict[str, float]],
    current_roster_names: set[str],
) -> tuple[list[str], dict]:
    """
    Build and solve the MILP for optimal roster construction.
    
    Args:
        candidates: DataFrame of candidate players (filtered projections).
                    Must have: Name, Position, player_type, and all stat columns.
        opponent_totals: Dict mapping team_id to category totals.
        current_roster_names: Names currently on my roster (for logging changes).
    
    Returns:
        optimal_roster_names: List of player Names for the optimal roster
        solution_info: Dict with:
            - 'objective': float (opponent-category wins, max 60)
            - 'solve_time': float (seconds)
            - 'status': str
    
    Implementation Steps:
    
    1. Build index sets:
        I = list(range(len(candidates)))
        I_H = [i for i in I if candidates.iloc[i]['player_type'] == 'hitter']
        I_P = [i for i in I if candidates.iloc[i]['player_type'] == 'pitcher']
        J = [1, 2, 3, 4, 5, 6]
    
    2. Create decision variables:
        x = {i: LpVariable(f"x_{i}", cat='Binary') for i in I}
        a = {}  # Only for eligible (i, s) pairs
        y = {(j, c): LpVariable(f"y_{j}_{c}", cat='Binary') for j in J for c in ALL_CATEGORIES}
    
    3. Set objective:
        prob += lpSum(y[j, c] for j in J for c in ALL_CATEGORIES)
    
    4. Add constraints C1-C8 as described above
    
    5. Pre-solve validation:
        For each slot type, assert enough eligible candidates exist
    
    6. Solve:
        solver = pulp.HiGHS_CMD(msg=True, timeLimit=300)
        status = prob.solve(solver)
        
    7. Assert optimal:
        assert status == pulp.LpStatusOptimal, (
            f"Solver failed: {pulp.LpStatus[status]}. "
            f"Check position slot eligibility."
        )
    
    8. Extract solution:
        roster = [candidates.iloc[i]['Name'] for i in I if value(x[i]) > 0.5]
    
    Print:
        "Building MILP with {N} candidates..."
        "  Variables: {X} player, {Y} slot, {Z} beat"
        "  Constraints: {W} total"
        "Solving..."
        "Solved in {T:.1f}s — objective: {obj}/60 opponent-category wins"
    """
```

---

## Solution Extraction and Output

```python
def compute_standings(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
) -> pd.DataFrame:
    """
    Compute projected standings for each category.
    
    Returns:
        DataFrame with columns:
            category, my_value, opp_1, opp_2, ..., opp_6, my_rank, wins
        
        my_rank: 1 = first place, 7 = last place
        wins: number of opponents I beat (0-6)
    
    For negative categories (ERA, WHIP), lower value = better rank.
    """


def print_roster_summary(
    roster_names: list[str],
    projections: pd.DataFrame,
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    old_roster_names: set[str] | None = None,
) -> None:
    """
    Print a formatted summary of the optimal roster.
    
    Sections:
    
    1. ROSTER (26 players)
       Split into Hitters and Pitchers.
       Show: position, name (suffix stripped), team, relevant stats.
       
    2. CHANGES (if old_roster_names provided)
       List players added and dropped with their stats.
       Flag low-PA/IP players with warning symbols.
    
    3. STANDINGS PROJECTION
       Table showing my value vs each opponent in each category.
       Mark wins with indicator.
       
    4. SUMMARY
       "Total opponent-category wins: X / 60"
       "Projected roto points: Y / 70"
       
    Roto points = Σ (8 - rank) across 10 categories.
    """
```

---

## Sensitivity Analysis

This is computationally expensive but provides valuable insights.

```python
def compute_player_sensitivity(
    optimal_roster_names: list[str],
    candidates: pd.DataFrame,
    opponent_totals: dict[int, dict[str, float]],
) -> pd.DataFrame:
    """
    Compute sensitivity of objective to each player.
    
    For each candidate player:
        - If ON optimal roster: solve MILP forcing x[i] = 0 (exclude them)
        - If NOT on roster: solve MILP forcing x[i] = 1 (include them)
        - Compare resulting objective to unconstrained optimum
    
    Returns:
        DataFrame with columns:
            Name, player_type, Position, on_optimal_roster,
            forced_objective, objective_delta
        
        objective_delta = forced_objective - optimal_objective
        (negative means forcing this player's in/out makes us worse)
    
    Implementation:
        This requires solving len(candidates) MILPs.
        Each solve is independent — build fresh MILP with forcing constraint.
        HiGHS does NOT support warm-starting in PuLP.
        
        Use tqdm: "Computing player sensitivities"
        Expected time: ~1-2s per player, ~5-15 minutes for 300 candidates.
    
    Print:
        "Computing sensitivity for {N} candidates (est. {T} minutes)"
        "Note: Each solve starts fresh (HiGHS doesn't warm-start)"
    """
```

---

## Edge Cases and Implementation Notes

1. **Position eligibility:** Test if `player['Position'] in SLOT_ELIGIBILITY[slot]`.

2. **OPS already exists:** The FanGraphs CSV has OPS. Do NOT recompute.

3. **Strikeouts:** Rename `SO` → `K` during pitcher load.

4. **Pitcher positions:** `'SP' if GS >= 3 else 'RP'`

5. **Hitter position fallback:** Default to `'DH'` if MLBAMID not in database.

6. **Ratio stat linearization signs:**
   - OPS: `PA * (OPS_player - OPS_opponent)`
   - ERA: `IP * (ERA_opponent - ERA_player)` — note flipped order!
   - WHIP: `IP * (WHIP_opponent - WHIP_player)`

7. **Filter by player_type:** Hitting constraints sum only over I_H. Pitching constraints sum only over I_P. Common bug: accidentally including pitchers in OPS calculation.

8. **Big-M values:**
   - Counting stats: 10000
   - Ratio stats: 5000
   - Too small = falsely constrains; too large = numerical issues

9. **Epsilon values:**
   - Counting stats: 0.5 (ensures strict inequality for integers)
   - Ratio stats: 0.001 (small positive for continuous values)

10. **Variable naming:** Use `f"x_{i}"` with integer index i. Never put player names in variable names (special characters break PuLP).

11. **Zero PA or IP:** Assert `sum(PA) > 0` for hitters, `sum(IP) > 0` for pitchers for ALL teams including opponents.

12. **Infeasibility:** If solver returns non-optimal, identify which position slot is problematic in the error message.

---

## Validation Checklist

Before the optimizer is complete:

- [ ] All functions are module-level (no classes)
- [ ] All assertions have descriptive messages
- [ ] No try/except blocks
- [ ] tqdm for sensitivity analysis loop
- [ ] print() for status at each stage
- [ ] Uses `pulp.HiGHS_CMD()` with `highspy` package installed
- [ ] Ratio stat linearization has correct signs
- [ ] Beat constraints filter by player_type
- [ ] Position eligibility via set membership
- [ ] Pre-solve validation for slot eligibility
- [ ] Infeasibility error identifies problematic slot
- [ ] sum(PA) > 0 and sum(IP) > 0 asserted for all teams
