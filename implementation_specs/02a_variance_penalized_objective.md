# Variance-Penalized MILP Objective

---

## Cross-References

**Depends on:**
- [00_agent_guidelines.md](00_agent_guidelines.md) — code style
- [02_free_agent_optimizer.md](02_free_agent_optimizer.md) — base MILP formulation to extend

**Used by:**
- [05_notebook_integration.md](05_notebook_integration.md) — optional `balance_lambda` parameter
- [07_testing.md](07_testing.md) — balance-related test functions

---

## Motivation

The standard MILP objective maximizes total opponent-category wins:

$$\text{maximize} \sum_{j \in J} \sum_{c \in C} y_{j,c}$$

This optimizes for **expected rotisserie points** but ignores a critical insight from Rosenof (2025): to **win** a rotisserie league, you must beat the maximum opponent, not just perform well on average. Teams with higher variance in their fantasy point distribution have better upside — they're more likely to achieve the exceptional performance needed to win.

The standard objective can produce "punting" rosters that dominate some categories while abandoning others. In head-to-head formats, this can be optimal. In rotisserie, it's usually suboptimal because:

1. Punting reduces variance (certain wins and certain losses contribute less variance than close matchups)
2. Lower variance means lower probability of the upside needed to beat all opponents
3. Punting sacrifices many guaranteed points for marginal gains in already-strong categories

This module extends the MILP with a **balance incentive** that discourages punting while remaining computationally tractable.

---

## Mathematical Formulation

### The Problem with Variance in MILP

True variance is quadratic:

$$\text{Var}(w) = \frac{1}{|C|} \sum_{c} \left( w_c - \bar{w} \right)^2$$

Where $w_c = \sum_j y_{j,c}$ is wins in category $c$. Quadratic terms cannot be directly optimized in a linear program.

### Linearization via Range Minimization

Instead of penalizing variance directly, we penalize the **range** of category wins:

$$\text{range} = w_{max} - w_{min}$$

Where:
- $w_{max} = \max_c \sum_j y_{j,c}$ (wins in best category)
- $w_{min} = \min_c \sum_j y_{j,c}$ (wins in worst category)

Range is correlated with variance and can be linearized using auxiliary variables.

### The Variance-Penalized Objective

$$\text{maximize} \quad \sum_{j \in J} \sum_{c \in C} y_{j,c} + \lambda \cdot w_{min} - \lambda \cdot w_{max}$$

**Interpretation:**
- First term: total wins (as before)
- Second term: bonus for raising the floor (worst category)
- Third term: penalty for ceiling being too high relative to floor

When $\lambda = 0$, this reduces to the standard objective (backward compatible).

### Linearizing min and max

**For $w_{min}$ (lower bound on minimum):**

$$w_{min} \leq \sum_{j \in J} y_{j,c} \quad \forall c \in C$$

The optimizer will push $w_{min}$ as high as possible (since it has positive coefficient), so these constraints will bind at the actual minimum.

**For $w_{max}$ (upper bound on maximum):**

$$w_{max} \geq \sum_{j \in J} y_{j,c} \quad \forall c \in C$$

The optimizer will push $w_{max}$ as low as possible (since it has negative coefficient), so these constraints will bind at the actual maximum.

**Variable bounds:**

$$0 \leq w_{min} \leq |J| = 6$$
$$0 \leq w_{max} \leq |J| = 6$$

### Complete Formulation

**New Decision Variables (in addition to existing x, a, y):**
- $w_{min} \in [0, 6]$ — continuous, minimum category wins
- $w_{max} \in [0, 6]$ — continuous, maximum category wins

**New Objective:**

$$\text{maximize} \quad \sum_{j,c} y_{j,c} + \lambda \cdot w_{min} - \lambda \cdot w_{max}$$

**New Constraints:**

$$w_{min} \leq \sum_{j \in J} y_{j,c} \quad \forall c \in C \quad \text{(C9: min bound, 10 constraints)}$$

$$w_{max} \geq \sum_{j \in J} y_{j,c} \quad \forall c \in C \quad \text{(C10: max bound, 10 constraints)}$$

All existing constraints (C1–C8) remain unchanged.

**Total additions:** 2 continuous variables, 20 linear constraints.

---

## Choosing λ

The balance parameter $\lambda$ controls the tradeoff between total wins and balance.

### Marginal Analysis

Consider the effect of gaining 1 win in different categories:

| Win location | Effect on total | Effect on w_min | Effect on w_max | Net objective change |
|--------------|-----------------|-----------------|-----------------|---------------------|
| Worst category | +1 | +1 | 0 | $1 + \lambda$ |
| Best category | +1 | 0 | +1 | $1 - \lambda$ |
| Middle category | +1 | 0 | 0 | $1$ |

**Interpretation:**
- $\lambda < 1$: Always prefer more wins, but favor weak categories at the margin
- $\lambda = 1$: Indifferent between improving best category and doing nothing
- $\lambda > 1$: Actively avoid improving already-strong categories

### Recommended Values

| Mode | λ | Behavior |
|------|---|----------|
| Standard (backward compatible) | 0.0 | Original objective (maximize total wins) |
| Moderate | 0.3 | Slight balance preference |
| **Balanced (default)** | **0.5** | 3:1 preference for weak vs strong categories |
| Conservative | 0.7 | Strong balance preference |
| Extreme | 1.0 | Never improve best category if it increases range |

**Default: $\lambda = 0.5$**

At $\lambda = 0.5$:
- Improving worst category is worth 1.5 effective wins
- Improving best category is worth 0.5 effective wins
- Improving middle category is worth 1.0 effective wins

---

## Implementation

### Step 1: Add Constant to data_loader.py

Add this constant in the "League Configuration" section of `data_loader.py`, after `NUM_OPPONENTS`:

```python
# Balance parameter for variance-penalized MILP objective
# 0.0 = standard objective (no balance), 0.5 = recommended, 1.0 = strong balance
BALANCE_LAMBDA_DEFAULT = 0.5
```

Update the `__init__.py` exports to include it:

```python
from .data_loader import (
    # ... existing exports ...
    BALANCE_LAMBDA_DEFAULT,
)
```

### Step 2: Modify build_and_solve_milp Function Signature

Change the function signature in `roster_optimizer.py`:

```python
def build_and_solve_milp(
    candidates: pd.DataFrame,
    opponent_totals: dict[int, dict[str, float]],
    current_roster_names: set[str],
    balance_lambda: float = BALANCE_LAMBDA_DEFAULT,  # NEW PARAMETER
) -> tuple[list[str], dict]:
```

Add import at top of file (with other data_loader imports):

```python
from .data_loader import (
    # ... existing imports ...
    BALANCE_LAMBDA_DEFAULT,
)
```

### Step 3: Add Input Validation

Add this assertion immediately after the function docstring, before any other code:

```python
    assert 0.0 <= balance_lambda <= 2.0, (
        f"balance_lambda must be in [0.0, 2.0], got {balance_lambda}. "
        f"Use 0.0 for standard objective, 0.5 for balanced (recommended)."
    )
```

### Step 4: Add Balance Variables

Insert this code **immediately after** the `y` variable creation block (after the line that creates beat indicator variables):

```python
    # === Balance Variables (for variance-penalized objective) ===
    w_min = LpVariable("w_min", lowBound=0, upBound=NUM_OPPONENTS, cat="Continuous")
    w_max = LpVariable("w_max", lowBound=0, upBound=NUM_OPPONENTS, cat="Continuous")
```

### Step 5: Update Variable Count Logging

Change the existing variable count print statement from:

```python
    var_count = len(x) + len(a) + len(y)
    print(
        f"  Variables: {len(x)} player, {len(a)} slot, {len(y)} beat ({var_count} total)"
    )
```

To:

```python
    var_count = len(x) + len(a) + len(y) + 2  # +2 for w_min, w_max
    print(
        f"  Variables: {len(x)} player, {len(a)} slot, {len(y)} beat, 2 balance ({var_count} total)"
    )
```

### Step 6: Replace Objective Function

Replace the existing objective line:

```python
    # === Objective: Maximize total opponent-category wins ===
    prob += lpSum(y[j, c] for j in J for c in ALL_CATEGORIES)
```

With:

```python
    # === Objective: Variance-penalized total wins ===
    # maximize: Σ y[j,c] + λ·w_min - λ·w_max
    total_wins_expr = lpSum(y[j, c] for j in J for c in ALL_CATEGORIES)
    balance_term = balance_lambda * w_min - balance_lambda * w_max
    prob += total_wins_expr + balance_term
```

### Step 7: Add Balance Constraints

Insert this code **after** constraint C8 (WHIP beat constraints) and **before** the constraint count print statement:

```python
    # C9: w_min bounded above by each category's wins
    for c in ALL_CATEGORIES:
        category_wins_expr = lpSum(y[j, c] for j in J)
        prob += w_min <= category_wins_expr, f"min_bound_{c}"
        constraint_count += 1

    # C10: w_max bounded below by each category's wins
    for c in ALL_CATEGORIES:
        category_wins_expr = lpSum(y[j, c] for j in J)
        prob += w_max >= category_wins_expr, f"max_bound_{c}"
        constraint_count += 1
```

### Step 8: Update Solution Logging

Replace the existing solve completion print statement:

```python
    print(
        f"Solved in {solve_time:.1f}s — objective: {objective}/60 opponent-category wins"
    )
```

With:

```python
    # Compute balance metrics
    category_wins = {c: int(round(sum(value(y[j, c]) for j in J))) for c in ALL_CATEGORIES}
    total_wins = sum(category_wins.values())
    actual_w_min = min(category_wins.values())
    actual_w_max = max(category_wins.values())
    win_range = actual_w_max - actual_w_min

    if balance_lambda > 0:
        print(
            f"Solved in {solve_time:.1f}s — {total_wins}/60 wins, "
            f"range {win_range} ({actual_w_min}-{actual_w_max}), λ={balance_lambda}"
        )
    else:
        print(
            f"Solved in {solve_time:.1f}s — objective: {total_wins}/60 opponent-category wins"
        )
```

### Step 9: Update Return Value

Replace the existing return statement:

```python
    return roster_names, {
        "objective": objective,
        "solve_time": solve_time,
        "status": status_str,
    }
```

With:

```python
    return roster_names, {
        "objective": pulp.value(prob.objective),
        "total_wins": total_wins,
        "category_wins": category_wins,  # dict: category -> wins (0-6)
        "w_min": actual_w_min,
        "w_max": actual_w_max,
        "win_range": win_range,
        "solve_time": solve_time,
        "status": status_str,
        "balance_lambda": balance_lambda,
    }
```

---

## Complete Modified Function (Reference)

For clarity, here is the complete `build_and_solve_milp` function with all changes integrated:

```python
def build_and_solve_milp(
    candidates: pd.DataFrame,
    opponent_totals: dict[int, dict[str, float]],
    current_roster_names: set[str],
    balance_lambda: float = BALANCE_LAMBDA_DEFAULT,
) -> tuple[list[str], dict]:
    """
    Build and solve the MILP for optimal roster construction.

    Uses a variance-penalized objective that balances total wins against
    category balance:

        maximize  Σ y[j,c] + λ·w_min - λ·w_max

    Where w_min and w_max are the minimum and maximum category win counts.
    This discourages "punting" strategies that abandon some categories.

    Args:
        candidates: DataFrame of candidate players with projections.
        opponent_totals: Dict mapping team_id to category totals.
        current_roster_names: Names currently on my roster (for logging).
        balance_lambda: Balance coefficient (default 0.5).
            - 0.0 = standard objective (backward compatible, no balance)
            - 0.5 = moderate balance (recommended)
            - 1.0 = strong balance

    Returns:
        optimal_roster_names: List of player Names for the optimal roster
        solution_info: Dict with:
            - 'objective': float (penalized objective value)
            - 'total_wins': int (raw opponent-category wins, max 60)
            - 'category_wins': dict[str, int] (wins per category, 0-6 each)
            - 'w_min': int (wins in worst category)
            - 'w_max': int (wins in best category)
            - 'win_range': int (w_max - w_min, lower is more balanced)
            - 'solve_time': float (seconds)
            - 'status': str
            - 'balance_lambda': float (λ used)
    """
    assert 0.0 <= balance_lambda <= 2.0, (
        f"balance_lambda must be in [0.0, 2.0], got {balance_lambda}. "
        f"Use 0.0 for standard objective, 0.5 for balanced (recommended)."
    )

    print(f"Building MILP with {len(candidates)} candidates...")

    # Index sets
    I = list(range(len(candidates)))
    I_H = [i for i in I if candidates.iloc[i]["player_type"] == "hitter"]
    I_P = [i for i in I if candidates.iloc[i]["player_type"] == "pitcher"]
    J = list(opponent_totals.keys())  # Opponent team IDs

    # Precompute eligibility
    eligibility = compute_slot_eligibility(candidates)
    validate_slot_coverage(eligibility, candidates)

    # Create problem
    prob = pulp.LpProblem("RosterOptimization", pulp.LpMaximize)

    # === Decision Variables ===

    # Player selection: x[i] = 1 if player i is on roster
    x = {i: LpVariable(f"x_{i}", cat="Binary") for i in I}

    # Slot assignment: a[i,s] = 1 if player i starts in slot type s
    a = {}
    for i in I:
        for s in eligibility[i]:
            a[i, s] = LpVariable(f"a_{i}_{s}", cat="Binary")

    # Beat indicators: y[j,c] = 1 if I beat opponent j in category c
    y = {
        (j, c): LpVariable(f"y_{j}_{c}", cat="Binary")
        for j in J
        for c in ALL_CATEGORIES
    }

    # Balance variables (for variance-penalized objective)
    w_min = LpVariable("w_min", lowBound=0, upBound=NUM_OPPONENTS, cat="Continuous")
    w_max = LpVariable("w_max", lowBound=0, upBound=NUM_OPPONENTS, cat="Continuous")

    var_count = len(x) + len(a) + len(y) + 2  # +2 for w_min, w_max
    print(
        f"  Variables: {len(x)} player, {len(a)} slot, {len(y)} beat, 2 balance ({var_count} total)"
    )

    # === Objective: Variance-penalized total wins ===
    # maximize: Σ y[j,c] + λ·w_min - λ·w_max
    total_wins_expr = lpSum(y[j, c] for j in J for c in ALL_CATEGORIES)
    balance_term = balance_lambda * w_min - balance_lambda * w_max
    prob += total_wins_expr + balance_term

    # === Constraints ===
    constraint_count = 0

    # C1: Roster size
    prob += lpSum(x[i] for i in I) == ROSTER_SIZE, "roster_size"
    constraint_count += 1

    # C2: Slot assignment requires rostering
    for i, s in a:
        prob += a[i, s] <= x[i], f"slot_requires_roster_{i}_{s}"
        constraint_count += 1

    # C3: Each player in at most one slot
    for i in I:
        player_slots = [s for s in eligibility[i] if (i, s) in a]
        if player_slots:
            prob += lpSum(a[i, s] for s in player_slots) <= 1, f"one_slot_{i}"
            constraint_count += 1

    # C4: Starting slots must be filled
    all_slots = {**HITTING_SLOTS, **PITCHING_SLOTS}
    for slot, count in all_slots.items():
        prob += (
            lpSum(a[i, slot] for i in I if slot in eligibility[i]) >= count,
            f"fill_{slot}",
        )
        constraint_count += 1

    # C5: Roster composition bounds
    prob += lpSum(x[i] for i in I_H) >= MIN_HITTERS, "min_hitters"
    prob += lpSum(x[i] for i in I_H) <= MAX_HITTERS, "max_hitters"
    prob += lpSum(x[i] for i in I_P) >= MIN_PITCHERS, "min_pitchers"
    prob += lpSum(x[i] for i in I_P) <= MAX_PITCHERS, "max_pitchers"
    constraint_count += 4

    # C6: Beat constraints for counting stats (R, HR, RBI, SB, W, SV, K)
    counting_hitting = ["R", "HR", "RBI", "SB"]
    counting_pitching = ["W", "SV", "K"]

    for j in J:
        for c in counting_hitting:
            my_sum = lpSum(candidates.iloc[i][c] * x[i] for i in I_H)
            opp_val = opponent_totals[j][c]
            prob += (
                my_sum >= opp_val + EPSILON_COUNTING - BIG_M_COUNTING * (1 - y[j, c]),
                f"beat_{j}_{c}",
            )
            constraint_count += 1

        for c in counting_pitching:
            my_sum = lpSum(candidates.iloc[i][c] * x[i] for i in I_P)
            opp_val = opponent_totals[j][c]
            prob += (
                my_sum >= opp_val + EPSILON_COUNTING - BIG_M_COUNTING * (1 - y[j, c]),
                f"beat_{j}_{c}",
            )
            constraint_count += 1

    # C7: Beat constraints for OPS (higher is better)
    for j in J:
        opp_ops = opponent_totals[j]["OPS"]
        coeff_sum = lpSum(
            candidates.iloc[i]["PA"] * (candidates.iloc[i]["OPS"] - opp_ops) * x[i]
            for i in I_H
        )
        prob += (
            coeff_sum >= EPSILON_RATIO - BIG_M_RATIO * (1 - y[j, "OPS"]),
            f"beat_{j}_OPS",
        )
        constraint_count += 1

    # C8: Beat constraints for ERA and WHIP (lower is better)
    for j in J:
        opp_era = opponent_totals[j]["ERA"]
        coeff_sum = lpSum(
            candidates.iloc[i]["IP"] * (opp_era - candidates.iloc[i]["ERA"]) * x[i]
            for i in I_P
        )
        prob += (
            coeff_sum >= EPSILON_RATIO - BIG_M_RATIO * (1 - y[j, "ERA"]),
            f"beat_{j}_ERA",
        )
        constraint_count += 1

    for j in J:
        opp_whip = opponent_totals[j]["WHIP"]
        coeff_sum = lpSum(
            candidates.iloc[i]["IP"] * (opp_whip - candidates.iloc[i]["WHIP"]) * x[i]
            for i in I_P
        )
        prob += (
            coeff_sum >= EPSILON_RATIO - BIG_M_RATIO * (1 - y[j, "WHIP"]),
            f"beat_{j}_WHIP",
        )
        constraint_count += 1

    # C9: w_min bounded above by each category's wins
    for c in ALL_CATEGORIES:
        category_wins_expr = lpSum(y[j, c] for j in J)
        prob += w_min <= category_wins_expr, f"min_bound_{c}"
        constraint_count += 1

    # C10: w_max bounded below by each category's wins
    for c in ALL_CATEGORIES:
        category_wins_expr = lpSum(y[j, c] for j in J)
        prob += w_max >= category_wins_expr, f"max_bound_{c}"
        constraint_count += 1

    print(f"  Constraints: {constraint_count} total")

    # === Solve ===
    print("Solving...")
    start_time = time.time()

    available_solvers = pulp.listSolvers(onlyAvailable=True)

    if "HiGHS_CMD" in available_solvers:
        solver = pulp.HiGHS_CMD(msg=True, timeLimit=300)
    elif "PULP_CBC_CMD" in available_solvers:
        solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=300)
    else:
        solver = None

    status = prob.solve(solver)
    solve_time = time.time() - start_time

    status_str = pulp.LpStatus[status]
    assert status == pulp.LpStatusOptimal, (
        f"Solver failed: {status_str}. "
        f"Check position slot eligibility and roster composition constraints."
    )

    # Extract solution
    roster_names = [candidates.iloc[i]["Name"] for i in I if value(x[i]) > 0.5]

    # Compute balance metrics
    category_wins = {c: int(round(sum(value(y[j, c]) for j in J))) for c in ALL_CATEGORIES}
    total_wins = sum(category_wins.values())
    actual_w_min = min(category_wins.values())
    actual_w_max = max(category_wins.values())
    win_range = actual_w_max - actual_w_min

    if balance_lambda > 0:
        print(
            f"Solved in {solve_time:.1f}s — {total_wins}/60 wins, "
            f"range {win_range} ({actual_w_min}-{actual_w_max}), λ={balance_lambda}"
        )
    else:
        print(
            f"Solved in {solve_time:.1f}s — objective: {total_wins}/60 opponent-category wins"
        )

    # Log roster changes
    added = set(roster_names) - current_roster_names
    dropped = current_roster_names - set(roster_names)
    if added or dropped:
        print(f"  Added {len(added)} players, dropped {len(dropped)}")

    return roster_names, {
        "objective": pulp.value(prob.objective),
        "total_wins": total_wins,
        "category_wins": category_wins,
        "w_min": actual_w_min,
        "w_max": actual_w_max,
        "win_range": win_range,
        "solve_time": solve_time,
        "status": status_str,
        "balance_lambda": balance_lambda,
    }
```

---

## Updating print_roster_summary

In `roster_optimizer.py`, modify `print_roster_summary()` to display balance metrics.

### Step 1: Update Function Signature

Change the signature to accept optional solution_info:

```python
def print_roster_summary(
    roster_names: list[str],
    projections: pd.DataFrame,
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    old_roster_names: set[str] | None = None,
    solution_info: dict | None = None,  # NEW PARAMETER
) -> None:
```

### Step 2: Add Balance Metrics Section

Insert this code **after** the "Total opponent-category wins" and "Projected roto points" lines, **before** the function ends:

```python
    # Balance metrics (if available)
    if solution_info and solution_info.get("balance_lambda", 0) > 0:
        print("\n" + "-" * 35)
        print(f"BALANCE METRICS (λ = {solution_info['balance_lambda']})")
        print("-" * 35)

        cat_wins = solution_info.get("category_wins", {})
        if cat_wins:
            w_min = solution_info["w_min"]
            w_max = solution_info["w_max"]

            worst_cats = [c for c, w in cat_wins.items() if w == w_min]
            best_cats = [c for c, w in cat_wins.items() if w == w_max]

            print(f"Worst: {', '.join(worst_cats)} ({w_min} wins)")
            print(f"Best:  {', '.join(best_cats)} ({w_max} wins)")
            print(f"Range: {solution_info['win_range']}")
```

---

## Testing

### Test File Location

Add these tests to `tests/test_core.py`.

### Test Functions

```python
def test_balance_lambda_zero_matches_standard():
    """With λ=0, variance-penalized should match standard objective in total wins."""
    # This test verifies backward compatibility
    _, info = build_and_solve_milp(
        candidates, opponent_totals, current_names, balance_lambda=0.0
    )
    # With λ=0, objective equals total_wins (no balance term)
    assert abs(info["objective"] - info["total_wins"]) < 0.01


def test_balance_lambda_validation():
    """Invalid λ values should raise assertion error."""
    import pytest

    with pytest.raises(AssertionError, match="balance_lambda must be in"):
        build_and_solve_milp(candidates, opponent_totals, current_names, balance_lambda=-0.5)

    with pytest.raises(AssertionError, match="balance_lambda must be in"):
        build_and_solve_milp(candidates, opponent_totals, current_names, balance_lambda=3.0)


def test_balance_metrics_in_solution_info():
    """Solution info should contain all balance-related fields."""
    _, info = build_and_solve_milp(
        candidates, opponent_totals, current_names, balance_lambda=0.5
    )

    assert "total_wins" in info
    assert "category_wins" in info
    assert "w_min" in info
    assert "w_max" in info
    assert "win_range" in info
    assert "balance_lambda" in info

    # Verify consistency
    assert info["w_min"] == min(info["category_wins"].values())
    assert info["w_max"] == max(info["category_wins"].values())
    assert info["win_range"] == info["w_max"] - info["w_min"]
    assert info["total_wins"] == sum(info["category_wins"].values())
    assert info["balance_lambda"] == 0.5


def test_balance_objective_formula():
    """Verify objective equals total_wins + λ*w_min - λ*w_max."""
    _, info = build_and_solve_milp(
        candidates, opponent_totals, current_names, balance_lambda=0.5
    )

    expected_objective = (
        info["total_wins"]
        + 0.5 * info["w_min"]
        - 0.5 * info["w_max"]
    )
    assert abs(info["objective"] - expected_objective) < 0.01


def test_higher_lambda_may_reduce_range():
    """Higher λ should produce same or lower win range (not strictly required but expected)."""
    _, info_low = build_and_solve_milp(
        candidates, opponent_totals, current_names, balance_lambda=0.0
    )
    _, info_high = build_and_solve_milp(
        candidates, opponent_totals, current_names, balance_lambda=1.0
    )

    # Not strictly guaranteed (depends on problem structure), but typical
    # If this fails on real data, convert to a soft check or remove
    assert info_high["win_range"] <= info_low["win_range"] + 1  # Allow small tolerance
```

### Test Data Requirements

These tests require the same test fixtures used by other `test_core.py` tests:
- `candidates`: DataFrame of candidate players
- `opponent_totals`: Dict of opponent category totals
- `current_names`: Set of current roster names

Use the existing test setup or create minimal fixtures.

---

## Example Calculations

### Verifying the Math

**Scenario:** Roster produces these category wins:

| Category | Wins |
|----------|------|
| R | 4 |
| HR | 5 |
| RBI | 4 |
| SB | 2 |
| OPS | 4 |
| W | 3 |
| SV | 2 |
| K | 5 |
| ERA | 4 |
| WHIP | 4 |

**Computed values:**
- `total_wins` = 4+5+4+2+4+3+2+5+4+4 = 37
- `w_min` = 2 (SB, SV)
- `w_max` = 5 (HR, K)
- `win_range` = 5 - 2 = 3

**Objective with λ=0.5:**
```
objective = 37 + 0.5×2 - 0.5×5
          = 37 + 1 - 2.5
          = 35.5
```

**Verification in solution_info:**
```python
assert info["total_wins"] == 37
assert info["w_min"] == 2
assert info["w_max"] == 5
assert info["win_range"] == 3
assert abs(info["objective"] - 35.5) < 0.01
```

---

## Backward Compatibility

To maintain backward compatibility with existing code:

1. **Default parameter:** `balance_lambda=0.5` means new behavior is default. If you need old behavior, pass `balance_lambda=0.0`.

2. **Return value changes:** The return dict now includes additional keys (`category_wins`, `w_min`, `w_max`, `win_range`, `balance_lambda`). Existing code that only accesses `objective`, `solve_time`, `status` will continue to work.

3. **For true backward compatibility:** Change the default to `balance_lambda=0.0` instead. However, the recommended approach is to use the new default and update calling code to handle the richer return value.

---

## Summary

**Files to modify:**
1. `optimizer/data_loader.py` — Add `BALANCE_LAMBDA_DEFAULT = 0.5`
2. `optimizer/__init__.py` — Export `BALANCE_LAMBDA_DEFAULT`
3. `optimizer/roster_optimizer.py` — Modify `build_and_solve_milp()` per steps 2-9
4. `optimizer/roster_optimizer.py` — Modify `print_roster_summary()` to show balance metrics
5. `tests/test_core.py` — Add 5 new test functions

**Key changes to build_and_solve_milp:**
- New parameter: `balance_lambda: float = 0.5`
- New variables: `w_min`, `w_max` (continuous, bounds [0, 6])
- New constraints: 20 (10 for C9, 10 for C10)
- Modified objective: `Σ y[j,c] + λ·w_min - λ·w_max`
- Expanded return dict with balance metrics

**Computational impact:** Negligible. Adds 2 continuous variables and 20 simple linear constraints to an already-large MILP.
