
# Technical Specification: Probabilistic Roster Optimization & Trade Logic

## 1. Overview

This system replaces standard deterministic fantasy metrics with a probability-based optimization engine using a functional, data-oriented architecture.

**The Core Problem:**
Standard metrics (like "Points Above Replacement") assume fixed outcomes (e.g., "Player X gets 25 HR"). In reality, projections are probability distributions. The true value of a player is the *change in probability* that a team improves its rank in specific categories against specific opponents.

**The Solution:**

1. **Evaluation Metric:** **Expected Standings Points (ESP)**. A continuous function calculating the expected roto points a team will earn, given the mean and variance of their roster's performance.
2. **Search Algorithm:** **Bilateral Knapsack**. A targeted MILP (Mixed-Integer Linear Programming) solver that identifies mutually beneficial player swaps between two specific teams.

---

## 2. Metric: Expected Standings Points (ESP)

### 2.1 Concept

In a Rotisserie league, a team's score in a category is determined by how many opponents they defeat. We model every team's final category total as a **Normal Distribution** defined by a Mean (`mu`) and Variance (`var`).

### 2.2 Core Math Logic (Functional Pseudocode)

All functions below assume input data is a dictionary or struct representing the league state.

**1. Pairwise Win Probability**
The probability Team A beats Team B is the area under the curve where the difference `(A - B)` is positive.

```python
def calculate_win_probability(mu_a: float, var_a: float, mu_b: float, var_b: float) -> float:
    diff_mean = mu_a - mu_b
    diff_std = sqrt(var_a + var_b)
    
    # Use standard normal cumulative distribution function (CDF)
    # This returns probability that a value is <= z_score
    z_score = diff_mean / diff_std
    return norm.cdf(z_score) 

```

**2. Expected Category Score**
A team's expected points in a category is the sum of probabilities of beating each opponent, plus 1 (the base point).

```python
def calculate_expected_category_points(team_id: str, category: str, league_stats: dict) -> float:
    """
    league_stats: dict mapping team_id -> {category -> {mu, var}}
    """
    target_stats = league_stats[team_id][category]
    expected_points = 1.0
    
    for opponent_id, opponent_stats in league_stats.items():
        if team_id == opponent_id: continue
        
        opp_cat_stats = opponent_stats[category]
        
        prob_win = calculate_win_probability(
            target_stats['mu'], 
            target_stats['var'],
            opp_cat_stats['mu'], 
            opp_cat_stats['var']
        )
        expected_points += prob_win
        
    return expected_points

```

**3. Total Expected Standings Points (ESP)**
Sum the expected points across all scoring categories.

```python
def calculate_total_esp(team_id: str, league_stats: dict, categories: list[str]) -> float:
    total_esp = 0.0
    for cat in categories:
        total_esp += calculate_expected_category_points(team_id, cat, league_stats)
    return total_esp

```

### 2.3 Variance Calculations (Aggregation Logic)

Projections provide the Mean. We must derive the Variance based on historical reliability.

**A. Counting Stats (HR, R, RBI, SB, W, K, SV)**
Assume independence between players. Variances sum linearly.

```python
def aggregate_counting_stats(roster_projections: list[dict]) -> dict:
    # Returns {mu, var}
    return {
        'mu': sum(p['mu'] for p in roster_projections),
        'var': sum(p['var'] for p in roster_projections)
    }

```

**B. Ratio Stats (AVG, ERA, WHIP)**
Ratio stats are non-linear. Use the **Delta Method approximation**.

```python
def aggregate_ratio_stats(numerator_projections: list[dict], denominator_projections: list[dict], multiplier: float = 1.0) -> dict:
    """
    Calculates mu and var for a Ratio (Numerator / Denominator).
    """
    # 1. Sum components
    mu_n = sum(p['mu'] for p in numerator_projections)
    mu_d = sum(p['mu'] for p in denominator_projections)
    var_n = sum(p['var'] for p in numerator_projections)
    var_d = sum(p['var'] for p in denominator_projections) # Often 0 for AB/IP
    
    # 2. Taylor Series Approximation for Variance of a Ratio
    # var(N/D) ≈ (mu_n/mu_d)^2 * (var_n/mu_n^2 + var_d/mu_d^2)
    term1 = (mu_n / mu_d)**2
    term2 = (var_n / (mu_n**2)) if mu_n > 0 else 0
    term3 = (var_d / (mu_d**2)) if mu_d > 0 else 0
    
    ratio_var = term1 * (term2 + term3)
    ratio_mu = mu_n / mu_d

    return {
        'mu': multiplier * ratio_mu,
        'var': (multiplier**2) * ratio_var
    }

```

**C. Default Variances (Constants)**
If `InterSD` is unavailable, use these hardcoded standard deviations (`sigma`) to derive variance (`sigma^2`):

* **H:** 15.0
* **ER:** 10.0
* **BB (Pitcher):** 8.0
* **H (Pitcher):** 12.0
* **HR:** 7.0
* **SB:** 8.0
* **W:** 3.0
* **K:** 25.0
* **SV:** 8.0

---

## 3. Algorithm: Bilateral Knapsack Trade Solver

### 3.1 Concept

We solve a specific "negotiation" between **Me (Team A)** and **Target (Team B)**.
**Goal:** Find subsets of players to swap such that *both* teams improve their ESP.

### 3.2 Optimization Strategy (Two-Stage)

Because the `norm.cdf` function in ESP is non-linear, it is hard to optimize directly. We use a linear proxy to find candidates, then verify them with the heavy math.

#### Stage 1: The MILP Search (Candidate Generation)

**Objective:** Maximize a linear proxy (Standard SGP) for Team A.
**Variables:** Binary `x_p` for every player `p` currently on Team A or Team B.

* `x_p = 1`: Player ends up on Team A.
* `x_p = 0`: Player ends up on Team B.

**Constraints:**

1. **Roster Count:** `sum(x_p) == size(Team_A)` (Team A size stays constant).
2. **Positions:** Team A must satisfy all positional requirements (e.g., `sum(x_p for p in catchers) >= 1`).
3. **Pareto Improvement (The "Dealbreaker"):**
`ProxyValue(Team_B_New) >= ProxyValue(Team_B_Current)`
*(Team B must "win" or "break even" on the trade by the linear metric)*.
4. **Transaction Size:**
`sum(1 - x_p for p in Team_A_Original) <= 3`
*(Don't trade more than 3 players at once)*.

#### Stage 2: The ESP Verifier

The MILP returns a list of players for Team A and Team B.

1. Construct the hypothetical new rosters.
2. Run `calculate_total_esp` for the entire league using the new state.
3. **Filter:** If `ESP(Team_B_New) < ESP(Team_B_Current)`, discard the trade.
4. **Rank:** Sort valid trades by `ESP(Team_A_New) - ESP(Team_A_Current)`.

---

## 4. Implementation Guidelines

### 4.1 Data Schemas (TypedDict / Dataclass)

Strict data separation. No methods attached to data.

**Player:**

```python
{
    "id": str,
    "name": str,
    "position": str,
    "team_id": str,
    "projections": { "HR": 25.0, "SB": 10.0, ... },
    "variances": { "HR": 49.0, "SB": 64.0, ... } # derived from sigma^2
}

```

**LeagueState:**

```python
{
    "teams": {
        "team_1": [player1, player2, ...],
        "team_2": [player3, player4, ...]
    },
    "stats_cache": {
        # Calculated via aggregation functions
        "team_1": { "HR": {mu: 100, var: 50}, ... } 
    }
}

```

### 4.2 Workflow Execution

1. **Ingest:** Load players, apply projections, assign variances.
2. **Baseline Calculation:**
* Run a linear solver (MILP) to determine the optimal *Active Starting Lineup* for every team (Starter vs Bench).
* Compute `baseline_esp` for every team.


3. **Trade Search:**
* Input: `my_team_id`, `target_team_id`.
* Run `solve_bilateral_knapsack()` (MILP function).
* Output: `proposed_roster_A`, `proposed_roster_B`.
* Compute `new_esp_A`, `new_esp_B`.
* Return trade if `delta_A > 0` and `delta_B >= 0`.



### 4.3 Handling "Drops"

To simplify V1, assume trades are **N-for-N** (e.g., 2 players for 2 players).

* If the user wants to test an uneven trade (2-for-1), create a dummy "Empty Slot" player or force the inclusion of a generic "Replacement Level FA" in the set of the team receiving fewer players.