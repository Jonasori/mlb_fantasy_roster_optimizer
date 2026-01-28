# Trade Engine: Implementation Spec (Revised)

## Overview

The Trade Engine evaluates and proposes trades for a dynasty fantasy baseball league with rotisserie scoring. It integrates with the existing roster optimizer pipeline, using the same data structures and following the same code conventions.

**Core insight:** In rotisserie scoring, a player's value is *context-dependent*. Production in a category where you're safely ahead of all opponents has near-zero marginal value. Production in a category where you're narrowly trailing opponents is extremely valuable. The trade engine exploits this asymmetry to find trades that improve your expected wins while appearing fair in generic value terms.

**Key design decision:** The engine uses *two-sided marginal category values*. The value of acquiring production in a category (based on contested losses you could flip) is computed separately from the cost of losing production (based on contested wins you could lose). This asymmetry is essential for accurate trade evaluation.

**Module boundaries:**
- **Input:** Projections DataFrame, my roster names, opponent rosters, opponent totals (all from data module)
- **Output:** Trade recommendations with evaluation metrics
- **Does NOT:** Modify rosters, run the MILP, or interact with external systems

---

## Constraints on Implementation Style

**Mandatory (same as existing pipeline):**
- No classes. No OOP. Only functions and plain data structures (dicts, lists, numpy arrays, pandas DataFrames).
- Fail fast: use `assert` liberally. If something is wrong, crash immediately with a clear message. Never write try/except for error handling. Never write fallback logic.
- Use `print()` for status updates.
- All configuration is passed as function arguments. No global state.

**File structure:**
```
mlb_fantasy_roster_optimizer/
├── optimizer/
│   ├── __init__.py
│   ├── data_loader.py        # Data preparation module (future refactor)
│   ├── roster_optimizer.py   # MILP module (existing, to be refactored)
│   ├── trade_engine.py       # THIS SPEC
│   └── visualizations.py     # Plotting functions
├── data/
│   └── ...
├── notebook.py
└── pyproject.toml
```

---

## Configuration Constants

Add to the top of `trade_engine.py`:

```python
# === TRADE ENGINE CONFIGURATION ===

# Import shared constants from roster_optimizer
from .roster_optimizer import (
    HITTING_CATEGORIES,
    PITCHING_CATEGORIES,
    ALL_CATEGORIES,
    NEGATIVE_CATEGORIES,
    RATIO_STATS,
    MIN_HITTERS,
    MAX_HITTERS,
    MIN_PITCHERS,
    MAX_PITCHERS,
)

# Gap classification thresholds
# These define what counts as a "close" race vs a "safe" lead/deficit
#
# CALIBRATION GUIDANCE:
#   1. Compute std of each category across all 7 teams (you + 6 opponents)
#   2. Set threshold ≈ 0.5 to 1.0 standard deviations
#   3. Alternatively, set threshold ≈ 5-10% of the category mean
#
# Current values assume a 7-team dynasty league with typical production.
# For shallower leagues, increase thresholds (more variance).
# For deeper leagues, decrease thresholds (tighter races).
#
# VALIDATION: Run classify_races and check that roughly 30-50% of races
# are "contested" (either contested_win or contested_loss). If too few
# are contested, thresholds are too tight; if too many, too loose.

GAP_THRESHOLDS = {
    # Counting stats: absolute thresholds
    'R': 25,      # 25 runs ≈ 3% of typical team total
    'HR': 12,     # 12 HR
    'RBI': 25,    # 25 RBI
    'SB': 10,     # 10 SB (more volatile category)
    'W': 5,       # 5 wins
    'SV': 8,      # 8 saves
    'K': 50,      # 50 strikeouts
    # Ratio stats: absolute difference thresholds
    'OPS': 0.010,  # .010 OPS points
    'ERA': 0.20,   # 0.20 ERA
    'WHIP': 0.05,  # 0.05 WHIP
}

# Trade fairness threshold (absolute)
# A trade is "fair" if |Δ_generic| <= FAIRNESS_THRESHOLD
# Generic value is sum of z-scores, so this is in standard deviation units.
# 2.0 means the total z-score difference must be within 2 standard deviations.
FAIRNESS_THRESHOLD = 2.0

# Maximum players per side in a trade
MAX_TRADE_SIZE = 3

# Minimum standard deviation for z-score calculation
# If std < MIN_STD, treat as zero variance (z-score = 0)
MIN_STD = 0.001
```

---

## Core Data Structures

The trade engine uses these intermediate data structures (all plain dicts/DataFrames):

### Gap Matrix

```python
# Dict[str, Dict[int, float]]
# gap_matrix[category][opponent_id] = my_total - opp_total (normalized so positive = winning)
gap_matrix = {
    'R':   {1: 45.0, 2: -12.0, 3: 30.0, 4: -5.0, 5: 88.0, 6: 15.0},
    'HR':  {1: 10.0, 2: -25.0, ...},
    'ERA': {1: 0.15, 2: -0.08, ...},  # Positive means MY ERA is LOWER (winning)
    ...
}
```

### Race Classification

```python
# Dict[str, Dict[int, str]]
# race_class[category][opponent_id] = one of {'safe_win', 'contested_win', 'contested_loss', 'lost_cause'}
race_class = {
    'R':   {1: 'safe_win', 2: 'contested_loss', 3: 'safe_win', ...},
    'HR':  {1: 'contested_win', 2: 'lost_cause', ...},
    ...
}
```

### Two-Sided Marginal Category Value (MCV)

```python
# Tuple of two Dict[str, float]
# mcv_acquire[category] = marginal value of GAINING production (to flip contested losses)
# mcv_protect[category] = marginal cost of LOSING production (risking contested wins)

mcv_acquire = {
    'R': 2.0,    # 2 contested losses in R → gaining R is valuable
    'HR': 1.0,
    'SB': 4.0,   # 4 contested losses → high value to acquire SB
    ...
}

mcv_protect = {
    'R': 1.0,    # 1 contested win in R → losing R is costly
    'HR': 3.0,   # 3 contested wins → high cost to lose HR
    'SB': 0.0,   # 0 contested wins → losing SB doesn't hurt
    ...
}
```

### Player Values

```python
# DataFrame with columns:
#   Name, player_type, Position,
#   pcv_acquire (value if I acquire this player),
#   pcv_protect (cost if I lose this player),
#   generic_value (context-free value),
#   contrib_z_R, contrib_z_HR, ... (z-scored contributions)

player_values = pd.DataFrame({
    'Name': ['Trea Turner-H', 'Zack Wheeler-P', ...],
    'player_type': ['hitter', 'pitcher', ...],
    'Position': ['SS', 'SP', ...],
    'pcv_acquire': [12.5, 8.3, ...],
    'pcv_protect': [8.2, 7.1, ...],
    'generic_value': [10.1, 7.8, ...],
    ...
})
```

---

## Mathematical Formulation

### Gap Computation

For counting stats (R, HR, RBI, SB, W, SV, K):
```
gap[c, j] = my_total[c] - opp_total[j][c]
```

For negative ratio stats (ERA, WHIP) where lower is better:
```
gap[c, j] = opp_total[j][c] - my_total[c]
```
This normalization ensures **positive gap = I'm winning** for all categories.

For OPS (ratio stat, higher is better):
```
gap['OPS', j] = my_total['OPS'] - opp_total[j]['OPS']
```

### Race Classification

```
if gap[c, j] > +threshold[c]:
    race_class[c, j] = 'safe_win'
elif gap[c, j] > 0:
    race_class[c, j] = 'contested_win'
elif gap[c, j] > -threshold[c]:
    race_class[c, j] = 'contested_loss'
else:
    race_class[c, j] = 'lost_cause'
```

### Two-Sided Marginal Category Value (MCV)

```
mcv_acquire[c] = count of j where race_class[c, j] == 'contested_loss'
mcv_protect[c] = count of j where race_class[c, j] == 'contested_win'
```

**Intuition:**
- `mcv_acquire[c]` = number of matchups I could flip from loss to win by gaining production in c
- `mcv_protect[c]` = number of matchups I could flip from win to loss by losing production in c

These are separate because the trade decision is asymmetric: I'm acquiring different players than I'm sending.

### Player Contribution by Category

**Counting stats** (R, HR, RBI, SB, W, SV, K):
```
contrib[p, c] = projection[p, c]
```

**OPS** (ratio stat, higher is better):
For linearization, we compute the marginal impact on team OPS relative to average opponent:
```
avg_opp_OPS = mean(opp_total[j]['OPS'] for j in opponents)
contrib[p, 'OPS'] = PA[p] * (OPS[p] - avg_opp_OPS)
```

**ERA** (ratio stat, lower is better):
```
avg_opp_ERA = mean(opp_total[j]['ERA'] for j in opponents)
contrib[p, 'ERA'] = IP[p] * (avg_opp_ERA - ERA[p])
```
Note the sign flip: a pitcher with ERA below opponent average has **positive** contribution.

**WHIP** (ratio stat, lower is better):
```
avg_opp_WHIP = mean(opp_total[j]['WHIP'] for j in opponents)
contrib[p, 'WHIP'] = IP[p] * (avg_opp_WHIP - WHIP[p])
```

**Note on units:** Counting stat contributions are in natural units (runs, HRs, etc.). Ratio stat contributions are in weighted-difference units (PA × OPS_diff, IP × ERA_diff). These are NOT directly comparable—a contribution of 50 in R is not equivalent to 50 in OPS. The z-score normalization step converts all contributions to standardized units, making them comparable across categories.

### Z-Score Normalization (Within Player Type)

**Critical:** Z-scores must be computed *within* player type, not across all players. A hitter has zero contribution to pitching categories, and vice versa.

For hitters:
```
For each c in HITTING_CATEGORIES:
    mean_c = mean(contrib[p, c] for all hitters p)
    std_c = std(contrib[p, c] for all hitters p)
    if std_c < MIN_STD:
        contrib_z[p, c] = 0 for all hitters p
    else:
        contrib_z[p, c] = (contrib[p, c] - mean_c) / std_c

For each c in PITCHING_CATEGORIES:
    contrib_z[p, c] = 0 for all hitters p
```

For pitchers:
```
For each c in PITCHING_CATEGORIES:
    mean_c = mean(contrib[p, c] for all pitchers p)
    std_c = std(contrib[p, c] for all pitchers p)
    if std_c < MIN_STD:
        contrib_z[p, c] = 0 for all pitchers p
    else:
        contrib_z[p, c] = (contrib[p, c] - mean_c) / std_c

For each c in HITTING_CATEGORIES:
    contrib_z[p, c] = 0 for all pitchers p
```

### Player Contextual Value (Two-Sided PCV)

```
pcv_acquire[p] = sum over c of (contrib_z[p, c] * mcv_acquire[c])
pcv_protect[p] = sum over c of (contrib_z[p, c] * mcv_protect[c])
```

- `pcv_acquire[p]` = how much value I gain by adding player p to my roster
- `pcv_protect[p]` = how much value I lose by removing player p from my roster

### Generic Value (for Fairness Comparison)

Generic value ignores roster context—it's what a "trade value chart" would show:

```
generic_value[p] = sum over relevant c of contrib_z[p, c]
```

Where "relevant c" means hitting categories for hitters, pitching categories for pitchers.

### Trade Evaluation

For a trade where I send players S and receive players R:

```
Δ_contextual = sum(pcv_acquire[p] for p in R) - sum(pcv_protect[p] for p in S)
Δ_generic = sum(generic_value[p] for p in R) - sum(generic_value[p] for p in S)
```

**Good trade criteria:**
1. `Δ_contextual > 0` — improves my expected wins
2. `|Δ_generic| <= FAIRNESS_THRESHOLD` — approximately fair (opponent likely to accept)

---

## Function Specifications

### Gap and Classification Functions

```python
def compute_gap_matrix(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
) -> dict[str, dict[int, float]]:
    """
    Compute the gap between my team and each opponent in each category.
    
    Args:
        my_totals: Dict mapping category to my team's total.
                   Example: {'R': 823, 'HR': 245, ..., 'ERA': 3.85, ...}
        opponent_totals: Dict mapping opponent_id to their category totals.
                         Example: {1: {'R': 800, ...}, 2: {...}, ...}
    
    Returns:
        Dict[category, Dict[opponent_id, gap]].
        Gap is normalized so POSITIVE = I'm winning.
        
        For counting stats and OPS: gap = my_total - opp_total
        For ERA and WHIP: gap = opp_total - my_total (lower is better, so flip)
    
    Example output:
        {
            'R': {1: 23.0, 2: -15.0, 3: 45.0, 4: 8.0, 5: -30.0, 6: 12.0},
            'ERA': {1: 0.12, 2: -0.05, ...},  # positive = my ERA is lower
            ...
        }
    
    Assertions:
        - my_totals has all 10 categories in ALL_CATEGORIES
        - opponent_totals has exactly 6 opponents (ids 1-6)
        - Each opponent has all 10 categories
    
    Print:
        "Computed gap matrix for 10 categories × 6 opponents"
    """
```

```python
def classify_races(
    gap_matrix: dict[str, dict[int, float]],
    thresholds: dict[str, float] = GAP_THRESHOLDS,
) -> dict[str, dict[int, str]]:
    """
    Classify each (category, opponent) race into one of four buckets.
    
    Args:
        gap_matrix: Output of compute_gap_matrix()
        thresholds: Dict mapping category to threshold value.
                    Default: GAP_THRESHOLDS from module constants.
    
    Returns:
        Dict[category, Dict[opponent_id, classification]]
        Classification is one of:
            'safe_win' — gap > +threshold (comfortable lead)
            'contested_win' — 0 < gap <= +threshold (narrow lead, must protect)
            'contested_loss' — -threshold <= gap < 0 (narrow deficit, can catch up)
            'lost_cause' — gap < -threshold (too far behind)
    
    Example output:
        {
            'R': {1: 'contested_win', 2: 'contested_loss', 3: 'safe_win', ...},
            ...
        }
    
    Assertions:
        - gap_matrix has all 10 categories
        - thresholds has all 10 categories
        - All threshold values are positive
    
    Print:
        Summary table showing count of each classification per category:
        "Race classification summary:"
        "Category  SafeWin  ContestedWin  ContestedLoss  LostCause"
        "R         2        1             2              1"
        ...
        "Total contested races: {N}/60 ({N/60:.0%})"
    """
```

```python
def compute_mcv(
    race_classification: dict[str, dict[int, str]],
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Compute two-sided Marginal Category Value for each category.
    
    mcv_acquire = value of gaining production (based on contested losses)
    mcv_protect = cost of losing production (based on contested wins)
    
    Formula:
        mcv_acquire[c] = count of j where race_class[c, j] == 'contested_loss'
        mcv_protect[c] = count of j where race_class[c, j] == 'contested_win'
    
    Args:
        race_classification: Output of classify_races()
    
    Returns:
        Tuple of (mcv_acquire, mcv_protect)
        Each is a dict mapping category to its MCV.
        Range for each: 0 (no contested races) to 6 (all 6 opponents contested)
    
    Example output:
        mcv_acquire = {'R': 2, 'HR': 1, 'SB': 4, 'ERA': 3, ...}
        mcv_protect = {'R': 1, 'HR': 3, 'SB': 0, 'ERA': 1, ...}
    
    Print:
        "Category priorities:"
        "  Acquire (contested losses): SB=4, ERA=3, R=2, ..."
        "  Protect (contested wins):   HR=3, OPS=2, R=1, ..."
        (each sorted descending by MCV)
    """
```

### Player Valuation Functions

```python
def compute_player_contributions(
    player_names: set[str],
    projections: pd.DataFrame,
    opponent_totals: dict[int, dict[str, float]],
) -> pd.DataFrame:
    """
    Compute each player's raw contribution to each category.
    
    Args:
        player_names: Set of player names to evaluate.
                      Typically: my_roster | all_opponent_roster_names
        projections: Combined projections DataFrame with all stats.
        opponent_totals: Opponent totals (used for ratio stat baselines).
    
    Returns:
        DataFrame with columns:
            Name, player_type, Position,
            contrib_R, contrib_HR, contrib_RBI, contrib_SB, contrib_OPS,
            contrib_W, contrib_SV, contrib_K, contrib_ERA, contrib_WHIP
        
        For counting stats: contrib = raw stat value
        For ratio stats: contrib = weighted marginal impact
        
        Hitters have contrib = 0 for all pitching categories.
        Pitchers have contrib = 0 for all hitting categories.
    
    Implementation:
        1. Filter projections to player_names
        2. Compute average opponent values for ratio stats:
           avg_opp_OPS = mean of opponent OPS values
           avg_opp_ERA = mean of opponent ERA values
           avg_opp_WHIP = mean of opponent WHIP values
        3. For each hitter:
           - Counting stats: contrib = stat value
           - OPS: contrib = PA * (OPS - avg_opp_OPS)
           - Pitching categories: contrib = 0
        4. For each pitcher:
           - Counting stats: contrib = stat value
           - ERA: contrib = IP * (avg_opp_ERA - ERA)
           - WHIP: contrib = IP * (avg_opp_WHIP - WHIP)
           - Hitting categories: contrib = 0
        5. Return as DataFrame
    
    Assertions:
        - All player_names found in projections (crash with list of missing names)
        - No null values in contribution columns
    
    Print:
        "Computed contributions for {N} players ({H} hitters, {P} pitchers)"
    """
```

```python
def compute_player_values(
    contributions: pd.DataFrame,
    mcv_acquire: dict[str, float],
    mcv_protect: dict[str, float],
) -> pd.DataFrame:
    """
    Compute contextual (PCV) and generic value for each player.
    
    Args:
        contributions: Output of compute_player_contributions()
        mcv_acquire: Marginal value of gaining production (from compute_mcv)
        mcv_protect: Marginal cost of losing production (from compute_mcv)
    
    Returns:
        DataFrame with columns:
            Name, player_type, Position,
            pcv_acquire (value if acquired),
            pcv_protect (cost if lost),
            generic_value (context-free value),
            contrib_z_R, contrib_z_HR, ..., contrib_z_WHIP (z-scored contributions)
        
        Sorted by pcv_acquire descending.
    
    Implementation:
        1. Separate hitters and pitchers
        
        2. For hitters, z-score normalize hitting categories within hitters:
           For each c in HITTING_CATEGORIES:
               std_c = contributions[hitters]['contrib_' + c].std()
               if std_c < MIN_STD:
                   contrib_z[c] = 0 for all hitters
               else:
                   contrib_z[c] = (contrib[c] - mean) / std
           For pitching categories: contrib_z = 0 for all hitters
        
        3. For pitchers, z-score normalize pitching categories within pitchers:
           For each c in PITCHING_CATEGORIES:
               std_c = contributions[pitchers]['contrib_' + c].std()
               if std_c < MIN_STD:
                   contrib_z[c] = 0 for all pitchers
               else:
                   contrib_z[c] = (contrib[c] - mean) / std
           For hitting categories: contrib_z = 0 for all pitchers
        
        4. Recombine hitters and pitchers
        
        5. Compute generic_value = sum of z-scores across relevant categories
           (hitting categories for hitters, pitching for pitchers)
        
        6. Compute pcv_acquire = sum of (contrib_z[c] * mcv_acquire[c]) for all c
        
        7. Compute pcv_protect = sum of (contrib_z[c] * mcv_protect[c]) for all c
        
        8. Sort by pcv_acquire descending
    
    Assertions:
        - mcv_acquire and mcv_protect both have all 10 categories
        - No null values in output
    
    Print:
        "Player values computed:"
        "  Top 5 by acquire value: {name1} ({pcv1:.1f}), {name2} ({pcv2:.1f}), ..."
        "  Top 5 by protect value: {name1} ({pcv1:.1f}), {name2} ({pcv2:.1f}), ..."
        "  Top 5 by generic value: {name1} ({gv1:.1f}), {name2} ({gv2:.1f}), ..."
    """
```

### Trade Candidate Identification

```python
def identify_trade_targets(
    player_values: pd.DataFrame,
    my_roster_names: set[str],
    opponent_rosters: dict[int, set[str]],
    mcv_acquire: dict[str, float],
    n_targets: int = 10,
) -> pd.DataFrame:
    """
    Identify players to TARGET (acquire) in trades.
    
    Targets are players on opponent rosters who have high pcv_acquire.
    
    Args:
        player_values: Output of compute_player_values()
        my_roster_names: Set of player names on my roster
        opponent_rosters: Dict mapping opponent_id to set of their player names
        mcv_acquire: Marginal value of acquiring (to identify primary value category)
        n_targets: Number of targets to return per player type (hitters/pitchers)
    
    Returns:
        DataFrame of trade targets with columns:
            Name, player_type, Position, pcv_acquire, pcv_protect, generic_value,
            primary_value_category (which high-MCV category they help most),
            owner_id (int, which opponent owns this player)
        
        Sorted by pcv_acquire descending.
    
    Implementation:
        1. Build set of all opponent player names: 
           opponent_roster_names = union of all opponent_rosters.values()
        2. Build mapping from player name to owner:
           owner_map = {name: opp_id for opp_id, names in opponent_rosters.items() for name in names}
        3. Filter player_values to players IN opponent_roster_names
        4. For each player, find their "primary value category":
           - The category c where (contrib_z[c] * mcv_acquire[c]) is highest
        5. Add owner_id column using owner_map
        6. Take top n_targets hitters and top n_targets pitchers by pcv_acquire
        7. Return combined DataFrame sorted by pcv_acquire descending
    
    Assertions:
        - All players in opponent_rosters exist in player_values
    
    Print:
        "Trade targets identified: {N} hitters, {M} pitchers"
        "  Top hitter: {name} from opponent {owner_id} (helps {category})"
        "  Top pitcher: {name} from opponent {owner_id} (helps {category})"
    """
```

```python
def identify_trade_pieces(
    player_values: pd.DataFrame,
    my_roster_names: set[str],
    mcv_acquire: dict[str, float],
    mcv_protect: dict[str, float],
    n_pieces: int = 10,
) -> pd.DataFrame:
    """
    Identify players to OFFER (trade away) from my roster.
    
    Trade pieces are players with high generic value but low pcv_protect
    (valuable to others but not critical to me).
    
    Args:
        player_values: Output of compute_player_values()
        my_roster_names: Set of player names on my roster
        mcv_acquire: For identifying expendable categories
        mcv_protect: For computing expendability
        n_pieces: Number of trade pieces to identify
    
    Returns:
        DataFrame of tradeable players with columns:
            Name, player_type, Position, pcv_acquire, pcv_protect, generic_value,
            expendability_score (generic_value - pcv_protect),
            expendable_category (which low-MCV category their value comes from)
        
        Sorted by expendability_score descending.
        High expendability = valuable generically but not critical to protect.
    
    Implementation:
        1. Filter player_values to players IN my_roster_names
        2. Compute expendability_score = generic_value - pcv_protect
           High score = player has trade value but losing them doesn't hurt much
        3. For each player, find their "expendable category":
           - The category c where contrib_z[c] is highest AND mcv_protect[c] is below median
           - If no such category, use the category with highest contrib_z
        4. Take top n_pieces by expendability_score
        5. Return DataFrame
    
    Print:
        "Trade pieces identified: {N} players"
        "  Most expendable: {name} (excess {category}, expendability={score:.1f})"
    """
```

### Trade Evaluation Functions

```python
def evaluate_trade(
    send_players: list[str],
    receive_players: list[str],
    player_values: pd.DataFrame,
) -> dict:
    """
    Evaluate a specific trade proposal.
    
    Args:
        send_players: List of player names I would send
        receive_players: List of player names I would receive
        player_values: Output of compute_player_values()
    
    Returns:
        Dict with keys:
            'send_players': list of names
            'receive_players': list of names
            'delta_contextual': float (positive = good for me)
                Computed as: sum(pcv_acquire for received) - sum(pcv_protect for sent)
            'delta_generic': float (positive = I'm getting more generic value)
            'is_fair': bool (|delta_generic| <= FAIRNESS_THRESHOLD)
            'is_good_for_me': bool (delta_contextual > 0)
            'recommendation': str ('ACCEPT', 'REJECT', 'UNFAIR')
            'send_total_pcv_protect': float
            'send_total_generic': float
            'receive_total_pcv_acquire': float
            'receive_total_generic': float
            'send_details': DataFrame of sent players with their values
            'receive_details': DataFrame of received players with their values
    
    Implementation:
        1. Look up pcv_acquire, pcv_protect, generic_value for all players
        2. For sent players, sum pcv_protect (what I lose) and generic_value
        3. For received players, sum pcv_acquire (what I gain) and generic_value
        4. delta_contextual = receive_total_pcv_acquire - send_total_pcv_protect
        5. delta_generic = receive_total_generic - send_total_generic
        6. is_fair = abs(delta_generic) <= FAIRNESS_THRESHOLD
        7. is_good_for_me = delta_contextual > 0
        8. Determine recommendation:
           - If not is_fair and delta_generic > 0: 'UNFAIR' (I'm overpaying)
           - If not is_fair and delta_generic < 0: 'STEAL' (they won't accept)
           - If is_good_for_me: 'ACCEPT'
           - Else: 'REJECT'
    
    Assertions:
        - All send_players found in player_values
        - All receive_players found in player_values
        - len(send_players) >= 1 and len(receive_players) >= 1
        - len(send_players) <= MAX_TRADE_SIZE and len(receive_players) <= MAX_TRADE_SIZE
    
    Print:
        "Trade evaluation:"
        "  Send: {player1}, {player2} (protect_value={X:.1f}, generic={Y:.1f})"
        "  Receive: {player1} (acquire_value={X:.1f}, generic={Y:.1f})"
        "  Δ Contextual: {+Z:.1f} | Δ Generic: {+W:.1f}"
        "  Fair: {Yes/No} | Good for me: {Yes/No}"
        "  Recommendation: {ACCEPT/REJECT/UNFAIR/STEAL}"
    """
```

```python
def generate_trade_candidates(
    my_roster_names: set[str],
    player_values: pd.DataFrame,
    mcv_acquire: dict[str, float],
    mcv_protect: dict[str, float],
    opponent_rosters: dict[int, set[str]],
    max_send: int = 2,
    max_receive: int = 2,
    n_targets: int = 10,
    n_pieces: int = 10,
    n_candidates: int = 20,
) -> list[dict]:
    """
    Generate candidate trades to consider.
    
    Uses a combinatorial approach to find trades that maximize contextual value
    gain while maintaining fairness.
    
    Args:
        my_roster_names: Set of player names on my roster
        player_values: Output of compute_player_values()
        mcv_acquire: Marginal value of acquiring
        mcv_protect: Marginal cost of losing
        opponent_rosters: Dict mapping opponent_id to their player names
        max_send: Maximum players to send in a trade (1 to MAX_TRADE_SIZE)
        max_receive: Maximum players to receive in a trade (1 to MAX_TRADE_SIZE)
        n_targets: Number of trade targets to consider
        n_pieces: Number of trade pieces to consider
        n_candidates: Number of trade candidates to return
    
    Returns:
        List of trade evaluation dicts (output of evaluate_trade),
        sorted by delta_contextual descending.
        Only includes trades where is_fair=True and is_good_for_me=True.
        Returns empty list if no favorable fair trades found.
    
    Implementation:
        1. Call identify_trade_targets to get top n_targets
        2. Call identify_trade_pieces to get top n_pieces
        3. Generate all combinations:
           - All 1-for-1 trades
           - All 2-for-1 trades (if max_send >= 2 or max_receive >= 2)
           - All 1-for-2 trades (if max_receive >= 2)
           - All 2-for-2 trades (if both max >= 2)
        4. Use tqdm progress bar over all combinations
        5. Evaluate each trade using evaluate_trade
        6. Filter to trades where is_fair=True and is_good_for_me=True
        7. Sort by delta_contextual descending
        8. Return top n_candidates
    
    Assertions:
        - max_send >= 1 and max_send <= MAX_TRADE_SIZE
        - max_receive >= 1 and max_receive <= MAX_TRADE_SIZE
    
    Print:
        "Generating trade candidates..."
        "  Targets: {N} | Pieces: {M}"
        "  Evaluating {K} possible trades..."
        [tqdm progress bar]
        "  Found {F} favorable fair trades"
        If F == 0: "  No favorable fair trades found. Consider adjusting thresholds or expanding player pools."
    """
```

### Verification Function

```python
def verify_trade_impact(
    send_players: list[str],
    receive_players: list[str],
    my_roster_names: set[str],
    projections: pd.DataFrame,
    opponent_totals: dict[int, dict[str, float]],
) -> dict:
    """
    Verify a trade's impact by computing actual win changes.
    
    This is the ground truth check—it computes team totals before and after
    the trade and counts how many opponent-category matchups change.
    
    Args:
        send_players: List of player names I would send
        receive_players: List of player names I would receive
        my_roster_names: Current roster (set of names)
        projections: Combined projections DataFrame
        opponent_totals: Opponent totals dict
    
    Returns:
        Dict with keys:
            'old_wins': int (current opponent-category wins, max 60)
            'new_wins': int (post-trade opponent-category wins)
            'delta_wins': int (new - old)
            'old_totals': dict of my current category totals
            'new_totals': dict of my post-trade category totals
            'category_changes': dict mapping category to change in that category's total
            'flipped_matchups': list of dicts with keys:
                'opponent_id', 'category', 'direction' ('gained' or 'lost')
    
    Implementation:
        1. Import compute_team_totals from roster_optimizer
        2. Compute old_totals = compute_team_totals(my_roster_names, projections)
        3. Compute new_roster = (my_roster_names - set(send_players)) | set(receive_players)
        4. Validate roster composition:
           - Count hitters: sum(1 for name in new_roster if projections[name].player_type == 'hitter')
           - Count pitchers: similarly
           - Assert MIN_HITTERS <= hitter_count <= MAX_HITTERS
           - Assert MIN_PITCHERS <= pitcher_count <= MAX_PITCHERS
        5. Compute new_totals = compute_team_totals(new_roster, projections)
        6. For each (opponent, category):
           - Determine old_win: did I beat them before? (handle negative categories)
           - Determine new_win: do I beat them after?
           - If changed, add to flipped_matchups
        7. Count old_wins, new_wins
        8. Compute category_changes = {c: new_totals[c] - old_totals[c] for c in ALL_CATEGORIES}
    
    Assertions:
        - All send_players are in my_roster_names
        - All receive_players are in projections
        - No receive_players are already in my_roster_names
        - Post-trade roster meets composition bounds (see step 4)
    
    Print:
        "Trade verification:"
        "  Roster: {old_size} → {new_size} players"
        "  Current wins: {old}/60 | Post-trade wins: {new}/60 | Net: {+/-delta}"
        "  Category changes:"
        "    R: {old:.0f} → {new:.0f} ({+/-diff:+.0f})"
        "    HR: ..."
        "  Flipped matchups ({N} total):"
        "    {category} vs opponent {id}: {gained/lost}"
        ...
    """
```

### Convenience/Summary Functions

```python
def compute_roster_situation(
    my_roster_names: set[str],
    projections: pd.DataFrame,
    opponent_totals: dict[int, dict[str, float]],
) -> dict:
    """
    Compute full roster situation analysis in one call.
    
    This is the main entry point for trade analysis. It computes all
    intermediate data structures and returns a summary.
    
    Args:
        my_roster_names: Set of player names on my roster
        projections: Combined projections DataFrame
        opponent_totals: Dict mapping opponent_id to their category totals
    
    Returns:
        Dict with keys:
            'my_totals': dict of my category totals
            'gap_matrix': output of compute_gap_matrix
            'race_classification': output of classify_races
            'mcv_acquire': first element of compute_mcv output
            'mcv_protect': second element of compute_mcv output
            'category_priorities_acquire': list of (category, mcv) sorted descending
            'category_priorities_protect': list of (category, mcv) sorted descending
            'strengths': list of categories where I have >= 4 safe wins
            'weaknesses': list of categories where I have >= 3 contested losses
            'current_wins': int (total opponent-category wins)
            'projected_roto_points': int (sum of 8-rank across categories)
    
    Implementation:
        1. Import compute_team_totals from roster_optimizer
        2. my_totals = compute_team_totals(my_roster_names, projections)
        3. gap_matrix = compute_gap_matrix(my_totals, opponent_totals)
        4. race_classification = classify_races(gap_matrix)
        5. mcv_acquire, mcv_protect = compute_mcv(race_classification)
        6. Count current_wins by iterating over race_classification
           (safe_win and contested_win both count as wins)
        7. Compute projected_roto_points:
           For each category, rank my_totals[c] among all 7 teams
           Sum (8 - rank) across all categories
        8. Identify strengths (categories with >= 4 safe_win)
        9. Identify weaknesses (categories with >= 3 contested_loss)
    
    Print:
        Formatted summary (see print_trade_report for format)
    """
```

```python
def print_trade_report(
    situation: dict,
    trade_candidates: list[dict],
    player_values: pd.DataFrame,
    top_n: int = 5,
) -> None:
    """
    Print a formatted trade recommendation report.
    
    Args:
        situation: Output of compute_roster_situation
        trade_candidates: Output of generate_trade_candidates
        player_values: Output of compute_player_values
        top_n: Number of top trades to show in detail
    
    Output format:
    
    ═══════════════════════════════════════════════════════════════════
    ROSTER SITUATION
    ═══════════════════════════════════════════════════════════════════
    Current wins: 42/60 | Projected roto points: 58/70
    
    CATEGORY PRIORITIES:
    
    To Acquire (contested losses → flip to wins):
      HIGH:   SB (4), ERA (3)
      MEDIUM: RBI (2), R (2), WHIP (2)
      LOW:    HR (1), K (1), OPS (1), W (0), SV (0)
    
    To Protect (contested wins → could lose):
      HIGH:   HR (3), OPS (2)
      MEDIUM: R (1), W (1), K (1)
      LOW:    SB (0), RBI (0), ERA (0), WHIP (0), SV (0)
    
    STRENGTHS (safe in 4+ matchups): W, SV
    WEAKNESSES (trailing in 3+ matchups): SB, ERA
    
    ═══════════════════════════════════════════════════════════════════
    TOP TRADE RECOMMENDATIONS
    ═══════════════════════════════════════════════════════════════════
    
    #1: HIGHLY RECOMMENDED
        Send:    Gunnar Henderson (protect=5.2, generic=8.1)
        Receive: Trea Turner [from opponent 3] (acquire=8.4, generic=7.8)
        ────────────────────────────────────────────────────────
        Δ Contextual: +3.2 | Δ Generic: -0.3 (Fair)
        You gain: SB production, R production
        You lose: HR production
        Recommendation: ACCEPT
    
    #2: ...
    
    ═══════════════════════════════════════════════════════════════════
    
    Note: Player names are displayed WITHOUT -H/-P suffix for readability.
    
    If trade_candidates is empty:
        "No favorable fair trades found."
        "Consider:"
        "  - Adjusting GAP_THRESHOLDS (currently may be too tight/loose)"
        "  - Expanding trade piece pool (increase n_pieces)"
        "  - Looking at 2-for-1 or 2-for-2 trades"
    """
```

---

## Display Name Handling

The following functions strip the `-H/-P` suffix from player names for cleaner output:
- `print_trade_report` — all player names in formatted output
- `identify_trade_targets` — when printing top targets
- `identify_trade_pieces` — when printing most expendable
- `evaluate_trade` — when printing trade details
- `verify_trade_impact` — when printing roster changes

All other functions preserve the suffix for internal consistency:
- `compute_player_contributions` — DataFrame uses full names
- `compute_player_values` — DataFrame uses full names
- All data structures (gap_matrix, player_values, etc.)

Helper function:

```python
def _strip_suffix(name: str) -> str:
    """Strip -H or -P suffix from player name for display."""
    if name.endswith('-H') or name.endswith('-P'):
        return name[:-2]
    return name
```

---

## Integration with Existing Pipeline

The trade engine imports these from `roster_optimizer.py`:

```python
from .roster_optimizer import (
    compute_team_totals,      # Compute category totals for a roster
    HITTING_CATEGORIES,
    PITCHING_CATEGORIES,
    ALL_CATEGORIES,
    NEGATIVE_CATEGORIES,
    RATIO_STATS,
    MIN_HITTERS,
    MAX_HITTERS,
    MIN_PITCHERS,
    MAX_PITCHERS,
)
```

The trade engine does NOT import:
- MILP functions (build_and_solve_milp, etc.)
- Candidate filtering functions
- Visualization functions

---

## Example Notebook Usage

```python
# Cell 1: Imports
from optimizer.roster_optimizer import (
    load_all_data,
    compute_team_totals,
    compute_all_opponent_totals,
)
from optimizer.trade_engine import (
    compute_roster_situation,
    compute_player_contributions,
    compute_player_values,
    identify_trade_targets,
    identify_trade_pieces,
    generate_trade_candidates,
    evaluate_trade,
    verify_trade_impact,
    print_trade_report,
)

# Cell 2: Load data (same as existing pipeline)
projections, my_roster_names, opponent_rosters = load_all_data(
    HITTER_PROJ_PATH,
    PITCHER_PROJ_PATH,
    MY_ROSTER_PATH,
    OPPONENT_ROSTERS_PATH,
    DB_PATH,
)
opponent_totals = compute_all_opponent_totals(opponent_rosters, projections)

# Cell 3: Analyze roster situation
situation = compute_roster_situation(my_roster_names, projections, opponent_totals)

# Cell 4: Compute player values for trade analysis
# Include my roster + all opponent rosters (potential trade partners)
all_roster_names = my_roster_names | set().union(*opponent_rosters.values())
contributions = compute_player_contributions(all_roster_names, projections, opponent_totals)
player_values = compute_player_values(
    contributions, 
    situation['mcv_acquire'], 
    situation['mcv_protect']
)

# Cell 5: Identify trade targets and pieces
targets = identify_trade_targets(
    player_values, 
    my_roster_names, 
    opponent_rosters,
    situation['mcv_acquire']
)
pieces = identify_trade_pieces(
    player_values, 
    my_roster_names, 
    situation['mcv_acquire'],
    situation['mcv_protect']
)

# Cell 6: Generate trade candidates
candidates = generate_trade_candidates(
    my_roster_names=my_roster_names,
    player_values=player_values,
    mcv_acquire=situation['mcv_acquire'],
    mcv_protect=situation['mcv_protect'],
    opponent_rosters=opponent_rosters,
    max_send=2,
    max_receive=2,
    n_targets=10,
    n_pieces=10,
    n_candidates=20,
)

# Cell 7: Print trade report
print_trade_report(situation, candidates, player_values, top_n=5)

# Cell 8: Evaluate a specific trade idea
result = evaluate_trade(
    send_players=['Gunnar Henderson-H'],
    receive_players=['Trea Turner-H'],
    player_values=player_values,
)

# Cell 9: Verify with ground truth calculation
verification = verify_trade_impact(
    send_players=['Gunnar Henderson-H'],
    receive_players=['Trea Turner-H'],
    my_roster_names=my_roster_names,
    projections=projections,
    opponent_totals=opponent_totals,
)
```

---

## Edge Cases and Implementation Notes

1. **Zero MCV categories:** If mcv_acquire[c] = 0 (no contested losses), acquiring production in c doesn't help flip any matchups. Similarly for mcv_protect. This is correct behavior.

2. **All-zero contributions:** A player with 0 in all stats has contribution = 0 everywhere. After z-score normalization, they'll have below-average z-scores. Their PCV will depend on which categories have high MCV.

3. **Standard deviation < MIN_STD:** When computing z-scores, if std < MIN_STD for a category, set z-score = 0 for all players in that category. This prevents division by near-zero and treats uniform distributions as having no differentiating value.

4. **Empty trade candidates:** If no trades pass the fairness and goodness filters, `generate_trade_candidates` returns an empty list. The `print_trade_report` function handles this gracefully with suggestions.

5. **Player on multiple opponent rosters:** This shouldn't happen (load_opponent_rosters asserts no duplicates). If it did, the owner_map in `identify_trade_targets` would arbitrarily pick one owner.

6. **Roster composition after trade:** Trades don't need to be player-count balanced. A 2-for-1 trade changes roster size. `verify_trade_impact` checks that the post-trade roster still meets MIN/MAX bounds for hitters and pitchers.

7. **Ratio stat edge cases:** Players with very low PA or IP have small absolute contributions to ratio categories regardless of their rate. This is handled correctly—the contribution is PA × (rate - baseline), so low PA means low contribution.

8. **Name suffix handling:** All player names include -H or -P suffix throughout internal processing. Display functions strip suffixes. The `_strip_suffix` helper is used consistently.

9. **Negative PCV values:** A player can have negative pcv_acquire if they hurt you in high-MCV categories (e.g., a pitcher with bad ERA when you're in contested ERA races). Similarly for pcv_protect. These players should generally not be acquired/traded for.

10. **Self-trades:** The system doesn't prevent evaluating "trades" where send and receive are the same player. The assertions check that receive_players are not in my_roster, which prevents this.

11. **Opponent roster changes:** The trade engine uses a snapshot of opponent rosters. If opponents make moves, re-run the analysis with updated data.

12. **Tie-breaking in races:** A gap of exactly 0 is classified as contested_loss (gap >= -threshold and gap < 0 fails, but gap > 0 also fails). Add explicit handling: `elif gap[c,j] == 0: race_class = 'tied'` and treat ties as contested_loss for MCV purposes (you need to gain ground to win).

---

## Final Checklist

Before implementation is complete:

### Core Functions
1. ☐ `compute_gap_matrix` handles sign flip for ERA and WHIP
2. ☐ `classify_races` uses configurable thresholds from GAP_THRESHOLDS
3. ☐ `classify_races` handles exact ties (gap = 0) as contested_loss
4. ☐ `compute_mcv` returns tuple of (mcv_acquire, mcv_protect)
5. ☐ `compute_player_contributions` sets cross-type contributions to 0
6. ☐ `compute_player_contributions` handles ratio stats with correct signs
7. ☐ `compute_player_values` z-score normalizes within player type only
8. ☐ `compute_player_values` handles std < MIN_STD edge case
9. ☐ `compute_player_values` computes both pcv_acquire and pcv_protect
10. ☐ `identify_trade_targets` filters to opponent roster players only
11. ☐ `identify_trade_targets` includes owner_id column
12. ☐ `evaluate_trade` uses pcv_acquire for received, pcv_protect for sent
13. ☐ `evaluate_trade` uses absolute FAIRNESS_THRESHOLD
14. ☐ `generate_trade_candidates` handles empty results gracefully
15. ☐ `verify_trade_impact` validates roster composition bounds

### Integration
16. ☐ Imports `compute_team_totals` from roster_optimizer
17. ☐ Uses same category constants as roster_optimizer
18. ☐ Player names include -H/-P suffix in all data structures
19. ☐ Display functions use `_strip_suffix` for clean output

### Edge Cases
20. ☐ Handles zero MCV categories correctly
21. ☐ Handles std < MIN_STD in z-score computation
22. ☐ Handles exact ties (gap = 0) in race classification
23. ☐ Validates roster composition after trade
24. ☐ Filters trade targets to opponent-owned players only
25. ☐ Returns empty list (not error) when no fair trades found

### Code Style
26. ☐ No classes—all module-level functions
27. ☐ All assertions have descriptive error messages
28. ☐ No try/except blocks anywhere
29. ☐ Print statements for progress at each major step
30. ☐ tqdm for loop over trade combinations in generate_trade_candidates