# Configuration and Constants

## Overview

This document defines all league configuration constants and shared utility functions. These are the foundational pieces imported by all other modules.

**Module:** `optimizer/data_loader.py` (configuration section)

---

## League Configuration

```python
# === LEAGUE CONFIGURATION ===

# Fantrax identification
FANTRAX_LEAGUE_ID = "f7cc72ecmkfnc7kl"
MY_TEAM_NAME = "The Big Dumpers"
MY_TEAM_ID = "zhh2uwcamkfnc7kw"

# All teams in the league
FANTRAX_TEAM_IDS = {
    "Aidangonnawin": "oluroo3mmkfnc7kw",
    "Future AL MVP, Evan Carter": "a80hb31gmkfnc7kw",
    "Oliver Wendell Homers": "ea7d523tmkfnc7kw",
    "paranoia_in_z_major": "2n4v2dn8mkfnc7kw",
    "Reasonable Doubtfielders": "03xns3dwmkfnc7kw",
    "Shohei Me The (Betting) Money!": "9z72hkg2mkfnc7kw",
    "The Big Dumpers": "zhh2uwcamkfnc7kw",
}

# Scoring categories
HITTING_CATEGORIES = ['R', 'HR', 'RBI', 'SB', 'OPS']
PITCHING_CATEGORIES = ['W', 'SV', 'K', 'ERA', 'WHIP']
ALL_CATEGORIES = HITTING_CATEGORIES + PITCHING_CATEGORIES

# Categories where lower is better
NEGATIVE_CATEGORIES = {'ERA', 'WHIP'}

# Ratio stats and their weighting denominators
RATIO_STATS = {
    'OPS': 'PA',   # Team OPS = sum(PA * OPS) / sum(PA)
    'ERA': 'IP',   # Team ERA = sum(IP * ERA) / sum(IP)
    'WHIP': 'IP',  # Team WHIP = sum(IP * WHIP) / sum(IP)
}

# Roster construction
ROSTER_SIZE = 26  # Active roster only

# Starting lineup slots
HITTING_SLOTS = {'C': 1, '1B': 1, '2B': 1, 'SS': 1, '3B': 1, 'OF': 3, 'UTIL': 1}
PITCHING_SLOTS = {'SP': 5, 'RP': 2}

# Position eligibility: which player positions can fill which lineup slots
SLOT_ELIGIBILITY = {
    'C': {'C'}, '1B': {'1B'}, '2B': {'2B'}, 'SS': {'SS'}, '3B': {'3B'},
    'OF': {'OF'}, 'UTIL': {'C', '1B', '2B', 'SS', '3B', 'OF', 'DH'},
    'SP': {'SP'}, 'RP': {'RP'},
}

# Roster composition bounds
MIN_HITTERS = 12
MAX_HITTERS = 16
MIN_PITCHERS = 10
MAX_PITCHERS = 14

# League structure
NUM_OPPONENTS = 6  # 7-team league

# Minimum standard deviation for calculations
MIN_STD = 0.001

# Status mapping for Fantrax roster data
FANTRAX_STATUS_MAP = {
    "Act": "active",
    "Res": "active",    # Reserve still counts toward 26-man
    "Min": "prospect",
    "IR": "IR",
    # API status IDs
    "1": "active",
    "2": "reserve", 
    "3": "IR",
    "9": "minors",
}
```

---

## SGP Configuration

Standing Gain Points (SGP) is a context-free player valuation metric that estimates how many rotisserie standings points a player contributes.

**Key insight from Smart Fantasy Baseball:** Rate stats (OPS, ERA, WHIP) must be weighted by playing time. A .850 OPS player with 600 PA contributes more to team OPS than the same player with 100 PA.

```python
# === SGP (STANDING GAIN POINTS) CONFIGURATION ===
# Denominators: "how much of stat X gains one standing point"
# Based on Smart Fantasy Baseball analysis of 12-team leagues, adjusted for 7-team.
# See: https://www.smartfantasybaseball.com/2013/03/create-your-own-fantasy-baseball-rankings-part-5-understanding-standings-gain-points/

SGP_DENOMINATORS = {
    "R": 20.0, "HR": 8.0, "RBI": 20.0, "SB": 7.0,  # Hitting counting
    "W": 3.5, "SV": 8.0, "K": 35.0,                 # Pitching counting
}

# Rate stat SGP config: (denominator, league_average)
# Denominator is per-unit change in team ratio that gains one standings point
SGP_RATE_STATS = {
    "OPS": (0.010, 0.750),   # Higher is better
    "ERA": (0.18, 4.00),     # Lower is better
    "WHIP": (0.030, 1.25),   # Lower is better
}

# Baseline team totals for SGP calculation (14 hitters, 9 pitchers typical)
SGP_BASELINE_PA = 7000   # 14 hitters × 500 PA average
SGP_BASELINE_IP = 1200   # 9 pitchers × ~133 IP average
```

---

## Core Utility Functions

### Name Suffix Handling

**All player names include `-H` or `-P` suffix** to ensure uniqueness:
- `"Mike Trout-H"`, `"Gerrit Cole-P"`
- `"Shohei Ohtani-H"`, `"Shohei Ohtani-P"` (two-way players appear twice)

```python
def strip_name_suffix(name: str) -> str:
    """
    Strip -H or -P suffix from player name for display.
    
    This is the SINGLE source of truth for suffix stripping.
    All other modules MUST import this function, not redefine it.
    
    Example: "Mike Trout-H" → "Mike Trout"
    """
    if name.endswith("-H") or name.endswith("-P"):
        return name[:-2]
    return name
```

### Name Normalization

Used for matching names between FanGraphs and Fantrax (handles accents, suffixes like Jr.).

```python
import unicodedata

def normalize_name(name: str) -> str:
    """
    Normalize player name for fuzzy comparison.
    
    CRITICAL: Preserves -H/-P suffix!
    
    Handles:
        - Accented characters (Rodríguez → rodriguez)
        - Name suffixes like Jr., Sr. (removed)
        - Apostrophe variants
    
    Example: "Ronald Acuña Jr.-H" → "ronald acuna-h"
    """
    # Preserve suffix
    suffix = ""
    if name.endswith("-H"):
        suffix = "-H"
        name = name[:-2]
    elif name.endswith("-P"):
        suffix = "-P"
        name = name[:-2]
    
    # Normalize unicode (decompose accented characters)
    name = unicodedata.normalize("NFD", name)
    # Remove combining characters (accents)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    
    # Lowercase
    name = name.lower()
    
    # Remove suffixes
    for suffix_remove in [" jr.", " jr", " sr.", " sr", " ii", " iii", " iv"]:
        name = name.replace(suffix_remove, "")
    
    # Normalize apostrophes
    name = name.replace("\u2019", "'").replace("`", "'")
    
    return name.strip() + suffix.lower()
```

### SGP Computation

```python
def compute_sgp_value(row: pd.Series) -> float:
    """
    Compute Standing Gain Points (SGP) for a single player.
    
    IMPORTANT: Rate stats (OPS, ERA, WHIP) are weighted by playing time.
    This follows the canonical SGP methodology from Smart Fantasy Baseball.
    
    Args:
        row: Player row with: player_type, PA, R, HR, RBI, SB, OPS (hitters)
             or IP, W, SV, K, ERA, WHIP (pitchers)
    
    Returns:
        Total SGP value. Typical ranges: 0-30 for hitters, 0-20 for pitchers.
    
    Implementation:
        Counting stats: SGP = stat_value / denominator
        
        Rate stats (WITH playing time weighting):
            1. Compute how player changes team's ratio stat
            2. team_stat = (player_weight * player_stat + baseline_weight * baseline_stat) / total_weight
            3. impact = team_stat - league_average
            4. SGP = impact / denominator
            
        For "lower is better" stats (ERA, WHIP), flip sign so positive = good.
    """
    sgp = 0.0
    
    if row["player_type"] == "hitter":
        # Counting stats: R, HR, RBI, SB
        for stat in ["R", "HR", "RBI", "SB"]:
            sgp += row[stat] / SGP_DENOMINATORS[stat]
        
        # OPS (rate stat with PA weighting)
        denom, league_avg = SGP_RATE_STATS["OPS"]
        player_pa = row["PA"]
        player_ops = row["OPS"]
        
        # Replace one "average" player slot with this player
        avg_player_pa = SGP_BASELINE_PA / 14  # ~500
        remaining_pa = SGP_BASELINE_PA - avg_player_pa
        
        # Team OPS with this player
        team_ops = (player_pa * player_ops + remaining_pa * league_avg) / (player_pa + remaining_pa)
        ops_impact = team_ops - league_avg
        sgp += ops_impact / denom
        
    else:  # pitcher
        # Counting stats: W, SV, K
        for stat in ["W", "SV", "K"]:
            sgp += row[stat] / SGP_DENOMINATORS[stat]
        
        # ERA (rate stat with IP weighting, lower is better)
        denom, league_avg = SGP_RATE_STATS["ERA"]
        player_ip = row["IP"]
        player_era = row["ERA"]
        
        avg_player_ip = SGP_BASELINE_IP / 9  # ~133
        remaining_ip = SGP_BASELINE_IP - avg_player_ip
        
        # Team ERA with this player (ER-based calculation)
        player_er = player_era * player_ip / 9
        remaining_er = league_avg * remaining_ip / 9
        team_era = (player_er + remaining_er) * 9 / (player_ip + remaining_ip)
        
        # Flip sign: lower ERA = positive SGP
        era_impact = league_avg - team_era
        sgp += era_impact / denom
        
        # WHIP (rate stat with IP weighting, lower is better)
        denom, league_avg = SGP_RATE_STATS["WHIP"]
        player_whip = row["WHIP"]
        
        # Team WHIP with this player
        player_baserunners = player_whip * player_ip
        remaining_baserunners = league_avg * remaining_ip
        team_whip = (player_baserunners + remaining_baserunners) / (player_ip + remaining_ip)
        
        # Flip sign: lower WHIP = positive SGP
        whip_impact = league_avg - team_whip
        sgp += whip_impact / denom
    
    return sgp
```

### Projection Uncertainty Estimation

```python
# Historical priors for category standard deviations (regularization)
PRIOR_SIGMAS = {
    "R": 40.0, "HR": 25.0, "RBI": 40.0, "SB": 20.0, "OPS": 0.015,
    "W": 8.0, "SV": 15.0, "K": 80.0, "ERA": 0.25, "WHIP": 0.05,
}

def estimate_projection_uncertainty(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
) -> dict[str, float]:
    """
    Estimate standard deviation of team totals for each category.
    
    Uses a hybrid approach:
    1. Empirical: std dev across 7 teams (captures league-specific variation)
    2. Prior: historical baselines (regularization for small sample)
    
    Args:
        my_totals: My team's category totals
        opponent_totals: Dict of opponent totals (6 opponents)
    
    Returns:
        Dict mapping category name to estimated standard deviation.
        Example: {'R': 45.2, 'HR': 28.1, ..., 'ERA': 0.32, ...}
    
    Implementation:
        final_sigma = sqrt(empirical_var * 0.6 + prior_var * 0.4)
        
        With only 7 teams, we weight empirical at 60% and use priors
        to prevent extreme values from small sample noise.
    """
    EMPIRICAL_WEIGHT = 0.6
    
    category_sigmas = {}
    for category in ALL_CATEGORIES:
        # Collect all 7 team values
        values = [my_totals[category]]
        for opp_id in sorted(opponent_totals.keys()):
            values.append(opponent_totals[opp_id][category])
        
        # Empirical std (population, since we have all teams)
        empirical_sigma = np.std(values, ddof=0)
        
        # Blend with prior
        prior_sigma = PRIOR_SIGMAS[category]
        blended_var = (
            empirical_sigma**2 * EMPIRICAL_WEIGHT + 
            prior_sigma**2 * (1 - EMPIRICAL_WEIGHT)
        )
        
        category_sigmas[category] = max(np.sqrt(blended_var), MIN_STD)
    
    return category_sigmas
```

---

## Validation

Add these assertions to your implementation:

```python
# In data_loader.py module initialization
assert len(ALL_CATEGORIES) == 10, "Must have exactly 10 scoring categories"
assert len(FANTRAX_TEAM_IDS) == 7, "Must have exactly 7 teams"
assert sum(HITTING_SLOTS.values()) == 9, "Must have 9 hitting slots"
assert sum(PITCHING_SLOTS.values()) == 7, "Must have 7 pitching slots"
assert MIN_HITTERS + MIN_PITCHERS <= ROSTER_SIZE, "Composition bounds must fit roster"
```
