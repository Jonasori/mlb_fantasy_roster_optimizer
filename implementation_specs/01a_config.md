# Configuration and Constants

## Overview

This document defines all league configuration constants and shared utility functions. These are the foundational pieces imported by all other modules.

**Module:** `optimizer/config.py` (config loading) and `optimizer/data_loader.py` (utility functions)

**Config File:** `config.json` at project root

---

## Cross-References

**Depends on:** [00_agent_guidelines.md](00_agent_guidelines.md) for code style

**Used by:**
- [01b_fangraphs_loading.md](01b_fangraphs_loading.md) — imports constants and `compute_sgp_value()`
- [01c_fantrax_api.md](01c_fantrax_api.md) — imports `FANTRAX_LEAGUE_ID`, `FANTRAX_TEAM_IDS`
- [01d_database.md](01d_database.md) — imports constants for schema validation
- [02_free_agent_optimizer.md](02_free_agent_optimizer.md) — imports slot eligibility, roster bounds
- [03_trade_engine.md](03_trade_engine.md) — imports `compute_team_totals()`, `estimate_projection_uncertainty()`, SGP metric selection, trade engine config
- [04_visualizations.md](04_visualizations.md) — imports category lists, `NEGATIVE_CATEGORIES`

---

## Configuration File Structure

All configuration is stored in `config.json` at the project root. The `optimizer/config.py` module loads and validates this file, exposing constants as module-level variables.

**File:** `config.json`

```json
{
  "league": {
    "fantrax_league_id": "f7cc72ecmkfnc7kl",
    "my_team_name": "The Big Dumpers",
    "fantrax_team_ids": {
      "Aidangonnawin": "oluroo3mmkfnc7kw",
      "Future AL MVP, Evan Carter": "a80hb31gmkfnc7kw",
      "Oliver Wendell Homers": "ea7d523tmkfnc7kw",
      "paranoia_in_z_major": "2n4v2dn8mkfnc7kw",
      "Reasonable Doubtfielders": "03xns3dwmkfnc7kw",
      "Shohei Me The (Betting) Money!": "9z72hkg2mkfnc7kw",
      "The Big Dumpers": "zhh2uwcamkfnc7kw"
    },
    "roster_size": 26,
    "min_hitters": 12,
    "max_hitters": 16,
    "min_pitchers": 10,
    "max_pitchers": 14,
    "hitting_slots": {"C": 1, "1B": 1, "2B": 1, "SS": 1, "3B": 1, "OF": 3, "UTIL": 1},
    "pitching_slots": {"SP": 5, "RP": 2},
    "slot_eligibility": {
      "C": ["C"],
      "1B": ["1B"],
      "2B": ["2B"],
      "SS": ["SS"],
      "3B": ["3B"],
      "OF": ["OF"],
      "UTIL": ["C", "1B", "2B", "SS", "3B", "OF", "DH"],
      "SP": ["SP"],
      "RP": ["RP"]
    },
    "hitting_categories": ["R", "HR", "RBI", "SB", "OPS"],
    "pitching_categories": ["W", "SV", "K", "ERA", "WHIP"],
    "negative_categories": ["ERA", "WHIP"],
    "ratio_stats": {
      "OPS": "PA",
      "ERA": "IP",
      "WHIP": "IP"
    },
    "fantrax_active_status_ids": ["1", "2"],
    "min_stat_standard_deviation": 0.001,
    "balance_lambda_default": 0.5
  },
  "trade_engine": {
    "fairness_threshold_percent": 0.10,
    "max_trade_size": 3,
    "min_meaningful_improvement": 0.1,
    "lose_cost_scale": 2
  },
  "sgp": {
    "denominators": {
      "R": 20.0,
      "HR": 8.0,
      "RBI": 20.0,
      "SB": 7.0,
      "W": 3.5,
      "SV": 8.0,
      "K": 35.0
    },
    "rate_stats": {
      "OPS": [0.010, 0.750, true],
      "ERA": [0.18, 4.00, false],
      "WHIP": [0.030, 1.25, false]
    },
    "metric": "raw"
  },
  "projections": {
    "data_dir": "data/",
    "raw_hitters": "data/fangraphs-atc-projections-hitters.csv",
    "raw_pitchers": "data/fangraphs-atc-projections-pitchers.csv",
    "adjusted_hitters": "data/fangraphs-atc-pt-adjusted-hitters.csv",
    "adjusted_pitchers": "data/fangraphs-atc-pt-adjusted-pitchers.csv",
    "use_adjusted": true
  }
}
```

**Key Configuration Options:**

1. **`sgp.metric`**: `"raw"` or `"dynasty"` — selects which SGP metric to use for trade fairness evaluation
   - `"raw"`: Single-season SGP (default, simpler, more predictable)
   - `"dynasty"`: Age-adjusted dynasty SGP (for dynasty leagues)

2. **`projections.use_adjusted`**: `true` or `false` — whether to use playing-time-adjusted projections
   - `true`: Use adjusted projections (default)
   - `false`: Use raw FanGraphs projections

3. **`trade_engine.fairness_threshold_percent`**: `0.10` (10%) — maximum SGP differential percentage for a trade to be considered "fair"
   - A fair trade has SGP differential within this percentage of total SGP involved
   - Example: 45.0 SGP for 62.0 SGP = 17 diff / 107 total = 16% → UNFAIR
   - Example: 30.0 SGP for 35.0 SGP = 5 diff / 65 total = 8% → FAIR

4. **`trade_engine.max_trade_size`**: `3` — maximum players per side in a trade

5. **`trade_engine.min_meaningful_improvement`**: `0.1` — minimum expected win probability improvement (in EWA terms) to recommend ACCEPT
   - Trades with smaller EWA are marked NEUTRAL even if positive

6. **`trade_engine.lose_cost_scale`**: `2` — scale factor for converting ewa_lose to SGP scale for expendability calculation

**Derived Values:**

- **`my_team_id`**: Inferred from `fantrax_team_ids[my_team_name]` — no need to specify separately
- **`num_opponents`**: Calculated as `len(fantrax_team_ids) - 1` — automatically computed from team count

---

## Config Loading Module

**Module:** `optimizer/config.py`

```python
import json
from pathlib import Path

def load_config(config_path: str = "config.json") -> dict:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to config.json file
    
    Returns:
        Dict containing all configuration values
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    config_file = Path(config_path)
    assert config_file.exists(), f"Config file not found: {config_path}"
    
    with open(config_file) as f:
        config = json.load(f)
    
    # Validate required sections
    assert "league" in config, "Config must have 'league' section"
    assert "sgp" in config, "Config must have 'sgp' section"
    assert "projections" in config, "Config must have 'projections' section"
    assert "trade_engine" in config, "Config must have 'trade_engine' section"
    
    # Validate SGP metric
    assert config["sgp"]["metric"] in ["raw", "dynasty"], \
        "sgp.metric must be 'raw' or 'dynasty'"
    
    return config

# Load config at module level
_CONFIG = load_config()

# Expose config sections
LEAGUE = _CONFIG["league"]
SGP_CONFIG = _CONFIG["sgp"]
PROJECTIONS_CONFIG = _CONFIG["projections"]
TRADE_ENGINE_CONFIG = _CONFIG["trade_engine"]

# Convenience accessors for league constants
FANTRAX_LEAGUE_ID = LEAGUE["fantrax_league_id"]
MY_TEAM_NAME = LEAGUE["my_team_name"]
FANTRAX_TEAM_IDS = LEAGUE["fantrax_team_ids"]

# Infer my_team_id from fantrax_team_ids
assert MY_TEAM_NAME in FANTRAX_TEAM_IDS, (
    f"my_team_name '{MY_TEAM_NAME}' not found in fantrax_team_ids"
)
MY_TEAM_ID = FANTRAX_TEAM_IDS[MY_TEAM_NAME]

# Calculate num_opponents from fantrax_team_ids
NUM_OPPONENTS = len(FANTRAX_TEAM_IDS) - 1

ROSTER_SIZE = LEAGUE["roster_size"]
MIN_HITTERS = LEAGUE["min_hitters"]
MAX_HITTERS = LEAGUE["max_hitters"]
MIN_PITCHERS = LEAGUE["min_pitchers"]
MAX_PITCHERS = LEAGUE["max_pitchers"]
HITTING_SLOTS = LEAGUE["hitting_slots"]
PITCHING_SLOTS = LEAGUE["pitching_slots"]
SLOT_ELIGIBILITY = {k: set(v) for k, v in LEAGUE["slot_eligibility"].items()}
HITTING_CATEGORIES = LEAGUE["hitting_categories"]
PITCHING_CATEGORIES = LEAGUE["pitching_categories"]
ALL_CATEGORIES = HITTING_CATEGORIES + PITCHING_CATEGORIES
NEGATIVE_CATEGORIES = set(LEAGUE["negative_categories"])
RATIO_STATS = LEAGUE["ratio_stats"]
FANTRAX_ACTIVE_STATUS_IDS = set(LEAGUE["fantrax_active_status_ids"])
MIN_STAT_STANDARD_DEVIATION = LEAGUE["min_stat_standard_deviation"]
BALANCE_LAMBDA_DEFAULT = LEAGUE["balance_lambda_default"]

# SGP constants
SGP_DENOMINATORS = SGP_CONFIG["denominators"]
SGP_RATE_STATS = {
    k: tuple(v) for k, v in SGP_CONFIG["rate_stats"].items()
}
SGP_METRIC = SGP_CONFIG["metric"]  # "raw" or "dynasty"

# Trade engine constants
TRADE_FAIRNESS_THRESHOLD_PERCENT = TRADE_ENGINE_CONFIG["fairness_threshold_percent"]
TRADE_MAX_SIZE = TRADE_ENGINE_CONFIG["max_trade_size"]
TRADE_MIN_MEANINGFUL_IMPROVEMENT = TRADE_ENGINE_CONFIG["min_meaningful_improvement"]
TRADE_LOSE_COST_SCALE = TRADE_ENGINE_CONFIG["lose_cost_scale"]

# Projections paths
DATA_DIR = PROJECTIONS_CONFIG["data_dir"]
RAW_HITTERS_PATH = PROJECTIONS_CONFIG["raw_hitters"]
RAW_PITCHERS_PATH = PROJECTIONS_CONFIG["raw_pitchers"]
ADJUSTED_HITTERS_PATH = PROJECTIONS_CONFIG["adjusted_hitters"]
ADJUSTED_PITCHERS_PATH = PROJECTIONS_CONFIG["adjusted_pitchers"]
USE_ADJUSTED_PROJECTIONS = PROJECTIONS_CONFIG["use_adjusted"]

# Determine which projection files to use
if USE_ADJUSTED_PROJECTIONS:
    HITTER_PROJ_PATH = ADJUSTED_HITTERS_PATH
    PITCHER_PROJ_PATH = ADJUSTED_PITCHERS_PATH
else:
    HITTER_PROJ_PATH = RAW_HITTERS_PATH
    PITCHER_PROJ_PATH = RAW_PITCHERS_PATH
```

---

## League Configuration

All league constants are loaded from `config.json` via `optimizer/config.py`. The following constants are available:

All constants are loaded from `config.json` and exposed via `optimizer/config.py`. Import them like:

```python
from optimizer.config import (
    FANTRAX_LEAGUE_ID,
    MY_TEAM_NAME,
    MY_TEAM_ID,  # Inferred from fantrax_team_ids[my_team_name]
    FANTRAX_TEAM_IDS,
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
    NUM_OPPONENTS,  # Calculated as len(fantrax_team_ids) - 1
    FANTRAX_ACTIVE_STATUS_IDS,
    MIN_STAT_STANDARD_DEVIATION,
    BALANCE_LAMBDA_DEFAULT,
    TRADE_FAIRNESS_THRESHOLD_PERCENT,
    TRADE_MAX_SIZE,
    TRADE_MIN_MEANINGFUL_IMPROVEMENT,
    TRADE_LOSE_COST_SCALE,
)
```

See `config.json` structure above for the exact values. All constants match the structure shown in the config file.

---

## SGP Configuration

Standing Gain Points (SGP) is a context-free player valuation metric that estimates how many rotisserie standings points a player contributes.

**Design note:** Rate stats (OPS, ERA, WHIP) are NOT weighted by playing time in this implementation. Playing time value is already captured in the counting stats (more PA = more R, HR, RBI, SB; more IP = more W, K). This simpler approach avoids double-counting and produces sensible rankings.

SGP configuration is loaded from `config.json`:

```python
from optimizer.config import (
    SGP_DENOMINATORS,
    SGP_RATE_STATS,
    SGP_METRIC,  # "raw" or "dynasty"
)
```

**SGP Denominators:** "how much of stat X gains one standing point"
- Based on Smart Fantasy Baseball analysis of 12-team leagues, adjusted for 7-team.
- See: https://www.smartfantasybaseball.com/2013/03/create-your-own-fantasy-baseball-rankings-part-5-understanding-standings-gain-points/

**SGP Rate Stats:** `(denominator, league_average, higher_is_better)`
- Denominator is per-unit change in team ratio that gains one standings point

**SGP Metric Selection:** `SGP_METRIC` determines which metric to use for trade fairness:
- `"raw"`: Single-season SGP (default, simpler, more predictable)
- `"dynasty"`: Age-adjusted dynasty SGP (for dynasty leagues)

The trade engine uses `SGP_METRIC` to select the appropriate column from projections (`SGP` or `dynasty_SGP`) for trade fairness evaluation.

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
    
    Args:
        row: Player row with: player_type, R, HR, RBI, SB, OPS (hitters)
             or W, SV, K, ERA, WHIP (pitchers)
    
    Returns:
        Total SGP value. Typical ranges: 0-25 for hitters, 0-15 for pitchers.
    
    Implementation:
        Counting stats: SGP = stat_value / denominator
        Rate stats: SGP = (player_stat - league_avg) / denominator
        For "lower is better" stats (ERA, WHIP), flip sign so positive = good.
        
    Note: Rate stats are NOT weighted by playing time. Playing time value is
    already captured in the counting stats.
    """
    sgp = 0.0
    
    if row["player_type"] == "hitter":
        # Counting stats: R, HR, RBI, SB
        for stat in ["R", "HR", "RBI", "SB"]:
            sgp += row[stat] / SGP_DENOMINATORS[stat]
        
        # OPS (rate stat, higher is better)
        denom, league_avg, _ = SGP_RATE_STATS["OPS"]
        sgp += (row["OPS"] - league_avg) / denom
        
    else:  # pitcher
        # Counting stats: W, SV, K
        for stat in ["W", "SV", "K"]:
            sgp += row[stat] / SGP_DENOMINATORS[stat]
        
        # ERA (rate stat, lower is better)
        denom, league_avg, _ = SGP_RATE_STATS["ERA"]
        # Flip sign: (league_avg - player_ERA) so lower ERA = positive SGP
        sgp += (league_avg - row["ERA"]) / denom
        
        # WHIP (rate stat, lower is better)
        denom, league_avg, _ = SGP_RATE_STATS["WHIP"]
        sgp += (league_avg - row["WHIP"]) / denom
    
    return sgp
```

### Projection Uncertainty Estimation

```python
def estimate_projection_uncertainty(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
) -> dict[str, float]:
    """
    Estimate standard deviation of team totals for each category.
    
    Uses pure empirical approach: std dev across all 7 teams.
    
    Args:
        my_totals: My team's category totals
        opponent_totals: Dict of opponent totals (6 opponents)
    
    Returns:
        Dict mapping category name to estimated standard deviation.
        Example: {'R': 45.2, 'HR': 18.3, ..., 'ERA': 0.35, ...}
    
    Note:
        This empirical approach captures real league variation.
        The σ_c values are used in the Rosenof (2025) formulation
        for normalizing matchup gaps.
    """
    category_sigmas = {}
    
    for category in ALL_CATEGORIES:
        # Collect all 7 team values
        values = [my_totals[category]]
        for opp_id in sorted(opponent_totals.keys()):
            values.append(opponent_totals[opp_id][category])
        
        # Compute standard deviation
        sigma = np.std(values, ddof=0)  # Population std (all teams)
        
        # Ensure minimum to avoid division by zero
        if sigma < MIN_STAT_STANDARD_DEVIATION:
            sigma = MIN_STAT_STANDARD_DEVIATION
        
        category_sigmas[category] = sigma
    
    return category_sigmas
```

---

## Validation

Add these assertions to `optimizer/config.py` after loading:

```python
# Validate config after loading
assert len(ALL_CATEGORIES) == 10, "Must have exactly 10 scoring categories"
assert len(FANTRAX_TEAM_IDS) == 7, "Must have exactly 7 teams"
assert sum(HITTING_SLOTS.values()) == 9, "Must have 9 hitting slots"
assert sum(PITCHING_SLOTS.values()) == 7, "Must have 7 pitching slots"
assert MIN_HITTERS + MIN_PITCHERS <= ROSTER_SIZE, "Composition bounds must fit roster"
assert SGP_METRIC in ["raw", "dynasty"], "SGP_METRIC must be 'raw' or 'dynasty'"
```
