"""
Configuration loading and validation.

Loads all configuration from config.json and exposes constants as module-level variables.
"""

import json
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent.parent / "config.json"


def load_config(config_path: Path | str | None = None) -> dict:
    """
    Load configuration from JSON file.

    Args:
        config_path: Path to config.json file (defaults to project root)

    Returns:
        Dict containing all configuration values

    Raises:
        FileNotFoundError: If config file doesn't exist
        AssertionError: If config is invalid
    """
    if config_path is None:
        config_path = _CONFIG_PATH
    else:
        config_path = Path(config_path)

    assert config_path.exists(), f"Config file not found: {config_path}"

    with open(config_path) as f:
        config = json.load(f)

    # Validate required sections
    assert "league" in config, "Config must have 'league' section"
    assert "sgp" in config, "Config must have 'sgp' section"
    assert "projections" in config, "Config must have 'projections' section"
    assert "trade_engine" in config, "Config must have 'trade_engine' section"
    assert "fantrax" in config, "Config must have 'fantrax' section"
    assert "cookies" in config["fantrax"], "Config must have 'fantrax.cookies' section"

    # Validate SGP metric
    assert config["sgp"]["metric"] in ["raw", "dynasty"], (
        "sgp.metric must be 'raw' or 'dynasty'"
    )

    return config


# Load config at module level
_CONFIG = load_config()

# Expose config sections
LEAGUE = _CONFIG["league"]
SGP_CONFIG = _CONFIG["sgp"]
PROJECTIONS_CONFIG = _CONFIG["projections"]
TRADE_ENGINE_CONFIG = _CONFIG["trade_engine"]
FANTRAX_CONFIG = _CONFIG["fantrax"]

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
SGP_RATE_STATS = {k: tuple(v) for k, v in SGP_CONFIG["rate_stats"].items()}
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

# Fantrax cookies
FANTRAX_COOKIES = FANTRAX_CONFIG["cookies"]

# Validation assertions
assert len(ALL_CATEGORIES) == 10, "Must have exactly 10 scoring categories"
assert len(FANTRAX_TEAM_IDS) == 7, "Must have exactly 7 teams"
assert sum(HITTING_SLOTS.values()) == 9, "Must have 9 hitting slots"
assert sum(PITCHING_SLOTS.values()) == 7, "Must have 7 pitching slots"
assert MIN_HITTERS + MIN_PITCHERS <= ROSTER_SIZE, "Composition bounds must fit roster"
assert SGP_METRIC in ["raw", "dynasty"], "SGP_METRIC must be 'raw' or 'dynasty'"
