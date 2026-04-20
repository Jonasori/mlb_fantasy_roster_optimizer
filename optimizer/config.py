"""
Configuration loading from repo-root config.json.

Exposes league constants as module-level variables.
"""

import json
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent.parent / "config.json"


def load_config() -> dict:
    """Load configuration from repo-root config.json."""
    assert _CONFIG_PATH.exists(), f"Config file not found: {_CONFIG_PATH}"
    with open(_CONFIG_PATH) as f:
        return json.load(f)


_CONFIG = load_config()
LEAGUE = _CONFIG["league"]

# Categories
HITTING_CATEGORIES: list[str] = LEAGUE[
    "hitting_categories"
]  # ['R', 'HR', 'RBI', 'SB', 'OPS']
PITCHING_CATEGORIES: list[str] = LEAGUE[
    "pitching_categories"
]  # ['W', 'SV', 'K', 'ERA', 'WHIP']
ALL_CATEGORIES: list[str] = HITTING_CATEGORIES + PITCHING_CATEGORIES
NEGATIVE_CATEGORIES: set[str] = set(LEAGUE["negative_categories"])  # {'ERA', 'WHIP'}

# League structure
NUM_OPPONENTS: int = len(LEAGUE["fantrax_team_ids"]) - 1  # 6
ROSTER_SIZE: int = LEAGUE["roster_size"]  # 28
HITTING_SLOTS: dict[str, int] = LEAGUE[
    "hitting_slots"
]  # {"C": 1, "1B": 1, ..., "UTIL": 1}
PITCHING_SLOTS: dict[str, int] = LEAGUE["pitching_slots"]  # {"SP": 5, "RP": 2}
SLOT_ELIGIBILITY: dict[str, set[str]] = {
    k: set(v) for k, v in LEAGUE["slot_eligibility"].items()
}

# Team identity
MY_TEAM_NAME: str = LEAGUE["my_team_name"]  # "The Big Dumpers"
ALL_TEAM_NAMES: list[str] = sorted(LEAGUE["fantrax_team_ids"].keys())

# Numeric safety
MIN_STAT_STANDARD_DEVIATION: float = LEAGUE["min_stat_standard_deviation"]  # 0.001

# Derived
N_STARTER_SLOTS: int = sum(HITTING_SLOTS.values()) + sum(PITCHING_SLOTS.values())  # 18

# Validation
assert len(ALL_CATEGORIES) == 10, "Must have exactly 10 scoring categories"
assert len(LEAGUE["fantrax_team_ids"]) == 7, "Must have exactly 7 teams"
