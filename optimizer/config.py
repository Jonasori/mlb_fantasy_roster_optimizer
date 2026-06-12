"""
Configuration loading from repo-root config.json.

Exposes league constants as module-level variables.
"""

import datetime
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
MIN_HITTERS: int = LEAGUE["min_hitters"]  # 12
MAX_HITTERS: int = LEAGUE["max_hitters"]  # 18
MIN_PITCHERS: int = LEAGUE["min_pitchers"]  # 10
MAX_PITCHERS: int = LEAGUE["max_pitchers"]  # 14
HITTING_SLOTS: dict[str, int] = LEAGUE[
    "hitting_slots"
]  # {"C": 1, "1B": 1, ..., "UTIL": 1}
PITCHING_SLOTS: dict[str, int] = LEAGUE["pitching_slots"]  # {"SP": 5, "RP": 2}
SLOT_ELIGIBILITY: dict[str, set[str]] = {
    k: set(v) for k, v in LEAGUE["slot_eligibility"].items()
}

# Team identity
MY_TEAM_NAME: str = LEAGUE["my_team_name"]  # "The Big Dumpers"
FANTRAX_TEAM_IDS: dict[str, str] = LEAGUE["fantrax_team_ids"]  # {team_name: team_id}
# Inverse map for reconciling APIs that key by team_id (e.g. standings, whose
# display names can differ from the roster/owner names used elsewhere).
TEAM_ID_TO_NAME: dict[str, str] = {v: k for k, v in FANTRAX_TEAM_IDS.items()}
ALL_TEAM_NAMES: list[str] = sorted(LEAGUE["fantrax_team_ids"].keys())

# Numeric safety
MIN_STAT_STANDARD_DEVIATION: float = LEAGUE["min_stat_standard_deviation"]  # 0.001

# Perceived Value (trade-market perception) tuning. See player_scoring.add_perceived_value.
_PV_CONFIG: dict = _CONFIG.get("perceived_value", {})
FAME_WAR_THRESHOLD: float = _PV_CONFIG.get("fame_war_threshold", 3.0)
FAME_WAR_SLOPE: float = _PV_CONFIG.get("fame_war_slope", 3.0)

# Trade fairness: max fraction of PV an opponent will accept losing.
_TRADE_CONFIG: dict = _CONFIG.get("trade_engine", {})
PV_MAX_LOSS_FRAC: float = _TRADE_CONFIG.get("pv_max_loss_frac", 0.15)

# Regular-season window (used to compute the fraction of season remaining,
# e.g. for scaling rest-of-season WAR back to a full-season fame premium).
SEASON_START: datetime.date = datetime.date.fromisoformat(LEAGUE["season_start"])
SEASON_END: datetime.date = datetime.date.fromisoformat(LEAGUE["season_end"])


def season_fraction_remaining(today: datetime.date | None = None) -> float:
    """Fraction of the regular season still to be played, clamped to [0, 1].

    1.0 before opening day, 0.0 after the final day. Used to rescale
    rest-of-season quantities (e.g. WAR) to a full-season-equivalent scale.
    """
    if today is None:
        today = datetime.date.today()
    span = (SEASON_END - SEASON_START).days
    assert span > 0, f"SEASON_END must be after SEASON_START; got {span} days"
    remaining = (SEASON_END - today).days
    return max(0.0, min(1.0, remaining / span))


# Derived
N_STARTER_SLOTS: int = sum(HITTING_SLOTS.values()) + sum(PITCHING_SLOTS.values())  # 18

# Validation
assert len(ALL_CATEGORIES) == 10, "Must have exactly 10 scoring categories"
assert len(LEAGUE["fantrax_team_ids"]) == 7, "Must have exactly 7 teams"
