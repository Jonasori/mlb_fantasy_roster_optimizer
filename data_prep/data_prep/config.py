"""
Configuration for data prep: Fantrax, projection paths, league IDs.

Loads shared config.json from the repo root.
"""

import json
import re
from pathlib import Path

# Project root: .../data_prep (contains pyproject.toml, data/)
_DATA_PREP_ROOT = Path(__file__).resolve().parent.parent
# Repo root: parent of data_prep
_REPO_ROOT = _DATA_PREP_ROOT.parent

_DATA_DIR = _DATA_PREP_ROOT / "data"

# Default artifact for v1/v2 and CLI
SILVER_TABLE_DEFAULT_PATH = _DATA_DIR / "silver_table.parquet"

# Expose data directory for scripts and docs
DATA_DIR = _DATA_DIR


def _resolve_config_path(config_path: Path | str | None) -> Path:
    if config_path is not None:
        p = Path(config_path)
        assert p.exists(), f"Config file not found: {p}"
        return p
    repo = _REPO_ROOT / "config.json"
    assert repo.exists(), (
        f"No config.json at repo root ({repo}). "
        "Pass config_path= to load_config() or add config.json."
    )
    return repo


def find_latest_projection_folder(data_dir: Path | None = None) -> Path | None:
    """
    Find the most recent pulled_YYYYMMDD folder under data/.

    Returns:
        Path to the newest projection folder, or None if no such folders exist.
    """
    root = Path(data_dir) if data_dir is not None else _DATA_DIR
    assert root.is_dir(), f"Data directory not found: {root}"

    pattern = re.compile(r"^pulled_(\d{8})$")
    candidates: list[tuple[str, Path]] = []

    for entry in root.iterdir():
        if entry.is_dir():
            match = pattern.match(entry.name)
            if match:
                date_str = match.group(1)
                candidates.append((date_str, entry))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def load_config(config_path: Path | str | None = None) -> dict:
    """
    Load configuration from JSON file.

    Args:
        config_path: Path to config.json (defaults to repo root config.json)

    Returns:
        Full config dict (caller may read league, projections, fantrax sections).

    Raises:
        AssertionError: If config file is missing or invalid for data prep.
    """
    path = _resolve_config_path(config_path)

    with open(path) as f:
        config = json.load(f)

    assert "league" in config, "Config must have 'league' section"
    assert "projections" in config, "Config must have 'projections' section"
    assert "fantrax" in config, "Config must have 'fantrax' section"
    assert "cookies" in config["fantrax"], "Config must have 'fantrax.cookies' section"

    return config


_CONFIG = load_config()

LEAGUE = _CONFIG["league"]
PROJECTIONS_CONFIG = _CONFIG["projections"]
FANTRAX_CONFIG = _CONFIG["fantrax"]

FANTRAX_LEAGUE_ID = LEAGUE["fantrax_league_id"]
MY_TEAM_NAME = LEAGUE["my_team_name"]
FANTRAX_TEAM_IDS = LEAGUE["fantrax_team_ids"]

assert MY_TEAM_NAME in FANTRAX_TEAM_IDS, (
    f"my_team_name '{MY_TEAM_NAME}' not found in fantrax_team_ids"
)

_NUM_TEAMS = len(FANTRAX_TEAM_IDS)
assert _NUM_TEAMS == 7, f"Data prep Fantrax helpers expect 7 teams, got {_NUM_TEAMS}"

_LATEST_FOLDER = find_latest_projection_folder()
if _LATEST_FOLDER is not None:
    _DATA_PREFIX = str(_LATEST_FOLDER) + "/"
    print(f"Using projections from: {_LATEST_FOLDER.name}")
else:
    _DATA_PREFIX = str(_DATA_DIR) + "/"
    print("No pulled_YYYYMMDD folders found — using data/ directly")

# The `use_adjusted` flag selects the projection system: True -> Steamer,
# False -> ATC. Both load RAW rest-of-season feeds. The preseason playing-time
# adjustment (data_prep/playing_time.py) is intentionally NOT applied: RoS
# projections already reflect current-season playing time, and the adjustment's
# full-season historical baselines are meaningless against remaining-season PA/IP.
USE_ADJUSTED_PROJECTIONS = PROJECTIONS_CONFIG["use_adjusted"]

STEAMER_HITTERS_PATH = _DATA_PREFIX + "fangraphs-steamer-projections-hitters_ros.csv"
STEAMER_PITCHERS_PATH = _DATA_PREFIX + "fangraphs-steamer-projections-pitchers_ros.csv"
ATC_HITTERS_PATH = _DATA_PREFIX + "fangraphs-atc-projections-hitters_ros.csv"
ATC_PITCHERS_PATH = _DATA_PREFIX + "fangraphs-atc-projections-pitchers_ros.csv"

if USE_ADJUSTED_PROJECTIONS:
    HITTER_PROJ_PATH = STEAMER_HITTERS_PATH
    PITCHER_PROJ_PATH = STEAMER_PITCHERS_PATH
else:
    HITTER_PROJ_PATH = ATC_HITTERS_PATH
    PITCHER_PROJ_PATH = ATC_PITCHERS_PATH

FANTRAX_COOKIES = FANTRAX_CONFIG["cookies"]
