"""
Configuration for data prep: Fantrax credentials and league identity.

Loads the shared config.json from the repo root.

Projection FILE paths used to live here (latest-pulled-folder discovery, a
`use_adjusted` flag selecting Steamer vs ATC, four hardcoded CSV paths). The
raw layer replaced all of it: snapshots are addressed as
(source, date) via `raw_io`, so "the newest Steamer pull" is
`read_latest_raw("projections/steamer")` and needs no config.
"""

import json
from pathlib import Path

from .raw_io import DATA_DIR, REPO_ROOT

__all__ = [
    "DATA_DIR",
    "FANTRAX_COOKIES",
    "FANTRAX_LEAGUE_ID",
    "FANTRAX_TEAM_IDS",
    "LEAGUE",
    "MY_TEAM_NAME",
    "REPO_ROOT",
    "load_config",
]


def load_config(config_path: Path | str | None = None) -> dict:
    """Load configuration from JSON.

    Args:
        config_path: Path to config.json. Defaults to the repo-root file.

    Returns:
        Full config dict (caller reads the league / fantrax sections).
    """
    if config_path is not None:
        path = Path(config_path)
        assert path.exists(), f"Config file not found: {path}"
    else:
        path = REPO_ROOT / "config.json"
        assert path.exists(), (
            f"No config.json at repo root ({path}). "
            "Pass config_path= to load_config() or add config.json."
        )

    with open(path) as f:
        config = json.load(f)

    assert "league" in config, "Config must have 'league' section"
    assert "fantrax" in config, "Config must have 'fantrax' section"
    assert "cookies" in config["fantrax"], "Config must have 'fantrax.cookies' section"

    return config


_CONFIG = load_config()

LEAGUE = _CONFIG["league"]
FANTRAX_COOKIES = _CONFIG["fantrax"]["cookies"]

FANTRAX_LEAGUE_ID = LEAGUE["fantrax_league_id"]
MY_TEAM_NAME = LEAGUE["my_team_name"]
FANTRAX_TEAM_IDS = LEAGUE["fantrax_team_ids"]

assert MY_TEAM_NAME in FANTRAX_TEAM_IDS, (
    f"my_team_name '{MY_TEAM_NAME}' not found in fantrax_team_ids"
)

_NUM_TEAMS = len(FANTRAX_TEAM_IDS)
assert _NUM_TEAMS == 7, f"Data prep Fantrax helpers expect 7 teams, got {_NUM_TEAMS}"
