"""FanGraphs + Fantrax + MLB API pipeline producing the silver `players` table."""

from .build import (
    build_silver_table,
    load_fangraphs,
    merge_fantrax,
    merge_mlb_ages,
)
from .config import (
    DATA_DIR,
    FANTRAX_LEAGUE_ID,
    FANTRAX_TEAM_IDS,
    HITTER_PROJ_PATH,
    MY_TEAM_NAME,
    PITCHER_PROJ_PATH,
    SILVER_TABLE_DEFAULT_PATH,
    find_latest_projection_folder,
    load_config,
)
from .fantrax_api import (
    FANTRAX_NAME_CORRECTIONS,
    create_session,
    fetch_all_fantrax_data,
)
from .names import normalize_name, strip_name_suffix
from .playing_time import adjust_projections
from .silver_io import write_silver_table

__all__ = [
    "DATA_DIR",
    "FANTRAX_LEAGUE_ID",
    "FANTRAX_NAME_CORRECTIONS",
    "FANTRAX_TEAM_IDS",
    "HITTER_PROJ_PATH",
    "MY_TEAM_NAME",
    "PITCHER_PROJ_PATH",
    "SILVER_TABLE_DEFAULT_PATH",
    "adjust_projections",
    "build_silver_table",
    "create_session",
    "fetch_all_fantrax_data",
    "find_latest_projection_folder",
    "load_config",
    "load_fangraphs",
    "merge_fantrax",
    "merge_mlb_ages",
    "normalize_name",
    "strip_name_suffix",
    "write_silver_table",
]
