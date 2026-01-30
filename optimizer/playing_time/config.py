"""Configuration constants for playing time adjustment."""

from pathlib import Path

# Paths
MLB_STATS_DB = Path("../mlb_player_comps_dashboard/mlb_stats.db")
OPTIMIZER_DB = Path("data/optimizer.db")
ATC_HITTERS = Path("data/fangraphs-atc-projections-hitters.csv")
ATC_PITCHERS = Path("data/fangraphs-atc-projections-pitchers.csv")
OUTPUT_PATH = Path("data/adjusted_projections.csv")

# Playing time baselines
FULL_SEASON_PA = 600  # Full-time hitter
FULL_SEASON_IP_SP = 180  # Full-time starter
FULL_SEASON_IP_RP = 70  # Full-time reliever

# Age penalty thresholds
AGE_PENALTY_START_HITTER = 32
AGE_PENALTY_START_PITCHER = 33
AGE_PENALTY_PER_YEAR = 0.05  # 5% reduction per year over threshold

# Talent penalty
TALENT_PENALTY_PERCENTILE = 0.25  # Bottom quartile
TALENT_PENALTY_FACTOR = 0.85  # 15% reduction

# Blend ratio (from Zimmerman's research)
PROJECTION_WEIGHT = 0.65
ADJUSTMENT_WEIGHT = 0.35
