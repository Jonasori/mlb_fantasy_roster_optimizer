"""Playing time adjustment module.

This standalone module adjusts ATC projections for playing time bias.
All projection systems systematically overproject playing time.
This module corrects that using historical data.

Usage:
    python -m optimizer.playing_time.main
"""

from .adjust import apply_adjustments
from .load import load_ages, load_atc_projections, load_historical_actuals

__all__ = [
    "apply_adjustments",
    "load_ages",
    "load_atc_projections",
    "load_historical_actuals",
]
