# Visualizations

## Overview

This module provides all plotting functions for the roster optimizer. Visualizations help users understand:
- Team strength relative to opponents
- Category-by-category breakdown
- Player contributions
- Trade impact analysis

**Module:** `optimizer/visualizations.py`

**Critical Rule:** All functions return `matplotlib.Figure` objects. **Never call `plt.show()`** — the marimo notebook handles display.

---

## Imports and Setup

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Iterable

from .data_loader import (
    HITTING_CATEGORIES,
    PITCHING_CATEGORIES,
    ALL_CATEGORIES,
    NEGATIVE_CATEGORIES,
)

# Consistent styling
plt.style.use('seaborn-v0_8-whitegrid')
TEAM_COLORS = {
    'me': '#2E86AB',
    'opponent': '#A23B72',
}
WIN_COLOR = '#2ECC71'
LOSS_COLOR = '#E74C3C'
```

---

## Team Comparison Visualizations

### Radar Chart

```python
def plot_team_radar(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    title: str = "Team Comparison Across Categories",
) -> plt.Figure:
    """
    Radar chart comparing all 7 teams across all 10 categories.
    
    Display:
        - One polygon per team (7 total)
        - My team: thick solid line, distinct color
        - Opponents: thin dashed lines, muted colors
        - Legend identifying each team
    
    Normalization:
        Convert each category to percentile rank among the 7 teams.
        This puts all categories on [0, 1] scale.
        For NEGATIVE_CATEGORIES (ERA, WHIP), flip so better = higher on chart.
    
    Implementation:
        1. Collect all 7 teams' totals
        2. For each category, compute percentile rank (0-1)
        3. For ERA/WHIP, use 1 - percentile (lower is better)
        4. Create polar plot with 10 spokes
        5. Plot each team as a filled polygon
        6. Close polygons by appending first point
    
    Returns:
        Figure with radar chart.
    """
```

### Category Margins Bar Chart

```python
def plot_category_margins(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
) -> plt.Figure:
    """
    Grouped bar chart showing my margin over each opponent in each category.
    
    X-axis: 10 categories
    Bars: 6 bars per category (one per opponent)
    Colors: Green if positive (I win), red if negative (I lose)
    
    For NEGATIVE_CATEGORIES (ERA, WHIP), flip sign:
        margin = opponent_value - my_value
        (so positive still means I win)
    
    Returns:
        Figure with grouped bar chart.
    """
```

### Win Matrix Heatmap

```python
def plot_win_matrix(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
) -> plt.Figure:
    """
    Heatmap showing win/loss for each opponent-category pair.
    
    Rows: 6 opponents
    Columns: 10 categories
    Cell color: Green gradient if I win, red gradient if I lose
    Cell text: Margin (formatted appropriately for category type)
    
    For NEGATIVE_CATEGORIES, flip sign for display consistency.
    
    Implementation:
        Use seaborn heatmap or matplotlib imshow.
        Add text annotations in each cell.
        Color scale: diverging around 0.
    
    Returns:
        Figure with heatmap.
    """
```

---

## Player Contribution Visualizations

### Single Category Breakdown

```python
def plot_category_contributions(
    roster_names: list[str],
    projections: pd.DataFrame,
    category: str,
) -> plt.Figure:
    """
    Horizontal bar chart showing each player's contribution to one category.
    
    For counting stats (R, HR, RBI, SB, W, SV, K):
        Contribution = player's raw value
        All bars positive, sorted by magnitude
    
    For ratio stats (OPS, ERA, WHIP):
        Contribution = "impact" on team ratio
        Impact = weight * (player_value - team_average)
        For ERA/WHIP, flip sign so positive = helps team
        
        This shows who's helping vs hurting the team ratio.
        Bars can be positive (helps) or negative (hurts).
    
    Args:
        roster_names: Players on the roster (with -H/-P suffix)
        projections: Projections DataFrame
        category: One of ALL_CATEGORIES
    
    Returns:
        Figure with horizontal bar chart.
        Player names displayed without suffix.
    """
```

### Player Contribution Radar

```python
def plot_player_contribution_radar(
    roster_names: list[str],
    projections: pd.DataFrame,
    player_type: str = "hitter",
    top_n: int = 12,
) -> plt.Figure:
    """
    Radar chart showing each player's contributions across all relevant categories.
    
    Each player is a polygon. Shows all category contributions at once.
    
    Args:
        roster_names: Players on the roster
        projections: Projections DataFrame
        player_type: "hitter" or "pitcher"
        top_n: Maximum players to show (avoid clutter)
    
    Categories:
        Hitters: R, HR, RBI, SB, OPS (5 axes)
        Pitchers: W, SV, K, ERA, WHIP (5 axes)
    
    Normalization:
        Values normalized to [0, 1] based on min/max within roster.
        For ERA/WHIP, flip so positive = good.
        Players closer to edge are better in that category.
    
    Implementation:
        1. Filter by player_type
        2. Compute contributions for each category
        3. Normalize to [0, 1]
        4. Select top N by total contribution
        5. Plot using polar projection
        6. Different color per player
        7. Strip -H/-P suffix for legend
    
    Returns:
        Figure with radar chart.
    """
```

---

## Roster Change Visualizations

### Roster Diff

```python
def plot_roster_changes(
    added_df: pd.DataFrame,
    dropped_df: pd.DataFrame,
) -> plt.Figure:
    """
    Diverging bar chart showing roster changes sorted by WPA.
    
    Players are sorted by priority (highest WPA magnitude at top).
    Bar length is proportional to WPA. Names include SGP and WPA values.
    
    Args:
        added_df: DataFrame with columns: Name, Position, WPA, SGP
                  (sorted by WPA descending - from compute_roster_change_values)
        dropped_df: DataFrame with columns: Name, Position, WPA, SGP
                    (sorted by WPA descending, i.e., least harmful to drop first)
    
    Layout:
        - Single figure with diverging bars from center vertical axis
        - Left side: DROPPED players (red bars extending left)
        - Right side: ADDED players (green bars extending right)
        - Top row = highest priority (most WPA impact)
        - Bar length proportional to |WPA|
        - Labels: "POS Name (SGP: X.X, WPA: +Y.Y%)"
    
    Styling:
        - Red (#E74C3C) for drops, Green (#2ECC71) for adds
        - Vertical center line
        - Column headers: "DROP (N)" and "ADD (N)"
        - Title: "Waiver Priority List (sorted by Win Probability Added)"
    
    Returns:
        Figure with diverging bar chart.
    
    Note:
        This is called from the notebook AFTER compute_roster_change_values()
        computes the WPA and SGP DataFrames. The old signature taking
        (old_roster_names, new_roster_names, projections) is deprecated.
    """
```

### Trade Impact Visualization

```python
def plot_trade_impact(
    trade_eval: dict,
    my_totals_before: dict[str, float],
    my_totals_after: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
) -> plt.Figure:
    """
    Visualize the impact of a proposed trade.
    
    Layout: 2x2 grid
    
    Top-left: Trade summary text
        - Players sent/received
        - Win probability change
        - Recommendation
    
    Top-right: Category changes
        - Horizontal bars showing Δ for each category
        - Green/red coloring
    
    Bottom-left: Win probability before/after
        - Two donut charts or gauges
        - Shows V_before and V_after
    
    Bottom-right: Matchup changes
        - Mini heatmap showing which matchups flip
        - Before: win/lose indicators
        - After: win/lose indicators
        - Highlight flips
    
    Returns:
        Figure with trade impact visualization.
    """
```

---

## Sensitivity Analysis Visualizations

### Player Sensitivity

```python
def plot_player_sensitivity(
    sensitivity_df: pd.DataFrame,
    top_n: int = 15,
) -> plt.Figure:
    """
    Horizontal bar chart showing most impactful players.
    
    Two panels, stacked vertically:
    
    Top panel: "Most Valuable Rostered Players"
        Players ON optimal roster, sorted by objective_delta (most negative first).
        These are players whose removal hurts the most.
        Bars extend left (negative = losing them hurts).
        Show top N.
    
    Bottom panel: "Best Available Non-Rostered"
        Players NOT on roster, sorted by objective_delta.
        Usually most are zero.
        Interesting if some are close substitutes.
        Show top N by |objective_delta|.
    
    Bar labels: player name and position.
    Color by magnitude.
    
    Returns:
        Figure with two-panel sensitivity visualization.
    """
```

### Constraint Analysis

```python
def plot_constraint_analysis(
    optimal_roster_names: list[str],
    projections: pd.DataFrame,
) -> plt.Figure:
    """
    Visualize which roster constraints are binding.
    
    Args:
        optimal_roster_names: Players on the optimal roster (with -H/-P suffix)
        projections: Combined projections (to get Position for each player)
    
    Bar chart showing:
        - For each position slot: rostered count vs required count
        - For hitter/pitcher bounds: current count vs min/max
    
    Color coding:
        - Red: at minimum (binding, might want more)
        - Yellow: at maximum (binding, can't add more)
        - Green: between min and max (slack available)
    
    Returns:
        Figure with constraint visualization.
    """
```

---

## Trade Engine Specific Visualizations

### Win Probability Breakdown

```python
def plot_win_probability_breakdown(
    diagnostics: dict,
) -> plt.Figure:
    """
    Visualize the components of win probability calculation.
    
    Layout: 2x2 grid
    
    Top-left: Matchup probability heatmap
        - 10 categories × 6 opponents
        - Color = P(win), 0 to 1
        - Text = probability
    
    Top-right: Expected wins pie chart
        - Segment per category
        - Size = contribution to μ_T
    
    Bottom-left: Distribution visualization
        - Bell curve for my fantasy point distribution
        - Bell curve for "target to beat" distribution
        - Shaded overlap = loss region
        - V shown as area to the right
    
    Bottom-right: Summary statistics
        - μ_T, σ_T
        - μ_L, σ_L
        - μ_D, σ_D
        - V
    
    Args:
        diagnostics: Output from compute_win_probability
    
    Returns:
        Figure with win probability breakdown.
    """
```

### Category Value Comparison

```python
def plot_category_marginal_values(
    gradient: dict[str, dict[int, float]],
) -> plt.Figure:
    """
    Visualize marginal value of improvement in each category.
    
    Heatmap showing:
        - Rows: categories
        - Columns: opponents
        - Color: gradient magnitude (marginal value of improving this matchup)
    
    High values indicate high-value targets for improvement.
    
    Returns:
        Figure with gradient heatmap.
    """
```

### Player Value Scatter

```python
def plot_player_value_scatter(
    player_values: pd.DataFrame,
    my_roster_names: set[str],
) -> plt.Figure:
    """
    Scatter plot of player values: generic vs contextual.
    
    X-axis: generic_value (context-free, z-score sum)
    Y-axis: delta_V_acquire (contextual, win probability impact)
    
    Point styling:
        - My roster: filled circles
        - Opponents: hollow circles
        - Color by player_type (hitter/pitcher)
        - Size by PA or IP
    
    Quadrants:
        - Top-right: High generic, high contextual (great targets)
        - Top-left: Low generic, high contextual (undervalued for me)
        - Bottom-right: High generic, low contextual (overvalued for me)
        - Bottom-left: Low generic, low contextual (avoid)
    
    Returns:
        Figure with scatter plot.
    """
```

---

## Utility Functions

```python
# Import the shared utility function - do NOT redefine it
from .data_loader import strip_name_suffix


def _format_stat(value: float, category: str) -> str:
    """Format a stat value for display."""
    if category in {'ERA', 'WHIP', 'OPS'}:
        return f"{value:.3f}"
    return f"{value:.0f}"


def _get_category_color(category: str, value: float, opponent_value: float) -> str:
    """
    Get color for a category comparison.
    
    Green if winning, red if losing.
    Handles NEGATIVE_CATEGORIES appropriately.
    """
    if category in NEGATIVE_CATEGORIES:
        return WIN_COLOR if value < opponent_value else LOSS_COLOR
    return WIN_COLOR if value > opponent_value else LOSS_COLOR
```

---

## Validation Checklist

- [ ] All functions return `plt.Figure` objects
- [ ] No `plt.show()` calls anywhere
- [ ] Player names stripped using `strip_name_suffix()` imported from data_loader
- [ ] NEGATIVE_CATEGORIES handled correctly (ERA, WHIP)
- [ ] Consistent color scheme across visualizations
- [ ] Figure sizes appropriate for notebook display
- [ ] Legends included where needed
- [ ] Axis labels clear and descriptive
- [ ] Title on every plot
- [ ] `plot_roster_changes` uses new signature (added_df, dropped_df) with WPA/SGP
