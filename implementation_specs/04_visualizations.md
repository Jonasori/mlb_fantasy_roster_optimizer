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

## Cross-References

**Depends on:**
- [00_agent_guidelines.md](00_agent_guidelines.md) — code style, return figures (no `plt.show()`)
- [01a_config.md](01a_config.md) — `HITTING_CATEGORIES`, `PITCHING_CATEGORIES`, `NEGATIVE_CATEGORIES`

**Used by:**
- [05_notebook_integration.md](05_notebook_integration.md) — all plot functions called from notebook
- [06_streamlit_dashboard.md](06_streamlit_dashboard.md) — `plot_team_dashboard()`, `plot_comparison_dashboard()`

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
    team_names: dict[int, str] | None = None,
) -> plt.Figure:
    """
    Radar chart comparing all teams across all 10 categories.
    
    Display:
        - One polygon per team (7 total)
        - My team: thick solid line (2.5), filled with transparency, DRAWN LAST (on top)
        - Opponents: thin lines (1.0), muted colors, alpha=0.5, DRAWN FIRST (behind)
        - Legend showing team names with "My Team" listed first
    
    Args:
        my_totals: My team's category totals
        opponent_totals: Dict mapping opponent ID to their totals
        title: Chart title
        team_names: Optional dict mapping opponent ID to team name for legend
                    (if None, uses "Opponent 1", "Opponent 2", etc.)
    
    Visual Design:
        - Figure size: 8x8 inches (high resolution, display size controlled by Streamlit)
        - My Team color: #2E86AB (blue)
        - Opponent colors: muted palette (light red, light green, light purple, etc.)
        - Radial bounds: [-0.5, 1.1] for cleaner center and visual buffer
        - Reference circles drawn at r=0 and r=1 (black lines, alpha=0.5)
        - Y-tick labels only at [0, 0.5, 1] (within the valid range)
        - Y-axis label: "League Percentile" (labelpad=30)
        - Legend placed outside plot (bbox_to_anchor=(1.02, 1.0)), fontsize 10
        - Category labels at fontsize 11
        - Title at fontsize 14, bold
    
    Category Sorting:
        Uses sort_categories_for_radar() - single source of truth.
        - Hitting: descending (best first, clockwise from top)
        - Pitching: ascending (worst first, so best ends adjacent to hitting's best)
        This consolidates the team's strengths visually.
    
    Normalization:
        Convert each category to percentile rank among the 7 teams.
        This puts all categories on [0, 1] scale (displayed on [-0.5, 1.1] axis).
        For NEGATIVE_CATEGORIES (ERA, WHIP), flip so better = higher on chart.
    
    Drawing Order (critical for visibility):
        1. Draw opponents FIRST (they appear behind)
        2. Draw "My Team" LAST with zorder=10 (always on top)
    
    Returns:
        Figure with radar chart.
    """
    # Use shared sorting logic
    categories = sort_categories_for_radar(my_totals, opponent_totals)
    # ... rest of implementation
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

## Combined Dashboard Visualizations

These functions create multi-panel figures for use in the Streamlit dashboard. They combine multiple visualizations into a single figure for consistent layout and sizing.

### Team Dashboard (3-Panel)

```python
def plot_team_dashboard(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    optimal_roster_names: list[str],
    projections: pd.DataFrame,
    team_names: dict[int, str] | None = None,
) -> plt.Figure:
    """
    Combined 3-panel visualization for team performance dashboard.
    
    Used on the My Team page to show all visualizations in one figure.
    
    Panel 1 (left): Radar chart showing league percentile across categories
        - Uses sort_categories_for_radar() for ordering
        - Opponents drawn first (faded), My Team on top
        - Reference circles at r=0 and r=1
        - Y-ticks at [0, 0.5, 1], label "League Percentile"
    
    Panel 2 (center): Win/Loss heatmap vs each opponent
        - Rows: opponent team names (truncated if >20 chars)
        - Columns: all 10 categories
        - Green = winning, Red = losing
        - Annotated with margin values
    
    Panel 3 (right): Roster composition (horizontal bar chart)
        - Positions on Y-axis, counts on X-axis
        - Required (gray) vs Actual (green) bars
        - Summary text with hitter/pitcher counts and bounds
    
    Layout:
        - Figure size: 28x8 inches (wide format)
        - Width ratios: [4, 5, 2.5]
        - Uses constrained_layout=True for automatic spacing
    
    Returns:
        Figure with 3 panels.
    """
```

### Comparison Dashboard (Before/After)

```python
def plot_comparison_dashboard(
    before_totals: dict[str, float],
    after_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    team_names: dict[int, str] | None = None,
    title: str = "Trade/Roster Change Analysis",
) -> plt.Figure:
    """
    Combined 3-panel visualization comparing before/after a roster change.
    
    Used in Trade Builder and Free Agents simulator to show impact analysis.
    
    Panel 1 (left): Before/After radar chart
        - Before: blue solid line with fill
        - After: green dashed line with fill
        - Opponents: faded gray in background
        - Reference circles at r=0 and r=1
        - Normalizes against league baseline (excludes after from baseline)
    
    Panel 2 (center): Win/Loss heatmap for AFTER state
        - Shows projected standings after the change
        - Same format as team dashboard heatmap
    
    Panel 3 (right): Category-by-category delta bars
        - Horizontal bars showing change in each category
        - Green = improvement, Red = decline
        - For ERA/WHIP, sign is flipped (lower = better = green)
        - Summary showing total gains/losses count
    
    Layout:
        - Figure size: 28x8 inches (wide format)
        - Width ratios: [4, 5, 2.5]
        - Uses constrained_layout=True for automatic spacing
        - Title displayed via fig.suptitle()
    
    Returns:
        Figure with 3 panels.
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
    Diverging bar chart showing roster changes sorted by EWA (Expected Wins Added).
    
    Bar length is proportional to EWA. Names include SGP values.
    
    Args:
        added_df: DataFrame with columns: Name, Position, EWA, SGP
                  (sorted by EWA descending - from compute_roster_change_values)
        dropped_df: DataFrame with columns: Name, Position, EWA, SGP
                    (sorted by EWA descending, i.e., least harmful to drop first)
    
    Layout:
        - Two side-by-side panels: DROP (left) and ADD (right)
        - TOP of chart = first priority (what to do first)
        - DROP panel: TOP = safest to drop (EWA closest to 0)
        - ADD panel: TOP = most valuable to add (highest positive EWA)
        - Bar length proportional to |EWA|
        - Labels: "POS Name (SGP: X.X)"
    
    ⚠️ CHART ORDERING: Since matplotlib barh puts index 0 at the bottom,
    you must REVERSE the DataFrames before plotting so that the first
    priority item appears at the TOP of the visual chart.
    
    ```python
    # Reverse for display so TOP of chart = first priority
    dropped_display = dropped_df.iloc[::-1]
    added_display = added_df.iloc[::-1]
    ```
    
    Styling:
        - Red (#E74C3C) for drops, Green (#2ECC71) for adds
        - Vertical zero line in each panel
        - Column headers: "DROP (N)" and "ADD (N)"
        - Title: "Waiver Priority List (sorted by Expected Wins Added)"
    
    Returns:
        Figure with two-panel bar chart.
    
    Note:
        This is called from the notebook AFTER compute_roster_change_values()
        computes the EWA and SGP DataFrames.
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

## Position Sensitivity Visualizations

These visualizations work with the outputs from `compute_position_sensitivity()` and `compute_percentile_sensitivity()` in roster_optimizer.py. See [02b_position_sensitivity.md](02b_position_sensitivity.md) for the underlying analysis.

### Position Sensitivity Dashboard

```python
def plot_position_sensitivity_dashboard(
    ewa_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    slot_data: dict[str, pd.DataFrame],
) -> plt.Figure:
    """
    Combined 4-panel visualization for position sensitivity analysis.

    Panel 1 (top-left): EWA per SGP by position
        - Horizontal bar chart
        - Shows which positions give most "bang for buck"
        - Sorted by ewa_per_sgp ascending (best at top)
        - Color: green if positive, red if negative

    Panel 2 (top-right): EWA from best available FA
        - Horizontal bar chart
        - For each position: EWA if you upgrade to best available player
        - Derived from ewa_df where candidate_rank == 1
        - Color: green if positive (upgrade available), red if negative

    Panel 3 (bottom-left): SGP vs EWA scatter
        - X-axis: sgp_delta (candidate SGP - my worst SGP)
        - Y-axis: EWA
        - One point per (slot, candidate) combination
        - Color by slot type
        - Slope shows position sensitivity

    Panel 4 (bottom-right): Position scarcity curves
        - Line chart showing SGP vs rank for each position
        - X-axis: Rank (1 = best available)
        - Y-axis: SGP
        - Circles mark your current players
        - Shows where talent drops off at each position

    Args:
        ewa_df: DataFrame from compute_position_sensitivity with columns:
                slot, candidate, candidate_sgp, sgp_delta, ewa, candidate_rank
        sensitivity_df: DataFrame with columns:
                slot, ewa_per_sgp, better_fas_count, best_fa_ewa
        slot_data: Dict mapping slot name to eligible players DataFrame

    Layout:
        - Figure size: 14x10 inches
        - 2x2 grid with plt.subplots(2, 2)
        - Use constrained_layout=True

    Returns:
        Figure with 4 panels.
    """
```

### Percentile EWA Curves

```python
def plot_percentile_ewa_curves(
    pctl_ewa_df: pd.DataFrame,
    slots: list[str] | None = None,
) -> plt.Figure:
    """
    Plot Expected Wins Added vs Percentile for each position.

    Shows how much EWA changes as you upgrade to higher-percentile players.
    The vertical line shows your current worst player's percentile.

    Args:
        pctl_ewa_df: DataFrame from compute_percentile_sensitivity with columns:
                     slot, target_pctl, ewa, my_worst_avail_pctl
        slots: List of slots to include (default: C, SS, OF, SP, RP, 2B)

    Layout:
        - Subplot grid: ceil(n_slots / 3) rows × 3 columns
        - Figure size: 5*n_cols × 4*n_rows

    Each subplot:
        - X-axis: Percentile (45 to 100)
        - Y-axis: Expected Wins Added
        - Blue line with markers: EWA at each percentile
        - Red dashed vertical line: my current player's percentile (among available)
        - Annotation: "+5%ile → {EWA}" showing marginal gain
        - Zero line: horizontal black line at y=0

    Key insight:
        If your player is at 99th percentile among available, most percentile
        targets will show NEGATIVE EWA (downgrade). The curve reveals where
        meaningful upgrades exist.

    Returns:
        Figure with subplots for each position.
    """
```

### Position Distributions

```python
def plot_position_distributions(
    slot_data: dict[str, pd.DataFrame],
    slots: list[str] | None = None,
) -> plt.Figure:
    """
    Boxplots showing SGP distribution at each position with my players marked.

    Args:
        slot_data: Dict mapping slot name to eligible players DataFrame
                   with columns: Name, SGP, status
        slots: Optional list of slots to show (default: all hitting + pitching)

    Layout:
        - Two rows: Hitting positions (top), Pitching positions (bottom)
        - Boxplot for each position showing SGP distribution of AVAILABLE players
        - Red dots overlay showing MY rostered players at each position

    Implementation:
        For each slot:
            1. Get available players (status == 'available')
            2. Create boxplot of their SGP values
            3. Overlay scatter of my_roster players as red dots

    Visual elements:
        - Boxplot shows quartiles and outliers for available player pool
        - Red dots show where your players rank
        - Dots above the box = you have strong players
        - Dots in/below the box = upgrade opportunity

    Returns:
        Figure with boxplot distributions.
    """
```

### Upgrade Opportunities

```python
def plot_upgrade_opportunities(
    slot_data: dict[str, pd.DataFrame],
) -> plt.Figure:
    """
    Horizontal bar chart showing SGP gap between best FA and my worst player.

    This directly visualizes: "How much better is the best available player
    than my worst player at each position?"

    Args:
        slot_data: Dict mapping slot name to eligible players DataFrame

    For each position:
        - my_worst_sgp = min(SGP) among my_roster players
        - best_fa_sgp = max(SGP) among available players
        - gap = best_fa_sgp - my_worst_sgp

    Layout:
        - Horizontal bar chart
        - Positions sorted by gap (largest opportunity at top)
        - Positive gap = green (upgrade available)
        - Negative gap = red (you have better than best FA)

    Labels:
        - Bar labels: "{my_worst_name} → {best_fa_name}"
        - X-axis: "SGP Gap (Best FA - My Worst)"

    Implementation:
        Use explicit .min() and .max() on SGP column, not iloc which
        depends on sort order.

    Returns:
        Figure with horizontal bar chart.
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


def sort_categories_for_radar(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    categories: list[str] | None = None,
) -> list[str]:
    """
    Sort categories for radar chart display.
    
    SINGLE SOURCE OF TRUTH for category sorting in radar charts.
    Used by both plot_team_radar() and dashboard/components.py radar_chart_with_overlay().
    
    Sorting Logic:
        - Hitting: sorted DESCENDING (best first, clockwise from top)
        - Pitching: sorted ASCENDING (worst first, so best ends adjacent to hitting's best)
    
    This makes the "center of mass" consolidated - the team's best hitting stat 
    appears at the top-left, and best pitching stat appears at the bottom-left,
    so maxima meet in the middle of the chart.
    
    Args:
        my_totals: My team's category totals
        opponent_totals: Dict mapping opponent ID to their totals
        categories: Optional list to sort (defaults to ALL_CATEGORIES)
    
    Returns:
        Sorted list of category names
    
    Example:
        If my team is best at R, HR (hitting) and W, K (pitching),
        the order might be: [R, HR, RBI, SB, OPS, ERA, WHIP, SV, K, W]
        
        Hitting goes clockwise from 12 o'clock: best → worst
        Pitching continues clockwise: worst → best (so W ends up at ~4 o'clock,
        adjacent to R at 12 o'clock when the circle wraps)
    """
    if categories is None:
        categories = ALL_CATEGORIES

    def sort_key(cat):
        all_vals = [my_totals[cat]] + [opp[cat] for opp in opponent_totals.values()]
        min_val, max_val = min(all_vals), max(all_vals)
        if max_val > min_val:
            norm = (my_totals[cat] - min_val) / (max_val - min_val)
            return 1 - norm if cat in NEGATIVE_CATEGORIES else norm
        return 0.5

    sorted_hitting = sorted(
        [c for c in categories if c in HITTING_CATEGORIES], key=sort_key, reverse=True
    )
    sorted_pitching = sorted(
        [c for c in categories if c in PITCHING_CATEGORIES], key=sort_key, reverse=False
    )

    return sorted_hitting + sorted_pitching
```

---

## Validation Checklist

- [ ] All functions return `plt.Figure` objects
- [ ] No `plt.show()` calls anywhere
- [ ] Player names stripped using `strip_name_suffix()` imported from data_loader
- [ ] NEGATIVE_CATEGORIES handled correctly (ERA, WHIP)
- [ ] Consistent color scheme across visualizations
- [ ] Legends included where needed
- [ ] Axis labels clear and descriptive
- [ ] Title on every plot
- [ ] `plot_roster_changes` uses new signature (added_df, dropped_df) with EWA/SGP
- [ ] `sort_categories_for_radar()` is the SINGLE source of truth for radar chart category ordering
- [ ] Radar charts sort hitting DESCENDING and pitching ASCENDING (maxima meet in middle)
- [ ] Radar charts use muted colors for opponents, bold color for My Team
- [ ] Radar charts support `team_names` parameter for showing real team names in legend
- [ ] Radar charts draw opponents FIRST, then My Team LAST (on top with zorder)
- [ ] Radar chart radial bounds: [-0.5, 1.1] for visual buffer and clean center
- [ ] Radar charts have reference circles at r=0 and r=1
- [ ] Radar charts have Y-ticks only at [0, 0.5, 1] with "League Percentile" label
- [ ] Radar chart figure size: 8x8 inches (high resolution for crisp display)
- [ ] `radar_chart_with_overlay` normalizes against LEAGUE only (excludes after_totals from baseline)
- [ ] `plot_team_dashboard` combines radar, heatmap, and roster chart in one 28x8 figure
- [ ] `plot_comparison_dashboard` shows before/after with radar, heatmap, and delta bars
- [ ] Combined dashboard functions use `constrained_layout=True` for proper spacing

### Position Sensitivity Visualizations
- [ ] `plot_position_sensitivity_dashboard` creates 4-panel figure (14x10 inches)
- [ ] Panel 4 (scarcity curves) marks my players with circles
- [ ] `plot_percentile_ewa_curves` shows vertical line at my_worst_avail_pctl
- [ ] `plot_percentile_ewa_curves` x-axis range: 45 to 100
- [ ] `plot_position_distributions` shows boxplots for available players only
- [ ] `plot_position_distributions` overlays my players as red dots
- [ ] `plot_upgrade_opportunities` uses explicit .min()/.max() not iloc
- [ ] All position sensitivity functions handle empty slot data gracefully
