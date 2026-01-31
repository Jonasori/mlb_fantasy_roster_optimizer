"""
Visualization functions for the roster optimizer.

All functions return matplotlib.Figure objects. NEVER call plt.show() -
the marimo notebook handles display.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .data_loader import (
    ALL_CATEGORIES,
    HITTING_CATEGORIES,
    NEGATIVE_CATEGORIES,
    PITCHING_CATEGORIES,
    strip_name_suffix,
)

# =============================================================================
# STYLING
# =============================================================================

plt.style.use("seaborn-v0_8-whitegrid")

TEAM_COLORS = {
    "me": "#2E86AB",
    "opponent": "#A23B72",
}
WIN_COLOR = "#2ECC71"
LOSS_COLOR = "#E74C3C"
NEUTRAL_COLOR = "#95A5A6"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def _format_stat(value: float, category: str) -> str:
    """Format a stat value for display."""
    if category in {"ERA", "WHIP", "OPS"}:
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

    Hitting categories are sorted descending (best first, going clockwise).
    Pitching categories are sorted ascending (worst first, going clockwise).
    This makes the maxima of hitting and pitching meet in the middle of the chart.

    Args:
        my_totals: My team's category totals
        opponent_totals: Dict mapping opponent ID to their totals
        categories: Optional list of categories to sort (defaults to ALL_CATEGORIES)

    Returns:
        Sorted list of category names
    """
    if categories is None:
        categories = ALL_CATEGORIES

    # Compute normalized score (higher = better for my team)
    def sort_key(cat):
        all_vals = [my_totals[cat]] + [opp[cat] for opp in opponent_totals.values()]
        min_val, max_val = min(all_vals), max(all_vals)
        if max_val > min_val:
            norm = (my_totals[cat] - min_val) / (max_val - min_val)
            return 1 - norm if cat in NEGATIVE_CATEGORIES else norm
        return 0.5

    # Hitting: sort descending (best first, clockwise from top)
    sorted_hitting = sorted(
        [c for c in categories if c in HITTING_CATEGORIES], key=sort_key, reverse=True
    )

    # Pitching: sort ASCENDING (worst first, so best ends up adjacent to hitting's best)
    sorted_pitching = sorted(
        [c for c in categories if c in PITCHING_CATEGORIES], key=sort_key, reverse=False
    )

    return sorted_hitting + sorted_pitching


# =============================================================================
# TEAM COMPARISON VISUALIZATIONS
# =============================================================================


def plot_team_radar(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    title: str = "Team Comparison Across Categories",
    team_names: dict[int, str] | None = None,
) -> plt.Figure:
    """
    Radar chart comparing all teams across all 10 categories.

    Categories are sorted so hitting and pitching maxima meet:
    - Hitting: sorted descending (best first, clockwise from top)
    - Pitching: sorted ascending (worst first, so best ends adjacent to hitting's best)

    Args:
        my_totals: My team's category totals
        opponent_totals: Dict mapping opponent ID to their totals
        title: Chart title
        team_names: Optional dict mapping opponent ID to team name for legend
    """
    # Use shared sorting logic
    categories = sort_categories_for_radar(my_totals, opponent_totals)
    n_categories = len(categories)

    # Compute angles for radar chart
    angles = np.linspace(0, 2 * np.pi, n_categories, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    # Collect opponent teams first (so My Team is drawn last, on top)
    opponent_teams = []
    for opp_id, totals in opponent_totals.items():
        if team_names and opp_id in team_names:
            name = team_names[opp_id]
        else:
            name = f"Opponent {opp_id}"
        opponent_teams.append((name, totals))

    # Color palette - muted colors for opponents
    opponent_colors = [
        "#E57373",  # Light red
        "#81C784",  # Light green
        "#BA68C8",  # Light purple
        "#FFB74D",  # Light orange
        "#4DB6AC",  # Light teal
        "#F06292",  # Light pink
        "#A1887F",  # Light brown
        "#90A4AE",  # Light blue-gray
        "#FF8A65",  # Light deep orange
    ]

    # High-resolution figure (display size controlled by Streamlit)
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Helper to compute percentile for a team
    def get_percentile_values(totals):
        values = []
        for cat in categories:
            # Get all values for this category (league-wide)
            all_values = [my_totals[cat]] + [
                opponent_totals[o][cat] for o in opponent_totals
            ]

            # Compute percentile rank
            if cat in NEGATIVE_CATEGORIES:
                # Lower is better - flip
                rank = sum(1 for v in all_values if v < totals[cat]) / (
                    len(all_values) - 1
                )
                pct = 1 - rank
            else:
                rank = sum(1 for v in all_values if v < totals[cat]) / (
                    len(all_values) - 1
                )
                pct = rank

            values.append(pct)
        return values + values[:1]  # Close polygon

    # Draw opponents FIRST (so they're behind My Team)
    for idx, (team_name, totals) in enumerate(opponent_teams):
        values = get_percentile_values(totals)
        color = opponent_colors[idx % len(opponent_colors)]

        ax.plot(
            angles,
            values,
            "-",
            linewidth=0.8,
            label=team_name,
            alpha=0.5,
            color=color,
        )

    # Draw My Team LAST (on top)
    my_values = get_percentile_values(my_totals)
    ax.plot(
        angles,
        my_values,
        "o-",
        linewidth=2.5,
        label="My Team",
        color="#2E86AB",
        zorder=10,  # Ensure on top
    )
    ax.fill(angles, my_values, alpha=0.25, color="#2E86AB", zorder=9)

    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(-0.5, 1.1)  # Buffer for cleaner center and edge

    # Draw reference circles at r=0 and r=1
    theta_circle = np.linspace(0, 2 * np.pi, 100)
    ax.plot(theta_circle, [0] * 100, "k-", linewidth=1.0, alpha=0.5)
    ax.plot(theta_circle, [1] * 100, "k-", linewidth=1.0, alpha=0.5)

    # Set tick labels only within [0, 1] range
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0", "0.5", "1"], fontsize=8)

    # Add radial axis label
    ax.set_ylabel("League Percentile", labelpad=35, fontsize=9)

    ax.set_title(title, size=14, fontweight="bold", pad=15)

    # Legend outside plot - reorder to put My Team first
    handles, labels = ax.get_legend_handles_labels()
    # Move My Team to front
    my_team_idx = labels.index("My Team")
    handles = (
        [handles[my_team_idx]] + handles[:my_team_idx] + handles[my_team_idx + 1 :]
    )
    labels = [labels[my_team_idx]] + labels[:my_team_idx] + labels[my_team_idx + 1 :]

    ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=10,
        framealpha=0.9,
    )

    plt.tight_layout()
    return fig


def plot_category_margins(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
) -> plt.Figure:
    """
    Grouped bar chart showing my margin over each opponent in each category.
    """
    categories = ALL_CATEGORIES
    n_categories = len(categories)
    n_opponents = len(opponent_totals)

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(n_categories)
    width = 0.12

    for i, (opp_id, opp_totals) in enumerate(sorted(opponent_totals.items())):
        margins = []
        colors = []

        for cat in categories:
            if cat in NEGATIVE_CATEGORIES:
                margin = opp_totals[cat] - my_totals[cat]  # Positive = I win
            else:
                margin = my_totals[cat] - opp_totals[cat]

            margins.append(margin)
            colors.append(WIN_COLOR if margin > 0 else LOSS_COLOR)

        offset = (i - n_opponents / 2 + 0.5) * width
        bars = ax.bar(x + offset, margins, width, label=f"vs Opp {opp_id}", alpha=0.7)

        for bar, color in zip(bars, colors):
            bar.set_color(color)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Category")
    ax.set_ylabel("Margin (positive = winning)")
    ax.set_title("Category Margins vs Each Opponent", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    return fig


def plot_win_matrix(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
) -> plt.Figure:
    """
    Heatmap showing win/loss for each opponent-category pair.
    """
    categories = ALL_CATEGORIES
    opponents = sorted(opponent_totals.keys())

    # Build matrix
    data = []
    for opp_id in opponents:
        row = []
        for cat in categories:
            if cat in NEGATIVE_CATEGORIES:
                margin = opponent_totals[opp_id][cat] - my_totals[cat]
            else:
                margin = my_totals[cat] - opponent_totals[opp_id][cat]
            row.append(margin)
        data.append(row)

    df = pd.DataFrame(data, index=[f"Opp {o}" for o in opponents], columns=categories)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Use diverging colormap
    vmax = max(abs(df.min().min()), abs(df.max().max()))

    sns.heatmap(
        df,
        ax=ax,
        cmap="RdYlGn",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        annot=True,
        fmt=".1f",
        cbar_kws={"label": "Margin"},
    )

    ax.set_title("Win/Loss Matrix (positive margin = winning)", fontweight="bold")

    plt.tight_layout()
    return fig


# =============================================================================
# COMBINED TEAM DASHBOARD VISUALIZATION
# =============================================================================


def plot_team_dashboard(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    optimal_roster_names: list[str],
    projections: pd.DataFrame,
    team_names: dict[int, str] | None = None,
) -> plt.Figure:
    """
    Combined 3-panel visualization for team performance dashboard.

    Panel 1: Radar chart showing league percentile across categories
    Panel 2: Win/Loss heatmap vs each opponent
    Panel 3: Roster composition (horizontal bar chart)

    Args:
        my_totals: Dict of category -> total for my team
        opponent_totals: Dict of opponent_id -> {category -> total}
        optimal_roster_names: List of player names on roster
        projections: Full projections DataFrame
        team_names: Optional dict mapping opponent_id -> team name

    Returns:
        matplotlib Figure with 3 panels
    """
    from .data_loader import (
        HITTING_SLOTS,
        MAX_HITTERS,
        MAX_PITCHERS,
        MIN_HITTERS,
        MIN_PITCHERS,
        PITCHING_SLOTS,
    )

    # Create wide figure with constrained layout for automatic spacing
    fig = plt.figure(figsize=(28, 8), constrained_layout=True)

    # Use GridSpec with explicit spacing
    gs = fig.add_gridspec(1, 3, width_ratios=[4, 5, 2.5])

    ax1 = fig.add_subplot(gs[0], polar=True)  # Radar chart
    ax2 = fig.add_subplot(gs[1])  # Heatmap
    ax3 = fig.add_subplot(gs[2])  # Bar chart

    # =========================================================================
    # PANEL 1: Radar Chart (Category Comparison)
    # =========================================================================
    categories = sort_categories_for_radar(my_totals, opponent_totals, ALL_CATEGORIES)
    n_categories = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_categories, endpoint=False).tolist()
    angles += angles[:1]

    # Collect all values for percentile calculation
    all_team_totals = [my_totals] + list(opponent_totals.values())

    def get_percentile_values(totals):
        values = []
        for cat in categories:
            all_values = [t[cat] for t in all_team_totals]
            if cat in NEGATIVE_CATEGORIES:
                rank = sum(1 for v in all_values if v < totals[cat]) / (
                    len(all_values) - 1
                )
                pct = 1 - rank
            else:
                rank = sum(1 for v in all_values if v < totals[cat]) / (
                    len(all_values) - 1
                )
                pct = rank
            values.append(pct)
        return values + values[:1]

    # Colors for opponents
    opponent_colors = ["#E8A87C", "#C38D9E", "#41B3A3", "#E27D60", "#85DCB8", "#659DBD"]

    # Prepare opponent teams with names (truncated for legend)
    opponent_teams = []
    for opp_id, opp_totals in opponent_totals.items():
        if team_names:
            full_name = team_names.get(opp_id, f"Opp {opp_id}")
            # Truncate for legend readability
            name = full_name[:12] + "..." if len(full_name) > 15 else full_name
        else:
            name = f"Opp {opp_id}"
        opponent_teams.append((name, opp_totals))

    # Draw opponents first (behind My Team)
    for idx, (team_name, totals) in enumerate(opponent_teams):
        values = get_percentile_values(totals)
        color = opponent_colors[idx % len(opponent_colors)]
        ax1.plot(angles, values, "-", linewidth=1.0, alpha=0.5, color=color)

    # Draw My Team last (on top)
    my_values = get_percentile_values(my_totals)
    ax1.plot(
        angles,
        my_values,
        "o-",
        linewidth=2.5,
        label="My Team",
        color="#2E86AB",
        zorder=10,
    )
    ax1.fill(angles, my_values, alpha=0.25, color="#2E86AB", zorder=9)

    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories, fontsize=11)
    ax1.set_ylim(-0.5, 1.1)

    # Reference circles and ticks
    theta_circle = np.linspace(0, 2 * np.pi, 100)
    ax1.plot(theta_circle, [0] * 100, "k-", linewidth=1.0, alpha=0.5)
    ax1.plot(theta_circle, [1] * 100, "k-", linewidth=1.0, alpha=0.5)
    ax1.set_yticks([0, 0.5, 1])
    ax1.set_yticklabels(["0", "0.5", "1"], fontsize=9)
    ax1.set_ylabel("League Percentile", labelpad=30, fontsize=10)
    ax1.set_title(
        "Category Comparison\n(vs League)", fontweight="bold", fontsize=12, pad=10
    )

    # =========================================================================
    # PANEL 2: Win/Loss Heatmap
    # =========================================================================
    opponents = sorted(opponent_totals.keys())
    heatmap_data = []
    for opp_id in opponents:
        row = []
        for cat in ALL_CATEGORIES:
            if cat in NEGATIVE_CATEGORIES:
                margin = opponent_totals[opp_id][cat] - my_totals[cat]
            else:
                margin = my_totals[cat] - opponent_totals[opp_id][cat]
            row.append(margin)
        heatmap_data.append(row)

    # Use team names if available (truncated for display)
    row_labels = []
    for opp_id in opponents:
        if team_names and opp_id in team_names:
            full_name = team_names[opp_id]
            name = full_name[:18] + "..." if len(full_name) > 20 else full_name
            row_labels.append(name)
        else:
            row_labels.append(f"Opp {opp_id}")

    df_heatmap = pd.DataFrame(heatmap_data, index=row_labels, columns=ALL_CATEGORIES)
    vmax = max(abs(df_heatmap.min().min()), abs(df_heatmap.max().max()))

    sns.heatmap(
        df_heatmap,
        ax=ax2,
        cmap="RdYlGn",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        annot=True,
        fmt=".0f",
        annot_kws={"fontsize": 9},
        cbar_kws={"label": "Margin", "shrink": 0.7},
    )
    ax2.set_title(
        "Win/Loss Matrix\n(positive = winning)", fontweight="bold", fontsize=12
    )
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    ax2.tick_params(axis="x", labelsize=10)
    ax2.tick_params(axis="y", labelsize=9)

    # =========================================================================
    # PANEL 3: Roster Composition (Horizontal Bar Chart)
    # =========================================================================
    roster_df = projections[projections["Name"].isin(optimal_roster_names)]
    position_counts = roster_df["Position"].value_counts()
    all_slots = {**HITTING_SLOTS, **PITCHING_SLOTS}

    positions = list(all_slots.keys())
    required = [all_slots[p] for p in positions]
    actual = [position_counts.get(p, 0) for p in positions]

    y = np.arange(len(positions))
    height = 0.35

    ax3.barh(
        y - height / 2,
        required,
        height,
        label="Required",
        color=NEUTRAL_COLOR,
        alpha=0.7,
    )
    ax3.barh(y + height / 2, actual, height, label="Actual", color=WIN_COLOR, alpha=0.7)

    ax3.set_yticks(y)
    ax3.set_yticklabels(positions, fontsize=10)
    ax3.set_xlabel("Count", fontsize=10)
    ax3.set_title("Roster Composition", fontweight="bold", fontsize=12)
    ax3.legend(loc="lower right", fontsize=9)
    ax3.invert_yaxis()  # Top position at top

    # Add composition summary
    n_hitters = (roster_df["player_type"] == "hitter").sum()
    n_pitchers = (roster_df["player_type"] == "pitcher").sum()
    ax3.text(
        0.95,
        0.05,
        f"Hitters: {n_hitters} ({MIN_HITTERS}-{MAX_HITTERS})\nPitchers: {n_pitchers} ({MIN_PITCHERS}-{MAX_PITCHERS})",
        transform=ax3.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
    )

    return fig


# =============================================================================
# PLAYER CONTRIBUTION VISUALIZATIONS
# =============================================================================


def plot_comparison_dashboard(
    before_totals: dict[str, float],
    after_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    team_names: dict[int, str] | None = None,
    title: str = "Trade/Roster Change Analysis",
) -> plt.Figure:
    """
    Combined 3-panel visualization comparing before/after a roster change.

    Panel 1: Radar chart showing Before vs After (league percentile)
    Panel 2: Win/Loss heatmap for AFTER state
    Panel 3: Category-by-category delta bars

    Args:
        before_totals: Dict of category -> total before change
        after_totals: Dict of category -> total after change
        opponent_totals: Dict of opponent_id -> {category -> total}
        team_names: Optional dict mapping opponent_id -> team name
        title: Title for the figure

    Returns:
        matplotlib Figure with 3 panels
    """
    # Create wide figure with constrained layout
    fig = plt.figure(figsize=(28, 8), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[4, 5, 2.5])

    ax1 = fig.add_subplot(gs[0], polar=True)  # Radar chart
    ax2 = fig.add_subplot(gs[1])  # Heatmap
    ax3 = fig.add_subplot(gs[2])  # Delta bars

    # =========================================================================
    # PANEL 1: Before/After Radar Chart
    # =========================================================================
    categories = sort_categories_for_radar(
        before_totals, opponent_totals, ALL_CATEGORIES
    )
    n_categories = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_categories, endpoint=False).tolist()
    angles += angles[:1]

    # Collect all values for percentile calculation (league baseline)
    all_team_totals = [before_totals] + list(opponent_totals.values())

    def get_percentile_values(totals):
        values = []
        for cat in categories:
            all_values = [t[cat] for t in all_team_totals]
            if cat in NEGATIVE_CATEGORIES:
                rank = sum(1 for v in all_values if v < totals[cat]) / (
                    len(all_values) - 1
                )
                pct = 1 - rank
            else:
                rank = sum(1 for v in all_values if v < totals[cat]) / (
                    len(all_values) - 1
                )
                pct = rank
            # Clamp in case after is outside league range
            pct = max(0, min(1, pct))
            values.append(pct)
        return values + values[:1]

    # Draw opponents faintly in background
    for opp_totals in opponent_totals.values():
        values = get_percentile_values(opp_totals)
        ax1.plot(angles, values, "-", linewidth=0.5, alpha=0.2, color="gray")

    # Draw Before (blue solid)
    before_values = get_percentile_values(before_totals)
    ax1.plot(
        angles,
        before_values,
        "o-",
        linewidth=2.5,
        label="Before",
        color="#2E86AB",
        zorder=10,
    )
    ax1.fill(angles, before_values, alpha=0.15, color="#2E86AB", zorder=9)

    # Draw After (green dashed)
    after_values = get_percentile_values(after_totals)
    ax1.plot(
        angles,
        after_values,
        "o--",
        linewidth=2.5,
        label="After",
        color="#2ECC71",
        zorder=11,
    )
    ax1.fill(angles, after_values, alpha=0.15, color="#2ECC71", zorder=8)

    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories, fontsize=11)
    ax1.set_ylim(-0.5, 1.1)

    # Reference circles and ticks
    theta_circle = np.linspace(0, 2 * np.pi, 100)
    ax1.plot(theta_circle, [0] * 100, "k-", linewidth=1.0, alpha=0.5)
    ax1.plot(theta_circle, [1] * 100, "k-", linewidth=1.0, alpha=0.5)
    ax1.set_yticks([0, 0.5, 1])
    ax1.set_yticklabels(["0", "0.5", "1"], fontsize=9)
    ax1.set_ylabel("League Percentile", labelpad=30, fontsize=10)
    ax1.set_title(
        "Before vs After\n(League Percentile)", fontweight="bold", fontsize=12, pad=10
    )
    ax1.legend(loc="upper right", fontsize=10)

    # =========================================================================
    # PANEL 2: Win/Loss Heatmap (After state)
    # =========================================================================
    opponents = sorted(opponent_totals.keys())
    heatmap_data = []
    for opp_id in opponents:
        row = []
        for cat in ALL_CATEGORIES:
            if cat in NEGATIVE_CATEGORIES:
                margin = opponent_totals[opp_id][cat] - after_totals[cat]
            else:
                margin = after_totals[cat] - opponent_totals[opp_id][cat]
            row.append(margin)
        heatmap_data.append(row)

    # Use team names if available
    row_labels = []
    for opp_id in opponents:
        if team_names and opp_id in team_names:
            full_name = team_names[opp_id]
            name = full_name[:18] + "..." if len(full_name) > 20 else full_name
            row_labels.append(name)
        else:
            row_labels.append(f"Opp {opp_id}")

    df_heatmap = pd.DataFrame(heatmap_data, index=row_labels, columns=ALL_CATEGORIES)
    vmax = max(abs(df_heatmap.min().min()), abs(df_heatmap.max().max()))

    sns.heatmap(
        df_heatmap,
        ax=ax2,
        cmap="RdYlGn",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        annot=True,
        fmt=".0f",
        annot_kws={"fontsize": 9},
        cbar_kws={"label": "Margin", "shrink": 0.7},
    )
    ax2.set_title(
        "Win/Loss Matrix (After)\n(positive = winning)", fontweight="bold", fontsize=12
    )
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    ax2.tick_params(axis="x", labelsize=10)
    ax2.tick_params(axis="y", labelsize=9)

    # =========================================================================
    # PANEL 3: Category Deltas (horizontal bar)
    # =========================================================================
    deltas = []
    colors = []
    for cat in ALL_CATEGORIES:
        if cat in NEGATIVE_CATEGORIES:
            # For ERA/WHIP, lower is better so negative delta is good
            delta = before_totals[cat] - after_totals[cat]
        else:
            delta = after_totals[cat] - before_totals[cat]
        deltas.append(delta)
        colors.append(
            WIN_COLOR if delta > 0 else LOSS_COLOR if delta < 0 else NEUTRAL_COLOR
        )

    y = np.arange(len(ALL_CATEGORIES))
    ax3.barh(y, deltas, color=colors, alpha=0.8)
    ax3.set_yticks(y)
    ax3.set_yticklabels(ALL_CATEGORIES, fontsize=10)
    ax3.axvline(x=0, color="black", linewidth=0.8)
    ax3.set_xlabel("Change (+ is better)", fontsize=10)
    ax3.set_title("Category Impact", fontweight="bold", fontsize=12)
    ax3.invert_yaxis()

    # Add summary annotation
    gains = sum(1 for d in deltas if d > 0)
    losses = sum(1 for d in deltas if d < 0)
    ax3.text(
        0.95,
        0.05,
        f"Gains: {gains}\nLosses: {losses}",
        transform=ax3.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
    )

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    return fig


def plot_category_contributions(
    roster_names: list[str],
    projections: pd.DataFrame,
    category: str,
) -> plt.Figure:
    """
    Horizontal bar chart showing each player's contribution to one category.
    """
    roster_df = projections[projections["Name"].isin(roster_names)].copy()

    # Filter to appropriate player type
    if category in HITTING_CATEGORIES:
        roster_df = roster_df[roster_df["player_type"] == "hitter"]
    else:
        roster_df = roster_df[roster_df["player_type"] == "pitcher"]

    # For ratio stats, compute impact relative to team average
    if category in ["OPS", "ERA", "WHIP"]:
        if category == "OPS":
            weight_col = "PA"
        else:
            weight_col = "IP"

        total_weight = roster_df[weight_col].sum()
        if total_weight > 0:
            team_avg = (
                roster_df[weight_col] * roster_df[category]
            ).sum() / total_weight
        else:
            team_avg = roster_df[category].mean()

        roster_df = roster_df.copy()

        if category in NEGATIVE_CATEGORIES:
            roster_df["contribution"] = roster_df[weight_col] * (
                team_avg - roster_df[category]
            )
        else:
            roster_df["contribution"] = roster_df[weight_col] * (
                roster_df[category] - team_avg
            )
    else:
        roster_df["contribution"] = roster_df[category]

    # Sort by contribution
    roster_df = roster_df.sort_values("contribution", ascending=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(6, len(roster_df) * 0.4)))

    colors = [WIN_COLOR if v >= 0 else LOSS_COLOR for v in roster_df["contribution"]]

    ax.barh(
        [strip_name_suffix(n) for n in roster_df["Name"]],
        roster_df["contribution"],
        color=colors,
        alpha=0.7,
    )

    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.set_xlabel(f"Contribution to {category}")
    ax.set_title(f"Player Contributions to {category}", fontweight="bold")

    plt.tight_layout()
    return fig


def plot_player_contribution_radar(
    roster_names: list[str],
    projections: pd.DataFrame,
    player_type: str = "hitter",
    top_n: int = 12,
) -> plt.Figure:
    """
    Radar chart showing each player's contributions across all relevant categories.
    """
    roster_df = projections[projections["Name"].isin(roster_names)].copy()
    roster_df = roster_df[roster_df["player_type"] == player_type]

    if player_type == "hitter":
        categories = ["R", "HR", "RBI", "SB", "OPS"]
    else:
        categories = ["W", "SV", "K", "ERA", "WHIP"]

    # Normalize values to [0, 1]
    for cat in categories:
        min_val = roster_df[cat].min()
        max_val = roster_df[cat].max()

        if cat in NEGATIVE_CATEGORIES:
            roster_df[f"{cat}_norm"] = 1 - (roster_df[cat] - min_val) / (
                max_val - min_val + 0.001
            )
        else:
            roster_df[f"{cat}_norm"] = (roster_df[cat] - min_val) / (
                max_val - min_val + 0.001
            )

    # Sort by total normalized value
    roster_df["total_norm"] = roster_df[[f"{c}_norm" for c in categories]].sum(axis=1)
    roster_df = roster_df.nlargest(top_n, "total_norm")

    # Create radar chart
    n_categories = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_categories, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    colors = plt.cm.tab20(np.linspace(0, 1, len(roster_df)))

    for i, (_, row) in enumerate(roster_df.iterrows()):
        values = [row[f"{c}_norm"] for c in categories]
        values += values[:1]

        ax.plot(
            angles,
            values,
            "o-",
            linewidth=1.5,
            label=strip_name_suffix(row["Name"]),
            color=colors[i],
        )
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title(
        f"{player_type.title()} Contributions (normalized)", fontweight="bold", pad=20
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=8)

    plt.tight_layout()
    return fig


# =============================================================================
# ROSTER CHANGE VISUALIZATIONS
# =============================================================================


def plot_roster_changes(
    added_df: pd.DataFrame,
    dropped_df: pd.DataFrame,
) -> plt.Figure:
    """
    Diverging bar chart showing roster changes sorted by EWA (Expected Wins Added).
    """
    n_adds = len(added_df)
    n_drops = len(dropped_df)
    n_rows = max(n_adds, n_drops)

    if n_rows == 0:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.5, "No roster changes", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    fig, (ax_drop, ax_add) = plt.subplots(1, 2, figsize=(14, max(6, n_rows * 0.5)))

    # DROPS (left side) - sorted by EWA descending, so least harmful first
    # Reverse for display so TOP of chart = first priority (least harmful to drop)
    if not dropped_df.empty:
        dropped_display = dropped_df.iloc[::-1]  # Reverse for top-to-bottom display
        y_pos = range(len(dropped_display))
        bars = ax_drop.barh(
            y_pos,
            dropped_display["EWA"],  # Expected wins added (raw number)
            color=LOSS_COLOR,
            alpha=0.7,
        )

        labels = [
            f"{row['Position']} {strip_name_suffix(row['Name'])} (SGP: {row['SGP']:.1f})"
            for _, row in dropped_display.iterrows()
        ]
        ax_drop.set_yticks(y_pos)
        ax_drop.set_yticklabels(labels)
        ax_drop.set_xlabel("EWA (Expected Wins)")
        ax_drop.set_title(f"DROP ({n_drops})", fontweight="bold", color=LOSS_COLOR)
        ax_drop.axvline(x=0, color="black", linewidth=0.5)
        ax_drop.invert_xaxis()  # Negative values go left
    else:
        ax_drop.text(0.5, 0.5, "No drops", ha="center", va="center")
        ax_drop.axis("off")

    # ADDS (right side) - sorted by EWA descending, so most valuable first
    # Reverse for display so TOP of chart = first priority (most valuable to add)
    if not added_df.empty:
        added_display = added_df.iloc[::-1]  # Reverse for top-to-bottom display
        y_pos = range(len(added_display))
        bars = ax_add.barh(
            y_pos,
            added_display["EWA"],
            color=WIN_COLOR,
            alpha=0.7,
        )

        labels = [
            f"{row['Position']} {strip_name_suffix(row['Name'])} (SGP: {row['SGP']:.1f})"
            for _, row in added_display.iterrows()
        ]
        ax_add.set_yticks(y_pos)
        ax_add.set_yticklabels(labels)
        ax_add.set_xlabel("EWA (Expected Wins)")
        ax_add.set_title(f"ADD ({n_adds})", fontweight="bold", color=WIN_COLOR)
        ax_add.axvline(x=0, color="black", linewidth=0.5)
    else:
        ax_add.text(0.5, 0.5, "No adds", ha="center", va="center")
        ax_add.axis("off")

    fig.suptitle(
        "Waiver Priority List (sorted by Expected Wins Added)",
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    return fig


def plot_trade_impact(
    trade_eval: dict,
    my_totals_before: dict[str, float],
    my_totals_after: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
) -> plt.Figure:
    """
    Visualize the impact of a proposed trade.
    """
    fig = plt.figure(figsize=(14, 10))

    # 2x2 grid
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    # Top-left: Trade summary
    ax1.axis("off")
    summary_text = f"""
TRADE SUMMARY

Send: {", ".join(strip_name_suffix(p) for p in trade_eval["send_players"])}
Receive: {", ".join(strip_name_suffix(p) for p in trade_eval["receive_players"])}

Expected Wins: {trade_eval["ew_before"]:.1f} → {trade_eval["ew_after"]:.1f}
Change: {trade_eval["ewa"]:+.2f} EWA

SGP: {trade_eval["delta_generic"]:+.1f}
Fairness: {"Fair" if trade_eval["is_fair"] else "Unfair"}

RECOMMENDATION: {trade_eval["recommendation"]}
"""
    ax1.text(0.1, 0.5, summary_text, fontsize=11, family="monospace", va="center")
    ax1.set_title("Trade Summary", fontweight="bold")

    # Top-right: Category changes
    categories = ALL_CATEGORIES
    changes = [trade_eval["category_impact"].get(c, 0) for c in categories]
    colors = [WIN_COLOR if c > 0 else LOSS_COLOR for c in changes]

    ax2.barh(categories, changes, color=colors, alpha=0.7)
    ax2.axvline(x=0, color="black", linewidth=0.5)
    ax2.set_xlabel("Change")
    ax2.set_title("Category Impact", fontweight="bold")

    # Bottom-left: Win probability gauges
    ax3.axis("off")

    # Simple text representation of before/after
    gauge_text = f"""
EXPECTED WINS (out of 60)

Before: {trade_eval["ew_before"]:.1f}
After:  {trade_eval["ew_after"]:.1f}

Change: {trade_eval["ewa"]:+.2f}
"""
    ax3.text(0.3, 0.5, gauge_text, fontsize=14, family="monospace", va="center")
    ax3.set_title("Expected Wins", fontweight="bold")

    # Bottom-right: Value comparison
    ax4.axis("off")

    send_sgp = sum(g[1] for g in trade_eval["send_generics"])
    receive_sgp = sum(g[1] for g in trade_eval["receive_generics"])

    value_text = f"""
VALUE COMPARISON

Sending: {send_sgp:.1f} dynasty SGP
Receiving: {receive_sgp:.1f} dynasty SGP

Net: {trade_eval["delta_generic"]:+.1f} dynasty SGP
"""
    ax4.text(0.3, 0.5, value_text, fontsize=14, family="monospace", va="center")
    ax4.set_title("Dynasty Value", fontweight="bold")

    plt.tight_layout()
    return fig


# =============================================================================
# SENSITIVITY ANALYSIS VISUALIZATIONS
# =============================================================================


def plot_player_sensitivity(
    sensitivity_df: pd.DataFrame,
    top_n: int = 15,
) -> plt.Figure:
    """
    Horizontal bar chart showing most impactful players.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # Top panel: Most valuable rostered players
    on_roster = sensitivity_df[sensitivity_df["on_optimal_roster"]]
    on_roster = on_roster.nsmallest(
        top_n, "objective_delta"
    )  # Most negative = most valuable

    if not on_roster.empty:
        ax1.barh(
            [strip_name_suffix(n) for n in on_roster["Name"]],
            on_roster["objective_delta"],
            color=WIN_COLOR,
            alpha=0.7,
        )
        ax1.axvline(x=0, color="black", linewidth=0.5)
        ax1.set_xlabel("Objective Delta (if removed)")
        ax1.set_title("Most Valuable Rostered Players", fontweight="bold")
    else:
        ax1.text(0.5, 0.5, "No data", ha="center", va="center")

    # Bottom panel: Best available non-rostered
    off_roster = sensitivity_df[~sensitivity_df["on_optimal_roster"]]
    off_roster = off_roster.nlargest(top_n, "objective_delta")

    if not off_roster.empty:
        ax2.barh(
            [strip_name_suffix(n) for n in off_roster["Name"]],
            off_roster["objective_delta"],
            color=NEUTRAL_COLOR,
            alpha=0.7,
        )
        ax2.axvline(x=0, color="black", linewidth=0.5)
        ax2.set_xlabel("Objective Delta (if added)")
        ax2.set_title("Best Available Non-Rostered", fontweight="bold")
    else:
        ax2.text(0.5, 0.5, "No data", ha="center", va="center")

    plt.tight_layout()
    return fig


def plot_constraint_analysis(
    optimal_roster_names: list[str],
    projections: pd.DataFrame,
) -> plt.Figure:
    """
    Visualize which roster constraints are binding.
    """
    from .data_loader import (
        HITTING_SLOTS,
        MAX_HITTERS,
        MAX_PITCHERS,
        MIN_HITTERS,
        MIN_PITCHERS,
        PITCHING_SLOTS,
    )

    roster_df = projections[projections["Name"].isin(optimal_roster_names)]

    # Count by position
    position_counts = roster_df["Position"].value_counts()

    # Slot requirements
    all_slots = {**HITTING_SLOTS, **PITCHING_SLOTS}

    fig, ax = plt.subplots(figsize=(10, 4))

    positions = list(all_slots.keys())
    required = [all_slots[p] for p in positions]
    actual = [position_counts.get(p, 0) for p in positions]

    x = np.arange(len(positions))
    width = 0.35

    ax.bar(
        x - width / 2, required, width, label="Required", color=NEUTRAL_COLOR, alpha=0.7
    )
    ax.bar(x + width / 2, actual, width, label="Actual", color=WIN_COLOR, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(positions)
    ax.set_ylabel("Count")
    ax.set_title("Position Slot Analysis", fontweight="bold")
    ax.legend()

    # Add composition info
    n_hitters = (roster_df["player_type"] == "hitter").sum()
    n_pitchers = (roster_df["player_type"] == "pitcher").sum()

    ax.text(
        0.98,
        0.98,
        f"Hitters: {n_hitters} (bounds: {MIN_HITTERS}-{MAX_HITTERS})\n"
        f"Pitchers: {n_pitchers} (bounds: {MIN_PITCHERS}-{MAX_PITCHERS})",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    return fig


# =============================================================================
# TRADE ENGINE VISUALIZATIONS
# =============================================================================


def plot_win_probability_breakdown(
    diagnostics: dict,
) -> plt.Figure:
    """
    Visualize the components of win probability calculation.
    """
    fig = plt.figure(figsize=(14, 10))

    # 2x2 grid
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    matchup_probs = diagnostics.get("matchup_probs", {})

    # Top-left: Matchup probability heatmap
    if matchup_probs:
        categories = ALL_CATEGORIES
        opponents = sorted(list(matchup_probs.get(categories[0], {}).keys()))

        data = []
        for opp in opponents:
            row = [matchup_probs.get(cat, {}).get(opp, 0.5) for cat in categories]
            data.append(row)

        df = pd.DataFrame(
            data, index=[f"Opp {o}" for o in opponents], columns=categories
        )

        sns.heatmap(
            df, ax=ax1, cmap="RdYlGn", vmin=0, vmax=1, annot=True, fmt=".2f", center=0.5
        )
        ax1.set_title("Matchup Win Probabilities", fontweight="bold")
    else:
        ax1.text(0.5, 0.5, "No matchup data", ha="center", va="center")
        ax1.axis("off")

    # Top-right: Expected wins by category
    if matchup_probs:
        cat_wins = []
        for cat in categories:
            wins = sum(matchup_probs.get(cat, {}).values())
            cat_wins.append(wins)

        ax2.barh(categories, cat_wins, color=WIN_COLOR, alpha=0.7)
        ax2.set_xlabel("Expected Wins (out of 6)")
        ax2.set_title("Expected Wins by Category", fontweight="bold")
        ax2.set_xlim(0, 6)
    else:
        ax2.text(0.5, 0.5, "No data", ha="center", va="center")
        ax2.axis("off")

    # Bottom-left: Distribution visualization
    mu_D = diagnostics.get("mu_D", 0)
    sigma_D = diagnostics.get("sigma_D", 1)

    x = np.linspace(mu_D - 4 * sigma_D, mu_D + 4 * sigma_D, 100)
    y = (1 / (sigma_D * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((x - mu_D) / sigma_D) ** 2
    )

    ax3.plot(x, y, color=TEAM_COLORS["me"], linewidth=2)
    ax3.fill_between(x[x > 0], y[x > 0], alpha=0.3, color=WIN_COLOR, label="Win Region")
    ax3.fill_between(
        x[x <= 0], y[x <= 0], alpha=0.3, color=LOSS_COLOR, label="Lose Region"
    )
    ax3.axvline(x=0, color="black", linestyle="--", linewidth=1)
    ax3.set_xlabel("Differential (my points - best opponent)")
    ax3.set_ylabel("Probability Density")
    ax3.set_title("Win Probability Distribution", fontweight="bold")
    ax3.legend()

    # Bottom-right: Summary statistics
    ax4.axis("off")

    stats_text = f"""
SUMMARY STATISTICS

Expected Wins (μ_T): {diagnostics.get("mu_T", 0):.1f}
Variance (σ_T²): {diagnostics.get("sigma_T_sq", 0):.2f}

Target Lead (μ_L): {diagnostics.get("mu_L", 0):.2f}
Target Var (σ_L²): {diagnostics.get("sigma_L_sq", 0):.2f}

Differential Mean (μ_D): {diagnostics.get("mu_D", 0):.2f}
Differential Std (σ_D): {diagnostics.get("sigma_D", 1):.2f}

Expected Roto Points: {diagnostics.get("expected_roto_points", 0)}/70
"""
    ax4.text(0.1, 0.5, stats_text, fontsize=11, family="monospace", va="center")
    ax4.set_title("Summary", fontweight="bold")

    plt.tight_layout()
    return fig


def plot_category_marginal_values(
    gradient: dict[str, dict[int, float]],
) -> plt.Figure:
    """
    Visualize marginal value of improvement in each category.
    """
    categories = ALL_CATEGORIES
    opponents = sorted(list(gradient.get(categories[0], {}).keys()))

    data = []
    for opp in opponents:
        row = [gradient.get(cat, {}).get(opp, 0) for cat in categories]
        data.append(row)

    df = pd.DataFrame(data, index=[f"Opp {o}" for o in opponents], columns=categories)

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.heatmap(df, ax=ax, cmap="YlOrRd", annot=True, fmt=".3f")
    ax.set_title("Marginal Value of Improvement (Gradient)", fontweight="bold")

    plt.tight_layout()
    return fig


# =============================================================================
# POSITION SENSITIVITY VISUALIZATIONS
# =============================================================================


def plot_position_sensitivity_dashboard(
    ewa_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    slot_data: dict[str, pd.DataFrame],
) -> plt.Figure:
    """
    Combined 4-panel visualization for position sensitivity analysis.

    Panel 1: EWA per SGP by position (which positions give most bang for buck)
    Panel 2: EWA from upgrading to best FA at each position
    Panel 3: SGP vs EWA scatter by position
    Panel 4: Position scarcity curves with my players marked

    Args:
        ewa_df: DataFrame with columns: slot, sgp_delta, ewa, candidate_rank, etc.
        sensitivity_df: DataFrame with columns: slot, ewa_per_sgp, better_fas_count, best_fa_ewa
        slot_data: Dict mapping slot name to eligible players DataFrame

    Returns:
        matplotlib Figure with 4 panels
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: EWA per SGP by position
    ax1 = axes[0, 0]
    sens_plot = sensitivity_df.sort_values("ewa_per_sgp", ascending=True)
    colors = [WIN_COLOR if w > 0 else LOSS_COLOR for w in sens_plot["ewa_per_sgp"]]
    ax1.barh(sens_plot["slot"], sens_plot["ewa_per_sgp"], color=colors, alpha=0.7)
    ax1.set_xlabel("Expected Wins Added per 1 SGP")
    ax1.set_title(
        "Position Sensitivity: EWA per SGP\n(Higher = upgrades more impactful)",
        fontweight="bold",
    )
    ax1.axvline(x=0, color="black", linewidth=0.5)
    ax1.grid(True, alpha=0.3, axis="x")

    # Plot 2: EWA from upgrading to best FA
    ax2 = axes[0, 1]
    best_fa_ewa = ewa_df[ewa_df["candidate_rank"] == 1].set_index("slot")["ewa"]
    best_fa_ewa = best_fa_ewa.sort_values()
    colors = [WIN_COLOR if w > 0 else LOSS_COLOR for w in best_fa_ewa]
    ax2.barh(best_fa_ewa.index, best_fa_ewa.values, color=colors, alpha=0.7)
    ax2.set_xlabel("Expected Wins Added")
    ax2.set_title(
        "EWA from Upgrading to Best FA\n(Replaces your worst at each position)",
        fontweight="bold",
    )
    ax2.axvline(x=0, color="black", linewidth=0.5)
    ax2.grid(True, alpha=0.3, axis="x")

    # Plot 3: SGP vs EWA scatter
    ax3 = axes[1, 0]
    colors_map = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, slot in enumerate(ewa_df["slot"].unique()):
        slot_plot = ewa_df[ewa_df["slot"] == slot]
        if not slot_plot.empty:
            ax3.scatter(
                slot_plot["sgp_delta"],
                slot_plot["ewa"],
                label=slot,
                alpha=0.7,
                s=80,
                color=colors_map[i % 10],
            )

    ax3.set_xlabel("SGP Gain (candidate - my worst)")
    ax3.set_ylabel("Expected Wins Added")
    ax3.set_title(
        "SGP vs EWA by Position\n(Slope = position sensitivity)", fontweight="bold"
    )
    ax3.axhline(y=0, color="black", linewidth=0.5)
    ax3.axvline(x=0, color="black", linewidth=0.5)
    ax3.legend(loc="best", fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Position scarcity curves
    ax4 = axes[1, 1]
    highlight_slots = ["C", "SS", "OF", "SP", "RP"]
    colors_line = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, slot in enumerate(highlight_slots):
        if slot not in slot_data:
            continue
        eligible = slot_data[slot]
        available = eligible[eligible["status"] != "opponent"].copy()
        available = available.sort_values("SGP", ascending=False).reset_index(drop=True)

        n = min(50, len(available))
        ax4.plot(
            range(1, n + 1),
            available["SGP"].head(n),
            label=slot,
            linewidth=2,
            color=colors_line[i % len(colors_line)],
        )

        # Mark my players with circles
        my_at_slot = eligible[eligible["status"] == "my_roster"]
        for _, player in my_at_slot.iterrows():
            player_in_avail = available[available["Name"] == player["Name"]]
            if not player_in_avail.empty:
                idx = player_in_avail.index[0]
                if idx < n:
                    ax4.scatter(
                        idx + 1,
                        player["SGP"],
                        marker="o",
                        s=100,
                        edgecolors="black",
                        linewidth=2,
                        zorder=5,
                        color=colors_line[i % len(colors_line)],
                    )

    ax4.set_xlabel("Rank (1 = best)")
    ax4.set_ylabel("SGP")
    ax4.set_title(
        "Position Scarcity + My Players\n(Circles = my players)", fontweight="bold"
    )
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(1, 50)

    plt.tight_layout()
    return fig


def plot_percentile_ewa_curves(
    pctl_ewa_df: pd.DataFrame,
    slots: list[str] | None = None,
) -> plt.Figure:
    """
    Plot Expected Wins Added vs EWA Percentile for each position.

    Players are ranked by EWA (team-specific value), so curves are monotonic.
    The slope tells you how valuable upgrades are at each position.
    Includes all active players (FA + opponent rosters) for trade context.

    Args:
        pctl_ewa_df: DataFrame with columns: slot, ewa_pctl, ewa, status, my_worst_ewa_pctl
        slots: List of slots to include (default: C, SS, OF, SP, RP, 2B)

    Returns:
        matplotlib Figure
    """
    if slots is None:
        slots = ["C", "SS", "OF", "SP", "RP", "2B"]

    n_slots = len(slots)
    n_cols = min(3, n_slots)
    n_rows = (n_slots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)

    for idx, slot in enumerate(slots):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        slot_pctl = pctl_ewa_df[pctl_ewa_df["slot"] == slot]
        if slot_pctl.empty:
            ax.text(
                0.5,
                0.5,
                f"No data for {slot}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        # Plot EWA vs EWA percentile (monotonic by construction)
        slot_pctl = slot_pctl.sort_values("ewa_pctl")

        # Color points by status (available vs opponent)
        available = slot_pctl[slot_pctl["status"] == "available"]
        opponent = slot_pctl[slot_pctl["status"] == "opponent"]

        # Draw the line through all points
        ax.plot(
            slot_pctl["ewa_pctl"],
            slot_pctl["ewa"],
            "b-",
            linewidth=2,
            alpha=0.5,
        )

        # Plot FA and opponent points differently
        if not available.empty:
            ax.scatter(
                available["ewa_pctl"],
                available["ewa"],
                c="green",
                s=60,
                marker="o",
                label="Available (FA)",
                zorder=3,
            )
        if not opponent.empty:
            ax.scatter(
                opponent["ewa_pctl"],
                opponent["ewa"],
                c="red",
                s=60,
                marker="^",
                label="Opponent roster",
                zorder=3,
            )

        # Mark my current worst percentile
        my_pctl = slot_pctl["my_worst_ewa_pctl"].iloc[0]
        ax.axvline(
            x=my_pctl,
            color="purple",
            linestyle="--",
            alpha=0.7,
            linewidth=2,
            label=f"My worst: {my_pctl:.0f}%ile",
        )

        # Zero line (no improvement)
        ax.axhline(y=0, color="black", linewidth=0.5)

        # Compute slope at key region (around 60-80th percentile)
        mid_range = slot_pctl[
            (slot_pctl["ewa_pctl"] >= 55) & (slot_pctl["ewa_pctl"] <= 85)
        ]
        if len(mid_range) >= 2:
            # Simple linear slope
            x_vals = mid_range["ewa_pctl"].values
            y_vals = mid_range["ewa"].values
            slope = (y_vals.max() - y_vals.min()) / (x_vals.max() - x_vals.min() + 0.01)
            ax.text(
                0.05,
                0.95,
                f"Slope: {slope:.3f} EWA/%ile",
                transform=ax.transAxes,
                fontsize=9,
                va="top",
                color="blue",
            )

        ax.set_xlabel("EWA Percentile (ranked by team value)")
        ax.set_ylabel("Expected Wins Added")
        ax.set_title(f"{slot}", fontweight="bold")
        ax.legend(loc="lower right", fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(5, 100)

    # Hide any unused subplots
    for idx in range(n_slots, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")

    plt.suptitle(
        "EWA by Percentile (Ranked by Team-Specific Value)\n"
        "Green=FA, Red=Opponent (trade targets), Purple line=My worst player",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    return fig


def plot_position_distributions(
    slot_data: dict[str, pd.DataFrame],
    slots: list[str] | None = None,
) -> plt.Figure:
    """
    Violin plots showing SGP distribution at each position with my players marked.

    Violin plots show the full distribution shape without cluttering with individual
    outlier points like boxplots do.

    Args:
        slot_data: Dict mapping slot name to eligible players DataFrame
                   with columns: Name, SGP, status (available/my_roster/opponent)
        slots: Optional list of slots to include

    Returns:
        matplotlib Figure with 2 panels (hitting and pitching)
    """
    from .data_loader import HITTING_SLOTS, PITCHING_SLOTS

    hitting_slots = slots if slots else list(HITTING_SLOTS.keys())
    pitching_slots = slots if slots else list(PITCHING_SLOTS.keys())

    # Filter to actual slots in data
    hitting_slots = [s for s in hitting_slots if s in slot_data and s in HITTING_SLOTS]
    pitching_slots = [
        s for s in pitching_slots if s in slot_data and s in PITCHING_SLOTS
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Hitting positions
    ax1 = axes[0]
    hitting_data = []
    for slot in hitting_slots:
        # Include available (FA) players only for distribution (not opponents)
        available = slot_data[slot][slot_data[slot]["status"] == "available"]
        for _, row in available.iterrows():
            hitting_data.append({"slot": slot, "sgp": row["SGP"]})

    if hitting_data:
        hitting_df = pd.DataFrame(hitting_data)

        # Violin plot with seaborn
        sns.violinplot(
            data=hitting_df,
            x="slot",
            y="sgp",
            order=hitting_slots,
            ax=ax1,
            color="lightblue",
            alpha=0.7,
            inner="quartile",  # Show quartile lines inside violin
            cut=0,  # Don't extend past data range
        )

        # Overlay my players as red dots
        for i, slot in enumerate(hitting_slots):
            my_at_slot = slot_data[slot][slot_data[slot]["status"] == "my_roster"]
            for _, row in my_at_slot.iterrows():
                ax1.scatter(
                    i,  # 0-indexed for seaborn
                    row["SGP"],
                    color="red",
                    s=120,
                    zorder=5,
                    edgecolors="black",
                    linewidth=1.5,
                )

    ax1.set_xlabel("")
    ax1.set_ylabel("SGP")
    ax1.set_title(
        "Hitting Position SGP Distributions\n(Red dots = my players)", fontweight="bold"
    )
    ax1.grid(True, alpha=0.3, axis="y")

    # Pitching positions
    ax2 = axes[1]
    pitching_data = []
    for slot in pitching_slots:
        available = slot_data[slot][slot_data[slot]["status"] == "available"]
        for _, row in available.iterrows():
            pitching_data.append({"slot": slot, "sgp": row["SGP"]})

    if pitching_data:
        pitching_df = pd.DataFrame(pitching_data)

        sns.violinplot(
            data=pitching_df,
            x="slot",
            y="sgp",
            order=pitching_slots,
            ax=ax2,
            color="lightblue",
            alpha=0.7,
            inner="quartile",
            cut=0,
        )

        for i, slot in enumerate(pitching_slots):
            my_at_slot = slot_data[slot][slot_data[slot]["status"] == "my_roster"]
            for _, row in my_at_slot.iterrows():
                ax2.scatter(
                    i,  # 0-indexed for seaborn
                    row["SGP"],
                    color="red",
                    s=120,
                    zorder=5,
                    edgecolors="black",
                    linewidth=1.5,
                )

    ax2.set_xlabel("")
    ax2.set_ylabel("SGP")
    ax2.set_title(
        "Pitching Position SGP Distributions\n(Red dots = my players)",
        fontweight="bold",
    )
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


def plot_upgrade_opportunities(
    slot_data: dict[str, pd.DataFrame],
) -> plt.Figure:
    """
    Horizontal bar chart showing SGP gap between best FA and my worst player at each position.

    Positive gap = there's a better free agent available than my worst player.

    Args:
        slot_data: Dict mapping slot name to eligible players DataFrame
                   with columns: Name, SGP, status

    Returns:
        matplotlib Figure
    """
    opportunities = []

    for slot, eligible in slot_data.items():
        my_at_slot = eligible[eligible["status"] == "my_roster"]
        available = eligible[eligible["status"] == "available"]

        if my_at_slot.empty or available.empty:
            continue

        # Explicitly get worst (min SGP) and best (max SGP) - don't rely on sort order
        my_worst_sgp = my_at_slot["SGP"].min()
        best_fa_sgp = available["SGP"].max()

        opportunities.append(
            {
                "slot": slot,
                "my_worst_sgp": my_worst_sgp,
                "best_fa_sgp": best_fa_sgp,
                "gap": best_fa_sgp - my_worst_sgp,
            }
        )

    if not opportunities:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No upgrade opportunities found", ha="center", va="center")
        ax.axis("off")
        return fig

    opp_df = pd.DataFrame(opportunities).sort_values("gap", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [WIN_COLOR if g > 0 else LOSS_COLOR for g in opp_df["gap"]]
    ax.barh(opp_df["slot"], opp_df["gap"], color=colors, alpha=0.7)
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.set_xlabel("SGP Gap (Best FA - My Worst)")
    ax.set_ylabel("Position Slot")
    ax.set_title(
        "Upgrade Opportunity by Position\n(Green = clear upgrade available)",
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    return fig


def plot_player_value_scatter(
    player_values: pd.DataFrame,
    my_roster_names: set[str],
) -> plt.Figure:
    """
    Scatter plot of player values: generic vs contextual.
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Separate by roster status
    on_roster = player_values[player_values["on_my_roster"]]
    off_roster = player_values[~player_values["on_my_roster"]]

    # Plot off-roster (hollow circles)
    hitters_off = off_roster[off_roster["player_type"] == "hitter"]
    pitchers_off = off_roster[off_roster["player_type"] == "pitcher"]

    ax.scatter(
        hitters_off["generic_value"],
        hitters_off["ewa_acquire"],
        s=50,
        facecolors="none",
        edgecolors="blue",
        alpha=0.6,
        label="Hitters (available)",
    )
    ax.scatter(
        pitchers_off["generic_value"],
        pitchers_off["ewa_acquire"],
        s=50,
        facecolors="none",
        edgecolors="red",
        alpha=0.6,
        label="Pitchers (available)",
    )

    # Plot on-roster (filled circles)
    hitters_on = on_roster[on_roster["player_type"] == "hitter"]
    pitchers_on = on_roster[on_roster["player_type"] == "pitcher"]

    ax.scatter(
        hitters_on["generic_value"],
        hitters_on["ewa_acquire"],
        s=80,
        c="blue",
        alpha=0.8,
        label="Hitters (my roster)",
    )
    ax.scatter(
        pitchers_on["generic_value"],
        pitchers_on["ewa_acquire"],
        s=80,
        c="red",
        alpha=0.8,
        label="Pitchers (my roster)",
    )

    # Add quadrant lines
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.axvline(
        x=player_values["generic_value"].median(),
        color="gray",
        linestyle="--",
        linewidth=0.5,
    )

    ax.set_xlabel("Generic Value (SGP)")
    ax.set_ylabel("Contextual Value (EWA)")
    ax.set_title("Player Value: Generic vs Contextual", fontweight="bold")
    ax.legend(loc="upper right")

    # Annotate quadrants
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    ax.text(
        xlim[1] * 0.95,
        ylim[1] * 0.95,
        "Great targets\n(high both)",
        ha="right",
        va="top",
        fontsize=9,
        alpha=0.7,
    )
    ax.text(
        xlim[0] * 1.05,
        ylim[1] * 0.95,
        "Hidden gems\n(high context)",
        ha="left",
        va="top",
        fontsize=9,
        alpha=0.7,
    )
    ax.text(
        xlim[1] * 0.95,
        ylim[0] * 1.05,
        "Overvalued\n(high generic)",
        ha="right",
        va="bottom",
        fontsize=9,
        alpha=0.7,
    )
    ax.text(
        xlim[0] * 1.05,
        ylim[0] * 1.05,
        "Avoid\n(low both)",
        ha="left",
        va="bottom",
        fontsize=9,
        alpha=0.7,
    )

    plt.tight_layout()
    return fig
