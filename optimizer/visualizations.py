"""
Visualizations for MLB Fantasy Roster Optimizer.

All functions return matplotlib.Figure objects.
NEVER call plt.show() — the marimo notebook handles display.
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .data_loader import (
    ALL_CATEGORIES,
    HITTING_CATEGORIES,
    HITTING_SLOTS,
    MAX_HITTERS,
    MAX_PITCHERS,
    MIN_HITTERS,
    MIN_PITCHERS,
    NEGATIVE_CATEGORIES,
    PITCHING_CATEGORIES,
    PITCHING_SLOTS,
    strip_name_suffix,
)

# Consistent styling
plt.style.use("seaborn-v0_8-whitegrid")
TEAM_COLORS = {
    "me": "#2E86AB",
    "opponent": "#A23B72",
}
WIN_COLOR = "#2ECC71"
LOSS_COLOR = "#E74C3C"
NEUTRAL_COLOR = "#95A5A6"


# === UTILITY FUNCTIONS ===


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


def _compute_percentile_rank(
    values: list[float], target_idx: int = 0, higher_is_better: bool = True
) -> float:
    """Compute percentile rank for target value among all values."""
    target = values[target_idx]
    if higher_is_better:
        rank = sum(1 for v in values if v <= target) / len(values)
    else:
        rank = sum(1 for v in values if v >= target) / len(values)
    return rank


# === TEAM COMPARISON VISUALIZATIONS ===


def plot_team_radar(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    title: str = "Team Comparison Across Categories",
) -> plt.Figure:
    """
    Radar chart comparing all 7 teams across all 10 categories.

    One polygon per team. My team uses thick solid line.
    Values normalized to percentile rank among all teams.
    """
    categories = ALL_CATEGORIES
    n_cats = len(categories)

    # Compute angles for radar
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    # Collect all team values
    all_teams = {"Me": my_totals}
    for opp_id, opp_totals in opponent_totals.items():
        all_teams[f"Opp {opp_id}"] = opp_totals

    # Compute percentile ranks for each category
    team_ranks = {}
    for team_name, totals in all_teams.items():
        ranks = []
        for cat in categories:
            all_vals = [all_teams[t][cat] for t in all_teams]
            target_idx = list(all_teams.keys()).index(team_name)
            higher_is_better = cat not in NEGATIVE_CATEGORIES
            rank = _compute_percentile_rank(all_vals, target_idx, higher_is_better)
            ranks.append(rank)
        ranks += ranks[:1]  # Close polygon
        team_ranks[team_name] = ranks

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

    # Plot each team
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_teams)))

    for i, (team_name, ranks) in enumerate(team_ranks.items()):
        if team_name == "Me":
            ax.plot(
                angles,
                ranks,
                "o-",
                linewidth=3,
                color=TEAM_COLORS["me"],
                label=team_name,
            )
            ax.fill(angles, ranks, alpha=0.25, color=TEAM_COLORS["me"])
        else:
            ax.plot(
                angles,
                ranks,
                "--",
                linewidth=1,
                color=colors[i],
                alpha=0.6,
                label=team_name,
            )

    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)

    # Set radial limits
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], size=8)

    ax.set_title(title, size=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    return fig


def plot_category_margins(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
) -> plt.Figure:
    """
    Grouped bar chart showing my margin over each opponent in each category.

    Green if positive (I win), red if negative (I lose).
    """
    categories = ALL_CATEGORIES
    n_cats = len(categories)
    n_opps = len(opponent_totals)

    # Compute margins
    margins = {}
    for cat in categories:
        margins[cat] = []
        for opp_id in sorted(opponent_totals.keys()):
            if cat in NEGATIVE_CATEGORIES:
                # Lower is better, so positive margin = opponent higher (I win)
                margin = opponent_totals[opp_id][cat] - my_totals[cat]
            else:
                margin = my_totals[cat] - opponent_totals[opp_id][cat]
            margins[cat].append(margin)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(n_cats)
    width = 0.12

    for i, opp_id in enumerate(sorted(opponent_totals.keys())):
        opp_margins = [margins[cat][i] for cat in categories]
        colors = [WIN_COLOR if m > 0 else LOSS_COLOR for m in opp_margins]

        offset = (i - n_opps / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            opp_margins,
            width,
            label=f"vs Opp {opp_id}",
            color=colors,
            alpha=0.7,
        )

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Category")
    ax.set_ylabel("Margin (positive = winning)")
    ax.set_title("My Margin vs Each Opponent by Category")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc="upper right")

    plt.tight_layout()
    return fig


def plot_win_matrix(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
) -> plt.Figure:
    """
    Heatmap showing win/loss for each opponent-category pair.

    Rows: opponents, Columns: categories
    Color: Green if I win, red if I lose
    """
    categories = ALL_CATEGORIES
    opp_ids = sorted(opponent_totals.keys())

    # Build matrix
    matrix = np.zeros((len(opp_ids), len(categories)))
    annotations = []

    for i, opp_id in enumerate(opp_ids):
        row_annot = []
        for j, cat in enumerate(categories):
            my_val = my_totals[cat]
            opp_val = opponent_totals[opp_id][cat]

            if cat in NEGATIVE_CATEGORIES:
                margin = opp_val - my_val  # Positive = I win
            else:
                margin = my_val - opp_val

            matrix[i, j] = margin
            row_annot.append(
                _format_stat(margin, cat) if abs(margin) < 100 else f"{margin:.0f}"
            )
        annotations.append(row_annot)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # Create custom colormap
    cmap = sns.diverging_palette(10, 130, as_cmap=True)

    # Normalize matrix for coloring
    max_abs = max(abs(matrix.min()), abs(matrix.max()))

    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=-max_abs, vmax=max_abs)

    # Add annotations
    for i in range(len(opp_ids)):
        for j in range(len(categories)):
            color = "white" if abs(matrix[i, j]) > max_abs * 0.5 else "black"
            ax.text(
                j,
                i,
                annotations[i][j],
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )

    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_yticks(np.arange(len(opp_ids)))
    ax.set_yticklabels([f"Opp {oid}" for oid in opp_ids])

    ax.set_xlabel("Category")
    ax.set_ylabel("Opponent")
    ax.set_title("Win/Loss Matrix (positive margin = I win)")

    plt.colorbar(im, ax=ax, label="Margin")
    plt.tight_layout()
    return fig


# === PLAYER CONTRIBUTION VISUALIZATIONS ===


def plot_category_contributions(
    roster_names: list[str],
    projections: pd.DataFrame,
    category: str,
) -> plt.Figure:
    """
    Horizontal bar chart showing each player's contribution to one category.

    For counting stats: contribution = player's raw value
    For ratio stats: contribution = impact on team ratio
    """
    roster_df = projections[projections["Name"].isin(roster_names)].copy()

    # Filter by player type
    if category in HITTING_CATEGORIES:
        roster_df = roster_df[roster_df["player_type"] == "hitter"]
    else:
        roster_df = roster_df[roster_df["player_type"] == "pitcher"]

    if category in ["OPS", "ERA", "WHIP"]:
        # Ratio stat: compute impact on team average
        if category == "OPS":
            weight_col = "PA"
        else:
            weight_col = "IP"

        total_weight = roster_df[weight_col].sum()
        team_avg = (roster_df[weight_col] * roster_df[category]).sum() / total_weight

        roster_df["contribution"] = roster_df[weight_col] * (
            roster_df[category] - team_avg
        )

        if category in NEGATIVE_CATEGORIES:
            roster_df["contribution"] = -roster_df[
                "contribution"
            ]  # Flip so positive = helps
    else:
        roster_df["contribution"] = roster_df[category]

    # Sort by contribution
    roster_df = roster_df.sort_values("contribution", ascending=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(6, len(roster_df) * 0.4)))

    colors = [WIN_COLOR if c > 0 else LOSS_COLOR for c in roster_df["contribution"]]

    y_pos = np.arange(len(roster_df))
    ax.barh(y_pos, roster_df["contribution"], color=colors, alpha=0.8)

    # Labels
    names = [strip_name_suffix(n) for n in roster_df["Name"]]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

    ax.set_xlabel(f"{category} Contribution")
    ax.set_title(f"Player Contributions to {category}")

    plt.tight_layout()
    return fig


def plot_player_contribution_radar(
    roster_names: list[str],
    projections: pd.DataFrame,
    player_type: str = "hitter",
    top_n: int = 12,
) -> plt.Figure:
    """
    Radar chart showing each player's contributions across relevant categories.
    """
    roster_df = projections[projections["Name"].isin(roster_names)].copy()
    roster_df = roster_df[roster_df["player_type"] == player_type]

    if player_type == "hitter":
        categories = ["R", "HR", "RBI", "SB", "OPS"]
    else:
        categories = ["W", "SV", "K", "ERA", "WHIP"]

    n_cats = len(categories)

    # Normalize values to [0, 1]
    normalized = pd.DataFrame()
    for cat in categories:
        if cat in NEGATIVE_CATEGORIES:
            # Lower is better, flip
            normalized[cat] = 1 - (roster_df[cat] - roster_df[cat].min()) / (
                roster_df[cat].max() - roster_df[cat].min() + 1e-10
            )
        else:
            normalized[cat] = (roster_df[cat] - roster_df[cat].min()) / (
                roster_df[cat].max() - roster_df[cat].min() + 1e-10
            )

    normalized["Name"] = roster_df["Name"].values

    # Select top N by total contribution
    normalized["total"] = normalized[categories].sum(axis=1)
    normalized = normalized.nlargest(top_n, "total")

    # Create radar
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

    colors = plt.cm.tab20(np.linspace(0, 1, len(normalized)))

    for i, (_, row) in enumerate(normalized.iterrows()):
        values = [row[cat] for cat in categories]
        values += values[:1]

        ax.plot(
            angles,
            values,
            "o-",
            linewidth=2,
            color=colors[i],
            label=strip_name_suffix(row["Name"]),
            alpha=0.7,
        )
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)

    title = "Hitter" if player_type == "hitter" else "Pitcher"
    ax.set_title(
        f"{title} Contributions by Category", size=14, fontweight="bold", pad=20
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=8)

    plt.tight_layout()
    return fig


# === ROSTER CHANGE VISUALIZATIONS ===


def plot_roster_changes(
    added_df: pd.DataFrame,
    dropped_df: pd.DataFrame,
) -> plt.Figure:
    """
    Diverging bar chart showing roster changes sorted by WPA.

    Players are sorted by priority (highest WPA magnitude at top).
    Bar length is proportional to WPA. Names include WAR and WPA values.

    Parameters
    ----------
    added_df : pd.DataFrame
        DataFrame with columns: Name, Position, WPA, WAR (sorted by WPA descending)
    dropped_df : pd.DataFrame
        DataFrame with columns: Name, Position, WPA, WAR (sorted by WPA descending,
        i.e., least harmful to drop at top)
    """
    n_added = len(added_df)
    n_dropped = len(dropped_df)
    n_rows = max(n_added, n_dropped, 1)

    fig, ax = plt.subplots(figsize=(14, max(6, n_rows * 0.45)))

    # Determine scale for bar lengths (use max absolute WPA across both)
    max_wpa = 0.0
    if n_added > 0:
        max_wpa = max(max_wpa, added_df["WPA"].abs().max())
    if n_dropped > 0:
        max_wpa = max(max_wpa, dropped_df["WPA"].abs().max())

    # Avoid division by zero
    if max_wpa < 0.001:
        max_wpa = 0.001

    # Y positions (0 at bottom, n_rows-1 at top for highest priority)
    y_positions = np.arange(n_rows)

    # Plot dropped players (left side, negative x)
    for i, (_, row) in enumerate(dropped_df.iterrows()):
        y = n_rows - 1 - i  # Top row = highest priority
        # WPA for drops is negative (loss of value), so abs() for bar length
        bar_length = abs(row["WPA"]) / max_wpa
        ax.barh(y, -bar_length, color=LOSS_COLOR, alpha=0.7, height=0.8)

        # Label on the left of center
        name = strip_name_suffix(row["Name"])
        pos = row["Position"]
        wpa = row["WPA"]
        sgp = row["SGP"]
        label = f"{pos} {name}  (SGP: {sgp:.1f}, WPA: {wpa:+.1%})"
        ax.text(-0.02, y, label, va="center", ha="right", fontsize=9)

    # Plot added players (right side, positive x)
    for i, (_, row) in enumerate(added_df.iterrows()):
        y = n_rows - 1 - i  # Top row = highest priority
        bar_length = row["WPA"] / max_wpa
        ax.barh(y, bar_length, color=WIN_COLOR, alpha=0.7, height=0.8)

        # Label on the right of center
        name = strip_name_suffix(row["Name"])
        pos = row["Position"]
        wpa = row["WPA"]
        sgp = row["SGP"]
        label = f"{pos} {name}  (SGP: {sgp:.1f}, WPA: {wpa:+.1%})"
        ax.text(0.02, y, label, va="center", ha="left", fontsize=9)

    # Styling
    ax.axvline(x=0, color="black", linewidth=1)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.5, n_rows - 0.5)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_frame_on(False)

    # Column headers
    ax.text(
        -0.5,
        n_rows + 0.3,
        f"DROP ({n_dropped})",
        fontsize=12,
        fontweight="bold",
        color=LOSS_COLOR,
        ha="center",
    )
    ax.text(
        0.5,
        n_rows + 0.3,
        f"ADD ({n_added})",
        fontsize=12,
        fontweight="bold",
        color=WIN_COLOR,
        ha="center",
    )

    plt.suptitle(
        "Waiver Priority List (sorted by Win Probability Added)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
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
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Top-left: Trade summary
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis("off")

    summary_text = f"""
Trade Summary
─────────────
Send:    {[strip_name_suffix(p) for p in trade_eval["send_players"]]}
Receive: {[strip_name_suffix(p) for p in trade_eval["receive_players"]]}

Win Probability: {trade_eval["V_before"] * 100:.1f}% → {trade_eval["V_after"] * 100:.1f}%
Change: {trade_eval["delta_V"] * 100:+.1f}%

SGP Change: {trade_eval["delta_generic"]:+.1f}
Fair: {"Yes" if trade_eval["is_fair"] else "No"}

Recommendation: {trade_eval["recommendation"]}
"""
    ax1.text(
        0.1,
        0.9,
        summary_text,
        transform=ax1.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Top-right: Category changes
    ax2 = fig.add_subplot(gs[0, 1])

    categories = ALL_CATEGORIES
    changes = [my_totals_after[c] - my_totals_before[c] for c in categories]

    # Flip sign for negative categories for display
    display_changes = []
    for cat, change in zip(categories, changes):
        if cat in NEGATIVE_CATEGORIES:
            display_changes.append(-change)  # Negative change is good
        else:
            display_changes.append(change)

    colors = [WIN_COLOR if c > 0 else LOSS_COLOR for c in display_changes]

    y_pos = np.arange(len(categories))
    ax2.barh(y_pos, display_changes, color=colors, alpha=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(categories)
    ax2.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
    ax2.set_xlabel("Change (positive = improvement)")
    ax2.set_title("Category Impact")

    # Bottom-left: Win probability gauge
    ax3 = fig.add_subplot(gs[1, 0])

    before = trade_eval["V_before"]
    after = trade_eval["V_after"]

    ax3.barh(
        [0, 1],
        [before, after],
        color=[NEUTRAL_COLOR, WIN_COLOR if after > before else LOSS_COLOR],
        alpha=0.8,
    )
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(["Before", "After"])
    ax3.set_xlim(0, 1)
    ax3.set_xlabel("Win Probability")
    ax3.set_title("Win Probability Comparison")

    for i, val in enumerate([before, after]):
        ax3.text(val + 0.02, i, f"{val * 100:.1f}%", va="center")

    # Bottom-right: Category value comparison
    ax4 = fig.add_subplot(gs[1, 1])

    x = np.arange(len(categories))
    width = 0.35

    before_vals = [my_totals_before[c] for c in categories]
    after_vals = [my_totals_after[c] for c in categories]

    # Normalize for display
    max_vals = [max(abs(b), abs(a)) for b, a in zip(before_vals, after_vals)]
    before_norm = [b / m if m > 0 else 0 for b, m in zip(before_vals, max_vals)]
    after_norm = [a / m if m > 0 else 0 for a, m in zip(after_vals, max_vals)]

    ax4.bar(
        x - width / 2,
        before_norm,
        width,
        label="Before",
        color=NEUTRAL_COLOR,
        alpha=0.7,
    )
    ax4.bar(
        x + width / 2,
        after_norm,
        width,
        label="After",
        color=TEAM_COLORS["me"],
        alpha=0.7,
    )

    ax4.set_xticks(x)
    ax4.set_xticklabels(categories, rotation=45, ha="right")
    ax4.set_ylabel("Normalized Value")
    ax4.set_title("Category Values (normalized)")
    ax4.legend()

    plt.suptitle("Trade Impact Analysis", fontsize=14, fontweight="bold")
    return fig


# === SENSITIVITY ANALYSIS VISUALIZATIONS ===


def plot_player_sensitivity(
    sensitivity_df: pd.DataFrame,
    top_n: int = 15,
) -> plt.Figure:
    """
    Horizontal bar chart showing most impactful players.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Top panel: Most valuable rostered players
    ax1 = axes[0]
    rostered = sensitivity_df[sensitivity_df["on_optimal_roster"]].copy()
    rostered = rostered.nsmallest(
        top_n, "objective_delta"
    )  # Most negative = most valuable

    if len(rostered) > 0:
        y_pos = np.arange(len(rostered))
        ax1.barh(y_pos, rostered["objective_delta"], color=LOSS_COLOR, alpha=0.8)

        names = [
            f"{row['Position']} {strip_name_suffix(row['Name'])}"
            for _, row in rostered.iterrows()
        ]
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(names)
        ax1.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

    ax1.set_xlabel("Objective Change if Removed")
    ax1.set_title("Most Valuable Rostered Players")

    # Bottom panel: Best available non-rostered
    ax2 = axes[1]
    non_rostered = sensitivity_df[~sensitivity_df["on_optimal_roster"]].copy()
    non_rostered["abs_delta"] = non_rostered["objective_delta"].abs()
    non_rostered = non_rostered.nlargest(top_n, "abs_delta")

    if len(non_rostered) > 0:
        y_pos = np.arange(len(non_rostered))
        ax2.barh(y_pos, non_rostered["objective_delta"], color=WIN_COLOR, alpha=0.8)

        names = [
            f"{row['Position']} {strip_name_suffix(row['Name'])}"
            for _, row in non_rostered.iterrows()
        ]
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(names)
        ax2.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

    ax2.set_xlabel("Objective Change if Added")
    ax2.set_title("Best Available Non-Rostered")

    plt.suptitle("Player Sensitivity Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_constraint_analysis(
    optimal_roster_names: list[str],
    projections: pd.DataFrame,
) -> plt.Figure:
    """
    Visualize which roster constraints are binding.
    """
    roster_df = projections[projections["Name"].isin(optimal_roster_names)]

    # Count by position
    position_counts = roster_df["Position"].value_counts().to_dict()

    # Count by type
    n_hitters = (roster_df["player_type"] == "hitter").sum()
    n_pitchers = (roster_df["player_type"] == "pitcher").sum()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Position slots
    ax1 = axes[0]
    all_slots = {**HITTING_SLOTS, **PITCHING_SLOTS}
    slot_names = list(all_slots.keys())
    required = [all_slots[s] for s in slot_names]

    # Get actual counts (approximate - positions don't map 1:1 to slots)
    position_to_slot = {
        "C": "C",
        "1B": "1B",
        "2B": "2B",
        "SS": "SS",
        "3B": "3B",
        "OF": "OF",
        "DH": "UTIL",
        "SP": "SP",
        "RP": "RP",
    }

    actual = []
    for slot in slot_names:
        if slot == "UTIL":
            actual.append(position_counts.get("DH", 0))
        elif slot == "OF":
            actual.append(position_counts.get("OF", 0))
        else:
            actual.append(position_counts.get(slot, 0))

    x = np.arange(len(slot_names))
    width = 0.35

    ax1.bar(
        x - width / 2, required, width, label="Required", color=NEUTRAL_COLOR, alpha=0.7
    )
    ax1.bar(
        x + width / 2,
        actual,
        width,
        label="Rostered",
        color=TEAM_COLORS["me"],
        alpha=0.7,
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(slot_names, rotation=45, ha="right")
    ax1.set_ylabel("Count")
    ax1.set_title("Position Slot Analysis")
    ax1.legend()

    # Composition bounds
    ax2 = axes[1]

    bounds_data = {
        "Hitters": (n_hitters, MIN_HITTERS, MAX_HITTERS),
        "Pitchers": (n_pitchers, MIN_PITCHERS, MAX_PITCHERS),
    }

    x = np.arange(2)

    for i, (label, (actual, min_val, max_val)) in enumerate(bounds_data.items()):
        if actual == min_val:
            color = LOSS_COLOR  # At minimum
        elif actual == max_val:
            color = "#F39C12"  # At maximum (yellow/orange)
        else:
            color = WIN_COLOR  # Has slack

        ax2.bar(i, actual, color=color, alpha=0.8, label=label)
        ax2.hlines(
            min_val, i - 0.3, i + 0.3, colors="black", linestyles="--", linewidth=2
        )
        ax2.hlines(
            max_val, i - 0.3, i + 0.3, colors="black", linestyles="--", linewidth=2
        )
        ax2.text(i, min_val - 0.5, f"min={min_val}", ha="center", fontsize=9)
        ax2.text(i, max_val + 0.5, f"max={max_val}", ha="center", fontsize=9)

    ax2.set_xticks(x)
    ax2.set_xticklabels(["Hitters", "Pitchers"])
    ax2.set_ylabel("Count")
    ax2.set_title("Roster Composition")

    # Legend for colors
    legend_patches = [
        mpatches.Patch(color=LOSS_COLOR, alpha=0.8, label="At minimum"),
        mpatches.Patch(color="#F39C12", alpha=0.8, label="At maximum"),
        mpatches.Patch(color=WIN_COLOR, alpha=0.8, label="Slack available"),
    ]
    ax2.legend(handles=legend_patches, loc="upper right")

    plt.suptitle("Constraint Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


# === TRADE ENGINE SPECIFIC VISUALIZATIONS ===


def plot_win_probability_breakdown(
    diagnostics: dict,
) -> plt.Figure:
    """
    Visualize the components of win probability calculation.
    """
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Top-left: Matchup probability heatmap
    ax1 = fig.add_subplot(gs[0, 0])

    matchup_probs = diagnostics["matchup_probs"]
    categories = ALL_CATEGORIES
    opp_ids = sorted(list(matchup_probs[categories[0]].keys()))

    matrix = np.array(
        [[matchup_probs[cat][opp] for cat in categories] for opp in opp_ids]
    )

    im = ax1.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    for i in range(len(opp_ids)):
        for j in range(len(categories)):
            ax1.text(j, i, f"{matrix[i, j]:.0%}", ha="center", va="center", fontsize=8)

    ax1.set_xticks(np.arange(len(categories)))
    ax1.set_xticklabels(categories, rotation=45, ha="right")
    ax1.set_yticks(np.arange(len(opp_ids)))
    ax1.set_yticklabels([f"Opp {o}" for o in opp_ids])
    ax1.set_title("Matchup Win Probabilities")
    plt.colorbar(im, ax=ax1)

    # Top-right: Expected wins by category
    ax2 = fig.add_subplot(gs[0, 1])

    expected_per_cat = [sum(matchup_probs[cat].values()) for cat in categories]

    colors = [
        WIN_COLOR if e > len(opp_ids) / 2 else LOSS_COLOR for e in expected_per_cat
    ]
    ax2.bar(range(len(categories)), expected_per_cat, color=colors, alpha=0.8)
    ax2.axhline(
        y=len(opp_ids) / 2,
        color="black",
        linestyle="--",
        linewidth=1,
        label="Break-even",
    )
    ax2.set_xticks(range(len(categories)))
    ax2.set_xticklabels(categories, rotation=45, ha="right")
    ax2.set_ylabel("Expected Wins")
    ax2.set_title("Expected Wins by Category")
    ax2.legend()

    # Bottom-left: Distribution visualization
    ax3 = fig.add_subplot(gs[1, 0])

    mu_T = diagnostics["mu_T"]
    sigma_T = np.sqrt(diagnostics["sigma_T_sq"])
    mu_D = diagnostics["mu_D"]
    sigma_D = diagnostics["sigma_D"]

    # Plot my distribution vs target
    x = np.linspace(mu_D - 4 * sigma_D, mu_D + 4 * sigma_D, 100)
    y_me = (
        1 / (sigma_D * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu_D) / sigma_D) ** 2)
    )

    ax3.fill_between(
        x[x > 0], y_me[x > 0], alpha=0.3, color=WIN_COLOR, label="Win region"
    )
    ax3.fill_between(
        x[x <= 0], y_me[x <= 0], alpha=0.3, color=LOSS_COLOR, label="Loss region"
    )
    ax3.plot(x, y_me, color="black", linewidth=2)
    ax3.axvline(x=0, color="black", linestyle="--", linewidth=1)
    ax3.set_xlabel("Differential (vs best opponent)")
    ax3.set_ylabel("Density")
    ax3.set_title("Win Probability Distribution")
    ax3.legend()

    # Bottom-right: Summary statistics
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    from scipy import stats as scipy_stats

    V = scipy_stats.norm.cdf(mu_D / sigma_D) if sigma_D > 0.001 else 0.5

    summary_text = f"""
Summary Statistics
──────────────────
Expected Wins (μ_T):    {mu_T:.1f} / 60
Std Dev (σ_T):          {sigma_T:.2f}

Target to Beat (μ_L):   {diagnostics["mu_L"]:.2f}
Target Std (σ_L):       {np.sqrt(diagnostics["sigma_L_sq"]):.2f}

Differential (μ_D):     {mu_D:.2f}
Differential Std (σ_D): {sigma_D:.2f}

Win Probability (V):    {V:.1%}
"""
    ax4.text(
        0.1,
        0.9,
        summary_text,
        transform=ax4.transAxes,
        fontsize=12,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.suptitle("Win Probability Breakdown", fontsize=14, fontweight="bold")
    return fig


def plot_category_marginal_values(
    gradient: dict[str, dict[int, float]],
) -> plt.Figure:
    """
    Visualize marginal value of improvement in each category.
    """
    categories = ALL_CATEGORIES
    opp_ids = sorted(list(gradient[categories[0]].keys()))

    matrix = np.array([[gradient[cat][opp] for opp in opp_ids] for cat in categories])

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    for i in range(len(categories)):
        for j in range(len(opp_ids)):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=9)

    ax.set_xticks(np.arange(len(opp_ids)))
    ax.set_xticklabels([f"Opp {o}" for o in opp_ids])
    ax.set_yticks(np.arange(len(categories)))
    ax.set_yticklabels(categories)

    ax.set_xlabel("Opponent")
    ax.set_ylabel("Category")
    ax.set_title("Marginal Value of Improvement")

    plt.colorbar(im, ax=ax, label="Gradient")
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

    # Separate by roster status and player type
    for player_type in ["hitter", "pitcher"]:
        subset = player_values[player_values["player_type"] == player_type]

        my_players = subset[subset["on_my_roster"]]
        opp_players = subset[~subset["on_my_roster"]]

        # Plot opponent players (hollow)
        if len(opp_players) > 0:
            ax.scatter(
                opp_players["generic_value"],
                opp_players["delta_V_acquire"],
                c=TEAM_COLORS["opponent"] if player_type == "hitter" else "#8B4513",
                s=50,
                alpha=0.5,
                marker="o",
                facecolors="none",
                edgecolors=TEAM_COLORS["opponent"]
                if player_type == "hitter"
                else "#8B4513",
                label=f"Opponent {player_type}s",
            )

        # Plot my players (filled)
        if len(my_players) > 0:
            ax.scatter(
                my_players["generic_value"],
                my_players["delta_V_acquire"],
                c=TEAM_COLORS["me"] if player_type == "hitter" else "#228B22",
                s=80,
                alpha=0.8,
                marker="o",
                label=f"My {player_type}s",
            )

    # Add quadrant lines
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.5)

    # Add quadrant labels
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    ax.text(
        xlim[1] * 0.7,
        ylim[1] * 0.8,
        "Great targets\n(high generic, high context)",
        fontsize=9,
        alpha=0.7,
        ha="center",
    )
    ax.text(
        xlim[0] * 0.7,
        ylim[1] * 0.8,
        "Undervalued for me\n(low generic, high context)",
        fontsize=9,
        alpha=0.7,
        ha="center",
    )
    ax.text(
        xlim[1] * 0.7,
        ylim[0] * 0.8,
        "Overvalued for me\n(high generic, low context)",
        fontsize=9,
        alpha=0.7,
        ha="center",
    )
    ax.text(
        xlim[0] * 0.7,
        ylim[0] * 0.8,
        "Avoid\n(low generic, low context)",
        fontsize=9,
        alpha=0.7,
        ha="center",
    )

    ax.set_xlabel("Generic Value (z-score sum)")
    ax.set_ylabel("Contextual Value (ΔV if acquired)")
    ax.set_title("Player Values: Generic vs Contextual")
    ax.legend(loc="upper left")

    plt.tight_layout()
    return fig
