"""
Dynasty Roto Roster Optimizer - Visualization functions.

All functions return matplotlib Figure objects. No plt.show() calls.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from tqdm.auto import tqdm

from .roster_optimizer import (
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
    RATIO_STATS,
    SLOT_ELIGIBILITY,
    _build_milp,
    build_and_solve_milp,
    compute_team_totals,
)


def plot_team_radar(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
) -> plt.Figure:
    """
    Radar chart comparing all 7 teams across all 10 categories.

    Display:
        - One polygon per team (7 total)
        - My team: thick solid line, distinct color (blue)
        - Opponents: thin dashed lines, muted colors
        - Legend identifying each team

    Normalization:
        Convert each category to percentile rank among the 7 teams.
        This puts all categories on [0, 1] scale.
        For negative categories (ERA, WHIP), flip so that better = higher on chart.
    """
    # Collect all teams' values
    all_teams = {"Me": my_totals}
    for tid in range(1, 7):
        all_teams[f"Opp {tid}"] = opponent_totals[tid]

    # Compute percentile ranks for each category
    normalized = {team: {} for team in all_teams}

    for cat in ALL_CATEGORIES:
        values = [all_teams[team][cat] for team in all_teams]

        if cat in NEGATIVE_CATEGORIES:
            # Lower is better - flip ranking
            ranked = np.argsort(np.argsort(values))  # rank indices
            percentiles = 1 - (
                ranked / (len(values) - 1)
            )  # flip: low value = high percentile
        else:
            # Higher is better
            ranked = np.argsort(np.argsort(values))
            percentiles = ranked / (len(values) - 1)

        for i, team in enumerate(all_teams.keys()):
            normalized[team][cat] = percentiles[i]

    # Set up radar chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Angles for each category
    angles = np.linspace(0, 2 * np.pi, len(ALL_CATEGORIES), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    # Plot each team
    colors = plt.cm.Set2(np.linspace(0, 1, 7))

    for i, (team, values) in enumerate(normalized.items()):
        team_values = [values[cat] for cat in ALL_CATEGORIES]
        team_values += team_values[:1]  # Close the polygon

        if team == "Me":
            ax.plot(angles, team_values, "o-", linewidth=3, label=team, color="blue")
            ax.fill(angles, team_values, alpha=0.25, color="blue")
        else:
            ax.plot(
                angles,
                team_values,
                "--",
                linewidth=1.5,
                label=team,
                color=colors[i],
                alpha=0.7,
            )

    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(ALL_CATEGORIES, size=12)

    # Set y-axis limits
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], size=8)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    ax.set_title("Team Comparison (Percentile Ranks)", size=14, pad=20)

    plt.tight_layout()
    return fig


def plot_category_margins(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
) -> plt.Figure:
    """
    Grouped bar chart showing my margin over each opponent in each category.

    X-axis: 10 categories
    Bars: 6 bars per category (one per opponent), showing my_value - opponent_value
    Colors: Green if positive (I win), red if negative (I lose)

    For negative categories (ERA, WHIP), flip sign: opponent_value - my_value,
    so positive still means I win.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(ALL_CATEGORIES))
    width = 0.12  # Width of each bar

    for opp_idx, tid in enumerate(range(1, 7)):
        margins = []
        colors = []

        for cat in ALL_CATEGORIES:
            my_val = my_totals[cat]
            opp_val = opponent_totals[tid][cat]

            if cat in NEGATIVE_CATEGORIES:
                # Lower is better - flip so positive = I win
                margin = opp_val - my_val
            else:
                margin = my_val - opp_val

            margins.append(margin)
            colors.append("green" if margin > 0 else "red")

        offset = (opp_idx - 2.5) * width
        bars = ax.bar(x + offset, margins, width, label=f"vs Opp {tid}", alpha=0.7)

        # Color each bar based on win/loss
        for bar, color in zip(bars, colors):
            bar.set_color(color)

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Category")
    ax.set_ylabel("Margin (positive = I win)")
    ax.set_title("Category Margins vs Each Opponent")
    ax.set_xticks(x)
    ax.set_xticklabels(ALL_CATEGORIES)

    # Custom legend for win/loss
    legend_elements = [
        Patch(facecolor="green", alpha=0.7, label="Win"),
        Patch(facecolor="red", alpha=0.7, label="Loss"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    return fig


def plot_win_matrix(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
) -> plt.Figure:
    """
    Heatmap showing win/loss for each opponent-category pair.

    Rows: 6 opponents
    Columns: 10 categories
    Cell color: Green if I win, red if I lose
    Cell text: Margin (my_value - opponent_value), formatted appropriately
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Build matrix of wins (1) and losses (0)
    win_matrix = np.zeros((6, len(ALL_CATEGORIES)))
    margin_matrix = np.zeros((6, len(ALL_CATEGORIES)))

    for row, tid in enumerate(range(1, 7)):
        for col, cat in enumerate(ALL_CATEGORIES):
            my_val = my_totals[cat]
            opp_val = opponent_totals[tid][cat]

            if cat in NEGATIVE_CATEGORIES:
                # Lower is better
                win = my_val < opp_val
                margin = opp_val - my_val  # positive = I win
            else:
                win = my_val > opp_val
                margin = my_val - opp_val

            win_matrix[row, col] = 1 if win else 0
            margin_matrix[row, col] = margin

    # Create color map: red (0) to green (1)
    cmap = plt.cm.RdYlGn

    # Plot heatmap
    im = ax.imshow(win_matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    # Add text annotations
    for row in range(6):
        for col in range(len(ALL_CATEGORIES)):
            margin = margin_matrix[row, col]
            cat = ALL_CATEGORIES[col]

            # Format margin based on category type
            if cat in ["OPS"]:
                text = f"{margin:+.3f}"
            elif cat in ["ERA", "WHIP"]:
                text = f"{margin:+.2f}"
            else:
                text = f"{margin:+.0f}"

            # Text color based on background
            text_color = "white" if win_matrix[row, col] == 0 else "black"
            ax.text(
                col,
                row,
                text,
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
                fontweight="bold",
            )

    # Labels
    ax.set_xticks(range(len(ALL_CATEGORIES)))
    ax.set_xticklabels(ALL_CATEGORIES)
    ax.set_yticks(range(6))
    ax.set_yticklabels([f"Opponent {i}" for i in range(1, 7)])

    ax.set_xlabel("Category")
    ax.set_ylabel("Opponent")
    ax.set_title(
        "Win/Loss Matrix (Green = Win, Red = Loss)\nNumbers show margin (positive = I win)"
    )

    plt.tight_layout()
    return fig


def plot_category_contributions(
    roster_names: list[str],
    projections: pd.DataFrame,
    category: str,
) -> plt.Figure:
    """
    Horizontal bar chart showing each player's contribution to one category.

    For counting stats: Player's raw value.
    For ratio stats (OPS, ERA, WHIP): Player's "impact" on team ratio.
    """
    roster_df = projections[projections["Name"].isin(roster_names)].copy()

    # Filter to relevant player type
    if category in HITTING_CATEGORIES:
        players = roster_df[roster_df["player_type"] == "hitter"].copy()
        weight_col = "PA" if category in RATIO_STATS else None
    else:
        players = roster_df[roster_df["player_type"] == "pitcher"].copy()
        weight_col = "IP" if category in RATIO_STATS else None

    fig, ax = plt.subplots(figsize=(10, max(6, len(players) * 0.3)))

    if category in RATIO_STATS:
        # Compute weighted average for team
        weight = weight_col
        total_weight = players[weight].sum()
        team_avg = (players[weight] * players[category]).sum() / total_weight

        # Impact = weight * (player_value - team_avg)
        players["contribution"] = players[weight] * (players[category] - team_avg)

        # For negative categories, flip sign so positive = helps team
        if category in NEGATIVE_CATEGORIES:
            players["contribution"] = -players["contribution"]

        title_suffix = f" (Impact on Team {category})"
    else:
        # Counting stat - raw value
        players["contribution"] = players[category]
        title_suffix = ""

    # Sort by contribution magnitude
    players = players.sort_values("contribution", ascending=True)

    # Create bars
    colors = ["green" if c >= 0 else "red" for c in players["contribution"]]
    bars = ax.barh(
        range(len(players)), players["contribution"], color=colors, alpha=0.7
    )

    # Labels
    ax.set_yticks(range(len(players)))
    ax.set_yticklabels(
        [f"{row['Name']} ({row['Position']})" for _, row in players.iterrows()]
    )
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

    ax.set_xlabel(
        "Contribution" if category not in RATIO_STATS else "Impact (weighted)"
    )
    ax.set_title(f"{category} Contributions{title_suffix}")

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

    Each player is a line on the radar. Categories are normalized to [0, 1] scale
    based on min/max within the roster for comparability.

    Args:
        roster_names: List of player names on the roster
        projections: Combined projections DataFrame
        player_type: "hitter" or "pitcher" - determines which 5 categories to show
        top_n: Maximum number of players to show (to avoid clutter)

    Returns:
        matplotlib Figure with radar chart
    """
    roster_df = projections[projections["Name"].isin(roster_names)].copy()

    # Filter to relevant player type
    if player_type == "hitter":
        players = roster_df[roster_df["player_type"] == "hitter"].copy()
        categories = list(HITTING_CATEGORIES)
        weight_col = "PA"
    else:
        players = roster_df[roster_df["player_type"] == "pitcher"].copy()
        categories = list(PITCHING_CATEGORIES)
        weight_col = "IP"

    if len(players) == 0:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.text(0.5, 0.5, f"No {player_type}s on roster", ha="center", va="center")
        return fig

    # Compute contribution scores for each category
    contribution_data = {}
    for cat in categories:
        if cat in RATIO_STATS:
            # For ratio stats, compute impact = weight * (value - team_avg)
            total_weight = players[weight_col].sum()
            team_avg = (players[weight_col] * players[cat]).sum() / total_weight
            contrib = players[weight_col] * (players[cat] - team_avg)

            # Flip negative categories so positive = good
            if cat in NEGATIVE_CATEGORIES:
                contrib = -contrib
        else:
            # Counting stat - raw value
            contrib = players[cat]

        contribution_data[cat] = contrib.values

    # Build contribution matrix
    contrib_df = pd.DataFrame(contribution_data, index=players["Name"].values)

    # Normalize each category to [0, 1] based on min/max
    for cat in categories:
        col_min = contrib_df[cat].min()
        col_max = contrib_df[cat].max()
        if col_max > col_min:
            contrib_df[cat] = (contrib_df[cat] - col_min) / (col_max - col_min)
        else:
            contrib_df[cat] = 0.5  # All same value

    # Select top N players by total contribution (sum of normalized values)
    contrib_df["total"] = contrib_df[categories].sum(axis=1)
    contrib_df = contrib_df.sort_values("total", ascending=False).head(top_n)
    contrib_df = contrib_df.drop(columns=["total"])

    # Set up radar chart
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))

    # Color map for players
    colors = plt.cm.tab20(np.linspace(0, 1, len(contrib_df)))

    # Plot each player
    for i, (player_name, row) in enumerate(contrib_df.iterrows()):
        values = row[categories].values.tolist()
        values += values[:1]  # Close the polygon

        # Strip -H/-P suffix for display
        display_name = player_name
        if display_name.endswith("-H") or display_name.endswith("-P"):
            display_name = display_name[:-2]

        ax.plot(angles, values, "o-", linewidth=2, label=display_name, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12)

    # Set radial limits
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], size=8)

    # Title and legend
    title = f"{'Hitter' if player_type == 'hitter' else 'Pitcher'} Contributions by Category"
    ax.set_title(title, size=14, y=1.08)

    # Place legend outside
    ax.legend(loc="upper left", bbox_to_anchor=(1.1, 1.0), fontsize=9)

    plt.tight_layout()
    return fig


def plot_roster_changes(
    old_roster_names: set[str],
    new_roster_names: set[str],
    projections: pd.DataFrame,
) -> plt.Figure:
    """
    Visual comparison of old vs new roster.

    Top section: Two-column table showing dropped and added players
    Bottom section: Bar chart showing net change in each projected category total
    """
    added = new_roster_names - old_roster_names
    dropped = old_roster_names - new_roster_names

    # Compute totals for comparison
    old_totals = compute_team_totals(old_roster_names, projections)
    new_totals = compute_team_totals(new_roster_names, projections)

    fig = plt.figure(figsize=(14, 10))

    # Top: Table of changes
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.axis("off")

    # Build table data
    dropped_list = sorted(dropped)
    added_list = sorted(added)
    max_rows = max(len(dropped_list), len(added_list), 1)

    # Pad lists to same length
    dropped_list += [""] * (max_rows - len(dropped_list))
    added_list += [""] * (max_rows - len(added_list))

    # Add position and key stat info
    def format_player_info(name, projections):
        if not name:
            return ""
        matches = projections[projections["Name"] == name]
        if len(matches) == 0:
            return f"{name} (NOT FOUND)"

        # Names are globally unique with -H/-P suffix, so there's exactly one match
        player = matches.iloc[0]
        pos = player["Position"]

        # Strip the -H/-P suffix for cleaner display (position already tells us the type)
        display_name = name[:-2] if name.endswith(("-H", "-P")) else name

        if player["player_type"] == "hitter":
            pa = player["PA"]
            warn = " △" if pa < 50 else ""
            return f"{display_name} ({pos}, {pa:.0f} PA){warn}"
        else:
            ip = player["IP"]
            warn = " △" if ip < 20 else ""
            return f"{display_name} ({pos}, {ip:.0f} IP){warn}"

    dropped_with_pos = [format_player_info(name, projections) for name in dropped_list]
    added_with_pos = [format_player_info(name, projections) for name in added_list]

    table_data = list(zip(dropped_with_pos, added_with_pos))

    table = ax1.table(
        cellText=table_data,
        colLabels=["DROPPED", "ADDED"],
        loc="center",
        cellLoc="left",
        colColours=["#ffcccc", "#ccffcc"],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    ax1.set_title("Roster Changes", fontsize=14, fontweight="bold")

    # Bottom: Category changes bar chart
    ax2 = fig.add_subplot(2, 1, 2)

    changes = []
    for cat in ALL_CATEGORIES:
        change = new_totals[cat] - old_totals[cat]
        if cat in NEGATIVE_CATEGORIES:
            # For ERA/WHIP, negative change is good
            change = -change
        changes.append(change)

    colors = ["green" if c >= 0 else "red" for c in changes]
    bars = ax2.bar(ALL_CATEGORIES, changes, color=colors, alpha=0.7)

    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax2.set_xlabel("Category")
    ax2.set_ylabel("Change (positive = improvement)")
    ax2.set_title(
        "Net Change in Category Totals\n(ERA/WHIP flipped so positive = improvement)"
    )

    # Add value labels on bars
    for bar, change, cat in zip(bars, changes, ALL_CATEGORIES):
        original_change = new_totals[cat] - old_totals[cat]
        if cat in ["OPS"]:
            label = f"{original_change:+.3f}"
        elif cat in ["ERA", "WHIP"]:
            label = f"{original_change:+.2f}"
        else:
            label = f"{original_change:+.0f}"

        ax2.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom" if change >= 0 else "top",
            fontsize=9,
        )

    plt.tight_layout()
    return fig


def compute_player_sensitivity(
    optimal_roster_names: list[str],
    candidates: pd.DataFrame,
    opponent_totals: dict[int, dict[str, float]],
) -> pd.DataFrame:
    """
    Compute sensitivity of objective to each player.

    For each player in candidates:
        - If player is ON optimal roster: solve MILP forcing x[i] = 0 (exclude them)
        - If player is NOT on optimal roster: solve MILP forcing x[i] = 1 (include them)
        - Compare resulting objective to unconstrained optimum

    Returns:
        DataFrame with columns:
            - Name: player name
            - player_type: hitter or pitcher
            - Position: player's position(s)
            - on_optimal_roster: bool
            - forced_objective: objective value when this player is forced in/out
            - objective_delta: forced_objective - optimal_objective
    """
    import pulp

    optimal_set = set(optimal_roster_names)

    # Get the baseline objective
    _, baseline_info = build_and_solve_milp(candidates, opponent_totals, optimal_set)
    baseline_objective = baseline_info["objective"]

    n_candidates = len(candidates)
    est_minutes = n_candidates * 1.5 / 60

    print(
        f"Computing sensitivity for {n_candidates} candidates (estimated time: {est_minutes:.0f} minutes)"
    )

    results = []
    candidates = candidates.reset_index(drop=True)

    for idx in tqdm(range(len(candidates)), desc="Computing player sensitivities"):
        player = candidates.iloc[idx]
        player_name = player["Name"]
        on_roster = player_name in optimal_set

        # Build MILP with forcing constraint using shared logic
        prob, x, a, y, I_H, I_P = _build_milp(
            candidates, opponent_totals, check_eligibility=False
        )

        # Add forcing constraint
        if on_roster:
            prob += x[idx] == 0  # Force player OUT
        else:
            prob += x[idx] == 1  # Force player IN

        # Solve silently
        solver = pulp.HiGHS(msg=0, timeLimit=60)
        status = prob.solve(solver)

        forced_obj = pulp.value(prob.objective) if status == pulp.LpStatusOptimal else None

        results.append(
            {
                "Name": player_name,
                "player_type": player["player_type"],
                "Position": player["Position"],
                "on_optimal_roster": on_roster,
                "forced_objective": forced_obj,
                "objective_delta": forced_obj - baseline_objective
                if forced_obj is not None
                else None,
            }
        )

    return pd.DataFrame(results)


def plot_player_sensitivity(
    sensitivity_df: pd.DataFrame,
    top_n: int = 15,
) -> plt.Figure:
    """
    Horizontal bar chart showing most impactful players.

    Two panels, stacked vertically:

    Top panel: "Most Valuable Rostered Players"
        Players currently on optimal roster, sorted by objective_delta (most negative first).

    Bottom panel: "Best Available Non-Rostered"
        Players NOT on optimal roster, sorted by objective_delta.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Top panel: Most valuable rostered players
    rostered = sensitivity_df[sensitivity_df["on_optimal_roster"]].copy()
    rostered = rostered.dropna(subset=["objective_delta"])
    rostered = rostered.sort_values("objective_delta").head(top_n)

    if len(rostered) > 0:
        colors = ["red" if d < 0 else "gray" for d in rostered["objective_delta"]]
        ax1.barh(
            range(len(rostered)), rostered["objective_delta"], color=colors, alpha=0.7
        )
        ax1.set_yticks(range(len(rostered)))
        ax1.set_yticklabels(
            [f"{row['Name']} ({row['Position']})" for _, row in rostered.iterrows()]
        )
        ax1.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

    ax1.set_xlabel("Objective Delta (if removed)")
    ax1.set_title("Most Valuable Rostered Players\n(Negative = removing them hurts)")

    # Bottom panel: Best available non-rostered
    non_rostered = sensitivity_df[~sensitivity_df["on_optimal_roster"]].copy()
    non_rostered = non_rostered.dropna(subset=["objective_delta"])
    non_rostered["abs_delta"] = non_rostered["objective_delta"].abs()
    non_rostered = non_rostered.sort_values("objective_delta", ascending=False).head(
        top_n
    )

    if len(non_rostered) > 0:
        colors = ["green" if d > 0 else "gray" for d in non_rostered["objective_delta"]]
        ax2.barh(
            range(len(non_rostered)),
            non_rostered["objective_delta"],
            color=colors,
            alpha=0.7,
        )
        ax2.set_yticks(range(len(non_rostered)))
        ax2.set_yticklabels(
            [f"{row['Name']} ({row['Position']})" for _, row in non_rostered.iterrows()]
        )
        ax2.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

    ax2.set_xlabel("Objective Delta (if forced onto roster)")
    ax2.set_title(
        "Best Available Non-Rostered Players\n(Positive = forcing them in helps)"
    )

    plt.tight_layout()
    return fig


def plot_constraint_analysis(
    candidates: pd.DataFrame,
    optimal_roster_names: list[str],
    projections: pd.DataFrame,
) -> plt.Figure:
    """
    Visualize which roster constraints are binding.

    Bar chart showing:
        - For each position slot: how many eligible players are rostered vs required
        - For hitter/pitcher bounds: current count vs min/max
    """
    roster_df = projections[projections["Name"].isin(optimal_roster_names)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Position slot fill rates
    slot_data = []
    all_slots = {**HITTING_SLOTS, **PITCHING_SLOTS}

    for slot, required in all_slots.items():
        eligible_positions = SLOT_ELIGIBILITY[slot]
        eligible_rostered = roster_df[roster_df["Position"].isin(eligible_positions)]
        count = len(eligible_rostered)
        slot_data.append(
            {
                "slot": slot,
                "required": required,
                "eligible_rostered": count,
            }
        )

    slot_df = pd.DataFrame(slot_data)

    x = np.arange(len(slot_df))
    width = 0.35

    ax1.bar(
        x - width / 2,
        slot_df["required"],
        width,
        label="Required",
        color="steelblue",
        alpha=0.7,
    )
    ax1.bar(
        x + width / 2,
        slot_df["eligible_rostered"],
        width,
        label="Eligible Rostered",
        color="green",
        alpha=0.7,
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(slot_df["slot"])
    ax1.set_ylabel("Count")
    ax1.set_title("Position Slot Requirements vs Eligible Rostered")
    ax1.legend()

    # Right: Hitter/Pitcher composition
    hitter_count = (roster_df["player_type"] == "hitter").sum()
    pitcher_count = (roster_df["player_type"] == "pitcher").sum()

    comp_labels = ["Hitters", "Pitchers"]
    comp_counts = [hitter_count, pitcher_count]
    comp_mins = [MIN_HITTERS, MIN_PITCHERS]
    comp_maxs = [MAX_HITTERS, MAX_PITCHERS]

    x = np.arange(2)

    # Draw bars for current count
    bars = ax2.bar(x, comp_counts, 0.5, label="Current", color="steelblue", alpha=0.7)

    # Draw min/max lines
    for i, (min_val, max_val) in enumerate(zip(comp_mins, comp_maxs)):
        ax2.hlines(
            min_val, i - 0.3, i + 0.3, colors="red", linestyles="--", linewidth=2
        )
        ax2.hlines(
            max_val, i - 0.3, i + 0.3, colors="orange", linestyles="--", linewidth=2
        )

    # Add value labels
    for bar, count in zip(bars, comp_counts):
        ax2.annotate(
            f"{count}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(comp_labels)
    ax2.set_ylabel("Count")
    ax2.set_title("Roster Composition\n(Red = Min, Orange = Max)")
    ax2.legend(["Current", "Min", "Max"])

    # Color bars based on binding constraints
    for i, (count, min_val, max_val) in enumerate(
        zip(comp_counts, comp_mins, comp_maxs)
    ):
        if count == min_val:
            bars[i].set_color("red")
            bars[i].set_alpha(0.5)
        elif count == max_val:
            bars[i].set_color("orange")
            bars[i].set_alpha(0.5)

    plt.tight_layout()
    return fig
