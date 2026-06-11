"""
Visualization functions for the v2 roster optimizer.

All functions return matplotlib.Figure objects. Never call plt.show() --
the marimo notebook handles display via fig_to_png.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch

from .config import (
    ALL_CATEGORIES,
    HITTING_CATEGORIES,
    NEGATIVE_CATEGORIES,
    PITCHING_CATEGORIES,
)
from .players import strip_name_suffix

plt.style.use("seaborn-v0_8-whitegrid")

WIN_COLOR = "#2ECC71"
LOSS_COLOR = "#E74C3C"
NEUTRAL_COLOR = "#95A5A6"


def plot_gradient_profile(gradient: dict[str, float]) -> plt.Figure:
    """Horizontal bar chart of category gradients sorted by magnitude.

    Bars show |g_c|; annotations show signed g_c values (4 decimal places).
    Categories with larger |g_c| have more competitive matchups — this is the
    most important chart in the dashboard.

    Args:
        gradient: {category: dEW/d(my_c)} from compute_ew_gradient.

    Returns:
        Figure with horizontal bar chart.
    """
    cats = sorted(ALL_CATEGORIES, key=lambda c: abs(gradient[c]))
    magnitudes = [abs(gradient[c]) for c in cats]
    max_mag = max(magnitudes) if magnitudes else 1.0

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(cats, magnitudes, color=WIN_COLOR, edgecolor="white")

    for bar, cat in zip(bars, cats):
        ax.text(
            bar.get_width() + max_mag * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{gradient[cat]:+.2f}",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel("|∂EW/∂(category)|")
    ax.set_title("Category Gradient: Where Investment Pays Off")
    fig.tight_layout()
    return fig


def plot_category_heatmap(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    opponent_teams: list[str],
) -> plt.Figure:
    """Heatmap of win/loss margins vs each opponent across 10 categories.

    Positive cell = I am winning. Rate stats (OPS, ERA, WHIP) scaled ×1000
    so they display alongside counting stats.

    Args:
        my_totals: My team's projected totals per category.
        opponent_totals: {opp_id (1-indexed): {cat: total}}.
        opponent_teams: Alphabetically sorted opponent team names.

    Returns:
        Figure with annotated seaborn heatmap.
    """
    rate_cats = {"OPS", "ERA", "WHIP"}
    n_opps = len(opponent_teams)
    margin_data = np.zeros((n_opps, len(ALL_CATEGORIES)))

    for i in range(n_opps):
        opp_id = i + 1
        for j, cat in enumerate(ALL_CATEGORIES):
            if cat in NEGATIVE_CATEGORIES:
                margin = opponent_totals[opp_id][cat] - my_totals[cat]
            else:
                margin = my_totals[cat] - opponent_totals[opp_id][cat]
            if cat in rate_cats:
                margin *= 1000
            margin_data[i, j] = margin

    df = pd.DataFrame(margin_data, index=opponent_teams, columns=ALL_CATEGORIES)

    fig, ax = plt.subplots(figsize=(14, max(4, n_opps * 0.8 + 1)))
    sns.heatmap(
        df,
        cmap="RdYlGn",
        center=0,
        annot=True,
        fmt=".0f",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Win/Loss Matrix (positive = winning, rate stats ×1000)")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig


def plot_move_impact(
    before_totals: dict[str, float],
    after_totals: dict[str, float],
    before_ew: float,
    after_ew: float,
) -> plt.Figure:
    """Horizontal bar chart showing per-category delta from a roster move.

    Positive bars (green) = improvement; negative bars (red) = regression.
    For C- stats (ERA, WHIP), delta is flipped so positive always means better.

    Args:
        before_totals: Team totals before the move.
        after_totals: Team totals after the move.
        before_ew: Expected wins before.
        after_ew: Expected wins after.

    Returns:
        Figure with impact chart, title shows EW change.
    """
    deltas = []
    for cat in ALL_CATEGORIES:
        if cat in NEGATIVE_CATEGORIES:
            deltas.append(before_totals[cat] - after_totals[cat])
        else:
            deltas.append(after_totals[cat] - before_totals[cat])

    colors = [WIN_COLOR if d >= 0 else LOSS_COLOR for d in deltas]
    max_abs = max((abs(d) for d in deltas), default=1.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(ALL_CATEGORIES, deltas, color=colors, edgecolor="white")

    for i, delta in enumerate(deltas):
        offset = max_abs * 0.03
        x_pos = delta + (offset if delta >= 0 else -offset)
        ax.text(
            x_pos,
            i,
            f"{delta:+.2f}",
            va="center",
            fontsize=9,
            ha="left" if delta >= 0 else "right",
        )

    ax.axvline(0, color="black", linewidth=0.5)
    ew_delta = after_ew - before_ew
    ax.set_title(
        f"Move Impact: EW {before_ew:.2f} \u2192 {after_ew:.2f} ({ew_delta:+.2f})"
    )
    fig.tight_layout()
    return fig


def plot_mew_breakdown(
    player_name: str,
    players: pd.DataFrame,
    gradient: dict[str, float],
    my_totals: dict[str, float],
) -> plt.Figure:
    """Three-column breakdown: player z-scores × gradient → MEW contribution.

    Column 1: z-scores (FV components) within the player's type population.
    Column 2: gradient weights for the relevant 5 categories.
    Column 3: per-category MEW contribution (counting: g_c * stat; ratio:
              g_c * weight * (stat - team_avg) / total_weight).
    ConnectionPatch arrows show the flow between panels.

    Args:
        player_name: Internal name with -H/-P suffix.
        players: Full DataFrame with stat columns, FV, and MEW.
        gradient: Converged gradient dict {cat: dEW/d(my_c)}.
        my_totals: Team totals dict (needs PA, IP, OPS, ERA, WHIP).

    Returns:
        Figure with three-panel breakdown.
    """
    matches = players[players["Name"] == player_name]
    assert len(matches) > 0, (
        f"Player '{player_name}' not found in players DataFrame. "
        f"Check spelling and ensure -H/-P suffix is correct."
    )
    row = matches.iloc[0]

    is_hitter = row["player_type"] == "hitter"
    display_name = strip_name_suffix(player_name)
    position = row["Position"]
    fv = float(row["FV"])
    mew = float(row["MEW"])

    cats = list(HITTING_CATEGORIES if is_hitter else PITCHING_CATEGORIES)
    population = players[players["player_type"] == row["player_type"]]

    total_pa = my_totals["PA"]
    total_ip = my_totals["IP"]

    zscores: list[float] = []
    raw_values: list[float] = []
    for cat in cats:
        val = float(row[cat])
        raw_values.append(val)
        std = population[cat].std()
        mean = population[cat].mean()
        if cat in NEGATIVE_CATEGORIES:
            zscores.append(-(val - mean) / std)
        else:
            zscores.append((val - mean) / std)

    grad_values = [gradient[cat] for cat in cats]

    mew_contribs: list[float] = []
    for cat in cats:
        if cat == "OPS":
            c = gradient["OPS"] * row["PA"] * (row["OPS"] - my_totals["OPS"]) / total_pa
        elif cat == "ERA":
            c = gradient["ERA"] * row["IP"] * (row["ERA"] - my_totals["ERA"]) / total_ip
        elif cat == "WHIP":
            c = (
                gradient["WHIP"]
                * row["IP"]
                * (row["WHIP"] - my_totals["WHIP"])
                / total_ip
            )
        else:
            c = gradient[cat] * row[cat]
        mew_contribs.append(float(c))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    y = np.arange(len(cats))

    colors1 = [WIN_COLOR if z >= 0 else LOSS_COLOR for z in zscores]
    ax1.barh(y, zscores, color=colors1, edgecolor="white")
    ax1.set_yticks(y)
    labels1 = []
    for cat, raw in zip(cats, raw_values):
        fmt = ".3f" if cat in {"OPS", "ERA", "WHIP"} else ".0f"
        labels1.append(f"{cat} ({raw:{fmt}})")
    ax1.set_yticklabels(labels1)
    ax1.axvline(0, color="black", linewidth=0.5)
    ax1.set_title("Player Stats (z-score)")
    ax1.text(
        0.5,
        -0.12,
        f"FV = {fv:.2f}",
        transform=ax1.transAxes,
        ha="center",
        fontsize=10,
        fontweight="bold",
    )

    median_g = float(np.median(np.abs(grad_values)))
    colors2 = [WIN_COLOR if abs(g) > median_g else NEUTRAL_COLOR for g in grad_values]
    ax2.barh(y, grad_values, color=colors2, edgecolor="white")
    ax2.set_yticks(y)
    ax2.set_yticklabels(cats)
    ax2.axvline(0, color="black", linewidth=0.5)
    ax2.set_title("Gradient Weight (g_c)")
    max_g = max(abs(g) for g in grad_values) if grad_values else 1.0
    for i, g in enumerate(grad_values):
        offset = max_g * 0.05
        ax2.text(
            g + (offset if g >= 0 else -offset),
            i,
            f"{g:.2f}",
            va="center",
            fontsize=8,
            ha="left" if g >= 0 else "right",
        )

    colors3 = [WIN_COLOR if c >= 0 else LOSS_COLOR for c in mew_contribs]
    ax3.barh(y, mew_contribs, color=colors3, edgecolor="white")
    ax3.set_yticks(y)
    ax3.set_yticklabels(cats)
    ax3.axvline(0, color="black", linewidth=0.5)
    ax3.set_title("MEW Contribution")
    ax3.text(
        0.5,
        -0.12,
        f"MEW = {mew:+.2f}",
        transform=ax3.transAxes,
        ha="center",
        fontsize=10,
        fontweight="bold",
    )

    for i in range(len(cats)):
        for left_ax, right_ax in [(ax1, ax2), (ax2, ax3)]:
            con = ConnectionPatch(
                xyA=(left_ax.get_xlim()[1], y[i]),
                xyB=(right_ax.get_xlim()[0], y[i]),
                coordsA="data",
                coordsB="data",
                axesA=left_ax,
                axesB=right_ax,
                arrowstyle="->",
                color=NEUTRAL_COLOR,
                alpha=0.3,
                linewidth=0.8,
            )
            fig.add_artist(con)

    fig.suptitle(f"{display_name} \u00b7 {position}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    return fig


def plot_player_comparison_radar(
    selected_names: list[str],
    players: pd.DataFrame,
    player_type: str,
    comparison_names: list[str] | None = None,
    max_players: int = 8,
    figsize: tuple[float, float] = (5, 5),
) -> plt.Figure:
    """Radar chart comparing players' stats as league-wide percentiles.

    Percentiles computed across ALL players of the given type. For
    NEGATIVE_CATEGORIES (ERA, WHIP), lower stat → higher percentile.

    Args:
        selected_names: Internal names (with -H/-P suffix) to compare.
        players: Full DataFrame with stat columns and FV.
        player_type: "hitter" or "pitcher".
        comparison_names: Optional roster players to overlay. Drawn with
            the same color as the corresponding selected player but dashed
            linestyle, so pairs are visually linked.
        max_players: If more selected, keep top by FV.
        figsize: Figure dimensions.

    Returns:
        Figure with radar chart.
    """
    cats = HITTING_CATEGORIES if player_type == "hitter" else PITCHING_CATEGORIES
    type_df = players[players["player_type"] == player_type].copy()

    pctl_cols: list[str] = []
    for cat in cats:
        col = f"_pctl_{cat}"
        type_df[col] = type_df[cat].rank(
            pct=True, ascending=cat not in NEGATIVE_CATEGORIES
        )
        pctl_cols.append(col)

    sel = type_df[type_df["Name"].isin(selected_names)]
    assert len(sel) > 0, (
        f"None of {selected_names} found among {player_type}s in players DataFrame."
    )
    if len(sel) > max_players:
        sel = sel.nlargest(max_players, "FV")

    comparison_names = comparison_names or []
    comp = type_df[type_df["Name"].isin(comparison_names)]

    n_cats = len(cats)
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"polar": True})
    cmap = plt.cm.tab10

    # Map comparison players to their paired color index
    comp_color_map: dict[str, int] = {}
    if len(comp) > 0:
        for idx, (_, row) in enumerate(sel.iterrows()):
            if idx < len(comparison_names):
                comp_color_map[comparison_names[idx]] = idx

    for idx, (_, row) in enumerate(sel.iterrows()):
        values = [float(row[c]) for c in pctl_cols]
        values += values[:1]
        color = cmap(idx % 10)
        ax.plot(
            angles,
            values,
            "o-",
            color=color,
            linewidth=2,
            label=strip_name_suffix(row["Name"]),
        )
        ax.fill(angles, values, color=color, alpha=0.1)

    # Overlay comparison players with dashed lines, same color as their pair
    for _, row in comp.iterrows():
        values = [float(row[c]) for c in pctl_cols]
        values += values[:1]
        color_idx = comp_color_map.get(row["Name"], len(sel))
        color = cmap(color_idx % 10)
        ax.plot(
            angles,
            values,
            "--",
            color=color,
            linewidth=1.5,
            alpha=0.7,
            label=strip_name_suffix(row["Name"]) + " (roster)",
        )

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, fontsize=8)
    ax.set_ylim(0, 1)
    type_label = "Hitter" if player_type == "hitter" else "Pitcher"
    ax.set_title(f"{type_label} Comparison (league percentiles)", fontsize=10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.0), fontsize=8)
    fig.tight_layout()
    return fig


def plot_roster_value_map(
    players: pd.DataFrame,
    my_roster_names: set[str],
    my_starters: set[str],
    highlight_names: set[str] | None = None,
) -> plt.Figure:
    """Scatter plot of FV vs MEW for my roster with quadrant labels.

    Points colored by role: green = starter, blue = bench. Reference lines
    at median FV (vertical) and MEW = 0 (horizontal). Quadrants labeled:
    Core, Hidden value, Trade chip, Droppable.

    Args:
        players: Full DataFrame with FV and MEW columns.
        my_roster_names: Names on my roster.
        my_starters: Names of current starters.
        highlight_names: Optional set of names to highlight as red stars.

    Returns:
        Figure with scatter plot.
    """
    highlight_names = highlight_names or set()
    roster = players[players["Name"].isin(my_roster_names)].copy()
    is_starter = roster["Name"].isin(my_starters)
    colors = np.where(is_starter, WIN_COLOR, "#3498db")

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(
        roster["FV"].values,
        roster["MEW"].values,
        c=colors,
        s=80,
        edgecolors="white",
        zorder=5,
    )

    # Overlay highlighted players as red stars
    if highlight_names:
        hl = players[players["Name"].isin(highlight_names)]
        if len(hl) > 0:
            ax.scatter(
                hl["FV"].values,
                hl["MEW"].values,
                c="red",
                s=200,
                marker="*",
                edgecolors="darkred",
                linewidths=0.5,
                zorder=10,
            )
            for _, row in hl.iterrows():
                ax.annotate(
                    strip_name_suffix(row["Name"]),
                    (row["FV"], row["MEW"]),
                    textcoords="offset points",
                    xytext=(7, 7),
                    fontsize=8,
                    fontweight="bold",
                    color="red",
                )

    median_fv = float(roster["FV"].median())
    ax.axvline(median_fv, color=NEUTRAL_COLOR, linewidth=0.8, linestyle="--")
    ax.axhline(0, color=NEUTRAL_COLOR, linewidth=0.8, linestyle="--")

    for _, row in roster.iterrows():
        ax.annotate(
            strip_name_suffix(row["Name"]),
            (row["FV"], row["MEW"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=7,
        )

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    pad_x = (x_max - x_min) * 0.02
    pad_y = (y_max - y_min) * 0.02
    kw = {"fontsize": 11, "alpha": 0.3, "fontweight": "bold"}
    ax.text(x_max - pad_x, y_max - pad_y, "Core", ha="right", va="top", **kw)
    ax.text(x_min + pad_x, y_max - pad_y, "Hidden value", ha="left", va="top", **kw)
    ax.text(x_max - pad_x, y_min + pad_y, "Trade chip", ha="right", va="bottom", **kw)
    ax.text(x_min + pad_x, y_min + pad_y, "Droppable", ha="left", va="bottom", **kw)

    ax.set_xlabel("FV (Fantasy Value)")
    ax.set_ylabel("MEW (Marginal Expected Wins)")
    ax.set_title("Roster Value Map")

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=WIN_COLOR,
            markersize=10,
            label="Starter",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#3498db",
            markersize=10,
            label="Bench",
        ),
    ]
    if highlight_names:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                markerfacecolor="red",
                markeredgecolor="darkred",
                markersize=14,
                label="Selected",
            )
        )
    ax.legend(handles=legend_elements)
    fig.tight_layout()
    return fig


_SLOT_ORDER: list[str] = [
    "C",
    "1B",
    "2B",
    "SS",
    "3B",
    "OF",
    "UTIL",
    "SP",
    "RP",
]


def plot_starter_contributions(
    my_lineup: dict[str, str],
    players: pd.DataFrame,
) -> plt.Figure:
    """Horizontal bar chart of each starter's MEW, grouped by position slot.

    Y-axis shows "Slot: Player Name", sorted by slot order then MEW within
    multi-slot positions (OF×5, SP×5, RP×2). Bars colored by sign (green
    positive, red negative). Annotations show MEW value.

    Args:
        my_lineup: {player_name: slot} from compute_league_state.
        players: DataFrame with Name, MEW columns.

    Returns:
        Figure with horizontal bar chart.
    """
    mew_lookup = players.set_index("Name")["MEW"].to_dict()

    rows: list[tuple[str, str, float]] = []
    for name, slot in my_lineup.items():
        rows.append((slot, strip_name_suffix(name), mew_lookup.get(name, 0.0)))

    def _sort_key(row: tuple[str, str, float]) -> tuple[int, float]:
        slot_idx = _SLOT_ORDER.index(row[0]) if row[0] in _SLOT_ORDER else 99
        return (slot_idx, -row[2])

    rows.sort(key=_sort_key)

    labels = [f"{slot}: {name}" for slot, name, _ in rows]
    mew_vals = [m for _, _, m in rows]
    colors = [WIN_COLOR if m >= 0 else LOSS_COLOR for m in mew_vals]
    max_abs = max((abs(m) for m in mew_vals), default=1.0)

    n = len(rows)
    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.35)))
    y = np.arange(n)
    ax.barh(y, mew_vals, color=colors, edgecolor="white")

    for i, mew in enumerate(mew_vals):
        offset = max_abs * 0.02
        ax.text(
            mew + (offset if mew >= 0 else -offset),
            i,
            f"{mew:+.2f}",
            va="center",
            fontsize=8,
            ha="left" if mew >= 0 else "right",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("MEW (Marginal Value — higher = more impactful)")
    ax.set_title("Starter Rankings by Marginal Value")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def plot_transaction_summary(
    before_totals: dict[str, float],
    after_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    opponent_teams: list[str],
    before_ew: float,
    after_ew: float,
) -> plt.Figure:
    """Two-panel summary of a transaction's projected impact.

    Left panel: per-category delta bars (same as plot_move_impact).
    Right panel: before/after standings rank per category, showing how the
    transaction shifts your position in the league for each category.

    Args:
        before_totals: Team totals before the transaction.
        after_totals: Team totals after the transaction.
        opponent_totals: {opp_id (1-indexed): {cat: total}}.
        opponent_teams: Alphabetically sorted opponent team names.
        before_ew: Expected wins before.
        after_ew: Expected wins after.

    Returns:
        Figure with two panels.
    """
    n_opps = len(opponent_teams)

    # ── Left panel data: category deltas ──
    deltas = []
    for cat in ALL_CATEGORIES:
        if cat in NEGATIVE_CATEGORIES:
            deltas.append(before_totals[cat] - after_totals[cat])
        else:
            deltas.append(after_totals[cat] - before_totals[cat])
    delta_colors = [WIN_COLOR if d >= 0 else LOSS_COLOR for d in deltas]

    # ── Right panel data: before/after rank per category ──
    def _rank(my_val: float, cat: str) -> int:
        """1 = best. For negative cats, lower value = better rank."""
        opp_vals = [opponent_totals[i + 1][cat] for i in range(n_opps)]
        if cat in NEGATIVE_CATEGORIES:
            better_count = sum(1 for v in opp_vals if v < my_val)
        else:
            better_count = sum(1 for v in opp_vals if v > my_val)
        return better_count + 1

    before_ranks = [_rank(before_totals[cat], cat) for cat in ALL_CATEGORIES]
    after_ranks = [_rank(after_totals[cat], cat) for cat in ALL_CATEGORIES]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"wspace": 0.35})

    # ── Left: category deltas ──
    y = np.arange(len(ALL_CATEGORIES))
    max_abs = max((abs(d) for d in deltas), default=1.0)
    ax1.barh(y, deltas, color=delta_colors, edgecolor="white")
    for i, delta in enumerate(deltas):
        offset = max_abs * 0.03
        x_pos = delta + (offset if delta >= 0 else -offset)
        ax1.text(
            x_pos,
            i,
            f"{delta:+.2f}",
            va="center",
            fontsize=9,
            ha="left" if delta >= 0 else "right",
        )
    ax1.set_yticks(y)
    ax1.set_yticklabels(ALL_CATEGORIES)
    ax1.axvline(0, color="black", linewidth=0.5)
    ew_delta = after_ew - before_ew
    ax1.set_title(
        f"Category Deltas (EW {before_ew:.2f} → {after_ew:.2f}, {ew_delta:+.2f})"
    )

    # ── Right: before/after rank ──
    bar_height = 0.35
    ax2.barh(
        y - bar_height / 2,
        before_ranks,
        bar_height,
        color=NEUTRAL_COLOR,
        label="Before",
        alpha=0.6,
    )
    ax2.barh(
        y + bar_height / 2,
        after_ranks,
        bar_height,
        color=[
            WIN_COLOR if a <= b else LOSS_COLOR
            for b, a in zip(before_ranks, after_ranks)
        ],
        label="After",
    )
    for i, (br, ar) in enumerate(zip(before_ranks, after_ranks)):
        shift = ar - br
        if shift != 0:
            arrow = "↑" if shift < 0 else "↓"
            ax2.text(
                max(br, ar) + 0.3,
                i,
                f"{arrow}{abs(shift)}",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=WIN_COLOR if shift < 0 else LOSS_COLOR,
            )
    ax2.set_yticks(y)
    ax2.set_yticklabels(ALL_CATEGORIES)
    ax2.set_xlim(0, n_opps + 2)
    ax2.invert_xaxis()
    ax2.set_xlabel("Rank (1 = best)")
    ax2.set_title("Standings Rank Shift")
    ax2.legend(loc="lower left", fontsize=9)

    fig.suptitle(
        f"Transaction Impact: EW {before_ew:.2f} → {after_ew:.2f} ({ew_delta:+.2f})",
        fontsize=13,
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.90, wspace=0.4)
    return fig


def plot_player_contribution_waterfall(
    drop_names: set[str],
    add_names: set[str],
    before_totals: dict[str, float],
    after_totals: dict[str, float],
    players: pd.DataFrame,
) -> plt.Figure:
    """Stacked bar chart comparing raw stats of transacted players.

    For each scoring category, shows stacked bars: dropped players on the
    left (red, negative) and added players on the right (green, positive).
    This directly shows what the user is gaining and losing in the
    transaction, using each player's projected stats.

    The "net" annotation on each row shows the actual team-total delta
    (which accounts for lineup re-optimization and may differ from the
    raw player stat comparison).

    Counting stats (R, HR, RBI, SB, W, SV, K): each player's raw
    projected stat.

    Ratio stats (OPS, ERA, WHIP): shown as the stat value itself
    (e.g., a player with 3.20 ERA vs one with 4.10 ERA).

    All values are displayed in "positive = improvement" orientation
    (ERA/WHIP signs are flipped for display).

    Args:
        drop_names: Internal names of players leaving the roster.
        add_names: Internal names of players joining the roster.
        before_totals: Team totals before (from compute_totals_for_starters).
        after_totals: Team totals after.
        players: Full DataFrame with stat columns.

    Returns:
        Figure with stacked horizontal bars per category.
    """
    plookup = players.set_index("Name")

    if not drop_names and not add_names:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(
            0.5,
            0.5,
            "No players specified",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return fig

    _RATIO_CATS = {"OPS", "ERA", "WHIP"}
    _HITTING_CATS = set(HITTING_CATEGORIES)
    _PITCHING_CATS = set(PITCHING_CATEGORIES)

    def _relevant_for(name: str, cat: str) -> bool:
        """True if this player's type matches the category."""
        ptype = str(plookup.loc[name, "player_type"])
        if ptype == "hitter":
            return cat in _HITTING_CATS
        return cat in _PITCHING_CATS

    def _player_bars(cat: str) -> list[tuple[str, float, str]]:
        """Raw stat bars for one category, filtered by player type.

        Returns list of (display_name, display_value, "out"|"in").
        display_value is in "positive = improvement" orientation.
        Only includes players whose type matches the category
        (hitters on hitting cats, pitchers on pitching cats).
        """
        bars: list[tuple[str, float, str]] = []
        is_negative = cat in NEGATIVE_CATEGORIES

        for name in sorted(drop_names):
            if not _relevant_for(name, cat):
                continue
            val = float(plookup.loc[name, cat])
            display = val if is_negative else -val
            bars.append((strip_name_suffix(name), display, "out"))

        for name in sorted(add_names):
            if not _relevant_for(name, cat):
                continue
            val = float(plookup.loc[name, cat])
            display = -val if is_negative else val
            bars.append((strip_name_suffix(name), display, "in"))

        return bars

    all_lost_names = sorted(strip_name_suffix(n) for n in drop_names)
    all_gained_names = sorted(strip_name_suffix(n) for n in add_names)

    n_out = max(len(all_lost_names), 1)
    n_in = max(len(all_gained_names), 1)
    out_colors = {
        name: plt.cm.Reds(0.35 + 0.45 * i / n_out)
        for i, name in enumerate(all_lost_names)
    }
    in_colors = {
        name: plt.cm.Greens(0.35 + 0.45 * i / n_in)
        for i, name in enumerate(all_gained_names)
    }

    n_cats = len(ALL_CATEGORIES)
    fig, ax = plt.subplots(figsize=(14, max(5, n_cats * 0.7)))
    y_positions = np.arange(n_cats)

    for cat_idx, cat in enumerate(ALL_CATEGORIES):
        bars = _player_bars(cat)
        if not bars:
            continue

        pos_offset = 0.0
        neg_offset = 0.0

        for name, val, side in bars:
            color = out_colors[name] if side == "out" else in_colors[name]
            if val >= 0:
                ax.barh(
                    cat_idx,
                    val,
                    left=pos_offset,
                    height=0.6,
                    color=color,
                    edgecolor="white",
                    linewidth=0.5,
                )
                pos_offset += val
            else:
                ax.barh(
                    cat_idx,
                    val,
                    left=neg_offset,
                    height=0.6,
                    color=color,
                    edgecolor="white",
                    linewidth=0.5,
                )
                neg_offset += val

        # Annotate actual team-total net delta
        if cat in NEGATIVE_CATEGORIES:
            net = before_totals[cat] - after_totals[cat]
        else:
            net = after_totals[cat] - before_totals[cat]
        fmt = ".3f" if cat in _RATIO_CATS else ".1f"
        if net >= 0:
            label_x = -0.3
            label_ha = "right"
        else:
            label_x = 0.3
            label_ha = "left"
        ax.text(
            label_x,
            cat_idx,
            f"team Δ {net:{fmt}}",
            va="center",
            ha=label_ha,
            fontsize=8,
            fontweight="bold",
            color=WIN_COLOR if net >= 0 else LOSS_COLOR,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(ALL_CATEGORIES)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Player stat value (positive = improvement)")
    ax.set_title("Per-Player Stat Comparison by Category")

    legend_elements = []
    for name in all_lost_names:
        legend_elements.append(
            Line2D([0], [0], color=out_colors[name], lw=6, label=f"▼ {name} (out)")
        )
    for name in all_gained_names:
        legend_elements.append(
            Line2D([0], [0], color=in_colors[name], lw=6, label=f"▲ {name} (in)")
        )
    if legend_elements:
        ax.legend(
            handles=legend_elements,
            loc="upper right",
            fontsize=7,
            ncol=max(1, (len(legend_elements) + 3) // 4),
        )

    fig.tight_layout()
    return fig


def plot_ew_category_decomposition(
    before_totals: dict[str, float],
    after_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
    opponent_teams: list[str],
) -> plt.Figure:
    """Before/after stacked bar showing EW contribution per category.

    Each category contributes 0–N_opponents expected wins (sum of Φ(z) across
    opponents). This plot decomposes total EW into per-category slices, shown
    as a side-by-side stacked bar, so you can see where wins come from and
    how the transaction redistributes them.

    Args:
        before_totals: Team totals before.
        after_totals: Team totals after.
        opponent_totals: {opp_id: {cat: total}}.
        category_sigmas: σ_c per category.
        opponent_teams: Opponent team names (for count).

    Returns:
        Figure with paired stacked bars.
    """
    from scipy import stats as sp_stats

    def _cat_ew(my_totals: dict[str, float]) -> dict[str, float]:
        """Sum of Φ(z) across all opponents for each category."""
        cat_wins: dict[str, float] = {}
        for cat in ALL_CATEGORIES:
            sigma = max(category_sigmas[cat], 1e-6)
            denom = sigma * np.sqrt(2)
            total = 0.0
            for opp_id, opp in opponent_totals.items():
                if cat in NEGATIVE_CATEGORIES:
                    z = (opp[cat] - my_totals[cat]) / denom
                else:
                    z = (my_totals[cat] - opp[cat]) / denom
                total += float(sp_stats.norm.cdf(z))
            cat_wins[cat] = total
        return cat_wins

    before_ew = _cat_ew(before_totals)
    after_ew = _cat_ew(after_totals)

    deltas = {cat: after_ew[cat] - before_ew[cat] for cat in ALL_CATEGORIES}
    ordered = sorted(ALL_CATEGORIES, key=lambda c: deltas[c])

    total_delta = sum(deltas.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(ordered))
    vals = [deltas[c] for c in ordered]
    colors = [WIN_COLOR if v >= 0 else LOSS_COLOR for v in vals]

    ax.barh(y, vals, color=colors, height=0.6, edgecolor="white", linewidth=0.5)

    for i, (cat, v) in enumerate(zip(ordered, vals)):
        ha = "left" if v >= 0 else "right"
        offset = 0.02 if v >= 0 else -0.02
        ax.text(
            v + offset,
            i,
            f"{v:+.2f}",
            va="center",
            ha=ha,
            fontsize=9,
            fontweight="bold",
            color=colors[i],
        )

    ax.set_yticks(y)
    ax.set_yticklabels(ordered, fontsize=10)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("ΔEW (expected wins gained/lost)")
    ax.set_title(
        f"Expected Win Change by Category  (total: {total_delta:+.2f} EW)",
        fontsize=12,
        fontweight="bold",
    )

    fig.tight_layout()
    return fig


def plot_matchup_delta_heatmap(
    before_totals: dict[str, float],
    after_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    opponent_teams: list[str],
) -> plt.Figure:
    """Heatmap of how your win margin changes against each opponent per category.

    Each cell is margin_after − margin_before. Positive = you gained ground
    against that opponent in that category. Rate stats scaled ×1000.

    This is the transaction-specific complement to the dashboard's static
    category heatmap: it isolates the marginal competitive shift.

    Args:
        before_totals: Team totals before the transaction.
        after_totals: Team totals after the transaction.
        opponent_totals: {opp_id (1-indexed): {cat: total}}.
        opponent_teams: Alphabetically sorted opponent team names.

    Returns:
        Figure with annotated heatmap.
    """
    rate_cats = {"OPS", "ERA", "WHIP"}
    n_opps = len(opponent_teams)
    delta_data = np.zeros((n_opps, len(ALL_CATEGORIES)))

    for i in range(n_opps):
        opp_id = i + 1
        for j, cat in enumerate(ALL_CATEGORIES):
            opp_val = opponent_totals[opp_id][cat]
            if cat in NEGATIVE_CATEGORIES:
                margin_before = opp_val - before_totals[cat]
                margin_after = opp_val - after_totals[cat]
            else:
                margin_before = before_totals[cat] - opp_val
                margin_after = after_totals[cat] - opp_val
            delta = margin_after - margin_before
            if cat in rate_cats:
                delta *= 1000
            delta_data[i, j] = delta

    df = pd.DataFrame(delta_data, index=opponent_teams, columns=ALL_CATEGORIES)

    fig, ax = plt.subplots(figsize=(14, max(4, n_opps * 0.8 + 1)))
    sns.heatmap(
        df,
        cmap="RdYlGn",
        center=0,
        annot=True,
        fmt="+.0f",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(
        "Margin Change vs Each Opponent (positive = gained ground, rate stats ×1000)"
    )
    ax.set_ylabel("")
    fig.tight_layout()
    return fig


def plot_gap_to_ceiling(
    current_ew: float,
    ceiling_ew: float,
    starting_ew: float | None = None,
) -> plt.Figure:
    """Horizontal gauge showing current EW vs ceiling on a 0-60 scale.

    Args:
        current_ew: Current expected wins.
        ceiling_ew: Best achievable EW from full candidate pool.
        starting_ew: EW before optimizer ran (optional marker).

    Returns:
        Wide, short figure with gauge visualization.
    """
    fig, ax = plt.subplots(figsize=(10, 1.5))

    ax.barh(0, 60, color="#ecf0f1", height=0.6)
    ax.barh(0, current_ew, color=WIN_COLOR, height=0.6)

    ax.axvline(ceiling_ew, color="#2C3E50", linewidth=2, linestyle="--")
    ax.text(ceiling_ew, 0.5, f" Ceiling: {ceiling_ew:.1f}", va="bottom", fontsize=9)

    if starting_ew is not None:
        ax.axvline(starting_ew, color=LOSS_COLOR, linewidth=1.5, linestyle=":")
        ax.text(starting_ew, -0.5, f" Before: {starting_ew:.1f}", va="top", fontsize=9)

    gap = ceiling_ew - current_ew
    ax.text(
        30,
        -0.8,
        f"Gap to ceiling: {gap:.2f} EW",
        ha="center",
        fontsize=11,
        fontweight="bold",
    )

    ax.set_xlim(0, 60)
    ax.set_ylim(-1.2, 1.2)
    ax.set_yticks([])
    ax.set_xlabel("Expected Wins (0\u201360)")
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig
