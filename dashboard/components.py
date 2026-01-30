"""
Reusable UI components for the Streamlit dashboard.
"""

import io
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


def display_figure(fig: plt.Figure, width: int = 400) -> None:
    """
    Display a matplotlib figure at a specific pixel width.

    This gives precise control over display size while maintaining
    high resolution from the underlying figure.

    Args:
        fig: matplotlib Figure object
        width: display width in pixels (default 400)
    """
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    buf.seek(0)
    st.image(buf, width=width)
    plt.close(fig)


def metric_card(label: str, value: str, delta: str | None = None) -> None:
    """
    Display a metric in a styled card.

    Example:
        metric_card("Win Probability", "31.4%", "+2.3%")
    """
    if delta:
        delta_color = "green" if delta.startswith("+") else "red"
        st.metric(label=label, value=value, delta=delta)
    else:
        st.metric(label=label, value=value)


def player_table(
    df: pd.DataFrame,
    columns: list[str],
    sortable: bool = True,
    on_row_click: Callable | None = None,
) -> None:
    """
    Display a formatted player table with optional sorting.
    """
    display_df = df[columns].copy()

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
    )


def category_rank_chart(
    ranks: dict[str, int], title: str = "Category Ranks"
) -> plt.Figure:
    """
    Horizontal bar chart of category ranks (1-7).
    Color-coded by rank tier.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = list(ranks.keys())
    values = list(ranks.values())

    colors = []
    for rank in values:
        if rank <= 2:
            colors.append("#2ECC71")  # Green
        elif rank <= 4:
            colors.append("#F1C40F")  # Yellow
        else:
            colors.append("#E74C3C")  # Red

    ax.barh(categories, values, color=colors)
    ax.set_xlim(0, 7)
    ax.set_xlabel("Rank")
    ax.set_title(title)
    ax.invert_xaxis()  # 1st is best

    return fig


def radar_chart_with_overlay(
    current_totals: dict[str, float],
    after_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    categories: list[str],
) -> plt.Figure:
    """
    Radar chart comparing current team vs simulated team.

    Normalization: Min-max normalized against LEAGUE values only (current + opponents).
    This ensures both "current" and "after" are compared to the same baseline.
    Values show where you stand relative to the league (0 = worst, 1 = best).

    For negative categories (ERA, WHIP), lower is better so normalization is inverted.

    Uses shared sorting logic from visualizations module:
    - Hitting: sorted descending (best first, clockwise from top)
    - Pitching: sorted ascending (worst first, so best ends adjacent to hitting's best)
    """
    import numpy as np

    from optimizer.data_loader import NEGATIVE_CATEGORIES
    from optimizer.visualizations import sort_categories_for_radar

    # Use shared sorting logic (current_totals is "my team" for sorting purposes)
    sorted_categories = sort_categories_for_radar(
        current_totals, opponent_totals, categories
    )

    n_categories = len(sorted_categories)
    angles = np.linspace(0, 2 * np.pi, n_categories, endpoint=False).tolist()
    angles += angles[:1]

    # High-resolution figure (display size controlled by Streamlit)
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Compute league min/max for normalization (EXCLUDE after_totals from baseline)
    league_min = {}
    league_max = {}
    for cat in sorted_categories:
        # League = current team + all opponents
        league_vals = [current_totals[cat]] + [
            opp[cat] for opp in opponent_totals.values()
        ]
        league_min[cat] = min(league_vals)
        league_max[cat] = max(league_vals)

    # Helper to normalize values against league baseline
    def normalize(totals):
        values = []
        for cat in sorted_categories:
            min_val = league_min[cat]
            max_val = league_max[cat]

            if max_val > min_val:
                norm = (totals[cat] - min_val) / (max_val - min_val)
                # For negative categories (ERA, WHIP), lower is better
                if cat in NEGATIVE_CATEGORIES:
                    norm = 1 - norm
            else:
                norm = 0.5

            # Clamp to [0, 1] in case after_totals is outside league range
            norm = max(0, min(1, norm))
            values.append(norm)
        return values + values[:1]

    # Plot opponents first (faded background)
    for opp_id, opp_totals in opponent_totals.items():
        opp_vals = normalize(opp_totals)
        ax.plot(angles, opp_vals, "-", linewidth=0.5, alpha=0.2, color="gray")

    # Plot current (blue, solid)
    current_vals = normalize(current_totals)
    ax.plot(angles, current_vals, "o-", linewidth=2, label="Current", color="blue")
    ax.fill(angles, current_vals, alpha=0.15, color="blue")

    # Plot after (green, dashed)
    after_vals = normalize(after_totals)
    ax.plot(angles, after_vals, "o--", linewidth=2, label="After", color="green")
    ax.fill(angles, after_vals, alpha=0.15, color="green")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(sorted_categories, fontsize=11)
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

    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.0), fontsize=10)

    plt.tight_layout()
    return fig
