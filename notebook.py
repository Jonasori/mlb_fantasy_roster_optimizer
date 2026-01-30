import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    refresh = mo.ui.refresh(label="Refresh", options=["1m", "10m", "1hr"])
    return mo, refresh


@app.cell
def _(mo, refresh):
    mo.hstack([refresh, refresh.value])
    return


@app.cell
def imports():
    import matplotlib.pyplot as plt
    import pandas as pd

    # Data utilities (constants from data_loader, the single source of truth)
    from optimizer.data_loader import (
        FANTRAX_TEAM_IDS,
        MY_TEAM_NAME,
        NUM_OPPONENTS,
        compute_all_opponent_totals,
        compute_quality_scores,
        compute_team_totals,
        estimate_projection_uncertainty,
    )

    # Database is the primary data source
    from optimizer.database import (
        get_free_agents,
        get_projections,
        get_roster_names,
        refresh_all_data,
    )

    # Free agent optimizer
    from optimizer.roster_optimizer import (
        build_and_solve_milp,
        compute_percentile_sensitivity,
        compute_position_sensitivity,
        compute_roster_change_values,
        compute_standings,
        filter_candidates,
        print_roster_summary,
    )

    # Trade engine
    from optimizer.trade_engine import (
        compute_player_values,
        compute_roster_situation,
        compute_win_probability,
        evaluate_trade,
        generate_trade_candidates,
        identify_trade_pieces,
        identify_trade_targets,
        print_trade_report,
        verify_trade_impact,
    )

    # Visualizations
    from optimizer.visualizations import (
        plot_category_contributions,
        plot_category_margins,
        plot_constraint_analysis,
        plot_percentile_ewa_curves,
        plot_player_contribution_radar,
        plot_player_sensitivity,
        plot_player_value_scatter,
        plot_position_distributions,
        plot_position_sensitivity_dashboard,
        plot_roster_changes,
        plot_team_radar,
        plot_trade_impact,
        plot_upgrade_opportunities,
        plot_win_matrix,
        plot_win_probability_breakdown,
    )
    return (
        build_and_solve_milp,
        compute_all_opponent_totals,
        compute_percentile_sensitivity,
        compute_player_values,
        compute_position_sensitivity,
        compute_quality_scores,
        compute_roster_change_values,
        compute_roster_situation,
        compute_team_totals,
        estimate_projection_uncertainty,
        filter_candidates,
        generate_trade_candidates,
        plot_category_contributions,
        plot_category_margins,
        plot_constraint_analysis,
        plot_percentile_ewa_curves,
        plot_player_contribution_radar,
        plot_player_value_scatter,
        plot_position_distributions,
        plot_position_sensitivity_dashboard,
        plot_roster_changes,
        plot_team_radar,
        plot_upgrade_opportunities,
        plot_win_matrix,
        plot_win_probability_breakdown,
        print_roster_summary,
        print_trade_report,
        refresh_all_data,
    )


@app.cell
def config():
    from pathlib import Path

    # File paths for FanGraphs CSVs (input to database)
    DATA_DIR = "data/"
    HITTER_PROJ_PATH = DATA_DIR + "fangraphs-atc-projections-hitters.csv"
    PITCHER_PROJ_PATH = DATA_DIR + "fangraphs-atc-projections-pitchers.csv"
    DB_PATH = DATA_DIR + "optimizer.db"

    # =======================================================
    # CONFIGURATION FLAGS
    # =======================================================
    SKIP_FANTRAX = True  # Set False to fetch rosters from Fantrax API
    SKIP_MLB_API = True  # Set True to skip MLB API age fetching (faster)
    # =======================================================

    # Check if cookies exist
    cookie_file = Path(DATA_DIR) / "fantrax_cookies.json"
    if not cookie_file.exists():
        print("⚠️  WARNING: Fantrax cookies file not found!")
        print(f"   Expected at: {cookie_file}")
        print("   To set up cookies:")
        print("   1. Log into https://www.fantrax.com")
        print("   2. Open DevTools → Application → Cookies")
        print("   3. Copy JSESSIONID and FX_RM values")
        print(f"   4. Save to {cookie_file} as JSON:")
        print('      {"JSESSIONID": "...", "FX_RM": "..."}')
        SKIP_FANTRAX = True
    elif SKIP_FANTRAX:
        print("ℹ️  SKIP_FANTRAX=True — using cached database data")
    else:
        print("✅ Fantrax cookies found — will fetch fresh roster data")

    if SKIP_MLB_API:
        print("ℹ️  SKIP_MLB_API=True — using cached ages")
    else:
        print("✅ Will fetch ages from MLB Stats API")
    return (
        DB_PATH,
        HITTER_PROJ_PATH,
        PITCHER_PROJ_PATH,
        SKIP_FANTRAX,
        SKIP_MLB_API,
    )


@app.cell
def load_data(
    DB_PATH,
    HITTER_PROJ_PATH,
    PITCHER_PROJ_PATH,
    SKIP_FANTRAX,
    SKIP_MLB_API,
    mo,
    refresh_all_data,
):
    mo.md("## Data Loading")

    # Refresh all data: FanGraphs + MLB API + Fantrax → database
    data = refresh_all_data(
        hitter_proj_path=HITTER_PROJ_PATH,
        pitcher_proj_path=PITCHER_PROJ_PATH,
        db_path=DB_PATH,
        skip_fantrax=SKIP_FANTRAX,
        skip_mlb_api=SKIP_MLB_API,
    )

    # Extract what we need (all from database queries)
    projections = data["projections"]
    my_roster_names = data["my_roster"]
    opponent_rosters = data["opponent_rosters"]

    print(f"\nProjections: {len(projections)} players")
    print(f"My roster: {len(my_roster_names)} players")
    print(f"Opponents: {len(opponent_rosters)} teams")

    # Check age coverage
    with_age = projections["age"].notna().sum()
    print(
        f"Players with age: {with_age}/{len(projections)} ({100 * with_age / len(projections):.0f}%)"
    )

    if len(my_roster_names) == 0:
        print("\n⚠️  No roster data! Set SKIP_FANTRAX=False in the config cell above.")
    return my_roster_names, opponent_rosters, projections


@app.cell
def compute_totals(
    compute_all_opponent_totals,
    compute_team_totals,
    estimate_projection_uncertainty,
    my_roster_names,
    opponent_rosters,
    projections,
):
    assert len(my_roster_names) > 0, "No roster data. Set SKIP_FANTRAX=False in config."

    my_totals = compute_team_totals(my_roster_names, projections)

    # Convert opponent rosters to dict[int, set[str]] format
    opponent_rosters_indexed = {
        i + 1: names for i, (team, names) in enumerate(opponent_rosters.items())
    }
    opponent_totals = compute_all_opponent_totals(opponent_rosters_indexed, projections)

    # Compute category sigmas for win probability and sensitivity analysis
    category_sigmas = estimate_projection_uncertainty(my_totals, opponent_totals)
    return (
        category_sigmas,
        my_totals,
        opponent_rosters_indexed,
        opponent_totals,
    )


@app.cell
def team_radar(mo, my_totals, opponent_totals, plot_team_radar):
    mo.vstack(
        [
            mo.md("## Team Comparison"),
            plot_team_radar(my_totals, opponent_totals),
        ]
    )
    return


@app.cell
def win_matrix(mo, my_totals, opponent_totals, plot_win_matrix):
    mo.vstack(
        [
            mo.md("## Win/Loss Matrix"),
            plot_win_matrix(my_totals, opponent_totals),
        ]
    )
    return


@app.cell
def category_margins(mo, my_totals, opponent_totals, plot_category_margins):
    mo.vstack(
        [
            mo.md("## Category Margins"),
            plot_category_margins(my_totals, opponent_totals),
        ]
    )
    return


@app.cell
def filter_cands(
    compute_quality_scores,
    filter_candidates,
    mo,
    my_roster_names,
    opponent_rosters,
    projections,
):
    mo.md("## Free Agent Optimizer")

    assert len(my_roster_names) > 0, "No roster data. Set SKIP_FANTRAX=False in config."

    quality_scores = compute_quality_scores(projections)
    opponent_roster_names = set().union(*opponent_rosters.values())

    candidates = filter_candidates(
        projections,
        quality_scores,
        my_roster_names,
        opponent_roster_names,
        top_n_per_position=30,
        top_n_per_category=10,
    )
    return candidates, opponent_roster_names


@app.cell
def solve_milp(
    build_and_solve_milp,
    candidates,
    mo,
    my_roster_names,
    opponent_totals,
):
    mo.md("### Running Optimizer...")

    assert len(candidates) > 0, "No candidates available for optimization"

    optimal_roster_names, solution_info = build_and_solve_milp(
        candidates,
        opponent_totals,
        my_roster_names,
    )

    print(f"Total wins: {solution_info['total_wins']}/60")
    print(
        f"Balance: range {solution_info['win_range']} ({solution_info['w_min']}-{solution_info['w_max']}), λ={solution_info['balance_lambda']}"
    )
    print(f"Solve time: {solution_info['solve_time']:.1f}s")
    print(f"Status: {solution_info['status']}")
    return optimal_roster_names, solution_info


@app.cell
def roster_summary(
    compute_team_totals,
    my_roster_names,
    opponent_totals,
    optimal_roster_names,
    print_roster_summary,
    projections,
    solution_info,
):
    optimal_totals = compute_team_totals(optimal_roster_names, projections)

    print_roster_summary(
        optimal_roster_names,
        projections,
        optimal_totals,
        opponent_totals,
        old_roster_names=my_roster_names,
        solution_info=solution_info,
    )
    return (optimal_totals,)


@app.cell
def roster_changes_viz(
    compute_roster_change_values,
    mo,
    my_roster_names,
    opponent_totals,
    optimal_roster_names,
    plot_roster_changes,
    projections,
):
    added = set(optimal_roster_names) - my_roster_names
    dropped = my_roster_names - set(optimal_roster_names)

    if added or dropped:
        print(f"Players to add: {len(added)}")
        print(f"Players to drop: {len(dropped)}")
        added_df, dropped_df = compute_roster_change_values(
            added, dropped, my_roster_names, projections, opponent_totals
        )
        output = mo.vstack(
            [
                mo.md("### Waiver Priority List"),
                plot_roster_changes(added_df, dropped_df),
            ]
        )
    else:
        output = mo.md("### Waiver Priority List\n\n*No roster changes needed*")
    output
    return


@app.cell
def trade_analysis_setup(
    compute_roster_situation,
    mo,
    opponent_totals,
    optimal_roster_names,
    projections,
):
    mo.md("## Trade Analysis (from optimized roster)")

    trade_roster_names = set(optimal_roster_names)
    situation = compute_roster_situation(
        trade_roster_names, projections, opponent_totals
    )

    print(f"Win probability: {situation['win_probability']:.1%}")
    print(f"Expected wins: {situation['expected_wins']:.1f}/60")
    print(f"Strengths: {', '.join(situation['strengths']) or 'None'}")
    print(f"Weaknesses: {', '.join(situation['weaknesses']) or 'None'}")
    return situation, trade_roster_names


@app.cell
def win_prob_breakdown(mo, plot_win_probability_breakdown, situation):
    mo.vstack(
        [
            mo.md("### Win Probability Breakdown"),
            plot_win_probability_breakdown(situation["diagnostics"]),
        ]
    )
    return


@app.cell
def player_values_calc(
    category_sigmas,
    compute_player_values,
    mo,
    opponent_roster_names,
    opponent_totals,
    optimal_totals,
    projections,
    trade_roster_names,
):
    mo.md("### Player Values")

    all_roster_names = trade_roster_names | opponent_roster_names

    player_values = compute_player_values(
        player_names=all_roster_names,
        my_roster_names=trade_roster_names,
        projections=projections,
        my_totals=optimal_totals,
        opponent_totals=opponent_totals,
        category_sigmas=category_sigmas,
    )

    player_values.head(20)
    return (player_values,)


@app.cell
def player_value_scatter_viz(
    mo,
    player_values,
    plot_player_value_scatter,
    trade_roster_names,
):
    mo.vstack(
        [
            mo.md("### Player Value Scatter"),
            plot_player_value_scatter(player_values, trade_roster_names),
        ]
    )
    return


@app.cell
def trade_candidates_gen(
    category_sigmas,
    generate_trade_candidates,
    mo,
    opponent_rosters_indexed,
    opponent_totals,
    optimal_totals,
    player_values,
    projections,
    trade_roster_names,
):
    mo.md("### Trade Recommendations")

    trade_candidates = generate_trade_candidates(
        my_roster_names=trade_roster_names,
        player_values=player_values,
        opponent_rosters=opponent_rosters_indexed,
        projections=projections,
        my_totals=optimal_totals,
        opponent_totals=opponent_totals,
        category_sigmas=category_sigmas,
        max_send=2,
        max_receive=2,
        n_targets=20,
        n_pieces=20,
        n_candidates=30,
    )
    return (trade_candidates,)


@app.cell
def trade_report(
    player_values,
    print_trade_report,
    situation,
    trade_candidates,
):
    print_trade_report(situation, trade_candidates, player_values, top_n=5)
    return


@app.cell
def deep_dive_hr(
    mo,
    my_roster_names,
    plot_category_contributions,
    projections,
):
    mo.vstack(
        [
            mo.md("## Deep Dive Analysis"),
            mo.md("### Home Run Contributions"),
            plot_category_contributions(list(my_roster_names), projections, "HR"),
        ]
    )
    return


@app.cell
def hitter_radar(
    mo,
    my_roster_names,
    plot_player_contribution_radar,
    projections,
):
    mo.vstack(
        [
            mo.md("### Hitter Contributions"),
            plot_player_contribution_radar(
                list(my_roster_names), projections, "hitter", top_n=10
            ),
        ]
    )
    return


@app.cell
def constraint_viz(
    mo,
    optimal_roster_names,
    plot_constraint_analysis,
    projections,
):
    mo.vstack(
        [
            mo.md("### Constraint Analysis"),
            plot_constraint_analysis(optimal_roster_names, projections),
        ]
    )
    return


@app.cell
def position_sensitivity_header(mo):
    mo.md("""
    ## Position Sensitivity Analysis
    """)
    return


@app.cell
def position_sensitivity_compute(
    category_sigmas,
    compute_percentile_sensitivity,
    compute_position_sensitivity,
    mo,
    my_roster_names,
    opponent_roster_names,
    opponent_totals,
    projections,
):
    """Compute position-by-position sensitivity analysis."""
    mo.md("Computing position sensitivities...")

    # Compute position sensitivity (EWA for swaps at each position)
    sensitivity_results = compute_position_sensitivity(
        my_roster_names=my_roster_names,
        opponent_roster_names=opponent_roster_names,
        projections=projections,
        opponent_totals=opponent_totals,
        category_sigmas=category_sigmas,
    )

    slot_data = sensitivity_results["slot_data"]
    ewa_df = sensitivity_results["ewa_df"]
    sensitivity_df = sensitivity_results["sensitivity_df"]
    baseline_ew = sensitivity_results["baseline_expected_wins"]

    # Compute percentile sensitivity (EWA vs percentile curves)
    pctl_ewa_df = compute_percentile_sensitivity(
        my_roster_names=my_roster_names,
        projections=projections,
        opponent_totals=opponent_totals,
        category_sigmas=category_sigmas,
        slot_data=slot_data,
        baseline_expected_wins=baseline_ew,
    )

    print(f"\nBaseline expected wins: {baseline_ew:.1f} / 60")
    print("\nPosition Upgrade Opportunities:")
    print(
        sensitivity_df[
            [
                "slot",
                "my_worst_name",
                "better_fas_count",
                "best_fa_sgp_gap",
                "best_fa_ewa",
            ]
        ].to_string(index=False)
    )
    return ewa_df, pctl_ewa_df, sensitivity_df, slot_data


@app.cell
def position_sensitivity_dashboard(
    ewa_df,
    mo,
    plot_position_sensitivity_dashboard,
    sensitivity_df,
    slot_data,
):
    """4-panel dashboard showing position sensitivity analysis."""
    mo.vstack(
        [
            mo.md("### Position Sensitivity Dashboard"),
            mo.md(
                "Shows which positions offer the most EWA per SGP, "
                "best free agent upgrades, and position scarcity curves."
            ),
            plot_position_sensitivity_dashboard(ewa_df, sensitivity_df, slot_data),
        ]
    )
    return


@app.cell
def upgrade_opportunities_viz(mo, plot_upgrade_opportunities, slot_data):
    """Bar chart showing SGP gap between best FA and my worst player."""
    mo.vstack(
        [
            mo.md("### Upgrade Opportunities by Position"),
            mo.md(
                "SGP difference between the best available free agent "
                "and my worst rostered player at each position."
            ),
            plot_upgrade_opportunities(slot_data),
        ]
    )
    return


@app.cell
def percentile_ewa_curves_viz(mo, pctl_ewa_df, plot_percentile_ewa_curves):
    """EWA vs percentile curves for each position."""
    mo.vstack(
        [
            mo.md("### EWA vs Percentile Curves"),
            mo.md(
                "Shows how much EWA you gain from upgrading to players at different "
                "percentile levels. The star shows your current player."
            ),
            plot_percentile_ewa_curves(pctl_ewa_df),
        ]
    )
    return


@app.cell
def position_distributions_viz(mo, plot_position_distributions, slot_data):
    """Boxplot distributions of SGP by position."""
    mo.vstack(
        [
            mo.md("### Player Distribution by Position"),
            mo.md(
                "SGP distributions for each position. "
                "Red dots indicate your current rostered players."
            ),
            plot_position_distributions(slot_data),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
