import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    """Imports and setup."""
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd

    # Data loading
    from optimizer.data_loader import (
        apply_name_corrections,
        compute_all_opponent_totals,
        compute_quality_scores,
        compute_team_totals,
        convert_fantrax_rosters_from_dir,
        estimate_projection_uncertainty,
        load_all_data,
        load_projections,
    )

    # Free agent optimizer
    from optimizer.roster_optimizer import (
        build_and_solve_milp,
        compute_player_sensitivity,
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
        plot_player_contribution_radar,
        plot_player_sensitivity,
        plot_player_value_scatter,
        plot_roster_changes,
        plot_team_radar,
        plot_trade_impact,
        plot_win_matrix,
        plot_win_probability_breakdown,
    )
    return (
        apply_name_corrections,
        build_and_solve_milp,
        compute_all_opponent_totals,
        compute_player_sensitivity,
        compute_player_values,
        compute_quality_scores,
        compute_roster_change_values,
        compute_roster_situation,
        compute_team_totals,
        convert_fantrax_rosters_from_dir,
        filter_candidates,
        generate_trade_candidates,
        load_all_data,
        load_projections,
        mo,
        pd,
        plot_category_contributions,
        plot_category_margins,
        plot_constraint_analysis,
        plot_player_contribution_radar,
        plot_player_sensitivity,
        plot_player_value_scatter,
        plot_roster_changes,
        plot_team_radar,
        plot_win_matrix,
        plot_win_probability_breakdown,
        print_roster_summary,
        print_trade_report,
    )


@app.cell
def _():
    """Configuration - File paths."""
    DATA_DIR = "data/"
    RAW_ROSTERS_DIR = DATA_DIR + "raw_rosters/"
    HITTER_PROJ_PATH = DATA_DIR + "fangraphs-steamer-projections-hitters.csv"
    PITCHER_PROJ_PATH = DATA_DIR + "fangraphs-steamer-projections-pitchers.csv"
    DB_PATH = "../mlb_player_comps_dashboard/mlb_stats.db"
    return DB_PATH, HITTER_PROJ_PATH, PITCHER_PROJ_PATH, RAW_ROSTERS_DIR


@app.cell
def _(RAW_ROSTERS_DIR, convert_fantrax_rosters_from_dir, mo):
    """Convert Fantrax roster exports to pipeline format."""
    mo.md("## 1. Data Pipeline")

    # Convert raw Fantrax exports to pipeline format
    my_roster_path, opponent_rosters_path = convert_fantrax_rosters_from_dir(
        raw_rosters_dir=RAW_ROSTERS_DIR,
        my_team_filename="my_team.csv",
    )
    return my_roster_path, opponent_rosters_path


@app.cell
def _(
    DB_PATH,
    HITTER_PROJ_PATH,
    PITCHER_PROJ_PATH,
    apply_name_corrections,
    load_projections,
    my_roster_path,
    opponent_rosters_path,
):
    """Apply name corrections to roster files."""
    # Load projections temporarily for name matching
    _projections_temp = load_projections(HITTER_PROJ_PATH, PITCHER_PROJ_PATH, DB_PATH)

    # Auto-correct accents, apostrophes, known mismatches
    apply_name_corrections(my_roster_path, _projections_temp)
    apply_name_corrections(
        opponent_rosters_path, _projections_temp, is_opponent_file=True
    )
    return


@app.cell
def _(
    DB_PATH,
    HITTER_PROJ_PATH,
    PITCHER_PROJ_PATH,
    compute_all_opponent_totals,
    compute_team_totals,
    load_all_data,
    my_roster_path,
    opponent_rosters_path,
):
    """Load all validated data."""
    projections, my_roster_names, opponent_rosters = load_all_data(
        HITTER_PROJ_PATH,
        PITCHER_PROJ_PATH,
        my_roster_path,
        opponent_rosters_path,
        DB_PATH,
    )

    # Compute opponent totals
    opponent_totals = compute_all_opponent_totals(opponent_rosters, projections)

    # Compute my current totals
    my_totals = compute_team_totals(my_roster_names, projections)
    return (
        my_roster_names,
        my_totals,
        opponent_rosters,
        opponent_totals,
        projections,
    )


@app.cell
def _(mo, opponent_totals, pd):
    """Display opponent totals summary."""
    mo.md("## 2. Opponent Analysis")
    pd.DataFrame(opponent_totals).T.round(2)
    return


@app.cell
def _(mo, my_totals, opponent_totals, plot_team_radar):
    """Team comparison radar chart."""
    mo.md("### Team Comparison Radar")
    fig_radar = plot_team_radar(my_totals, opponent_totals)
    fig_radar
    return


@app.cell
def _(mo, my_totals, opponent_totals, plot_win_matrix):
    """Win/loss matrix heatmap."""
    mo.md("### Win/Loss Matrix")
    fig_matrix = plot_win_matrix(my_totals, opponent_totals)
    fig_matrix
    return


@app.cell
def _(mo, my_totals, opponent_totals, plot_category_margins):
    """Category margins bar chart."""
    mo.md("### Category Margins")
    fig_margins = plot_category_margins(my_totals, opponent_totals)
    fig_margins
    return


@app.cell
def _(
    compute_quality_scores,
    filter_candidates,
    mo,
    my_roster_names,
    opponent_rosters,
    projections,
):
    """Filter candidates for optimization."""
    mo.md("## 3. Free Agent Optimizer")

    # Compute quality scores for prefiltering
    quality_scores = compute_quality_scores(projections)

    # Get all opponent player names (unavailable)
    opponent_roster_names = set().union(*opponent_rosters.values())

    # Filter to optimization candidates
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
def _(build_and_solve_milp, candidates, mo, my_roster_names, opponent_totals):
    """Solve the MILP for optimal roster."""
    mo.md("### Running Optimizer...")

    optimal_roster_names, solution_info = build_and_solve_milp(
        candidates,
        opponent_totals,
        my_roster_names,
    )

    print(f"\nObjective: {solution_info['objective']}/60 wins")
    print(f"Solve time: {solution_info['solve_time']:.1f}s")
    print(f"Status: {solution_info['status']}")
    return (optimal_roster_names,)


@app.cell
def _(
    compute_team_totals,
    my_roster_names,
    opponent_totals,
    optimal_roster_names,
    print_roster_summary,
    projections,
):
    """Print optimal roster summary."""
    optimal_totals = compute_team_totals(optimal_roster_names, projections)

    print_roster_summary(
        optimal_roster_names,
        projections,
        optimal_totals,
        opponent_totals,
        old_roster_names=my_roster_names,
    )
    return (optimal_totals,)


@app.cell
def _(
    compute_roster_change_values,
    mo,
    my_roster_names,
    opponent_totals,
    optimal_roster_names,
    plot_roster_changes,
    projections,
):
    """Visualize roster changes with WPA-based priority."""
    mo.md("### Waiver Priority List")

    added = set(optimal_roster_names) - my_roster_names
    dropped = my_roster_names - set(optimal_roster_names)

    added_df, dropped_df = compute_roster_change_values(
        added, dropped, my_roster_names, projections, opponent_totals
    )
    fig_changes = plot_roster_changes(added_df, dropped_df)
    fig_changes
    return


@app.cell
def _(
    compute_roster_situation,
    mo,
    opponent_totals,
    optimal_roster_names,
    projections,
):
    """Analyze post-free-agency roster situation for trade analysis."""
    mo.md("## 4. Trade Analysis (from optimized roster)")

    # Use optimized roster for trade analysis - this is post-free-agency
    trade_roster_names = set(optimal_roster_names)
    situation = compute_roster_situation(
        trade_roster_names, projections, opponent_totals
    )
    category_sigmas = situation["category_sigmas"]

    print(f"Win probability: {situation['win_probability']:.1%}")
    print(f"Expected wins: {situation['expected_wins']:.1f}/60")
    print(
        f"\nStrengths: {', '.join(situation['strengths']) if situation['strengths'] else 'None'}"
    )
    print(
        f"Weaknesses: {', '.join(situation['weaknesses']) if situation['weaknesses'] else 'None'}"
    )
    return category_sigmas, situation, trade_roster_names


@app.cell
def _(mo, plot_win_probability_breakdown, situation):
    """Visualize win probability breakdown."""
    mo.md("### Win Probability Breakdown")
    fig_wp = plot_win_probability_breakdown(situation["diagnostics"])
    fig_wp
    return


@app.cell
def _(
    category_sigmas,
    compute_player_values,
    mo,
    opponent_roster_names,
    opponent_totals,
    optimal_totals,
    projections,
    trade_roster_names,
):
    """Compute player values for trade analysis (from optimized roster)."""
    mo.md("### Player Values")

    # Include optimized roster + all opponent rosters
    all_roster_names = trade_roster_names | opponent_roster_names

    player_values = compute_player_values(
        player_names=all_roster_names,
        my_roster_names=trade_roster_names,
        projections=projections,
        my_totals=optimal_totals,
        opponent_totals=opponent_totals,
        category_sigmas=category_sigmas,
    )

    # Show top players
    player_values.head(20)
    return (player_values,)


@app.cell
def _(mo, player_values, plot_player_value_scatter, trade_roster_names):
    """Scatter plot of player values."""
    mo.md("### Player Value Scatter")
    fig_scatter = plot_player_value_scatter(player_values, trade_roster_names)
    fig_scatter
    return


@app.cell
def _(
    category_sigmas,
    generate_trade_candidates,
    mo,
    opponent_rosters,
    opponent_totals,
    optimal_totals,
    player_values,
    projections,
    trade_roster_names,
):
    """Generate trade candidates (from optimized roster)."""
    mo.md("### Trade Recommendations")

    trade_candidates = generate_trade_candidates(
        my_roster_names=trade_roster_names,
        player_values=player_values,
        opponent_rosters=opponent_rosters,
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
def _(player_values, print_trade_report, situation, trade_candidates):
    """Print trade recommendations report."""
    print_trade_report(situation, trade_candidates, player_values, top_n=5)
    return


@app.cell
def _(mo, my_roster_names, plot_category_contributions, projections):
    """Category contribution analysis - Stolen Bases."""
    mo.md("## 5. Deep Dive Analysis")
    mo.md("### Stolen Bases Contributions")
    fig_sb = plot_category_contributions(list(my_roster_names), projections, "HR")
    fig_sb
    return


@app.cell
def _(mo, my_roster_names, plot_player_contribution_radar, projections):
    """Hitter contribution radar."""
    mo.md("### Hitter Contributions")
    fig_hitter_radar = plot_player_contribution_radar(
        list(my_roster_names), projections, "hitter", top_n=10
    )
    fig_hitter_radar
    return


@app.cell
def _(mo, my_roster_names, plot_player_contribution_radar, projections):
    """Pitcher contribution radar."""
    mo.md("### Pitcher Contributions")
    fig_pitcher_radar = plot_player_contribution_radar(
        list(my_roster_names), projections, "pitcher", top_n=10
    )
    fig_pitcher_radar
    return


@app.cell
def _(mo, optimal_roster_names, plot_constraint_analysis, projections):
    """Constraint analysis."""
    mo.md("### Constraint Analysis")
    fig_constraints = plot_constraint_analysis(optimal_roster_names, projections)
    fig_constraints
    return


@app.cell
def _(
    candidates,
    compute_player_sensitivity,
    mo,
    opponent_totals,
    optimal_roster_names,
    plot_player_sensitivity,
):
    """Sensitivity analysis placeholder."""
    mo.md("""
    ### Sensitivity Analysis (Optional - Slow)

    *Uncomment the code below to run sensitivity analysis. This takes 5-15 minutes.*
    """)

    # Uncomment to run sensitivity analysis:
    sensitivity = compute_player_sensitivity(optimal_roster_names, candidates, opponent_totals)
    fig_sensitivity = plot_player_sensitivity(sensitivity)
    fig_sensitivity
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
