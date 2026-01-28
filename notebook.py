import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Dynasty Roto Roster Optimizer

    This notebook optimizes a 26-player fantasy baseball roster using Mixed-Integer
    Linear Programming (MILP) to maximize wins against opponents across 10 scoring categories.

    **Key Features:**
    - All rostered players' stats count (not just starters)
    - Starting lineup slots enforce positional validity
    - Optimizes across 10 roto categories: R, HR, RBI, SB, OPS, W, SV, K, ERA, WHIP
    """)
    return


@app.cell
def _():
    # Imports
    import pandas as pd
    import matplotlib.pyplot as plt
    from optimizer.roster_optimizer import (
        # Fantrax roster conversion
        convert_fantrax_rosters_from_dir,
        load_projections,
        apply_name_corrections,
        # Data loading
        load_all_data,
        compute_all_opponent_totals,
        compute_team_totals,
        compute_quality_scores,
        filter_candidates,
        build_and_solve_milp,
        compute_standings,
        print_roster_summary,
    )
    from optimizer.visualizations import (
        plot_team_radar,
        plot_category_margins,
        plot_win_matrix,
        plot_category_contributions,
        plot_roster_changes,
        compute_player_sensitivity,
        plot_player_sensitivity,
        plot_constraint_analysis,
    )
    return (
        apply_name_corrections,
        build_and_solve_milp,
        compute_all_opponent_totals,
        compute_player_sensitivity,
        compute_quality_scores,
        compute_team_totals,
        convert_fantrax_rosters_from_dir,
        filter_candidates,
        load_all_data,
        load_projections,
        pd,
        plot_category_contributions,
        plot_category_margins,
        plot_constraint_analysis,
        plot_player_sensitivity,
        plot_roster_changes,
        plot_team_radar,
        plot_win_matrix,
        print_roster_summary,
    )


@app.cell
def _(mo):
    mo.md("""
    ## Configuration
    """)
    return


@app.cell
def _():
    # File paths
    DATA_DIR = "data/"
    RAW_ROSTERS_DIR = DATA_DIR + "raw_rosters/"
    HITTER_PROJ_PATH = DATA_DIR + "fangraphs-steamer-projections-hitters.csv"
    PITCHER_PROJ_PATH = DATA_DIR + "fangraphs-steamer-projections-pitchers.csv"
    MY_ROSTER_PATH = DATA_DIR + "my-roster.csv"
    OPPONENT_ROSTERS_PATH = DATA_DIR + "opponent-rosters.csv"
    DB_PATH = "../mlb_player_comps_dashboard/mlb_stats.db"
    return (
        DB_PATH,
        HITTER_PROJ_PATH,
        MY_ROSTER_PATH,
        OPPONENT_ROSTERS_PATH,
        PITCHER_PROJ_PATH,
        RAW_ROSTERS_DIR,
    )


@app.cell
def _(mo):
    mo.md("""
    ## Convert Fantrax Rosters

    This step converts raw Fantrax roster exports to the pipeline format and applies
    automatic name corrections (accents, apostrophes, etc.).

    **To update rosters:**
    1. Download roster CSVs from Fantrax
    2. Place them in `data/raw_rosters/` (my_team.csv + team_*.csv for opponents)
    3. Re-run this notebook
    """)
    return


@app.cell
def _(
    DB_PATH,
    HITTER_PROJ_PATH,
    PITCHER_PROJ_PATH,
    RAW_ROSTERS_DIR,
    apply_name_corrections,
    convert_fantrax_rosters_from_dir,
    load_projections,
):
    # Step 1: Convert Fantrax exports to pipeline format
    converted_my_roster_path, converted_opponent_rosters_path = (
        convert_fantrax_rosters_from_dir(
            raw_rosters_dir=RAW_ROSTERS_DIR,
        )
    )

    # Step 2: Load projections for name validation
    _projections_for_correction = load_projections(
        HITTER_PROJ_PATH,
        PITCHER_PROJ_PATH,
        DB_PATH,
    )

    # Step 3: Apply automatic name corrections (accents, apostrophes, etc.)
    apply_name_corrections(converted_my_roster_path, _projections_for_correction)
    apply_name_corrections(
        converted_opponent_rosters_path,
        _projections_for_correction,
        is_opponent_file=True,
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Load Data
    """)
    return


@app.cell
def _(
    DB_PATH,
    HITTER_PROJ_PATH,
    MY_ROSTER_PATH,
    OPPONENT_ROSTERS_PATH,
    PITCHER_PROJ_PATH,
    load_all_data,
):
    # Load all data using the converted roster files
    projections, my_roster_names, opponent_rosters = load_all_data(
        HITTER_PROJ_PATH,
        PITCHER_PROJ_PATH,
        MY_ROSTER_PATH,
        OPPONENT_ROSTERS_PATH,
        DB_PATH,
    )
    return my_roster_names, opponent_rosters, projections


@app.cell
def _(mo):
    mo.md("""
    ## Compute Opponent Totals
    """)
    return


@app.cell
def _(compute_all_opponent_totals, opponent_rosters, pd, projections):
    opponent_totals = compute_all_opponent_totals(opponent_rosters, projections)

    # Display opponent totals as a table with roto points
    opponent_totals_df = pd.DataFrame(opponent_totals).T
    opponent_totals_df.index.name = "Team"

    # Compute roto points: rank each category, sum (7 - rank) for points
    # For most categories, higher is better (rank descending)
    # For ERA/WHIP, lower is better (rank ascending)
    negative_cats = {"ERA", "WHIP"}
    roto_points = pd.Series(0, index=opponent_totals_df.index)
    for cat in opponent_totals_df.columns:
        ascending = cat in negative_cats  # Lower is better for ERA/WHIP
        ranks = opponent_totals_df[cat].rank(ascending=ascending, method="min")
        roto_points += (7 - ranks)  # 7 - rank gives points (best = 6 pts)

    opponent_totals_df["Roto Pts"] = roto_points.astype(int)
    opponent_totals_df = opponent_totals_df.sort_values("Roto Pts", ascending=False)
    opponent_totals_df
    return (opponent_totals,)


@app.cell
def _(mo):
    mo.md("""
    ## Filter Candidates
    """)
    return


@app.cell
def _(
    compute_quality_scores,
    filter_candidates,
    my_roster_names,
    opponent_rosters,
    projections,
):
    quality_scores = compute_quality_scores(projections)

    # Combine all opponent names into a single set
    opponent_roster_names = set()
    for team_names in opponent_rosters.values():
        opponent_roster_names.update(team_names)

    candidates = filter_candidates(
        projections,
        quality_scores,
        my_roster_names,
        opponent_roster_names,
        top_n_per_position=30,
        top_n_per_category=10,
    )
    return (candidates,)


@app.cell
def _(mo):
    mo.md("""
    ## Solve Optimization
    """)
    return


@app.cell
def _(build_and_solve_milp, candidates, my_roster_names, opponent_totals):
    optimal_roster_names, solution_info = build_and_solve_milp(
        candidates,
        opponent_totals,
        my_roster_names,
    )
    return (optimal_roster_names,)


@app.cell
def _(mo):
    mo.md("""
    ## Results Summary
    """)
    return


@app.cell
def _(
    compute_team_totals,
    my_roster_names,
    opponent_totals,
    optimal_roster_names,
    print_roster_summary,
    projections,
):
    # Compute totals for optimal roster
    my_totals = compute_team_totals(optimal_roster_names, projections)

    # Print detailed summary
    print_roster_summary(
        optimal_roster_names,
        projections,
        my_totals,
        opponent_totals,
        old_roster_names=my_roster_names,
    )
    return (my_totals,)


@app.cell
def _(mo):
    mo.md("""
    ## Visualizations
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Team Comparison Radar
    """)
    return


@app.cell
def _(my_totals, opponent_totals, plot_team_radar):
    fig_radar = plot_team_radar(my_totals, opponent_totals)
    fig_radar
    return


@app.cell
def _(mo):
    mo.md("""
    ### Win/Loss Matrix
    """)
    return


@app.cell
def _(my_totals, opponent_totals, plot_win_matrix):
    fig_win_matrix = plot_win_matrix(my_totals, opponent_totals)
    fig_win_matrix
    return


@app.cell
def _(mo):
    mo.md("""
    ### Category Margins
    """)
    return


@app.cell
def _(my_totals, opponent_totals, plot_category_margins):
    fig_margins = plot_category_margins(my_totals, opponent_totals)
    fig_margins
    return


@app.cell
def _(mo):
    mo.md("""
    ### Roster Changes
    """)
    return


@app.cell
def _(my_roster_names, optimal_roster_names, plot_roster_changes, projections):
    fig_changes = plot_roster_changes(
        my_roster_names, set(optimal_roster_names), projections
    )
    fig_changes
    return


@app.cell
def _(mo):
    mo.md("""
    ### Category Contributions
    """)
    return


@app.cell
def _(optimal_roster_names, projections):
    from optimizer.visualizations import plot_player_contribution_radar
    # Hitter contributions radar (top 10 hitters)
    plot_player_contribution_radar(list(optimal_roster_names), projections, "hitter", top_n=10)
    return (plot_player_contribution_radar,)


@app.cell
def _(optimal_roster_names, plot_player_contribution_radar, projections):

    # Pitcher contributions radar (top 10 pitchers)  
    plot_player_contribution_radar(list(optimal_roster_names), projections, "pitcher", top_n=10)
    return


@app.cell
def _():
    return


@app.cell
def _(optimal_roster_names, plot_category_contributions, projections):
    # Show stolen base contributions
    fig_sb = plot_category_contributions(optimal_roster_names, projections, "SB")
    fig_sb
    return


@app.cell
def _(optimal_roster_names, plot_category_contributions, projections):
    # Show ERA contributions (impact on team ERA)
    fig_era = plot_category_contributions(optimal_roster_names, projections, "SV")
    fig_era
    return


@app.cell
def _(mo):
    mo.md("""
    ### Constraint Analysis
    """)
    return


@app.cell
def _(candidates, optimal_roster_names, plot_constraint_analysis, projections):
    fig_constraints = plot_constraint_analysis(
        candidates, optimal_roster_names, projections
    )
    fig_constraints
    return


@app.cell
def _(mo):
    mo.md("""
    ## Sensitivity Analysis

    **Warning:** This section takes 5-15 minutes to run as it solves a separate
    MILP for each candidate player. Run the cells below only when needed.
    """)
    return


@app.cell
def _(
    candidates,
    compute_player_sensitivity,
    opponent_totals,
    optimal_roster_names,
):
    # Uncomment to run sensitivity analysis (takes ~5-15 minutes)
    sensitivity_df = compute_player_sensitivity(
        optimal_roster_names,
        candidates,
        opponent_totals
    )
    # sensitivity_df = None  # Placeholder
    return (sensitivity_df,)


@app.cell
def _(plot_player_sensitivity, sensitivity_df):
    if sensitivity_df is not None:
        fig_sensitivity = plot_player_sensitivity(sensitivity_df)
        fig_sensitivity
    else:
        print("Run sensitivity analysis cell above to see results")
    return


if __name__ == "__main__":
    app.run()
