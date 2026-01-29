# MLB Fantasy Roster Optimizer
#
# This package provides tools for optimizing fantasy baseball rosters:
# - data_loader: Load projections, rosters, and compute team totals
# - roster_optimizer: MILP-based optimal roster construction
# - trade_engine: Probabilistic win model and trade evaluation
# - visualizations: All plotting functions

from .data_loader import (
    ALL_CATEGORIES,
    HITTING_CATEGORIES,
    NEGATIVE_CATEGORIES,
    PITCHING_CATEGORIES,
    ROSTER_SIZE,
    SGP_DENOMINATORS,
    apply_name_corrections,
    compute_all_opponent_totals,
    compute_quality_scores,
    compute_sgp_value,
    compute_team_totals,
    convert_fantrax_rosters_from_dir,
    estimate_projection_uncertainty,
    load_all_data,
    load_projections,
    strip_name_suffix,
)
from .roster_optimizer import (
    build_and_solve_milp,
    compute_player_sensitivity,
    compute_roster_change_values,
    compute_standings,
    filter_candidates,
    print_roster_summary,
)
from .trade_engine import (
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
from .visualizations import (
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
