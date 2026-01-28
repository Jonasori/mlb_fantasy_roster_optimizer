# Notebook Integration Guide

## Overview

This document shows how to use the optimizer in a marimo notebook. It demonstrates the complete workflow from data loading through analysis and visualization.

**File:** `notebook.py` (at project root)

---

## Notebook Structure

The notebook should be organized into logical sections:

1. **Setup** - Imports and configuration
2. **Data Pipeline** - Fantrax conversion, name correction, data loading
3. **Current Situation** - Analyze current roster
4. **Free Agent Analysis** - Run optimizer, review recommendations
5. **Trade Analysis** - Identify trade opportunities
6. **Deep Dives** - Sensitivity analysis, visualizations

---

## Complete Example

### Cell 1: Imports

```python
import marimo as mo
import pandas as pd
import matplotlib.pyplot as plt

# Data loading
from optimizer.data_loader import (
    load_projections,
    convert_fantrax_rosters_from_dir,
    apply_name_corrections,
    load_all_data,
    compute_team_totals,
    compute_all_opponent_totals,
    compute_quality_scores,
)

# Free agent optimizer
from optimizer.roster_optimizer import (
    filter_candidates,
    build_and_solve_milp,
    compute_standings,
    print_roster_summary,
    compute_player_sensitivity,
)

# Trade engine
from optimizer.trade_engine import (
    compute_win_probability,
    compute_player_values,
    identify_trade_targets,
    identify_trade_pieces,
    generate_trade_candidates,
    evaluate_trade,
    verify_trade_impact,
    compute_roster_situation,
    print_trade_report,
)

# Visualizations
from optimizer.visualizations import (
    plot_team_radar,
    plot_category_margins,
    plot_win_matrix,
    plot_category_contributions,
    plot_player_contribution_radar,
    plot_roster_changes,
    plot_player_sensitivity,
    plot_win_probability_breakdown,
    plot_player_value_scatter,
    plot_trade_impact,
)
```

### Cell 2: Configuration

```python
# File paths
DATA_DIR = "data/"
RAW_ROSTERS_DIR = DATA_DIR + "raw_rosters/"
HITTER_PROJ_PATH = DATA_DIR + "fangraphs-steamer-projections-hitters.csv"
PITCHER_PROJ_PATH = DATA_DIR + "fangraphs-steamer-projections-pitchers.csv"
DB_PATH = "../mlb_player_comps_dashboard/mlb_stats.db"
```

### Cell 3: Convert Fantrax Rosters

```python
# Convert raw Fantrax exports to pipeline format
# Returns paths to the generated files
my_roster_path, opponent_rosters_path = convert_fantrax_rosters_from_dir(
    raw_rosters_dir=RAW_ROSTERS_DIR,
    my_team_filename='my_team.csv',
)
```

### Cell 4: Apply Name Corrections

```python
# Load projections temporarily for name matching
_projections_temp = load_projections(HITTER_PROJ_PATH, PITCHER_PROJ_PATH, DB_PATH)

# Auto-correct accents, apostrophes, known mismatches
apply_name_corrections(my_roster_path, _projections_temp)
apply_name_corrections(opponent_rosters_path, _projections_temp, is_opponent_file=True)
```

### Cell 5: Load All Data

```python
# Load validated data (uses the converted paths from Cell 3)
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
```

### Cell 6: Display Opponent Summary

```python
# Show opponent totals as a table
mo.md("## Opponent Totals")
pd.DataFrame(opponent_totals).T.round(2)
```

### Cell 7: Team Comparison Radar

```python
mo.md("## Team Comparison")
fig = plot_team_radar(my_totals, opponent_totals)
fig
```

### Cell 8: Win Matrix

```python
mo.md("## Win/Loss Matrix")
fig = plot_win_matrix(my_totals, opponent_totals)
fig
```

---

## Free Agent Optimizer Section

### Cell 9: Filter Candidates

```python
mo.md("## Free Agent Optimizer")

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
```

### Cell 10: Solve MILP

```python
# Run the optimizer
optimal_roster_names, solution_info = build_and_solve_milp(
    candidates,
    opponent_totals,
    my_roster_names,
)

print(f"Objective: {solution_info['objective']}/60 wins")
print(f"Solve time: {solution_info['solve_time']:.1f}s")
```

### Cell 11: Roster Summary

```python
# Compute optimal roster totals
optimal_totals = compute_team_totals(optimal_roster_names, projections)

# Print detailed summary
print_roster_summary(
    optimal_roster_names,
    projections,
    optimal_totals,
    opponent_totals,
    old_roster_names=my_roster_names,
)
```

### Cell 12: Visualize Changes

```python
mo.md("## Roster Changes")
fig = plot_roster_changes(my_roster_names, set(optimal_roster_names), projections)
fig
```

---

## Trade Analysis Section

### Cell 13: Roster Situation Analysis

```python
mo.md("## Trade Analysis")

# Analyze current situation (computes category_sigmas internally)
situation = compute_roster_situation(my_roster_names, projections, opponent_totals)

# Extract for later use
category_sigmas = situation['category_sigmas']

print(f"Win probability: {situation['win_probability']:.1%}")
print(f"Expected wins: {situation['expected_wins']:.1f}/60")
```

### Cell 14: Win Probability Breakdown

```python
fig = plot_win_probability_breakdown(situation['diagnostics'])
fig
```

### Cell 15: Compute Player Values

```python
# Include my roster + all opponent rosters for comparison
all_roster_names = my_roster_names | opponent_roster_names

# Compute probabilistic player values
player_values = compute_player_values(
    player_names=all_roster_names,
    my_roster_names=my_roster_names,
    projections=projections,
    my_totals=my_totals,
    opponent_totals=opponent_totals,
    category_sigmas=category_sigmas,
)

# Show top players by contextual value
player_values.head(20)
```

### Cell 16: Player Value Scatter

```python
fig = plot_player_value_scatter(player_values, my_roster_names)
fig
```

### Cell 17: Generate Trade Candidates

```python
# Generate trade recommendations
trade_candidates = generate_trade_candidates(
    my_roster_names=my_roster_names,
    player_values=player_values,
    opponent_rosters=opponent_rosters,
    projections=projections,
    my_totals=my_totals,
    opponent_totals=opponent_totals,
    category_sigmas=category_sigmas,
    max_send=2,
    max_receive=2,
    n_targets=15,
    n_pieces=15,
    n_candidates=20,
)
```

### Cell 18: Trade Report

```python
print_trade_report(situation, trade_candidates, player_values, top_n=5)
```

### Cell 19: Evaluate Specific Trade

```python
mo.md("## Evaluate Specific Trade")

# Example: evaluate a specific trade idea
result = evaluate_trade(
    send_players=['Gunnar Henderson-H'],  # Adjust to your roster
    receive_players=['Trea Turner-H'],     # Adjust to opponent roster
    player_values=player_values,
    my_roster_names=my_roster_names,
    projections=projections,
    my_totals=my_totals,
    opponent_totals=opponent_totals,
    category_sigmas=category_sigmas,
)

# Visualize the trade impact
new_totals = compute_team_totals(
    (my_roster_names - set(result['send_players'])) | set(result['receive_players']),
    projections,
)
fig = plot_trade_impact(result, my_totals, new_totals, opponent_totals)
fig
```

### Cell 20: Verify Trade

```python
# Ground-truth verification
verification = verify_trade_impact(
    send_players=['Gunnar Henderson-H'],
    receive_players=['Trea Turner-H'],
    my_roster_names=my_roster_names,
    projections=projections,
    opponent_totals=opponent_totals,
    category_sigmas=category_sigmas,
)
```

---

## Deep Dive Section

### Cell 21: Category Deep Dive

```python
mo.md("## Category Analysis")

# Analyze SB contributions (example weak category)
fig = plot_category_contributions(list(my_roster_names), projections, 'SB')
fig
```

### Cell 22: Hitter Contribution Radar

```python
fig = plot_player_contribution_radar(list(my_roster_names), projections, "hitter", top_n=10)
fig
```

### Cell 23: Pitcher Contribution Radar

```python
fig = plot_player_contribution_radar(list(my_roster_names), projections, "pitcher", top_n=10)
fig
```

### Cell 24: Sensitivity Analysis (Optional - Slow)

```python
mo.md("## Sensitivity Analysis")
mo.md("*This takes 5-15 minutes. Run only when needed.*")

# Uncomment to run:
# sensitivity = compute_player_sensitivity(optimal_roster_names, candidates, opponent_totals)
# fig = plot_player_sensitivity(sensitivity)
# fig
```

---

## Running the Notebook

```bash
# Install dependencies
uv sync

# Run the notebook
marimo edit notebook.py
```

---

## Workflow Summary

1. **Data Pipeline**
   - Convert Fantrax exports → pipeline format
   - Apply name corrections (accents, spelling)
   - Load validated projections and rosters

2. **Free Agent Optimizer** (for rebuilding roster)
   - Filter to candidate players
   - Solve MILP for optimal roster
   - Review changes and projected standings

3. **Trade Engine** (for evaluating trades)
   - Compute win probability and player values
   - Generate trade candidates
   - Evaluate specific proposals
   - Verify impact before executing

4. **Deep Dives** (for understanding)
   - Visualize category contributions
   - Run sensitivity analysis
   - Identify bottlenecks and opportunities

---

## Tips

1. **Start with Trade Analysis** if you just want trade ideas without changing roster.

2. **Use Free Agent Optimizer** when you want to see the globally optimal roster (might recommend many changes).

3. **Verify trades** with `verify_trade_impact` before proposing — ensures gradient-based evaluation matches full recomputation.

4. **Sensitivity analysis is expensive** — only run when you need to understand which players are truly irreplaceable.

5. **Update data regularly** — re-download projections and re-export rosters from Fantrax before each analysis session.
