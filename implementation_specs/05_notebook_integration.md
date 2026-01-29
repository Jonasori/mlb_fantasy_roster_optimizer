# Notebook Integration Guide

## Overview

This document shows how to use the optimizer in a marimo notebook. It demonstrates the complete workflow from data loading through analysis and visualization.

**File:** `notebook.py` (at project root)

**Key principle:** All data comes from the database via `refresh_all_data()`. The notebook never loads CSVs directly.

---

## Notebook Structure

1. **Setup** - Imports and configuration
2. **Data Loading** - Single call to `refresh_all_data()` (syncs FanGraphs + Fantrax → database)
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

# Database is the primary data source
from optimizer.database import (
    refresh_all_data,
    get_projections,
    get_roster_names,
    get_free_agents,
)

# Data utilities (constants from data_loader, the single source of truth)
from optimizer.data_loader import (
    MY_TEAM_NAME,
    NUM_OPPONENTS,
    FANTRAX_TEAM_IDS,
    compute_team_totals,
    compute_all_opponent_totals,
    compute_quality_scores,
    estimate_projection_uncertainty,
)

# Free agent optimizer
from optimizer.roster_optimizer import (
    filter_candidates,
    build_and_solve_milp,
    compute_standings,
    print_roster_summary,
    compute_roster_change_values,
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
    plot_constraint_analysis,
)
```

### Cell 2: Configuration

```python
# File paths for FanGraphs CSVs (input to database)
DATA_DIR = "data/"
HITTER_PROJ_PATH = DATA_DIR + "fangraphs-steamer-projections-hitters.csv"
PITCHER_PROJ_PATH = DATA_DIR + "fangraphs-steamer-projections-pitchers.csv"
DB_PATH = DATA_DIR + "optimizer.db"
```

### Cell 3: Load Data (Full Refresh)

```python
mo.md("## Data Loading")

# Refresh all data: FanGraphs + Fantrax API → database
# This is the SINGLE data loading call
data = refresh_all_data(
    hitter_proj_path=HITTER_PROJ_PATH,
    pitcher_proj_path=PITCHER_PROJ_PATH,
    db_path=DB_PATH,
)

# Extract what we need (all from database queries)
projections = data["projections"]
my_roster_names = data["my_roster"]
opponent_rosters = data["opponent_rosters"]

print(f"Projections: {len(projections)} players")
print(f"My roster: {len(my_roster_names)} players")
print(f"Opponents: {len(opponent_rosters)} teams")
```

### Cell 4: Compute Totals

```python
# Compute team totals for comparison
my_totals = compute_team_totals(my_roster_names, projections)

# Opponent rosters need to be converted to dict[int, set[str]] format
opponent_rosters_indexed = {
    i+1: names for i, (team, names) in enumerate(opponent_rosters.items())
}
opponent_totals = compute_all_opponent_totals(opponent_rosters_indexed, projections)
```

### Cell 5: Team Comparison Radar

```python
mo.md("## Team Comparison")
fig = plot_team_radar(my_totals, opponent_totals)
fig
```

### Cell 6: Win Matrix

```python
mo.md("## Win/Loss Matrix")
fig = plot_win_matrix(my_totals, opponent_totals)
fig
```

### Cell 7: Category Margins

```python
fig = plot_category_margins(my_totals, opponent_totals)
fig
```

### Cell 8: Filter Candidates

```python
mo.md("## Free Agent Optimizer")

# Compute quality scores for filtering
quality_scores = compute_quality_scores(projections)

# All opponent player names (unavailable)
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

### Cell 9: Solve MILP

```python
mo.md("### Running Optimizer...")

optimal_roster_names, solution_info = build_and_solve_milp(
    candidates,
    opponent_totals,
    my_roster_names,
)

print(f"Objective: {solution_info['objective']}/60 wins")
print(f"Solve time: {solution_info['solve_time']:.1f}s")
print(f"Status: {solution_info['status']}")
```

### Cell 10: Roster Summary

```python
# Compute optimal roster totals
optimal_totals = compute_team_totals(optimal_roster_names, projections)

# Print summary with waiver priority
print_roster_summary(
    optimal_roster_names,
    projections,
    optimal_totals,
    opponent_totals,
    old_roster_names=my_roster_names,
)
```

### Cell 11: Roster Changes Visualization

```python
mo.md("### Waiver Priority List")

added = set(optimal_roster_names) - my_roster_names
dropped = my_roster_names - set(optimal_roster_names)

added_df, dropped_df = compute_roster_change_values(
    added, dropped, my_roster_names, projections, opponent_totals
)
fig = plot_roster_changes(added_df, dropped_df)
fig
```

### Cell 12: Trade Analysis Setup

```python
mo.md("## Trade Analysis (from optimized roster)")

# Use optimized roster for trade analysis
trade_roster_names = set(optimal_roster_names)
situation = compute_roster_situation(
    trade_roster_names, projections, opponent_totals
)

print(f"Win probability: {situation['win_probability']:.1%}")
print(f"Expected wins: {situation['expected_wins']:.1f}/60")
print(f"Strengths: {', '.join(situation['strengths']) or 'None'}")
print(f"Weaknesses: {', '.join(situation['weaknesses']) or 'None'}")

category_sigmas = situation["category_sigmas"]
```

### Cell 13: Win Probability Breakdown

```python
fig = plot_win_probability_breakdown(situation["diagnostics"])
fig
```

### Cell 14: Player Values

```python
mo.md("### Player Values")

# Include my roster + all opponent rosters
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
```

### Cell 15: Player Value Scatter

```python
fig = plot_player_value_scatter(player_values, trade_roster_names)
fig
```

### Cell 16: Generate Trade Candidates

```python
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
```

### Cell 17: Trade Report

```python
print_trade_report(situation, trade_candidates, player_values, top_n=5)
```

### Cell 18: Deep Dive - Category Contributions

```python
mo.md("## Deep Dive Analysis")
mo.md("### Home Run Contributions")
fig = plot_category_contributions(list(my_roster_names), projections, "HR")
fig
```

### Cell 19: Player Contribution Radar

```python
mo.md("### Hitter Contributions")
fig = plot_player_contribution_radar(
    list(my_roster_names), projections, "hitter", top_n=10
)
fig
```

### Cell 20: Constraint Analysis

```python
mo.md("### Constraint Analysis")
fig = plot_constraint_analysis(optimal_roster_names, projections)
fig
```

### Cell 21: Sensitivity Analysis (Optional)

```python
mo.md("""
### Sensitivity Analysis (Optional - Slow)

*Uncomment the code below to run sensitivity analysis. Takes 5-15 minutes.*
""")

# Uncomment to run:
# from optimizer.roster_optimizer import compute_player_sensitivity
# sensitivity = compute_player_sensitivity(optimal_roster_names, candidates, opponent_totals)
# fig = plot_player_sensitivity(sensitivity)
# fig
```

---

## Key Points

1. **Single data source:** All data comes from `refresh_all_data()`, which:
   - Loads FanGraphs CSVs
   - Fetches Fantrax API data
   - Syncs everything to SQLite database
   - Returns data from database queries

2. **Skip Fantrax during development:** Use `skip_fantrax=True` to avoid API calls:
   ```python
   data = refresh_all_data(..., skip_fantrax=True)
   ```

3. **Trade analysis uses optimized roster:** After running the free agent optimizer, trade analysis starts from the improved roster position.

4. **Visualizations return figures:** Never call `plt.show()` — marimo handles display.

5. **Progress reporting:** Long operations use `print()` for status and `tqdm` for progress bars.
