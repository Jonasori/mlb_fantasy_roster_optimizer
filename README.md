# MLB Fantasy Roster Optimizer

A mathematical optimization system for dynasty fantasy baseball (rotisserie format). Uses Mixed Integer Linear Programming (MILP) to construct optimal rosters and a probabilistic model to evaluate trades.

## General philosophy

This optimizer embraces an aggressive win-now approach calibrated for small-league dynasty formats (≤10 teams). The justification is mathematical: in a 7-team league, the replacement-level floor is extraordinarily high—freely available players are genuinely good. A prospect is only worth rostering if there's a realistic probability they become an above-replacement-level contributor, and that's a much harder bar to clear when replacement level includes quite good players. Most prospects won't clear that bar. Most will bust, or settle into replacement-level production that you could have acquired for free anyway. Meanwhile, the aging veteran you dropped to stash that prospect is putting up points now, and when he declines, someone comparable will almost certainly be available to replace him. The optimizer therefore treats prospect value with extreme skepticism, holding only those with a credible path to elite production. Everyone else gets evaluated on current-year output. The goal is to win every year by maintaining an optimally constructed roster of productive players, rather than sacrificing present competitiveness for a future payoff that -— in a small league -- you could likely acquire off waivers anyway.

## Features

- **Free Agent Optimizer**: MILP solver finds the globally optimal roster from available players
- **Trade Engine**: Probabilistic win model evaluates trade proposals based on marginal category value
- **Dynasty Valuation**: Multi-year projections with aging curves and discount rates
- **Visualizations**: Radar charts, heatmaps, and trade impact analysis
- **Dual Interfaces**: Marimo notebook for analysis, Streamlit dashboard for in-season use

## Quick Start

```bash
# Install dependencies
uv sync

# Run the notebook (primary interface)
marimo edit notebook.py

# Or run the dashboard
streamlit run dashboard/app.py
```

## Data Setup

### FanGraphs Projections

Download Steamer projections from FanGraphs and save to `data/`:
- `fangraphs-steamer-projections-hitters.csv`
- `fangraphs-steamer-projections-pitchers.csv`

### Fantrax Authentication

Create `data/fantrax_cookies.json` with your session cookies:

```json
{
  "JSESSIONID": "your_session_id",
  "FX_RM": "your_remember_me_token"
}
```

To get these values:
1. Log into [fantrax.com](https://www.fantrax.com)
2. Open DevTools → Application → Cookies
3. Copy the `JSESSIONID` and `FX_RM` values

## Project Structure

```
mlb_fantasy_roster_optimizer/
├── notebook.py              # Marimo notebook (primary interface)
├── dashboard/               # Streamlit dashboard
│   ├── app.py
│   └── components.py
├── optimizer/               # Core library
│   ├── data_loader.py       # Configuration, FanGraphs loading, team totals
│   ├── fantrax_api.py       # Roster/age/standings from Fantrax API
│   ├── database.py          # SQLite schema and queries
│   ├── roster_optimizer.py  # MILP free agent optimizer
│   ├── trade_engine.py      # Probabilistic trade evaluation
│   └── visualizations.py    # All plotting functions
├── data/                    # Data files (gitignored except examples)
│   ├── optimizer.db         # SQLite database (primary data source)
│   └── *.csv                # FanGraphs projection files
├── implementation_specs/    # Detailed specifications for each module
└── tests/                   # Pytest test suite
```

## How It Works

### Data Flow

```
FanGraphs CSVs ──► load_projections() ──►┐
                                          │
Fantrax API ─────► fetch_all_fantrax() ──►├──► optimizer.db ──► Analysis
                                          │    (primary source)
                                          │
                                          ▼
                              ┌──────────────────────┐
                              │  Free Agent Optimizer │
                              │  Trade Engine         │
                              │  Visualizations       │
                              └──────────────────────┘
```

### Free Agent Optimizer

Uses MILP to maximize expected category wins against all opponents:

- Handles roster position constraints (C, 1B, 2B, SS, 3B, OF, UTIL, SP, RP)
- Linearizes ratio stat comparisons (OPS, ERA, WHIP)
- Finds globally optimal solution (not greedy heuristic)

### Trade Engine

Based on [Rosenof (2025)](https://arxiv.org/abs/2501.00933) probabilistic win model:

```
Win Probability = Φ(μ_D / σ_D)
```

Where:
- `μ_D` = expected differential vs best opponent
- `σ_D` = standard deviation of that differential
- `Φ` = standard normal CDF

Player marginal values computed via numerical differentiation.

### Dynasty Valuation

- Aging curves model player decline/growth
- 25% annual discount rate (prioritizes current production)
- Trade fairness = dynasty SGP differential within 10%

## Scoring Categories

**Hitting (5 categories):** R, HR, RBI, SB, OPS

**Pitching (5 categories):** W, SV, K, ERA, WHIP

## Configuration

Key constants in `optimizer/data_loader.py`:

```python
ROSTER_SIZE = 30
MY_TEAM_NAME = "Your Team Name"
HITTING_CATEGORIES = ["R", "HR", "RBI", "SB", "OPS"]
PITCHING_CATEGORIES = ["W", "SV", "K", "ERA", "WHIP"]
```

### Position Sensitivity Analysis

The EWA percentile curves include all active players (free agents + opponent rosters) by default. This provides trade context: red points on the plots indicate players on opponent rosters who would be valuable trade targets.

To show only free agent opportunities, pass `include_all_active=False`:

```python
pctl_ewa_df = compute_percentile_sensitivity(
    ...,
    include_all_active=False,  # Only show FA opportunities
)
```

**Minimum playing time thresholds:** Free agents are filtered to PA >= 50 (hitters) and IP >= 20 (pitchers) to exclude minor leaguers with negligible projections. All rostered players (your team + opponents) are included regardless of PA/IP. These thresholds are defined in `compute_position_sensitivity()` as `MIN_PA_FOR_FA` and `MIN_IP_FOR_FA`.

## Development

```bash
# Run tests
uv run pytest tests/ -v

# Type checking (optional)
uv run mypy optimizer/
```

## Architecture Notes

- **No OOP**: All code uses module-level functions and plain data structures
- **Fail fast**: Assertions with descriptive messages, no silent error handling
- **Database-centric**: SQLite consolidates all data; queries pull from DB, not raw files
- **Return figures**: Visualization functions return `plt.Figure`, never call `plt.show()`

## References

- Rosenof, Z. (2025). "Optimizing for Rotisserie Fantasy Basketball." arXiv:2501.00933
- Smart Fantasy Baseball. "How to Analyze SGP Denominators"
- Ryan Brock (2016). "On the Use of Aging Curves for Fantasy Baseball." FanGraphs Community

## License

MIT
