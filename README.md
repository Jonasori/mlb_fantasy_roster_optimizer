# MLB Fantasy Roster Optimizer

A mathematical optimization system for dynasty fantasy baseball (rotisserie format). Uses Mixed Integer Linear Programming (MILP) to construct optimal rosters and a probabilistic model to evaluate trades.

## What This Does

This tool helps you make data-driven decisions in your fantasy baseball league by:

- **Finding the best free agents** - Automatically constructs the optimal roster from available players, considering position requirements, category needs, and opponent strengths
- **Evaluating trades** - Calculates how much a trade improves your win probability using probabilistic modeling
- **Identifying weaknesses** - Shows which categories and positions need improvement
- **Visualizing your team** - Creates charts and graphs to understand your roster's strengths and weaknesses

## Philosophy

This optimizer is designed for **small dynasty leagues (≤10 teams)** with an aggressive win-now approach. In small leagues, replacement-level players are genuinely good—freely available players can contribute meaningfully. The optimizer prioritizes current production over prospect stashing, since comparable players are often available on waivers. Prospects are only valued if they have a credible path to elite production.

## Features

### Free Agent Optimizer
Uses mathematical optimization (MILP) to find the **globally optimal** roster from available players. Unlike greedy heuristics that pick players one-by-one, this finds the best combination of players that maximizes expected category wins.

**Key capabilities:**
- Handles all position constraints (C, 1B, 2B, SS, 3B, OF, UTIL, SP, RP)
- Accounts for multi-position eligibility
- Balances counting stats (R, HR, RBI, SB, W, SV, K) and ratio stats (OPS, ERA, WHIP)
- Considers your opponents' rosters when optimizing

### Trade Engine
Evaluates trade proposals using a probabilistic win model based on [Rosenof (2025)](https://arxiv.org/abs/2501.00933). Calculates how much a trade changes your win probability by modeling category differentials and their variance.

**What it shows:**
- Expected win probability change from the trade
- Category-by-category impact (which stats improve/decline)
- Fairness assessment (dynasty value comparison)
- Trade candidate suggestions

### Dynasty Valuation
Multi-year player valuation with:
- Aging curves that model player decline/growth
- 25% annual discount rate (prioritizes current production)
- Trade fairness thresholds (within 10% dynasty value)

### Visualizations
Comprehensive charts and graphs:
- Team radar charts showing category strengths
- Position sensitivity analysis (where to upgrade)
- Trade impact visualizations
- Win probability breakdowns
- Category contribution analysis

### Dual Interfaces
- **Marimo Notebook** (`notebook.py`) - Interactive analysis environment for deep dives
- **Streamlit Dashboard** (`dashboard/app.py`) - Quick in-season roster checks and simulations

## Quick Start

### Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager (install with `curl -LsSf https://astral.sh/uv/install.sh | sh`)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd mlb_fantasy_roster_optimizer

# Install dependencies
uv sync

# Install development dependencies (optional, for testing)
uv sync --group dev
```

### Data Setup

#### 1. FanGraphs Projections

Download Steamer or ATC projections from [FanGraphs](https://www.fangraphs.com/projections) and save to the `data/` directory:

- `fangraphs-steamer-projections-hitters.csv` (or `fangraphs-atc-projections-hitters.csv`)
- `fangraphs-steamer-projections-pitchers.csv` (or `fangraphs-atc-projections-pitchers.csv`)

**Note:** The optimizer supports both Steamer and ATC projections. Configure which to use in `config.json` (see Configuration section).

#### 2. Fantrax Authentication

To fetch your league rosters and standings, you need to authenticate with Fantrax:

1. Log into [fantrax.com](https://www.fantrax.com)
2. Open your browser's Developer Tools (F12 or right-click → Inspect)
3. Go to **Application** → **Cookies** → `https://www.fantrax.com`
4. Copy the values for:
   - `JSESSIONID`
   - `FX_RM`
5. Add them to `config.json`:

```json
{
  "fantrax": {
    "cookies": {
      "JSESSIONID": "your_session_id_here",
      "FX_RM": "your_remember_me_token_here"
    }
  }
}
```

**Security Note:** Never commit `config.json` with real credentials. It's already in `.gitignore`.

#### 3. Configure Your League

Edit `config.json` to match your league settings:

- `league.my_team_name` - Your team name (must match Fantrax exactly)
- `league.fantrax_league_id` - Your league ID (found in Fantrax URL)
- `league.fantrax_team_ids` - Map of team names to Fantrax team IDs
- `league.roster_size` - Total roster size
- `league.hitting_slots` / `league.pitching_slots` - Position requirements

See the [Configuration](#configuration) section for full details.

### Running the Optimizer

#### Option 1: Marimo Notebook (Recommended for Analysis)

```bash
marimo edit notebook.py
```

The notebook provides an interactive environment where you can:
- Refresh data from Fantrax
- Run the free agent optimizer
- Evaluate trades
- Generate visualizations
- Explore position sensitivity

#### Option 2: Streamlit Dashboard (Quick Checks)

```bash
streamlit run dashboard/app.py
```

The dashboard is optimized for quick in-season roster checks and trade evaluations.

## Usage Examples

### Finding Optimal Free Agents

```python
from optimizer.database import refresh_all_data, get_free_agents
from optimizer.roster_optimizer import build_and_solve_milp, print_roster_summary
from optimizer.data_loader import compute_all_opponent_totals

# Refresh data from Fantrax
refresh_all_data()

# Get free agents and opponent totals
free_agents = get_free_agents()
opponent_totals = compute_all_opponent_totals()
current_roster = get_roster_names("Your Team Name")

# Find optimal roster
optimal_roster, solution_info = build_and_solve_milp(
    candidates=free_agents,
    opponent_totals=opponent_totals,
    current_roster_names=current_roster
)

# Print summary
print_roster_summary(optimal_roster, solution_info)
```

### Evaluating a Trade

```python
from optimizer.trade_engine import evaluate_trade, print_trade_report

# Evaluate a trade
trade_result = evaluate_trade(
    players_giving=["Player A", "Player B"],
    players_receiving=["Player C", "Player D"]
)

# Print detailed report
print_trade_report(trade_result)
```

### Position Sensitivity Analysis

```python
from optimizer.roster_optimizer import compute_position_sensitivity
from optimizer.visualizations import plot_position_sensitivity_dashboard

# Analyze which positions need upgrades
sensitivity_df = compute_position_sensitivity(
    include_all_active=True  # Shows FA opportunities + trade targets
)

# Visualize
fig = plot_position_sensitivity_dashboard(sensitivity_df)
```

## Configuration

All configuration is stored in `config.json` at the project root. The file is organized into sections:

### League Settings

```json
{
  "league": {
    "fantrax_league_id": "your_league_id",
    "my_team_name": "Your Team Name",
    "roster_size": 26,
    "min_hitters": 12,
    "max_hitters": 16,
    "min_pitchers": 10,
    "max_pitchers": 14,
    "hitting_slots": {
      "C": 1,
      "1B": 1,
      "2B": 1,
      "SS": 1,
      "3B": 1,
      "OF": 3,
      "UTIL": 1
    },
    "pitching_slots": {
      "SP": 5,
      "RP": 2
    }
  }
}
```

### Scoring Categories

The optimizer supports standard 5x5 rotisserie categories:

**Hitting:** R, HR, RBI, SB, OPS  
**Pitching:** W, SV, K, ERA, WHIP

These are configured in `config.json`:

```json
{
  "league": {
    "hitting_categories": ["R", "HR", "RBI", "SB", "OPS"],
    "pitching_categories": ["W", "SV", "K", "ERA", "WHIP"],
    "negative_categories": ["ERA", "WHIP"]
  }
}
```

### SGP (Standings Gain Points) Configuration

SGP denominators control how player value is calculated. Adjust these based on your league's scoring distribution:

```json
{
  "sgp": {
    "denominators": {
      "R": 20.0,
      "HR": 8.0,
      "RBI": 20.0,
      "SB": 7.0,
      "W": 3.5,
      "SV": 8.0,
      "K": 35.0
    },
    "rate_stats": {
      "OPS": [0.010, 0.750, true],
      "ERA": [0.18, 4.00, false],
      "WHIP": [0.030, 1.25, false]
    }
  }
}
```

### Projection Settings

Configure which projection system to use:

```json
{
  "projections": {
    "data_dir": "data/",
    "raw_hitters": "data/fangraphs-atc-projections-hitters.csv",
    "raw_pitchers": "data/fangraphs-atc-projections-pitchers.csv",
    "adjusted_hitters": "data/fangraphs-atc-pt-adjusted-hitters.csv",
    "adjusted_pitchers": "data/fangraphs-atc-pt-adjusted-pitchers.csv",
    "use_adjusted": true
  }
}
```

Set `use_adjusted: true` to use playing-time-adjusted projections (recommended).

### Trade Engine Settings

```json
{
  "trade_engine": {
    "fairness_threshold_percent": 0.10,
    "max_trade_size": 3,
    "min_meaningful_improvement": 0.1,
    "lose_cost_scale": 2
  }
}
```

## How It Works

### Data Flow

```
FanGraphs CSVs ──► load_projections() ──►┐
                                          │
Fantrax API ─────► fetch_all_fantrax() ──►├──► optimizer.db ──► Analysis
                                          │    (SQLite database)
                                          │
                                          ▼
                              ┌──────────────────────┐
                              │  Free Agent Optimizer │
                              │  Trade Engine         │
                              │  Visualizations       │
                              └──────────────────────┘
```

All data is consolidated into a SQLite database (`data/optimizer.db`). This ensures consistency and allows efficient queries.

### Free Agent Optimizer

The optimizer uses **Mixed Integer Linear Programming (MILP)** to find the globally optimal roster. It:

1. Models each player as a binary variable (selected or not)
2. Adds constraints for:
   - Position requirements (must fill all slots)
   - Roster composition (min/max hitters and pitchers)
   - Multi-position eligibility
3. Maximizes expected category wins against all opponents
4. Linearizes ratio stats (OPS, ERA, WHIP) for the solver

The solver finds the **best possible combination**, not just a good one.

### Trade Engine

Based on the probabilistic win model from [Rosenof (2025)](https://arxiv.org/abs/2501.00933):

```
Win Probability = Φ(μ_D / σ_D)
```

Where:
- `μ_D` = expected category differential vs best opponent
- `σ_D` = standard deviation of that differential
- `Φ` = standard normal cumulative distribution function

The trade engine:
1. Calculates your current win probability
2. Simulates the trade (adds/removes players)
3. Recalculates win probability
4. Reports the change and category impacts

### Key Concepts

**SGP (Standings Gain Points)** - Context-free player value. Answers "How good is this player in general?" Same value regardless of which team they're on.

**EWA (Expected Wins Added)** - Context-dependent value. Answers "How many more category matchups will I win if I add this player?" Varies based on your roster's strengths/weaknesses.

**Example:** A high-SGP player may have low EWA if you're already dominant in their categories. A lower-SGP player may have high EWA if they fill contested category gaps.

## Project Structure

```
mlb_fantasy_roster_optimizer/
├── notebook.py              # Marimo notebook (primary interface)
├── dashboard/               # Streamlit dashboard
│   ├── app.py              # Main dashboard app
│   └── components.py       # Reusable dashboard components
├── optimizer/               # Core library
│   ├── config.py           # Configuration loading
│   ├── data_loader.py      # FanGraphs loading, team totals, utilities
│   ├── database.py         # SQLite schema and queries
│   ├── fantrax_api.py      # Roster/age/standings from Fantrax API
│   ├── mlb_api.py          # Player age data from MLB Stats API
│   ├── playing_time.py     # Playing time adjustments
│   ├── roster_optimizer.py # MILP free agent optimizer
│   ├── trade_engine.py     # Probabilistic trade evaluation
│   └── visualizations.py   # All plotting functions
├── data/                    # Data files (gitignored)
│   ├── optimizer.db        # SQLite database (primary data source)
│   └── *.csv               # FanGraphs projection files
├── implementation_specs/   # Detailed technical specifications
└── tests/                   # Pytest test suite
```

## Troubleshooting

### "Config file not found" Error

Make sure `config.json` exists in the project root. Copy `config.json.example` if needed (if provided) or create it from the template above.

### "Player names don't match" Error

Player names must match exactly between Fantrax rosters and FanGraphs projections. Common issues:
- Nicknames vs full names
- Special characters (e.g., "José" vs "Jose")
- Suffixes (e.g., "Jr." vs "Jr")

The optimizer will list all unmatched names. Update your projection CSV files to match Fantrax names exactly.

### "Infeasible solution" from Optimizer

This means no valid roster exists that satisfies all constraints. Common causes:
- Not enough players eligible for a position slot
- Roster size constraints too restrictive
- Min/max hitter/pitcher bounds incompatible

The optimizer will identify which position slot is problematic. Adjust your `config.json` constraints or check your free agent pool.

### Fantrax API Errors

If you get authentication errors:
1. Verify your cookies are still valid (they expire after inactivity)
2. Re-copy `JSESSIONID` and `FX_RM` from browser DevTools
3. Make sure you're logged into Fantrax in the same browser

### Database Locked Errors

If you see SQLite "database is locked" errors:
- Close any other processes accessing `data/optimizer.db`
- Make sure the Marimo notebook or Streamlit dashboard isn't running in multiple instances

## Development

### Running Tests

```bash
uv run pytest tests/ -v
```

### Code Style

This project follows strict conventions:
- **No OOP** - All code uses module-level functions and plain data structures
- **Fail fast** - Assertions with descriptive messages, no silent error handling
- **Database-centric** - SQLite consolidates all data; queries pull from DB
- **Return figures** - Visualization functions return `plt.Figure`, never call `plt.show()`

See `.cursor/rules/mlb-optimizer.mdc` for complete coding guidelines.

### Type Checking

```bash
uv run mypy optimizer/
```

### Adding New Features

1. Read the relevant specification in `implementation_specs/`
2. Follow the existing code patterns (no classes, fail fast, database-centric)
3. Add tests in `tests/`
4. Update this README if adding user-facing features

## References

- Rosenof, Z. (2025). "Optimizing for Rotisserie Fantasy Basketball." arXiv:2501.00933
- Smart Fantasy Baseball. "How to Analyze SGP Denominators"
- Ryan Brock (2016). "On the Use of Aging Curves for Fantasy Baseball." FanGraphs Community

## License

MIT
