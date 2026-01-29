# MLB Fantasy Roster Optimizer: Implementation Specifications

## Purpose

These specifications enable an AI coding agent to implement the complete roster optimization system in a single pass. Each document is self-contained with enough detail for implementation without ambiguity.

## Reading Order

Read the documents in numerical order:

| # | Document | Description |
|---|----------|-------------|
| 00 | [Agent Guidelines](00_agent_guidelines.md) | Code style, constraints, implementation philosophy. **Read first.** |
| 01a | [Configuration](01a_config.md) | League constants, SGP config, core utilities. |
| 01b | [FanGraphs Loading](01b_fangraphs_loading.md) | Projection CSVs, position loading, team totals. |
| 01c | [Fantrax API](01c_fantrax_api.md) | Roster/age/standings data from API. |
| 01d | [Database](01d_database.md) | SQLite schema, sync functions, queries. |
| 01e | [Dynasty Valuation](01e_dynasty_valuation.md) | Aging curves, dynasty SGP, multi-year value. |
| 02 | [Free Agent Optimizer](02_free_agent_optimizer.md) | MILP formulation for optimal roster construction. |
| 03 | [Trade Engine](03_trade_engine.md) | Probabilistic win model and trade evaluation. |
| 04 | [Visualizations](04_visualizations.md) | All plotting functions. |
| 05 | [Notebook Integration](05_notebook_integration.md) | Marimo notebook workflow examples. |
| 06 | [Streamlit Dashboard](06_streamlit_dashboard.md) | In-season dashboard with simulator. |
| 07 | [Testing](07_testing.md) | Minimal pytest suite (16 tests, no classes). **Run after implementation.** |

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    notebook.py / dashboard/                      │
│                    (User Interfaces)                             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         optimizer/                               │
├──────────┬──────────┬──────────┬──────────┬─────────────────────┤
│data_     │fantrax_  │database  │roster_   │trade_engine.py      │
│loader.py │api.py    │.py       │optimizer │                     │
│          │          │          │.py       │                     │
│• Config  │• Rosters │• Schema  │• MILP    │• Win probability    │
│• FG load │• Ages    │• Sync    │• Filter  │• Player values      │
│• Totals  │• Stands  │• Queries │• Solve   │• Trade eval         │
└──────────┴──────────┴──────────┴──────────┴─────────────────────┘
                                │
                                ▼
                   ┌────────────────────────┐
                   │   visualizations.py    │
                   │   • Radar charts       │
                   │   • Heatmaps           │
                   │   • Trade impact       │
                   └────────────────────────┘
```

## Data Flow

```
FanGraphs CSVs ─────► load_projections() ─────►┐
(projections)                                   │
                                                ▼
Fantrax API ───────► fetch_all_fantrax_data() ─► refresh_all_data() ─► optimizer.db
(rosters, ages,                                 │                      (PRIMARY SOURCE)
 standings, positions)                          │
                                                ▼
                                    ┌──────────────────────┐
                                    │   Database Queries   │
                                    │   get_projections()  │
                                    │   get_roster_names() │
                                    │   get_free_agents()  │
                                    └──────────────────────┘
                                                │
                        ┌───────────────────────┴───────────────────────┐
                        ▼                                               ▼
                Free Agent Optimizer                            Trade Engine
                (MILP for optimal roster)                    (Win probability model)
```

## Key Design Decisions

### 1. Database Is Primary Source of Truth
The SQLite database (`optimizer.db`) consolidates all data. All queries in the optimizer and trade engine pull from the database, not from CSVs or API calls directly.

### 2. No Object-Oriented Programming
All code uses module-level functions and plain data structures. No classes.

### 3. Two Data Sources (Inputs)
- **FanGraphs CSVs:** Player projections (PA, IP, WAR, all stats)
- **Fantrax API:** Roster ownership, ages, standings, positions

### 4. Two Optimization Approaches

| Approach | Use Case | Method |
|----------|----------|--------|
| **Free Agent Optimizer** | "What's my optimal roster?" | MILP (global optimum) |
| **Trade Engine** | "Which trades should I propose?" | Probabilistic model |

### 5. Probabilistic Win Model (Trade Engine)

Based on Rosenof (2025):

```
V = Φ(μ_D / σ_D)
```

Where:
- `μ_D` = expected differential vs best opponent
- `σ_D` = std dev of that differential
- `Φ` = standard normal CDF

Marginal value computed via **numerical differentiation** (simple, accurate).

### 6. Player Name Uniqueness

All names include `-H` or `-P` suffix:
- `"Mike Trout-H"`, `"Gerrit Cole-P"`
- Display functions use `strip_name_suffix()` (single source in data_loader)

### 7. Dynasty SGP for Trade Fairness

Trade fairness uses `dynasty_SGP` (net present value of future production):
- Accounts for aging curves
- 25% discount rate (prioritizes current-year production)
- Trade fairness = dynasty_SGP differential within 10%

### 8. SGP Weights Rate Stats by Playing Time

Following the canonical SGP methodology (Smart Fantasy Baseball), rate stats are weighted by playing time. A .850 OPS player with 600 PA contributes more to team OPS than the same player with 100 PA.

### 9. Fail Fast

No try/except blocks. No fallback logic. Assert with descriptive messages.

## Implementation Sequence

**Note:** An existing implementation exists in `optimizer/`. It should be used as **guidance only** — the implementing agent should follow these specs, not the existing code, when they differ. The existing code predates the database-centric architecture and may have bugs.

**Existing files to handle:**
- `optimizer/*.py` — Rewrite to match specs (keep useful patterns, fix issues)
- `notebook.py` — Rewrite to match spec (05)
- `test_fantrax_api.py` — Delete (functionality moves to `tests/test_core.py`)
- `data/*.csv` — Keep (input data files)

1. **Set up project** (pyproject.toml, directories)
2. **data_loader.py** (01a, 01b) — rewrite with specs
3. **fantrax_api.py** (01c) — rewrite with specs
4. **database.py** (01d) — new file
5. **Dynasty valuation** in data_loader (01e)
6. **roster_optimizer.py** (02) — rewrite with specs
7. **trade_engine.py** (03) — new file
8. **visualizations.py** (04) — rewrite with specs
9. **notebook.py** (05) — rewrite with specs
10. **dashboard/** (06) — new directory
11. **tests/** (07) — new file, run to verify implementation

## Dependencies

```toml
dependencies = [
    "pandas>=2.0",
    "numpy>=1.24",
    "pulp>=2.8",
    "highspy>=1.5",
    "scipy>=1.10",
    "matplotlib>=3.7",
    "seaborn>=0.12",
    "tqdm>=4.65",
    "marimo>=0.8",
    "streamlit>=1.30",
    "requests>=2.28",
]

[tool.uv]
dev-dependencies = [
    "pytest>=8.0",
]
```

## Validation

Each spec includes a validation checklist. Implementation is complete when:
1. All checklists pass
2. `uv run pytest tests/ -v` — all tests pass
3. Notebook runs end-to-end
4. Dashboard loads and functions

## References

- Rosenof, Z. (2025). "Optimizing for Rotisserie Fantasy Basketball." arXiv:2501.00933.
- Smart Fantasy Baseball. "How to Analyze SGP Denominators."
- Ryan Brock (2016). "On the Use of Aging Curves for Fantasy Baseball." FanGraphs Community.
