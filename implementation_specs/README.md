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
| 02a | [Variance-Penalized Objective](02a_variance_penalized_objective.md) | Balance incentive extension to MILP (optional). |
| 02b | [Position Sensitivity](02b_position_sensitivity.md) | EWA-based position analysis, SGP limitations. |
| 03 | [Trade Engine](03_trade_engine.md) | Probabilistic win model and trade evaluation. |
| 04 | [Visualizations](04_visualizations.md) | All plotting functions. |
| 05 | [Notebook Integration](05_notebook_integration.md) | Marimo notebook workflow examples. |
| 06 | [Streamlit Dashboard](06_streamlit_dashboard.md) | In-season dashboard with simulator. |
| 07 | [Testing](07_testing.md) | Minimal pytest suite (17 tests, no classes). **Run after implementation.** |

## Glossary

Quick reference for key terms used throughout these specifications:

| Term | Definition |
|------|------------|
| **SGP** | Standing Gain Points. Context-free player value measuring how many rotisserie standings points a player contributes. Same value regardless of team context. |
| **EWA** | Expected Wins Added. Context-dependent value measuring how many additional category matchups (out of 60) you win by adding a player to YOUR specific roster. |
| **MILP** | Mixed-Integer Linear Programming. Optimization technique used by the Free Agent Optimizer to find the globally optimal roster. |
| **V** | Win probability. The probability of winning the rotisserie league, computed via `Φ(μ_D / σ_D)`. |
| **μ_D** | Expected differential between your fantasy points and the best opponent's. |
| **σ_D** | Standard deviation of the differential. |
| **Φ** | Standard normal CDF (cumulative distribution function). |
| **PA** | Plate Appearances. Used as weight for hitter ratio stats (OPS). |
| **IP** | Innings Pitched. Used as weight for pitcher ratio stats (ERA, WHIP). |
| **Ratio stats** | OPS, ERA, WHIP — computed as weighted averages, never summed directly. |
| **Counting stats** | R, HR, RBI, SB, W, SV, K — summed directly across players. |
| **dynasty_SGP** | Age-adjusted SGP representing net present value of future production. |
| **Big-M** | Large constant used in MILP indicator constraints (10000 for counting, 5000 for ratio). |
| **ε (epsilon)** | Small constant for strict inequality (0.5 for counting, 0.001 for ratio). |

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

### 7. Raw SGP for Trade Fairness

Trade fairness uses raw single-season `SGP`, not dynasty_SGP:
- Simpler and more predictable
- Avoids aging curve complexity and potential bugs
- Trade fairness = SGP differential within 10%
- Dynasty leagues can optionally enable age-adjusted values

### 8. SGP Weights Rate Stats by Playing Time

Following the canonical SGP methodology (Smart Fantasy Baseball), rate stats are weighted by playing time. A .850 OPS player with 600 PA contributes more to team OPS than the same player with 100 PA.

### 9. Fail Fast

No try/except blocks. No fallback logic. Assert with descriptive messages.

### 10. SGP vs EWA: Context-Free vs Context-Dependent Value

Two different value metrics serve different purposes:

**SGP (Standings Gain Points)** - Context-free player value:
- "How good is this player in general?"
- Same value regardless of which team they're on
- Based on projected stats relative to league averages
- Used for trade fairness (both sides can agree on it)

**EWA (Expected Wins Added)** - Context-dependent value:
- "How many more category matchups will I win if I add this player?"
- Varies based on your roster's category strengths/weaknesses
- A high-SGP player may have low EWA if you're already dominant in their categories
- A lower-SGP player may have high EWA if they fill contested category gaps
- More intuitive than league win probability (V): "you gain 1.5 expected wins" vs "you gain 0.3%"

**Standardization:** All team-specific value metrics use EWA (expected category wins out of 60), not WPA (league win probability). EWA is more linear, more intuitive, and avoids the non-linearities of the league probability model.

**Example:** Bo Bichette (SGP: 10.7) may be recommended for dropping despite high SGP because:
- Your team is already winning R, HR, RBI matchups (diminishing returns)
- You're losing pitching categories where adding pitchers flips more matchups
- EWA captures this context; SGP does not

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

## Final Step: Generate README

After all implementation is complete and validated, generate a `README.md` at the project root. The README should include:
- Project overview and features
- Quick start instructions (`uv sync`, `marimo edit notebook.py`)
- Data setup (FanGraphs CSVs, Fantrax cookies)
- Project structure
- How the system works (data flow, optimizer, trade engine)
- Configuration options
- Development commands
- References

## ⚠️ Common Pitfalls (Lessons Learned)

These bugs were discovered during implementation and are documented here to prevent recurrence:

### 1. Fantrax Two-Way Player Names
**Problem:** Fantrax returns some players with `-H`/`-P` suffix already attached (e.g., "Shohei Ohtani-H").  
**Bug:** Code added another suffix → "Shohei Ohtani-H-H" → didn't match database.  
**Fix:** Always check `if name.endswith("-H") or name.endswith("-P")` before adding suffix.

### 2. Fantrax getPlayerStats Pagination
**Problem:** The API reports 9,719 players but pagination parameters are ignored.  
**Bug:** Infinite loop fetching the same 20 players repeatedly.  
**Fix:** Use `maxResultsPerPage=5000` in a single request. Don't paginate.

### 3. Trade Expendability Formula Sign Error
**Problem:** Formula `-(SGP + ewa_lose * scale)` made stars MORE expendable.  
**Bug:** `ewa_lose` is negative when losing hurts → double negative flipped the logic.  
**Fix:** Use `-SGP + ewa_lose * scale` (no outer negation on the lose cost term).

### 4. Marimo Output vs Return
**Problem:** Using `return mo.vstack(...)` caused syntax errors.  
**Bug:** Marimo uses last expression for display, `return` for variable export.  
**Fix:** Make display the last expression, use `return (var,)` only for exports.

### 5. Trade Targets with Zero Value
**Problem:** "Best" trade target had +0.00 EWA value.  
**Bug:** No filter for positive-value players before sorting.  
**Fix:** Filter to `ewa_acquire > 0.01` before ranking targets.

### 6. Defensive Programming in Notebook
**Problem:** Cells had `if data is not None` checks everywhere.  
**Bug:** Silent failures hid real problems; code was 50 lines longer.  
**Fix:** Use assertions with clear messages. Fail fast, don't defend.

### 7. EWA Calculation for Adds
**Problem:** Each add's EWA was computed after removing ALL dropped players.  
**Bug:** Every add looked equally bad (adding one player to a gutted roster).  
**Fix:** Evaluate each add in ISOLATION: "what if I add just this player to my current roster?"

### 8. Chart Ordering (barh)
**Problem:** Priority lists showed least important items at top.  
**Bug:** matplotlib `barh` puts index 0 at the BOTTOM of the chart.  
**Fix:** Reverse the DataFrame before plotting: `df_display = df.iloc[::-1]`

### 9. Default Position: UTIL not DH
**Problem:** Unknown hitter positions defaulted to "DH".  
**Bug:** DH is a specific position (designated hitter). UTIL is the universal slot term.  
**Fix:** Default to "UTIL" and add "UTIL" to SLOT_ELIGIBILITY["UTIL"].

### 10. SGP is a Poor Proxy for RP Value
**Problem:** Position sensitivity ranked RPs by SGP, suggesting a player at "99th percentile" was optimal.  
**Bug:** SGP treats all categories equally, but saves don't correlate with K/ERA/WHIP. High-K relievers get high SGP even with 0 saves. A team weak in saves needs closers specifically.  
**Evidence:** For RPs, SGP→EWA correlation is only 0.591. Saves→EWA correlation is 0.925. For hitters, SGP→EWA correlation is 0.997.  
**Fix:** Use EWA (Expected Wins Added) as the primary metric for position sensitivity. Show `better_fas_count` and `best_fa_ewa` instead of misleading SGP-based percentiles.

### 11. Percentile Among ALL Players is Misleading
**Problem:** Computing percentile among all 4,000+ eligible players produces useless metrics.  
**Bug:** A player at "99th percentile" among all RP-eligible players could still have 40 available upgrades because most of those 4,000 players are minor leaguers with near-zero SGP.  
**Fix:** Compute percentiles among AVAILABLE players only. Better yet, show `better_fas_count` directly.

### 12. UTIL/RP Slots Have 4000+ Eligible Players
**Problem:** Position sensitivity computed EWA for 4000+ players at UTIL and RP slots.  
**Bug:** UTIL includes all hitters (4100+) and RP includes all pitchers (4300+), including minor leaguers with 1 PA/IP projected. Computing EWA for each is slow and meaningless.  
**Evidence:** Only ~500 hitters have PA >= 50, and ~400 pitchers have IP >= 20.  
**Fix:** Filter FA players to MIN_PA_FOR_FA=50 and MIN_IP_FOR_FA=20. Keep all rostered players (my + opponent) regardless of PA/IP. This reduces UTIL from 4100→545 and RP from 4300→446.

## References

- Rosenof, Z. (2025). "Optimizing for Rotisserie Fantasy Basketball." arXiv:2501.00933.
- Smart Fantasy Baseball. "How to Analyze SGP Denominators."
- Ryan Brock (2016). "On the Use of Aging Curves for Fantasy Baseball." FanGraphs Community.
