# Streamlit Dashboard

## Overview

A Streamlit dashboard for in-season fantasy baseball management. Surfaces optimizer results, enables trade/roster simulation, and provides a searchable player database.

**Module:** `dashboard/app.py`

**Data layer:** Uses functions from:
- `optimizer/database.py` — `refresh_all_data()`, queries (see `01d_database.md`)
- `optimizer/fantrax_api.py` — API calls, `MY_TEAM_NAME` constant (see `01c_fantrax_api.md`)
- `optimizer/data_loader.py` — `compute_team_totals()`, config constants (see `01a_config.md`, `01b_fangraphs_loading.md`)
- `optimizer/trade_engine.py` — `compute_roster_situation()`, `compute_player_values()`, trade evaluation
- `optimizer/roster_optimizer.py` — `build_and_solve_milp()`, `compute_roster_change_values()`
- `optimizer/visualizations.py` — All plotting functions

---

## Goals

1. **Visibility**: Surface computed metrics (SGP, WPA, win probability, trade recommendations) in one place
2. **Simulation**: Explore "what-if" scenarios (trades, add/drops) with instant visual feedback
3. **Actionability**: Clear recommendations on what to do right now

---

## File Structure

```
dashboard/
├── app.py              # Main Streamlit entry point
├── components.py       # Reusable UI components (cards, tables, charts)
└── pages/
    ├── 1_my_team.py
    ├── 2_free_agents.py
    ├── 3_trades.py
    ├── 4_simulator.py
    ├── 5_players.py
    └── 6_settings.py
```

---

## Page Specifications

### Page 1: My Team Overview

**Purpose:** Quick health check on current roster status.

**Components:**

1. **Win Probability Card**
   ```
   ┌────────────────────────┐
   │  Win Probability       │
   │       31.4%            │
   │  Projected: 2nd-3rd    │
   └────────────────────────┘
   ```

2. **Category Strength/Weakness Chart**
   - Horizontal bar chart showing rank in each category (1-7)
   - Color coding: green (1st-2nd), yellow (3rd-5th), red (6th-7th)
   - Identifies categories where improvement is most impactful

3. **Head-to-Head Matchup Grid**
   - Heatmap: my team vs. each opponent in each category
   - Cell value: projected win probability for that matchup
   - Reuse existing `plot_win_matrix()` from `visualizations.py`

4. **Current vs. Optimal Roster Diff**
   - Side-by-side table comparison
   - Highlighted rows for players to add/drop
   - "Apply Optimal" button → populates simulator with changes

5. **Roster Table**
   - All 26 players with key stats (Position, Team, PA/IP, key category stats, SGP)
   - Sortable columns
   - Click row to expand detailed stats

---

### Page 2: Free Agent Recommendations

**Purpose:** Prioritized waiver wire targets.

**Components:**

1. **Top Free Agents Table**

   | Rank | Player | Pos | Team | WPA | SGP | Key Stats | Recommendation |
   |------|--------|-----|------|-----|-----|-----------|----------------|
   | 1 | J. Doe-H | OF | NYY | +0.023 | 12.5 | HR: 25, SB: 15 | **Take Now** |
   | 2 | J. Smith-P | RP | LAD | +0.018 | 8.2 | SV: 28 | Monitor |

   **Recommendation Logic:**
   - **Take Now**: WPA > 0.015 AND fills a category weakness
   - **Monitor**: WPA > 0.005
   - **Pass**: WPA ≤ 0.005

2. **Filters**
   - Position dropdown (C, 1B, 2B, SS, 3B, OF, SP, RP, All)
   - Category focus dropdown (show players who help specific category)
   - Min SGP slider

3. **Quick Actions**
   - "Simulate Add" button → navigates to simulator with player pre-selected

---

### Page 3: Trade Analysis

**Purpose:** Find and evaluate trade opportunities.

**Components:**

1. **Trade Recommendations Table**

   | Partner | I Give | I Get | ΔWin% | Fairness | Recommendation |
   |---------|--------|-------|-------|----------|----------------|
   | Team A | Player X | Player Y | +2.3% | Fair | **Accept** |

   - Sortable by ΔWin%, Fairness
   - Exposes existing trade engine output

2. **Trade Targets Panel** (sidebar or collapsible)
   - Players to acquire, ranked by acquirability score
   - Shows: player name, owner, value-to-me, market value (SGP)

3. **Trade Pieces Panel** (sidebar or collapsible)
   - Players to offer, ranked by expendability score
   - Shows: player name, my value loss, market value (SGP)

4. **Trade Builder**
   - Two multi-select boxes: "I Give" / "I Get"
   - Real-time evaluation as players are added/removed
   - Shows: ΔWin%, SGP differential, fairness assessment, recommendation

---

### Page 4: Roster Simulator

**Purpose:** Interactive "what-if" scenario exploration.

**Layout:**

```
┌─────────────────────────────────────────────────────────────────┐
│ ROSTER SIMULATOR                                                │
├───────────────────────────────┬─────────────────────────────────┤
│ INPUTS                        │ RESULTS                         │
│                               │                                 │
│ Players to ADD:               │ ┌─────────────────────────────┐ │
│ [Multi-select dropdown]       │ │     Radar Chart             │ │
│   - Free agents               │ │  (All teams + "After")      │ │
│   - Opponent players          │ │                             │ │
│                               │ │  Current: solid line        │ │
│ Players to DROP:              │ │  After: dashed line         │ │
│ [Multi-select dropdown]       │ │  Opponents: gray lines      │ │
│   - My roster only            │ └─────────────────────────────┘ │
│                               │                                 │
│ [Simulate] [Reset]            │ IMPACT SUMMARY                  │
│                               │ ┌─────────────────────────────┐ │
│                               │ │ Win Prob: 23.1% → 28.4%     │ │
│                               │ │ Change: +5.3%               │ │
│                               │ │                             │ │
│                               │ │ Category Changes:           │ │
│                               │ │  HR: 4th → 3rd (+4 wins)    │ │
│                               │ │  SB: 5th → 4th (+2 wins)    │ │
│                               │ │  ERA: 2nd → 2nd (no change) │ │
│                               │ └─────────────────────────────┘ │
└───────────────────────────────┴─────────────────────────────────┘
```

**Inputs:**
- "Players to ADD" multi-select: populated from free agents + all opponent rosters
- "Players to DROP" multi-select: populated from my roster only
- "Simulate" button: triggers recomputation
- "Reset" button: clears selections

**Outputs:**
- Radar chart with overlay (current vs after)
- Win probability before/after with delta
- Category-by-category rank changes with win impact

**Radar Chart Implementation:**
- Base: `plot_team_radar()` from `visualizations.py`
- Add second trace for "After" team totals (dashed line, different color)
- Legend distinguishes Current, After, and Opponents

---

### Page 5: Player Database

**Purpose:** Searchable reference for all players.

**Components:**

1. **Search Bar**
   - Search by name (case-insensitive, partial match)
   - Instant filtering as you type

2. **Filter Panel** (sidebar)
   - Position: multi-select checkboxes
   - Owner: My Team / Free Agent / [Opponent names] / All
   - Player type: Hitter / Pitcher / All
   - Min SGP slider

3. **Results Table**

   | Name | Pos | Team | Owner | PA/IP | R/W | HR/SV | RBI/K | SB/ERA | OPS/WHIP | WAR | SGP |
   |------|-----|------|-------|-------|-----|-------|-------|--------|----------|-----|-----|

   - Sortable columns (click header to sort)
   - Paginated (50 per page)
   - Click row to expand detailed view

4. **Player Detail View** (expandable row or modal)
   - Full projected stat line
   - SGP breakdown by category
   - Current owner
   - "Simulate Add" button (if not on my team)

---

### Page 6: Settings & Data Refresh

**Purpose:** Configure data sources and trigger refreshes.

**Components:**

1. **Data Status Panel**
   - Last refresh timestamp
   - Data source: "Fantrax API" or "CSV files"
   - Player count, roster counts

2. **Refresh Actions**
   - "Refresh All Data" button → calls `refresh_all_data()` from data layer
   - Progress indicator during refresh

3. **Current Standings** (from Fantrax)
   - Table showing all teams: rank, total points, category breakdown
   - My team row highlighted

4. **Configuration Display** (read-only, for reference)
   - Fantrax League ID
   - My Team Name
   - File paths

---

## Shared Components

Define reusable components in `dashboard/components.py`:

```python
def metric_card(label: str, value: str, delta: str | None = None) -> None:
    """
    Display a metric in a styled card.
    
    Example:
        metric_card("Win Probability", "31.4%", "+2.3%")
    """


def player_table(
    df: pd.DataFrame,
    columns: list[str],
    sortable: bool = True,
    on_row_click: Callable | None = None,
) -> None:
    """
    Display a formatted player table with optional sorting.
    """


def category_rank_chart(ranks: dict[str, int], title: str = "Category Ranks") -> plt.Figure:
    """
    Horizontal bar chart of category ranks (1-7).
    Color-coded by rank tier.
    """


def radar_chart_with_overlay(
    current_totals: dict[str, float],
    after_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    categories: list[str],
) -> plt.Figure:
    """
    Radar chart comparing current team, simulated team, and opponents.
    """
```

---

## Session State

Streamlit reruns the script on each interaction. Use `st.session_state` to persist computed results:

```python
# Initialize once at app startup
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.projections = None
    st.session_state.my_roster = None
    st.session_state.opponent_rosters = None
    st.session_state.optimizer_results = None
    st.session_state.last_refresh = None

# After data refresh
def on_refresh():
    from optimizer.data_loader import MY_TEAM_NAME  # Single source of truth
    from optimizer.database import refresh_all_data
    from datetime import datetime
    
    # refresh_all_data returns dict with projections, rosters, standings
    data = refresh_all_data()
    
    st.session_state.projections = data["projections"]
    st.session_state.rosters = data["rosters"]  # Dict: team_name → set of player names
    st.session_state.standings = data["standings"]
    st.session_state.my_roster = data["rosters"][MY_TEAM_NAME]
    st.session_state.opponent_rosters = {
        k: v for k, v in data["rosters"].items() if k != MY_TEAM_NAME
    }
    st.session_state.data_loaded = True
    st.session_state.last_refresh = datetime.now()
```

---

## Dependencies

Add to `pyproject.toml`:

```toml
dependencies = [
    # ... existing ...
    "streamlit>=1.30",
]
```

---

## Running the Dashboard

```bash
uv sync
streamlit run dashboard/app.py
```

---

## Implementation Priorities

### Phase 1: Foundation
1. Basic app structure with navigation
2. Settings page with data refresh
3. Player Database page (query from database)

### Phase 2: Core Views
4. My Team Overview (win prob, category chart, roster table)
5. Free Agent Recommendations (waiver priority list)
6. Current vs Optimal roster diff

### Phase 3: Simulation
7. Roster Simulator with radar overlay
8. Real-time impact calculation

### Phase 4: Trade Analysis
9. Trade recommendations table
10. Trade builder with evaluation

---

## Notes

- All data operations use functions from `optimizer/database.py` and `optimizer/fantrax_api.py`
- Visualizations reuse existing functions from `optimizer/visualizations.py` where possible
- The dashboard is read-only — it recommends actions but doesn't execute them on Fantrax
