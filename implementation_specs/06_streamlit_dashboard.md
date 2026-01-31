# Streamlit Dashboard

## Overview

A Streamlit dashboard for in-season fantasy baseball management. Surfaces optimizer results, enables trade/roster simulation, and provides a searchable player database.

**Module:** `dashboard/app.py` (single file with all pages)
**Components:** `dashboard/components.py` (reusable UI elements)

---

## Cross-References

**Depends on:**
- [00_agent_guidelines.md](00_agent_guidelines.md) — code style
- [01a_config.md](01a_config.md) — `compute_team_totals()`, config constants
- [01c_fantrax_api.md](01c_fantrax_api.md) — `MY_TEAM_NAME` constant
- [01d_database.md](01d_database.md) — `refresh_all_data()`, database queries
- [02_free_agent_optimizer.md](02_free_agent_optimizer.md) — `build_and_solve_milp()`, `compute_roster_change_values()`
- [02b_position_sensitivity.md](02b_position_sensitivity.md) — `compute_position_sensitivity()`, `compute_percentile_sensitivity()`
- [03_trade_engine.md](03_trade_engine.md) — `compute_roster_situation()`, `compute_player_values()`, trade evaluation
- [04_visualizations.md](04_visualizations.md) — `plot_team_dashboard()`, `plot_comparison_dashboard()`, position sensitivity plots

**Used by:**
- [07_testing.md](07_testing.md) — dashboard browser tests

---

**Data layer:** Uses functions from:
- `optimizer/database.py` — `refresh_all_data()`, queries (see [01d_database.md](01d_database.md))
- `optimizer/fantrax_api.py` — API calls, `MY_TEAM_NAME` constant (see [01c_fantrax_api.md](01c_fantrax_api.md))
- `optimizer/data_loader.py` — `compute_team_totals()`, `estimate_projection_uncertainty()`, config constants (see [01a_config.md](01a_config.md), [01b_fangraphs_loading.md](01b_fangraphs_loading.md))
- `optimizer/trade_engine.py` — `compute_roster_situation()`, `compute_player_values()`, trade evaluation
- `optimizer/roster_optimizer.py` — `build_and_solve_milp()`, `compute_roster_change_values()`, `compute_position_sensitivity()`, `compute_percentile_sensitivity()`
- `optimizer/visualizations.py` — All plotting functions (including `plot_position_sensitivity_dashboard()`, `plot_upgrade_opportunities()`, `plot_percentile_ewa_curves()`, `plot_position_distributions()`)

---

## Goals

1. **Visibility**: Surface computed metrics (SGP, EWA, win probability, trade recommendations) in one place
2. **Simulation**: Explore "what-if" scenarios (trades, add/drops) with instant visual feedback
3. **Actionability**: Clear recommendations on what to do right now

---

## Architecture

### Single-File Structure

The dashboard uses a single `app.py` with page routing via navigation buttons (not Streamlit's multi-page system):

```
dashboard/
├── app.py              # Main entry point with all page functions
└── components.py       # Reusable UI components (radar chart, etc.)
```

### Navigation Pattern

Use session state for navigation with button-based navigation:

```python
PAGES = [
    ("🏠 Overview", "Overview"),
    ("📊 My Team", "My Team"),
    ("🔄 Trades", "Trades"),
    ("🔍 Free Agents", "Free Agents"),  # Combined free agent browser + roster simulator
    ("📋 All Players", "All Players"),
]

def navigate_to(page: str):
    """Navigate to a specific page programmatically."""
    st.session_state.nav_page = page
```

In sidebar:
```python
st.markdown("---")
st.markdown("#### Navigation")

current_page = st.session_state.nav_page

for page_key, page_label in PAGES:
    # Highlight current page with primary button style
    is_current = current_page == page_key
    button_type = "primary" if is_current else "secondary"
    
    if st.button(
        page_key,
        key=f"nav_{page_key}",
        width="stretch",  # NOTE: use width="stretch" not use_container_width (deprecated)
        type=button_type,
    ):
        navigate_to(page_key)
        st.rerun()
```

**Benefits of button navigation:**
- Clear visual distinction between current page (primary) and others (secondary)
- Works consistently across light/dark themes
- No custom CSS required
- Larger, more clickable targets

---

## Session State

Initialize all state variables at startup:

```python
def init_session_state():
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.projections = None
        st.session_state.my_roster = None
        st.session_state.opponent_rosters = None
        st.session_state.standings = None  # Actual standings from Fantrax
        st.session_state.last_refresh = None
    if "nav_page" not in st.session_state:
        st.session_state.nav_page = "🏠 Overview"
    # Trade analysis cache
    if "trade_results" not in st.session_state:
        st.session_state.trade_results = None
    if "player_values" not in st.session_state:
        st.session_state.player_values = None
    if "situation" not in st.session_state:
        st.session_state.situation = None
    if "trade_targets" not in st.session_state:
        st.session_state.trade_targets = None
    if "trade_pieces" not in st.session_state:
        st.session_state.trade_pieces = None
    if "opponent_totals" not in st.session_state:
        st.session_state.opponent_totals = None
    # Position sensitivity analysis cache (My Team page)
    if "position_sensitivity" not in st.session_state:
        st.session_state.position_sensitivity = None
```

**Critical:** Store all data including standings, and clear cached results when data refreshes:
```python
def load_data():
    data = refresh_all_data(skip_fantrax=True)  # Uses cached Fantrax data from DB
    
    st.session_state.projections = data["projections"]
    st.session_state.my_roster = data["my_roster"]
    st.session_state.opponent_rosters = data["opponent_rosters"]
    st.session_state.standings = data["standings"]  # Actual Fantrax standings
    st.session_state.data_loaded = True
    
    # Clear cached results
    st.session_state.trade_results = None
    st.session_state.player_values = None
    st.session_state.situation = None
    st.session_state.trade_targets = None
    st.session_state.trade_pieces = None
```

---

## Page Specifications

### Page 1: Overview (`show_overview`)

**Purpose:** League overview with standings and navigation shortcuts.

**Title:** "League Overview"

**Components:**

1. **Quick Actions** (3 buttons at top)
   - "📊 View My Team" → navigates to My Team
   - "🔍 Free Agents" → navigates to Free Agents
   - "🔄 Analyze Trades" → navigates to Trades
   - Use `navigate_to()` + `st.rerun()` on click

2. **Actual League Standings** (from Fantrax)
   - Check `st.session_state.standings` for actual Fantrax standings
   - If available, display with `st.subheader("League Standings")`
   - Caption shows date: "Actual standings from Fantrax (as of {date})"
   - Columns: (indicator), Rank, Team, Total Points
   - User's team indicated with "👉" emoji (matching MY_TEAM_NAME)
   - Status message: "Season hasn't started yet" if total_points is null
   - If no standings available: "No standings data available. Click 'Refresh Data' to fetch from Fantrax."

3. **Projected Standings** (in expander)
   - `st.expander("📊 Projected Standings (based on FanGraphs projections)")`
   - Caption: "How teams would rank if projections played out perfectly."
   - Computes roto standings from projected category totals
   - Shows DataFrame with: (indicator), Rank, Team, Total Points, R, HR, RBI, SB, OPS, W, SV, K, ERA, WHIP
   - Each category column shows team's projected rank (1-7)
   - User's team indicated with "👉" emoji for "My Team"

4. **My Roster Summary** (if roster loaded)
   - 4 metric columns: Hitters count, Pitchers count, Total SGP, Avg SGP

---

### Page 2: My Team (`show_my_team`)

**Purpose:** View current roster composition, team totals, performance visualizations, and position sensitivity analysis.

**Components:**

1. **Two-Column Roster Display**
   - Left: Hitters table (Name, Position, Team, PA, R, HR, RBI, SB, OPS, SGP)
   - Right: Pitchers table (Name, Position, Team, IP, W, SV, K, ERA, WHIP, SGP)
   - Both sorted by SGP descending

2. **Team Totals**
   - Hitting totals: R, HR, RBI, SB, OPS
   - Pitching totals: W, SV, K, ERA, WHIP

3. **Team Performance Visualizations** (if opponent data available)
   - Uses combined `plot_team_dashboard()` function (all 3 charts in one figure)
   - Display via `display_figure(fig_dashboard, width=2200)`
   
   The combined figure shows:
   - **Panel 1 (left):** Radar chart - league percentile across categories
   - **Panel 2 (center):** Win/Loss heatmap vs each opponent
   - **Panel 3 (right):** Roster composition (horizontal bar chart)

4. **Position Sensitivity Analysis** (after `st.divider()`)
   
   **Purpose:** Analyze which positions offer the most upgrade opportunity and where your roster is strongest/weakest.
   
   **Session State:**
   ```python
   if "position_sensitivity" not in st.session_state:
       st.session_state.position_sensitivity = None
   ```
   
   **"Analyze Positions" Button:**
   - Shows "📊 Analyze Positions" initially, "🔄 Refresh Analysis" after computation
   - Triggers `_compute_position_sensitivity()` which:
     1. Computes `category_sigmas` via `estimate_projection_uncertainty()`
     2. Gets `opponent_roster_names` from opponent rosters
     3. Calls `compute_position_sensitivity()` from `roster_optimizer.py`
     4. Calls `compute_percentile_sensitivity()` from `roster_optimizer.py`
     5. Stores all results in `st.session_state.position_sensitivity`
   - Takes ~30 seconds to compute (uses spinner)
   
   **Displayed Results** (via `_display_position_sensitivity_plots()`):
   
   - **Baseline Expected Wins:** `st.info()` showing current expected wins (e.g., "19.2 / 60")
   
   - **Position Sensitivity Dashboard** (always visible after computation)
     - Uses `plot_position_sensitivity_dashboard(ewa_df, sensitivity_df, slot_data)`
     - 4-panel visualization:
       - Panel 1: EWA per SGP by position (which positions give most bang for buck)
       - Panel 2: EWA from upgrading to best FA at each position
       - Panel 3: SGP vs EWA scatter by position
       - Panel 4: Position scarcity curves with my players marked
     - Display via `display_figure(fig_dashboard, width=1400)`
   
   - **Upgrade Opportunities by Position** (in expander)
     - Uses `plot_upgrade_opportunities(slot_data)`
     - Horizontal bar chart showing SGP gap between best FA and my worst player
     - Display via `display_figure(fig, width=1000)`
   
   - **EWA vs Percentile Curves** (in expander)
     - Uses `plot_percentile_ewa_curves(pctl_ewa_df)`
     - Grid of 6 small charts (C, SS, OF, SP, RP, 2B)
     - Shows how EWA changes at different percentile levels
     - Red dashed line marks current worst player's percentile
     - Display via `display_figure(fig, width=1400)`
   
   - **Player Distribution by Position** (in expander)
     - Uses `plot_position_distributions(slot_data)`
     - Boxplots showing SGP distributions for hitting and pitching positions
     - Red dots mark rostered players
     - Display via `display_figure(fig, width=1400)`

---

### Page 3: Trade Analysis (`show_trades`)

**Purpose:** Find and evaluate trade opportunities.

**Layout:**

```
┌────────────────────────────────────────────────────────────────┐
│ TRADE ANALYSIS                                                 │
├────────────────────────────────────────────────────────────────┤
│ [Success banner: Win probability X% | Expected wins Y/60]      │
│ (auto-computed when entering page)                             │
├────────────────────────────────────────────────────────────────┤
│ 🔧 TRADE BUILDER (available immediately!)                      │
│ ┌─────────────────────┐ ┌─────────────────────┐                │
│ │ Players you SEND    │ │ Players you RECEIVE │                │
│ │ [multiselect]       │ │ [multiselect]       │                │
│ └─────────────────────┘ └─────────────────────┘                │
│ [📊 Evaluate Trade]                                            │
├────────────────────────────────────────────────────────────────┤
│ 🔍 FIND RECOMMENDED TRADES                                     │
│ [🔍 Find Trade Targets]                                        │
├────────────────────────────────────────────────────────────────┤
│ CATEGORY ANALYSIS                                              │
│ ┌────────────────────────────────────────────────────────────┐ │
│ │ Category | My Value | Avg Opponent | Win Prob | Status     │ │
│ └────────────────────────────────────────────────────────────┘ │
│ Strengths: [list]        Weaknesses: [list]                    │
├────────────────────────────────────────────────────────────────┤
│ (after Find Trade Targets)                                     │
│ ┌─────────────────────┐ ┌─────────────────────┐                │
│ │ 🎯 Players to       │ │ 📤 Players to       │                │
│ │    Acquire          │ │    Offer            │                │
│ └─────────────────────┘ └─────────────────────┘                │
├────────────────────────────────────────────────────────────────┤
│ ⚙️ TRADE SEARCH SETTINGS                                       │
│ Fairness Threshold: [slider 0-50%, default 30%]                │
│ Min Win Prob Improvement: [slider -1% to 2%, default 0.3%]     │
│ [🔄 Re-run with New Settings]                                  │
├────────────────────────────────────────────────────────────────┤
│ 💱 RECOMMENDED TRADES                                          │
│ ┌────────────────────────────────────────────────────────────┐ │
│ │ # | Partner | You Send | You Get | ΔWin% | Fair? | Rec     │ │
│ └────────────────────────────────────────────────────────────┘ │
│ [Expandable details for each trade]                            │
└────────────────────────────────────────────────────────────────┘
```

**Key Architecture: Trade Builder is Independent**

The Trade Builder appears immediately when entering the Trades tab, separate from recommended trades.
This is achieved by computing `player_values` during initial situation computation.

**Implementation Details:**

1. **Auto-Compute Situation + Player Values** (on page load via `_compute_roster_situation`)
   - If `st.session_state.situation` is `None`, automatically computes:
     - `opponent_totals` via `compute_all_opponent_totals()`
     - `situation` via `compute_roster_situation()`
     - `player_values` via `compute_player_values()` - enables Trade Builder immediately
   - Stores all in session state and calls `st.rerun()`
   - No separate button needed - Trade Builder ready on first load

2. **Trade Builder** (`_show_trade_builder`) - SHOWN FIRST, always available
   - Uses pre-computed `player_values` from session state
   - Two multiselects: "Players you SEND" (your roster) and "Players you RECEIVE" (opponents)
   - "📊 Evaluate Trade" button triggers `_evaluate_custom_trade()`
   - Completely independent from recommended trades search

3. **Find Trade Targets Button** (`_compute_trade_targets`)
   - Computes trade_targets, trade_pieces, and trade_results (recommendations)
   - Uses already-computed player_values
   - Stores results in session state
   - Calls `st.rerun()`

4. **Category Analysis Table**
   - Shows after situation computed
   - Columns: Category, My Value, Avg Opponent, Win Prob, Status
   - Format ratio stats (OPS, ERA, WHIP) with `.3f`
   - Format counting stats as integers

5. **Trade Targets/Pieces Panels** (side by side)
   - Shows after trade targets computed
   - Targets: Player, Pos, Owner, Value (+X.XX%), SGP
   - Pieces: Player, Pos, Lose Cost, SGP
   - Use `strip_name_suffix()` for display names
   - Map opponent IDs to team names

6. **Trade Search Settings** (below targets, above recommended trades)
   - Visible section with `st.subheader("⚙️ Trade Search Settings")`
   - **Fairness Threshold slider (0-50%, default 30%)**
   - **Min Win Prob Improvement slider (-1% to 2%, default 0.3%)**
   - "🔄 Re-run with New Settings" button to recompute with new parameters

7. **Recommended Trades Table**
   - Check `if st.session_state.trade_results is not None:` (handles empty list)
   - Columns: #, Partner, You Send, You Get, ΔWin%, Fairness, Rec
   - Show "No favorable fair trades found" if empty list
   - Expandable details for first 10 trades
   - Options format: `"PlayerName (SGP: X.X)"`
   - "Evaluate Trade" button calls `_evaluate_custom_trade()`

8. **Custom Trade Evaluation** (`_evaluate_custom_trade`)
   - Uses `evaluate_trade()` from trade_engine
   - Checks for `invalid_reason` key in result and displays error if present
   - Uses `plot_comparison_dashboard()` for full visual analysis:
     - Panel 1: Before/After radar chart
     - Panel 2: Win/Loss heatmap (after state)
     - Panel 3: Category-by-category delta bars
   - Display via `display_figure(fig_dashboard, width=2200)`
   - Summary row with 3 columns: Trade Summary, You Send, You Receive
   - Displays: Win Prob Change, Dynasty SGP Change, Fairness, Recommendation
   - Color-coded results (st.success/st.error/st.info)
   - Shows player details with SGP totals
   - Expandable category impact table showing before/after for all 10 categories

---

### Page 4: Free Agents (`show_simulator`)

**Purpose:** Combined free agent browser + interactive roster simulator with radar visualization.

**Layout:**

```
┌─────────────────────────────────────────────────────────────────┐
│ FREE AGENTS                                                     │
├─────────────────────────────────────────────────────────────────┤
│ 🔍 FREE AGENT BROWSER                                           │
│ Browse available free agents, then use the simulator below      │
├───────────────────┬───────────────────┬─────────────────────────┤
│ Player Type ▼     │ Position ▼        │ Min SGP [slider]        │
├───────────────────┴───────────────────┴─────────────────────────┤
│ **X free agents found**                                         │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Name | Position | Team | Stats... | SGP                     │ │
│ └─────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│ ─────────────────── [divider] ───────────────────               │
├─────────────────────────────────────────────────────────────────┤
│ 🎮 SIMULATE ROSTER CHANGES                                      │
│ Test roster changes and see their impact on your win probability│
├───────────────────────────────┬─────────────────────────────────┤
│ **Players to ADD**            │ **Players to DROP**             │
│ [multiselect: FA + opponents] │ [multiselect: my roster]        │
├───────────────────────────────┴─────────────────────────────────┤
│ [🎮 Simulate]                 [🔄 Reset]                        │
├─────────────────────────────────────────────────────────────────┤
│ (after Simulate)                                                │
│ ┌─────────────────────────┐ ┌─────────────────────────────────┐ │
│ │   Radar Chart           │ │ Impact Summary                  │ │
│ │   Comparison            │ │ Win Probability: XX% → YY%      │ │
│ │                         │ │ Change: +Z.ZZ%                  │ │
│ │   (sorted by perf)      │ │                                 │ │
│ │                         │ │ Players Added: [list]           │ │
│ │                         │ │ Players Dropped: [list]         │ │
│ └─────────────────────────┘ └─────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│ Category Impact Details                                         │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Category | Before | After | Change                          │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

**Implementation Details:**

**Section 1: Free Agent Browser**

1. **Filter Controls:** Player Type (All/Hitters/Pitchers), Position dropdown, Min SGP slider (0-20, default 3.0)

2. **Free Agent Table:** Filters `projections[projections["owner"].isna()]`. Columns vary by player type filter. Sorted by SGP descending, limited to top 50. Shows count: "**X free agents found**"

**Section 2: Roster Simulator** (after `st.divider()`)

3. **Players to ADD Multiselect:** Combines top 100 free agents + top 50 opponent players. Format: `"PlayerName (Position, SGP: X.X)"`

4. **Players to DROP Multiselect:** My roster only, sorted by SGP ascending (worst first)

5. **Action Buttons:** "🎮 Simulate" (primary) triggers `_run_simulation()`, "🔄 Reset" clears session state

6. **Simulation Results:** Computes old/new totals and win probability.

7. **Comparison Dashboard:** Uses `plot_comparison_dashboard()` from `optimizer/visualizations.py`
   - Panel 1: Before/After radar chart
   - Panel 2: Win/Loss heatmap (after state)
   - Panel 3: Category-by-category delta bars
   - Display via `display_figure(fig_dashboard, width=2200)`

8. **Summary Row:** 3 columns with Impact Summary, Players Added, Players Dropped
   - Win probability change displayed inline

9. **Category Impact Table:** In expander, showing all 10 categories with Before/After/Change columns

---

### Page 5: All Player Data (`show_players`)

**Purpose:** Searchable reference for all players with full stats.

**Title:** "All Player Data"

**Components:**

1. **Search Bar**
   - Text input with placeholder
   - Case-insensitive partial match on Name

2. **Filters** (3 columns)
   - Player Type: All / Hitters / Pitchers
   - Ownership: All / Free Agents / Rostered
   - Min SGP slider (0-20)

3. **Dynamic Column Display**
   - Core columns: Name, Position, Team, owner
   - Hitter stats: PA, R, HR, RBI, SB, OPS
   - Pitcher stats: IP, W, SV, K, ERA, WHIP, GS
   - Value columns: SGP, WAR
   - Extra columns: age, dynasty_SGP

   Column set depends on Player Type filter:
   - Hitters: core + hitter stats + value + extra
   - Pitchers: core + pitcher stats + value + extra
   - All: core + player_type + all stats + value + extra

4. **Table Configuration**
   Use `st.column_config` for formatting:
   ```python
   column_config = {
       "Name": st.column_config.TextColumn("Name", width="medium"),
       "PA": st.column_config.NumberColumn("PA", format="%d"),
       "OPS": st.column_config.NumberColumn("OPS", format="%.3f"),
       "ERA": st.column_config.NumberColumn("ERA", format="%.2f"),
       # etc.
   }
   ```

5. **Stat Legend** (expandable)
   - Explains all hitter stats
   - Explains all pitcher stats
   - Explains value stats (SGP, WAR, Dynasty SGP)

---

## Shared Components (`dashboard/components.py`)

```python
import io

def display_figure(fig: plt.Figure, width: int = 400) -> None:
    """
    Display a matplotlib figure at a specific pixel width.
    
    This gives precise control over display size while maintaining
    high resolution from the underlying figure.
    
    Implementation:
        1. Save figure to BytesIO buffer as PNG (dpi=200, bbox_inches="tight")
        2. Display via st.image(buf, width=width)
        3. Close the figure with plt.close(fig)
    
    This approach decouples figure resolution (set in matplotlib figsize)
    from display size (controlled by width parameter).
    
    Common widths:
        - 600: Single radar chart or comparison
        - 2200: Full-width combined dashboard
    """

def metric_card(label: str, value: str, delta: str | None = None) -> None:
    """Display a metric in a styled card using st.metric()."""

def player_table(
    df: pd.DataFrame,
    columns: list[str],
    sortable: bool = True,
    on_row_click: Callable | None = None,
) -> None:
    """Display a formatted player table."""

def category_rank_chart(
    ranks: dict[str, int],
    title: str = "Category Ranks"
) -> plt.Figure:
    """Horizontal bar chart of category ranks (1-7), color-coded."""

def radar_chart_with_overlay(
    current_totals: dict[str, float],
    after_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    categories: list[str],
) -> plt.Figure:
    """
    Radar chart comparing current team vs simulated team.
    
    Visual Design:
        - Figure size: 8x8 inches (high resolution, display controlled by display_figure)
        - Radial bounds: [-0.5, 1.1] for cleaner center
        - Reference circles at r=0 and r=1 (black, alpha=0.5)
        - Y-ticks only at [0, 0.5, 1] with "League Percentile" label
        - Current: blue solid line with fill
        - After: green dashed line with fill
        - Opponents: faded gray lines in background (linewidth=0.5, alpha=0.2)
    
    Normalization (CRITICAL):
        - Min-max normalized against LEAGUE values only (current + opponents)
        - Does NOT include after_totals in baseline (would distort comparison)
        - For negative categories (ERA, WHIP), normalization is inverted
        - Values clamped to [0, 1] if after-trade values fall outside league range
    
    Uses shared sorting logic from optimizer/visualizations.py:
    - Hitting: sorted DESCENDING (best first, clockwise from top)
    - Pitching: sorted ASCENDING (worst first, so best ends adjacent to hitting's best)
    """
```

**Radar Chart Sorting Logic:**

Implemented in `optimizer/visualizations.py` as `sort_categories_for_radar()`:

```python
from optimizer.visualizations import sort_categories_for_radar

# Single source of truth for sorting - used by both plot_team_radar() and 
# radar_chart_with_overlay()
sorted_categories = sort_categories_for_radar(current_totals, opponent_totals, categories)
```

The sorting works as follows:
1. Compute normalized performance score for each category (higher = better for my team)
2. For NEGATIVE_CATEGORIES (ERA, WHIP), invert so lower values = higher scores
3. Sort hitting categories DESCENDING (best first, clockwise from 12 o'clock)
4. Sort pitching categories ASCENDING (worst first, so best is last)
5. Concatenate: `sorted_hitting + sorted_pitching`

This makes hitting maxima appear at the top-left and pitching maxima appear at the bottom-left, consolidating the team's strengths visually.

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

## Implementation Notes

1. **Session State is Critical**: Always use `st.rerun()` after modifying session state from button callbacks to see updates.

2. **Button State Display**: Use `type="primary"` for main action buttons, `type="secondary"` for already-completed actions.

3. **Empty List Handling**: Check `if results is not None:` not `if results:` to distinguish between "not computed yet" and "computed but empty".

4. **Name Suffix Stripping**: Always use `strip_name_suffix()` when displaying player names to remove `-H`/`-P` suffixes.

5. **Opponent Roster Indexing**: Convert opponent rosters dict to indexed format before passing to trade engine:
   ```python
   opponent_rosters_indexed = {
       i + 1: names for i, (team, names) in enumerate(opponent_rosters.items())
   }
   ```

6. **Column Availability**: Always filter display columns to those that exist in the DataFrame:
   ```python
   available_cols = [c for c in display_cols if c in df.columns]
   ```

7. **Streamlit Deprecation**: Use `width="stretch"` instead of `use_container_width=True` (deprecated after 2025-12-31) for `st.button`, `st.dataframe`, and similar widgets.

8. **Figure Display**: Use `display_figure(fig, width=N)` from `dashboard/components.py` instead of `st.pyplot()` for precise control over display size. This saves the figure to a buffer and uses `st.image()` with explicit width.

9. **Combined Dashboards**: For Trade Builder and Free Agents, use `plot_comparison_dashboard()` from `optimizer/visualizations.py` to show before/after analysis. Display at `width=2200` for full-width view.

10. **Actual vs Projected Standings**: The Overview page shows both:
    - Actual standings from `st.session_state.standings` (fetched from Fantrax, stored in DB)
    - Projected standings computed from projections (in a collapsible expander)

---

## Notes

- All data operations use functions from `optimizer/database.py` and `optimizer/fantrax_api.py`
- Visualizations reuse existing functions from `optimizer/visualizations.py` where possible
- The dashboard is read-only — it recommends actions but doesn't execute them on Fantrax
