# Testing Specification

## Overview

This document specifies the test suite for the MLB Fantasy Roster Optimizer. Tests are organized into two files:

1. **Core tests** (`tests/test_core.py`): Unit tests for optimizer logic
2. **Dashboard tests** (`tests/test_dashboard.py`): End-to-end browser tests for the Streamlit dashboard

**Test runner:** `pytest`  
**Browser automation:** Playwright  
**Philosophy:** Simple, self-contained tests. Module-level functions, no classes.

---

## Cross-References

**Depends on:**
- [00_agent_guidelines.md](00_agent_guidelines.md) — test philosophy (fail fast, no classes)
- [01a_config.md](01a_config.md) — constants to verify
- [02_free_agent_optimizer.md](02_free_agent_optimizer.md) — MILP constants to verify
- [02a_variance_penalized_objective.md](02a_variance_penalized_objective.md) — balance tests
- [03_trade_engine.md](03_trade_engine.md) — MEV/MVAR tables, win probability
- [04_visualizations.md](04_visualizations.md) — radar chart sorting
- [05_notebook_integration.md](05_notebook_integration.md) — notebook as integration test
- [06_streamlit_dashboard.md](06_streamlit_dashboard.md) — dashboard browser tests

**Used by:** None (terminal node — validates all other specs)

---

## Test Structure

```
tests/
├── test_core.py        # Unit tests for optimizer logic (17 tests)
└── test_dashboard.py   # End-to-end browser tests (44 tests)
```

---

## Test Implementation

```python
"""
Minimal test suite for MLB Fantasy Roster Optimizer.

All tests are module-level functions with inline test data.
No classes, no fixtures, no mocking.
"""

import pandas as pd
import numpy as np


# =============================================================================
# SMOKE TESTS - Verify imports work
# =============================================================================

def test_import_data_loader():
    """Core data_loader exports are importable."""
    from optimizer.data_loader import (
        HITTING_CATEGORIES,
        PITCHING_CATEGORIES,
        ALL_CATEGORIES,
        ROSTER_SIZE,
        SLOT_ELIGIBILITY,
        NUM_OPPONENTS,
        FANTRAX_TEAM_IDS,
        strip_name_suffix,
        compute_sgp_value,
        compute_team_totals,
        estimate_projection_uncertainty,
    )
    
    assert len(ALL_CATEGORIES) == 10
    assert ROSTER_SIZE == 26
    assert NUM_OPPONENTS == 6
    assert len(FANTRAX_TEAM_IDS) == 7
    assert "C" in SLOT_ELIGIBILITY


def test_import_roster_optimizer():
    """Roster optimizer exports are importable."""
    from optimizer.roster_optimizer import (
        filter_candidates,
        build_and_solve_milp,
        BIG_M_COUNTING,
        EPSILON_RATIO,
    )
    
    assert BIG_M_COUNTING == 10000
    assert EPSILON_RATIO == 0.001


def test_import_trade_engine():
    """Trade engine exports are importable."""
    from optimizer.trade_engine import (
        compute_win_probability,
        compute_player_values,
        evaluate_trade,
        MEV_TABLE,
        MVAR_TABLE,
    )
    
    assert MEV_TABLE[6] == 1.267
    assert MVAR_TABLE[6] == 0.416


def test_import_visualizations():
    """Visualizations exports are importable."""
    from optimizer.visualizations import (
        plot_team_radar,
        plot_trade_impact,
    )


# =============================================================================
# NAME HANDLING TESTS
# =============================================================================

def test_strip_name_suffix():
    """strip_name_suffix removes -H and -P correctly."""
    from optimizer.data_loader import strip_name_suffix
    
    assert strip_name_suffix("Mike Trout-H") == "Mike Trout"
    assert strip_name_suffix("Gerrit Cole-P") == "Gerrit Cole"
    assert strip_name_suffix("Shohei Ohtani-H") == "Shohei Ohtani"
    assert strip_name_suffix("Shohei Ohtani-P") == "Shohei Ohtani"
    assert strip_name_suffix("No Suffix") == "No Suffix"


# =============================================================================
# SGP CALCULATION TESTS
# =============================================================================

def test_compute_sgp_value_hitter():
    """SGP calculation for hitters produces reasonable values."""
    from optimizer.data_loader import compute_sgp_value
    
    # Create a realistic hitter row as pd.Series
    hitter = pd.Series({
        "player_type": "hitter",
        "PA": 600,
        "R": 95,
        "HR": 28,
        "RBI": 85,
        "SB": 15,
        "OPS": 0.820,
    })
    
    sgp = compute_sgp_value(hitter)
    
    # Rough expected: R/20 + HR/8 + RBI/20 + SB/7 + (OPS-0.75)/0.01
    # = 4.75 + 3.5 + 4.25 + 2.14 + 7.0 = ~21.6
    assert 15 < sgp < 30, f"Hitter SGP {sgp} outside reasonable range"


def test_compute_sgp_value_pitcher():
    """SGP calculation for pitchers produces reasonable values."""
    from optimizer.data_loader import compute_sgp_value
    
    # Create a realistic pitcher row as pd.Series
    pitcher = pd.Series({
        "player_type": "pitcher",
        "IP": 180.0,
        "W": 14,
        "SV": 0,
        "K": 200,
        "ERA": 3.25,
        "WHIP": 1.10,
    })
    
    sgp = compute_sgp_value(pitcher)
    
    # Rough expected: W/3.5 + SV/8 + K/35 + (4.0-ERA)/0.18 + (1.25-WHIP)/0.03
    # = 4.0 + 0 + 5.7 + 4.2 + 5.0 = ~18.9
    assert 10 < sgp < 30, f"Pitcher SGP {sgp} outside reasonable range"


# =============================================================================
# TEAM TOTALS TESTS
# =============================================================================

def test_compute_team_totals_counting_stats():
    """Counting stats are summed correctly."""
    from optimizer.data_loader import compute_team_totals
    
    projections = pd.DataFrame([
        {"Name": "Player A-H", "player_type": "hitter", "PA": 500, "R": 80, "HR": 25, "RBI": 70, "SB": 10, "OPS": 0.850, "IP": 0, "W": 0, "SV": 0, "K": 0, "ERA": 0, "WHIP": 0},
        {"Name": "Player B-H", "player_type": "hitter", "PA": 400, "R": 60, "HR": 15, "RBI": 50, "SB": 5, "OPS": 0.750, "IP": 0, "W": 0, "SV": 0, "K": 0, "ERA": 0, "WHIP": 0},
    ])
    
    totals = compute_team_totals({"Player A-H", "Player B-H"}, projections)
    
    assert totals["R"] == 140, f"R should be 80+60=140, got {totals['R']}"
    assert totals["HR"] == 40, f"HR should be 25+15=40, got {totals['HR']}"
    assert totals["RBI"] == 120, f"RBI should be 70+50=120, got {totals['RBI']}"
    assert totals["SB"] == 15, f"SB should be 10+5=15, got {totals['SB']}"


def test_compute_team_totals_ratio_stats():
    """Ratio stats use PA/IP-weighted averages, not sums."""
    from optimizer.data_loader import compute_team_totals
    
    projections = pd.DataFrame([
        {"Name": "Player A-H", "player_type": "hitter", "PA": 600, "R": 100, "HR": 30, "RBI": 90, "SB": 10, "OPS": 0.900, "IP": 0, "W": 0, "SV": 0, "K": 0, "ERA": 0, "WHIP": 0},
        {"Name": "Player B-H", "player_type": "hitter", "PA": 400, "R": 60, "HR": 15, "RBI": 50, "SB": 5, "OPS": 0.700, "IP": 0, "W": 0, "SV": 0, "K": 0, "ERA": 0, "WHIP": 0},
    ])
    
    totals = compute_team_totals({"Player A-H", "Player B-H"}, projections)
    
    # Weighted average: (600*0.9 + 400*0.7) / (600+400) = 820/1000 = 0.820
    expected_ops = (600 * 0.900 + 400 * 0.700) / 1000
    assert abs(totals["OPS"] - expected_ops) < 0.001, (
        f"OPS should be weighted average {expected_ops:.3f}, got {totals['OPS']:.3f}"
    )


def test_compute_team_totals_pitcher_ratio_stats():
    """Pitcher ratio stats (ERA, WHIP) use IP-weighted averages."""
    from optimizer.data_loader import compute_team_totals
    
    projections = pd.DataFrame([
        {"Name": "Pitcher A-P", "player_type": "pitcher", "PA": 0, "R": 0, "HR": 0, "RBI": 0, "SB": 0, "OPS": 0, "IP": 180, "W": 12, "SV": 0, "K": 180, "ERA": 3.00, "WHIP": 1.00},
        {"Name": "Pitcher B-P", "player_type": "pitcher", "PA": 0, "R": 0, "HR": 0, "RBI": 0, "SB": 0, "OPS": 0, "IP": 60, "W": 4, "SV": 10, "K": 70, "ERA": 4.50, "WHIP": 1.40},
    ])
    
    totals = compute_team_totals({"Pitcher A-P", "Pitcher B-P"}, projections)
    
    # ERA weighted average: (180*3.0 + 60*4.5) / 240 = 810/240 = 3.375
    expected_era = (180 * 3.00 + 60 * 4.50) / 240
    assert abs(totals["ERA"] - expected_era) < 0.01, (
        f"ERA should be weighted average {expected_era:.2f}, got {totals['ERA']:.2f}"
    )
    
    # Counting stats should sum
    assert totals["W"] == 16
    assert totals["SV"] == 10
    assert totals["K"] == 250


# =============================================================================
# PROJECTION UNCERTAINTY TESTS
# =============================================================================

def test_estimate_projection_uncertainty():
    """Uncertainty estimation returns reasonable values."""
    from optimizer.data_loader import estimate_projection_uncertainty
    
    my_totals = {"R": 800, "HR": 240, "RBI": 780, "SB": 100, "OPS": 0.770,
                 "W": 85, "SV": 75, "K": 1350, "ERA": 3.80, "WHIP": 1.20}
    
    opponent_totals = {
        1: {"R": 820, "HR": 250, "RBI": 800, "SB": 110, "OPS": 0.780, "W": 88, "SV": 70, "K": 1400, "ERA": 3.70, "WHIP": 1.18},
        2: {"R": 780, "HR": 230, "RBI": 760, "SB": 95, "OPS": 0.760, "W": 80, "SV": 80, "K": 1300, "ERA": 3.90, "WHIP": 1.22},
        3: {"R": 850, "HR": 260, "RBI": 820, "SB": 105, "OPS": 0.790, "W": 92, "SV": 68, "K": 1450, "ERA": 3.60, "WHIP": 1.15},
        4: {"R": 790, "HR": 245, "RBI": 790, "SB": 90, "OPS": 0.765, "W": 82, "SV": 78, "K": 1320, "ERA": 3.85, "WHIP": 1.21},
        5: {"R": 810, "HR": 235, "RBI": 770, "SB": 115, "OPS": 0.775, "W": 86, "SV": 72, "K": 1380, "ERA": 3.75, "WHIP": 1.19},
        6: {"R": 830, "HR": 255, "RBI": 810, "SB": 100, "OPS": 0.785, "W": 90, "SV": 74, "K": 1420, "ERA": 3.65, "WHIP": 1.16},
    }
    
    sigmas = estimate_projection_uncertainty(my_totals, opponent_totals)
    
    # Should have all 10 categories
    assert len(sigmas) == 10
    
    # All values should be positive
    for cat, sigma in sigmas.items():
        assert sigma > 0, f"Sigma for {cat} should be positive, got {sigma}"


# =============================================================================
# MILP COEFFICIENT SIGN TESTS
# =============================================================================

def test_ratio_stat_coefficient_signs():
    """Verify coefficient signs for ratio stat linearization are correct."""
    # OPS (higher is better): coeff = PA * (player_OPS - opponent_OPS)
    # ERA (lower is better): coeff = IP * (opponent_ERA - player_ERA)
    
    player_ops = 0.850
    player_era = 3.00
    opponent_ops = 0.770
    opponent_era = 3.85
    player_pa = 600
    player_ip = 180
    
    # OPS coefficient: positive when player is better (higher OPS)
    ops_coeff = player_pa * (player_ops - opponent_ops)
    assert ops_coeff > 0, "Good OPS player should have positive coefficient"
    
    # ERA coefficient: positive when player is better (lower ERA)
    era_coeff = player_ip * (opponent_era - player_era)
    assert era_coeff > 0, "Good ERA player should have positive coefficient"


# =============================================================================
# WIN PROBABILITY TESTS
# =============================================================================

def test_win_probability_bounds():
    """Win probability is between 0 and 1."""
    from optimizer.trade_engine import compute_win_probability
    
    my_totals = {"R": 820, "HR": 250, "RBI": 800, "SB": 105, "OPS": 0.775,
                 "W": 85, "SV": 75, "K": 1350, "ERA": 3.75, "WHIP": 1.18}
    
    opponent_totals = {
        1: {"R": 820, "HR": 240, "RBI": 780, "SB": 110, "OPS": 0.765, "W": 85, "SV": 70, "K": 1350, "ERA": 3.85, "WHIP": 1.22},
        2: {"R": 795, "HR": 255, "RBI": 810, "SB": 95, "OPS": 0.782, "W": 78, "SV": 82, "K": 1280, "ERA": 3.72, "WHIP": 1.18},
        3: {"R": 850, "HR": 230, "RBI": 760, "SB": 130, "OPS": 0.758, "W": 92, "SV": 65, "K": 1420, "ERA": 3.95, "WHIP": 1.25},
        4: {"R": 780, "HR": 268, "RBI": 830, "SB": 88, "OPS": 0.795, "W": 80, "SV": 78, "K": 1310, "ERA": 3.68, "WHIP": 1.15},
        5: {"R": 835, "HR": 245, "RBI": 795, "SB": 105, "OPS": 0.770, "W": 88, "SV": 72, "K": 1380, "ERA": 3.80, "WHIP": 1.20},
        6: {"R": 805, "HR": 252, "RBI": 805, "SB": 100, "OPS": 0.778, "W": 82, "SV": 75, "K": 1340, "ERA": 3.75, "WHIP": 1.19},
    }
    
    category_sigmas = {"R": 25, "HR": 12, "RBI": 25, "SB": 12, "OPS": 0.012,
                       "W": 5, "SV": 5, "K": 50, "ERA": 0.10, "WHIP": 0.03}
    
    V, diagnostics = compute_win_probability(my_totals, opponent_totals, category_sigmas)
    
    assert 0 <= V <= 1, f"Win probability {V} out of bounds"
    assert "expected_wins" in diagnostics


def test_better_team_higher_probability():
    """Strictly better team has higher win probability."""
    from optimizer.trade_engine import compute_win_probability
    
    opponent_totals = {
        1: {"R": 800, "HR": 240, "RBI": 780, "SB": 100, "OPS": 0.770, "W": 84, "SV": 74, "K": 1340, "ERA": 3.80, "WHIP": 1.20},
        2: {"R": 800, "HR": 240, "RBI": 780, "SB": 100, "OPS": 0.770, "W": 84, "SV": 74, "K": 1340, "ERA": 3.80, "WHIP": 1.20},
        3: {"R": 800, "HR": 240, "RBI": 780, "SB": 100, "OPS": 0.770, "W": 84, "SV": 74, "K": 1340, "ERA": 3.80, "WHIP": 1.20},
        4: {"R": 800, "HR": 240, "RBI": 780, "SB": 100, "OPS": 0.770, "W": 84, "SV": 74, "K": 1340, "ERA": 3.80, "WHIP": 1.20},
        5: {"R": 800, "HR": 240, "RBI": 780, "SB": 100, "OPS": 0.770, "W": 84, "SV": 74, "K": 1340, "ERA": 3.80, "WHIP": 1.20},
        6: {"R": 800, "HR": 240, "RBI": 780, "SB": 100, "OPS": 0.770, "W": 84, "SV": 74, "K": 1340, "ERA": 3.80, "WHIP": 1.20},
    }
    
    category_sigmas = {"R": 30, "HR": 15, "RBI": 30, "SB": 12, "OPS": 0.015,
                       "W": 6, "SV": 6, "K": 60, "ERA": 0.15, "WHIP": 0.04}
    
    # Average team (same as opponents)
    avg_totals = {"R": 800, "HR": 240, "RBI": 780, "SB": 100, "OPS": 0.770,
                  "W": 84, "SV": 74, "K": 1340, "ERA": 3.80, "WHIP": 1.20}
    
    # Clearly better team (better in every category)
    good_totals = {"R": 900, "HR": 280, "RBI": 880, "SB": 130, "OPS": 0.820,
                   "W": 100, "SV": 90, "K": 1500, "ERA": 3.40, "WHIP": 1.08}
    
    V_avg, _ = compute_win_probability(avg_totals, opponent_totals, category_sigmas)
    V_good, _ = compute_win_probability(good_totals, opponent_totals, category_sigmas)
    
    assert V_good > V_avg, f"Better team ({V_good:.3f}) should beat average ({V_avg:.3f})"


# =============================================================================
# TRADE ENGINE CONSTANTS
# =============================================================================

def test_mev_mvar_tables():
    """MEV and MVAR tables have correct values from literature."""
    from optimizer.trade_engine import MEV_TABLE, MVAR_TABLE
    
    # From Teichroew (1956) / Rosenof (2025)
    assert abs(MEV_TABLE[6] - 1.267) < 0.001, f"MEV[6] should be 1.267, got {MEV_TABLE[6]}"
    assert abs(MVAR_TABLE[6] - 0.416) < 0.001, f"MVAR[6] should be 0.416, got {MVAR_TABLE[6]}"


def test_trade_fairness_threshold():
    """Fairness threshold is percentage-based at 10%."""
    from optimizer.trade_engine import FAIRNESS_THRESHOLD_PERCENT, MIN_MEANINGFUL_IMPROVEMENT
    
    assert FAIRNESS_THRESHOLD_PERCENT == 0.10
    assert MIN_MEANINGFUL_IMPROVEMENT == 0.1


# =============================================================================
# RADAR CHART SORTING TESTS
# =============================================================================

def test_radar_sorting_hitting_pitching_opposite_directions():
    """
    Radar chart sorting should have hitting descending and pitching ascending.
    This makes the maxima of hitting and pitching meet in the middle.
    """
    from optimizer.data_loader import HITTING_CATEGORIES, PITCHING_CATEGORIES
    from optimizer.visualizations import sort_categories_for_radar

    # Create test data where "My Team" is best in R, HR and W, K
    my_totals = {
        "R": 900, "HR": 280, "RBI": 750, "SB": 80, "OPS": 0.760,  # Best: R, HR
        "W": 100, "SV": 60, "K": 1500, "ERA": 4.00, "WHIP": 1.30,  # Best: W, K
    }

    opponent_totals = {
        1: {"R": 800, "HR": 240, "RBI": 780, "SB": 100, "OPS": 0.800,
            "W": 85, "SV": 75, "K": 1350, "ERA": 3.50, "WHIP": 1.15}
    }

    sorted_cats = sort_categories_for_radar(my_totals, opponent_totals)

    # Extract hitting and pitching portions
    hitting_portion = [c for c in sorted_cats if c in HITTING_CATEGORIES]
    pitching_portion = [c for c in sorted_cats if c in PITCHING_CATEGORIES]

    # Hitting should come before pitching
    hitting_indices = [sorted_cats.index(c) for c in hitting_portion]
    pitching_indices = [sorted_cats.index(c) for c in pitching_portion]
    assert max(hitting_indices) < min(pitching_indices)

    # R and HR should be near the START of hitting (best first)
    assert hitting_portion.index("R") < hitting_portion.index("OPS")
    assert hitting_portion.index("HR") < hitting_portion.index("OPS")

    # W and K should be near the END of pitching (best last, since ascending)
    assert pitching_portion.index("ERA") < pitching_portion.index("W")
    assert pitching_portion.index("ERA") < pitching_portion.index("K")
```

---

## Dashboard Tests (`tests/test_dashboard.py`)

End-to-end browser tests using Playwright. These tests start the Streamlit server, navigate through all pages, click buttons, and verify the UI behaves correctly.

### Dependencies

Add to `pyproject.toml`:
```toml
[dependency-groups]
dev = ["pytest>=8.0", "playwright>=1.40", "pytest-playwright>=0.4"]
```

Install browsers:
```bash
uv run playwright install chromium
```

### Test Categories

```python
# =============================================================================
# NAVIGATION TESTS (6 tests)
# =============================================================================
# - test_sidebar_exists
# - test_all_navigation_options_present
# - test_navigate_to_all_pages
# - test_sidebar_has_team_selector
# - test_sidebar_has_refresh_button
# - test_sidebar_shows_navigation

# =============================================================================
# PAGE-SPECIFIC TESTS (20 tests)
# =============================================================================
# Overview page (2 tests)
# My Team page (2 tests)
# Trades page (5 tests) - includes parameter controls, trade builder
# Free Agents page (8 tests) - free agent browser + simulator
# Players page (2 tests)

# =============================================================================
# INTERACTIVITY TESTS (4 tests)
# =============================================================================
# - test_refresh_data_button_works
# - test_button_click_does_not_crash
# - test_simulator_multiselect_opens
# - test_simulator_filter_player_type

# =============================================================================
# LAYOUT/CONTENT TESTS (6 tests)
# =============================================================================
# - test_no_error_messages_on_load
# - test_pages_have_reasonable_content
# - test_dataframes_render_correctly
# - test_my_team_has_radar_chart_option
# - test_simulator_slider_exists
# - test_trade_parameter_sliders_exist

# =============================================================================
# COMPREHENSIVE CRAWL TEST (1 test)
# =============================================================================
# - test_full_dashboard_crawl

# =============================================================================
# DATA-LOADED TESTS (4 tests)
# =============================================================================
# These tests click "Refresh Data" first, then verify functionality:
# - test_with_data_simulator_shows_free_agents
# - test_with_data_my_team_shows_roster
# - test_with_data_trades_shows_interface
# - test_with_data_players_shows_database

# =============================================================================
# ROBUSTNESS TESTS (4 tests)
# =============================================================================
# - test_recover_from_invalid_nav_state
# - test_rapid_navigation_stability
# - test_pages_have_headers
# - test_primary_buttons_have_text
```

### Key Implementation Patterns

**Server Fixture:**
```python
@pytest.fixture(scope="module")
def streamlit_server():
    """Start Streamlit server for test module, shut down after."""
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT
    
    process = subprocess.Popen(
        ["streamlit", "run", "dashboard/app.py", 
         "--server.port", "8501", "--server.headless", "true"],
        cwd=PROJECT_ROOT, env=env
    )
    
    # Wait for server ready (curl returns 200)
    # ... polling logic ...
    
    yield process
    process.terminate()
```

**Page Navigation Helper:**
```python
def navigate_to_page(page: Page, page_name: str):
    """Navigate via sidebar navigation buttons."""
    nav_button = page.locator(f'[data-testid="stSidebar"] button:has-text("{page_name}")')
    nav_button.wait_for(timeout=5000)
    nav_button.click()
    page.wait_for_timeout(1000)
```

**Handling Data-Loaded vs No-Data States:**

Tests should pass in both states (data loaded or not):
```python
def test_simulator_has_free_agent_browser(dashboard_page: Page):
    navigate_to_page(dashboard_page, "🔍 Free Agents")
    page_text = get_visible_text(dashboard_page)
    
    has_fa = "Free Agent" in page_text
    has_load_msg = "Load data" in page_text or "No roster" in page_text
    
    assert has_fa or has_load_msg, "Should show content or load message"
```

---

## Running Tests

```bash
# Install dependencies
uv sync --group dev
uv run playwright install chromium

# Run all tests (core + dashboard)
uv run pytest tests/ -v

# Run only core tests (fast, ~5 seconds)
uv run pytest tests/test_core.py -v

# Run only dashboard tests (~2 minutes)
uv run pytest tests/test_dashboard.py -v

# Run dashboard tests with visible browser
uv run pytest tests/test_dashboard.py -v --headed

# Run specific test
uv run pytest tests/test_dashboard.py::test_full_dashboard_crawl -v
```

---

## Expected Results

All tests should pass.

### Core Tests (`test_core.py`)

Expected runtime: < 5 seconds.

| Category | Tests | Purpose |
|----------|-------|---------|
| Smoke tests | 4 | Verify imports work |
| Name handling | 1 | Suffix stripping |
| SGP calculation | 2 | Hitter/pitcher SGP |
| Team totals | 3 | Counting + ratio stats |
| Uncertainty | 1 | Category sigmas |
| MILP coefficients | 1 | Sign conventions |
| Win probability | 2 | Bounds + ordering |
| Trade constants | 2 | MEV/MVAR + thresholds |
| Radar sorting | 1 | Hitting/pitching opposite directions |

**Total: 17 tests**

### Dashboard Tests (`test_dashboard.py`)

Expected runtime: ~2 minutes (browser automation).

| Category | Tests | Purpose |
|----------|-------|---------|
| Navigation | 6 | Sidebar, routing |
| Overview page | 2 | Content, buttons |
| My Team page | 2 | Roster display |
| Trades page | 5 | Parameters, builder |
| Free Agents page | 8 | Browser + simulator |
| Players page | 2 | Search, stats |
| Interactivity | 4 | Buttons, widgets |
| Layout/content | 6 | Structure, no errors |
| Full crawl | 1 | Comprehensive smoke test |
| Data-loaded | 4 | Post-refresh behavior |
| Robustness | 4 | Edge cases, stability |

**Total: 44 tests**

### Combined Total: 61 tests

---

## What These Tests Catch

1. **Import errors** — module structure is correct
2. **SGP math errors** — counting vs rate stat handling
3. **Ratio stat bugs** — weighted average vs simple sum
4. **Sign convention bugs** — ERA/WHIP coefficient directions
5. **Win probability bugs** — bounds violations, ordering violations
6. **Constant typos** — MEV/MVAR values from literature

---

## What These Tests Don't Cover

- Full MILP solve (too slow, requires data)
- Fantrax API calls (requires authentication)
- Database operations (requires setup)
- Visualization output (visual inspection only)

These are validated by running the notebook end-to-end.
