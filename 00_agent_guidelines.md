# Agent Guidelines and Code Style

## Overview

This document establishes the coding standards, architectural constraints, and implementation philosophy for the MLB Fantasy Roster Optimizer. All subsequent implementation specs assume adherence to these guidelines.

**Target audience:** An AI coding agent implementing this system from scratch.

---

## Cross-References

**Depends on:** None (foundational document)

**Used by:** All other specs — every module must follow these guidelines

---

## Implementation Priorities

When making decisions, prioritize in this order:

1. **Correctness** — Code must work. Crash loudly with clear messages if something is wrong.
2. **Simplicity** — Fewer abstractions, plain data structures, obvious code paths.
3. **Specification compliance** — Follow these docs exactly. Don't improvise or add unrequested features.

---

## Mandatory Constraints

### No Object-Oriented Programming

**This is non-negotiable.** The entire codebase must use:
- Module-level functions only
- Plain data structures: `dict`, `list`, `tuple`, `set`
- NumPy arrays and pandas DataFrames
- No classes, no methods, no inheritance, no class-based patterns whatsoever

```python
# CORRECT
def compute_team_totals(player_names: set[str], projections: pd.DataFrame) -> dict[str, float]:
    """Compute totals as a pure function."""
    ...

# WRONG - Never do this
class Team:
    def __init__(self, players):
        self.players = players
    
    def compute_totals(self):
        ...
```

### Fail Fast Philosophy

**Never write try/except blocks.** Not even for I/O operations.

**Never write fallback logic.** If something is wrong, the program should crash immediately with a clear, actionable error message.

**Every `assert` must have a descriptive message:**

```python
# CORRECT
assert len(roster_names) == ROSTER_SIZE, (
    f"Expected {ROSTER_SIZE} players on roster, got {len(roster_names)}. "
    f"Check that roster file has correct number of active players."
)

# WRONG - No message
assert len(roster_names) == ROSTER_SIZE

# WRONG - Unhelpful message
assert len(roster_names) == ROSTER_SIZE, "Bad roster size"
```

### Status Reporting

Use `print()` for status updates at key stages:
```python
print(f"Loaded {len(hitters)} hitter projections ({n_with_pos} with positions from DB)")
print(f"Solving MILP with {len(candidates)} candidates...")
print(f"Solved in {elapsed:.1f}s — objective: {obj}/60 opponent-category wins")
```

Use `tqdm` for progress bars on any loop that takes more than a few seconds:
```python
from tqdm.auto import tqdm

for player in tqdm(candidates, desc="Computing sensitivities"):
    ...
```

**Import from `tqdm.auto`** for proper notebook/terminal display.

---

## Project Structure

```
mlb_fantasy_roster_optimizer/
├── optimizer/
│   ├── __init__.py           # Minimal exports
│   ├── data_loader.py        # Config, FanGraphs loading, team totals, SGP
│   ├── fantrax_api.py        # Fantrax API integration (rosters, ages, standings)
│   ├── database.py           # SQLite schema, sync, queries (PRIMARY DATA SOURCE)
│   ├── roster_optimizer.py   # MILP formulation and solving
│   ├── trade_engine.py       # Trade analysis and recommendations
│   └── visualizations.py     # All plotting functions
├── dashboard/
│   ├── app.py                # Streamlit main entry point
│   └── components.py         # Reusable UI components
├── tests/
│   └── test_core.py          # All tests in one file (minimal, no classes)
├── data/
│   ├── fangraphs-steamer-projections-hitters.csv
│   ├── fangraphs-steamer-projections-pitchers.csv
│   └── optimizer.db          # SQLite database (generated, PRIMARY DATA SOURCE)
├── config.json               # All configuration including Fantrax cookies (not in git)
├── implementation_specs/     # This documentation
├── notebook.py               # Marimo notebook (at project root)
└── pyproject.toml
```

### Database Is Primary

**Critical architectural decision:** The SQLite database (`data/optimizer.db`) is the single source of truth for all player data.

- FanGraphs CSVs and Fantrax API are **inputs** that sync to the database
- All optimizer and trade engine code queries the database via `get_projections()`, `get_roster_names()`, etc.
- DataFrames in the codebase are query results, not independent data structures
- This enables caching, fast queries, and consistent state

---

## Package Management

Use `uv` for dependency management. Create `pyproject.toml` at the project root:

```toml
[project]
name = "roster-optimizer"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.0",
    "numpy>=1.24",
    "pulp>=2.8",
    "highspy>=1.5",
    "matplotlib>=3.7",
    "seaborn>=0.12",
    "scipy>=1.10",
    "tqdm>=4.65",
    "marimo>=0.8",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["optimizer"]
```

**Installation:** `uv sync`

**Run notebook:** `marimo edit notebook.py`

---

## Function Design Principles

### Configuration via Arguments

Pass all configuration as function arguments. No global mutable state.

Module-level **constants** are acceptable for true constants:
```python
# OK - True constants
HITTING_CATEGORIES = ['R', 'HR', 'RBI', 'SB', 'OPS']
ROSTER_SIZE = 26

# NOT OK - Mutable global state
current_roster = set()  # Don't do this
```

### Type Hints

Use type hints for all function signatures. Use modern Python syntax:
```python
def filter_candidates(
    projections: pd.DataFrame,
    quality_scores: pd.DataFrame,
    my_roster_names: set[str],
    opponent_roster_names: set[str],
    top_n_per_position: int = 30,
) -> pd.DataFrame:
    ...
```

### Docstrings

Every public function needs a docstring with:
1. One-line summary
2. Args section (if parameters are non-obvious)
3. Returns section
4. Any critical implementation notes

```python
def compute_team_totals(
    player_names: Iterable[str],
    projections: pd.DataFrame,
) -> dict[str, float]:
    """
    Compute a team's projected totals across all 10 scoring categories.
    
    Args:
        player_names: Iterable of player names on the team.
        projections: Combined projections DataFrame with all stats.
    
    Returns:
        Dict mapping category name to team total.
        Example: {'R': 823, 'HR': 245, ..., 'ERA': 3.85, ...}
    
    Note:
        Ratio stats (OPS, ERA, WHIP) are computed as weighted averages,
        NOT simple sums. OPS uses PA weighting; ERA/WHIP use IP weighting.
    """
```

---

## Visualization Standards

All visualization functions must:

1. **Return a `matplotlib.Figure` object** — never call `plt.show()`
2. Accept data as arguments — no side effects or global state
3. Use consistent styling across the package

```python
def plot_category_margins(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
) -> plt.Figure:
    """
    Plot margin over each opponent in each category.
    
    Returns:
        Figure object. Caller is responsible for display.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    # ... plotting code ...
    return fig  # Never plt.show()
```

---

## Data Handling Principles

### Player Names Are Globally Unique

All player names include a `-H` (hitter) or `-P` (pitcher) suffix for uniqueness. Display functions strip the suffix using `strip_name_suffix()` from `data_loader.py`.

**Important:** `strip_name_suffix()` is defined ONLY in `data_loader.py` and imported everywhere else. Do NOT duplicate this function.

See `01a_config.md` and `01c_fantrax_api.md` for detailed name handling (corrections, normalization, suffix management).

### Ratio Stats and SGP Calculation

**Critical:** Ratio stats (OPS, ERA, WHIP) must be computed as weighted averages, not sums. SGP calculation weights rate stats by playing time.

See `01a_config.md` for detailed implementation of `compute_sgp_value()` and ratio stat handling.

### Sign Conventions for "Lower is Better" Stats

For ERA and WHIP, "winning" means having a LOWER value. All gap computations should normalize so that **positive = winning**:

```python
# For R, HR, OPS (higher is better):
gap = my_total - opponent_total

# For ERA, WHIP (lower is better):
gap = opponent_total - my_total  # Flip the sign!
```

---

## Solver Configuration

Use HiGHS via Python bindings (NOT command-line):

```python
import pulp

# CORRECT - Python bindings with highspy
solver = pulp.HiGHS_CMD(msg=True, timeLimit=300)

# Alternative if HiGHS_CMD doesn't work: use CBC as fallback
# solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=300)
```

**Note:** The exact solver class may vary by PuLP version. If `HiGHS_CMD` fails, the `highspy` package provides HiGHS. Check `pulp.listSolvers(onlyAvailable=True)` to see available solvers.

**HiGHS does NOT support warm-starting in PuLP.** Each MILP solve starts from scratch. Plan accordingly for sensitivity analysis.

---

## PuLP Variable Naming

Use only alphanumeric characters and underscores. Never put player names in variable names:

```python
# CORRECT - Integer indices
x = {i: pulp.LpVariable(f"x_{i}", cat='Binary') for i in range(n_players)}

# WRONG - Player names can have special characters
x = {name: pulp.LpVariable(f"x_{name}", cat='Binary') for name in player_names}
```

---

## Error Messages

When validation fails, provide actionable context:

```python
# Find unmatched names
unmatched = roster_names - set(projections['Name'])
assert len(unmatched) == 0, (
    f"Found {len(unmatched)} roster names not in projections:\n"
    f"  {sorted(unmatched)}\n"
    f"Run apply_name_corrections() to fix spelling/accent differences, "
    f"or manually update the roster file."
)
```

---

## Testing Philosophy

This codebase uses **minimal, focused testing**:

1. **Assertions embedded in functions** catch invariant violations at runtime
2. **Print statements** report progress and intermediate results
3. **The notebook serves as the primary integration test** — if it runs end-to-end, the system works
4. **A small pytest suite** provides guardrails for the implementing agent

### Test Design Principles

- **No classes** — all tests are module-level functions
- **No fixtures** — each test is self-contained with inline test data
- **No mocking** — tests use real functions with minimal test data
- **Fail fast** — tests assert clearly, no try/except
- **Skip integration tests** that require external dependencies (Fantrax cookies)

The test suite is intentionally minimal. Its purpose is to:
- Verify imports and module structure work
- Catch obvious bugs in core calculations (SGP, team totals, ratio stats)
- Ensure function signatures match expectations

If the tests pass and the notebook runs end-to-end, the system is working.

---

## Summary Checklist

Before considering any module complete:

- [ ] No classes anywhere
- [ ] No try/except blocks
- [ ] All assertions have descriptive messages
- [ ] All functions have type hints and docstrings
- [ ] Print statements at key stages
- [ ] tqdm for loops > few seconds
- [ ] Visualizations return Figure objects (no plt.show())
- [ ] Ratio stats computed as weighted averages (PA for OPS, IP for ERA/WHIP)
- [ ] SGP calculation weights rate stats by playing time
- [ ] Player names include -H/-P suffix internally
- [ ] Display functions strip suffix for output
- [ ] All data queries go through database functions (not direct CSV loading)
- [ ] Name corrections applied BEFORE adding -H/-P suffix
