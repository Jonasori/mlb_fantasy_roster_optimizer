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
2. **Specification compliance** — Follow these docs exactly. Don't improvise or add unrequested features.
3. **Simplicity** — No unnecessary abstractions, plain data structures, obvious code paths.

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

**Run notebook:** `uv run marimo edit notebook.py`
**Run notebook as dashboard:** `uv run marimo run notebook.py`

---

## Core architecture: one core `players` table (feature-style enrichment)

**This is the central architectural idea.** Keep the **current picture of the league at player grain** in a single primary table (`players`)—the same spirit as **feature engineering** on one base table: add derived per-player fields as **columns**, then run **offshoot analytics** (lineup solves, swap/trade logic, plots, reports) from that table **plus** any explicit non-tabular outputs.

The **column-level contract** for v2 is **`design_descriptions/IMPLEMENTATION_SPEC.md` §§1–2** (silver input table, gold columns added by the math pipeline). Stay aligned with that spec for names and invariants.

### Defaults (prefer this shape)

1. **Prefer new columns on `players`** for per-player quantities that later stages read—so roster/value state is not split across ad hoc parallel structures.

2. **Avoid scattered parallel per-player maps**—especially `dict` keyed by name (or any structure) that the caller must **manually align and merge** into the table. If the result is effectively one value per player in the same grain as `players`, put it **on the DataFrame** (new column or an immediate, documented merge on `Name` / row index in the same step). The problem is **orphan side data**, not the use of a `Series` per se.

3. **`pd.Series` is fine** when it is **aligned** with `players` (same index or merge key) and you **assign or merge it into a column** in the same logical step—so there is still one obvious row-level store for downstream code.

4. **Extra return values are fine** when they are **not** per-player merge-back data—e.g. `(players, lineup_solution)`, recommendation lists, league-level dicts, solver status. Do not force non-tabular artifacts into the table; make the split explicit at the API.

5. **Copy before mutating:** at the start of a function that adds columns, `players = players.copy()` so callers are not surprised by in-place mutation.

6. **Column names are part of the API:** in docstrings, state which columns are required and which are added; match **IMPLEMENTATION_SPEC** for gold columns.

### Illustrative flow

```python
# `players` already satisfies design_descriptions/IMPLEMENTATION_SPEC.md §1 (silver).
players = add_fantasy_value(players, ...)
players = assign_optimal_slots(players, ...)
players = add_perceived_value(players, ...)
players = add_mew(players, ...)
players = add_bench_value(players, ...)
# Exact stage names, order, and side outputs are defined in IMPLEMENTATION_SPEC.
```

Chaining with `pd.DataFrame.pipe()` is optional where it improves readability.

### When a function is not “add a column”

Not every stage enriches the wide table. Examples:

- **Silver table production** — upstream; yields initial `players` per IMPLEMENTATION_SPEC §1.
- **League-level aggregation** — dicts or small structs (team totals, fixed-point state, etc.).
- **Lineup / MILP solvers** — consume `players`; may return assignments, objectives, or status **beside** the table.
- **Swap / trade and similar outputs** — lists or structs; IMPLEMENTATION_SPEC does **not** require these as `players` columns.
- **Visualization** — takes tables and/or summaries; returns figures.
- **Persistence / sync** — I/O outside the math pipeline.

**The default still applies** to work that is “compute a per-player field that downstream roster or value logic depends on”: keep it on `players` (or merged in immediately), not in a parallel map the caller must remember to join.

### Example: avoid vs prefer

```python
# AVOID — per-player results only in a parallel dict; caller must align keys to rows
def mew_by_name(players: pd.DataFrame, ...) -> dict[str, float]:
    ...


# PREFER — same information lives on the table (names per IMPLEMENTATION_SPEC §2)
def add_mew(players: pd.DataFrame, ...) -> pd.DataFrame:
    """Requires columns: ... Adds columns: MEW."""
    players = players.copy()
    players["MEW"] = ...
    return players
```

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


## Data handling principles

### Silver table and upstream scope

For **v2 optimizer code**, the authoritative definition of the input `players` DataFrame—**required and optional columns**, **`-H` / `-P` `Name` values**, and **invariants**—is **`design_descriptions/IMPLEMENTATION_SPEC.md` §1**.

Building that silver table (FanGraphs CSVs, Fantrax/roster data, name corrections, position merge, etc.) is **outside** the v2 math spec. Those steps must run **before** `players` enters the pipeline so the contract holds.

### Display names

Internal rows use **`Name` with a `-H` (hitter) or `-P` (pitcher) suffix** so the two sides are distinct. For user-facing strings, call **`strip_name_suffix()`** from **`optimizer/players.py`** only. **Do not** duplicate that helper elsewhere.

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

- [ ] **Per-player fields for downstream roster/value logic live on `players`** — avoid leaving parallel per-player dicts or other side maps for the caller to merge by hand; align with `design_descriptions/IMPLEMENTATION_SPEC.md` §§1–2
- [ ] No classes anywhere
- [ ] No try/except blocks
- [ ] All assertions have descriptive messages
- [ ] All functions have type hints and docstrings
- [ ] Print statements at key stages
- [ ] tqdm for loops > few seconds
- [ ] Ratio stats computed as weighted averages (PA for OPS, IP for ERA/WHIP)
- [ ] Player names include -H/-P suffix internally
- [ ] Display functions strip suffix for output
- [ ] Silver `players` input matches `design_descriptions/IMPLEMENTATION_SPEC.md` §1; upstream owns ingestion and normalization before the optimizer runs
- [ ] `strip_name_suffix()` exists only in `optimizer/players.py` and is imported for display
