# MLB Fantasy Roster Optimizer: Implementation Specifications

## Purpose

These specifications are designed to enable an AI coding agent to implement the complete roster optimization system in a single pass. Each document is self-contained with enough detail for implementation without ambiguity.

## Reading Order

Read the documents in numerical order:

| # | Document | Description |
|---|----------|-------------|
| 00 | [Agent Guidelines](00_agent_guidelines.md) | Code style, constraints, and implementation philosophy. **Read first.** |
| 01 | [Data and Config](01_data_and_config.md) | League settings, data loading, name matching, team totals. |
| 02 | [Free Agent Optimizer](02_free_agent_optimizer.md) | MILP formulation for optimal roster construction. |
| 03 | [Trade Engine](03_trade_engine.md) | Probabilistic win model and trade evaluation. |
| 04 | [Visualizations](04_visualizations.md) | All plotting functions. |
| 05 | [Notebook Integration](05_notebook_integration.md) | Complete workflow examples. |

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         notebook.py                              │
│                    (Marimo notebook UI)                          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       optimizer/                                 │
├─────────────────┬─────────────────┬─────────────────────────────┤
│  data_loader.py │roster_optimizer │  trade_engine.py            │
│                 │      .py        │                             │
│  • Load projs   │  • MILP         │  • Win probability          │
│  • Convert      │  • Candidate    │  • Player values            │
│  • Name match   │    filtering    │  • Trade evaluation         │
│  • Team totals  │  • Solve        │  • Recommendations          │
└────────┬────────┴────────┬────────┴──────────────┬──────────────┘
         │                 │                        │
         └─────────────────┴────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   visualizations.py    │
              │                        │
              │  • Radar charts        │
              │  • Heatmaps            │
              │  • Contribution plots  │
              │  • Trade impact        │
              └────────────────────────┘
```

## Key Design Decisions

### 1. No Object-Oriented Programming
All code uses module-level functions and plain data structures. No classes.

### 2. Two Optimization Approaches

| Approach | Use Case | Method |
|----------|----------|--------|
| **Free Agent Optimizer** | "What's my optimal roster?" | MILP (global optimum) |
| **Trade Engine** | "Which trades should I propose?" | Probabilistic model (marginal values) |

The free agent optimizer finds the globally optimal roster via integer programming. The trade engine uses a probabilistic win model to evaluate marginal player value and identify beneficial trades.

### 3. Probabilistic Win Model (Trade Engine)

Based on Rosenof (2025), the trade engine models win probability as:

```
V = Φ(μ_D / σ_D)
```

Where:
- `μ_D` = expected differential vs best opponent
- `σ_D` = std dev of that differential
- `Φ` = standard normal CDF

This is superior to heuristic approaches because:
- Uses continuous probabilities (not arbitrary thresholds)
- Accounts for variance (which matters in rotisserie!)
- Marginal value computed via numerical differentiation (simple and accurate)

### 4. Player Name Uniqueness

All player names include `-H` or `-P` suffix:
- `"Mike Trout-H"`, `"Gerrit Cole-P"`
- Eliminates all name collision issues
- Display functions strip suffix for output

### 5. Fail Fast

No try/except blocks. No fallback logic. Assert with descriptive messages. Crash immediately if something is wrong.

## Implementation Sequence

For an agent implementing this system:

1. **Set up project structure** (pyproject.toml, directories)
2. **Implement data_loader.py** (spec 01)
3. **Implement roster_optimizer.py** (spec 02)
4. **Implement trade_engine.py** (spec 03)
5. **Implement visualizations.py** (spec 04)
6. **Create notebook.py** (spec 05)
7. **Test with real data**

## Dependencies

```
pandas>=2.0
numpy>=1.24
pulp>=2.8
highspy>=1.5
scipy>=1.10
matplotlib>=3.7
seaborn>=0.12
tqdm>=4.65
marimo>=0.8
```

## References

- Rosenof, Z. (2025). "Optimizing for Rotisserie Fantasy Basketball." arXiv:2501.00933.
  - Provides the probabilistic win model used in the trade engine
  - Establishes that balanced teams with variance perform better in rotisserie

## Validation

Each spec includes a validation checklist. An implementation is complete when all checklist items are satisfied and the notebook runs end-to-end without errors.
