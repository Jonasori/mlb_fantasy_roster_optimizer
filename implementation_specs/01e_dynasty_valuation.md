# Dynasty Valuation

## Overview

Dynasty value is the **net present value (NPV)** of a player's expected future SGP contributions, accounting for aging-related decline and time-discounting.

**Module:** `optimizer/data_loader.py`

---

## Sources

The aging curves are based on empirical research:

1. **Ryan Brock (2016)**, "On the Use of Aging Curves for Fantasy Baseball," FanGraphs Community
   - Peak age: 26 for hitters, 26 for SP, 28 for RP
   
2. **Zimmerman & Petti (2013)**, "Hitters No Longer Peak, Only Decline"
   - Post-PED era players only decline after reaching MLB

3. **Dynasty Process**, "Market Values" methodology
   - Future picks valued at ~80% of current year (≈20-25% discount rate)

---

## Configuration

```python
# Discount rate for future years (aggressive "win-now" mode)
DYNASTY_DISCOUNT_RATE = 0.25  # Year 1 = 80%, Year 2 = 64%, Year 3 = 51%

# Projection horizon
DYNASTY_PROJECTION_YEARS = 4  # Current + 3 future years

# Yearly SGP decline by age (negative = decline)
HITTER_AGING_FACTORS = {
    22: 0.0, 23: 0.0, 24: 0.0, 25: 0.0, 26: 0.0,      # Peak years
    27: -0.8, 28: -0.9, 29: -1.0, 30: -1.1,           # Gradual decline
    31: -1.2, 32: -1.4, 33: -1.6, 34: -1.8,           # Accelerating
    35: -2.2, 36: -2.6, 37: -3.0, 38: -3.5, 39: -4.0, 40: -4.5,  # Steep
}

SP_AGING_FACTORS = {
    22: 0.0, 23: 0.0, 24: 0.0, 25: 0.0, 26: 0.0,
    27: 0.0, 28: 0.0, 29: 0.0, 30: 0.0, 31: 0.0,      # Longer plateau
    32: -0.6, 33: -0.8, 34: -1.0,
    35: -1.3, 36: -1.6, 37: -2.0, 38: -2.5, 39: -3.0, 40: -3.5,
}

RP_AGING_FACTORS = {
    22: 0.0, 23: 0.0, 24: 0.0, 25: 0.0, 26: 0.0, 27: 0.0, 28: 0.0,  # Peak at 28
    29: -0.4, 30: -0.5, 31: -0.6, 32: -0.7,
    33: -0.9, 34: -1.1, 35: -1.4, 36: -1.7, 37: -2.0, 38: -2.5, 39: -3.0, 40: -3.5,
}
```

---

## Mathematical Framework

### Dynasty SGP Formula

```
dynasty_SGP = Σ(t=0 to T-1) [ SGP_t / (1 + r)^t ]
```

Where:
- `T` = projection years (4)
- `r` = discount rate (0.25)
- `SGP_t = max(0, SGP_0 + cumulative_aging_decline)`

### Discount Factors (r = 0.25)

| Year | Factor | Interpretation |
|------|--------|----------------|
| 0 | 1.000 | 100% weight |
| 1 | 0.800 | 80% weight |
| 2 | 0.640 | 64% weight |
| 3 | 0.512 | 51% weight |

### Example: Trout (33) vs Rodriguez (25)

**Trout (Age 33, SGP 18.0):**
- Year 0: 18.0 × 1.000 = 18.0
- Year 1: 16.4 × 0.800 = 13.1  (−1.6 decline)
- Year 2: 14.6 × 0.640 = 9.3   (−1.8 decline)
- Year 3: 12.4 × 0.512 = 6.3   (−2.2 decline)
- **Dynasty SGP: 46.7**

**Rodriguez (Age 25, SGP 16.0):**
- Year 0: 16.0 × 1.000 = 16.0
- Year 1: 16.0 × 0.800 = 12.8  (no decline)
- Year 2: 15.2 × 0.640 = 9.7   (−0.8 decline)
- Year 3: 14.3 × 0.512 = 7.3   (−0.9 decline)
- **Dynasty SGP: 45.8**

Despite lower current SGP, the younger player has nearly equal dynasty value.

---

## Age Data Source: Fantrax API

Ages come from **Fantrax API** via `fetch_player_pool()`, which provides ages for ALL ~9,718 players.

**Why Fantrax API (not CSV)?**
- Ages for ALL players including free agents
- Always current (no manual export)
- Single source of truth for roster and age data

---

## Function Specifications

```python
def compute_dynasty_sgp(
    row: pd.Series,
    projection_years: int = DYNASTY_PROJECTION_YEARS,
    discount_rate: float = DYNASTY_DISCOUNT_RATE,
) -> float:
    """
    Compute dynasty-adjusted SGP for a single player.
    
    Args:
        row: Player row with SGP, age, player_type, Position
    
    Returns:
        Dynasty SGP (NPV of future production).
        If age missing, returns current SGP.
    
    Implementation:
        1. Select aging factors: HITTER/SP/RP based on player_type + Position
        2. For each year t (0 to projection_years-1):
           - future_age = age + t
           - Accumulate aging decline
           - projected_sgp = max(0, SGP + cumulative_decline)
           - discount = 1 / (1 + discount_rate)^t
           - Add projected_sgp × discount
        3. Return dynasty_sgp
    
    Edge cases:
        - age > 40: use age 40 factors
        - age < 22: use age 22 factors (no improvement)
        - age missing: return SGP only
    """


def add_dynasty_sgp_to_projections(
    projections: pd.DataFrame,
    age_lookup: dict[str, int],
) -> pd.DataFrame:
    """
    Add age and dynasty_SGP columns to projections.
    
    Returns DataFrame with: age, dynasty_SGP columns added.
    Players without age get dynasty_SGP = SGP.
    
    Print:
        "Added dynasty values: {N} with age, {M} without"
        "Dynasty SGP range: {min:.1f} to {max:.1f}"
    """
```

---

## Trade Engine Integration

The trade engine uses `dynasty_SGP` for fairness:

- **Trade fairness:** SGP differential within 10% of total SGP involved
- **Acquirability:** `delta_V_acquire / (dynasty_SGP + 0.5)`
- **Expendability:** `-(dynasty_SGP + lose_cost × scale)`

Both `SGP` and `dynasty_SGP` remain in DataFrame for analysis.

---

## Validation Checklist

```python
# Verify implementation:
assert df['dynasty_SGP'].min() >= 0, "Dynasty SGP must be non-negative"
assert df[df['age'] < 27]['dynasty_SGP'].mean() > df[df['age'] < 27]['SGP'].mean() * 0.9, \
    "Young players should have dynasty_SGP close to SGP"
assert df[df['age'] > 34]['dynasty_SGP'].mean() < df[df['age'] > 34]['SGP'].mean() * 2.5, \
    "Old players should have lower dynasty_SGP relative to SGP"
```
