# True-Talent Residuals: Conditioning Projections on In-Season Evidence

**Status:** Design proposal (skeleton). Nothing here is implemented yet.
**Relates to:** `design_descriptions/IMPLEMENTATION_SPEC.md` (silver/gold contracts), `AGENTS.md` (coding standards).
**Scope:** How to fold actual-vs-projected residuals into roster decisions in a statistically disciplined way — and the open design choices to resolve before building.

---

## 1. Problem statement

Preseason projections (Steamer/ATC) are a prior formed before a single 2026 pitch was thrown. As the season unfolds, two very different things move a player's line away from that prior:

1. **Noise** — variance around the same underlying true talent (a hot BABIP month, a lucky cluster of RBI chances).
2. **A genuine true-talent change the projection has not yet absorbed** — a pitcher added a new pitch over the offseason, a hitter rebuilt his swing, a velocity gain, a new role.

The naive table ("actual YTD vs. projection") shows the *sum* of these two effects. **Only the second is decision-relevant.** The entire goal of this work is to estimate the part of a residual that reflects a real talent change not reflected in the underlying projection, and to feed *that* — not the raw residual — into roster value.

### Reframing (the important one)

Rather than asking "how far is the player from projection?", ask:

> **What is the player's current true talent, and how much does it differ from what the projection assumed?**

Estimate true talent from the most predictive, fastest-stabilizing evidence available, then **infer the stat bumps off that true-talent delta** and apply them to the rest-of-season (ROS) projection. The residual becomes an *output* of a true-talent model, not the input to a heuristic.

---

## 2. The five disciplines (report)

### 2.1 Bayesian shrinkage is the backbone

Treat the projection as a prior expressed in units of playing time, and YTD as data:

```
reliability  = n_actual / (n_actual + n_stabilize)
blended_rate = reliability · observed_rate + (1 − reliability) · projected_rate
```

`n_stabilize` is the sample at which split-half reliability ≈ 0.5 for that stat. This single mechanism encodes "don't overreact to small samples" — early in the year `reliability` is near zero and the estimate stays close to the prior.

### 2.2 Stat-specific stabilization is the heart of the discipline

Different stats become trustworthy at very different samples. Approximate published points (Carleton et al.):

| Fast (trust early) | Medium | Slow (mostly noise mid-season) |
|---|---|---|
| Hitter K% ~60 PA, swing/contact ~50 PA | BB% ~120 PA, ISO / HR-rate ~160 PA, SLG ~320 PA | OBP ~460 PA, AVG ~910 PA, **BABIP ~820 PA** |
| Pitcher K% ~70 BF, GB% ~70 BF | BB% ~170 BF | **ERA barely stabilizes all season** |

Consequence: a 280-PA OPS jump driven by **SLG/ISO** is far more believable than one driven by **BABIP**. A pitcher's ERA residual should lean on **FIP/xFIP/SIERA**, not raw ERA.

### 2.3 Prefer predictive inputs over outcome inputs

Statcast expected stats (xwOBA, xSLG, xBA, barrel%, hard-hit%, exit velocity; for pitchers: velocity, whiff%, pitch-level stuff) stabilize faster and predict the future better than outcome stats. "Is the breakout real?" is best answered by "did barrel%/EV/velo move and does xSLG support the line?" — exactly the signature of a swing change or a new pitch.

### 2.4 Down-weight context-dependent residuals

R, RBI, W, SV are heavily lineup/role/team driven; a residual there is often **not** the player's skill. Treat them as low talent-signal, and handle the legitimate cases (closer gained/lost the job, batting-order move, post-trade park) as **explicit structural adjustments**, entered deliberately — not inferred from noise.

### 2.5 Guardrails against fooling yourself

- **Winner's curse:** the biggest positive residuals are where regression bites hardest. Be *more* skeptical of large overperformers.
- **No double-counting:** once the projection is blended/updated, do not *also* manually bump for "hot."
- **Projection vintage (RESOLVED — see §2.6):** our pulled CSVs are **frozen preseason** projections. There is no double-counting risk: 100% of every residual is currently unaccounted-for by the projection.

### 2.6 Empirical finding: our projections are frozen at preseason

Checked directly (web + our own pulled data, June 2026):

- **FanGraphs behavior:** Steamer is updated *daily* in-season; ATC's in-season form (ATC DC) updates playing time daily, performance "as needed." So updated feeds exist.
- **What we actually pull:** `data_prep/scrape_fangraphs.py` requests `type=steamer` and `type=atc` (no in-season flag). Tracing Cal Raleigh, Wilyer Abreu, and Jo Adell across all 21 pulls (2026-01-29 → 06-11): values are **identical to many decimals from ~opening day (03-25) onward**. A full-file diff between every pair of in-season dates shows **0 of ~270 regulars changed** for both systems.
- **Conclusion:** `type=steamer` / `type=atc` return the **preseason-frozen full-season projection**. They are NOT FanGraphs' daily in-season feed. (Abreu's projected OPS sits at .788 all year while he's running ~.90 — none of that is absorbed.)

Two consequences:

1. The shrinkage method is not just valid but *necessary* — the prior is stale by construction, and blending cannot double-count.
2. We have a free improvement available: pull FanGraphs' **updated** feeds (likely `type=steameru` "Steamer Update" / `type=steamerr` "Steamer RoS", and ATC DC — exact params to verify against the API). These give either (a) a better, already-updated prior, or (b) a benchmark to test "do Steamer/ATC update *well enough*" before we build our own. The user's original doubt ("do they do a good enough job?") is currently **untestable from our data** because we never captured the updated version.

---

## 3. Proposed approach: true-talent estimate → inferred stat bumps

A two-layer model keeps the statistics honest and the integration clean.

### Layer A — Estimate current true talent per skill

For each **skill metric** (not each fantasy category), shrink observed toward prior using that metric's own stabilization constant, preferring the predictive version where available:

```
talent[skill] = shrink(observed[skill], prior[skill], n_actual, n_stabilize[skill])
```

Skills (candidate set):
- Hitters: K%, BB%, ISO (or barrel%/xISO), BABIP-skill (via xwOBAcon/contact quality), speed (sprint/SB-rate).
- Pitchers: K%, BB%, GB%, HR/FB-skill (or barrel-against), stuff/velocity.

The prior for each skill comes from the projection (decomposed into rates). The observed comes from YTD, ideally the expected-stat variant.

### Layer B — Infer ROS stat bumps from the talent delta

The fantasy categories are *functions* of skills. Convert the talent estimate back into category rates, take the delta vs. the projection's implied rates, and apply to remaining playing time:

```
delta_rate[cat]   = f(talent_skills) − f(prior_skills)
ros_projection[cat] = prior_rate[cat] · remaining_PA  +  delta_rate[cat] · remaining_PA
```

This is where "a new pitch" or "a swing change" enters: it shows up as a moved skill (whiff%, velo, barrel%), which propagates through `f(...)` into K, ERA, OPS, etc. — instead of being asserted by hand.

> **Open modeling choice:** `f(skills → categories)` can be (a) simple analytic mappings (e.g., K = K% · BF), (b) a fitted historical regression of category rates on skill rates, or (c) skip Layer B and shrink each category rate directly (simpler, but loses the "infer bumps from talent" property the user wants). Recommend starting with (a)+(c) hybrid and adding (b) only if it earns its keep.

---

## 4. Skeleton design doc

### 4.1 Inputs (new, upstream — not in current silver contract)

Per-player YTD evidence, joined on `MLBAMID`. Following `IMPLEMENTATION_SPEC.md §1`, building this is upstream of the math pipeline.

| Field | Type | Notes |
|---|---|---|
| pa_ytd / bf_ytd / ip_ytd | float | Sample size for reliability. |
| observed skill rates | float | K%, BB%, ISO, GB%, etc. (outcome). |
| expected skill rates | float \| None | xwOBA, xSLG, barrel%, EV, velo, whiff% (predictive). None when unavailable. |
| structural_flags | dict \| None | Explicit role/park/lineup changes (manual or rules-driven). |

### 4.2 Constants (module-level, true constants per `AGENTS.md`)

```python
# Stabilization points (PA for hitters, BF for pitchers). Source: cite.
STABILIZE_PA = {"K_pct": 60, "BB_pct": 120, "ISO": 160, "SLG": 320, "OBP": 460, "BABIP": 820}
STABILIZE_BF = {"K_pct": 70, "BB_pct": 170, "GB_pct": 70}
CONTEXT_STATS = {"R", "RBI", "W", "SV"}   # low talent-signal; down-weight
```

### 4.3 Functions (no classes; results land on `players`)

```python
def add_skill_priors(players: pd.DataFrame) -> pd.DataFrame:
    """Decompose projected category totals into per-PA/BF skill rates. Adds *_prior columns."""

def add_ytd_skills(players: pd.DataFrame, ytd: pd.DataFrame) -> pd.DataFrame:
    """Merge YTD observed + expected skill rates and sample sizes. Adds *_ytd, *_x, n_actual."""

def add_true_talent(players: pd.DataFrame) -> pd.DataFrame:
    """Layer A: reliability-weighted shrink toward prior, per skill. Adds *_talent, reliability."""

def add_ros_projection(players: pd.DataFrame, remaining: pd.DataFrame) -> pd.DataFrame:
    """Layer B: infer category bumps from talent delta; extrapolate to remaining PA/IP.
    Adds R_ros, HR_ros, ..., ERA_ros, WHIP_ros (mirror of the projection columns)."""

def add_talent_delta(players: pd.DataFrame) -> pd.DataFrame:
    """At-a-glance signal: ros_rate − prior_rate per category, plus a confidence badge
    from reliability. Adds talent_delta, talent_confidence."""
```

All follow the column-on-`players` contract in `AGENTS.md`; copy before mutating; descriptive asserts; no try/except.

### 4.4 Integration with the optimizer

The clean path: **once `*_ros` columns exist, point the value pipeline (FV/MEW/slotting/trades) at the ROS projections instead of the stale preseason ones.** Residuals then flow through value automatically — no bolted-on, decision-time heuristics. This keeps the "one core `players` table" architecture intact.

### 4.5 Display

- Reuse the YTD table's color scale, but drive it from `talent_delta` (the *believed* edge) rather than the raw residual.
- Add a `talent_confidence` badge so a large delta on a thin sample is visibly discounted.
- A sortable "biggest believable risers/fallers" view, ranked by `talent_delta` and filtered by confidence.

---

## 5. Open questions (resolve before implementation)

1. **Projection vintage** — RESOLVED (§2.6): pulled feeds are preseason-frozen. Follow-up action: add the updated/RoS FanGraphs feeds (verify exact API `type` params) to serve as a better prior and/or a benchmark for "are they good enough."
2. **Expected-stat source** — where do xStats/Statcast come from, and what's the coverage/latency? Layer A degrades to outcome-only when missing.
3. **`f(skills → categories)`** — analytic vs. fitted; how much complexity is justified.
4. **Remaining playing time** — naive (proj − YTD) vs. a real playing-time model (injuries, demotions).
5. **Structural changes** — manual entry vs. rule-derived (role/park/lineup), and how to keep them from double-counting with the statistical signal.
6. **Validation** — backtest the blended ROS rate against actual ROS outcomes vs. (a) raw projection and (b) raw YTD, to prove the method beats both naive baselines.
