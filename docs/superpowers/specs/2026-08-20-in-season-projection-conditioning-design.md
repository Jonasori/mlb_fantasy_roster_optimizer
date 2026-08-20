# Conditioning Projections on In-Season Evidence

**Status:** Design, approved for Parts 0–1–2a. Part 2b gated on Part 0 results.
**Date:** 2026-08-20
**Supersedes:** `design_descriptions/TRUE_TALENT_RESIDUALS.md` (its method survives; its central
empirical claim §2.6 is falsified — see §2.3 below).
**Depends on:** `AGENTS.md` (style), `design_descriptions/IMPLEMENTATION_SPEC.md` (silver/gold
column contracts), `design_descriptions/MATHEMATICAL_FRAMEWORK.md` (MEW, gradient, EW).

---

## 1. Problem and framing

### 1.1 The stated complaint

ATC's rest-of-season (RoS) projections lean too heavily on prior-season talent and are slow to
absorb the current season. The motivating example: Cal Raleigh is having a far worse 2026 than
2025, yet still projects well.

### 1.2 The framing that dissolves the tension

The naive formulation — "blend ATC with in-season stats, shifting weight toward in-season as the
year progresses" — has three defects:

1. **The weight is unidentifiable.** There is no ground truth to fit `w` against, so it becomes a
   knob tuned until a favoured player looks right. That encodes the analyst's prior as a
   hyperparameter.
2. **It double-counts, partially.** ATC's RoS feed does absorb some in-season information (§2.3).
   A blend adds it a second time, by an unknown amount.
3. **The stated evidence test is circular.** "Are a player's last N games predictive of his next
   M?" is *always* yes — both correlate with true talent. The only decision-relevant quantity is
   the **partial** predictiveness conditional on the projection.

Reformulate. We are not blending; we are **modelling ATC's residual**:

```
RoS_actual[c] = ATC[c] · λ[c](x) + ε
```

where `x` is decomposed in-season evidence and `λ` is fitted. Three properties follow:

- The blend weight becomes a **fitted coefficient**, not a knob. If in-season evidence adds nothing
  over ATC, `λ → 1` and the model says so.
- The early-vs-late-season schedule is **fit, not imposed**: put sample size in `x`.
- Success is **one falsifiable number** — out-of-sample RoS error against the ATC baseline.

### 1.3 The second reframe: measure error where it changes decisions

Projection accuracy is not the objective. MEW is, and MEW enters decisions only through
*differences between players*. A projection error uniform across a category is invisible to the
optimizer. Therefore the target is not "be accurate" but **"be accurate enough to flip a lineup
slot or a trade."**

That quantity is computable exactly:

```
ΔMEW = Σ_c  g[c] · ∂(team total[c]) / ∂(projection error)
```

Both terms are already produced every run. **We do not decide a priori which projections to
improve — we ask the gradient, and the answer changes weekly.**

### 1.4 Structural leverage

Decomposing `∂(team total)/∂(player projection)` gives a fixed structural factor:

| Stat class | Leverage | Reason |
|---|---|---|
| Counting (R, HR, RBI, SB, W, SV, K) | **1.0** | The player *is* his own total |
| Ratio (OPS, ERA, WHIP) | `player_volume / team_volume` | ≈ **1/18** for a hitter's OPS, ≈ **1/7** for a starter's WHIP |
| **Volume (PA, IP)** | **multiplier on all of the above** | Moves every counting category *and* re-weights the ratio |

A ratio stat needs a gradient ~18× larger than a counting stat to matter equally on the hitting
side. Sometimes it has one (§2.2).

**The load-bearing conclusion:** volume is the only **gradient-invariant** input. Its value does
not depend on which races are live. Rate corrections' value is contingent on the gradient and must
be re-justified each season. This is why Part 2a is unconditional and Part 2b is gated.

---

## 2. Measured findings

All numbers below were measured against repo data on 2026-08-20 and are reproducible.

### 2.1 No per-player in-season data enters valuation today

- `data/players.parquet` stat columns are **pure ATC RoS**. Nothing divides a full-season number.
- Season-to-date enters **only as team totals**, via `optimizer/banked.py` reading the Fantrax
  standings, blended in `optimizer/lineup_solver.py::blend_season_totals`. 79% of every category
  total is realized history.
- Per-player YTD exists in exactly one place — `notebook.py::game_logs_cell` — and is consumed into
  a **display-only** copy. It never touches FV, MEW, or EW.

### 2.2 Sensitivity: ΔMEW per 10% projection error

Measured across all 18 starters, cross-checked against exact ΔEW (linearization accurate to <1% for
hitters, ~4% worst case for pitcher WHIP).

| Input | median ΔMEW | vs tightest lineup gap (0.023) |
|---|---|---|
| Pitcher WHIP | 0.115 | 5.0× |
| Pitcher volume (IP) | 0.097 | 4.2× |
| Pitcher W | 0.080 | 3.5× |
| **Hitter volume (PA)** | **0.077** | 3.3× |
| Pitcher ERA | 0.060 | 2.6× |
| Hitter R | 0.039 | 1.7× |
| Hitter SB | 0.038 | 1.7× |
| Pitcher K | 0.025 | 1.1× |
| Hitter HR | 0.010 | 0.4× |
| Hitter RBI | 0.0016 | 0.07× |
| **Hitter OPS** | **0.0006** | **0.03×** |

**Decision thresholds:** median lineup-flip gap ≈ 0.10 MEW; minimum ≈ **0.023 MEW** (the OF/UTIL
pool is four players deep within 0.05). Adjacent starters differ by as little as 0.0009.

This ranking is **structural, not seasonal** — sensitivities scale linearly with
`season_fraction_remaining` and the ratios are preserved exactly (OPS/PA = 0.0061 at f=1.0 vs
0.0065 at f=0.205).

**But the gradient is not structural.** Each category's `g` as a fraction of its own ceiling (all
six opponents dead even):

| Category | g actual | g if tied | % of max |
|---|---|---|---|
| OPS | 0.392 | **311.2** | **0.13%** |
| RBI | 0.001 | 0.08 | 1.3% |
| K | 0.007 | 0.06 | 10.9% |
| HR | 0.020 | 0.17 | 11.4% |
| SV | 0.061 | 0.35 | 17.5% |
| R | 0.023 | 0.08 | 30.2% |
| ERA | 4.59 | 12.45 | 36.8% |
| WHIP | 28.6 | 74.7 | 38.3% |
| SB | 0.077 | 0.15 | 51.0% |
| W | 0.268 | 0.48 | 56.3% |

**Hitter OPS is irrelevant in 2026 only because that race is decided.** In a live OPS race
`g_OPS = 311` — 4× today's `g_WHIP` — making it the most important input in the model. Any design
that hardcodes "skip OPS" is learning the wrong lesson.

### 2.3 ATC updates volume aggressively and rate almost not at all

Over 68 days (2026-06-13 → 2026-08-20), hitters projected ≥100 PA:

| Component | Revision | Read |
|---|---|---|
| Rate (OPS) | median −0.006, SD 0.026; n=254 gives median \|Δ\| = 0.0100 | near-frozen for regulars |
| Volume (PA, net of clock decay) | **41.6% of players revised >20%**, IQR 0.87–1.20× | genuinely live |

Single-day cadence: PA changes for ~90% of players per pull; OPS for ~5%.

**Falsifies `TRUE_TALENT_RESIDUALS.md` §2.6.** That document concluded our feeds were
preseason-frozen, measured across a 13-day June window where rates happened not to refresh. The
pipeline now pulls FanGraphs' RoS endpoints (`steamerr`, `ratcdc` — `data_prep/scrape_fangraphs.py:41`),
and rates *do* move; they simply move slowly.

**Switching systems will not fix this.** Steamer's RoS rates revise just as little as ATC's over
the same window (median \|Δ\| 0.0089 vs 0.0100). Published FanGraphs documentation lists ATC as the
only system whose in-season rates are "updated as needed" rather than daily, so the literature
predicts a cadence gap — **we measured for it and it is not there.** Both systems are deliberately
stable, which the MGL result (§2.5) says is correct behaviour.

### 2.4 The exploitable bias: ATC forgives fast-stabilizing skills

Regressing `gap = ATC_RoS_OPS − YTD_OPS` on decomposed YTD components (hitters ≥250 PA, n=266,
R²=0.60):

| Component | coef | t | Should ATC forgive it? |
|---|---|---|---|
| BABIP | −0.97 | −3.02 | **Yes** — M≈433, mostly luck |
| K% | +0.37 | **+2.51** | **No** — M≈49, ~89% real at 400 PA |
| ISO | −0.61 | −1.80 | **No** — M≈69, ~85% real |
| BB% | +0.08 | 0.47 | — |
| YTD OPS | −0.02 | −0.10 | components fully span it |

Pool-wide ATC is **unbiased** (mean gap −0.006; above YTD for only 42% of players). The bias is not
in the level — it is in **which components get regressed**. ATC forgives a high strikeout rate
almost as readily as it forgives a low BABIP, though one is luck and the other is ~90% real.

**This describes ATC's behaviour; it does not prove ATC is wrong.** The stabilization literature
says it shouldn't do this. Confirming requires held-out outcomes — which is Part 0's entire job.

### 2.5 Case diagnosis

| | 2025 | 2026 | reliability at 2026 n | verdict |
|---|---|---|---|---|
| Raleigh K% | 26.7% | **31.9%** | 89% | real |
| Raleigh ISO | .342 | **.140** | ~85% | **real, huge** |
| Raleigh BABIP | .248 | .196 | 48% | half luck |
| Raleigh OPS | .948 | **.569** | — | ATC RoS: **.769** |
| Duran K% | 24.3% | 28.4% | 91% | real |
| Duran ISO | .186 | .153 | ~87% | real, small |
| Duran BABIP | .326 | **.253** | 54% | **mostly luck** |
| Duran OPS | .774 | .622 | — | ATC RoS: .731 |
| Durbin OPS | .721 | **.727** | — | ATC RoS: .708 |

- **Raleigh — the complaint is justified.** Power genuinely vanished (ISO halved, strikeouts +5.2pp,
  both in ~85–90%-reliable samples). ATC at .769, 200 points above his season, is hard to defend.
- **Duran — mostly not.** The decline is dominated by a 73-point BABIP drop that at 504 PA is
  roughly half noise. Regressing it is correct.
- **Durbin — no step change at season resolution** (.727 vs .721). The within-season-trajectory
  claim requires game logs, not season totals, and is unproven.

Raleigh's .569 and Duran's .622 look like the same story as OPS and decompose completely
differently. **This is the case for never regressing the composite.**

### 2.6 The sting

Raleigh's OPS error is ~26%. Correcting it perfectly moves his MEW by **0.0016** — 14× below the
0.023 lineup-flip threshold. **Diagnosing him correctly changes nothing you do this week.**

The channel that does pay is volume: MGL found cold hitters lose ~30 PA over five months relative to
equally-projected hot hitters, because managers bench slumping players even when the projection says
not to. The rate diagnosis is the **input**; playing time is the **channel**, and it carries 130× the
MEW leverage.

### 2.7 Literature summary (evidence grades in the source review)

- **Against an already-updated projection**, in-season rate signal is small and asymmetric: +5 to
  +7 points of wOBA for hot hitters, ~0 for cold, ~0 for pitchers (MGL 2007–2013). Implied
  `M ≈ 1,500–4,000 PA`. Pre-Statcast; against ATC's slower rates the residual should be larger.
- **Volume is where the error is.** ATC's preseason PA RMSE is **156** against **162 for assigning
  every hitter a flat 510 PA**. Systematic ~10% over-projection with three correctable drivers:
  prior-2-year playing time, age, projected talent. Zimmerman's correction cuts ensemble RMSE
  153.2 → 146.3 at 35–49% blend weight. Pitcher IP RMSE ≈ 49% of the mean; SV RMSE ≈ 68%.
- **Never regress composites.** M ranges from 49 (K%) to 433 (BABIP) for hitters, 93 (K%) to 1,409
  (BABIP) for pitchers. One constant applied to OPS is wrong for every component.
- **Regressing toward a projection is not regressing toward the league mean.**
  `M_vs_ATC ≈ M_league / (1 − R²_ATC)`, i.e. **2–15× the published constants.** Using Carleton's
  numbers directly against ATC would over-correct wildly. **This is the single biggest technical
  trap in the design.**
- **Shrinkage should be performance-dependent, not only n-dependent.** Binomial Fisher information
  is `n/(p̂(1−p̂))`, so a bad line in n PA is literally more informative than a good line in the same
  n. Shrink extreme-low hitter lines slightly less than extreme-high ones. (For pitchers the
  Poisson form reverses this.)
- **Within-season changepoints are real but only in physical inputs** — velocity, chase rate, whiff
  rate (Glazer 2025, 91% detection against a ground-truth set). Not in wOBA or ERA: the
  signal-to-noise in outcome stats is below the detection floor at in-season samples. Unreplicated,
  and never shown to improve a RoS forecast.
- **Hot/cold streaks do not survive to a RoS horizon.** Green & Zwiebel find real 25-PA effects;
  MGL's month-long lookbacks over multi-month horizons wash them out. A 25-PA hot streak is a DFS
  signal, not a rest-of-season one.
- **No published work regresses a commercial projection's RoS residuals on season-to-date
  components.** Part 0 is not reproducing prior art.

---

## 3. Architecture

Three parts. Part 0 gates everything; Part 2b is gated on Part 0's results.

### 3.1 Part 0 — Backtest harness

Not a feature. Without it, every constant becomes a knob tuned until Raleigh looks right — the
failure mode §1.2 exists to prevent.

**Primary test (calibration of `M_vs_ATC`):**

| | |
|---|---|
| Split date | 2026-06-11 |
| Baseline | ATC RoS snapshot `data_prep/data/pulled_20260611/` — a genuine dated held-out projection |
| Evidence | 2026 stats through 06-11, from MLB StatsAPI game logs, decomposed to skills |
| Target | actual 06-11 → 08-20 performance (~70 days, ~500 hitters, ~600 pitchers) |

Snapshots at 06-12/13/14/18/23/26 exist but are 15 days apart on the same players. **Report them,
but treat the whole set as approximately one independent window, not seven.** Any claim of
significance must be justified against n≈1 window, not n=7.

**Secondary test (estimation of per-skill `M_league`):** does not require projection archives at
all. For any past season and any split date, ask "given stats through D, how well does each
component predict D→end?" This yields the shrinkage constants from many seasons of data. The scarce
projection archive is then needed only to calibrate the **single** ratio `M_vs_ATC / M_league` per
stat — which one window can support.

This decomposition is what makes a thin projection archive sufficient.

**Metric — the part that matters.** Report error in **both** stat units and **MEW units**. Decide on
the MEW column. A method that cuts OPS RMSE 20% but moves no decision has earned nothing.

**Mandatory baselines.** Every candidate must beat all of:
1. Unadjusted ATC (the thing we are trying to improve)
2. Raw YTD, unshrunk (Brown 2008: for batting average this is *worse than the league grand mean*)
3. A flat constant for volume (ATC's PA RMSE is only 4% better than one)

### 3.2 Part 1 — In-season evidence layer

New upstream source following the existing raw-snapshot pattern
(`data/raw/ytd/YYYY-MM-DD.parquet`, via `data_prep/raw_io.py::write_raw`), registered in
`data_prep/cli.py::SOURCES` and fetched with `uv run fetch ytd`.

Per-player, keyed on `MLBAMID`, **decomposed to skills, never composites**:

| Hitters | Pitchers |
|---|---|
| PA, K%, BB%, ISO, BABIP, SB-attempt rate | BF, IP, K%, BB%, GB%, HR/FB, BABIP-against |

Source: MLB StatsAPI `people` endpoint, `hydrate=stats(group=[hitting,pitching],type=[season])`,
batched 100 ids per request — the pattern already proven in `notebook.py::game_logs_cell`.
**Measured cost: 7 seconds for two full seasons of 1,325 players.** Game logs (`type=[gameLog]`)
use the same call shape and are needed for Part 0's date splits.

This part is independently useful: the dashboard's YTD table is currently display-only and
disconnected from valuation.

### 3.3 Part 2a — Volume correction (unconditional)

One function at the single narrowest seam — `data_prep/build.py:717`, the return of
`build_players`, immediately before `add_fantasy_value`:

```python
def adjust_projection_volume(players: pd.DataFrame, ytd: pd.DataFrame) -> pd.DataFrame:
    """Correct RoS PA/IP using in-season evidence.

    Requires columns: PA, AB, IP, R, HR, RBI, SB, W, SV, K, MLBAMID, age
    Adds columns: PA_adj_factor, IP_adj_factor, volume_adj_reason
    Rewrites columns: PA, AB, IP, R, HR, RBI, SB, W, SV, K
    """
```

Model, from the published and validated Zimmerman correction plus one addition:

1. prior-2-year playing time (best available health proxy; IL data added nothing beyond it)
2. age (older players chronically over-projected)
3. projected talent (weak hitters get implausibly generous PT)
4. **slump-benching term** keyed to YTD performance — MGL's ~30-PA effect, and the channel through
   which the Raleigh diagnosis actually reaches a decision

Blend weight against ATC fit by Part 0, expected in the 35–49% range the literature reports.

**Invariants (verified; none are enforced by existing code):**

- Nothing anywhere checks OPS against PA, or ERA/WHIP against IP. Rates and volumes float free.
- **Scaling PA does not scale R/HR/RBI/SB.** They must be scaled explicitly or the result is a
  player who plays 10% more games and scores the same runs, with no error raised.
- Scale `AB` in lockstep with `PA`.
- Opposite-type columns must stay exactly `0.0` (`build.py:236`) or MEW picks up a phantom ratio term.
- Never scale a player to zero volume: it silently drops him from FV's ratio z-population
  (`player_scoring.py:71`, the `PA > 0` / `IP > 0` rate gates) and permanently benches IL players
  via the zero-`projected_volume` early return in `players.py::get_startable_slots`.
- Team-level `PA > 0` and `IP > 0` asserts fire in `lineup_solver.py` and `player_scoring.py`.
- **Non-local side effects, measured:** σ is computed from the RoS league mean *including our team*,
  and the gradient is recomputed from post-adjustment totals. A team-wide +10% hitter volume bump
  moved `g_HR` +74%, `g_OPS` −33%, `g_RBI` 10×, and EW 30.52 → 31.59. Adjustments are **not** local.
- RoS counting stats arrive **integer-rounded** (HR ranges 1–9 across current starters). A ±0.5
  rounding on a 3-HR projection is already ±17%. Any multiplicative adjustment sits on top of that
  quantization; corrections below it are theatre.

### 3.4 Part 2b — Rate correction (GATED on Part 0)

Design recorded now; **not built until Part 0 shows it beats unadjusted ATC in MEW units.**

Component-wise Bayesian shrinkage, per-skill `M`, regressing toward ATC rather than the league mean:

```
reliability[s] = n / (n + M_vs_ATC[s])
talent[s]      = reliability[s] · observed[s] + (1 − reliability[s]) · atc_implied[s]
```

then recompose to category rates and apply to remaining volume.

Requirements carried from §2.7:
- `M_vs_ATC`, not `M_league`. Fit it; do not assume it.
- Asymmetric — the hot-side and cold-side coefficients differ, and part of the published asymmetry
  is a playing-time survivorship artifact that Part 2a should absorb instead.
- Performance-dependent, not only n-dependent.
- Never on composites.

**Expected value, stated honestly.** Hitter rate correction has near-zero decision value in the
2026 league (§2.6). It has real value for **pitchers now** — `g_WHIP` and `g_ERA` at 38%/37% of
ceiling, leverage ~1/7 rather than 1/18, and a 10% WHIP error worth 0.115 MEW against a 0.023
threshold — and for **hitters in any season where the OPS race is live**, where `g_OPS = 311` would
make it the most important input in the model.

Caveat specific to pitchers: WHIP decomposes into BB% (M≈237 BF) and BABIP-against (M≈1,409 BIP,
very slow). In-season WHIP is therefore heavily luck-driven and must be regressed hard. The gain
comes from *discriminating* — trusting a low WHIP built on low BB%, regressing one built on low
BABIP — not from following WHIP itself.

### 3.5 Explicitly out of scope

- **Changepoint detection, Statcast, bat tracking.** Right third move, not the first: unreplicated,
  and never demonstrated to improve a RoS forecast (§2.7).
- **A greenfield projection system.** Considered and rejected on evidence: ATC is pool-wide
  unbiased with R²=0.60 explicable residual structure. We are correcting a good estimator, not
  replacing it.
- **Switching projection systems.** Measured and rejected (§2.3).
- **Trade-value / market-value changes.** Separate concern.

---

## 4. Testing

Per `AGENTS.md`: module-level functions only, no classes, no try/except, descriptive asserts,
`players = players.copy()` before adding columns.

- Part 0 is itself the validation apparatus for Parts 2a/2b.
- Unit tests for the shrinkage arithmetic: reliability → 0 at n=0 (estimate equals the prior),
  → 1 as n→∞ (estimate equals the observation), monotone between.
- Invariant tests for `adjust_projection_volume`: counting stats scale with PA; AB tracks PA;
  opposite-type columns stay exactly 0.0; no player reaches zero volume.
- A regression test pinning current EW so the non-local σ/gradient side effects of §3.3 surface as
  a visible diff rather than a silent drift.

---

## 5. Open questions

1. **Is one held-out window enough to calibrate `M_vs_ATC`?** The §3.1 decomposition is designed so
   that it only has to support a single ratio per stat, but this should be checked with a bootstrap
   before any fitted constant is trusted.
2. **Start snapshotting now.** Whatever we conclude this season, daily RoS snapshots make a properly
   fitted model possible next season. Cheap, and the archive we do have exists only because
   snapshots happened to be kept.
3. **Pitcher volume** — IP RMSE ≈ 49% of the mean and SV RMSE ≈ 68% are far worse than hitter PA.
   Part 2a's model is specified from hitter-based literature; the pitcher analogue (rotation slot,
   closer role churn, IL) may need a different feature set. Closer role in particular is a discrete
   observable, not a regression target.
4. **Where does the Fantrax live injury/roster status fit?** `injury_status`, `injury_detail`, and
   `roster_status` are already on `players` and are fresher than anything ATC sees. They are a
   natural Part 2a feature and are currently unused by valuation.
5. **Should the gradient-weighted effort ranking be surfaced in the dashboard?** §1.3's
   `g[c] × leverage[c]` is a few lines given quantities already computed, and it is what tells you
   when Part 2b matters. Possibly YAGNI; possibly the most useful diagnostic in the system.
