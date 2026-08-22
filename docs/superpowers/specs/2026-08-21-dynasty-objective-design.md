# Dynasty Objective Function — Design

Status: design. No implementation. Supersedes `data_prep/ceiling.py`'s scoring layer.

## 0. Invariant

```
ONE UNIT: Δ P(win the league).
Nothing may be summed that is not in that unit.
```

Deleted by this design: `ceiling_score`, `tier1_FV`, `tool_z`, `now_value`,
`now_vs_replacement`, `replacement_now`, `screen_*`, `horizon`.
Retained from `ceiling.py`: the Savant fetchers, `pct_*` tool percentiles,
`profile_flag`, `add_eligibility`. These become inputs to §4, not scores.

`FV` (z-score sum, `optimizer/player_scoring.py`) is retained ONLY for its
existing job: solving opponent lineups (W4). It is not a valuation.

## 1. The score

```
V(p; β) = Σ  β^t · u_t(p)          summed while β^t·|u_t| > ε
          t≥0

u_t(p) = E[m_t(p)]  +  Λ_t(p) · Var_t(p)

m_t(p) ≜ p's marginal Δ P(win) in season t, scored against G^(t)   [§2]
β      ≜ THE KNOB. Impatience, 0 < β ≤ 1. β→0 compete-now, β→1 rebuild.
ε      ≜ convergence tolerance. Numerical, not a modeling choice.
```

`m_t` is already in `Δ P(win)` units (§2.3), so there is no separate leverage
scalar multiplying it. Season-`t` leverage lives inside `G^(t)`.

There is no horizon parameter. `survival` (§3) decays geometrically and §5
truncates each branch at `H_k`, so the sum converges even at β = 1.

Opportunity cost is NOT inside `u_t`. See §6.

## 2. The championship gradient

### 2.1 Why there is no scalar `dP/dEW`

`P(win)` is a function of the 10-vector of category totals, not of the scalar
`EW`. Two rosters with identical `EW` and different category profiles have
different `P(win)` — punting saves and spreading thin are not interchangeable.
So `dP/dEW` is undefined until a perturbation direction is fixed, and a player is
not a uniform perturbation: he adds HR and R and no SV.

Drop the `EW` intermediary. Differentiate `P` against the totals directly.

```
G_c ≜ ∂P(win) / ∂my_c              the championship gradient, one per category
```

### 2.2 Computing G — finite differences on the existing simulator

```
P(c, ±δ_c) ≜ simulate_standings(my_totals with my_c ± δ_c,
                                opponent_totals, sigmas, seed=S)["p_win"][0]

G_c = ( P(c,+δ_c) − P(c,−δ_c) ) / (2 δ_c)          δ_c = k · σ_c,  k ≈ 0.25
```

Cost: `1 + 2·10 = 21` simulator calls per season. Negligible.

MANDATORY: identical `seed` on every call. Common random numbers. Without it each
`p_win` carries `se ≈ 0.0035` at `n_sims=20_000` and the differences are noise.

Sign handling: perturb `my_c` in RAW category units and let the simulator's own
`NEGATIVE_CATEGORIES` logic set the sign. `G_ERA` and `G_WHIP` come out negative.
Do not pre-negate.

Assert `sign(G_c) < 0` for `c ∈ NEGATIVE_CATEGORIES` and `> 0` otherwise. A
violation means `δ_c` is below the simulator's noise floor.

### 2.3 Scoring a player against G — zero new code

```
m_t(p) = Σ_c  G_c^(t) · stat_c(p)   + ratio terms
```

This is exactly `add_mew`'s formula. `add_mew(players, my_totals, gradient)`
accepts any gradient dict and is not tied to a roster, so pass `G` in place of
the EW gradient and the output is denominated in `Δ P(win)` directly. No new
scoring function.

### 2.4 Curvature

The second-order object is the Hessian `∂²P/∂my_c ∂my_c'` — 55 unique entries,
each a second finite difference. Too noisy and too expensive to compute per
player.

Reduce to a DIRECTIONAL second derivative along the player's own stat profile:

```
Λ(p) = ( P(+u_p) − 2·P(0) + P(−u_p) ) / (2 ‖u_p‖²)      u_p = p's stat vector
```

Three simulator calls per player, so compute it ONLY for screened candidates.
This mirrors the repo's existing screen-then-exact architecture (§4a, W13).

Shape, for interpretation only — assuming a fixed cutoff `c` and Gaussian gap:

```
Λ  ∝  −z·φ(z)         > 0 below cutoff, < 0 above, = 0 AT cutoff
```

`c` is in truth the max of six opponents' totals and is itself random, so this
closed form is documentation of shape, not a formula to implement. The shape that
matters: leverage is single-peaked AT the cutoff and falls off in BOTH
directions, so a runaway leader has low current-season leverage and should also
sell forward.

### 2.5 The reference team, and G in future seasons

`m_t` requires a reference team, because MEW is a property of (player, roster):
the ratio terms are `PA·(OPS − my_OPS)/total_PA`.

```
t = 0     G^(0)   at MY actual totals            (compute_league_state)
t ≥ 1     G^(t)   at the NOMINAL team            (§2.2 with the totals below)
```

```
nominal totals_t  =  my_totals + ψ^t · (my_totals − league_mean_totals)
                     ψ = 0.0   # A1: pure league-mean team every future season.
                               # Raise toward ~0.7 for mean reversion.
                               # ψ = 1 is persistence: you stay who you are.

σ_c,t             =  σ_c,0 · sqrt(1 + t·v)
                     v         # per-season roster-drift variance. CALIBRATE.
```

`σ` inflation is what makes future seasons matter less on their own, before β
acts: a wider `σ_c` flattens `Φ`, so every `G_c^(t)` shrinks with `t`. That is a
derived discount and it is NOT β.

Consequences to log, not to hide:
- `ψ = 0` is the most future-favorable choice, not the neutral one. A league-mean
  team sits at the peak of the leverage curve, so A1 tilts toward rebuild before
  β acts.
- At `ψ = 0` the nominal team is at the cutoff, so `Λ_t ≈ 0` for `t ≥ 1`. That is
  an artifact of `ψ = 0`, not a result. At this setting future-season variance is
  valued ONLY through §5's option.
- The nominal team's opponents are frozen at current-season totals. We do not age
  the other six rosters.

## 3. Forward stat lines

No multi-year projection systems. `zipsp1`/`zipsp2` are NOT used: they stop at
2028, publish null `SV`/`HLD` in every out-year, and regress prospects to a floor
(Walcott .631→.660→.675 OPS).

```
line_t(p) = line_0(p) ⊙ decay(age_p, t, category) · survival(age_p, t, type_p)
```

Two primitives, both hand-rolled from MLB StatsAPI, both auditable:

```
decay(age, t, category)     per-category multiplier, split by talent decile
survival(age, t, type)      P(still has meaningful playing time)
```

`decay` and `survival` are the ONLY forward-projection machinery. Deliverable is
the generating script, not a frozen table.

Required properties, from the measured tables:
- Split by talent decile. The ERA curve differs in shape between bad, good and
  great pitchers; a pooled curve is a different object from any player's curve.
- Correct the delta method per category, not globally. Strict-pairs understates
  decline by 26% (OPS), 35% (SP ERA), 100% (SP BB/9), and ~0% (SB, BB%).
- Do NOT age-grade pitcher survival. Starter one-year retention is flat at
  70–79% across ages 21–40. Apply a flat hazard; let the cumulative separate.
- Age playing time separately from rate. SB rate is near-flat ages 22–26 while
  PA attrition is steepest for speed profiles; a rate-only curve overvalues speed.
- `SV` is role persistence, not a skill curve. Reliever SV/65G peaks at 29–30 for
  role reasons. No system projects future `SV`; this is an assumption and must be
  labelled as one wherever it surfaces.
- Percentile inputs must be computed against the contemporaneous league. Raw
  cross-year fastball velocity understates individual decline by ~40% because the
  league mean rose 91.22 → 94.43 mph over the sample.

## 4. The distribution

One abstraction, two sources. Replaces the `horizon` string and its 4,116
`unclear` rows.

```
mixture(p) -> [ (prob_k, line_k, arrive_k) ]        Σ_k prob_k = 1
```

`Σ_k prob_k = 1` INCLUDES the never-arrives branch (an all-zero line). Assert it.
A mixture summing to less than 1 silently discounts every player by the missing
mass.

```
MLB-established → Steamer q10..q90 / tt_q10..tt_q90      MEASURED
                  type=steamer only. Hitters: wOBA quantiles.
                  Pitchers: RA9 quantiles, q90 is the BEST outcome.
                  Null wherever projected PA is trivial.

prospect        → empirical MiLB base rates              MODELED
                  conditioned on age-relative-to-level, level, performance.
                  Savant tool percentiles shift the prior. They are never
                  added to a score.
```

`E[m_t]` and `Var[m_t]` both derive from `mixture`. Nothing downstream reads
anything else.

Hard requirements:
- Every branch carries a COMPLETE 12-stat line. `add_mew` asserts no NaN.
  A washout branch is all zeros; `MEW(zero line) = 0` correctly, because `PA=0`
  zeroes the ratio terms.
- `line_0` for minor leaguers must come from a projection feed. Verified in
  `data/players.parquet`: of 87 `roster_status=="minors"` rows, **67 have every
  scoring stat at `0.0`** (`build._append_unprojected`) and 20 carry real RoS
  projections (max PA 135, max IP 34) because the feed covers near-MLB players.
  So `MEW` is identically zero for the 67 deep-minors rows — silently, since
  `add_mew`'s NaN assert passes on zeros. Assert nonzero volume per row before
  scoring; do not assert on the pool as a whole.
- Normalize accents before joining. The projections API returns `Jesús Made`,
  `Iván Herrera`, `Kevin Alcántara`. Route through `data_prep/names.py`.
- `OPS = OBP + SLG` where the feed publishes them separately.
- Published FV→outcome base rates CANNOT be pooled. Wang reports 21% and 10.4%
  bust for the same players in two papers; the gap is entirely definitional.
  Use one source against its own definition, or build our own. Prefer our own —
  no published table is denominated in roto categories.

## 5. Holding cost

A slot is occupied while the posterior is ambiguous. Drop when
`E[continuation] < 0`.

`H_k` TRUNCATES the branch's value stream. It does NOT charge a per-season rent —
opportunity cost is priced once, at the level of `V`, in §6. Charging rent here
as well would double-count it.

```
H_k ≜ E[ seasons a roster spot stays occupied | outcome tier k ]

u_t(p) = Σ_k prob_k · u_t(branch k) · 1{ t < H_k }
```

So the mid outcome is expensive not because it is charged rent, but because it
occupies the slot for `H_mid ≈ full horizon` while contributing `m_t ≈ 0` — and
§6 then differences that against an alternative occupant who would have
contributed something. `V_net < 0` iff the best available alternative has
positive `V_gross`. That is the testable form of the doctrine.

Why the mid outcome is worst — the mechanism, stated precisely:

```
bust → posterior sharpens DOWN     → drop early    → H short
star → posterior sharpens UP       → hold gladly   → H long, and worth it
mid  → posterior stays diffuse ≈ 0 → hold forever  → H long, and worth nothing

magnitude = (slot-years at m_t ≈ 0) = THE COST OF INFERENCE LAG
```

`H` is long for stars and mids alike. The difference is entirely what `m_t` is
worth during those years. Long `H` is not the defect; long `H` at zero
production is.

With instant perfect inference no player is ever worth negative. The negative
value is manufactured entirely by inference lag. This is empirically the modal
case, not an edge case: "regular" is the single most likely outcome at 50 FV
(30%) and 55 FV (30%).

Option value requires RESOLVABLE variance. Jensen gives
`E[max(C,0)] ≥ max(E[C],0)`, with the gap growing in the dispersion of the
posterior AT THE DECISION TIME — not the dispersion of the terminal outcome. A
player who stays uncertain forever has variance and no option value.

Two distinct variance channels. Do not conflate:

```
Λ_t     pays for TERMINAL outcome variance   — only BELOW the cutoff (§2.4)
option  pays for RESOLVABLE variance         — always, risk-preference-free
```

Star-hunting draws on both. "Mid is worst" draws only on the option.

IMPLEMENTATION: reduced form. Do not model the inference; measure its
consequence. `H_k` is one extra column off the same MiLB pull. No filtration,
no dynamic program. Applies to major leaguers too — you drop a declining
veteran on the same rule.

Do NOT index resolution time by outcome tier. Resolution IS the process of
learning the tier; indexing the learning time by the thing learned is circular.
Resolution speed tracks proximity to MLB, not outcome.

## 6. Opportunity cost

Differenced at the level of `V`, never as a per-season flow inside `u_t`.

```
V_net(p) = V_gross(p) − V_gross( best available alternative for p's slot path )
```

Rationale: a per-season rent flow is not dimensionally homogeneous across pools.
A MiLB slot generates ZERO current-season EW — it cannot be started — so its
value is entirely an option on future EW. `λ_MLB` would be a per-season EW flow
while `λ_MiLB` would be a discounted lifetime value. Differencing `V` avoids the
units error entirely.

```
pool(p,t) ∈ { MLB_active, MLB_bench, MiLB, dropped }
```

Three occupied pools, not two. A bench MLB player generates EW only through the
injury option; reuse `swap_evaluator.add_bench_value`, which already prices this.

Baseline selection is ONE PASS on `V_gross`, deliberately. Defining the baseline
by `V_net` would be circular.

`p`'s slot path is itself random — it depends on `arrive_k`. Difference against
the best available alternative for the slot `p` occupies AT ACQUISITION (MiLB for
a prospect, the relevant MLB slot for a big leaguer), not against a
probability-weighted blend of paths. This is a decision, not a derivation: a
blended baseline would price a prospect partly against major-league free agents
he cannot yet be substituted for.

Multi-position players are priced at their SCARCEST eligible slot
(`ceiling.add_eligibility` already computes the slot set).

`pool` changes at `arrive_k`: promotion is a pool transfer and falls out of the
mixture for free.

Not modelled today and required before this ships: the minor-league slot pool
does not exist in `config.json`, and `optimizer/rosters.py` excludes minors from
`get_main_roster`. Bench size is likewise implicit.

## 7. Reporting

```
V_p(β) − V_q(β) = Σ_t β^t · d_t          d_t = u_t(p) − u_t(q)

#crossings in β>0  ≤  #sign changes in (d_0, d_1, …, d_n)   [Descartes]
```

Truncate at `ε` FIRST. Descartes' rule applies to polynomials; the untruncated
object is a power series and the bound does not hold for it.

```
sign changes == 1  →  report "q passes p at β* = 0.xx"      UNIQUE, guaranteed
otherwise          →  report the full root set, or refuse to summarize
```

Prospect-vs-veteran has sign pattern `(−,−,+,+,+)` — one change — so the single
break-even is valid exactly in the case we care about. Two players can otherwise
cross more than once.

```
DOMINANCE — parameter-free. Run FIRST, before any calibration.
  u_t(p) ≥ u_t(q) ∀t, strict somewhere  ⇒  p ≽ q for ALL β
```

Reachability: weights lie on the moment curve `(1, β, β², …)`, so only
cyclic-polytope faces are exposed and players in non-convex dents of the Pareto
frontier are optimal at no single β. Portfolio selection over 28+10 additive
slots is an LP whose optimum mixes, so dents fill in. Dents invalidate only
claims of the form "p is never the single top-ranked player" — never "p is not
worth acquiring."

## 8. Module surface

No classes, no try/except, one wide frame, asserts carry actionable messages.

```
gradient.py   championship_gradient(my_totals, opp_totals, sigmas, seed) -> dict
                21 simulator calls, common random numbers. Returns G_c.
              directional_curvature(my_totals, opp_totals, sigmas, line, seed)
                -> float. 3 calls. Screened candidates only.
              nominal_totals(my_totals, league_mean, psi, t)          -> dict
              inflate_sigmas(sigmas, t, v)                            -> dict

decay.py      decay_table(seasons, group, decile)      -> DataFrame   [SCRIPT]
              survival_table(seasons, thresholds)      -> DataFrame   [SCRIPT]

outcomes.py   mixture(players, source)                 -> long frame
                cols: MLBAMID, branch, prob, arrive_t, <12 scoring stats>

value.py      season_payoff(mixture, L, Λ, gradient)   -> u_0..u_n on frame
              player_value(u, beta)                    -> V_gross
              net_value(V_gross, pools, slots)         -> V_net
              breakeven(u_p, u_q)                      -> β* | root set | None
```

## 9. Open parameters

```
β        THE knob. No default. Feel it out against real players.
v        roster-drift variance per season.        CALIBRATE
ψ        0.0 (A1).                               SETTLED
ε        convergence tolerance.                  numerical
k        finite-difference step, δ_c = k·σ_c.    k≈0.25; tune vs the noise floor
decile   conditioning granularity on decay/survival.  requested
n_sims   simulator draws. 20_000 default; raise if G_c fails its sign assert.
```

## 10. Known approximations

Each is a real hole. None is hidden.

1. **Variance double-count.** `Λ_t(p)·Var_t(p)` adds p's own outcome variance on
   top of `σ_c`, which `simulate_standings` already applies to the whole team —
   including p. `estimate_projection_uncertainty` builds `σ_c` from league means
   and fixed CVs, so it does not decompose into player contributions and the
   overlap cannot be subtracted exactly. First-order inconsistent. Quantify the
   size of `Λ·Var` against `E[m]` before trusting it; if it is small, say so and
   move on.

1b. **Steamer quantiles are not stat lines.** `q10..q90` are wOBA quantiles for
   hitters and RA9 quantiles for pitchers. `m_t` needs R/HR/RBI/SB/OPS and
   W/SV/K/ERA/WHIP. Converting a `q90` wOBA into a 12-stat line requires a
   component mapping that this design does not specify. Simplest defensible
   choice: scale the whole projected line by `q90/q50` and label it an
   approximation. Do not present a scaled line as a measured tail.

2. **`H_k` is not a function of β.** Measured holding durations reflect other
   managers' patience. The model's own impatience therefore does not feed back
   into the doctrine's magnitude, which is exactly the coupling §5 claims to
   describe. Reduced form; know its limit.

3. **σ inflation conflates two things.** `v` covers both forecast error and lack
   of control, but my future strength is not a random walk — I choose the roster
   each year. One parameter absorbing both means `v` is not separately
   identifiable from either.

4. **The nominal team is frozen.** §2.5 holds opponents' totals at current-season
   values for all `t ≥ 1`, so `G^(t)` sees an unaging league.

5. **`decay`/`survival` are fit on ages 21–40.** Scoring a 17-year-old's age-28
   peak extrapolates eleven years past the data, and no MLB data on 17-year-olds
   can exist. Assert that `arrive_k` falls inside the mixture's support and crash
   otherwise; do not silently extrapolate. This — not a horizon cutoff — is the
   correct guard.

6. **Pitcher star rates rest on n=30 and n=12.** Clemens reports 0% star at both
   55 and 60 FV, and reads a projection only three years out, so a year-five
   breakout cannot appear. Directionally strong, numerically fragile.

7. **Position effects do not transfer from the WAR literature.** Valancius's grid
   ranks 1B highest; in this league 1B is the CHEAPEST slot (replacement +6.60 vs
   C +2.80). Use the age-relative-to-level dimension, never the position one.

8. **~31% of 3+ WAR players were never top-100 prospects**, and never-ranked
   players supply ~39% of league WAR. The ranked pool is not the population.
   `V_gross` of the best available alternative is therefore higher than a
   list-based baseline implies. Conversely, list recall on genuine stars is ~92%,
   so a star-only screen is well served by lists.
