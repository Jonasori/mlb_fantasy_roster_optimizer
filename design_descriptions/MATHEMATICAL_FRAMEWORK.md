# Roster Optimization: Mathematical Framework

## 1. Problem Definition

### State

A **roster** R is a set of N_roster = 28 players, partitioned into:
- **Starters** S(R): 18 players assigned to lineup slots (C / 1B / 2B / SS / 3B / 5×OF / UTIL / 5×SP / 2×RP)
- **Bench** B(R) = R \ S(R): the remaining 10 players

### Scoring

Rotisserie league, 7 teams, 10 categories. Each team accumulates season-long totals. At season's end, teams are ranked 1–7 in each category; rank converts to standing points (7 for 1st, 6 for 2nd, ..., 1 for 7th). Total standing points across all categories determines the winner. Range: [10, 70].

Categories C = C⁺ ∪ C⁻ where:
- **Higher is better** C⁺ = {R, HR, RBI, SB, OPS, W, SV, K}
- **Lower is better** C⁻ = {ERA, WHIP}

Opponents: O = {1, ..., 6} (the other 6 teams, compared simultaneously)

### Transactions: The Move Abstraction

A **move** M is the universal operation for all roster changes. It is defined by:

- **out** ⊂ R₀: players leaving my roster
- **in**: players entering my roster
- |out| = |in| (roster size preserved)
- **dest**: out → {FA} ∪ Opponents (where each outgoing player goes)
- **src**: in → {FA} ∪ Opponents (where each incoming player comes from)

**Constraint model.** For each opponent o involved in the move, define:

```
send_o = {p ∈ out : dest(p) = o}    (players I send to opponent o)
recv_o = {p ∈ in  : src(p) = o}     (players I receive from opponent o)
```

If send_o ∪ recv_o ≠ ∅, a **trade-fairness constraint** must hold. Write V(X) = Σ_{p ∈ X} V(p) for the total **trade value** of a player set, where V is a per-player market-value function (Section 7). Two conditions, both relative to a tolerance δ ∈ [0, 1):

```
Aggregate:       [V(recv_o) − V(send_o)] / V(recv_o)  ≤  δ
Per-player max:  max_{p ∈ send_o} V(p)  ≥  (1 − δ) × max_{p ∈ recv_o} V(p)
```

The aggregate condition says the opponent accepts if they lose at most a fraction δ of the value they give up. The per-player condition says the best player they receive must be comparable to the best player they surrender — **you need to send a star to get a star**. See Section 7 for why both are needed.

Players with src/dest = FA have **no constraint** — free agent transactions are unilateral.

**Examples.** Every roster transaction is a special case:

| Move type | out (dest) | in (src) | Fairness constraint |
|---|---|---|---|
| FA swap | {A→FA} | {F←FA} | none |
| 1-for-1 trade | {A→opp_j} | {B←opp_j} | on V(A) vs. V(B) |
| 2-for-1 + FA fill | {A→opp_j, B→opp_j} | {C←opp_j, D←FA} | on V(A)+V(B) vs. V(C); max(V(A),V(B)) vs. V(C) |
| 1-for-2 + FA drop | {A→opp_j, C→FA} | {B₁←opp_j, B₂←opp_j} | on V(A) vs. V(B₁)+V(B₂); V(A) vs. max(V(B₁),V(B₂)) |
| 2-for-2 trade | {A→opp_j, B→opp_j} | {C←opp_j, D←opp_j} | on V(A)+V(B) vs. V(C)+V(D); max sent vs. max received |

The fairness constraint applies only to the opponent-routed portion. FA-routed players are unconstrained — note that in the 1-for-2 row, C is routed to FA and so does not count toward V(send_o). Imbalanced trades (2-for-1, 1-for-2) emerge naturally: the FA-routed components fill or vacate roster spots to preserve |out| = |in|.

Multi-opponent moves (three-way trades) are handled identically — one fairness constraint per opponent involved.

### Goal

Given current roster R₀, find the sequence of feasible moves that maximizes expected standing points for the remaining season.

---

## 2. Expected Wins: The Value Function

### Team Totals

Given a starting lineup S, team totals are computed as:

**Counting stats** (c ∈ {R, HR, RBI, SB, W, SV, K}):

```
my_c = Σ_{p ∈ S_type(c)} stat_c(p)
```

where S_type(c) selects hitters (for R, HR, RBI, SB) or pitchers (for W, SV, K).

**Ratio stats** (PA-weighted for OPS, IP-weighted for ERA/WHIP):

```
my_OPS  = Σ_{p ∈ S_H} PA(p) × OPS(p)  /  Σ_{p ∈ S_H} PA(p)
my_ERA  = Σ_{p ∈ S_P} IP(p) × ERA(p)  /  Σ_{p ∈ S_P} IP(p)
my_WHIP = Σ_{p ∈ S_P} IP(p) × WHIP(p) /  Σ_{p ∈ S_P} IP(p)
```

### The Rosenof Win Probability Model

For category c against opponent o, define the normalized gap:

```
         ⎧ (my_c − opp_{c,o}) / (σ_c √2)    if c ∈ C⁺  (higher is better)
z_{c,o} =⎨
         ⎩ (opp_{c,o} − my_c) / (σ_c √2)    if c ∈ C⁻  (lower is better)
```

In both cases, **positive z means I'm winning** the category against opponent o.

Win probability: P(beat o in c) = Φ(z_{c,o})

**Standing Points Decomposition:**

My standing points from category c = 1 + #{opponents I beat in c}. Therefore:

```
E[standing_points] = Σ_c [1 + Σ_o P(beat o in c)]
                   = 10 + Σ_c Σ_o Φ(z_{c,o})
                   = 10 + EW
```

where:

```
EW = Σ_{c ∈ C} Σ_{o ∈ O} Φ(z_{c,o})
```

Range: [0, |C| × |O|] = [0, 60]. This is the sum of 60 pairwise beat probabilities — one per category per opponent. The constant 10 does not affect optimization, so maximizing EW is equivalent to maximizing expected standing points.

**The σ_c parameter** is **projection uncertainty**: how much actual season outcomes could deviate from projections for category c. It is NOT observed cross-team variance (which reflects strategic choices like punting saves, not uncertainty).

For counting stats: σ_c = |league_mean_c| × CV_c, where CV is a fixed coefficient of variation per category (e.g., 0.06 for R, 0.15 for SB). For ratio stats: σ_c is a fixed absolute value (e.g., 0.012 for OPS, 0.30 for ERA).

σ_c can be calibrated via Monte Carlo simulation: run many simulated seasons with player-level projection noise, compute the empirical standard deviation of team totals, and use σ_c = σ_empirical / √2 (dividing by √2 because σ_c enters the formula as σ_c√2, representing the combined uncertainty of both teams). This gives the analytical model simulation-calibrated accuracy while retaining the speed of closed-form gradients.

---

## 3. The EW Gradient

The gradient ∂EW/∂(my_c) is the **marginal value of one additional unit of category c** to your expected win total. It is the central computational object in the framework.

### Derivation

```
∂EW/∂(my_c) = Σ_o ∂Φ(z_{c,o})/∂(my_c) = Σ_o φ(z_{c,o}) × ∂z_{c,o}/∂(my_c)
```

Since ∂z/∂(my_c) = +1/(σ_c√2) for c ∈ C⁺ and −1/(σ_c√2) for c ∈ C⁻:

```
For c ∈ C⁺:   g_c = ∂EW/∂(my_c) = +Σ_o φ(z_{c,o}) / (σ_c √2)   > 0
For c ∈ C⁻:   g_c = ∂EW/∂(my_c) = −Σ_o φ(z_{c,o}) / (σ_c √2)   < 0
```

### Interpretation

- g_c > 0 for C⁺: more runs, more HRs, etc. help you. Obvious, but the magnitude tells you HOW MUCH.
- g_c < 0 for C⁻: a higher ERA total hurts you. A pitcher who lowers your ERA has a positive contribution to EW (negative g × negative ΔERA = positive).
- **Large |g_c|**: category c is "in play" — you're in tight races against multiple opponents. High marginal ROI.
- **Small |g_c|**: category c is decided — you're either dominating or getting dominated. Low marginal ROI. This naturally encodes punt logic: if you're hopelessly behind in SV, |g_SV| ≈ 0, and the model correctly de-prioritizes SV investment without any special-case logic.

### Gradient Depends Only on Current State

The gradient is a function of (my_totals, opponent_totals, σ). It does not depend on individual player stats. Compute it once, then use it to score every player in the database.

---

## 4. Per-Player Scoring: MEW

### Marginal Swap Value (MSV)

For any move M = (out, in), the **Marginal Swap Value** is:

```
MSV(M) = EW((R₀ \ out) ∪ in) − EW(R₀)
```

Computing MSV exactly requires re-solving the lineup assignment for the modified roster and recomputing EW. This is the ground truth, used for final evaluation.

### First-Order Approximation

For fast screening, linearize EW around the current team totals:

```
MSV(M) ≈ Σ_c  g_c × Δmy_c(out, in)
```

where g_c is the gradient (Section 3) and Δmy_c is the change in team total.

**Counting stats** (c ∈ {R, HR, RBI, SB, W, SV, K}):

```
Δmy_c = Σ_{p ∈ in} stat_c(p) − Σ_{p ∈ out} stat_c(p)
```

**Ratio stats** — first-order approximation, valid when individual PA/IP are small relative to team total:

```
Δmy_OPS  ≈ [Σ_in PA(p)(OPS(p) − my_OPS) − Σ_out PA(p)(OPS(p) − my_OPS)]  / total_PA
Δmy_ERA  ≈ [Σ_in IP(p)(ERA(p) − my_ERA) − Σ_out IP(p)(ERA(p) − my_ERA)]  / total_IP
Δmy_WHIP ≈ [Σ_in IP(p)(WHIP(p) − my_WHIP) − Σ_out IP(p)(WHIP(p) − my_WHIP)] / total_IP
```

### The MEW Score

Because the Δmy_c terms are additive across players, we can define a per-player **Marginal Expected Wins** score. MEW is a single unified formula for all players — hitters and pitchers alike:

```
MEW(p) = Σ_{c ∈ C_count} g_c × stat_c(p)
       + g_OPS  × PA(p) × (OPS(p)  − my_OPS)  / total_PA
       + g_ERA  × IP(p) × (ERA(p)  − my_ERA)  / total_IP
       + g_WHIP × IP(p) × (WHIP(p) − my_WHIP) / total_IP
```

where C_count = {R, HR, RBI, SB, W, SV, K}.

No conditional logic is needed. For hitters, IP = 0, so ERA/WHIP/W/SV/K terms vanish. For pitchers, PA = 0, so OPS/R/HR/RBI/SB terms vanish. The data encodes the player type; the formula is universal.

Then for any move M = (out, in):

```
MSV(M) ≈ Σ_{p ∈ in} MEW(p) − Σ_{p ∈ out} MEW(p)
```

MEW scores are comparable across all players. They can be computed in O(1) per player after the gradient is known.

**Sign check for ERA (verify this in implementation):** g_ERA < 0. A good pitcher has ERA(p) < my_ERA, so (ERA(p) − my_ERA) < 0. Product: negative × negative = **positive MEW contribution**. A good pitcher scores well. ✓

### Rate Stat Separability

A natural question: can ratio stats be decomposed into counting-stat primitives to achieve separability? For ERA = 9 × ER / IP, the win condition against opponent o is:

```
my_ERA < opp_ERA_o
⟺  Σ_i ER_i × x_i  <  (opp_ERA_o / 9) × Σ_i IP_i × x_i
⟺  Σ_i IP_i × (opp_ERA_o − ERA_i) / 9 × x_i  >  0
```

Each player's contribution to the ERA win margin is `IP_i × (opp_ERA_o − ERA_i) / 9` — fully separable per player. This is exactly the MILP linearization coefficient.

The gradient-based MEW score achieves the same separability from a different direction: g_ERA × IP(p) × (ERA(p) − my_ERA) / total_IP is also per-player separable. The two approaches are mathematically equivalent at first order — the gradient linearization implicitly performs the counting-stat decomposition.

The decomposition does NOT help with the probabilistic model directly. The Rosenof model treats ERA as a single Gaussian with one σ_ERA. Decomposing into ER and IP would require modeling their joint distribution (they're correlated: more innings → more earned runs), losing the clean single-σ structure. The gradient linearization is the right way to get per-player separability while keeping the probabilistic model intact.

### When the First-Order Approximation Breaks Down

1. **Lineup cascades**: the incoming player displaces a different starter than the one being dropped, triggering a chain of lineup changes. The gradient doesn't see cascading effects.
2. **Large swaps**: if the swap shifts team totals enough to materially change the gradient (z-scores move significantly), the linear approximation is inaccurate. See Section 5 for quantitative bounds on gradient stability.
3. **Bench players**: a bench player's MEW reflects what they would contribute *as a starter*, but their actual EW contribution from the bench is zero — their value comes entirely from insurance (Section 6). So `MSV_approx = MEW(in) − MEW(out)` is badly wrong as an *estimate* of ΔEW when the outgoing player is on the bench: it attributes starter-level EW contribution to a non-starter. This is why the screening stage is used only to generate candidates, never to value them (see below).
4. **Multi-position interactions**: a 2B/SS player's value depends on the depth at both positions, which the per-player score doesn't capture.
5. **EW surface convexity in deeply-losing categories**: see Section 4a.

### The Screening + Exact Evaluation Architecture

The division of labour is deliberate and asymmetric: **screening owns recall and feasibility; evaluation owns value.**

1. **Screen** (O(N), fast): compute MEW for all available players and rank candidates by `MSV_approx = Σ_{p ∈ in} MEW(p) − Σ_{p ∈ out} MEW(p)`. Keep top K.

   No attempt is made to model the lineup at this stage. Given the case analysis above, a lineup-aware MSV_approx is achievable — but it buys nothing, because the exact evaluation recomputes ΔEW from a real MILP solve anyway. Every candidate that survives screening is re-valued from scratch, so screening error costs only **recall** (a good move ranked below the top K), never a wrong recommendation. The dominant recall failure is the bench case: a free agent who would merely sit can outrank a genuine upgrade. The league is small enough that raising K is the remedy.

   What screening *does* own is **feasibility**, and there the checks are exact, not approximate: protected players (the last startable cover for a required slot may only be swapped for someone who also covers it), roster-composition bounds, and injury-stash protection. These are hard constraints, so they are enforced before any MILP solve rather than discovered by a failed one.

2. **Evaluate** (O(K × lineup_solve), slower): for each of the top K candidates, compute exact MSV by re-solving the lineup and recomputing EW. Recompute ΔBV from updated MEW (Section 6). The exact MSV catches lineup cascades, multi-position interactions, and nonlinear effects that screening cannot see. **All reported numbers come from this stage.**

### 4a. EW Surface Convexity and Screening Bias

The first-order approximation (MEW) evaluates the gradient at the current team totals and assumes linearity. In practice, the EW surface has curvature determined by the second derivative:

```
∂²EW/∂(my_c)² = −Σ_o z_{c,o} × φ(z_{c,o}) / (σ_c √2)²
```

The sign depends on the z-scores:

- **Winning regime (z > 0)**: The second derivative is negative (concave). MEW **overestimates** the marginal value of further improvement — you're already winning, and additional gains yield diminishing returns. This is safe: the optimizer might slightly over-invest in categories you're winning, but the exact evaluation corrects it.

- **Losing regime (z < 0)**: The second derivative is positive (convex). MEW **underestimates** the marginal value of improvement — the actual gain from improving in a deeply-losing category is larger than the linear prediction. This is dangerous: the screening may systematically miss improvements in deeply-losing categories because it undervalues them.

**Quantitative example (OPS):** If team OPS gives z ≈ −1.5 against all opponents, the gradient says each OPS point is worth ~50 EW. But moving from z = −1.5 to z = −1.0 actually yields ~60–100% more EW than the linear prediction, because Φ(z) curves upward steeply in the left tail.

**Impact on the optimizer:** For small perturbations (single FA swaps), the bias is modest (< 0.05 EW) because the stat change per swap is small relative to σ_c. The exact evaluation corrects any screening error. For large perturbations (trades that significantly shift a category), the bias can be substantial — the gradient may undervalue a trade by 1.5–2× in deeply-losing categories.

**Mitigation:** When evaluating trades or multi-player moves that would significantly change a deeply-losing category, always compute exact EW rather than relying on MEW-based estimates. The gradient is reliable for ranking and small moves; it is unreliable for valuing large shifts in categories where |z| > 1.

---

## 5. Lineup Assignment (Inner Optimization)

Given a roster R, finding the starting lineup S*(R) that maximizes the season objective is the **inner optimization** — called as a subroutine whenever we need exact EW for a roster.

### MEW as the Lineup Objective for My Team

The lineup MILP maximizes Σ MEW(starters) subject to slot eligibility constraints. Because MEW is a linear function of player stats (given a fixed gradient), this is a standard weighted bipartite matching (players → slots), solvable by MILP in microseconds.

MEW is the right objective because it directly reflects what the team needs: the gradient g_c weights each category by current marginal value, so the lineup prioritizes players who contribute most to categories where improvement matters. A context-free metric (like FV) treats all categories equally, ignoring that +5 HR might be worth much more than +2 SB given the current standings.

### The Circularity and Why It's Benign

MEW depends on the gradient, which depends on team totals, which depend on the lineup. This creates a fixed-point problem: lineup → totals → gradient → MEW → lineup.

**Resolution:** Initialize with any reasonable lineup (e.g., using FV), compute the gradient, solve for the MEW-optimal lineup, and check whether the lineup changed. If it did, recompute the gradient from the new totals and re-solve. Convergence is fast because the gradient is **self-correcting**: improving in a category c increases z_{c,o}, which decreases φ(z_{c,o}), which decreases |g_c|. This is the concavity of Φ acting as a damper.

**Quantitative bound.** Consider swapping one marginal starter. A typical stat change is ~10 HR out of a team total of ~250. With σ_HR = 22.5 and an opponent at z = 0 (worst case — tightest race):

```
Δz = 10 / (σ √2) = 10 / 31.8 = 0.314
φ(0) = 0.399  →  φ(0.314) = 0.379
```

The gradient contribution from this opponent changes by ~5%. Summed across 6 opponents at various z-scores (most not at z = 0), the aggregate gradient change from a marginal lineup swap is 2–8%. This is well within the range where first-order linearity holds.

In the rare worst case (hyper-specialist player oscillating between start/bench), the iteration has exactly two states. Evaluate both, pick the one with higher actual EW. Total cost: at most 3 MILP solves (< 3ms).

### FV for Opponent Lineups

Opponent lineups are solved using **Fantasy Value (FV)** — a context-free z-score sum across each player type's 5 scoring categories (with ERA/WHIP negated before z-scoring).

FV is appropriate for opponents because we model them as optimizing context-free quality, not optimizing against specific category needs. FV does not depend on any team's gradient, so it introduces no circularity and can be computed once for all opponents.

FV is NOT used for my team's lineup assignment (MEW is). FV serves three roles: (1) opponent lineup modeling, (2) a context-free diagnostic for general player quality, and (3) the **default** trade value column (Section 7) — a placeholder standing in for a real market price until one is available.

### Scale and Performance

With 28 binary decision variables and ~18 slot constraints, the lineup MILP solves in under 1 millisecond with HiGHS. This is negligible cost. Even 500 candidate evaluations complete in under a second.

### MILP for Ceiling Analysis (not implemented)

A separate, larger MILP could answer: "What is the best possible roster from the full candidate pool?" — a **diagnostic**, never the move-planning tool. A small gap between current EW and the unconstrained ceiling means focus on bench depth and marginal trades; a large gap means major structural upgrades are available.

This is a design note only. No ceiling diagnostic exists in the code, and the greedy optimizer's complementary-move blind spot (Section 8) is currently unmeasured.

---

## 6. Season Value and Bench Insurance

### The Single Optimization Target

The question "what do we maximize?" has one answer: **expected standing points for the remaining season, accounting for all plausible absence scenarios.**

In roto, there are no weekly matchups — only season-end cumulative totals matter. The season is one scoring period:

```
EW_season(R) = E_a[ EW(L*(R, a)) ]
```

where a is the **availability vector** (a_i = 1 if player i is available, 0 if absent), L*(R, a) is the optimal lineup from the available players, and the expectation is over the distribution of absences.

This single number captures both starter quality and bench quality. A roster with elite starters and a terrible bench has high EW when healthy but collapses when starters miss time. A roster with good starters and a strong bench is more robust across scenarios.

### Perturbation Expansion

Expand EW_season around the all-healthy scenario:

```
EW_season(R) = P(healthy) × EW_healthy(R)
             + Σ_s P(only s absent) × EW(L*(R \ {s}))
             + [multi-absence terms]
```

where P(healthy) = Π_s (1 − P_absent(s)) and P(only s absent) ≈ P_absent(s) × Π_{t≠s} (1 − P_absent(t)).

**For absolute EW_season**, this first-order expansion may be inaccurate: with 18 starters each absent ~20–35% of the time, the all-healthy probability is very low (< 1%).

**For move evaluation (deltas), the expansion is much more accurate.** When comparing two rosters that differ by one player, the multi-absence terms largely cancel — both rosters have the same starters absent in those scenarios, differing only in how the one changed player affects each scenario.

### Bench Value as Derived Quantity

Define the bench value of player b as the EW advantage of having b on the bench versus the best available free agent:

```
BV(b | R) = Σ_{s ∈ starters(R)}  P_absent(s)  ×  max(0,
                EW(L*(R \ {s}))  −  EW(L*((R \ {b, s}) ∪ {FA_b}))
            )
```

where:
- L*(R \ {s}) = optimal lineup with starter s absent (b available as replacement)
- (R \ {b, s}) ∪ {FA_b} = roster if b's spot had been held by the best available FA
- P_absent(s) = expected fraction of remaining season starter s is absent

This is not a separate objective from EW. It is the first-order perturbation term of EW_season that captures roster resilience. The delta formulation for evaluating a move is:

```
ΔValue(M) = ΔEW_healthy(M) + ΔBV(M)
```

Both terms are natively in the same units (expected category wins). ΔBV is the change in total bench insurance value across the roster. A move is worth making if ΔValue > 0.

### Computing BV (Gradient-Based)

BV can be computed analytically using MEW with no MILP solves. When starter s is absent and bench player b fills in, the EW change is approximately MEW(b) − MEW(s). In the counterfactual (FA_b fills in instead), the change is MEW(FA_b) − MEW(s). The starter's MEW cancels:

```
BV(b) ≈ Σ_{k : best_bench(k) = b}  P_absent(k) × max(0, MEW(b) − MEW(best_FA(k)))
```

where k indexes starter slots, best_bench(k) is the highest-MEW bench player eligible for slot k, and best_FA(k) is the highest-MEW free agent eligible for slot k. FA_b selection uses highest-MEW free agent not on any roster — this correctly identifies the FA that would contribute most to my team's EW if rostered. A bench player who is the best option for multiple slots accumulates BV from all of them.

This is a pure function of MEW scores and absence probabilities — no MILP solves required. It misses lineup cascades (e.g., when filling slot k with b frees other slots), but BV is inherently a rough statistic: it relies on estimated absence rates, assumes independent single-starter absences, and evaluates bench players independently. The gradient-based approximation is well-matched to this level of precision. One formula, one code path, used for both screening and exact evaluation.

**Key properties:**
- **Position-aware**: a bench SS is worth more when your starting SS has a high absence rate
- **FA-pool-relative**: if great FAs are available at a position, bench depth there is less valuable
- **Identifies droppable players**: if BV(b) ≈ 0 for a bench player, they're not providing meaningful insurance

**Simplifying assumptions:**
- At most one starter absent at a time (ignore multi-absence scenarios; first-order approximation)
- Each bench player evaluated independently (ignore bench-bench interactions)
- P_absent(s) estimated from historical injury/rest rates by position slot
- Lineup cascades ignored (bench player directly fills the absent starter's slot)
- Multi-absence scenarios largely cancel in ΔBV for 1-for-1 move evaluation (see Perturbation Expansion above)

---

## 7. Trade-Specific Considerations

The move abstraction (Section 1) unifies FA swaps and trades at the evaluation layer — `MSV(M)` is computed identically regardless of player sources. This section covers the additional concerns specific to moves involving opponent rosters.

### The Trade-Fairness Constraint Model

**Trade value V(p)** is an exogenous per-player number standing in for what the rest of the league thinks a player is worth. It is deliberately **not a model** — no hand-tuned formula over projections, no fame premium. It is a single column, resolved at the call site (`value_column`, default the context-free FV), so it can be repointed at a real market price — auction salaries, an auction-value column, a consensus dynasty ranking — without touching any of the math below.

This is a firm design boundary. Any function of our own projections is by construction correlated with our own MEW, which defeats the purpose: the constraint exists to encode *someone else's* valuation. A real market price is the only thing that does that honestly.

The constraint defines the set of opponent-acceptable trades. For each opponent o involved, with tolerance δ:

```
Aggregate:       [V(recv_o) − V(send_o)] / V(recv_o)  ≤  δ
Per-player max:  max_{p ∈ send_o} V(p)  ≥  (1 − δ) × max_{p ∈ recv_o} V(p)
```

Both are **relative**, not absolute: δ is a fraction of the value at stake, so the same tolerance behaves sensibly for a swap of two waiver-wire arms and for a blockbuster.

**Why the aggregate check alone is not enough.** Value is superadditive at the top of the market. Real owners will not trade one star for three useful regulars of equal summed value, because the star is scarce and the regulars are replaceable. An aggregate-only constraint happily proposes exactly that trade, and every such proposal is wasted. The per-player max condition encodes the missing structure: **to get a star you must send a star.** Together the two checks bound both the total and the shape of the package.

**The goal is value balance ≈ 0** (opponent sees it as fair) **while MSV >> 0** (I gain significant EW). The information asymmetry — I optimize EW against my own category needs, they price by market consensus — is the source of exploitable surplus. A player can be simultaneously fairly priced and worth far more to my roster than to theirs.

**Why not model mutual EW benefit instead?** It requires computing opponent EW accurately, assumes opponents are rational optimizers, and creates circularity (cost depends on trade terms, which depend on cost). An exogenous value column resolves all three: it is a fixed input, defines a bounded search space, and within that space pure EW maximization applies.

### Opponent Totals Update

When evaluating a trade involving opponent o, the opponent's roster changes. The exact evaluation should re-solve the opponent's lineup to reflect this:

1. Compute my new lineup and totals (1 MILP solve, as for any move)
2. Compute opponent o's new lineup and totals from their modified roster (1 additional MILP solve, using FV as objective)
3. Compute EW using my new totals and updated opponent_totals (opponent o's totals change; all other opponents unchanged)

This captures both effects: (a) my totals improved, and (b) the opponent I traded with got stronger or weaker.

**Why this costs only 1 extra MILP, not 2.** The baseline state computation (Section 8) already MILP-optimizes every opponent's lineup to compute their pre-trade totals. So the pre-trade opponent totals are already MILP-optimal. The only new computation is the opponent's *post-trade* lineup from their modified roster — 1 extra MILP per trade evaluation. Both sides of the comparison (pre-trade and post-trade) use MILP-optimal lineups, so the comparison is internally consistent regardless of whether the opponent actually plays their optimal lineup in practice.

For FA swaps, opponent totals are unchanged and no extra solve is needed.

### Trade Search Heuristics

For each opponent o:
1. **Identify targets**: players on o's roster with high MEW for my team
2. **Identify trade chips**: my players with high trade value but low MEW (valuable to the market, expendable to me)
3. **Enumerate candidate moves**: 1-for-1, 2-for-2, and imbalanced (2-for-1+FA, 1-for-2+drop) from chips × targets
4. **Filter**: fairness constraints (aggregate + per-player max) on the opponent-routed portion
5. **Score**: MSV_approx via MEW
6. **Exact evaluation**: compute exact MSV + ΔBV for top candidates via lineup re-solve (+ opponent re-solve for trades)

Trade recommendations are presented alongside FA optimization results. The evaluation math is identical — the fairness constraint and opponent totals update are the only trade-specific additions.

### Trade Non-Convexity

Multiple trades with the same opponent interact: trade A and trade B may each satisfy the fairness constraint, but doing both changes both teams' rosters, potentially making the combined package infeasible or suboptimal. When considering multiple trades with one opponent, evaluate the **combined** move, not individual trades independently.

---

## 8. The Optimization Pipeline

### Unified Pipeline: State → Screen → Evaluate → Present

All roster decisions flow through one pipeline, parameterized by the candidate pool and constraint set:

```
COMPUTE STATE
    Solve my lineup (MEW-optimal, Section 5)
    Compute my_totals, opponent_totals, gradient, MEW for all players
    Compute BV for all bench players

SCREEN (O(N), fast)
    Input: candidate pool (FA pool, opponent rosters, or both), constraints (fairness for trades)
    For each candidate add, pair with best legal drop(s) to maximize MSV_approx
    Enforce: slot coverage, roster composition, injury stash (Section 9c)
    Enforce: fairness constraints on opponent-routed portion (for trade candidates)
    Rank by MSV_approx (ΔBV is exact-only — see Section 4)
    Keep top K candidates

EVALUATE (O(K × MILP), exact)
    For each top-K candidate:
        Compute exact MSV via lineup re-solve
        For trades: re-solve opponent lineup, update opponent totals
        Recompute ΔBV from updated MEW (lineup re-solve changes who starts)
        Value = MSV_exact + ΔBV
    Rank by Value

PRESENT
    Partition results by actionability:
        FA-only moves → "execute" recommendations (unilateral)
        Moves involving opponents → "trade" recommendations (require acceptance)
```

The **evaluation math is identical** for all move types. The only branching is at the presentation layer (can I execute this unilaterally?) and in the constraint filtering during screening (fairness applies to trade components only). This is inherent in the problem — FA moves are free, trades require consent — not an artifact of the algorithm.

### Greedy Optimizer (FA Moves)

FA moves can be executed unilaterally. The greedy loop executes one move at a time, recomputing state after each:

```
REPEAT:
    Run the unified pipeline with candidate_pool = FA_pool, constraints = []
    If best Value > threshold: execute the move, update roster
    Else: stop
UNTIL stopped

Output: batch recommendation (all drops and adds)
```

Each iteration recomputes the full gradient and MEW from scratch. This is essential: after each move, the team totals change, which changes the gradient, which changes which subsequent move is optimal. The greedy approach converges because EW exhibits approximate diminishing returns — improving in a category reduces |g_c| via the concavity of Φ.

**Limitation:** greedy can miss **complementary moves** — two swaps that are each negative individually but positive together. This blind spot is currently unmeasured: there is no EW-ceiling diagnostic (Section 5). When a complementary pair is suspected, evaluate it directly as a single 2-for-2 move — exact MSV is indifferent to how many players a move touches.

### Trade Search (On-Demand)

Trade search uses the same pipeline with different inputs:

```
Run the unified pipeline with candidate_pool = opponent_rosters, constraints = [fairness]
```

Trade recommendations can include imbalanced trades (2-for-1, 1-for-2) with FA fills/drops, because the move abstraction and screening naturally enumerate these.

Trades are NOT part of the greedy loop because they require opponent acceptance and cannot be executed unilaterally. They are evaluated on request and presented as recommendations.

---

## 9. Implementation Pitfalls

### 9a. Sign Conventions

This is the single most dangerous implementation detail. The sign of g_c × Δmy_c must be **positive when a move helps you**.

| Category | g_c sign | Good player has... | Δmy_c sign | Product |
|----------|----------|-------------------|------------|---------|
| R, HR, RBI, SB, W, SV, K | + | High stat | + (more is better) | + ✓ |
| OPS | + | High OPS | + (raises team OPS) | + ✓ |
| ERA | − | Low ERA | − (lowers team ERA) | + ✓ |
| WHIP | − | Low WHIP | − (lowers team WHIP) | + ✓ |

**Verification test:** compute MEW for a pitcher with 2.50 ERA and one with 4.50 ERA, same IP. The 2.50 ERA pitcher must score higher. If not, there's a sign error.

### 9b. The Ratio Stat Replacement Trap

Replacing a high-IP pitcher whose ERA is **below** team average with a low-IP pitcher whose ERA is **even lower** can worsen your team ERA.

**Example:** Team has 1000 IP, 3.00 ERA.
- Remove: 200 IP, 2.80 ERA pitcher (below average — this pitcher was pulling ERA down)
- Add: 50 IP, 2.50 ERA pitcher (even better rate, but far fewer innings)
- New ERA: (3000 − 560 + 125) / 850 = 3.018 — **worse by 0.018**

The volume loss (removing 200 IP of good performance) dominates the rate improvement (adding 50 IP of slightly better performance). The linearized Δmy_ERA formula captures this correctly:

```
Δmy_ERA ≈ [50 × (2.50 − 3.00) − 200 × (2.80 − 3.00)] / 1000
        = [−25 − (−40)] / 1000
        = +15 / 1000 = +0.015  (ERA increases = worsens)
```

Since g_ERA < 0, the MSV contribution from ERA is **negative** — the model correctly flags this as a bad move. ✓

### 9c. Roster Composition Constraints

Every candidate move must be guarded against making the roster infeasible. Screening enforces three hard constraints before any MILP runs, so infeasible pairs never reach the top-K list:

- **Slot coverage (conditional).** A player who is one of the last startable covers for a required slot may only be dropped for someone who *also* covers that slot — your only startable catcher can be swapped for a better catcher, not for an outfielder. Coverage is injury-aware: a player who cannot start provides no coverage and needs no protection.
- **Roster composition.** The post-move active roster must stay within the league's hitter and pitcher bounds. Players on the IR sit outside those bounds and are not counted.
- **Injury-stash protection.** A rest-of-season projection for an injured player is already their post-return value, while the lineup model scores them zero because they cannot start today. Dropping such a player therefore requires a strictly better replacement, so that future value is never traded for a lesser player.

The exact evaluation (MILP re-solve) enforces all position-slot constraints as a second guard: if the new roster cannot fill every slot, the MILP fails and the move is discarded.

### 9d. Multi-Position Player Value

A player eligible at 2B and SS affects both position markets simultaneously. The MEW screening score doesn't capture this — it evaluates the player at their best single slot. The exact lineup solve handles multi-position interactions automatically, which is why the exact evaluation phase is necessary.

### 9e. Category Punting

The gradient handles this naturally:
- If you're far behind in SV, |g_SV| ≈ 0 — the model deprioritizes SV investment
- If you're far ahead in K, |g_K| is also small — no need to invest further

The gradient cannot recommend **initiating** a punt (proactively trading away SV producers). That requires evaluating the full trade outcome, which the exact evaluation and trade search handle.

### 9f. In-Season Dynamics

Projections change over the season. A player projected for 500 PA in March may have only 200 PA remaining by August. The framework should be re-run as projections update, with "remaining season value" replacing "full season value." The gradient and MEW scores automatically reflect updated projections — no structural changes needed, just updated inputs.

### 9g. MEW Staleness After Roster Changes

MEW depends on the gradient, which depends on team totals, which depend on who's on the roster. After each greedy move, MEW must be recomputed from scratch. Reusing old MEW values will produce incorrect rankings.

### 9h. Lineup MILP Uses MEW for My Team, FV for Opponents

My lineup maximizes Σ MEW(starters) — context-aware, reflecting current category needs. Opponent lineups maximize Σ FV(starters) — context-free, modeling their optimization behavior. Do not mix these: using FV for my lineup forfeits gradient information; using MEW for opponents requires a gradient we don't have for them.

---

## 10. Summary

### Conceptual Map

| Concept | Fintech Analogy | Roster Problem |
|---|---|---|
| Portfolio | Collection of assets | Roster of 28 players |
| Expected return | E[portfolio return] | Expected wins (EW, range 0–60) |
| Risk | Variance of return | σ_c (category-level projection uncertainty) |
| Rebalancing | Selling X, buying Y | A move: out set, in set, tagged by source/dest |
| Transaction cost | Bid-ask spread | Trade-fairness constraint (must appear fair to opponent) |
| Options / insurance | Call option on asset price | Bench player as insurance against starter injury |
| Gradient / MCTR | Marginal contribution to risk | g_c = ∂EW/∂(my_c), the category gradient |
| Screening alpha | Factor model score | MEW score (fast per-player ranking) |

### The Core Pipeline

**One pipeline for all roster decisions:**

1. **State**: compute my totals (MEW-optimal lineup), opponent totals (FV-optimal lineups), gradient, MEW for all players
2. **Screen**: rank all candidates from any pool (FA, opponent, mixed) by MSV_approx, enforce constraints exactly (fairness for trades, protected players, roster composition). Screening owns recall and feasibility only — never reported value.
3. **Evaluate**: exact MSV + ΔBV for top K via lineup MILP re-solve (+ opponent re-solve for trades)
4. **Present**: partition by actionability — FA moves (execute) vs. trades (recommend)

**Greedy loop** applies this pipeline iteratively for FA moves, executing the best move and recomputing state each round.

**Trade search** applies the same pipeline on-demand with opponent rosters as the candidate pool and trade fairness as the constraint.

### Key Decisions Made in This Framework

- **One optimization target**: EW_season = E_a[EW(L*(R,a))], a single expected value across absence scenarios. BV is a derived first-order perturbation term, not a separate objective.
- **One player-evaluation metric**: MEW is used for my lineup assignment, move screening, and BV computation. FV is reserved for opponent lineup modeling, diagnostics, and as the default stand-in for trade value.
- **One BV formula**: gradient-based analytical BV, a pure function of MEW scores and absence probabilities. Used for both screening and evaluation. No MILP-based BV — the inherent roughness of absence estimation doesn't justify a more expensive computation.
- **One pipeline for all move types**: FA swaps, trades, and mixed moves flow through the same state → screen → evaluate → present pipeline. The only branching is in constraint enforcement (fairness for trades) and presentation (executable vs. recommendation).
- **The move abstraction unifies all transactions**: every roster change is (out, in) with tagged sources/destinations. Imbalanced trades emerge naturally as mixed opponent/FA components.
- **Gradient as central object**: the category-level gradient g_c drives all player evaluation. Computed once per iteration, used to score every player.
- **Screening + exact evaluation**: fast first-order pass identifies candidates; expensive exact computation verifies them.
- **Trade-fairness constraint from an exogenous value column**: models real opponent behavior (market consensus, not our own projections), avoids game-theoretic circularity, defines a clean bounded search space. Two conditions — aggregate loss and per-player max — because value is superadditive at the top of the market.
- **Opponent totals updated for trades**: one extra MILP solve per trade evaluation captures the effect of weakening/strengthening the trade partner.
