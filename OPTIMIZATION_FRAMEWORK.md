# Roster Optimization: A Complete Mathematical Framework Analysis

## 1. Problem Decomposition: What Are We Actually Optimizing?

Let me start by precisely defining the decision problem, because the current codebase conflates several distinct sub-problems that need different mathematical tools.

### The State Space

The state is a **roster** R of 28 players, partitioned into:
- **Starters** S(R): the optimal lineup assignment (18 players filling slots C/1B/2B/SS/3B/5xOF/UTIL/5xSP/2xRP)
- **Bench** B(R) = R \ S(R): the remaining 10 players

Given R, the **value function** V(R) is a function of:
1. **Team totals** T(S(R)) across 10 categories — determined entirely by the starting lineup
2. **Opponent totals** {T_j} for j = 1..6 — treated as fixed (or stochastic) inputs
3. **Projection uncertainty** σ_c for each category

The Rosenof model gives you V(R) = Φ(μ_D / σ_D), the win probability. But the intermediate quantity μ_T (expected matchup wins, range 0-60) is actually the more operationally useful objective, and that's what the code uses as "expected wins" (EW).

### The Decision Problem

We face a **portfolio rebalancing problem**: given current roster R₀, find a sequence of transactions that transform R₀ into some R* that maximizes V(R*), subject to transaction constraints.

The transactions come in two types:
1. **Free agent swaps** (zero cost): drop player a ∈ R₀, add player f ∈ FA. Constraint: 1-for-1.
2. **Trades** (negotiation cost): send {a₁,...,aₖ} ⊂ R₀, receive {b₁,...,bₖ} from opponent j. Constraint: opponent must accept (fairness constraint), N-for-N.

This is exactly analogous to **portfolio rebalancing with heterogeneous transaction costs**.

## 2. What the Fintech Analogy Actually Is

The correct analogy isn't basic Markowitz mean-variance. It's closer to the following three concepts combined:

### 2a. Portfolio Rebalancing with Transaction Costs

In finance, you have:
- Current portfolio w₀ (your roster R₀)
- Target portfolio w* (optimal roster R*)
- Transaction costs c(w₀ → w*) that differ by asset class

In the roster problem:
- **FA swaps** have zero monetary cost but a **waiver priority** cost (earlier claims burn priority for later claims — though this may not apply in all leagues)
- **Trades** have a **negotiation cost**: you must give up something the opponent values. The "cost" of acquiring player b from opponent j is the minimum value you must surrender in return for j to accept.

The key insight from portfolio theory: **you should not rebalance to the unconstrained optimum.** You should rebalance to the optimum *net of transaction costs.* A trade that moves you from 42.1 EW to 43.5 EW is only worth doing if the cost of the trade doesn't damage your position more than 1.4 EW elsewhere.

### 2b. Real Options / Insurance Pricing (for bench value)

Bench players are literally **call options on future starting value**. A bench player b has:
- **Intrinsic value** = 0 (they don't contribute to team totals right now)
- **Option value** = E[max(0, EWA(b) if starter s gets injured)]

This is a conditional expectation over injury scenarios:

```
BV(b) = Σ_s P(s injured) × max(0, EWA(b replacing s) - EWA(best_FA replacing s))
```

The current BV formula captures some of this with `max(0, FVAR) × Scarcity × P(need)`, but it's too crude — it doesn't account for **which specific starter** would be lost, and it doesn't properly discount against the free agent replacement pool.

The fintech equivalent: bench value = **premium of an insurance policy** against starter injury. The "strike price" is the quality of the best free agent replacement at the position.

### 2c. Gradient-Based Marginal Analysis (the key simplification)

This is where the Rosenof model pays off hugely. Because V(R) is differentiable with respect to the normalized matchup gaps μ_{c,o}, and each player's contribution to team totals is linear in whether they start, you can compute:

```
∂EW/∂(my_total_c) = Σ_o φ(z_{c,o}) / (σ_c √2)    [compute_ew_gradient]
```

This gradient tells you the **marginal value of one unit of category c** to your win total. This is the central object for evaluating all roster moves:

- **Player marginal value** = Σ_c (∂EW/∂total_c) × (player's contribution to category c)
- **Swap value** = marginal_value(player_in) - marginal_value(player_out)

This is the existing MEW (Marginal Expected Wins), and it's the right first-order tool.

## 3. What's Missing from the Problem Statement

### 3a. The Lineup Assignment Problem Is Entangled with Roster Construction

The problem statement says "figure out which swaps will increase my win probability." But a swap's value depends on *who starts after the swap*, which depends on the MILP lineup assignment. The current code handles this with heuristic perturbation (swap one player, re-solve lineup), but this creates a chicken-and-egg problem:

- You can't evaluate a swap without knowing the post-swap lineup
- You can't determine the post-swap lineup without knowing the swap

**The fix:** Treat lineup assignment as an inner optimization that's called as a subroutine. The outer loop proposes roster changes; the inner loop computes optimal lineup and team totals for any given roster. This is already how `compute_optimal_lineup` works, but the MILP overhead makes iterating over many candidate swaps expensive.

**Key insight:** For the inner problem (lineup assignment), you don't need a full MILP. It's a **weighted bipartite matching** problem (players → slots), solvable in O(n³) with the Hungarian algorithm or even faster with specialized assignment solvers. This is dramatically cheaper than a general MILP.

### 3b. Sequential Dependencies Between Swaps

The current MILP treats roster construction as a single big decision: "what's the optimal 28-player roster from the entire candidate pool?" But in reality, you make swaps **sequentially**, and each swap changes the state:

1. After FA swap #1, your roster changes, which changes which FA swap #2 is optimal
2. Your waiver priority may change after a claim
3. After a trade, the opponent's roster changes, which changes what other trades are available

**This is a sequential decision problem**, not a one-shot optimization. The MILP gives you the destination but not the path.

### 3c. Opponent Modeling Is Missing

Trades require the opponent to accept. The current fairness model (FV within 10-15% tolerance) is necessary but not sufficient. You need to model:

1. **What does the opponent value?** (Their category weaknesses, their positional needs)
2. **What would improve the opponent's EW?** (Symmetric to your own analysis)
3. **Is the trade mutually beneficial in EW terms?** (Both sides gain EW — a true surplus exists)

The Nash Bargaining Solution is the right framework here: the trade should maximize the **product of surplus** for both sides, where surplus = (EW_after_trade - EW_without_trade) for each party.

### 3d. Projection Uncertainty Isn't Propagated Properly

The Monte Carlo simulation in `win_model.py` models player-level uncertainty well, but this uncertainty isn't used in the optimization. The optimizer uses point projections as if they're certain. This leads to:

- **Overvaluing volatile players** who project well but have huge variance
- **Undervaluing stable players** who provide reliable but moderate production
- **Ignoring the value of diversification** across injury risk

In portfolio theory terms: you're optimizing expected return while ignoring risk. You need **risk-adjusted returns**.

### 3e. The "How Much Better Can We Get" Ceiling

The problem statement doesn't distinguish between:
1. Roster is 95th percentile optimal — marginal improvements are tiny, focus on bench depth
2. Roster is 60th percentile optimal — major upgrades available, aggressive trading warranted

The gradient magnitudes tell you this: if ∂EW/∂total_c is near zero for all c, you're near a local maximum. If some gradients are large, there's room to improve.

## 4. The Unified Framework: Marginal Swap Value (MSV)

Here's the mathematical framework I recommend. The key insight: **every roster move is a swap, and all swaps can be evaluated on a common currency.**

### 4a. Define a Unified Swap

A **swap** is a pair (out_set, in_set) where:
- `out_set ⊂ R₀` (players leaving your roster)
- `in_set` (players entering your roster)
- `|out_set| = |in_set|` (roster size preserved)

- **FA swap**: out_set = {a}, in_set = {f} where f is a free agent. Cost = 0.
- **Trade**: out_set = {a₁,...,aₖ}, in_set = {b₁,...,bₖ} where all bᵢ are on opponent j's roster. Cost = f(opponent_acceptance_probability).

### 4b. The Value Function

For any roster R, define:

```
EW(R) = μ_T(S*(R))
```

where S*(R) is the optimal starting lineup for roster R (solved via bipartite matching, not general MILP).

The **Marginal Swap Value** of swap (out, in) is:

```
MSV(out, in) = EW(R₀ - out + in) - EW(R₀)
```

### 4c. First-Order Approximation (The Gradient Trick)

Computing MSV exactly requires re-solving the lineup assignment for each candidate swap. With hundreds of candidate FAs and trade targets, this is expensive.

The **first-order approximation** avoids this:

```
MSV(out, in) ≈ Σ_c gradient_c × Δtotal_c(out, in)
```

where `gradient_c = ∂EW/∂total_c` is the EW gradient (already computed by `compute_ew_gradient`), and `Δtotal_c` is the change in team total from the swap.

For counting stats: `Δtotal_c = stat_c(player_in) - stat_c(player_out)`
For ratio stats: `Δtotal_c` requires the linearization already used (PA-weighted OPS shift, IP-weighted ERA/WHIP shift).

**When this approximation breaks down:**
- When the swap changes who starts (e.g., the incoming player displaces a different starter than the one being dropped)
- When the swap is large enough that the gradient changes significantly
- For bench players (gradient = 0 since they don't start, but they have option value)

**Fix:** Use the first-order approximation as a fast **screening** pass to identify the top ~50 candidate swaps, then compute exact MSV for those 50. This is O(N) for screening + O(50 × lineup_solve) for exact evaluation.

### 4d. Bench Value as Option Premium

For bench players, define:

```
BV(b) = Σ_{s ∈ starters} P(s_injured) × max(0, MSV_exact({s}, {b}) - MSV_exact({s}, {best_FA(s)}))
```

This is the **expected value of having b on the bench vs. having to pick up the best available FA** when starter s goes down. Key refinements over the current formula:

1. **It's position-aware**: a bench SS is worth more if your starting SS is fragile
2. **It accounts for the FA pool**: bench value is relative to what's available for free
3. **It can be negative for opportunity cost**: a bench spot occupied by a bad player prevents you from carrying a better option

### 4e. The Combined Objective

For any candidate swap (out, in), the **total value** is:

```
Total_Value(out, in) = MSV_starters(out, in) + ΔBV(out, in) - transaction_cost(out, in)
```

where:
- `MSV_starters` = direct impact on expected wins from starting lineup changes
- `ΔBV` = change in total bench option value
- `transaction_cost` = 0 for FA swaps; for trades, some function of the surplus you must share

For trades, the transaction cost can be modeled as:

```
trade_cost = max(0, -MSV_opponent(in_for_them, out_for_them))
```

i.e., you can't execute a trade that makes the opponent worse off in EW terms. This replaces the crude FV-matching heuristic with genuine mutual-benefit evaluation.

## 5. Implementation Architecture

### Phase 1: Gradient Screening (fast, O(N))

```python
def screen_all_swaps(players, my_roster, gradient, starters):
    """Score every non-rostered player by first-order MSV."""
    for each player p not on my roster:
        # Which starter would p replace?
        slot = best_slot_for(p)
        worst_starter = worst_starter_at(slot)
        delta = linearized_stat_delta(p, worst_starter)
        msv_approx = dot(gradient, delta)
        # Store (player, slot, msv_approx)
```

### Phase 2: Exact Evaluation (expensive, O(K))

```python
def evaluate_top_k_swaps(top_candidates, my_roster, projections, opponent_totals):
    """Exact MSV for top K candidates."""
    for each (player_in, player_out) in top_candidates:
        new_roster = my_roster - {player_out} + {player_in}
        new_starters = solve_assignment(new_roster, projections)  # Hungarian, not MILP
        new_totals = compute_totals(new_starters, projections)
        msv_exact = ew(new_totals) - ew(current_totals)
```

### Phase 3: Trade Search (combinatorial but bounded)

```python
def find_mutually_beneficial_trades(my_roster, opponent_rosters, projections):
    """Find trades where both sides gain EW."""
    for each opponent j:
        # What do I want from j? (top MSV players on j's roster)
        # What can I offer j? (players that improve j's EW)
        # Enumerate small swaps (1-for-1, 2-for-2) that are mutually beneficial
        # Rank by my_msv, filter by opponent_msv > 0
```

### Phase 4: Sequencing

```python
def plan_roster_moves(fa_swaps, trades, my_roster):
    """Order moves to maximize total improvement."""
    # Greedy: execute highest-MSV swap first, recompute, repeat
    # This works because EW is approximately submodular in roster composition
```

## 6. Critical Edge Cases

### 6a. The Ratio Stat Trap
Adding a high-OPS hitter doesn't help if they have few PA — the PA-weighting dilutes. Similarly, adding a low-IP pitcher with a 2.50 ERA can actually hurt your ERA if they replace a higher-IP pitcher with a 3.20 ERA. The linearization handles this correctly, but it needs to be propagated through the entire pipeline.

### 6b. Roster Composition Constraints
You can't have fewer than 10 pitchers or 12 hitters. A trade that sends 2 pitchers for 2 hitters might violate this. Every candidate swap must be checked against composition bounds **before** evaluation.

### 6c. The "Punt" Decision
Sometimes the best strategy is to punt a category entirely (abandon SV, invest in other categories). The MILP can discover this, but the gradient approach won't because the gradient near zero wins in a category is positive (you can still flip some matchups). The Rosenof insight is that punting hurts variance-of-fantasy-points, which hurts win probability in roto. But in a 7-team league this effect may differ from a 12-team league.

### 6d. Multi-Position Player Interactions
A player eligible at 2B and SS affects two position markets simultaneously. Trading away a 2B/SS player has different value depending on whether you have depth at both positions or are thin at one. The lineup assignment (inner optimization) handles this automatically, but the gradient screening doesn't.

### 6e. The In-Season Dynamic
Projections change over the season. A player projected for 500 PA in March may be projected for 300 PA by July (injury, demotion). The optimization needs to be re-run as projections update, and the concept of "remaining season value" vs "full season value" matters.

### 6f. Trade Feasibility is Non-Convex
The set of feasible trades is not convex — you can't "partially" trade a player. And mutual benefit doesn't compose: trade A and trade B with the same opponent may each be mutually beneficial, but doing both may not be (the second trade changes both teams' state). You must evaluate trade combinations, not just individual trades.

## 7. What Mathematical Tricks Simplify Everything

### Trick 1: Gradient Pre-computation
The EW gradient `∂EW/∂total_c` only depends on your current totals and opponent totals. Compute it once, then screen all candidates in O(1) per player. This is already done in `compute_ew_gradient` — it's the right approach.

### Trick 2: Hungarian Assignment Instead of MILP
The lineup assignment problem is a **minimum-cost bipartite matching**: players on one side, slots on the other, edge weights = -FV for eligible assignments. The Hungarian algorithm solves this in O(n³) guaranteed, vs. MILP which can be much slower and has no polynomial guarantee. For a 28-player roster with ~18 slots, n is small enough that this is microseconds.

### Trick 3: Rank-Based MSV Rather Than Absolute
Instead of computing MSV for every FA, compute it only for players who **dominate** the worst starter at each slot. Player p dominates starter s at slot k if p is eligible for slot k and MSV_approx(p replacing s) > 0. This prunes the search space dramatically.

### Trick 4: Category-Decomposition of Value
EWA = Σ_c EWA_c, where EWA_c is the per-category expected wins added. This decomposition (already computed in `compute_ew_breakdown`) lets you:
- Identify **which categories** a player helps/hurts
- Match trade partners by **complementary category needs** (I'm weak in SV, you're weak in SB → I send SB producers for SV producers)
- Avoid "value traps" where a player has high FV but helps categories you don't need

### Trick 5: Unified Transaction Framework
Model both FA and trade swaps as elements of the same set, differing only in transaction cost:

```
swap_value(out, in) = MSV(out, in) - cost(out, in)

cost_FA(out, in) = 0                          # free agents are free
cost_trade(out, in) = f(opponent_surplus)      # must share surplus with opponent
```

Then the optimizer just ranks all swaps by `swap_value` and executes greedily. No need for separate "FA optimizer" and "trade engine" — they're the same optimization with different cost functions.

### Trick 6: Simulation for Risk Calibration
Use the existing Monte Carlo simulation **not as the optimizer** but as a **calibration tool**: run 10K simulated seasons with player-level noise, extract the empirical distribution of team totals, and use those to set the σ_c values that feed the analytical model. This gives the best of both worlds: fast analytical gradient for optimization, simulation-calibrated uncertainty for accuracy.

## 8. Concrete Recommendations for the Rewrite

1. **Kill the big MILP in `roster_optimizer.py`.** Replace with gradient screening + exact evaluation of top-K candidates. The current MILP tries to find the optimal 28-player roster from scratch, which is the wrong question — you want to find the best *changes* to the current roster.

2. **Replace the lineup MILP with Hungarian algorithm.** The lineup assignment is a pure assignment problem. Using PuLP for this is like using a nuclear reactor to boil water.

3. **Unify FA and trade evaluation.** One function: `evaluate_swap(out_set, in_set, my_roster, projections, opponent_totals) → (my_msv, delta_bv, cost)`. FA swaps have cost=0. Trade swaps have cost=f(opponent model).

4. **Compute bench value properly as option premium.** For each bench player, compute the injury-weighted conditional EWA improvement over the best FA at each eligible position. This replaces the heuristic BV formula.

5. **Use the category-decomposed gradient as the primary ranking tool.** This is the MEW — it's fast, it's correct to first order, and it directly tells you which categories need investment.

6. **Evaluate trades from the opponent's perspective.** For each candidate trade, compute the opponent's MSV too. Only propose trades where opponent MSV > 0 (mutual benefit). Rank by my MSV, filter by opponent MSV > some threshold.

7. **Iterate greedily.** Execute the highest-value swap, recompute gradient and starters, find the next-best swap. Stop when the best remaining swap has MSV below some threshold (e.g., 0.1 expected wins). This accounts for sequential dependencies naturally.

## 9. Summary

| Concept | Fintech Analogy | Roster Problem |
|---|---|---|
| Portfolio | Collection of assets | Roster of players |
| Expected return | E[portfolio return] | Expected wins (μ_T) |
| Risk | Variance of return | σ_T² (fantasy point variance) |
| Rebalancing | Selling X, buying Y | Dropping player, adding player |
| Transaction cost | Bid-ask spread, taxes | Trade negotiation (must give up value to opponent) |
| Options/insurance | Call option on future asset price | Bench player as insurance against starter injury |
| Gradient/MCTR | Marginal contribution to risk | MEW gradient (∂EW/∂total_c) |
| Diversification | Don't concentrate in one sector | Don't punt categories; spread across categories |

The core pipeline should be:

1. **State**: Current roster + opponent totals + projections
2. **Gradient**: Compute ∂EW/∂total_c → tells you category needs
3. **Screen**: Score all available players by first-order MSV → top candidates
4. **Evaluate**: Exact MSV for top candidates (via Hungarian lineup reassignment)
5. **Trade search**: For each opponent, find mutually beneficial swaps
6. **Rank**: All moves on common MSV currency, net of transaction cost
7. **Execute**: Greedily, re-evaluating after each move

This eliminates the need for the big MILP, unifies FA and trade analysis, properly values bench players, and runs fast enough to be interactive.

## 10. Key Conclusions

**First, the current architecture is solving the wrong problem.** The big MILP asks "what's the globally optimal 28-player roster?" but you actually need "what's the best *change* to my current roster?" These are fundamentally different questions. The former is a one-shot combinatorial optimization; the latter is sequential portfolio rebalancing. The rebalancing formulation is both more correct (it respects transaction constraints) and more computationally tractable (gradient screening + exact top-K evaluation).

**Second, trades and FA moves should be unified.** They are the same mathematical operation — a swap of `out_set` for `in_set` — differing only in transaction cost. FA swaps are free; trades cost surplus-sharing with the opponent. A single `evaluate_swap` function with a cost parameter replaces both `roster_optimizer.py` and `trade_engine.py`.

**Third, bench value needs to be treated as option premium, not a heuristic.** The current BV formula (`max(0, FVAR) × Scarcity × P(need)`) doesn't condition on *which* starter is lost or what the FA replacement pool looks like. Real bench value is the expected EWA improvement of having the bench player vs. the best FA, weighted by the probability of needing a replacement at each eligible position. This naturally captures positional scarcity and injury risk without ad-hoc parameters.
