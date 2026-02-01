# New areas of work

## 1. Build playing time estimation

This is a combination of:

- Expected playing time: this should largely be 100% (162) for all players on the team, since fantasy league's are smaller than the number of MLB teams and so every team should be made up entirely of real-life starters, but could use things like this to help estimate (https://www.fangraphs.com/teams/red-sox/depth-chart)
- Expected injury time: need to build a risk profile for every player in the league. Simplest is an empiral lookup table from historical data: conditioned on position and age, what is the shape of the missed_games probability distribution: e.g. for older players it's higher across the board, 1B are probably more reliable than SS, and pitchers have higher tail probabilities (long durations missed) because of Tommy John surgery. Could do research to see whether past reliability predicts this year's reliability, whether to measure by calendar age vs years in service, etc.
  - This should be a subagent research program. I think current SOTA is pretty bad. Can we add player-specific features like spin rates, sprint speed, BMI, etc to implicitly model specific failure modes?

Target outcome is some distributional probabilities for games played (median, maybe p10/p90 or something? Depends how they end up being used downstream.)

## 2. Player valuations should be starter-aware

So far, we have been optimizing for total roster value, assuming that scoring counts all players equally. However, in reality only the players in our starting lineup count towards our scoring, with Reserves not counting. Therefore, when considering the addition of new players (either in free agency/waivers, or in trades) we must consider whether the player being added will be an upgrade over our existing starter, and if not, what value they bring as a reserve. This new roster improvement score (is there an industry-standard name for this?) must consider the following factors:

1. A player's statistical projections (SGP): How valuable the player actually is, in isolation.
2. the context of my league (like our existing EWA): which metrics are close and worth contending in, which are not.
3. My roster's specific needs. This will devalue players who are worse than my existing starter, and increase value of players who may not be all that great but are an improvement over my existing starter.

In other words, it's basically the delta of EWA between a candidate player and a baseline player, which would be chosen automatically: for all players except the target team's starter's, the baseline player is the current starter, and for the starters, it's the best reserve at that position.

We will have to think hard about how to value Reserves. The most obvious way is just the player's relative performance: if they're worth 0.1 less EWA than the starter, then they'd be -0.1. This ignores possible asymmetric valuation strategies for each (e.g. we might prioritize reliability more in a Reserve than a starter), but I'm ok with that.

This new roster improvement score (RIS) should eventually replace EWA in all our reporting metrics (on dashboard and in notebook).

Another agent has already done some thinking about this, documented in the implementation specs. They are not very smart. I need you to review that work critically and evaluate what it has done well and what it has done poorly: look for errors in the logic and math (are we actually implementing it correctly), errors in complexity, verbosity/ambiguity/clarity, and so on. Our goal is to have a clear path to go from projections to RIS.

## 3. General Approach: Focus on Core Dataset

I kinda think we should consider our Player database as the core product of this whole project. We've been focusing too much on the outcomes, but if we get the database nailed and the right metrics in it, everything else is basically a trivial downstream op from there

For every player, we already have:

- individual metadata: age, position, which fantasy team they are on
- Projected performance: STEAMER, ATC projections
- Projected reliability: derived from discussion above.

Based on these factors, we can calculate an RIS vector for every fantasy team across all players pretty trivially, as explored above:

- For each team: calculate current starters' values, bench values
- Run a two-stage MILP or something to give every other player conditioned on that team's particular state.
- For a team's current starters, their RIS scores are sorta definitionally zero/null, but maybe we report their marginal value (inverse of their replacement's value: if my reserve 2B is worth -0.1 RIS, then my starter is worth -1 * -0.1 RIS). Maybe this should just be a separate column to keep the score's purity.

The resulting gold table should have schema like:

- Metadata: IDs, name, real team, position, age, fantasy team
- Stat projections (from either ATC or STEAMER or both)
- projected reliability metrics (prob of missing X games, for a few values of X?)
- SGP
- RIS team 1
- ...
- RIS team N

With a table like that, everything else becomes trivial:

- The whole dashboard is just exposing various slices of it: the main page of the dashboard should basically just be this complete table with a bunch of filters.
- Free agency becomes a process of trying to get all the the players with positive RIS who aren't on a team already

If we can formalize this mathematically, then player aquisition, in trade or FA (FA is implicitly a trade, since roster spots are conserved, so a player would be dropped if one is added), becomes pretty trivial too: it's just an operation on the matrix of [players, team-specific RIS scores]: in the subset of two team's players, find mutually beneficial subsets, where

player | team | ris_team_1     | ris_team_2      | net
p1     | t1   | 0.1 (marginal) | 0.9             | 0.8
p2     | t2   | 1.0            | 0.2 (marginal)  | -0.8

If those [t1, t2] traded [p1, p2], then:

- t1 loses 0.1 and gains 0.9 = 0.8 net RIS
- t2 loses 0.2 and gains 1.0 = 0.8 net RIS

For free agency, it's the exact same thing, just without team 2 involved: we should just only have the highest-ranked (by RIS) available players rostered.

How do we formalize this mathematically? I feel like it should not be an unbounded problem; it's basically just an energy minimization on that [players, team_ris] matrix.

## 4. Waiver priorities

Given Fantrax scoring (as proxy for public value estimate) and our (internal) SGP/EWA/RIS estimates, optimize waiver order: who should I take early because other people will want them vs. who can I wait to take since they'll probably still be available?

## 5. Thinking about prospects (minor leaguers)

This is the one thing we are totally ignoring right now. I think we should basically value every prospect at zero unless there's real noise about them being legitimately generational. As discussed in the README, the bar they have to clear to be a positive RIS player is so high that they're basically worthless, so they should only be used as trade bait I think

## 6. Eventually

Think again about how to treat the dynastic component of this league. I'm not convinced that there's ever really a time to punt a season, and therefore prioritizing future value in trades is almost never optimal. However, it might be a good idea anyways to put some estimates on future performance: simple aging curves, estimates for how reproducible their current performance level is next year (e.g. Cal Raleigh probably had an outlier season in 2025 that he's unlikely to reproduce). I suppose this could also guide trade discussions by improving my sense of when to sell high on someone.

It might be interesting to simulate how this strategy (never punt, never think about future value) would play out over various league-engagement timescales: if I'm in the league for one year, then it's definitely not worth thinking about FV; if I'm in the league for 30 years, then it definitely (??) is. What is the shape of the space between those? Probably some function of player career half-lives: if I'm in the league for less than the half life of my stars (say, 5 years), I can ride one set of core guys the whole time, but if it's longer, then I have to turn over the upper end of my roster, which is very hard to do without a. trading present value for future value, b. focusing hard on prospects, or c. fleecing someone.

## 7. Bug fixing and maintenance work

- Inspect nulls in SGP projections. For example, why does Ceddanne Rafaela not show up in the Free Agents Simulate Roster Change section of the dashboard?
- Identify places where the spec is out of sync with the implementation. Be methodical. Our goal here is to make sure that the spec is at parity or ahead of the current implementation everywhere. Wherever you find that they disagree, tell me.
- The table in Free Agents Browser tab in the Dashboard should have all the available metrics. Many more columns.
- If we have access, via the Fantrax API, to Fantrax Scores or player ranks or ADP, those might be useful to use as a baseline for estimating a player's perceived value to the outside world. I've noticed that they tend to differ pretty substantially from SGP, which could lead to some pretty obvious trade opportunities (if we really trust SGP more).
- Build a new directory called "references". In it should be a series of markdown files (and necessary plots to embed in them), each of which methodically and rigorously defines, explains, and justifies the pillars of belief we are building this project on:
  - Raw projections: What they are, why we're using ATC instead of STEAMER, what evidence exists that they've historically performed well, reporting of ballpark error on them.
  - SGP: What it is, why we use it
  - Player aging: how we calculate it, why we think it's good
  - MILP
  - Others?

This will be a crucial form of in-depth documentation.
