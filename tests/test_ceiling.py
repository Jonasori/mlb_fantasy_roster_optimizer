"""
Offline tests for the ceiling screen.
Per AGENTS.md: no classes, no fixtures, no mocking, inline test data.

The fetchers are NOT tested — they need a FanGraphs member cookie and live
Savant leaderboards. What is tested is the logic that would silently produce a
wrong board: our own percentile computation, the two-part screen, and the
ownership join, whose failure mode is a rostered player reading as available.
"""

import numpy as np
import pandas as pd

from data_prep.ceiling import (
    add_ceiling_score,
    add_eligibility,
    add_now_value,
    add_ownership,
    add_positional_replacement,
    add_tool_percentiles,
    compare_at_position,
)
from optimizer.config import MY_TEAM_NAME


def _tool_frame() -> pd.DataFrame:
    """Six hitters spanning the archetypes the screen has to separate."""
    return pd.DataFrame(
        {
            "Name": [
                "Real Slugger", "Speed Only", "Contact Only",
                "Good Tools Bad Bat", "Unmeasured Prospect", "Filler",
            ],
            "Team": ["NYY"] * 6,
            "player_type": ["hitter"] * 6,
            "tier1_FV": [8.0, 6.0, 6.0, 1.0, 9.0, -2.0],
            # bat_speed / max_ev / barrel_rate are the CORE (tail-gating) tools;
            # sprint_speed is not.
            "bat_speed":    [78.0, 69.0, 70.0, 77.0, np.nan, 71.5],
            "max_ev":       [116.0, 103.0, 104.0, 115.0, np.nan, 106.0],
            "barrel_rate":  [18.0, 3.0, 4.0, 16.0, np.nan, 6.0],
            "sprint_speed": [27.0, 30.5, 27.5, 26.0, np.nan, 27.2],
            "fb_velo": [np.nan] * 6,
            "whiff_rate": [np.nan] * 6,
        }
    )


def _scored() -> pd.DataFrame:
    """The frame after both scoring stages, with a pitcher stub for symmetry."""
    frame = _tool_frame()
    pitchers = pd.DataFrame(
        {
            "Name": ["Ace", "Soft Tosser"],
            "Team": ["NYY", "NYY"],
            "player_type": ["pitcher", "pitcher"],
            "tier1_FV": [7.0, 5.0],
            "bat_speed": [np.nan, np.nan],
            "max_ev": [np.nan, np.nan],
            "barrel_rate": [np.nan, np.nan],
            "sprint_speed": [np.nan, np.nan],
            "fb_velo": [99.0, 89.0],
            "whiff_rate": [35.0, 18.0],
        }
    )
    frame = pd.concat([frame, pitchers], ignore_index=True)
    # The horizon columns are set by hand rather than by running add_now_value:
    # these eight rows exist to exercise the STAR bar, and every one of them is
    # a prospect so that the production bar cannot quietly rescue anybody. The
    # major-leaguer branch has its own test below.
    frame["horizon"] = "prospect"
    frame["now_value"] = np.nan
    frame["replacement_now"] = np.nan
    frame["replacement_slot"] = pd.NA
    frame["now_vs_replacement"] = np.nan
    return add_ceiling_score(
        add_tool_percentiles(frame), fv_bar=4.0, tool_pct_bar=0.75, core_pct_bar=0.50
    )


def test_percentiles_are_computed_within_player_type():
    out = add_tool_percentiles(_scored().drop(columns=["pct_bat_speed"], errors="ignore"))
    hitters = out[out["player_type"] == "hitter"]
    # 6 hitters: the best bat speed must be the top of the HITTER population,
    # not of the 8-row frame. rank(pct=True) on 6 non-null values gives 1.0.
    top = hitters.loc[hitters["Name"] == "Real Slugger"].iloc[0]
    assert top["pct_bat_speed"] == 1.0, (
        f"pct_bat_speed is {top['pct_bat_speed']} for the hardest-swinging "
        f"hitter; expected 1.0 ranked within the hitter population."
    )
    ace = out.loc[out["Name"] == "Ace"].iloc[0]
    assert ace["pct_fb_velo"] == 1.0, (
        f"pct_fb_velo is {ace['pct_fb_velo']} for the hardest thrower; pitcher "
        f"tools must be ranked within the pitcher population."
    )
    assert pd.isna(ace["pct_bat_speed"]), (
        f"pct_bat_speed is {ace['pct_bat_speed']} for a pitcher; a tool the "
        f"player type does not have must be NaN, not a percentile."
    )


def test_unmeasured_player_is_not_zeroth_percentile():
    out = add_tool_percentiles(_tool_frame())
    row = out.loc[out["Name"] == "Unmeasured Prospect"].iloc[0]
    assert row["n_tools"] == 0, f"n_tools is {row['n_tools']}, expected 0"
    for col in ("pct_bat_speed", "pct_max_ev", "pct_best", "pct_core"):
        assert pd.isna(row[col]), (
            f"{col} is {row[col]} for a player with no Savant row; must be NaN. "
            f"A 0th percentile is indistinguishable from a genuinely slow bat "
            f"and would be an invented measurement."
        )
    assert pd.isna(row["best_tool"]), (
        f"best_tool is {row['best_tool']} for an unmeasured player; expected NA."
    )


def test_pct_best_is_max_not_mean():
    out = add_tool_percentiles(_tool_frame())
    row = out.loc[out["Name"] == "Speed Only"].iloc[0]
    assert row["best_tool"] == "sprint_speed", (
        f"best_tool is {row['best_tool']}; the speed-only player's loudest tool "
        f"is his legs. pct_best must be a MAX over tools — averaging them "
        f"measures how well-rounded he is, which is an EV question, not a tail "
        f"question."
    )
    assert row["pct_best"] == 1.0, (
        f"pct_best is {row['pct_best']}, expected 1.0 (fastest of six hitters)."
    )


def test_screen_rejects_one_trick_players_with_a_reason():
    out = _scored().set_index("Name")

    passed = out.loc["Real Slugger"]
    assert passed["screen_pass"], (
        f"Real Slugger failed the screen: {passed['screen_reason']}. He has "
        f"both a top-quartile core tool and tier1_FV above the bar."
    )
    # screen_reason now names the bar for passers too: "accepted" is no more
    # actionable than "rejected" when two different bars are in play.
    assert "star" in passed["screen_reason"], (
        f"a passing prospect's reason {passed['screen_reason']!r} does not name "
        f"the star bar it cleared. The reason must say WHICH bar was applied."
    )

    # The whole point of the core-tool cut: pct_best looks elite for both of
    # these, and both must still fail.
    for name, expect in (("Speed Only", "power"), ("Contact Only", "power")):
        row = out.loc[name]
        assert not row["screen_pass"], (
            f"{name} passed the screen with pct_best={row['pct_best']} and "
            f"pct_core={row['pct_core']}; a player whose only real tool is not a "
            f"tail-gating one must be rejected."
        )
        assert expect in row["screen_reason"], (
            f"{name} was rejected for {row['screen_reason']!r}, which does not "
            f"mention the missing {expect} tool. The reason has to be actionable."
        )

    tools_no_bat = out.loc["Good Tools Bad Bat"]
    assert not tools_no_bat["screen_pass"], (
        "a player with elite tools but a tier1_FV of 1.0 must fail: in a league "
        "this shallow a median outcome is free."
    )
    assert "Tier 1" in tools_no_bat["screen_reason"], (
        f"expected a Tier-1 reason, got {tools_no_bat['screen_reason']!r}"
    )

    unmeasured = out.loc["Unmeasured Prospect"]
    assert not unmeasured["screen_pass"], (
        "an unmeasured player must fail the screen, not pass on Tier 1 alone."
    )
    assert "Savant" in unmeasured["screen_reason"], (
        f"expected the unmeasured reason to name the missing source, got "
        f"{unmeasured['screen_reason']!r}"
    )
    assert pd.isna(unmeasured["ceiling_score"]), (
        f"ceiling_score is {unmeasured['ceiling_score']} for a player with no "
        f"tool measurement; it must be NaN rather than a plausible number "
        f"invented from tier1_FV alone."
    )


def test_ceiling_score_is_monotone_in_both_tiers():
    out = _scored().set_index("Name")
    assert out.loc["Real Slugger", "ceiling_score"] > out.loc["Filler", "ceiling_score"], (
        "a better projection with better tools must score higher"
    )
    # Same tier1_FV (6.0), different core tools: Contact Only has slightly
    # better core percentiles than Speed Only, so it must not rank below it on
    # the tool term alone.
    assert (
        out.loc["Contact Only", "pct_core"] > out.loc["Speed Only", "pct_core"]
    ), "pct_core must order the two one-trick profiles by their power tools"


def test_unmatched_player_is_unknown_not_free_agent():
    """The single most dangerous silent failure: an owned player reading as available."""
    players = pd.DataFrame(
        {
            "Name": ["Shohei Ohtani", "Shohei Ohtani", "Nobody In League", "Free Guy"],
            "Team": ["LAD", "LAD", "LAD", "LAD"],
            "player_type": ["hitter", "pitcher", "hitter", "hitter"],
        }
    )
    # Fantrax spells a split two-way player with our -H/-P suffix, and in this
    # league his two halves are owned by DIFFERENT teams.
    fantrax = pd.DataFrame(
        {
            "name": ["Shohei Ohtani-H", "Shohei Ohtani-P", "Free Guy"],
            "player_type": ["hitter", "pitcher", "hitter"],
            "mlb_team": ["LAD", "LAD", "LAD"],
            # MY_TEAM_NAME rather than a literal: the mine/owned split is read
            # from config.json, and hardcoding the team name here would make this
            # test fail the day the league is renamed.
            "owner": [MY_TEAM_NAME, "Some Other Team", None],
            "Position": ["UT", "SP", "OF"],
            "age": [32, 32, 25],
            "minors_eligible": [False, False, False],
            "pct_rostered": [100.0, 100.0, 4.0],
            "fantrax_id": ["a", "b", "c"],
        }
    )
    out = add_ownership(players, fantrax).set_index(["Name", "player_type"])

    assert out.loc[("Nobody In League", "hitter"), "ownership"] == "UNKNOWN", (
        f"an unmatched player got ownership "
        f"{out.loc[('Nobody In League', 'hitter'), 'ownership']!r}; it must be "
        f"UNKNOWN. Defaulting to 'free agent' invents availability and is how a "
        f"rostered player ends up on a pickup list."
    )
    assert out.loc[("Free Guy", "hitter"), "ownership"] == "free agent", (
        "a player present in the Fantrax pool with a null owner IS a free agent"
    )
    # Suffix handling: the two sides must not swap owners.
    assert out.loc[("Shohei Ohtani", "hitter"), "ownership"] == "mine", (
        f"Ohtani's hitter side reads "
        f"{out.loc[('Shohei Ohtani', 'hitter'), 'ownership']!r}; Fantrax stores "
        f"him as 'Shohei Ohtani-H', so the join must strip the suffix and keep "
        f"the two sides apart."
    )
    assert out.loc[("Shohei Ohtani", "pitcher"), "ownership"] == "owned", (
        f"Ohtani's pitcher side reads "
        f"{out.loc[('Shohei Ohtani', 'pitcher'), 'ownership']!r}; his halves are "
        f"owned by different teams and must not collapse onto one roster spot."
    )


def test_does_not_mutate_input():
    frame = _tool_frame()
    before = list(frame.columns)
    add_tool_percentiles(frame)
    assert list(frame.columns) == before, (
        "add_tool_percentiles mutated its input; it must copy first (AGENTS.md)."
    )


# ── Change 1: position eligibility ────────────────────────────────────────


def test_eligibility_is_every_listed_position_not_just_the_first():
    """A board that reads only the first slot drops the manager's only 2B."""
    players = pd.DataFrame(
        {
            "Name": ["Utility Man", "Outfielder", "DH Only", "Unmatched"],
            "Position": ["2B,SS", "OF", "UT", None],
        }
    )
    out = add_eligibility(players)
    slots = dict(zip(out["Name"], out["eligible_slots"]))

    assert slots["Utility Man"] == frozenset({"2B", "SS", "UTIL"}), (
        f"'2B,SS' parsed to {sorted(slots['Utility Man'])}; a player is eligible "
        f"at EVERY comma-separated slot (plus UTIL), not just the first."
    )
    assert "OF" in slots["Outfielder"] and "2B" not in slots["Outfielder"], (
        f"'OF' parsed to {sorted(slots['Outfielder'])}"
    )
    assert slots["DH Only"] == frozenset({"UTIL"}), (
        f"'UT' parsed to {sorted(slots['DH Only'])}; a DH earns the UTIL slot "
        f"and no fielding position."
    )
    assert slots["Unmatched"] == frozenset(), (
        f"a player with no Fantrax Position got {sorted(slots['Unmatched'])}; "
        f"eligibility must be EMPTY rather than guessed — a guessed slot is how "
        f"an unmatched row acquires a roster spot it cannot fill."
    )


def _replacement_frame() -> pd.DataFrame:
    """Six 2B-eligible players with now_value 6..1, plus one UNKNOWN ringer."""
    return pd.DataFrame(
        {
            "Name": ["Best", "Second", "Third", "Fourth", "Fifth", "Sixth", "Ghost"],
            "Position": ["2B"] * 6 + ["2B"],
            "ownership": ["mine", "owned", "owned", "free agent", "free agent",
                          "free agent", "UNKNOWN"],
            "now_value": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 99.0],
        }
    )


def test_replacement_level_is_the_next_player_without_a_job():
    """With T teams and S slots, replacement is the (T*S + 1)-th best, no more."""
    out = add_positional_replacement(
        add_eligibility(_replacement_frame()), num_teams=2
    ).set_index("Name")

    # 2 teams x 1 starting 2B = 2 players with jobs, so #3 is replacement.
    assert out.loc["Best", "replacement_now"] == 4.0, (
        f"2B replacement came out {out.loc['Best', 'replacement_now']}; with "
        f"num_teams=2 and one 2B slot the first player WITHOUT a job is the 3rd "
        f"best (now_value 4.0)."
    )
    assert out.loc["Best", "now_vs_replacement"] == 2.0, (
        f"surplus is {out.loc['Best', 'now_vs_replacement']}, expected 6.0 - 4.0"
    )
    assert out.loc["Ghost", "replacement_now"] == 4.0, (
        "an UNKNOWN-ownership player must not enter the replacement population "
        "(he is not in this league's universe), though he still gets scored "
        "against it."
    )
    assert out.loc["Sixth", "now_vs_replacement"] == -3.0, (
        f"the worst player's surplus is "
        f"{out.loc['Sixth', 'now_vs_replacement']}, expected 1.0 - 4.0"
    )


def test_multi_position_player_is_counted_at_the_scarcest_slot():
    """Flexibility is worth something: the MIN bar he clears is his bar."""
    frame = _replacement_frame()
    # Add a 3B population that is uniformly WEAKER than the 2B population, so
    # 3B replacement lands below 2B replacement.
    weak = pd.DataFrame(
        {
            "Name": ["W1", "W2", "W3", "Flex"],
            "Position": ["3B", "3B", "3B", "2B,3B"],
            "ownership": ["owned", "free agent", "free agent", "free agent"],
            "now_value": [-1.0, -2.0, -3.0, 0.0],
        }
    )
    out = add_positional_replacement(
        add_eligibility(pd.concat([frame, weak], ignore_index=True)), num_teams=2
    ).set_index("Name")

    assert out.loc["Flex", "replacement_slot"] == "3B", (
        f"the 2B/3B player was priced at "
        f"{out.loc['Flex', 'replacement_slot']!r}; a multi-position player is "
        f"judged at the SCARCEST slot he can fill, which is where he generates "
        f"the most surplus and therefore where he would be deployed."
    )
    assert out.loc["Flex", "replacement_now"] == -2.0, (
        f"3B replacement is {out.loc['Flex', 'replacement_now']}, expected the "
        f"3rd-best 3B (-2.0): 2 teams x 1 slot = 2 players with jobs, and Flex "
        f"is himself in the 3B population."
    )


# ── Change 2: the horizon split ───────────────────────────────────────────


def _ytd_frame() -> pd.DataFrame:
    """A minimal StatsAPI-shaped year-to-date snapshot.

    Two full-time hitters with the SAME OPS but very different power/speed, a
    high-average slap hitter, a bench bat under the volume floor, and two
    pitchers (add_fantasy_value needs variance in both type populations).
    Deliberately carries NO `ops` column — StatsAPI does not publish one, and
    `_ytd_line` has to build it from obp + slg.
    """
    return pd.DataFrame(
        {
            "MLBAMID": [1, 2, 3, 4, 11, 12],
            "name": ["Power Bat", "Singles Hitter", "Balanced",
                     "Bench Guy", "Ace", "Mop Up"],
            "group": ["hitting"] * 4 + ["pitching"] * 2,
            "PA": [550.0, 533.0, 500.0, 40.0, 0.0, 0.0],
            "IP": [0.0, 0.0, 0.0, 0.0, 180.0, 45.0],
            "R": [95.0, 61.0, 75.0, 4.0, 0.0, 0.0],
            "HR": [34.0, 6.0, 18.0, 1.0, 0.0, 0.0],
            "RBI": [98.0, 54.0, 70.0, 5.0, 0.0, 0.0],
            "SB": [8.0, 10.0, 14.0, 0.0, 0.0, 0.0],
            "W": [0.0, 0.0, 0.0, 0.0, 15.0, 2.0],
            "SV": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "SOA": [0.0, 0.0, 0.0, 0.0, 220.0, 30.0],
            "ER": [0.0, 0.0, 0.0, 0.0, 60.0, 30.0],
            "HA": [0.0, 0.0, 0.0, 0.0, 140.0, 55.0],
            "BBA": [0.0, 0.0, 0.0, 0.0, 45.0, 25.0],
            "avg": [0.262, 0.319, 0.281, 0.150, np.nan, np.nan],
            "obp": [0.352, 0.353, 0.348, 0.200, np.nan, np.nan],
            "slg": [0.541, 0.440, 0.470, 0.250, np.nan, np.nan],
        }
    )


def _peak_side() -> pd.DataFrame:
    """The peak-feed side of the join: same ids, plus two players with no MLB."""
    return pd.DataFrame(
        {
            "Name": ["Power Bat", "Singles Hitter", "Balanced", "Bench Guy",
                     "Ace", "Mop Up", "Toolsy Kid", "Lost Veteran"],
            "MLBAMID": pd.array([1, 2, 3, 4, 11, 12, 500, 600], dtype="Int64"),
            "player_type": ["hitter"] * 4 + ["pitcher"] * 2 + ["hitter"] * 2,
            "minors_eligible": [False, False, False, False, False, False,
                                True, False],
        }
    )


def test_horizon_splits_on_current_season_major_league_volume():
    out = add_now_value(_peak_side(), _ytd_frame()).set_index("Name")

    assert out.loc["Power Bat", "horizon"] == "major-leaguer", (
        f"a hitter with 550 PA reads {out.loc['Power Bat', 'horizon']!r}"
    )
    assert out.loc["Ace", "horizon"] == "major-leaguer", (
        f"a pitcher with 180 IP reads {out.loc['Ace', 'horizon']!r}"
    )
    assert out.loc["Bench Guy", "horizon"] == "major-leaguer", (
        "40 PA is still major-league PA: the horizon split is about whether he "
        "is HERE, not about whether the sample is big enough to score."
    )
    assert pd.isna(out.loc["Bench Guy", "now_value"]), (
        f"now_value is {out.loc['Bench Guy', 'now_value']} on 40 PA; below the "
        f"volume floor it must be NULL, not an imputed zero. Zero reads as "
        f"'average', which is a number he never earned."
    )
    assert out.loc["Toolsy Kid", "horizon"] == "prospect", (
        f"a minors-eligible player with no MLB volume reads "
        f"{out.loc['Toolsy Kid', 'horizon']!r}"
    )
    assert out.loc["Lost Veteran", "horizon"] == "unclear", (
        f"a player with no MLB volume who is NOT minors-eligible reads "
        f"{out.loc['Lost Veteran', 'horizon']!r}; 'we do not know what this is' "
        f"must not be reported as 'we checked and he is a prospect'."
    )
    assert out.loc["Toolsy Kid", "now_PA"] == 0.0, (
        "no YTD row means zero major-league PA, which is a counting FACT and "
        "must be filled; the RATES stay null."
    )
    assert pd.isna(out.loc["Toolsy Kid", "now_OPS"]), (
        f"now_OPS is {out.loc['Toolsy Kid', 'now_OPS']} for a player with no "
        f"plate appearances; he has no OPS, and 0.000 is a fabricated line."
    )


def test_ops_is_obp_plus_slg_because_statsapi_has_no_ops_column():
    out = add_now_value(_peak_side(), _ytd_frame()).set_index("Name")
    assert abs(out.loc["Singles Hitter", "now_OPS"] - 0.793) < 1e-9, (
        f"now_OPS is {out.loc['Singles Hitter', 'now_OPS']}, expected "
        f"0.353 + 0.440 = 0.793. StatsAPI publishes no `ops` column at all, so "
        f"a missing-column fallback here would silently zero every hitter's "
        f"rate category."
    )


def test_now_value_joins_on_id_so_two_players_named_max_muncy_stay_apart():
    """The exact bug this join guards: 571970 (LAD, 35) vs 691777 (ATH, 23)."""
    ytd = _ytd_frame()
    ytd.loc[ytd["name"] == "Power Bat", "name"] = "Max Muncy"
    ytd.loc[ytd["name"] == "Bench Guy", "name"] = "Max Muncy"
    peak = _peak_side()
    peak.loc[peak["Name"] == "Power Bat", "Name"] = "Max Muncy"
    peak.loc[peak["Name"] == "Bench Guy", "Name"] = "Max Muncy"

    out = add_now_value(peak, ytd)
    assert len(out) == len(peak), (
        f"the YTD merge fanned {len(peak)} rows out to {len(out)}: two players "
        f"share a name and the join is picking up both."
    )
    by_id = out.set_index("MLBAMID")
    assert by_id.loc[1, "now_PA"] == 550.0, (
        f"id 1's now_PA is {by_id.loc[1, 'now_PA']}, expected his own 550. A "
        f"name join spliced one Max Muncy's season onto the other's row and "
        f"produced a completely fictional recommendation."
    )
    assert by_id.loc[4, "now_PA"] == 40.0, (
        f"id 4's now_PA is {by_id.loc[4, 'now_PA']}, expected his own 40."
    )
    assert pd.isna(by_id.loc[4, "now_value"]), (
        "the 40-PA Muncy must keep a NULL now_value; inheriting the other "
        "Muncy's 34 homers is exactly the failure this test exists for."
    )


def test_high_batting_average_does_not_rescue_a_low_power_profile():
    """Arraez: .319 with 6 HR is near-worthless in a 5x5 with OPS, not AVG."""
    out = add_now_value(_peak_side(), _ytd_frame()).set_index("Name")
    singles = out.loc["Singles Hitter"]
    power = out.loc["Power Bat"]
    balanced = out.loc["Balanced"]

    assert singles["now_avg"] > power["now_avg"], (
        "test setup is wrong: the singles hitter must own the best batting "
        "average, otherwise this test proves nothing."
    )
    assert singles["now_value"] < power["now_value"], (
        f"now_value rates the .319 slap hitter ({singles['now_value']:.2f}) at "
        f"or above the .262/34-homer bat ({power['now_value']:.2f}). The league "
        f"scores R/HR/RBI/SB/OPS and does NOT score batting average — a "
        f"now_value that rewards it is weighting the wrong column."
    )
    assert singles["now_value"] < balanced["now_value"], (
        f"the highest average on the board ({singles['now_value']:.2f}) still "
        f"outranks an ordinary regular ({balanced['now_value']:.2f}); a singles "
        f"hitter with no power and no speed has to rate LAST here."
    )
    assert singles["now_value"] < 0.0, (
        f"the singles hitter's now_value is {singles['now_value']:.2f}; scored "
        f"against other regulars a no-power no-speed profile must land below "
        f"the population mean, not above it."
    )


def _horizon_screen_frame() -> pd.DataFrame:
    """Four players with identical tools, differing only in horizon and now."""
    return pd.DataFrame(
        {
            "Name": ["Productive Vet", "Empty Vet", "Toolsy Stash", "Star Prospect"],
            "player_type": ["hitter"] * 4,
            "horizon": ["major-leaguer", "major-leaguer", "major-leaguer",
                        "prospect"],
            "tier1_FV": [2.60, -0.32, 8.0, 8.0],
            "pct_best": [0.99, 0.99, 0.99, 0.99],
            "pct_core": [0.99, 0.99, 0.99, 0.99],
            "best_tool": ["bat_speed"] * 4,
            "n_tools": [4, 4, 4, 4],
            "now_value": [5.43, 2.33, 1.0, np.nan],
            "replacement_now": [4.15, 4.15, 4.15, 4.15],
            "replacement_slot": ["2B", "2B", "2B", "2B"],
            "now_vs_replacement": [1.28, -1.82, -3.15, np.nan],
        }
    )


def test_productive_major_leaguer_passes_without_a_breakout_option():
    """Otto Lopez: tier1_FV 2.60 is a hold with no option value, not a reject."""
    out = add_ceiling_score(_horizon_screen_frame(), fv_bar=4.0).set_index("Name")

    vet = out.loc["Productive Vet"]
    assert vet["screen_pass"], (
        f"a major leaguer producing 5.43 against a 4.15 replacement level "
        f"failed the screen: {vet['screen_reason']}. Holding him costs NOTHING "
        f"— his median is free, so the prospect star bar must not apply."
    )
    assert vet["screen_bar"] == "now|star", (
        f"screen_bar is {vet['screen_bar']!r}; a major leaguer is held to "
        f"max(production, breakout)."
    )
    assert "production" in vet["screen_reason"], (
        f"the reason {vet['screen_reason']!r} does not say which bar he cleared."
    )
    assert "NO option value" in vet["screen_reason"], (
        f"the reason {vet['screen_reason']!r} must state that there is no "
        f"breakout to option on — that is what makes him displaceable by a "
        f"higher-ceiling alternative rather than simply good."
    )

    # Arraez-shaped: real playing time, real average, no power, no tail.
    empty = out.loc["Empty Vet"]
    assert not empty["screen_pass"], (
        f"a major leaguer BELOW replacement with a NEGATIVE peak FV passed: "
        f"{empty['screen_reason']}. Production above replacement is the bar, "
        f"not production at all."
    )
    assert "FAILED BOTH" in empty["screen_reason"], (
        f"expected a reason naming both misses, got {empty['screen_reason']!r}"
    )

    stash = out.loc["Toolsy Stash"]
    assert stash["screen_pass"] and "breakout" in stash["screen_reason"], (
        f"an unproductive major leaguer with a real peak must pass as a STASH "
        f"— the option is free because he is already here. Got "
        f"{stash['screen_pass']} / {stash['screen_reason']!r}"
    )


def test_prospect_keeps_the_star_bar_and_gets_no_production_credit():
    frame = _horizon_screen_frame()
    # Give the prospect a huge (impossible) now_value: it must be ignored.
    frame.loc[frame["Name"] == "Star Prospect", "now_value"] = 99.0
    frame.loc[frame["Name"] == "Star Prospect", "now_vs_replacement"] = 94.85
    frame.loc[frame["Name"] == "Star Prospect", "tier1_FV"] = 1.0
    out = add_ceiling_score(frame, fv_bar=4.0).set_index("Name")

    prospect = out.loc["Star Prospect"]
    assert prospect["screen_bar"] == "star", (
        f"screen_bar is {prospect['screen_bar']!r} for a prospect; his median "
        f"costs four or five years of a roster slot, so only the tail pays."
    )
    assert not prospect["screen_pass"], (
        f"a prospect with tier1_FV 1.0 passed on production: "
        f"{prospect['screen_reason']}. Production cannot apply to a player who "
        f"is not here — the whole point of the split is that his holding cost "
        f"is years, not zero."
    )
    assert "star only" in prospect["screen_reason"], (
        f"the reason {prospect['screen_reason']!r} must name the star bar and "
        f"why it applies."
    )


# ── Change 3: the trade-off, stated rather than resolved ───────────────────


def _compare_frame() -> pd.DataFrame:
    """Otto Lopez against three 2B alternatives, real numbers from 2026-08-21."""
    frame = pd.DataFrame(
        {
            "Name": ["Otto Lopez", "Colt Keith", "Brice Matthews", "Luis Arraez"],
            "Position": ["2B,SS", "2B,3B", "2B,OF", "1B,2B"],
            "age": [27, 25, 24, 29],
            "player_type": ["hitter"] * 4,
            "ownership": ["mine", "free agent", "free agent", "free agent"],
            "horizon": ["major-leaguer"] * 4,
            "screen_bar": ["now|star"] * 4,
            "screen_pass": [True, True, True, False],
            "now_PA": [538.0, 363.0, 226.0, 533.0],
            "now_IP": [0.0, 0.0, 0.0, 0.0],
            "now_OPS": [0.793, 0.745, 0.582, 0.793],
            "now_value": [5.43, -0.41, -4.31, 2.33],
            "tier1_FV": [2.60, 6.79, 7.78, -0.32],
            "wRC+": [98.4, 120.6, 100.8, 103.7],
        }
    )
    return add_eligibility(frame)


def test_compare_states_both_deltas_and_refuses_to_blend_them():
    board = compare_at_position(_compare_frame(), "Otto Lopez", "2B").set_index("Name")

    keith = board.loc["Colt Keith"]
    assert abs(keith["d_OPS"] + 0.048) < 1e-9, (
        f"d_OPS is {keith['d_OPS']}, expected 0.745 - 0.793 = -0.048"
    )
    assert abs(keith["d_peak wRC+"] - 22.2) < 1e-9, (
        f"d_peak wRC+ is {keith['d_peak wRC+']}, expected 120.6 - 98.4 = 22.2"
    )
    assert keith["trade_off"] == (
        "costs 0.048 OPS of current production, buys 22.2 peak wRC+"
    ), (
        f"trade_off reads {keith['trade_off']!r}. Both halves must be stated in "
        f"the units the categories are scored in; a single blended score would "
        f"price the manager's contention judgement for him."
    )

    assert keith["pareto"], (
        "Colt Keith is the strongest 2B alternative — marginally worse now, far "
        "higher peak — and nothing on the board beats him on both axes."
    )
    assert not board.loc["Brice Matthews", "pareto"], (
        f"Brice Matthews is worse than Keith on BOTH axes (OPS "
        f"{board.loc['Brice Matthews', 'now_OPS']} vs {keith['now_OPS']}, peak "
        f"wRC+ {board.loc['Brice Matthews', 'wRC+']} vs {keith['wRC+']}) and "
        f"must be off the frontier."
    )
    # Arraez is the case that shows why BOTH axes are reported and neither is
    # the verdict. On (OPS, peak wRC+) he looks like a free lateral upgrade —
    # same .793, five more points of peak — because wRC+ ignores stolen bases,
    # which is where Lopez's value lives. tier1_FV scores the actual five
    # categories and says the opposite, and the screen agrees with tier1_FV.
    arraez = board.loc["Luis Arraez"]
    assert arraez["d_tier1_FV"] < 0, (
        f"d_tier1_FV is {arraez['d_tier1_FV']}; a .319 hitter with 6 homers has "
        f"LESS five-category peak value than the incumbent, whatever wRC+ says "
        f"about him."
    )
    assert not arraez["screen_pass"], (
        "the singles hitter must not pass the screen; if the peak-wRC+ column "
        "on this board is ever read as the verdict instead of as one of two "
        "axes, this is the player it gets wrong."
    )


def test_compare_refuses_a_slot_the_league_does_not_have():
    frame = _compare_frame()
    caught = ""
    try:
        compare_at_position(frame, "Otto Lopez", "2nd base")
    except AssertionError as exc:  # noqa: BLE001 - asserting on the message
        caught = str(exc)
    assert "not a configured lineup slot" in caught, (
        f"a typo'd slot produced {caught!r}; it must fail loudly rather than "
        f"return an empty board that reads as 'no alternatives exist'."
    )
