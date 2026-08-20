"""Tests for the join step: cascade matching, volume floors, projection prep.

Self-contained inline data, no network, no real snapshots.
"""

import pandas as pd

from data_prep.build import (
    apply_volume_floors,
    match_rows,
    merge_fantrax,
    merge_market,
    prepare_projections,
)


def _raw_projection_rows() -> pd.DataFrame:
    """A raw projections snapshot: opposite-type stats NULL, no name suffix."""
    return pd.DataFrame(
        [
            # A two-way player appears on BOTH sides with the same ids.
            {
                "Name": "Shohei Ohtani", "Team": "LAD", "player_type": "hitter",
                "PlayerId": "19755", "MLBAMID": 660271, "WAR": 5.0,
                "PA": 600.0, "AB": 520.0, "R": 100.0, "HR": 40.0, "RBI": 95.0,
                "SB": 10.0, "OPS": 1.000,
                "IP": None, "W": None, "SV": None, "SO": None, "ERA": None, "WHIP": None,
            },
            {
                "Name": "Shohei Ohtani", "Team": "LAD", "player_type": "pitcher",
                "PlayerId": "19755", "MLBAMID": 660271, "WAR": 2.0,
                "PA": None, "AB": None, "R": None, "HR": None, "RBI": None,
                "SB": None, "OPS": None,
                "IP": 100.0, "W": 8.0, "SV": 0.0, "SO": 130.0, "ERA": 2.90, "WHIP": 1.00,
            },
            # Accented name, and Team blank (unsigned) -> should become "FA".
            {
                "Name": "Julio Rodríguez", "Team": None, "player_type": "hitter",
                "PlayerId": "25764", "MLBAMID": 677594, "WAR": 4.0,
                "PA": 650.0, "AB": 600.0, "R": 95.0, "HR": 30.0, "RBI": 90.0,
                "SB": 25.0, "OPS": 0.850,
                "IP": None, "W": None, "SV": None, "SO": None, "ERA": None, "WHIP": None,
            },
            # A prospect: real player, negligible projected MLB volume.
            {
                "Name": "Colt Emerson", "Team": "SEA", "player_type": "hitter",
                "PlayerId": "sa3020522", "MLBAMID": 700000, "WAR": 0.1,
                "PA": 5.0, "AB": 4.0, "R": 1.0, "HR": 0.0, "RBI": 1.0,
                "SB": 0.0, "OPS": 0.500,
                "IP": None, "W": None, "SV": None, "SO": None, "ERA": None, "WHIP": None,
            },
            {
                "Name": "Scrub Reliever", "Team": "OAK", "player_type": "pitcher",
                "PlayerId": "99999", "MLBAMID": 700001, "WAR": 0.0,
                "PA": None, "AB": None, "R": None, "HR": None, "RBI": None,
                "SB": None, "OPS": None,
                "IP": 1.0, "W": 0.0, "SV": 0.0, "SO": 1.0, "ERA": 6.00, "WHIP": 2.00,
            },
        ]
    )


def test_prepare_projections_suffixes_and_zero_fills():
    """Both sides of a two-way player survive, distinctly, with no NaN stats."""
    players = prepare_projections(_raw_projection_rows())

    names = set(players["Name"])
    assert "Shohei Ohtani-H" in names and "Shohei Ohtani-P" in names, (
        f"Two-way player must keep both sides distinct, got {sorted(names)}"
    )

    # SO is renamed to the league's category name K.
    assert "K" in players.columns and "SO" not in players.columns, (
        f"Pitcher SO must be renamed to K; columns are {sorted(players.columns)}"
    )

    # The unified MEW formula requires all 12 stats present, 0 for the wrong type.
    stats = ["PA", "R", "HR", "RBI", "SB", "OPS", "IP", "W", "SV", "K", "ERA", "WHIP"]
    assert not players[stats].isna().any().any(), (
        f"No stat may be NaN after prepare; NaNs at\n{players[stats].isna().sum()}"
    )
    hitter = players[players["Name"] == "Shohei Ohtani-H"].iloc[0]
    assert hitter["IP"] == 0.0 and hitter["K"] == 0.0, (
        f"Hitter row must zero-fill pitching stats, got IP={hitter['IP']}, K={hitter['K']}"
    )
    pitcher = players[players["Name"] == "Shohei Ohtani-P"].iloc[0]
    assert pitcher["PA"] == 0.0 and pitcher["HR"] == 0.0, (
        f"Pitcher row must zero-fill hitting stats, got PA={pitcher['PA']}"
    )

    assert players["Team"].isna().sum() == 0, "Blank Team must default to FA"
    assert (players.loc[players["Name"] == "Julio Rodríguez-H", "Team"] == "FA").all()


def test_volume_floor_never_drops_a_rostered_player():
    """A rostered prospect below the volume floor must survive the filter.

    This is the regression that matters for a dynasty league: a minor-league
    prospect's rest-of-season MLB projection is ~0 by definition, so filtering
    on volume before knowing ownership silently deletes the whole farm system.
    """
    players = prepare_projections(_raw_projection_rows())
    players["owner"] = None
    # The prospect is rostered; the scrub reliever is a free agent.
    players.loc[players["Name"] == "Colt Emerson-H", "owner"] = "The Big Dumpers"

    filtered = apply_volume_floors(players)
    survivors = set(filtered["Name"])

    assert "Colt Emerson-H" in survivors, (
        "A ROSTERED player below the volume floor was dropped. Prospects have "
        "near-zero projected MLB volume; the floor must exempt rostered players."
    )
    assert "Scrub Reliever-P" not in survivors, (
        f"An unrostered sub-floor pitcher should be filtered out, got {sorted(survivors)}"
    )
    assert "Shohei Ohtani-H" in survivors, "A full-volume player must never be dropped"


def test_match_rows_prefers_strong_key_then_falls_back():
    """Pass 1 (id) wins; pass 2 (name) only fills what pass 1 missed."""
    left = pd.DataFrame(
        [
            {"id": "1", "name": "alpha"},   # matches on id
            {"id": "",  "name": "beta"},    # no id -> must match on name
            {"id": "9", "name": "gamma"},   # matches on neither
        ]
    )
    right = pd.DataFrame(
        [
            {"id": "1", "name": "alpha", "tag": "by_id"},
            {"id": "",  "name": "beta",  "tag": "by_name"},
        ]
    )
    idx = match_rows([left["id"], left["name"]], [right["id"], right["name"]])
    tags = idx.map(right["tag"])

    assert tags.iloc[0] == "by_id", f"Row with an id must match on it, got {tags.iloc[0]}"
    assert tags.iloc[1] == "by_name", f"Row without an id falls back, got {tags.iloc[1]}"
    assert pd.isna(tags.iloc[2]), f"Unmatchable row must stay NA, got {tags.iloc[2]}"


def test_match_rows_does_not_let_a_weak_key_override_a_strong_one():
    """Once matched by id, a row is never re-matched by name."""
    left = pd.DataFrame([{"id": "1", "name": "shared"}])
    right = pd.DataFrame(
        [
            {"id": "1", "name": "other", "tag": "correct_by_id"},
            {"id": "2", "name": "shared", "tag": "wrong_by_name"},
        ]
    )
    idx = match_rows([left["id"], left["name"]], [right["id"], right["name"]])
    assert idx.map(right["tag"]).iloc[0] == "correct_by_id", (
        "The name pass overrode an id match. Later passes must only consider "
        "rows still unmatched."
    )


def test_market_price_reaches_both_sides_and_the_name_fallback():
    """Both sides of a two-way player get priced, and minor leaguers match by name.

    Value SPLITTING is covered separately by
    test_two_way_player_market_value_is_split_not_double_counted; this checks
    coverage — that neither side is left unpriced and that an Ottoneu row with
    no FanGraphs major-league id still lands via the name pass.
    """
    players = prepare_projections(_raw_projection_rows())
    ottoneu = pd.DataFrame(
        [
            # Priced by FanGraphs id.
            {"name": "Shohei Ohtani", "fg_id": "19755",
             "median_salary": 78.0, "salary_momentum": 3.6},
            # No major-league id (a minor leaguer) -> must match on name.
            {"name": "Colt Emerson", "fg_id": None,
             "median_salary": 3.0, "salary_momentum": 0.5},
        ]
    )
    priced = merge_market(players, {"ottoneu": ottoneu})

    two_way = priced[priced["Name"].str.startswith("Shohei Ohtani")]
    assert len(two_way) == 2 and two_way["market_value"].notna().all(), (
        f"Both sides of a two-way player must be priced, got\n{two_way}"
    )

    prospect = priced[priced["Name"] == "Colt Emerson-H"].iloc[0]
    assert prospect["market_value"] == 3.0, (
        "A minor leaguer with no FanGraphs major-league id must still be priced "
        f"via the name fallback, got {prospect['market_value']}"
    )

    unpriced = priced[priced["Name"] == "Scrub Reliever-P"].iloc[0]
    assert pd.isna(unpriced["market_value"]), (
        f"An unpriced player must be NaN, not 0, got {unpriced['market_value']}"
    )


def test_fantrax_name_corrections_are_applied_during_the_merge():
    """Known Fantrax misspellings must be bridged, not silently unmatched.

    The raw snapshot keeps Fantrax's own spelling, so the correction has to
    happen in the merge. "Logan OHoppe" is missing an apostrophe; without the
    correction he simply never matches his projection row.
    """
    from data_prep.fantrax_api import FANTRAX_NAME_CORRECTIONS

    fantrax_spelling, fangraphs_spelling = next(iter(FANTRAX_NAME_CORRECTIONS.items()))

    players = pd.DataFrame(
        [{"Name": f"{fangraphs_spelling}-H", "Team": "LAA", "player_type": "hitter",
          "owner": None, "Position": None, "roster_status": None, "age": None}]
    )
    fantrax = pd.DataFrame(
        [{"name": fantrax_spelling, "mlb_team": "LAA", "player_type": "hitter",
          "Position": "C", "owner": "The Big Dumpers", "roster_status": "active",
          "age": 26}]
    )
    merged = merge_fantrax(players, fantrax)

    assert merged.iloc[0]["owner"] == "The Big Dumpers", (
        f"'{fantrax_spelling}' must match '{fangraphs_spelling}' via "
        f"FANTRAX_NAME_CORRECTIONS, but ownership came back "
        f"{merged.iloc[0]['owner']!r}. The correction dict is not being applied."
    )
    assert merged.iloc[0]["Position"] == "C", "Position should merge through too"


def test_already_suffixed_fantrax_name_is_not_double_suffixed():
    """Fantrax pre-suffixes split two-way players; appending again loses them.

    In this league Ohtani's hitter and pitcher halves are owned by DIFFERENT
    teams, so Fantrax itself ships "Shohei Ohtani-H" / "-P". Appending a second
    suffix yields "Shohei Ohtani-H-H", which matches nothing — silently dropping
    the best player in the league.
    """
    players = prepare_projections(_raw_projection_rows())
    fantrax = pd.DataFrame(
        [
            {"name": "Shohei Ohtani-H", "mlb_team": "LAD", "player_type": "hitter",
             "Position": "UT", "owner": "Team A", "roster_status": "active"},
            {"name": "Shohei Ohtani-P", "mlb_team": "LAD", "player_type": "pitcher",
             "Position": "SP", "owner": "Team B", "roster_status": "reserve"},
        ]
    )
    merged = merge_fantrax(players, fantrax)

    hitter = merged[merged["Name"] == "Shohei Ohtani-H"].iloc[0]
    pitcher = merged[merged["Name"] == "Shohei Ohtani-P"].iloc[0]
    assert hitter["owner"] == "Team A", (
        f"Pre-suffixed hitter row must match, got owner={hitter['owner']!r}"
    )
    assert pitcher["owner"] == "Team B", (
        f"Pre-suffixed pitcher row must match, got owner={pitcher['owner']!r}. "
        f"His two halves can have different owners, so the sides must stay distinct."
    )


def test_two_players_with_the_same_normalized_name_do_not_share_ownership():
    """"José Ramírez" (CLE) and "Jose Ramirez" (DET) are different people.

    They normalize identically, so a many-to-one match hands the second player
    the first's roster spot — inventing a rostered player and inflating a team.
    """
    players = pd.DataFrame(
        [
            {"Name": "José Ramírez-H", "Team": "CLE", "player_type": "hitter",
             "owner": None, "Position": None, "roster_status": None, "age": None},
            {"Name": "Jose Ramirez-H", "Team": "DET", "player_type": "hitter",
             "owner": None, "Position": None, "roster_status": None, "age": None},
        ]
    )
    # Fantrax holds only the CLE star.
    fantrax = pd.DataFrame(
        [{"name": "Jose Ramirez", "mlb_team": "CLE", "player_type": "hitter",
          "Position": "3B", "owner": "Team A", "roster_status": "active",
          "fantrax_id": "01ub6"}]
    )
    merged = merge_fantrax(players, fantrax)
    owned = merged[merged["owner"].notna()]

    assert len(owned) == 1, (
        f"Exactly one player may hold a Fantrax roster spot, got {len(owned)}:\n"
        f"{owned[['Name', 'Team', 'owner']]}"
    )
    assert owned.iloc[0]["Team"] == "CLE", (
        f"The team-qualified match must win over the name-only fallback; "
        f"ownership landed on {owned.iloc[0]['Team']}"
    )


def test_unprojected_rostered_player_has_zero_volume_so_rates_are_inert():
    """An added row must carry zero volume, which makes its stored rates inert.

    Team ratio totals weight OPS by PA and ERA/WHIP by IP, and `add_mew`
    multiplies every ratio term by PA or IP, so zero volume means the stored
    rate can never influence a total. `add_fantasy_value` separately excludes
    zero-volume players from the ratio z-score population — see
    test_fv_ignores_undefined_rates_for_zero_volume_players in test_core.
    """
    players = prepare_projections(_raw_projection_rows())
    fantrax = pd.DataFrame(
        [{"name": "Jake Latz", "mlb_team": "TEX", "player_type": "pitcher",
          "Position": "SP,RP", "owner": "Team A", "roster_status": "reserve"}]
    )
    merged = merge_fantrax(players, fantrax)
    latz = merged[merged["Name"] == "Jake Latz-P"]

    assert len(latz) == 1, "An unmatched ROSTERED player must be added to the table"
    row = latz.iloc[0]
    assert row["owner"] == "Team A", "The added row must carry its ownership"
    assert row["IP"] == 0.0 and row["PA"] == 0.0, (
        f"Zero volume is what makes the rates inert; got IP={row['IP']}, PA={row['PA']}"
    )
    assert row["K"] == 0.0 and row["W"] == 0.0, "Counting stats must be zero"
    stats = ["PA", "R", "HR", "RBI", "SB", "OPS", "IP", "W", "SV", "K", "ERA", "WHIP"]
    assert not pd.isna(row[stats]).any(), (
        "No stat may be NaN — team totals multiply by PA/IP and NaN would "
        "poison the sum even at zero weight."
    )


def test_two_way_player_market_value_is_split_not_double_counted():
    """One player's price must not be counted twice because we store two rows.

    `search_trades` sums the value column over players sent and received, so
    giving each half the full price makes a $78 asset look like $156 and the
    fairness check demands twice the return.
    """
    players = prepare_projections(_raw_projection_rows())
    ottoneu = pd.DataFrame(
        [{"name": "Shohei Ohtani", "fg_id": "19755",
          "median_salary": 78.0, "salary_momentum": 3.6}]
    )
    hkb = pd.DataFrame([{"name": "Shohei Ohtani", "value": 10000, "rank": 1}])
    priced = merge_market(players, {"ottoneu": ottoneu, "hkb": hkb})

    sides = priced[priced["Name"].str.startswith("Shohei Ohtani")]
    assert len(sides) == 2, f"Expected both Ohtani sides, got {len(sides)}"
    assert abs(sides["market_value"].sum() - 78.0) < 1e-9, (
        f"The two halves must SUM to his one price (78), got "
        f"{sides['market_value'].sum()}"
    )
    assert (sides["market_value"] == 39.0).all(), (
        f"Even split expected, got {sides['market_value'].tolist()}"
    )
    assert abs(sides["dynasty_value"].sum() - 10000) < 1e-6, (
        "Dynasty value is additive too and must be split"
    )


def test_same_name_different_people_do_not_have_value_split():
    """Splitting is keyed on identity, not name.

    "José Fermín" (STL hitter, MLBAMID 665877) and "José Fermin" (LAA pitcher,
    MLBAMID 820862) are two different people who normalize identically. Halving
    on a name match would rob them both.
    """
    players = pd.DataFrame(
        [
            {"Name": "José Fermín-H", "player_type": "hitter",
             "PlayerId": "21746", "MLBAMID": 665877},
            {"Name": "José Fermin-P", "player_type": "pitcher",
             "PlayerId": "33908", "MLBAMID": 820862},
        ]
    )
    ottoneu = pd.DataFrame(
        [
            {"name": "Jose Fermin", "fg_id": "21746",
             "median_salary": 1.0, "salary_momentum": 0.0},
            {"name": "Jose Fermin", "fg_id": "33908",
             "median_salary": 1.0, "salary_momentum": 0.0},
        ]
    )
    priced = merge_market(players, {"ottoneu": ottoneu})
    assert (priced["market_value"] == 1.0).all(), (
        f"Different people must keep their own full price, got "
        f"{priced['market_value'].tolist()}"
    )
