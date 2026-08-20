"""
The join step: read the latest raw snapshot of every source, produce `players`.

This is the ONLY place cross-source identity reconciliation happens. Fetchers
write their source verbatim (see `raw_io`); everything downstream reads the one
wide table this builds. If a name has to be matched across two providers, it is
matched here or nowhere.

Identity keys, in the order they are trusted:
    MLBAMID    MLB's own id  — projections <-> identity
    PlayerId   FanGraphs id  — projections <-> Ottoneu (its `fg_id`)
    name       normalized, accent-folded, suffix-stripped — last resort

No provider covers everyone, so every merge is a CASCADE: try the strong key,
then fall back to name for whatever is left. That is measurably better than
either key alone — Ottoneu prices 951 players by FanGraphs id and 987 by
id-then-name, because ~245 of its rows are minor leaguers carrying only a
FanGraphs *minor*-league id.
"""

import pandas as pd

from .names import normalize_name, strip_name_suffix
from .raw_io import PLAYERS_TABLE_PATH, read_latest_raw

# Rest-of-season volume floors, applied to FREE AGENTS ONLY by
# apply_volume_floors (rostered players are exempt — see its docstring).
# Raw snapshots stay unfiltered; this is a join-step transform.
MIN_AB: float = 10.0
MIN_IP: float = 5.0

HITTING_STATS: list[str] = ["PA", "R", "HR", "RBI", "SB", "OPS"]
PITCHING_STATS: list[str] = ["IP", "W", "SV", "K", "ERA", "WHIP"]

def match_rows(
    left_keys: list[pd.Series],
    right_keys: list[pd.Series],
    unique: bool = False,
) -> pd.Series:
    """Cascade-match left rows to right rows, trying each key pair in order.

    Pass N only considers rows still unmatched after pass N-1, so a strong key
    (an id) is never overridden by a weak one (a name). Within a pass, duplicate
    right keys resolve first-wins.

    Args:
        left_keys: One Series per pass, all indexed like the left frame.
        right_keys: One Series per pass, all indexed like the right frame.
            Must be the same length as left_keys.
        unique: Enforce one-to-one. By default many left rows may share one
            right row, which is what market joins want — Ohtani's -H and -P rows
            both take his single Ottoneu price. Set True when a right row
            represents something only one left row can be, such as a Fantrax
            roster spot: "José Ramírez" (CLE) and "Jose Ramirez" (DET) are two
            different players who normalize identically, and without this the
            second silently inherits the first's ownership. Earlier passes and
            earlier rows win, so the team-qualified match beats the name-only
            fallback.

    Returns:
        Series indexed like the left frame holding the matched right-frame index
        label, or NA where no pass matched. Feed it to `.map(right[col])`.
    """
    assert len(left_keys) == len(right_keys), (
        f"match_rows: got {len(left_keys)} left key(s) and {len(right_keys)} "
        f"right key(s); each pass needs one of each."
    )
    matched = pd.Series(pd.NA, index=left_keys[0].index, dtype="object")
    claimed: set = set()

    for left_key, right_key in zip(left_keys, right_keys):
        pending = matched.isna()
        if not pending.any():
            break
        valid = right_key.notna() & (right_key.astype(str) != "")
        deduped = right_key[valid]
        if unique:
            deduped = deduped[~deduped.index.isin(claimed)]
        deduped = deduped[~deduped.duplicated(keep="first")]
        lookup = pd.Series(deduped.index, index=deduped.values)
        found = left_key[pending].map(lookup)

        if unique:
            # Keep only the first left row to claim each right row.
            found = found.where(~found.duplicated() | found.isna())
            claimed |= set(found.dropna())

        matched.loc[pending] = found

    return matched


def _suffixed_key(names: pd.Series) -> pd.Series:
    """Name key that KEEPS the -H/-P suffix, for joining two suffixed sides.

    `normalize_name` preserves the suffix on purpose, so a two-way player's
    hitter row can never match his pitcher row. Use this when both sides carry
    the suffix (the Fantrax merge).
    """
    return names.astype(str).map(normalize_name)


def _plain_key(names: pd.Series) -> pd.Series:
    """Name key with the -H/-P suffix REMOVED, for joining external providers.

    Market sources know nothing about our suffix convention, so the suffix has
    to come off or nothing matches. A two-way player's -H and -P rows then both
    match his single external row, which is what we want — one market price
    applies to the player, not to a side of him.
    """
    return names.astype(str).map(strip_name_suffix).map(normalize_name)


def prepare_projections(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn a raw projections snapshot into the base `players` frame.

    Drops duplicate player-sides, appends the `-H`/`-P` name suffix that keeps a
    two-way player's two sides distinct, and zero-fills the opposite type's stat
    columns so the unified MEW formula needs no hitter/pitcher branching.

    Does NOT filter by volume — that needs ownership, so it happens after the
    Fantrax merge in `apply_volume_floors`.

    Requires columns: Name, Team, player_type, PlayerId, MLBAMID, WAR,
        the hitting stats (+ optional AB), SO and the pitching stats.
    Returns: one row per player-side with Name suffixed and all 12 scoring stats
        present and non-null.
    """
    required = {"Name", "Team", "player_type", "PlayerId", "MLBAMID", "WAR"}
    missing = required - set(raw.columns)
    assert not missing, (
        f"prepare_projections: raw projections snapshot missing {sorted(missing)}. "
        f"Got: {sorted(raw.columns)}. The scraper's output contract changed."
    )

    hitters = raw[raw["player_type"] == "hitter"].copy()
    pitchers = raw[raw["player_type"] == "pitcher"].copy()
    assert len(hitters) > 0 and len(pitchers) > 0, (
        f"prepare_projections: expected both player types, got "
        f"{len(hitters)} hitters and {len(pitchers)} pitchers."
    )

    # K is the league's category name for what FanGraphs exports as SO.
    if "SO" in pitchers.columns:
        pitchers = pitchers.rename(columns={"SO": "K"})

    hitters["Name"] = hitters["Name"].astype(str) + "-H"
    pitchers["Name"] = pitchers["Name"].astype(str) + "-P"

    for frame, label in ((hitters, "hitter"), (pitchers, "pitcher")):
        dupes = frame["Name"].duplicated().sum()
        if dupes:
            print(f"  dropping {dupes} duplicate {label} name(s)")
    hitters = hitters.drop_duplicates(subset="Name", keep="first")
    pitchers = pitchers.drop_duplicates(subset="Name", keep="first")

    for col in PITCHING_STATS:
        hitters[col] = 0.0
    for col in HITTING_STATS:
        pitchers[col] = 0.0

    # AB is carried purely so apply_volume_floors can use it after the merge.
    for frame in (hitters, pitchers):
        if "AB" not in frame.columns:
            frame["AB"] = 0.0
    keep = (
        ["Name", "Team", "player_type", "PlayerId", "MLBAMID", "WAR", "AB"]
        + HITTING_STATS
        + PITCHING_STATS
    )
    players = pd.concat([hitters[keep], pitchers[keep]], ignore_index=True)
    players["WAR"] = players["WAR"].fillna(0.0)
    players["AB"] = players["AB"].fillna(0.0)
    # FanGraphs blanks Team for unsigned players; "FA" reads better than NaN.
    players["Team"] = players["Team"].fillna("FA")

    stat_cols = HITTING_STATS + PITCHING_STATS
    nan_rows = players[stat_cols].isna().any(axis=1)
    assert not nan_rows.any(), (
        f"prepare_projections: NaN scoring stats for "
        f"{sorted(players.loc[nan_rows, 'Name'])[:10]}. Downstream scoring "
        f"requires all 12 stats present (0 for the opposite player type)."
    )

    # Placeholder columns the merges below fill in.
    for col in (
        "Position",
        "owner",
        "roster_status",
        "injury_status",
        "injury_detail",
        "age",
        "fantrax_score",
        "pct_rostered",
    ):
        players[col] = None

    print(f"  projections: {len(players)} player-sides")
    return players


def _append_unprojected(
    players: pd.DataFrame, unmatched_rostered: pd.DataFrame
) -> pd.DataFrame:
    """Add rows for rostered players the projection feed has never heard of.

    The table used to be projection-anchored: no projection row meant no player
    row, so a rostered prospect was simply invisible. That is wrong for a
    dynasty league — a player you own exists whether or not FanGraphs projects
    him, and the market sources (Ottoneu, HarryKnowsBall) DO price him. Rows
    added here carry no production but full identity, ownership and market
    value, which is exactly what a develop-vs-win-now decision needs.

    How much this matters depends entirely on the projection feed: with ATC it
    is ~72 of 286 rostered players league-wide, with Steamer ~2. Doing it here
    makes roster completeness independent of which feed you chose.

    All stats are filled with 0.0, and for ratio stats that value is INERT by
    construction rather than meaningful:

      * counting stats -> 0.0 is a fact. They will produce nothing, and they
        correctly z-score far below the mean in `add_fantasy_value`.
      * PA/IP -> 0.0, so team ratio totals (which weight OPS by PA and ERA/WHIP
        by IP) give these rows zero weight. `add_mew` likewise multiplies every
        ratio term by PA or IP, so it is zero there too.
      * ratio stats -> 0.0, never read. A rate over zero playing time is
        UNDEFINED, so `add_fantasy_value` excludes zero-volume players from the
        ratio z-score population entirely. That is what makes 0.0 safe here: no
        sentinel value is asserting anything about the player. Do not "fix" this
        by filling a plausible-looking rate — that invents information, and for
        the negated categories a low fill is actively backwards (a stored ERA of
        0.00 would read as the best pitcher in the league).
    """
    if len(unmatched_rostered) == 0:
        return players

    rows = []
    for _, row in unmatched_rostered.iterrows():
        name = str(row["name"])
        is_pitcher = row.get("player_type") == "pitcher"
        if not name.endswith(("-H", "-P")):
            name += "-P" if is_pitcher else "-H"
        entry = {
            "Name": name,
            "Team": row.get("mlb_team") or "FA",
            "player_type": "pitcher" if is_pitcher else "hitter",
            "PlayerId": None,
            "MLBAMID": None,
            "WAR": 0.0,
            "AB": 0.0,
            "Position": row.get("Position"),
            "owner": row.get("owner"),
            "roster_status": row.get("roster_status"),
            "injury_status": row.get("injury_status"),
            "injury_detail": row.get("injury_detail"),
            "age": row.get("age"),
            "fantrax_score": row.get("fantrax_score"),
            "pct_rostered": row.get("pct_rostered"),
            "fantrax_id": row.get("fantrax_id"),
        }
        for col in HITTING_STATS + PITCHING_STATS:
            entry[col] = 0.0
        rows.append(entry)

    added = pd.DataFrame(rows)
    n_minors = int((added["roster_status"] == "minors").sum())
    print(
        f"  added {len(added)} rostered player(s) with no projection "
        f"({n_minors} on minor-league slots), zero production"
    )
    return pd.concat([players, added], ignore_index=True)


def apply_volume_floors(players: pd.DataFrame) -> pd.DataFrame:
    """Drop negligible-volume FREE AGENTS. Never drops anyone who is rostered.

    The floor exists to trim the free-agent pool: a player projected for almost
    no rest-of-season playing time is not a realistic pickup, but he does drag
    the z-score population that FV is computed against.

    It must run AFTER the Fantrax merge, because ownership is what decides who
    is exempt. A rostered player is on someone's roster whatever his projected
    volume — most importantly a minor-league prospect, whose rest-of-season MLB
    projection is near zero by definition. Filtering before the merge silently
    deleted exactly the dynasty assets worth reasoning about.

    Requires columns: owner, AB (hitters), IP (pitchers), player_type.
    """
    rostered = players["owner"].notna()
    is_hitter = players["player_type"] == "hitter"

    # AB is only carried for the floor; hitters keep PA as the volume stat.
    ab = players["AB"].fillna(0) if "AB" in players.columns else players["PA"].fillna(0)
    thin_hitter = is_hitter & (ab < MIN_AB)
    thin_pitcher = ~is_hitter & (players["IP"].fillna(0) < MIN_IP)

    drop = (thin_hitter | thin_pitcher) & ~rostered
    kept_anyway = (thin_hitter | thin_pitcher) & rostered

    print(
        f"  volume floors: dropped {int(drop.sum())} free agents "
        f"(AB<{MIN_AB:g} / IP<{MIN_IP:g}); kept {int(kept_anyway.sum())} "
        f"rostered players below the floor"
    )
    return players[~drop].reset_index(drop=True)


def merge_fantrax(players: pd.DataFrame, fantrax: pd.DataFrame) -> pd.DataFrame:
    """Merge Fantrax ownership, positions, status and injuries onto players.

    Fantrax is the authority on who owns whom and on position eligibility;
    projections are the authority on production. Matching is a cascade on
    (name, MLB team) then name alone, because Fantrax and FanGraphs disagree on
    spelling often enough that team disambiguation earns its place — and because
    two different players can share a name.

    Requires (fantrax): name, Position, owner, roster_status. Optional:
        mlb_team, injury_status, injury_detail, age, fantrax_score,
        pct_rostered, fantrax_id, player_type, status_id.
    Adds/fills: Position, owner, roster_status, injury_status, injury_detail,
        age, fantrax_score, pct_rostered, fantrax_id.
    """
    players = players.copy()
    assert "name" in fantrax.columns, (
        f"merge_fantrax: fantrax snapshot needs a 'name' column, "
        f"got {sorted(fantrax.columns)}"
    )
    from .fantrax_api import FANTRAX_NAME_CORRECTIONS, get_player_type

    fantrax = fantrax.copy()

    # Known Fantrax misspellings that no amount of normalization can bridge
    # ("Logan OHoppe" is missing an apostrophe; "Leodalis De Vries" is a
    # different first name than FanGraphs' "Leo De Vries"). The raw snapshot
    # keeps Fantrax's spelling verbatim, so the correction is applied here, at
    # the one place identity is reconciled.
    corrected = fantrax["name"].astype(str).replace(FANTRAX_NAME_CORRECTIONS)
    n_fixed = int((corrected != fantrax["name"].astype(str)).sum())
    if n_fixed:
        print(f"  applied {n_fixed} Fantrax name correction(s)")

    # Suffix the Fantrax side the same way so the keys are comparable — but
    # NEVER double-suffix. Fantrax already spells a split two-way player
    # "Shohei Ohtani-H" / "Shohei Ohtani-P" (in this league his two halves are
    # owned by different teams, so the suffix is load-bearing). Appending again
    # yields "Shohei Ohtani-H-H", which matches nothing and silently drops the
    # best player in the league from the table.
    if "player_type" in fantrax.columns:
        types = fantrax["player_type"]
    else:
        types = fantrax["Position"].astype(str).map(get_player_type)
    already_suffixed = corrected.str.endswith(("-H", "-P"))
    fantrax["_suffixed"] = corrected.where(
        already_suffixed,
        corrected + types.map(lambda t: "-P" if t == "pitcher" else "-H"),
    )
    if int(already_suffixed.sum()):
        print(
            f"  {int(already_suffixed.sum())} Fantrax name(s) already carry a "
            f"-H/-P suffix; left as-is"
        )

    left_name = _suffixed_key(players["Name"])
    right_name = _suffixed_key(fantrax["_suffixed"])
    team_col = "mlb_team" if "mlb_team" in fantrax.columns else None

    if team_col is not None:
        left_keys = [left_name + "|" + players["Team"].astype(str), left_name]
        right_keys = [right_name + "|" + fantrax[team_col].astype(str), right_name]
    else:
        left_keys, right_keys = [left_name], [right_name]

    idx = match_rows(left_keys, right_keys, unique=True)

    carry = [
        "Position",
        "owner",
        "roster_status",
        "injury_status",
        "injury_detail",
        "age",
        "fantrax_score",
        "pct_rostered",
        "fantrax_id",
    ]
    for col in carry:
        if col not in fantrax.columns:
            continue
        incoming = idx.map(fantrax[col])
        players[col] = incoming.where(incoming.notna(), players.get(col))

    matched = idx.notna().sum()
    print(f"  fantrax: matched {matched} players")

    unmatched_rostered = fantrax[
        fantrax["owner"].notna() & ~fantrax.index.isin(idx.dropna())
    ]
    players = _append_unprojected(players, unmatched_rostered)

    print(f"  fantrax: {int(players['owner'].notna().sum())} rostered on the table")
    return players


def merge_identity(players: pd.DataFrame, identity: pd.DataFrame) -> pd.DataFrame:
    """Merge birth date and age from the MLB Stats API identity snapshot.

    Joined on MLBAMID only — it is MLB's own id and needs no name fallback.
    Overwrites the age Fantrax supplied, since MLB is authoritative on it.

    Requires (identity): MLBAMID, and at least one of birth_date / age.
    Adds/fills: birth_date, age.
    """
    players = players.copy()
    assert "MLBAMID" in identity.columns, (
        f"merge_identity: identity snapshot needs MLBAMID, "
        f"got {sorted(identity.columns)}"
    )

    left = players["MLBAMID"].map(lambda v: str(int(v)) if pd.notna(v) else pd.NA)
    right = identity["MLBAMID"].map(lambda v: str(int(v)) if pd.notna(v) else pd.NA)
    idx = match_rows([left], [right])

    for col in ("birth_date", "age"):
        if col not in identity.columns:
            continue
        incoming = idx.map(identity[col])
        players[col] = incoming.where(incoming.notna(), players.get(col))

    print(f"  identity: matched {idx.notna().sum()} players on MLBAMID")
    return players


# Market columns denominated in value that SUMS over players (dollars, dynasty
# points), so a two-way player's two rows must share one player's worth. `adp`,
# `pct_owned` and `dynasty_rank` are ordinals or percentages — they do not sum,
# and halving them would be meaningless.
_ADDITIVE_VALUE_COLUMNS: tuple[str, ...] = (
    "market_value",
    "salary_momentum",
    "dynasty_value",
)


def _split_across_sides(players: pd.DataFrame) -> pd.DataFrame:
    """Divide additive market value across the rows of a split two-way player.

    The market prices a PLAYER; we represent a two-way player as two rows
    (Ohtani-H and Ohtani-P). Giving each row his full price double-counts him:
    `search_trades` sums the value column over the players sent and received, so
    a trade for both halves would appear to move $156 of a $78 asset, and the
    fairness check would demand twice the return.

    Splitting evenly is the honest default. It is not a claim that his bat and
    his arm are worth the same — it is a refusal to invent a split the market
    never quoted, while keeping the one thing we know true: the two halves sum
    to his one price.

    In this league the two halves can even be owned by different teams, so both
    rows must keep a value; dropping one is not an option.

    Sides are grouped by IDENTITY (MLBAMID, falling back to the FanGraphs
    PlayerId), never by name. A shared name proves nothing: "José Fermín" (STL,
    hitter, MLBAMID 665877) and "José Fermin" (LAA, pitcher, MLBAMID 820862) are
    two different people who normalize to the same string, and halving their
    prices would rob them both. Two rows are the same player only when an id
    says so.
    """
    key = players["MLBAMID"].where(
        players["MLBAMID"].notna(), players["PlayerId"]
    )
    key = key.map(lambda v: str(v) if pd.notna(v) else pd.NA)
    counts = key.value_counts()
    n_sides = key.map(counts)
    split = key.notna() & (n_sides > 1)
    if not split.any():
        return players

    for col in _ADDITIVE_VALUE_COLUMNS:
        if col in players.columns:
            players.loc[split, col] = players.loc[split, col] / n_sides[split]

    shared = sorted({strip_name_suffix(n) for n in players.loc[split, "Name"]})
    print(
        f"  split additive market value across {int(split.sum())} rows for "
        f"{len(shared)} two-way player(s): {shared}"
    )
    return players


def merge_market(players: pd.DataFrame, market: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge market-driven value signals onto players.

    These are the dynasty axis of the two-axis value model: MEW answers "does
    this player win me categories this season", market value answers "what is
    this player worth to other people". Sources are independent, so a missing
    one degrades that column rather than the table.

    Args:
        players: The table so far.
        market: {source_name: frame} for any of ottoneu / adp / espn / hkb.

    Adds: market_value + salary_momentum (ottoneu), adp (fantrax ADP),
        pct_owned (ESPN), dynasty_value + dynasty_rank (HarryKnowsBall).

    Additive value columns are SPLIT across a two-way player's two rows — see
    `_split_across_sides`.
    """
    players = players.copy()
    left_name = _plain_key(players["Name"])

    ottoneu = market.get("ottoneu")
    if ottoneu is not None:
        # Pass 1 on the FanGraphs id, pass 2 on name — see module docstring.
        left_keys = [players["PlayerId"].astype(str), left_name]
        right_keys = [ottoneu["fg_id"].astype(str), _plain_key(ottoneu["name"])]
        idx = match_rows(left_keys, right_keys)
        players["market_value"] = pd.to_numeric(idx.map(ottoneu["median_salary"]))
        players["salary_momentum"] = pd.to_numeric(
            idx.map(ottoneu["salary_momentum"])
        )
        print(
            f"  market/ottoneu: priced {int(players['market_value'].notna().sum())} "
            f"players (total ${players['market_value'].sum():.0f})"
        )

    for source, columns in (
        ("adp", {"adp": "adp"}),
        ("espn", {"pct_owned": "pct_owned"}),
        ("hkb", {"value": "dynasty_value", "rank": "dynasty_rank"}),
    ):
        frame = market.get(source)
        if frame is None:
            continue
        idx = match_rows([left_name], [_plain_key(frame["name"])])
        for src_col, dest_col in columns.items():
            if src_col in frame.columns:
                players[dest_col] = pd.to_numeric(idx.map(frame[src_col]))
        print(f"  market/{source}: matched {idx.notna().sum()} players")

    return _split_across_sides(players)


def build_players(
    system: str = "atc",
    on_or_before=None,
    include_market: bool = True,
    include_identity: bool = True,
) -> pd.DataFrame:
    """Build the wide `players` table from the latest raw snapshot of each source.

    Every source is optional except projections: a missing Fantrax snapshot
    yields a table with no ownership (useful for pure projection work), a
    missing market snapshot yields one without the dynasty axis. Only
    projections are load-bearing, and their absence is a hard error.

    Args:
        system: Projection system directory under raw/projections (atc, steamer).
        on_or_before: Ignore snapshots newer than this date, to reproduce a past
            day's table.
        include_market: Join the market sources.
        include_identity: Join the MLB identity snapshot.

    Returns:
        One row per player-side. Name carries the -H/-P suffix.
    """
    print(f"=== Building players table ({system}) ===")

    raw, proj_date = read_latest_raw(f"projections/{system}", on_or_before)
    print(f"projections/{system} snapshot: {proj_date}")
    players = prepare_projections(raw)

    from .raw_io import available_dates

    if available_dates("fantrax"):
        fantrax, date = read_latest_raw("fantrax", on_or_before)
        print(f"fantrax snapshot: {date}")
        players = merge_fantrax(players, fantrax)
    else:
        print("fantrax: no snapshot — table will have no ownership")
    players = apply_volume_floors(players)

    if include_identity and available_dates("identity"):
        identity, date = read_latest_raw("identity", on_or_before)
        print(f"identity snapshot: {date}")
        players = merge_identity(players, identity)

    if include_market:
        market: dict[str, pd.DataFrame] = {}
        for source in ("ottoneu", "adp", "espn", "hkb"):
            if available_dates(f"market/{source}"):
                frame, date = read_latest_raw(f"market/{source}", on_or_before)
                market[source] = frame
                print(f"market/{source} snapshot: {date}")
        if market:
            players = merge_market(players, market)
        else:
            print("market: no snapshots — no dynasty axis")

    print(f"=== players table complete: {len(players)} rows ===")
    return players


def write_players(players: pd.DataFrame, path=None):
    """Write the joined table to `data/players.parquet` (or `path`)."""
    path = PLAYERS_TABLE_PATH if path is None else path
    players.to_parquet(path, index=False)
    print(f"Wrote {len(players)} rows -> {path}")
    return path
