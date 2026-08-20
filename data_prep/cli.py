"""
Command line: fetch one source, or join what has been fetched.

    uv run fetch status         # how stale is each source?
    uv run fetch market         # no auth       — safe to run anytime
    uv run fetch identity       # no auth
    uv run fetch projections    # browser cookies (FanGraphs)
    uv run fetch fantrax        # PASTED cookies in config.json (they expire)
    uv run fetch build          # join latest snapshots -> data/players.parquet

Each source is a separate command on purpose: a stale Fantrax cookie must not
stop you refreshing projections or market prices. `build` then joins whatever
snapshots exist.
"""

import argparse

from .raw_io import snapshot_ages

SOURCES: list[str] = [
    "projections/steamer",
    "projections/atc",
    "fantrax",
    "standings",
    "identity",
    "market/ottoneu",
    "market/adp",
    "market/espn",
    "market/hkb",
]


def cmd_status() -> None:
    """Print snapshot staleness for every source, oldest first."""
    print(snapshot_ages(SOURCES).to_string(index=False))


def cmd_market() -> None:
    """Fetch all four market sources (no auth required)."""
    from .market import fetch_all_market

    fetch_all_market()


def cmd_projections() -> None:
    """Scrape FanGraphs rest-of-season projections (uses browser cookies)."""
    from .scrape_fangraphs import main as scrape_main

    scrape_main()


def cmd_fantrax() -> None:
    """Fetch Fantrax rosters and standings (needs fresh cookies in config.json)."""
    from .fantrax_api import main as fantrax_main

    fantrax_main()


def cmd_identity() -> None:
    """Fetch MLB Stats API identity (birth dates, ages) for known players.

    The id list is the UNION across every projection snapshot, not just one
    system: the systems cover very different player sets (Steamer runs ~11k rows
    deep into the minors, ATC ~1.3k), and identity should not go stale for
    whichever system you happen to build with next.
    """
    from .mlb_api import fetch_identity_snapshot
    from .raw_io import available_dates, read_latest_raw

    ids: set[int] = set()
    for source in SOURCES:
        if not source.startswith("projections/") or not available_dates(source):
            continue
        frame, date = read_latest_raw(source)
        found = set(frame["MLBAMID"].dropna().astype(int))
        ids |= found
        print(f"  {source} ({date}): {len(found)} ids")

    assert ids, (
        "cmd_identity: no projection snapshot has a non-null MLBAMID, so there "
        "is nobody to look up. Run `uv run fetch projections` first."
    )
    print(f"Fetching identity for {len(ids)} players (union of projection systems)")
    fetch_identity_snapshot(sorted(ids))


def cmd_build(system: str) -> None:
    """Join the latest snapshot of every source into data/players.parquet."""
    from .build import build_players, write_players

    write_players(build_players(system=system))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch one data source into the raw layer, or join them.",
    )
    parser.add_argument(
        "command",
        choices=["status", "projections", "fantrax", "market", "identity", "build", "all"],
        help="What to run. 'all' fetches every source then builds.",
    )
    parser.add_argument(
        "--system",
        default="atc",
        help="Projection system for 'build' (atc or steamer). Default: atc",
    )
    args = parser.parse_args()

    if args.command == "status":
        cmd_status()
    elif args.command == "projections":
        cmd_projections()
    elif args.command == "fantrax":
        cmd_fantrax()
    elif args.command == "market":
        cmd_market()
    elif args.command == "identity":
        cmd_identity()
    elif args.command == "build":
        cmd_build(args.system)
    elif args.command == "all":
        cmd_projections()
        cmd_fantrax()
        cmd_market()
        cmd_identity()
        cmd_build(args.system)


if __name__ == "__main__":
    main()
