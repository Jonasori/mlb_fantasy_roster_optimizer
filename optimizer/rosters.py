"""
Roster selection for optimizer workflows.

The optimizer works on the "main roster" — players with roster_status in
{active, reserve, IR}. Taxi and minors players are on developmental rosters
and not eligible for lineup assignment.
"""

import pandas as pd

from .config import MY_TEAM_NAME, ROSTER_SIZE

MAIN_ROSTER_STATUSES: set[str] = {"active", "reserve", "IR"}


def get_main_roster(players: pd.DataFrame, owner_name: str) -> set[str]:
    """Get the main roster (active + reserve + IR) for a team.

    Args:
        players: Silver table with owner and roster_status columns.
        owner_name: Team name in players['owner'].

    Returns:
        Set of player Names on the team's main roster.
    """
    mask = (players["owner"] == owner_name) & (
        players["roster_status"].isin(MAIN_ROSTER_STATUSES)
    )
    names = set(players[mask]["Name"])

    if owner_name == MY_TEAM_NAME:
        # The active 28-man roster is active+reserve; players on Injured
        # Reserve (IR) are held in addition to the 28, so the count can
        # exceed ROSTER_SIZE. IL players are kept off the starting lineup
        # by the injury-aware lineup solver, not by exclusion here.
        assert len(names) >= ROSTER_SIZE, (
            f"get_main_roster: {owner_name} has {len(names)} active+reserve+IR players, "
            f"expected at least {ROSTER_SIZE}. Fix data_prep roster assignments."
        )

    assert len(names) >= 18, (
        f"get_main_roster: {owner_name} has only {len(names)} main roster players, "
        f"need at least 18 to fill a lineup."
    )
    return names
