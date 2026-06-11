"""
Player identity: names, positions, eligibility.

Pure functions — bottom of the dependency DAG (depends only on config).
"""

import unicodedata

from .config import SLOT_ELIGIBILITY


def strip_diacritics(name: str) -> str:
    """Replace accented characters with ASCII equivalents (Suárez → Suarez)."""
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def strip_name_suffix(name: str) -> str:
    """Strip -H or -P suffix from player name for display.

    Defined ONLY here, imported everywhere else.
    """
    if name.endswith("-H") or name.endswith("-P"):
        return name[:-2]
    return name


def get_eligible_slots(position_str: str) -> set[str]:
    """Compute which lineup slots a player is eligible for.

    Args:
        position_str: Comma-separated position string (e.g., "SS,2B" or "OF").

    Returns:
        Set of eligible slot names (e.g., {"SS", "2B", "UTIL"}).
    """
    player_positions = set(p.strip() for p in str(position_str).split(","))
    return {
        slot
        for slot, valid_positions in SLOT_ELIGIBILITY.items()
        if player_positions & valid_positions
    }


def get_startable_slots(position_str: str, injury_status: str | None = None) -> set[str]:
    """Slots a player can START in, honoring real-world injury state.

    A player on the Injured List ("IL") cannot fill a starting slot, so the
    lineup MILP must never assign them. Day-to-Day ("DTD") players are still
    startable (short-term, not roster-blocking). Any other injury_status
    (including None) is treated as healthy.

    Args:
        position_str: Comma-separated position string (e.g., "SS,2B").
        injury_status: "IL", "DTD", None, or NaN (from the silver table's
            optional injury_status column).

    Returns:
        Eligible starting slots, or an empty set if the player is on the IL.
    """
    if injury_status == "IL":
        return set()
    return get_eligible_slots(position_str)
