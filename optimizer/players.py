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


def get_startable_slots(
    position_str: str,
    injury_status: str | None = None,
    projected_volume: float | None = None,
) -> set[str]:
    """Slots a player can START in, honoring real-world injury state.

    Day-to-Day ("DTD") players are startable (short-term, not roster-blocking).

    An "IL" player is excluded ONLY when their rest-of-season projection is
    empty. This is the §9c point applied to the lineup model: a RoS projection
    for an injured player is *already* their post-return value — the feed has
    discounted the games they will miss. Zeroing them again charges the same
    absence twice, which understates every roster holding such a player. A
    player with a genuinely dead season projects zero volume and is excluded on
    that basis instead of on the flag.

    Args:
        position_str: Comma-separated position string (e.g., "SS,2B").
        injury_status: "IL", "DTD", None, or NaN (from the silver table's
            optional injury_status column).
        projected_volume: Rest-of-season PA + IP. None means unknown, in which
            case an IL player is excluded (the conservative reading).

    Returns:
        Eligible starting slots, or an empty set if the player cannot start.
    """
    if injury_status == "IL" and (projected_volume is None or projected_volume <= 0.0):
        return set()
    return get_eligible_slots(position_str)
