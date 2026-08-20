"""
Player name normalization and display helpers (data prep layer).

Pure functions — no optimizer config.
"""

import unicodedata


def strip_diacritics(name: str) -> str:
    """Replace accented characters with ASCII equivalents (Suárez → Suarez).

    Preserves casing, suffixes (Jr., III), and -H/-P tags.
    """
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def strip_name_suffix(name: str) -> str:
    """Strip -H or -P suffix from player name for display."""
    if name.endswith("-H") or name.endswith("-P"):
        return name[:-2]
    return name


def normalize_name(name: str) -> str:
    """
    Normalize player name for fuzzy comparison.

    CRITICAL: Preserves -H/-P suffix!

    Handles:
        - Accented characters (Rodríguez → rodriguez)
        - Name suffixes like Jr., Sr. (removed)
    """
    suffix = ""
    if name.endswith("-H"):
        suffix = "-H"
        name = name[:-2]
    elif name.endswith("-P"):
        suffix = "-P"
        name = name[:-2]

    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = name.lower()

    for suffix_remove in [" jr.", " jr", " sr.", " sr", " ii", " iii", " iv"]:
        name = name.replace(suffix_remove, "")

    name = name.replace("\u2019", "'").replace("`", "'")

    return name.strip() + suffix.lower()
