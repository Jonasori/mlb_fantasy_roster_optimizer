"""
Player name normalization and display helpers (data prep layer).

Pure functions — no optimizer config.
"""

import re
import unicodedata

# Generational suffixes, matched only at the END of the name and longest-first.
# A plain substring replace is wrong twice over: " ii" fires inside " iii"
# (leaving "hasselli"), and " iv" fires mid-word (turning "Ivey" into "ey").
_SUFFIX_RE = re.compile(r"\s+(?:jr\.?|sr\.?|iii|ii|iv)$")

# Punctuation providers disagree on: "J.R. Ritchie" vs "JR Ritchie".
# Apostrophes and hyphens are kept — they are part of the name proper
# ("O'Hoppe", "Jung-hoo") and both sides spell them consistently.
_PUNCT_RE = re.compile(r"[^\w\s'\-]")


def strip_diacritics(name: str) -> str:
    """Replace accented characters with ASCII equivalents (Suárez → Suarez).

    Preserves casing, suffixes (Jr., III), and -H/-P tags.

    Asserts on a non-string rather than letting unicodedata raise. A NaN
    reaching here means an upstream source had a null name — StatsAPI omits
    `fullName` for a handful of real minor leaguers — and the fix belongs at
    that fetcher, which knows the player id and can label the row traceably.
    The bare TypeError surfaces eleven frames deep inside pandas' map_infer,
    which says nothing about where to look.
    """
    assert isinstance(name, str), (
        f"strip_diacritics: expected a string, got {type(name).__name__} "
        f"({name!r}). A null name means the upstream fetcher let one through — "
        f"fill it there with something traceable (e.g. 'mlbam-<id>'), do not "
        f"loosen this guard."
    )
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
        - Trailing generational suffixes: Jr., Sr., II, III, IV (removed)
        - Punctuation providers disagree on (J.R. Ritchie → jr ritchie)
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

    name = name.replace("\u2019", "'").replace("`", "'")
    name = _SUFFIX_RE.sub("", name)
    name = _PUNCT_RE.sub("", name)
    # Collapse whitespace left behind by removed punctuation ("j.r." -> "jr").
    name = " ".join(name.split())

    return name + suffix.lower()
