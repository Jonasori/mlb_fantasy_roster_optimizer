"""
Read a logged-in session out of whichever browser has one.

Both cookie-gated sources (FanGraphs and Fantrax) authenticate this way, so the
browser probe lives here once. Reading the browser's own cookie store beats
hand-pasting values into config.json: nothing to copy, nothing to re-paste when
a session rolls over, and no secrets in a file.
"""

import browser_cookie3
from http.cookiejar import CookieJar

# Browsers to probe, in priority order. browser_cookie3 exposes one loader per
# browser. (Chromium-based browsers like Arc store cookies under Chrome's path
# on macOS and are usually picked up by `chrome`.)
COOKIE_BROWSERS: tuple[str, ...] = (
    "brave",
    "chrome",
    "edge",
    "vivaldi",
    "opera",
    "firefox",
    "safari",
)


def load_browser_cookies(
    domain: str, required_cookie: str | None = None, label: str | None = None
) -> CookieJar:
    """Find a browser holding cookies for `domain` and return its cookie jar.

    Args:
        domain: Cookie domain to load, e.g. ".fangraphs.com" or "fantrax.com".
        required_cookie: If given, only accept a browser whose cookie names
            contain this substring — used when one specific cookie proves login
            (FanGraphs' `wordpress_logged_in`). When None, the first browser
            with ANY cookie for the domain wins and the caller is expected to
            verify the session itself (Fantrax has no single tell-tale cookie).
        label: Human-readable source name for status output. Defaults to domain.

    Returns:
        The winning browser's CookieJar, ready to assign to a requests.Session.
    """
    label = label or domain
    print(f"Loading {label} cookies (auto-detecting browser)...")

    for name in COOKIE_BROWSERS:
        loader = getattr(browser_cookie3, name, None)
        if loader is None:
            continue
        # Each browser may be absent or its cookie store locked. There is no
        # way to ask browser_cookie3 whether a browser is usable other than
        # calling it, so this probe is the one sanctioned try/except in the
        # codebase (AGENTS.md fail-fast); the real failure is the assert below.
        try:
            candidate = loader(domain_name=domain)
        except Exception as exc:  # noqa: BLE001 - browser_cookie3 raises many types
            print(f"  {name}: unavailable ({type(exc).__name__})")
            continue

        names = [c.name for c in candidate]
        if not names:
            print(f"  {name}: no {domain} cookies")
            continue
        if required_cookie is not None and not any(
            required_cookie in n for n in names
        ):
            print(f"  {name}: {len(names)} cookie(s), but not logged in")
            continue

        print(f"  {name}: {len(names)} cookie(s) — using this browser")
        return candidate

    raise AssertionError(
        f"No {label} cookies found in any supported browser "
        f"({', '.join(COOKIE_BROWSERS)}).\n"
        f"Log in to {label} in one of them, then re-run. If you are logged in, "
        f"the browser's cookie store may be locked — quit and reopen the "
        f"browser, or grant terminal access to it."
    )
