import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def imports():
    import io

    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd

    from optimizer.players import strip_name_suffix

    def fig_to_png(fig: plt.Figure, dpi: int = 150):
        """Convert matplotlib Figure to PNG mo.image to avoid huge SVG output."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return mo.image(buf.read())

    return fig_to_png, mo, pd, strip_name_suffix


@app.cell
def projection_controls(mo):
    from optimizer.config import ALL_TEAM_NAMES, MY_TEAM_NAME

    projection_system = mo.ui.radio(
        options={"Steamer": "steamer", "ATC": "atc"},
        value="ATC",
        label="Projection system",
    )
    rebuild_data = mo.ui.button(
        value=0,
        on_click=lambda v: v + 1,
        label="Rebuild pipeline (scrape + build silver table)",
    )
    team_perspective = mo.ui.dropdown(
        label="Team perspective",
        options=ALL_TEAM_NAMES,
        value=MY_TEAM_NAME,
    )
    return projection_system, rebuild_data, team_perspective


@app.cell
def data_pipeline(pd, rebuild_data, projection_system, team_perspective):
    import subprocess
    from datetime import date
    from pathlib import Path

    from optimizer.banked import standings_to_banked_totals
    from optimizer.config import season_fraction_remaining
    from optimizer.league_state import compute_league_state
    from optimizer.lineup_solver import assign_optimal_slots
    from optimizer.player_scoring import add_fantasy_value, add_mew, add_perceived_value
    from optimizer.players import strip_diacritics
    from optimizer.swap_evaluator import add_bench_value

    _rebuild_tick = rebuild_data.value
    _system = projection_system.value
    _team = team_perspective.value
    print(f"Pipeline trigger count: {_rebuild_tick}  system: {_system}  team: {_team}")

    _repo_root = Path(assign_optimal_slots.__code__.co_filename).resolve().parents[1]
    _data_prep_dir = _repo_root / "data_prep"
    assert _data_prep_dir.exists(), (
        f"data_prep directory not found: {_data_prep_dir}. "
        "Expected repo layout to include data_prep/ at repo root."
    )
    _data_dir = _data_prep_dir / "data"
    _silver = _data_dir / "silver_table.parquet"
    if not _silver.exists():
        _silver = _data_dir / "silver_table.csv"

    _file_map = {
        "steamer": (
            "fangraphs-steamer-projections-hitters_ros.csv",
            "fangraphs-steamer-projections-pitchers_ros.csv",
        ),
        "atc": (
            "fangraphs-atc-projections-hitters_ros.csv",
            "fangraphs-atc-projections-pitchers_ros.csv",
        ),
    }

    # ── On rebuild: scrape if stale, then build silver table (Fantrax refresh) ──
    if _rebuild_tick > 0:
        _today = date.today().strftime("%Y%m%d")
        _today_dir = _data_dir / f"pulled_{_today}"

        # Check for the actual files we need, not just the directory: a
        # pulled_YYYYMMDD dir created by an older scraper (different file
        # names) must not block scraping the rest-of-season feeds.
        _h_name, _p_name = _file_map[_system]
        _have_today = (_today_dir / _h_name).exists() and (
            _today_dir / _p_name
        ).exists()

        if not _have_today:
            print(f"Projections are stale — scraping FanGraphs for {_today}...")
            _scrape = subprocess.run(
                ["uv", "run", "scrape-fangraphs"],
                cwd=str(_data_prep_dir),
                capture_output=True,
                text=True,
            )
            print(_scrape.stdout)
            assert _scrape.returncode == 0, (
                f"FanGraphs scrape failed (exit {_scrape.returncode}):\n{_scrape.stderr}"
            )
        else:
            print(f"Today's projections already pulled: {_today_dir.name}")

        _hitter_csv = str(_today_dir / _h_name)
        _pitcher_csv = str(_today_dir / _p_name)

        print(f"Building silver table with {_system.upper()} projections...")
        _build = subprocess.run(
            [
                "uv",
                "run",
                "build-silver-table",
                "--hitter",
                _hitter_csv,
                "--pitcher",
                _pitcher_csv,
            ],
            cwd=str(_data_prep_dir),
            capture_output=True,
            text=True,
        )
        print(_build.stdout)
        assert _build.returncode == 0, (
            f"Silver table build failed (exit {_build.returncode}):\n{_build.stderr}"
        )

    # ── Find latest projections and load selected system ──
    # Only consider pulled_* dirs that actually contain the rest-of-season
    # files for the selected system (older dirs may hold legacy preseason
    # pulls under different filenames).
    _pulled_dirs = sorted(
        d
        for d in _data_dir.glob("pulled_*")
        if (d / _file_map[_system][0]).exists() and (d / _file_map[_system][1]).exists()
    )
    if not _pulled_dirs:
        print(
            "No rest-of-season projection pulls found — "
            "click Rebuild to scrape FanGraphs."
        )
        players = pd.DataFrame()
        players_fv = pd.DataFrame()
        state = {}
        banked_totals = None
        season_frac = season_fraction_remaining()
        data_ready = False
    else:
        _latest_dir = _pulled_dirs[-1]
        _h_csv = _latest_dir / _file_map[_system][0]
        _p_csv = _latest_dir / _file_map[_system][1]

        # Load hitter projections
        _h = pd.read_csv(str(_h_csv))
        _h["Name"] = _h["Name"].astype(str).apply(strip_diacritics) + "-H"
        _h["Team"] = _h["Team"].fillna("FA")
        _h["player_type"] = "hitter"
        _h["Position"] = "DH"
        _h = _h.drop_duplicates(subset="Name", keep="first")
        if "WAR" not in _h.columns:
            _h["WAR"] = 0.0
        _h["WAR"] = _h["WAR"].fillna(0.0)
        if "MLBAMID" not in _h.columns:
            _h["MLBAMID"] = None

        # Load pitcher projections
        _p = pd.read_csv(str(_p_csv))
        _p["Name"] = _p["Name"].astype(str).apply(strip_diacritics) + "-P"
        _p["Team"] = _p["Team"].fillna("FA")
        _p = _p.rename(columns={"SO": "K"})
        _p["player_type"] = "pitcher"
        _p["Position"] = "RP"
        _p = _p.drop_duplicates(subset="Name", keep="first")
        if "WAR" not in _p.columns:
            _p["WAR"] = 0.0
        _p["WAR"] = _p["WAR"].fillna(0.0)
        if "MLBAMID" not in _p.columns:
            _p["MLBAMID"] = None

        # Combine into single DataFrame
        _h_stats = ["PA", "R", "HR", "RBI", "SB", "OPS"]
        _p_stats = ["IP", "W", "SV", "K", "ERA", "WHIP"]
        for _c in _p_stats:
            _h[_c] = 0.0
        for _c in _h_stats:
            _p[_c] = 0.0

        _all_cols = (
            ["Name", "Team", "Position", "player_type"]
            + _h_stats
            + _p_stats
            + ["WAR", "MLBAMID"]
        )
        players = pd.concat([_h[_all_cols], _p[_all_cols]], ignore_index=True)
        players["owner"] = None
        players["roster_status"] = None
        players["age"] = None
        players["fantrax_score"] = None
        players["pct_rostered"] = None
        print(
            f"Loaded {_system.upper()} projections: "
            f"{len(_h)} hitters + {len(_p)} pitchers = {len(players)} from {_latest_dir.name}"
        )

        # ── Merge Fantrax data from silver table (ownership, positions, ages) ──
        if _silver.exists():
            _st = (
                pd.read_parquet(_silver)
                if _silver.suffix == ".parquet"
                else pd.read_csv(_silver)
            )
            _fantrax_fields = [
                "owner",
                "roster_status",
                "age",
                "fantrax_score",
                "pct_rostered",
            ]
            _ft = _st[["Name", "Position"] + _fantrax_fields].rename(
                columns={c: f"_ft_{c}" for c in ["Position"] + _fantrax_fields}
            )
            players = players.merge(_ft, on="Name", how="left")

            _has_pos = players["_ft_Position"].notna()
            players.loc[_has_pos, "Position"] = players.loc[_has_pos, "_ft_Position"]
            for _c in _fantrax_fields:
                players[_c] = players[f"_ft_{_c}"]
            players = players.drop(
                columns=[f"_ft_{_c}" for _c in ["Position"] + _fantrax_fields]
            )
            _rostered = players["owner"].notna().sum()
            print(f"Merged Fantrax data: {_rostered} rostered players")
        else:
            print("No silver table found — run Rebuild to fetch Fantrax data")

        # ── Load banked YTD totals (Fantrax standings) ──
        # Adds the banked half of season totals so the win model compares
        # full-season standings, not just rest-of-season. Degrades safely to
        # rest-of-season-only when standings are absent or fail validation.
        _standings_path = _data_dir / "standings.parquet"
        banked_totals = None
        season_frac = season_fraction_remaining()
        if _standings_path.exists():
            _standings = pd.read_parquet(_standings_path)
            banked_totals = standings_to_banked_totals(_standings)
        else:
            print(
                "No standings.parquet found — rest-of-season-only model. "
                "Click Rebuild to fetch banked YTD totals from Fantrax."
            )

        # ── Run math pipeline ──
        players = add_fantasy_value(players)
        players_fv = players.copy()

        state = compute_league_state(
            players,
            my_team_name=_team,
            banked_totals=banked_totals,
            season_fraction_remaining=season_frac,
        )
        players = assign_optimal_slots(
            players,
            state["my_lineup"],
            state["opponent_lineups"],
            state["opponent_teams"],
        )
        players = add_perceived_value(
            players, season_fraction_remaining=season_fraction_remaining()
        )
        players = add_mew(players, state["my_totals"], state["gradient"])
        players = add_bench_value(players, state["my_lineup"], state["my_roster_names"])
        print(f"Pipeline complete: {len(players)} players enriched")
        data_ready = True
    return banked_totals, data_ready, players, players_fv, season_frac, state


@app.cell
def game_logs_cell(data_ready, players, pd):
    """Fetch 2026 game logs from MLB Stats API for all players with MLBAMID."""
    import time

    import statsapi

    game_logs: dict[int, list[dict]] = {}
    if data_ready:
        _ids = (
            players.loc[players["MLBAMID"].notna(), "MLBAMID"]
            .astype(int)
            .unique()
            .tolist()
        )
        print(f"Fetching game logs for {len(_ids)} players...")
        _t0 = time.time()

        _BATCH = 100
        for _i in range(0, len(_ids), _BATCH):
            _batch = _ids[_i : _i + _BATCH]
            _ids_str = ",".join(str(x) for x in _batch)
            _data = statsapi.get(
                "people",
                {
                    "personIds": _ids_str,
                    "hydrate": "stats(group=[hitting,pitching],type=[gameLog],season=2026)",
                },
            )
            for _person in _data.get("people", []):
                _pid = _person["id"]
                for _sg in _person.get("stats", []):
                    _group = _sg["group"]["displayName"]
                    for _split in _sg.get("splits", []):
                        game_logs.setdefault(_pid, []).append(
                            {
                                "group": _group,
                                "date": _split.get("date"),
                                **_split["stat"],
                            }
                        )

        _with_data = sum(1 for v in game_logs.values() if v)
        print(
            f"  Fetched game logs in {time.time() - _t0:.1f}s: "
            f"{_with_data} players with data"
        )
    return (game_logs,)


@app.cell
def observed_windows(data_ready, game_logs, pd, players):
    """Attach observed YTD and last-14-day stat versions to each player.

    Alongside the rest-of-season projection columns already on `players`
    (PA, R, HR, ... / IP, W, ...), this adds observed-actual columns compiled
    directly from MLB game logs for two windows:
      - YTD (suffix _ytd): all 2026 games
      - L14 (suffix _l14): games in the last 14 calendar days

    Hitter columns: G/PA/R/HR/RBI/SB/OPS per window. Pitcher columns:
    G/IP/W/SV/K/ERA/WHIP per window. Values are NA where a player has no
    games. The RoS projection columns on `players` remain the forward-looking
    expectation; these are the realized actuals to compare against.

    Returns `players_obs` (a copy of `players` with the window columns added).
    """
    if not data_ready:
        players_obs = players
    else:

        def _agg_hit(logs: list[dict]) -> dict:
            pa = sum(g.get("plateAppearances", 0) for g in logs)
            ab = sum(g.get("atBats", 0) for g in logs)
            ops = (
                sum(float(g.get("obp", 0)) * g.get("plateAppearances", 0) for g in logs)
                / pa
                + sum(float(g.get("slg", 0)) * g.get("atBats", 0) for g in logs) / ab
                if pa > 0 and ab > 0
                else 0.0
            )
            return {
                "G": len(logs),
                "PA": pa,
                "R": sum(g.get("runs", 0) for g in logs),
                "HR": sum(g.get("homeRuns", 0) for g in logs),
                "RBI": sum(g.get("rbi", 0) for g in logs),
                "SB": sum(g.get("stolenBases", 0) for g in logs),
                "OPS": ops,
            }

        def _agg_pit(logs: list[dict]) -> dict:
            ip = sum(float(g.get("inningsPitched", 0)) for g in logs)
            er = sum(g.get("earnedRuns", 0) for g in logs)
            bb = sum(g.get("baseOnBalls", 0) for g in logs)
            h = sum(g.get("hits", 0) for g in logs)
            return {
                "G": len(logs),
                "IP": ip,
                "W": sum(g.get("wins", 0) for g in logs),
                "SV": sum(g.get("saves", 0) for g in logs),
                "K": sum(g.get("strikeOuts", 0) for g in logs),
                "ERA": (er * 9 / ip) if ip > 0 else 0.0,
                "WHIP": ((bb + h) / ip) if ip > 0 else 0.0,
            }

        _cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=14)

        def _recent(gs: list[dict]) -> list[dict]:
            return [
                g
                for g in gs
                if g.get("date") is not None and pd.Timestamp(g["date"]) >= _cutoff
            ]

        _window_cols = [
            f"{c}_{w}"
            for w in ("ytd", "l14")
            for c in [
                "G",
                "PA",
                "R",
                "HR",
                "RBI",
                "SB",
                "OPS",
                "IP",
                "W",
                "SV",
                "K",
                "ERA",
                "WHIP",
            ]
        ]

        _by_id: dict[int, dict] = {}
        for _pid, _logs in game_logs.items():
            _h = [g for g in _logs if g["group"] == "hitting"]
            _p = [g for g in _logs if g["group"] == "pitching"]
            _rec: dict = {}
            if _h:
                for _k, _v in _agg_hit(_h).items():
                    _rec[f"{_k}_ytd"] = _v
                for _k, _v in _agg_hit(_recent(_h)).items():
                    _rec[f"{_k}_l14"] = _v
            if _p:
                for _k, _v in _agg_pit(_p).items():
                    _rec[f"{_k}_ytd"] = _v
                for _k, _v in _agg_pit(_recent(_p)).items():
                    _rec[f"{_k}_l14"] = _v
            if _rec:
                _by_id[_pid] = _rec

        _rows = []
        for _, _r in players.iterrows():
            _mid = int(_r["MLBAMID"]) if pd.notna(_r["MLBAMID"]) else None
            _rows.append(_by_id.get(_mid, {}) if _mid is not None else {})
        _obs_df = pd.DataFrame(_rows, index=players.index)
        for _c in _window_cols:
            if _c not in _obs_df.columns:
                _obs_df[_c] = pd.NA
        players_obs = pd.concat([players, _obs_df[_window_cols]], axis=1)
        print(
            f"Observed windows attached: {len(_by_id)} players with game-log data "
            f"(YTD + last 14d through {_cutoff.date()})"
        )
    return (players_obs,)


@app.cell
def ytd_window_control(mo):
    """Window selector for the actual-vs-projection table."""
    ytd_window = mo.ui.radio(
        options=["Year to date", "Last 14 days"],
        value="Year to date",
        label="Observed window",
    )
    return (ytd_window,)


@app.cell
def ytd_comparison(
    data_ready,
    mo,
    my_roster_names,
    pd,
    players_obs,
    strip_name_suffix,
    ytd_window,
):
    """Observed actuals vs. rest-of-season projection rate for the user's roster.

    Counting-stat cells show `actual / expected`, where expected applies the
    RoS projection's per-PA (or per-IP) rate to the playing time the player has
    actually accrued in the selected window. This makes the two directly
    comparable and pace-independent. Color = rate over/under-performance vs the
    projection (green over, red under). Rate stats (OPS/ERA/WHIP) compare
    directly to the projected rate.
    """
    if not data_ready:
        ytd_section = mo.md("")
    else:
        _w = "l14" if ytd_window.value == "Last 14 days" else "ytd"
        _roster = players_obs[players_obs["Name"].isin(my_roster_names)]

        def _wv(row: pd.Series, col: str) -> float:
            v = row.get(col)
            return float(v) if v is not None and not pd.isna(v) else 0.0

        def _ratio_count(actual: float, expected: float) -> float | None:
            # actual vs the projection-rate expectation over the same volume.
            if expected <= 0:
                return None
            return actual / expected

        def _ratio_higher(actual: float, ref: float) -> float | None:
            # Rate stat, higher is better (OPS): actual rate vs projected rate.
            if ref <= 0:
                return None
            return actual / ref

        def _ratio_lower(actual: float, ref: float) -> float | None:
            # Rate stat, lower is better (ERA, WHIP); invert so >1 = good.
            if ref <= 0:
                return None
            if actual <= 0:
                return 2.0  # zero ERA/WHIP is elite; clamp to full green
            return ref / actual

        def _residual_color(ratio: float | None) -> dict[str, str]:
            # Map a performance ratio (1.0 == on projected rate) to a green
            # (over) / red (under) background. Saturates at +/-20% deviation.
            if ratio is None:
                return {}
            dev = ratio - 1.0
            alpha = round(min(abs(dev) / 0.20, 1.0) * 0.55, 3)
            if alpha == 0:
                return {}
            rgb = "34,160,34" if dev > 0 else "210,45,45"
            return {"backgroundColor": f"rgba({rgb},{alpha})"}

        def _avg_ratio(d: dict[str, float | None]) -> float:
            vals = [v for v in d.values() if v is not None]
            return sum(vals) / len(vals) if vals else float("-inf")

        _h_rows: list[dict] = []
        _p_rows: list[dict] = []
        _h_ratios_by_player: dict[str, dict[str, float | None]] = {}
        _p_ratios_by_player: dict[str, dict[str, float | None]] = {}
        for _, _r in _roster.iterrows():
            _name = strip_name_suffix(_r["Name"])

            if _r["player_type"] == "hitter":
                _vol = _wv(_r, f"PA_{_w}")
                _has = _vol > 0
                _proj_pa = float(_r["PA"])
                _ratios: dict[str, float | None] = {}
                _row: dict = {
                    "Player": _name,
                    "Pos": _r["Position"],
                    "G": int(_wv(_r, f"G_{_w}")),
                    "PA": f"{_vol:.0f}" if _has else "—",
                }
                for _stat, _dec in (("R", 0), ("HR", 0), ("RBI", 0), ("SB", 0)):
                    _actual = _wv(_r, f"{_stat}_{_w}")
                    _rate = float(_r[_stat]) / _proj_pa if _proj_pa > 0 else 0.0
                    _exp = _rate * _vol
                    _row[_stat] = f"{_actual:.0f} / {_exp:.1f}" if _has else "—"
                    _ratios[_stat] = _ratio_count(_actual, _exp) if _has else None
                _ops_actual = _wv(_r, f"OPS_{_w}")
                _ops_ref = float(_r["OPS"])
                _row["OPS"] = f"{_ops_actual:.3f} / {_ops_ref:.3f}" if _has else "—"
                _ratios["OPS"] = _ratio_higher(_ops_actual, _ops_ref) if _has else None
                _h_rows.append(_row)
                _h_ratios_by_player[_name] = _ratios
            else:
                _vol = _wv(_r, f"IP_{_w}")
                _has = _vol > 0
                _proj_ip = float(_r["IP"])
                _ratios = {}
                _row = {
                    "Player": _name,
                    "Pos": _r["Position"],
                    "G": int(_wv(_r, f"G_{_w}")),
                    "IP": f"{_vol:.1f}" if _has else "—",
                }
                for _stat in ("W", "SV", "K"):
                    _actual = _wv(_r, f"{_stat}_{_w}")
                    _rate = float(_r[_stat]) / _proj_ip if _proj_ip > 0 else 0.0
                    _exp = _rate * _vol
                    _row[_stat] = f"{_actual:.0f} / {_exp:.1f}" if _has else "—"
                    _ratios[_stat] = _ratio_count(_actual, _exp) if _has else None
                for _stat in ("ERA", "WHIP"):
                    _actual = _wv(_r, f"{_stat}_{_w}")
                    _ref = float(_r[_stat])
                    _row[_stat] = f"{_actual:.2f} / {_ref:.2f}" if _has else "—"
                    _ratios[_stat] = _ratio_lower(_actual, _ref) if _has else None
                _p_rows.append(_row)
                _p_ratios_by_player[_name] = _ratios

        _h_df = (
            pd.DataFrame(_h_rows)
            .sort_values(
                "Player",
                key=lambda s: s.map(
                    lambda n: _avg_ratio(_h_ratios_by_player.get(n, {}))
                ),
                ascending=False,
            )
            .reset_index(drop=True)
        )
        _p_df = (
            pd.DataFrame(_p_rows)
            .sort_values(
                "Player",
                key=lambda s: s.map(
                    lambda n: _avg_ratio(_p_ratios_by_player.get(n, {}))
                ),
                ascending=False,
            )
            .reset_index(drop=True)
        )

        _h_styles = {
            str(_i): _h_ratios_by_player.get(_row["Player"], {})
            for _i, _row in _h_df.iterrows()
        }
        _p_styles = {
            str(_i): _p_ratios_by_player.get(_row["Player"], {})
            for _i, _row in _p_df.iterrows()
        }

        def _h_style_cell(row_id: str, name: str, value: object) -> dict[str, str]:
            return _residual_color(_h_styles.get(row_id, {}).get(name))

        def _p_style_cell(row_id: str, name: str, value: object) -> dict[str, str]:
            return _residual_color(_p_styles.get(row_id, {}).get(name))

        _label = "year-to-date" if _w == "ytd" else "last 14 days"
        ytd_section = mo.vstack(
            [
                ytd_window,
                mo.md(
                    f"*Window: **{_label}**. Counting cells: "
                    "`actual / expected`, where expected = the rest-of-season "
                    "projection's per-PA (or per-IP) rate applied to the "
                    "playing time accrued in this window. Rate cells (OPS/ERA/"
                    "WHIP): `actual / projected rate`. Color = performance vs. "
                    "the projected rate: <span style='background-color:"
                    "rgba(34,160,34,0.45);padding:0 4px'>green = over</span>, "
                    "<span style='background-color:rgba(210,45,45,0.45);"
                    "padding:0 4px'>red = under</span> (saturates at \u00b120%).*"
                ),
                mo.md("**Hitters**"),
                mo.ui.table(_h_df, page_size=30, style_cell=_h_style_cell),
                mo.md("**Pitchers**"),
                mo.ui.table(_p_df, page_size=30, style_cell=_p_style_cell),
            ]
        )
    return (ytd_section,)


@app.cell
def derived_state(data_ready, players, state):
    if data_ready:
        my_roster_names = state["my_roster_names"]
        my_bench = my_roster_names - state["my_starters"]
        current_total_bv = float(players[players["Name"].isin(my_bench)]["BV"].sum())
        opponent_rosters = state["opponent_rosters"]
    else:
        my_roster_names = set()
        current_total_bv = 0.0
        opponent_rosters = {}
    return current_total_bv, my_roster_names, opponent_rosters


@app.cell
def resolve_name_cell(strip_name_suffix):
    import unicodedata

    def _normalize(s: str) -> str:
        """Strip diacritics and lowercase for accent-insensitive matching."""
        nfkd = unicodedata.normalize("NFKD", s)
        return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()

    def resolve_name(
        display_name: str, players_df, player_type: str | None = None
    ) -> str | None:
        """Resolve user-typed display name to internal Name with -H/-P suffix.

        Accent-insensitive: 'Edwin Diaz' matches 'Edwin Díaz-P'.
        Returns None if no match found (caller should show UI message).
        """
        name = display_name.strip()
        if name.endswith("-H") or name.endswith("-P"):
            if name in players_df["Name"].values:
                return name
            return None
        all_names = players_df["Name"].tolist()
        query = _normalize(name)
        candidates = [n for n in all_names if _normalize(strip_name_suffix(n)) == query]
        if player_type:
            suffix = "-H" if player_type == "hitter" else "-P"
            candidates = [c for c in candidates if c.endswith(suffix)]
        return candidates[0] if candidates else None

    return (resolve_name,)


# ============================================================================
# Dashboard tab
# ============================================================================


@app.cell
def title_odds(data_ready, mo, pd, state):
    """Monte Carlo final-standings simulation: P(win), P(top-2), point spread.

    EW is risk-neutral (expected points); this panel shows the placement
    DISTRIBUTION — what actually matters for deciding how much variance to
    seek as the season tightens.
    """
    if data_ready:
        from optimizer.win_model import simulate_standings

        _sim = simulate_standings(
            state["my_totals"],
            state["opponent_totals"],
            state["category_sigmas"],
        )
        _names = [state["my_team_name"]] + [
            state["opponent_teams"][i - 1]
            for i in sorted(state["opponent_lineups"].keys())
        ]
        _rows = [
            {
                "Team": _names[i],
                "Expected Pts": round(_sim["expected_points"][i], 1),
                "P(win) %": round(100 * _sim["p_win"][i], 1),
                "P(top 2) %": round(100 * _sim["p_top2"][i], 1),
            }
            for i in range(len(_names))
        ]
        _odds_df = (
            pd.DataFrame(_rows)
            .sort_values("Expected Pts", ascending=False)
            .reset_index(drop=True)
        )
        _q = _sim["my_points_quantiles"]
        title_odds_section = mo.vstack(
            [
                mo.md("### Title Odds (Monte Carlo, 20k seasons)"),
                mo.md(
                    f"*My final points: 5th pct **{_q[5]:.0f}**, median "
                    f"**{_q[50]:.0f}**, 95th pct **{_q[95]:.0f}** (of 70).*"
                ),
                mo.ui.table(_odds_df, selection=None, page_size=10),
            ]
        )
    else:
        title_odds_section = mo.md("")
    return (title_odds_section,)


@app.cell
def dashboard_content(
    data_ready,
    fig_to_png,
    mo,
    my_roster_names,
    pd,
    players,
    projection_system,
    rebuild_data,
    roster_map_section,
    state,
    strip_name_suffix,
    team_perspective,
    title_odds_section,
    ytd_section,
):
    if data_ready:
        from optimizer.swap_evaluator import compute_ew_ceiling
        from optimizer.visualizations import (
            plot_category_heatmap,
            plot_gap_to_ceiling,
            plot_starter_contributions,
        )
        from optimizer.win_model import compute_category_regime

        _ew = state["current_ew"]
        _starters_fig = plot_starter_contributions(state["my_lineup"], players)
        _heatmap_fig = plot_category_heatmap(
            state["my_totals"],
            state["opponent_totals"],
            state["opponent_teams"],
        )

        # --- My Roster table with Status and Optimal Slot ---
        _rdf = players[players["Name"].isin(my_roster_names)].copy()
        _rdf["Player"] = _rdf["Name"].apply(strip_name_suffix)
        _rdf["Status"] = _rdf["roster_status"].str.capitalize()
        _rdf["Optimal Slot"] = _rdf["optimal_slot"].fillna("Bench")
        _is_st = _rdf["optimal_slot"].notna()
        _sorted = pd.concat(
            [
                _rdf[_is_st].sort_values("optimal_slot"),
                _rdf[~_is_st].sort_values("MEW", ascending=False),
            ]
        )
        _roster_cols = [
            "Player",
            "Position",
            "Status",
            "Optimal Slot",
            "MEW",
            "BV",
            "FV",
            "PV",
        ]
        _display = _sorted[_roster_cols].round(2).reset_index(drop=True)
        _roster_table = mo.ui.table(_display, page_size=50)

        # --- Action tables ---
        _my_starters = set(state["my_lineup"].keys())
        _my_bench = my_roster_names - _my_starters

        _bench_df = players[players["Name"].isin(_my_bench)].copy()
        _bench_df["Player"] = _bench_df["Name"].apply(strip_name_suffix)

        from optimizer.swap_evaluator import (
            evaluate_top_k,
            find_protected_players,
            screen_swaps,
            validate_transaction,
        )

        _current_total_bv = float(players[players["Name"].isin(_my_bench)]["BV"].sum())

        _screened = screen_swaps(
            players,
            my_roster_names,
            state["my_lineup"],
            top_k=50,
        )

        if len(_screened) > 0:
            _evaluated = evaluate_top_k(
                _screened.head(20),
                my_roster_names,
                players,
                state["opponent_totals"],
                state["category_sigmas"],
                state["current_ew"],
                _current_total_bv,
                include_bv=True,
                my_banked_totals=state["my_banked"],
                my_lineup=state["my_lineup"],
            )
            _positive = _evaluated[_evaluated["value"] > 0]
            _n_fa_upgrades = len(_positive)
        else:
            _positive = pd.DataFrame()
            _n_fa_upgrades = 0

        _pos_lookup = players.set_index("Name")["Position"].to_dict()
        _mew_lookup = players.set_index("Name")["MEW"].to_dict()
        _bv_lookup = players.set_index("Name")["BV"].to_dict()

        # --- Section 1: Best FA Pickups (deduplicated by FA) ---
        if len(_positive) > 0:
            _seen_fa = set()
            _pickup_rows = []
            for _, row in _positive.iterrows():
                _fn = row["fa_name"]
                if _fn in _seen_fa:
                    continue
                _seen_fa.add(_fn)
                _pickup_rows.append(
                    {
                        "Player": strip_name_suffix(_fn),
                        "Position": _pos_lookup.get(_fn, ""),
                        "MEW": round(_mew_lookup.get(_fn, 0), 2),
                        "Swap Value": round(row["value"], 2),
                    }
                )
            _fa_pickups = pd.DataFrame(_pickup_rows)
        else:
            _fa_pickups = pd.DataFrame(
                columns=["Player", "Position", "MEW", "Swap Value"]
            )

        # --- Section 2: Droppable Players (all expendable bench, ranked by BV) ---
        _protected = find_protected_players(my_roster_names, players)
        _droppable_names = _my_bench - _protected
        _droppable_df = players[players["Name"].isin(_droppable_names)].copy()
        _droppable_df = _droppable_df.sort_values(["BV", "MEW"], ascending=[True, True])
        _droppable_disp = pd.DataFrame(
            [
                {
                    "Player": strip_name_suffix(row["Name"]),
                    "Position": row["Position"],
                    "MEW": round(row["MEW"], 2),
                    "Drop Cost (BV)": round(row["BV"], 2),
                }
                for _, row in _droppable_df.iterrows()
            ]
        )

        # --- Section 3: Recommended Swaps (concrete, validated, exact-evaluated) ---
        if len(_positive) > 0:
            _seen_pairs = set()
            _swap_rows = []
            for _, row in _positive.iterrows():
                _fn = row["fa_name"]
                _dn = row["drop_name"]
                _pair = (_fn, _dn)
                if _pair in _seen_pairs:
                    continue
                _seen_pairs.add(_pair)
                _val = validate_transaction({_dn}, {_fn}, my_roster_names, players)
                if not _val["valid"]:
                    continue
                _swap_rows.append(
                    {
                        "Add": strip_name_suffix(_fn),
                        "Drop": strip_name_suffix(_dn),
                        "ΔEW": round(row["msv_exact"], 2),
                        "ΔBV": round(row["delta_bv"], 2),
                        "Value": round(row["value"], 2),
                        "New EW": round(row["new_ew"], 2),
                    }
                )
            _swaps_disp = pd.DataFrame(_swap_rows).head(5)
        else:
            _swaps_disp = pd.DataFrame(
                columns=["Add", "Drop", "ΔEW", "ΔBV", "Value", "New EW"]
            )

        # --- Should Start / Should Bench ---
        _should_start = (
            _rdf[
                (_rdf["roster_status"].isin(["reserve", "taxi"]))
                & (_rdf["optimal_slot"].notna())
            ]
            .sort_values("MEW", ascending=False)
            .copy()
        )
        _start_disp = (
            _should_start[["Player", "Position", "Optimal Slot", "MEW"]]
            .round(2)
            .reset_index(drop=True)
        )

        _should_bench = (
            _rdf[(_rdf["roster_status"] == "active") & (_rdf["optimal_slot"].isna())]
            .sort_values("MEW")
            .copy()
        )
        _bench_disp = (
            _should_bench[["Player", "Position", "MEW"]].round(2).reset_index(drop=True)
        )

        def _action_table(title: str, df, page_size: int = 50):
            if len(df) == 0:
                return mo.vstack([mo.md(f"#### {title}"), mo.md("*None*")])
            return mo.vstack(
                [
                    mo.md(f"#### {title}"),
                    mo.ui.table(df, page_size=page_size),
                ]
            )

        _n_safe_drops = len(_droppable_disp[_droppable_disp["Drop Cost (BV)"] < 0.1])
        _actions = mo.vstack(
            [
                mo.md(
                    f"**FA Swaps** ({_n_fa_upgrades} improving pickups, "
                    f"{_n_safe_drops} safe drops)"
                ),
                mo.hstack(
                    [
                        _action_table("Best FA Pickups", _fa_pickups),
                        _action_table(
                            "Droppable Players (lowest BV = safest drop)",
                            _droppable_disp,
                        ),
                    ],
                    widths="equal",
                ),
                _action_table("Recommended Swaps (fully evaluated)", _swaps_disp),
                mo.hstack(
                    [
                        _action_table("Should Start (currently reserve)", _start_disp),
                        _action_table("Should Bench (currently active)", _bench_disp),
                    ],
                    widths="equal",
                ),
            ]
        )

        _ceiling = compute_ew_ceiling(
            players,
            state["opponent_totals"],
            state["category_sigmas"],
            my_roster_names,
            _ew,
            my_team_name=state["my_team_name"],
            my_banked_totals=state["my_banked"],
        )
        _gap_fig = plot_gap_to_ceiling(_ew, _ceiling["ceiling_ew"])

        # --- Category Regime table ---
        _regime_df = compute_category_regime(
            state["my_totals"],
            state["opponent_totals"],
            state["category_sigmas"],
            state["gradient"],
        )
        _regime_display = _regime_df.rename(
            columns={
                "category": "Category",
                "avg_z": "Avg z",
                "gradient": "Gradient (g_c)",
                "convexity_ratio": "Convexity Ratio",
                "regime": "Regime",
            }
        )[["Category", "Avg z", "Gradient (g_c)", "Convexity Ratio", "Regime"]]
        _regime_display = _regime_display.round(2)
        _regime_table = mo.ui.table(_regime_display, page_size=10)

        _undervalued = _regime_df[_regime_df["convexity_ratio"] > 1.05]
        if len(_undervalued) > 0:
            _uv_parts = ", ".join(
                f"**{row['category']}** ({row['convexity_ratio']:.0%})"
                for _, row in _undervalued.iterrows()
            )
            _regime_callout = mo.callout(
                mo.md(
                    f"Gradient undervalues improvements in: {_uv_parts}. "
                    f"Actual EW gains exceed MEW estimates — "
                    f"trades shifting these categories deserve exact EW evaluation."
                ),
                kind="warn",
            )
        else:
            _regime_callout = mo.md("")

        _controls = mo.hstack(
            [projection_system, team_perspective, rebuild_data], justify="start", gap=1
        )

        dashboard_tab = mo.vstack(
            [
                _controls,
                mo.md(
                    f"**{state['my_team_name']}** — "
                    f"Expected Wins: {_ew:.2f} / 60 "
                    f"| Projected Standing Points: {10 + _ew:.1f} / 70"
                ),
                title_odds_section,
                mo.md("### Recommended Actions"),
                fig_to_png(_gap_fig),
                _actions,
                fig_to_png(_starters_fig),
                fig_to_png(_heatmap_fig),
                mo.md("### Category Regime (Convexity Diagnostics)"),
                _regime_callout,
                _regime_table,
                mo.md(f"### {state['my_team_name']} Roster"),
                _roster_table,
                mo.md("### Actual YTD vs. Projection"),
                ytd_section,
                mo.md("### Roster Value Map"),
                roster_map_section,
            ]
        )
    else:
        _controls = mo.hstack(
            [projection_system, team_perspective, rebuild_data],
            justify="start",
            gap=1,
        )
        dashboard_tab = mo.vstack(
            [
                _controls,
                mo.callout(
                    mo.md(
                        "**No data available.** Click *Rebuild pipeline* to scrape "
                        "projections and build the silver table."
                    ),
                    kind="warn",
                ),
            ]
        )
    return (dashboard_tab,)


# ============================================================================
# FA Optimizer
# ============================================================================


@app.cell
def fa_ui(mo):
    run_fa = mo.ui.button(
        value=0,
        on_click=lambda v: v + 1,
        label="Run FA Optimizer",
    )
    fa_value_threshold = mo.ui.slider(
        start=0.0,
        stop=0.5,
        step=0.01,
        value=0.10,
        label="Min value threshold",
        show_value=True,
    )
    return fa_value_threshold, run_fa


@app.cell
def fa_results(
    banked_totals,
    data_ready,
    fa_value_threshold,
    mo,
    pd,
    players_fv,
    run_fa,
    season_frac,
    state,
    strip_name_suffix,
):
    if not data_ready or not run_fa.value:
        fa_table = None
        fa_moves = []
        fa_section = mo.md(
            "*Click 'Run FA Optimizer' to find the best free-agent swaps.*"
        )
    else:
        from optimizer.swap_evaluator import run_greedy_optimization

        _res = run_greedy_optimization(
            players_fv,
            value_threshold=fa_value_threshold.value,
            my_team_name=state["my_team_name"],
            banked_totals=banked_totals,
            season_fraction_remaining=season_frac,
        )
        fa_moves = _res["moves"]
        if not fa_moves:
            fa_table = None
            fa_section = mo.md(f"No improving moves found. EW = {_res['final_ew']:.2f}")
        else:
            _mdf = pd.DataFrame(
                [
                    {
                        "Drop": strip_name_suffix(m["drop"]),
                        "Add": strip_name_suffix(m["add"]),
                        "\u0394EW": round(m["delta_ew"], 2),
                        "\u0394BV": round(m["delta_bv"], 2),
                        "Value": round(m["value"], 2),
                    }
                    for m in fa_moves
                ]
            )
            fa_table = mo.ui.table(_mdf, selection="single", page_size=50)
            fa_section = mo.vstack(
                [
                    mo.md(
                        f"**{len(fa_moves)} moves found** | "
                        f"EW: {_res['starting_ew']:.2f} \u2192 "
                        f"{_res['final_ew']:.2f} | "
                        f"Total value: {_res['total_value']:.2f}"
                    ),
                    fa_table,
                ]
            )
    return fa_moves, fa_section, fa_table


@app.cell
def fa_impact(
    data_ready,
    fa_moves,
    fa_table,
    fig_to_png,
    mo,
    my_roster_names,
    players,
    state,
):
    if data_ready and fa_table is not None and len(fa_table.value) > 0:
        from optimizer.swap_evaluator import compute_exact_msv as _compute_exact_msv
        from optimizer.visualizations import plot_move_impact as _pmi

        _idx = fa_table.value.index[0]
        _move = fa_moves[_idx]
        _res = _compute_exact_msv(
            drop_names={_move["drop"]},
            add_names={_move["add"]},
            my_roster_names=my_roster_names,
            players=players,
            opponent_totals=state["opponent_totals"],
            category_sigmas=state["category_sigmas"],
            current_ew=state["current_ew"],
            my_banked_totals=state["my_banked"],
            baseline_lineup=state["my_lineup"],
        )
        _fig = _pmi(
            state["my_totals"],
            _res["new_totals"],
            state["current_ew"],
            _res["new_ew"],
        )
        fa_impact_section = fig_to_png(_fig)
    else:
        fa_impact_section = mo.md("")
    return (fa_impact_section,)


# ============================================================================
# Unified Trade section (evaluator + search)
# ============================================================================


@app.cell
def trade_controls(data_ready, mo, state):
    _opts = state["opponent_teams"] if data_ready else []
    trade_send = mo.ui.text(label="Send (comma-separated)", placeholder="Your players")
    trade_recv = mo.ui.text(
        label="Receive (comma-separated)", placeholder="Target players"
    )
    trade_opp = mo.ui.dropdown(label="Opponent", options=[""] + _opts)
    trade_pv_loss_pct = mo.ui.slider(
        start=0,
        stop=50,
        step=5,
        value=15,
        label="Max opp. PV loss %",
        show_value=True,
    )
    trade_min_value = mo.ui.slider(
        start=-0.5,
        stop=0.5,
        step=0.05,
        value=0.0,
        label="Min Value",
        show_value=True,
    )
    trade_min_msv = mo.ui.slider(
        start=-1.0,
        stop=1.0,
        step=0.05,
        value=-0.1,
        label="Min MSV (ΔEWA)",
        show_value=True,
    )
    trade_search_btn = mo.ui.button(
        value=0,
        on_click=lambda v: v + 1,
        label="Search Trades",
    )
    return (
        trade_min_msv,
        trade_min_value,
        trade_opp,
        trade_pv_loss_pct,
        trade_recv,
        trade_search_btn,
        trade_send,
    )


@app.cell
def trade_results(
    current_total_bv,
    data_ready,
    fig_to_png,
    mo,
    my_roster_names,
    opponent_rosters: dict[int, set[str]],
    pd,
    players,
    resolve_name,
    state,
    strip_name_suffix,
    trade_min_msv,
    trade_min_value,
    trade_opp,
    trade_pv_loss_pct,
    trade_recv,
    trade_search_btn,
    trade_send,
):
    _has_send = bool(trade_send.value and trade_send.value.strip())
    _has_recv = bool(trade_recv.value and trade_recv.value.strip())
    _has_opp = trade_opp.value is not None and trade_opp.value != ""
    _search_clicked = trade_search_btn.value > 0

    if not data_ready:
        trade_section = mo.md("")
        trade_results_data = []
        trade_results_table = None
    elif _has_send and _has_recv:
        # --- Evaluate a specific trade ---
        from optimizer.trade_finder import evaluate_trade as _eval_trade
        from optimizer.visualizations import plot_move_impact as _pmi

        _send_raw = [n.strip() for n in trade_send.value.split(",") if n.strip()]
        _recv_raw = [n.strip() for n in trade_recv.value.split(",") if n.strip()]
        _send = [resolve_name(n, players) for n in _send_raw]
        _recv = [resolve_name(n, players) for n in _recv_raw]
        _bad = [
            name
            for name, resolved in zip(_send_raw + _recv_raw, _send + _recv)
            if resolved is None
        ]

        if _bad:
            trade_section = mo.callout(
                mo.md(f"**Player(s) not found:** {', '.join(_bad)}. Check spelling."),
                kind="danger",
            )
            trade_results_data = []
            trade_results_table = None
        else:
            _recv_from_opp = []
            _recv_from_fa = []
            for _rn in _recv:
                _owner = players.loc[players["Name"] == _rn, "owner"].iloc[0]
                if pd.isna(_owner):
                    _recv_from_fa.append(_rn)
                else:
                    _recv_from_opp.append(_rn)

            if _recv_from_opp and not _has_opp:
                trade_section = mo.callout(
                    mo.md(
                        "**Select an opponent** to receive players from their roster."
                    ),
                    kind="warn",
                )
                trade_results_data = []
                trade_results_table = None
            else:
                if _has_opp:
                    _oid = state["opponent_teams"].index(trade_opp.value) + 1
                    _opp_roster = opponent_rosters[_oid]
                else:
                    _oid = 1
                    _opp_roster = set()

                _res = _eval_trade(
                    send_names=set(_send),
                    receive_names=set(_recv_from_opp + _recv_from_fa),
                    my_roster_names=my_roster_names,
                    opponent_roster_names=_opp_roster,
                    trade_opponent_id=_oid,
                    players=players,
                    opponent_totals=state["opponent_totals"],
                    category_sigmas=state["category_sigmas"],
                    current_ew=state["current_ew"],
                    current_total_bv=current_total_bv,
                    pv_max_loss_frac=trade_pv_loss_pct.value / 100,
                    my_lineup=state["my_lineup"],
                    my_banked_totals=state["my_banked"],
                    trade_opponent_banked=state["opponent_banked"].get(_oid),
                )

                if "error" in _res:
                    trade_section = mo.callout(
                        mo.md(f"**Error:** {_res['error']}"),
                        kind="danger",
                    )
                else:
                    _auto_notes = []
                    if _res.get("auto_fa_add"):
                        _auto_notes.append(
                            f"Auto-add FA: **{strip_name_suffix(_res['auto_fa_add'])}**"
                        )
                    if _res.get("auto_drop"):
                        _auto_notes.append(
                            f"Auto-drop: **{strip_name_suffix(_res['auto_drop'])}**"
                        )
                    _auto_line = " | ".join(_auto_notes) if _auto_notes else ""

                    _callout = mo.callout(
                        mo.md(
                            f"**MSV:** {_res['msv']:+.2f} | "
                            f"**ΔBV:** {_res['delta_bv']:+.2f} | "
                            f"**Value:** {_res['value']:+.2f}\n\n"
                            f"**New EW:** {_res['new_ew']:.2f} | "
                            f"**Opp PV loss:** {_res['opp_pv_loss_pct']:+.1f}% | "
                            f"**PV feasible:** "
                            f"{'Yes' if _res['pv_feasible'] else 'No'}"
                            + (f"\n\n{_auto_line}" if _auto_line else "")
                        ),
                        kind="success" if _res["pv_feasible"] else "danger",
                    )
                    _parts = [_callout]
                    if _res["new_totals"]:
                        _fig = _pmi(
                            state["my_totals"],
                            _res["new_totals"],
                            state["current_ew"],
                            _res["new_ew"],
                        )
                        _parts.append(fig_to_png(_fig))
                    trade_section = mo.vstack(_parts)
                trade_results_data = []
                trade_results_table = None
    elif _search_clicked:
        # --- Search trades (with optional filters) ---
        _opp_filter = None
        if _has_opp:
            _oid = state["opponent_teams"].index(trade_opp.value) + 1
            _opp_filter = {_oid}

        _opp_rosters_filtered = (
            {k: v for k, v in opponent_rosters.items() if k in _opp_filter}
            if _opp_filter
            else opponent_rosters
        )

        # Resolve send/recv names for filter
        _send_names = set()
        _recv_names = set()
        _bad_names = []
        if _has_send:
            for _n in trade_send.value.split(","):
                _n = _n.strip()
                if _n:
                    _resolved = resolve_name(_n, players)
                    if _resolved:
                        _send_names.add(_resolved)
                    else:
                        _bad_names.append(_n)
        if _has_recv:
            for _n in trade_recv.value.split(","):
                _n = _n.strip()
                if _n:
                    _resolved = resolve_name(_n, players)
                    if _resolved:
                        _recv_names.add(_resolved)
                    else:
                        _bad_names.append(_n)

        if _bad_names:
            trade_results_data = []
            trade_results_table = None
            trade_section = mo.callout(
                mo.md(
                    f"**Player(s) not found:** {', '.join(_bad_names)}. Check spelling."
                ),
                kind="danger",
            )
        elif _send_names or _recv_names:
            # Use targeted search that guarantees must_send/must_receive are included
            from optimizer.trade_finder import (
                search_trades_for_players as _search_targeted,
            )

            trade_results_data = _search_targeted(
                players=players,
                my_roster_names=my_roster_names,
                my_lineup=state["my_lineup"],
                opponent_rosters=_opp_rosters_filtered,
                opponent_totals=state["opponent_totals"],
                category_sigmas=state["category_sigmas"],
                current_ew=state["current_ew"],
                current_total_bv=current_total_bv,
                pv_max_loss_frac=trade_pv_loss_pct.value / 100,
                must_send=_send_names or None,
                must_receive=_recv_names or None,
                opponent_filter=_opp_filter,
                min_value=trade_min_value.value,
                my_team_name=state["my_team_name"],
                my_banked_totals=state["my_banked"],
                opponent_banked=state["opponent_banked"],
            )
        else:
            # No player filters — use general search
            from optimizer.trade_finder import search_trades as _search

            trade_results_data = _search(
                players=players,
                my_roster_names=my_roster_names,
                my_lineup=state["my_lineup"],
                opponent_rosters=_opp_rosters_filtered,
                opponent_totals=state["opponent_totals"],
                category_sigmas=state["category_sigmas"],
                current_ew=state["current_ew"],
                current_total_bv=current_total_bv,
                pv_max_loss_frac=trade_pv_loss_pct.value / 100,
                min_value=trade_min_value.value,
                my_team_name=state["my_team_name"],
                my_banked_totals=state["my_banked"],
                opponent_banked=state["opponent_banked"],
            )

        if not _bad_names:
            # Apply MSV filter
            trade_results_data = [
                t for t in trade_results_data if t["msv_exact"] >= trade_min_msv.value
            ]

            if not trade_results_data:
                trade_results_table = None
                trade_section = mo.callout(
                    mo.md("No trades found matching filters."),
                    kind="warn",
                )
            else:
                # Sort by Value descending
                trade_results_data.sort(key=lambda t: t["value"], reverse=True)
                _tdf = pd.DataFrame(
                    [
                        {
                            "Send": " + ".join(strip_name_suffix(n) for n in t["send"]),
                            "Receive": " + ".join(
                                strip_name_suffix(n) for n in t["receive"]
                            ),
                            "Opponent": t["opponent"],
                            "Value": round(t["value"], 2),
                            "MSV": round(t["msv_exact"], 2),
                            "ΔBV": round(t["delta_bv"], 2),
                            "Opp PV Loss%": round(t["opp_pv_loss_pct"], 1),
                            "New EW": round(t["new_ew"], 2),
                        }
                        for t in trade_results_data
                    ]
                )
                trade_results_table = mo.ui.table(
                    _tdf, selection="single", page_size=50
                )
                trade_section = mo.vstack(
                    [
                        mo.md(
                            f"**{len(trade_results_data)} trades found** (sorted by Value)"
                        ),
                        trade_results_table,
                    ]
                )
    else:
        trade_section = mo.md(
            "*Enter send + receive for a specific evaluation, "
            "or click **Search Trades** to find all PV-feasible trades. "
            "Send/receive/opponent fields act as filters on search.*"
        )
        trade_results_data = []
        trade_results_table = None
    return trade_results_data, trade_results_table, trade_section


@app.cell
def trade_impact(
    current_total_bv,
    data_ready,
    fig_to_png,
    mo,
    my_roster_names,
    opponent_rosters: dict[int, set[str]],
    pd,
    players,
    state,
    strip_name_suffix,
    trade_pv_loss_pct,
    trade_results_data,
    trade_results_table,
):
    if (
        data_ready
        and trade_results_table is not None
        and len(trade_results_table.value) > 0
    ):
        from optimizer.trade_finder import evaluate_trade as _eval_trade
        from optimizer.visualizations import plot_move_impact as _pmi

        _idx = trade_results_table.value.index[0]
        _trade = trade_results_data[_idx]
        _oid = state["opponent_teams"].index(_trade["opponent"]) + 1

        _res = _eval_trade(
            send_names=set(_trade["send"]),
            receive_names=set(_trade["receive"]),
            my_roster_names=my_roster_names,
            opponent_roster_names=opponent_rosters[_oid],
            trade_opponent_id=_oid,
            players=players,
            opponent_totals=state["opponent_totals"],
            category_sigmas=state["category_sigmas"],
            current_ew=state["current_ew"],
            current_total_bv=current_total_bv,
            pv_max_loss_frac=trade_pv_loss_pct.value / 100,
            my_lineup=state["my_lineup"],
            send_to_opp_names=set(_trade.get("send_to_opp", _trade["send"])),
            my_banked_totals=state["my_banked"],
            trade_opponent_banked=state["opponent_banked"].get(_oid),
        )

        _all_names = list(_trade["send"]) + list(_trade["receive"])
        _plookup = players.set_index("Name")
        _player_rows = []
        for _n in _all_names:
            _row = _plookup.loc[_n]
            _player_rows.append(
                {
                    "Player": strip_name_suffix(_n),
                    "Side": "Send" if _n in _trade["send"] else "Receive",
                    "Position": _row.get("Position", ""),
                    "FV": round(_row.get("FV", 0), 2),
                    "MEW": round(_row.get("MEW", 0), 2),
                    "PV": round(_row.get("PV", 0), 2),
                }
            )
        _player_df = pd.DataFrame(_player_rows)

        _parts = [
            mo.callout(
                mo.md(
                    f"**MSV:** {_res['msv']:+.2f} | "
                    f"**ΔBV:** {_res['delta_bv']:+.2f} | "
                    f"**Value:** {_res['value']:+.2f}\n\n"
                    f"**New EW:** {_res['new_ew']:.2f} | "
                    f"**Opp PV loss:** {_res['opp_pv_loss_pct']:+.1f}%"
                ),
                kind="success" if _res["pv_feasible"] else "danger",
            ),
            mo.ui.table(_player_df, selection=None),
        ]
        if _res["new_totals"]:
            _fig = _pmi(
                state["my_totals"],
                _res["new_totals"],
                state["current_ew"],
                _res["new_ew"],
            )
            _parts.append(fig_to_png(_fig))
        trade_impact_section = mo.vstack(_parts)
    else:
        trade_impact_section = mo.md("")
    return (trade_impact_section,)


# ============================================================================
# Moves tab assembly
# ============================================================================


@app.cell
def moves_assembly(
    data_ready,
    fa_impact_section,
    fa_section,
    fa_value_threshold,
    mo,
    run_fa,
    trade_controls_ui,
    trade_impact_section,
    trade_section,
):
    if data_ready:
        moves_content = mo.vstack(
            [
                mo.md("## Free Agent Optimizer"),
                mo.hstack([run_fa, fa_value_threshold], justify="start", gap=2),
                fa_section,
                fa_impact_section,
                mo.md("---"),
                mo.md("## Trades"),
                mo.md(
                    "*Enter send + receive for a specific evaluation, "
                    "or click **Search Trades** to find all PV-feasible trades. "
                    "Send/receive/opponent act as filters on search.*"
                ),
                trade_controls_ui,
                trade_section,
                trade_impact_section,
            ]
        )
    else:
        moves_content = mo.md("")
    return (moves_content,)


@app.cell
def trade_controls_layout(
    mo,
    trade_min_msv,
    trade_min_value,
    trade_opp,
    trade_pv_loss_pct,
    trade_recv,
    trade_search_btn,
    trade_send,
):
    trade_controls_ui = mo.vstack(
        [
            mo.hstack([trade_send, trade_recv, trade_opp]),
            mo.hstack(
                [trade_pv_loss_pct, trade_min_value, trade_min_msv, trade_search_btn],
                justify="start",
                gap=2,
            ),
        ]
    )
    return (trade_controls_ui,)


# ============================================================================
# Sandbox tab — arbitrary transaction evaluator
# ============================================================================


@app.cell
def sandbox_controls(mo):
    sandbox_out = mo.ui.text_area(
        label="Players OUT (one per line)",
        placeholder="e.g.\nMike Trout\nCorbin Burnes",
        rows=5,
        full_width=True,
    )
    sandbox_in = mo.ui.text_area(
        label="Players IN (one per line)",
        placeholder="e.g.\nJuan Soto\nGerrit Cole",
        rows=5,
        full_width=True,
    )
    sandbox_eval_btn = mo.ui.button(
        value=0,
        on_click=lambda v: v + 1,
        label="Evaluate Transaction",
    )
    return sandbox_eval_btn, sandbox_in, sandbox_out


@app.cell
def sandbox_results(
    current_total_bv,
    data_ready,
    fig_to_png,
    mo,
    my_roster_names,
    pd,
    players,
    resolve_name,
    sandbox_eval_btn,
    sandbox_in,
    sandbox_out,
    state,
    strip_name_suffix,
):
    if not data_ready or sandbox_eval_btn.value == 0:
        sandbox_section = mo.md(
            "Enter players to drop and add, then click **Evaluate Transaction**.\n\n"
            "- Put one player name per line in each box.\n"
            "- Players OUT leave your roster (trade away / drop).\n"
            "- Players IN join your roster (trade for / pick up from FA).\n"
            "- Unequal sides are OK — the system will warn about roster size changes.\n"
            "- Positional validity is checked before evaluation."
        )
    else:
        from optimizer.player_scoring import add_mew as _add_mew
        from optimizer.swap_evaluator import (
            add_bench_value as _add_bench_value,
        )
        from optimizer.swap_evaluator import (
            compute_exact_msv as _compute_exact_msv,
        )
        from optimizer.swap_evaluator import (
            validate_transaction as _validate_transaction,
        )
        from optimizer.visualizations import (
            plot_ew_category_decomposition as _plot_ew_decomp,
        )
        from optimizer.visualizations import (
            plot_player_comparison_radar as _plot_player_comparison_radar,
        )
        from optimizer.visualizations import (
            plot_player_contribution_waterfall as _plot_waterfall,
        )
        from optimizer.win_model import compute_ew_gradient as _compute_ew_gradient

        # Parse player names
        _out_raw = [n.strip() for n in sandbox_out.value.split("\n") if n.strip()]
        _in_raw = [n.strip() for n in sandbox_in.value.split("\n") if n.strip()]

        if not _out_raw and not _in_raw:
            sandbox_section = mo.callout(
                mo.md("Enter at least one player in either box."), kind="warn"
            )
        else:
            # Resolve names
            _out_resolved = []
            _in_resolved = []
            _bad_names = []

            for _name in _out_raw:
                _r = resolve_name(_name, players)
                if _r is None:
                    _bad_names.append(_name)
                else:
                    _out_resolved.append(_r)

            for _name in _in_raw:
                _r = resolve_name(_name, players)
                if _r is None:
                    _bad_names.append(_name)
                else:
                    _in_resolved.append(_r)

            if _bad_names:
                sandbox_section = mo.callout(
                    mo.md(
                        f"**Player(s) not found:** {', '.join(_bad_names)}.\n\n"
                        "Check spelling. Use `-H` or `-P` suffix for "
                        "two-way players (e.g., `Shohei Ohtani-H`)."
                    ),
                    kind="danger",
                )
            else:
                _drop_set = set(_out_resolved)
                _add_set = set(_in_resolved)

                # Validate positional feasibility
                _val = _validate_transaction(
                    _drop_set,
                    _add_set,
                    my_roster_names,
                    players,
                )

                _parts: list = []

                # Show warnings
                for _w in _val["warnings"]:
                    _parts.append(mo.callout(mo.md(_w), kind="warn"))

                if not _val["valid"]:
                    _err_md = "\n".join(f"- {e}" for e in _val["errors"])
                    _parts.append(
                        mo.callout(
                            mo.md(f"**Transaction blocked:**\n\n{_err_md}"),
                            kind="danger",
                        )
                    )
                    sandbox_section = mo.vstack(_parts)
                else:
                    # Pad unequal transaction to preserve roster size
                    _n_drop = len(_drop_set)
                    _n_add = len(_add_set)
                    _auto_notes: list[str] = []

                    if _n_drop > _n_add:
                        _deficit = _n_drop - _n_add
                        _mew_lookup = players.set_index("Name")["MEW"].to_dict()
                        _fa_pool = (
                            set(players[players["owner"].isna()]["Name"]) - _add_set
                        )
                        _fa_sorted = sorted(
                            _fa_pool,
                            key=lambda n: _mew_lookup.get(n, 0.0),
                            reverse=True,
                        )
                        for _i in range(_deficit):
                            if _i < len(_fa_sorted):
                                _add_set.add(_fa_sorted[_i])
                                _auto_notes.append(
                                    f"Auto-added FA: **{strip_name_suffix(_fa_sorted[_i])}**"
                                )
                    elif _n_add > _n_drop:
                        _deficit = _n_add - _n_drop
                        _mew_lookup = players.set_index("Name")["MEW"].to_dict()
                        _starters = set(state["my_lineup"].keys())
                        _bench = my_roster_names - _starters - _drop_set
                        _bench_sorted = sorted(
                            _bench, key=lambda n: _mew_lookup.get(n, 0.0)
                        )
                        for _i in range(_deficit):
                            if _i < len(_bench_sorted):
                                _drop_set.add(_bench_sorted[_i])
                                _auto_notes.append(
                                    f"Auto-dropped: **{strip_name_suffix(_bench_sorted[_i])}**"
                                )

                    if _auto_notes:
                        _parts.append(
                            mo.callout(
                                mo.md(
                                    "**Roster-size balancing:**\n\n"
                                    + "\n".join(f"- {n}" for n in _auto_notes)
                                ),
                                kind="info",
                            )
                        )

                    # Compute exact MSV
                    _msv_result = _compute_exact_msv(
                        _drop_set,
                        _add_set,
                        my_roster_names,
                        players,
                        state["opponent_totals"],
                        state["category_sigmas"],
                        state["current_ew"],
                        my_banked_totals=state["my_banked"],
                        baseline_lineup=state["my_lineup"],
                    )

                    # Compute ΔBV
                    _new_gradient = _compute_ew_gradient(
                        _msv_result["new_totals"],
                        state["opponent_totals"],
                        state["category_sigmas"],
                    )
                    _work = _add_mew(players, _msv_result["new_totals"], _new_gradient)
                    _new_roster = (my_roster_names - _drop_set) | _add_set
                    _scored = _add_bench_value(
                        _work, _msv_result["new_lineup"], _new_roster
                    )
                    _new_bench = _new_roster - set(_msv_result["new_lineup"].keys())
                    _new_total_bv = float(
                        _scored[_scored["Name"].isin(_new_bench)]["BV"].sum()
                    )
                    # Same-scale baseline: current roster's BV under the
                    # post-swap gradient (consistent MEW scale).
                    _base = _add_bench_value(_work, state["my_lineup"], my_roster_names)
                    _old_bench = my_roster_names - set(state["my_lineup"].keys())
                    _baseline_bv = float(
                        _base[_base["Name"].isin(_old_bench)]["BV"].sum()
                    )
                    _delta_bv = _new_total_bv - _baseline_bv
                    _value = _msv_result["msv"] + _delta_bv

                    # Summary callout
                    _parts.append(
                        mo.callout(
                            mo.md(
                                f"**ΔEW (MSV):** {_msv_result['msv']:+.2f} | "
                                f"**ΔBV:** {_delta_bv:+.2f} | "
                                f"**Total Value:** {_value:+.2f}\n\n"
                                f"**EW:** {state['current_ew']:.2f} → "
                                f"{_msv_result['new_ew']:.2f}"
                            ),
                            kind="success" if _value >= 0 else "warn",
                        )
                    )

                    # Player details table
                    _plookup = players.set_index("Name")
                    _player_rows = []
                    for _n in sorted(_drop_set):
                        _row = _plookup.loc[_n]
                        _player_rows.append(
                            {
                                "Player": strip_name_suffix(_n),
                                "Side": "OUT",
                                "Position": _row.get("Position", ""),
                                "Type": _row.get("player_type", ""),
                                "FV": round(float(_row.get("FV", 0)), 2),
                                "MEW": round(float(_row.get("MEW", 0)), 2),
                                "PV": round(float(_row.get("PV", 0)), 2),
                            }
                        )
                    for _n in sorted(_add_set):
                        _row = _plookup.loc[_n]
                        _player_rows.append(
                            {
                                "Player": strip_name_suffix(_n),
                                "Side": "IN",
                                "Position": _row.get("Position", ""),
                                "Type": _row.get("player_type", ""),
                                "FV": round(float(_row.get("FV", 0)), 2),
                                "MEW": round(float(_row.get("MEW", 0)), 2),
                                "PV": round(float(_row.get("PV", 0)), 2),
                            }
                        )
                    _player_df = pd.DataFrame(_player_rows)
                    _parts.append(mo.ui.table(_player_df, selection=None, page_size=20))

                    # Per-player contribution waterfall
                    _waterfall_fig = _plot_waterfall(
                        _drop_set,
                        _add_set,
                        state["my_totals"],
                        _msv_result["new_totals"],
                        players,
                    )
                    _parts.append(fig_to_png(_waterfall_fig))

                    # EW decomposition: where wins come from
                    _ew_fig = _plot_ew_decomp(
                        state["my_totals"],
                        _msv_result["new_totals"],
                        state["opponent_totals"],
                        state["category_sigmas"],
                        state["opponent_teams"],
                    )
                    _parts.append(fig_to_png(_ew_fig))

                    # Radar comparison: group by hitter/pitcher, compare IN vs OUT
                    for _pt in ("hitter", "pitcher"):
                        _pt_suffix = "-H" if _pt == "hitter" else "-P"
                        _in_names = [n for n in _add_set if n.endswith(_pt_suffix)]
                        _out_names = [n for n in _drop_set if n.endswith(_pt_suffix)]
                        if _in_names and _out_names:
                            _radar_fig = _plot_player_comparison_radar(
                                _in_names,
                                players,
                                _pt,
                                comparison_names=_out_names,
                            )
                            _parts.append(fig_to_png(_radar_fig))

                    sandbox_section = mo.vstack(_parts)
    return (sandbox_section,)


@app.cell
def sandbox_assembly(
    data_ready,
    mo,
    sandbox_eval_btn,
    sandbox_in,
    sandbox_out,
    sandbox_section,
):
    if data_ready:
        sandbox_content = mo.vstack(
            [
                mo.md("## Transaction Sandbox"),
                mo.md(
                    "Evaluate any hypothetical roster move — trades, FA pickups, "
                    "multi-player swaps. Positional validity is enforced."
                ),
                mo.hstack(
                    [sandbox_out, sandbox_in],
                    widths="equal",
                    gap=2,
                ),
                sandbox_eval_btn,
                sandbox_section,
            ]
        )
    else:
        sandbox_content = mo.md("")
    return (sandbox_content,)


# ============================================================================
# Starter comparison — "Should I start A or B?"
# ============================================================================


@app.cell
def compare_controls(mo):
    compare_input = mo.ui.text_area(
        label="Players to compare (one per line, must be on your roster)",
        placeholder="e.g.\nTrea Turner\nWilly Adames",
        rows=4,
        full_width=True,
    )
    compare_btn = mo.ui.button(
        value=0,
        on_click=lambda v: v + 1,
        label="Compare Starters",
    )
    return compare_btn, compare_input


@app.cell
def compare_results(
    compare_btn,
    compare_input,
    data_ready,
    fig_to_png,
    mo,
    my_roster_names,
    pd,
    players,
    resolve_name,
    state,
    strip_name_suffix,
):
    compare_section = mo.md("")
    if data_ready and compare_btn.value > 0 and compare_input.value.strip():
        from optimizer.config import ALL_CATEGORIES
        from optimizer.swap_evaluator import compare_starters as _compare_starters

        _lines = [
            ln.strip() for ln in compare_input.value.strip().splitlines() if ln.strip()
        ]
        _names: list[str] = []
        _errors: list[str] = []
        for _ln in _lines:
            _resolved = resolve_name(_ln, players)
            if _resolved is None:
                _errors.append(f"Not found: **{_ln}**")
            elif _resolved not in my_roster_names:
                _errors.append(
                    f"**{strip_name_suffix(_resolved)}** is not on your roster"
                )
            else:
                _names.append(_resolved)

        if _errors:
            compare_section = mo.callout(mo.md("\n\n".join(_errors)), kind="danger")
        elif len(_names) < 2:
            compare_section = mo.callout(
                mo.md("Enter at least 2 players to compare."), kind="warn"
            )
        else:
            _results = _compare_starters(
                _names,
                my_roster_names,
                players,
                state["opponent_totals"],
                state["category_sigmas"],
                my_banked_totals=state["my_banked"],
            )

            _parts: list = []

            # Summary table
            _rows = []
            _best_ew = max(r["ew"] for r in _results)
            for _r in _results:
                _row = {
                    "Player": strip_name_suffix(_r["name"]),
                    "Slot": _r["lineup"].get(_r["name"], "BENCH"),
                    "EW": round(_r["ew"], 2),
                    "ΔEW vs best": round(_r["ew"] - _best_ew, 2),
                }
                _rows.append(_row)
            _summary_df = pd.DataFrame(_rows)
            _parts.append(mo.ui.table(_summary_df, selection=None, page_size=10))

            # EW delta bar chart: each candidate vs the first (baseline)
            _baseline = _results[0]
            import matplotlib.pyplot as _plt
            import numpy as _np

            from optimizer.visualizations import LOSS_COLOR, WIN_COLOR

            _n_compare = len(_results) - 1
            if _n_compare >= 1:
                _fig, _axes = _plt.subplots(
                    1,
                    _n_compare,
                    figsize=(8 * _n_compare, 5),
                    squeeze=False,
                )
                for _ci, _r in enumerate(_results[1:]):
                    _ax = _axes[0][_ci]
                    _deltas = {
                        cat: _r["cat_ew"][cat] - _baseline["cat_ew"][cat]
                        for cat in ALL_CATEGORIES
                    }
                    _ordered = sorted(ALL_CATEGORIES, key=lambda c: _deltas[c])
                    _vals = [_deltas[c] for c in _ordered]
                    _colors = [WIN_COLOR if v >= 0 else LOSS_COLOR for v in _vals]
                    _y = _np.arange(len(_ordered))

                    _ax.barh(
                        _y,
                        _vals,
                        color=_colors,
                        height=0.6,
                        edgecolor="white",
                        linewidth=0.5,
                    )
                    for _yi, (_cat, _v) in enumerate(zip(_ordered, _vals)):
                        _ha = "left" if _v >= 0 else "right"
                        _off = 0.01 if _v >= 0 else -0.01
                        _ax.text(
                            _v + _off,
                            _yi,
                            f"{_v:+.2f}",
                            va="center",
                            ha=_ha,
                            fontsize=9,
                            fontweight="bold",
                            color=_colors[_yi],
                        )
                    _ax.set_yticks(_y)
                    _ax.set_yticklabels(_ordered, fontsize=10)
                    _ax.axvline(0, color="black", linewidth=0.8)
                    _ax.set_xlabel("ΔEW")

                    _total_d = _r["ew"] - _baseline["ew"]
                    _ax.set_title(
                        f"{strip_name_suffix(_r['name'])} vs "
                        f"{strip_name_suffix(_baseline['name'])}  "
                        f"({_total_d:+.2f} EW)",
                        fontsize=11,
                        fontweight="bold",
                    )
                _fig.tight_layout()
                _parts.append(fig_to_png(_fig))

            compare_section = mo.vstack(_parts)
    return (compare_section,)


@app.cell
def compare_assembly(
    compare_btn,
    compare_input,
    compare_section,
    data_ready,
    mo,
    sandbox_content,
):
    if data_ready:
        sandbox_full = mo.vstack(
            [
                sandbox_content,
                mo.md("---"),
                mo.md("## Starter Comparison"),
                mo.md(
                    "Compare starting one roster player vs another. "
                    "Forces each into the lineup and re-solves optimally."
                ),
                compare_input,
                compare_btn,
                compare_section,
            ]
        )
    else:
        sandbox_full = mo.md("")
    return (sandbox_full,)


# ============================================================================
# Explore tab
# ============================================================================


@app.cell
def explore_controls(mo):
    player_search = mo.ui.text(
        label="Search (comma-separated names, team, position, owner)"
    )
    stat_window = mo.ui.radio(
        options={"YTD": "ytd", "Last 15 G": "15", "Last 30 G": "30"},
        value="YTD",
        label="Stats window",
    )
    return player_search, stat_window


@app.cell
def explore_table(
    data_ready,
    game_logs,
    mo,
    pd,
    player_search,
    players,
    stat_window,
    strip_name_suffix,
):
    if data_ready:
        _df = players.copy()
        _df["Player"] = _df["Name"].apply(strip_name_suffix)
        _df["Owner"] = _df["owner"].fillna("FA")

        # Compute trailing stats from game logs
        _window = stat_window.value
        _h_stats: dict[int, dict] = {}
        _p_stats: dict[int, dict] = {}

        for _pid, _logs in game_logs.items():
            _h_logs = [g for g in _logs if g["group"] == "hitting"]
            _p_logs = [g for g in _logs if g["group"] == "pitching"]

            if _h_logs:
                if _window != "ytd":
                    _h_logs = _h_logs[-int(_window) :]
                _n = len(_h_logs)
                _pa = sum(g.get("plateAppearances", 0) for g in _h_logs)
                _ab = sum(g.get("atBats", 0) for g in _h_logs)
                _hits = sum(g.get("hits", 0) for g in _h_logs)
                _h_stats[_pid] = {
                    "G": _n,
                    "PA*": _pa,
                    "R*": sum(g.get("runs", 0) for g in _h_logs),
                    "HR*": sum(g.get("homeRuns", 0) for g in _h_logs),
                    "RBI*": sum(g.get("rbi", 0) for g in _h_logs),
                    "SB*": sum(g.get("stolenBases", 0) for g in _h_logs),
                    "AVG*": round(_hits / _ab, 3) if _ab > 0 else 0.0,
                    "OPS*": round(
                        sum(
                            float(g.get("obp", 0)) * g.get("plateAppearances", 0)
                            for g in _h_logs
                        )
                        / _pa
                        + sum(
                            float(g.get("slg", 0)) * g.get("atBats", 0) for g in _h_logs
                        )
                        / _ab,
                        3,
                    )
                    if _pa > 0 and _ab > 0
                    else 0.0,
                }

            if _p_logs:
                if _window != "ytd":
                    _p_logs = _p_logs[-int(_window) :]
                _n = len(_p_logs)
                _ip = sum(float(g.get("inningsPitched", 0)) for g in _p_logs)
                _p_stats[_pid] = {
                    "G": _n,
                    "IP*": round(_ip, 1),
                    "W*": sum(g.get("wins", 0) for g in _p_logs),
                    "SV*": sum(g.get("saves", 0) for g in _p_logs),
                    "K*": sum(g.get("strikeOuts", 0) for g in _p_logs),
                    "ERA*": round(
                        sum(g.get("earnedRuns", 0) for g in _p_logs) * 9 / _ip, 2
                    )
                    if _ip > 0
                    else 0.0,
                    "WHIP*": round(
                        (
                            sum(g.get("baseOnBalls", 0) for g in _p_logs)
                            + sum(g.get("hits", 0) for g in _p_logs)
                        )
                        / _ip,
                        2,
                    )
                    if _ip > 0
                    else 0.0,
                }

        # Map MLBAMID → trailing stats
        _trailing_rows = []
        for _idx, _row in _df.iterrows():
            _mid = _row["MLBAMID"]
            if pd.notna(_mid):
                _mid = int(_mid)
                if _row["player_type"] == "hitter" and _mid in _h_stats:
                    _trailing_rows.append({"_idx": _idx, **_h_stats[_mid]})
                elif _row["player_type"] == "pitcher" and _mid in _p_stats:
                    _trailing_rows.append({"_idx": _idx, **_p_stats[_mid]})
                else:
                    _trailing_rows.append({"_idx": _idx})
            else:
                _trailing_rows.append({"_idx": _idx})

        _trailing_df = pd.DataFrame(_trailing_rows).set_index("_idx")
        _df = _df.join(_trailing_df)

        # Column selection: projections + real stats
        _base_cols = ["Player", "Position", "Team", "Owner", "player_type", "FV", "MEW"]
        _real_h_cols = ["G", "PA*", "R*", "HR*", "RBI*", "SB*", "AVG*", "OPS*"]
        _real_p_cols = ["G", "IP*", "W*", "SV*", "K*", "ERA*", "WHIP*"]
        _all_real = list(dict.fromkeys(_real_h_cols + _real_p_cols))
        _present = [c for c in _all_real if c in _df.columns]
        _cols = _base_cols + _present

        _df = _df[_cols].round(3)
        for _c in _present:
            _df[_c] = _df[_c].fillna("")

        _raw_q = player_search.value.strip().lower()
        if _raw_q:
            _terms = [t.strip() for t in _raw_q.split(",") if t.strip()]
            _mask = _df["Player"].isna()  # all-False seed
            for _t in _terms:
                _mask = _mask | (
                    _df["Player"].str.lower().str.contains(_t, na=False)
                    | _df["Team"].str.lower().str.contains(_t, na=False)
                    | _df["Position"].str.lower().str.contains(_t, na=False)
                    | _df["Owner"].str.lower().str.contains(_t, na=False)
                )
            _df = _df[_mask]
        _df = _df.sort_values("FV", ascending=False).reset_index(drop=True)
        player_browser = mo.ui.table(_df, selection="multi", page_size=50)
    else:
        player_browser = None
    return (player_browser,)


@app.cell
def explore_radar(
    data_ready,
    fig_to_png,
    mo,
    my_roster_names,
    player_browser,
    players,
    state,
):
    if data_ready and player_browser is not None and len(player_browser.value) > 0:
        from optimizer.visualizations import (
            plot_player_comparison_radar as _plot_player_comparison_radar,
        )

        _sel = player_browser.value
        _parts = []
        _selected_internal_names: list[str] = []

        for _pt in ("hitter", "pitcher"):
            _rows = _sel[_sel["player_type"] == _pt]
            if len(_rows) == 0:
                continue
            _suffix = "-H" if _pt == "hitter" else "-P"
            _names = [r["Player"] + _suffix for _, r in _rows.iterrows()]
            _selected_internal_names.extend(_names)

            # For each selected player, find worst rostered player at same position
            _comp_names: list[str] = []
            _pos_lookup = players.set_index("Name")["Position"].to_dict()
            _mew_lookup = players.set_index("Name")["MEW"].to_dict()
            _roster_type = players[
                (players["Name"].isin(my_roster_names))
                & (players["player_type"] == _pt)
            ]

            for _name in _names:
                _player_positions = set(_pos_lookup.get(_name, "DH").split(","))
                # Find roster players eligible at any of these positions
                _candidates = []
                for _, _rrow in _roster_type.iterrows():
                    _rpos = set(_rrow["Position"].split(","))
                    if _rpos & _player_positions:
                        _candidates.append(_rrow["Name"])

                if _candidates:
                    # Worst by MEW
                    _worst = min(_candidates, key=lambda n: _mew_lookup.get(n, 0.0))
                    if _worst not in _names and _worst not in _comp_names:
                        _comp_names.append(_worst)

            _fig = _plot_player_comparison_radar(
                _names,
                players,
                _pt,
                comparison_names=_comp_names if _comp_names else None,
            )
            _parts.append(fig_to_png(_fig))

        radar_section = mo.hstack(_parts) if _parts else mo.md("")
    else:
        radar_section = mo.md("")
    return (radar_section,)


@app.cell
def mew_input(mo):
    mew_player_input = mo.ui.text(
        label="Player to analyze",
        placeholder="e.g., Aaron Judge",
    )
    return (mew_player_input,)


@app.cell
def mew_chart(
    data_ready,
    fig_to_png,
    mew_player_input,
    mo,
    players,
    resolve_name,
    state,
):
    if data_ready and mew_player_input.value.strip():
        from optimizer.visualizations import plot_mew_breakdown

        _name = resolve_name(mew_player_input.value, players)
        if _name is None:
            mew_section = mo.callout(
                mo.md(
                    f"No player found matching **{mew_player_input.value.strip()}**. "
                    "Check spelling or try adding a `-H`/`-P` suffix."
                ),
                kind="warn",
            )
        else:
            _fig = plot_mew_breakdown(
                _name,
                players,
                state["gradient"],
                state["my_totals"],
            )
            mew_section = fig_to_png(_fig)
    else:
        mew_section = mo.md("")
    return (mew_section,)


@app.cell
def roster_map(
    data_ready,
    fig_to_png,
    mo,
    my_roster_names,
    players,
    state,
):
    if data_ready:
        from optimizer.visualizations import plot_roster_value_map

        _fig = plot_roster_value_map(
            players,
            my_roster_names,
            state["my_starters"],
        )
        roster_map_section = fig_to_png(_fig)
    else:
        roster_map_section = mo.md("")
    return (roster_map_section,)


@app.cell
def explore_assembly(
    data_ready,
    mew_player_input,
    mew_section,
    mo,
    player_browser,
    player_search,
    radar_section,
    stat_window,
):
    if data_ready:
        _browser = player_browser if player_browser is not None else mo.md("")
        explore_content = mo.vstack(
            [
                mo.md("## Player Browser"),
                mo.hstack([player_search, stat_window], justify="start", gap=2),
                _browser,
                radar_section,
                mo.md("---"),
                mo.md("## MEW Breakdown"),
                mew_player_input,
                mew_section,
            ]
        )
    else:
        explore_content = mo.md("")
    return (explore_content,)


@app.cell
def tab_assembly(
    dashboard_tab,
    data_ready,
    explore_content,
    mo,
    moves_content,
    sandbox_full,
):
    if data_ready:
        app_view = mo.ui.tabs(
            {
                "Dashboard": dashboard_tab,
                "Moves": moves_content,
                "Sandbox": sandbox_full,
                "Explore": explore_content,
            }
        )
    else:
        app_view = dashboard_tab
    app_view
    return (app_view,)


if __name__ == "__main__":
    app.run()
