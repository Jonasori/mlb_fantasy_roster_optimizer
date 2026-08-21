"""
Offline tests for the backtest harness.
Per AGENTS.md: no classes, no fixtures, no mocking.
"""

import datetime

import numpy as np
import pandas as pd
import pytest

from optimizer.backtest import SEASON_END, assemble_backtest_frame


def _projection() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MLBAMID": [1, 2, 3],
            "Name": ["Kept", "Also Kept", "No Evidence"],
            "player_type": ["hitter", "hitter", "hitter"],
            "PA": [200.0, 150.0, 100.0],
            "R": [25.0, 18.0, 12.0], "HR": [8.0, 5.0, 3.0],
            "RBI": [26.0, 19.0, 13.0], "SB": [4.0, 2.0, 1.0],
            "OPS": [0.780, 0.700, 0.650],
        }
    )


def _evidence() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MLBAMID": [1, 2], "name": ["Kept", "Also Kept"],
            "group": ["hitting", "hitting"],
            "PA": [300.0, 280.0], "AB": [270.0, 250.0], "H": [70.0, 60.0],
            "HR": [12.0, 7.0], "R": [40.0, 33.0], "RBI": [41.0, 30.0],
            "SB": [6.0, 3.0], "CS": [2.0, 1.0], "BB": [27.0, 25.0],
            "SO": [70.0, 62.0], "SF": [3.0, 3.0],
            "avg": [0.259, 0.240], "obp": [0.330, 0.318],
            "slg": [0.440, 0.390], "babip": [0.300, 0.285],
        }
    )


def _actual() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MLBAMID": [1, 2], "name": ["Kept", "Also Kept"],
            "group": ["hitting", "hitting"],
            "PA": [190.0, 140.0], "AB": [170.0, 125.0], "H": [45.0, 30.0],
            "HR": [9.0, 4.0], "R": [27.0, 16.0], "RBI": [28.0, 17.0],
            "SB": [5.0, 1.0], "CS": [1.0, 1.0], "BB": [18.0, 13.0],
            "SO": [44.0, 33.0], "SF": [2.0, 2.0],
            "avg": [0.265, 0.240], "obp": [0.335, 0.312],
            "slg": [0.455, 0.376], "babip": [0.305, 0.280],
        }
    )


def test_season_end_dates_known():
    assert SEASON_END[2026] == datetime.date(2026, 9, 27), (
        f"2026 season end is {SEASON_END.get(2026)}, expected 2026-09-27"
    )


def test_assemble_joins_and_prefixes():
    # outcome_end must be explicit and in the past: the 2026 season's default
    # (SEASON_END[2026] = 2026-09-27) has not happened yet as of this run.
    frame = assemble_backtest_frame(
        2026, datetime.date(2026, 6, 11), _projection(),
        evidence=_evidence(), actual=_actual(),
        outcome_end=datetime.date(2026, 6, 20),
    )
    assert len(frame) == 2, (
        f"Expected 2 rows (inner join on players with both evidence and "
        f"outcome), got {len(frame)}"
    )
    assert 3 not in set(frame["MLBAMID"]), (
        "Player 3 has a projection but no evidence and no outcome; he must "
        "not appear — scoring him would credit the model for a phantom."
    )
    for col in ("proj_PA", "proj_OPS", "evid_K_pct", "n_evid", "actual_PA"):
        assert col in frame.columns, f"Missing column {col}: {list(frame.columns)}"

    kept = frame[frame["MLBAMID"] == 1].iloc[0]
    assert kept["proj_PA"] == 200.0, f"proj_PA is {kept['proj_PA']}, expected 200"
    assert kept["actual_PA"] == 190.0, f"actual_PA is {kept['actual_PA']}, expected 190"
    assert kept["n_evid"] == 300.0, f"n_evid is {kept['n_evid']}, expected 300 PA"
    assert abs(kept["evid_K_pct"] - 70.0 / 300.0) < 1e-9, (
        f"evid_K_pct is {kept['evid_K_pct']}, expected {70 / 300}"
    )


def test_actual_ops_is_derived_not_copied():
    frame = assemble_backtest_frame(
        2026, datetime.date(2026, 6, 11), _projection(),
        evidence=_evidence(), actual=_actual(),
        outcome_end=datetime.date(2026, 6, 20),
    )
    kept = frame[frame["MLBAMID"] == 1].iloc[0]
    assert abs(kept["actual_OPS"] - (0.335 + 0.455)) < 1e-9, (
        f"actual_OPS is {kept['actual_OPS']}, expected obp+slg = 0.790"
    )


def test_rejects_split_outside_season():
    try:
        assemble_backtest_frame(
            2026, datetime.date(2026, 12, 1), _projection(),
            evidence=_evidence(), actual=_actual(),
        )
    except AssertionError as exc:
        assert "split" in str(exc).lower(), (
            f"Expected an assertion about the split date, got: {exc}"
        )
    else:
        raise AssertionError(
            "A split date after the season end must fail loudly — it silently "
            "produces an empty outcome window otherwise."
        )


def test_proj_horizon_frac_is_one_at_full_season_end():
    # 2025 is fully complete, so the default outcome_end (SEASON_END[2025])
    # passes the not-in-the-future check, and covers the full projected
    # horizon: frac must be exactly 1.0.
    frame = assemble_backtest_frame(
        2025, datetime.date(2025, 6, 11), _projection(),
        evidence=_evidence(), actual=_actual(),
    )
    assert (frame["proj_horizon_frac"] == 1.0).all(), (
        f"proj_horizon_frac at the default (season-end) outcome_end should be "
        f"1.0 for every row, got {frame['proj_horizon_frac'].tolist()}"
    )


def test_proj_horizon_frac_short_window():
    # SEASON_END[2025] = 2025-09-28. split = 2025-08-29 puts the full horizon
    # at exactly 30 days (2 days left in August + 28 in September).
    # outcome_end 15 days later = 2025-09-13, so the window covers exactly
    # half the projected horizon.
    split = datetime.date(2025, 8, 29)
    outcome_end = datetime.date(2025, 9, 13)
    assert (SEASON_END[2025] - split).days == 30, (
        f"Test setup assumption broken: full horizon is "
        f"{(SEASON_END[2025] - split).days} days, expected 30"
    )
    frame = assemble_backtest_frame(
        2025, split, _projection(),
        evidence=_evidence(), actual=_actual(),
        outcome_end=outcome_end,
    )
    assert (abs(frame["proj_horizon_frac"] - 0.5) < 1e-9).all(), (
        f"proj_horizon_frac for a 15-of-30-day window should be 0.5, got "
        f"{frame['proj_horizon_frac'].tolist()}"
    )


def test_future_outcome_end_rejected_as_censored():
    with pytest.raises(AssertionError, match="censor"):
        assemble_backtest_frame(
            2026, datetime.date(2026, 6, 11), _projection(),
            evidence=_evidence(), actual=_actual(),
            outcome_end=datetime.date(2026, 9, 27),
        )


def test_outcome_end_before_split_rejected():
    # Both dates are in the past (season 2025), so this isolates the
    # outcome_end > split check from the censoring check above. Injected
    # evidence/actual bypass fetch_stats_range's own start<=end assert, so
    # this ordering must be checked independently here.
    with pytest.raises(AssertionError, match="not after"):
        assemble_backtest_frame(
            2025, datetime.date(2025, 6, 11), _projection(),
            evidence=_evidence(), actual=_actual(),
            outcome_end=datetime.date(2025, 6, 5),
        )


def test_rejects_duplicate_mlbamid_in_projection():
    dup_proj = pd.concat([_projection(), _projection().iloc[[0]]], ignore_index=True)
    with pytest.raises(AssertionError, match="projection has duplicate"):
        assemble_backtest_frame(
            2025, datetime.date(2025, 6, 11), dup_proj,
            evidence=_evidence(), actual=_actual(),
        )


def test_rejects_duplicate_mlbamid_surviving_join():
    # A traded player exported as two team-split rows is a real FanGraphs
    # pattern; a duplicate hiding in evidence/actual (not the projection)
    # must still be caught, downstream of the join.
    dup_evidence = pd.concat([_evidence(), _evidence().iloc[[0]]], ignore_index=True)
    with pytest.raises(AssertionError, match="survived the join"):
        assemble_backtest_frame(
            2025, datetime.date(2025, 6, 11), _projection(),
            evidence=dup_evidence, actual=_actual(),
        )


def _pitching_projection() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MLBAMID": [10, 20], "Name": ["Ace", "Pen"],
            "IP": [80.0, 60.0], "W": [6.0, 4.0], "SV": [0.0, 10.0],
            "K": [90.0, 70.0], "ERA": [3.20, 2.80], "WHIP": [1.10, 1.05],
        }
    )


def _pitching_evidence() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MLBAMID": [10, 20], "name": ["Ace", "Pen"],
            "group": ["pitching", "pitching"],
            "BF": [300.0, 250.0], "SOA": [90.0, 60.0], "BBA": [24.0, 20.0],
            "HRA": [5.0, 4.0], "groundOuts": [70.0, 55.0], "airOuts": [50.0, 45.0],
            "babip": [0.290, 0.280],
        }
    )


def _pitching_actual() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MLBAMID": [10, 20], "name": ["Ace", "Pen"],
            "group": ["pitching", "pitching"],
            "IP": [70.0, 50.0], "W": [5.0, 3.0], "SV": [0.0, 8.0],
            "ER": [21.0, 15.0], "HA": [55.0, 40.0], "BBA": [15.0, 10.0],
            "SOA": [75.0, 55.0],
        }
    )


def test_assemble_pitching_group():
    frame = assemble_backtest_frame(
        2025, datetime.date(2025, 6, 11), _pitching_projection(),
        group="pitching",
        evidence=_pitching_evidence(), actual=_pitching_actual(),
    )
    assert len(frame) == 2, f"Expected 2 pitchers, got {len(frame)}"
    for col in (
        "proj_IP", "proj_ERA", "evid_K_pct", "evid_GB_pct", "n_evid",
        "n_evid_bip", "actual_ERA", "actual_WHIP", "actual_K",
    ):
        assert col in frame.columns, f"Missing column {col}: {list(frame.columns)}"

    ace = frame[frame["MLBAMID"] == 10].iloc[0]
    assert ace["proj_IP"] == 80.0, f"proj_IP is {ace['proj_IP']}, expected 80"
    assert abs(ace["actual_ERA"] - 2.7) < 1e-9, (
        f"actual_ERA is {ace['actual_ERA']}, expected 2.7 (21 ER * 9 / 70 IP)"
    )
    assert abs(ace["actual_WHIP"] - 1.0) < 1e-9, (
        f"actual_WHIP is {ace['actual_WHIP']}, expected 1.0 ((55 HA + 15 BBA) / 70 IP)"
    )
    assert ace["actual_K"] == 75.0, f"actual_K is {ace['actual_K']}, expected 75 (SOA)"
    assert ace["n_evid"] == 300.0, f"n_evid is {ace['n_evid']}, expected 300 BF"
    assert ace["n_evid_bip"] == 120.0, (
        f"n_evid_bip is {ace['n_evid_bip']}, expected 120 (70 groundOuts + 50 airOuts)"
    )
    assert abs(ace["evid_K_pct"] - 90.0 / 300.0) < 1e-9, (
        f"evid_K_pct is {ace['evid_K_pct']}, expected {90 / 300} (90 SOA / 300 BF)"
    )


from optimizer.backtest import BASELINES, run_baselines, score_in_mew

_TOTALS = {
    "PA": 7000.0, "IP": 1000.0, "R": 897.0, "HR": 240.0, "RBI": 764.0,
    "SB": 163.0, "OPS": 0.7356, "W": 67.0, "SV": 65.0, "K": 1014.0,
    "ERA": 3.7553, "WHIP": 1.1300,
}
_GRADIENT = {
    "R": 0.02284, "HR": 0.01964, "RBI": 0.00101, "SB": 0.07693,
    "OPS": 0.39234, "W": 0.26798, "SV": 0.06077, "K": 0.00660,
    "ERA": -4.58521, "WHIP": -28.62424,
}


def _scored_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MLBAMID": [1, 2],
            "name": ["A", "B"],
            "proj_PA": [200.0, 150.0], "proj_R": [25.0, 18.0],
            "proj_HR": [8.0, 5.0], "proj_RBI": [26.0, 19.0],
            "proj_SB": [4.0, 2.0], "proj_OPS": [0.780, 0.700],
            "actual_PA": [190.0, 140.0], "actual_R": [27.0, 16.0],
            "actual_HR": [9.0, 4.0], "actual_RBI": [28.0, 17.0],
            "actual_SB": [5.0, 1.0], "actual_OPS": [0.790, 0.690],
            "n_evid": [300.0, 280.0],
        }
    )


def test_score_in_mew_zero_when_prediction_is_perfect():
    frame = _scored_frame()
    for stat in ("PA", "R", "HR", "RBI", "SB", "OPS"):
        frame[f"pred_{stat}"] = frame[f"actual_{stat}"]
    scored = score_in_mew(frame, _TOTALS, _GRADIENT, "hitting")
    assert scored["mew_error"].abs().max() < 1e-9, (
        f"A perfect prediction must have zero MEW error, got "
        f"{scored['mew_error'].abs().max()}"
    )


def test_score_in_mew_weights_by_gradient():
    """A 10-unit SB error must outweigh a 10-unit RBI error by ~g_SB/g_RBI."""
    frame = _scored_frame()
    for stat in ("PA", "R", "HR", "RBI", "SB", "OPS"):
        frame[f"pred_{stat}"] = frame[f"actual_{stat}"]

    sb_off = frame.copy()
    sb_off["pred_SB"] = sb_off["actual_SB"] + 10.0
    rbi_off = frame.copy()
    rbi_off["pred_RBI"] = rbi_off["actual_RBI"] + 10.0

    sb_err = score_in_mew(sb_off, _TOTALS, _GRADIENT, "hitting")["mew_error"].abs().sum()
    rbi_err = score_in_mew(rbi_off, _TOTALS, _GRADIENT, "hitting")["mew_error"].abs().sum()
    ratio = sb_err / rbi_err
    expected = _GRADIENT["SB"] / _GRADIENT["RBI"]
    assert abs(ratio - expected) < 0.01 * expected, (
        f"SB/RBI MEW-error ratio is {ratio:.1f}, expected ~{expected:.1f} "
        f"(g_SB/g_RBI). Scoring is not gradient-weighted."
    )


def test_score_in_mew_ops_is_volume_weighted():
    """OPS is the one ratio-shaped stat in the hitting formula (volume *
    (rate - team_rate) / team_volume) — the part of _mew_contribution that
    doesn't reduce to a linear sum. A zero-diff perfect-prediction test can
    pass even if this term is missing entirely (0 == 0 either way), so pin
    its exact magnitude here against a nonzero OPS error.
    """
    frame = _scored_frame()
    for stat in ("PA", "R", "HR", "RBI", "SB"):
        frame[f"pred_{stat}"] = frame[f"actual_{stat}"]
    frame["pred_OPS"] = frame["actual_OPS"] + 0.050

    scored = score_in_mew(frame, _TOTALS, _GRADIENT, "hitting")
    expected = _GRADIENT["OPS"] * frame["actual_PA"] * 0.050 / _TOTALS["PA"]
    diff = (scored["mew_error"] - expected).abs().max()
    assert diff < 1e-9, (
        f"OPS MEW error should equal g_OPS * PA * delta_OPS / team_PA "
        f"({expected.tolist()}), got {scored['mew_error'].tolist()} "
        f"(max diff {diff}). The ratio-stat term in _mew_contribution is "
        f"wrong or missing."
    )


def _pitch_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MLBAMID": [10, 20],
            "name": ["Ace", "Closer"],
            "proj_IP": [150.0, 60.0], "proj_W": [10.0, 4.0],
            "proj_SV": [0.0, 20.0], "proj_K": [140.0, 70.0],
            "proj_ERA": [3.80, 3.10], "proj_WHIP": [1.20, 1.05],
            "actual_IP": [140.0, 55.0], "actual_W": [9.0, 5.0],
            "actual_SV": [0.0, 18.0], "actual_K": [135.0, 68.0],
            "actual_ERA": [3.60, 3.30], "actual_WHIP": [1.15, 1.10],
        }
    )


def test_score_in_mew_pitching_group_end_to_end():
    """The pitching branch of _mew_contribution (IP volume, ERA/WHIP as the
    ratio-shaped stats, W/SV/K counting) is otherwise never exercised by this
    suite — every other test here only ever passes group='hitting'. Check
    both the perfect-prediction zero case and a pinned nonzero ERA error in
    one pass, since 'pitching' is a first-class value of the `group` param
    on score_in_mew.
    """
    frame = _pitch_frame()
    for stat in ("IP", "W", "SV", "K", "ERA", "WHIP"):
        frame[f"pred_{stat}"] = frame[f"actual_{stat}"]
    scored = score_in_mew(frame, _TOTALS, _GRADIENT, "pitching")
    assert scored["mew_error"].abs().max() < 1e-9, (
        f"A perfect pitching prediction must have zero MEW error, got "
        f"{scored['mew_error'].abs().max()}"
    )

    frame["pred_ERA"] = frame["actual_ERA"] - 0.20  # a lower (better) ERA
    scored = score_in_mew(frame, _TOTALS, _GRADIENT, "pitching")
    expected = _GRADIENT["ERA"] * frame["actual_IP"] * (-0.20) / _TOTALS["IP"]
    diff = (scored["mew_error"] - expected).abs().max()
    assert diff < 1e-9, (
        f"ERA MEW error should equal g_ERA * IP * delta_ERA / team_IP "
        f"({expected.tolist()}), got {scored['mew_error'].tolist()} "
        f"(max diff {diff})."
    )


def test_baselines_present_and_atc_is_identity():
    assert set(BASELINES) >= {"atc", "raw_ytd", "flat_volume"}, (
        f"Spec §3.1 requires all three mandatory baselines, got {sorted(BASELINES)}"
    )
    frame = _scored_frame()
    predicted = BASELINES["atc"](frame, "hitting")
    assert (predicted["pred_PA"] == frame["proj_PA"]).all(), (
        "The 'atc' baseline must pass the projection through unchanged — it is "
        "the thing every candidate has to beat."
    )


def test_raw_ytd_baseline_actually_differs_from_atc():
    """Guard against raw_ytd silently degenerating into the atc baseline.

    raw_ytd reads evid_R/HR/RBI/SB. If those columns ever go missing from the
    backtest frame, every .get() falls through to the projection and this
    baseline becomes 'atc with one OPS column swapped' — it would still
    produce a plausible number and no other test would notice.
    """
    frame = _scored_frame()
    for stat, value in (("R", 40.0), ("HR", 12.0), ("RBI", 41.0), ("SB", 6.0)):
        frame[f"evid_{stat}"] = [value, value * 0.8]
    frame["evid_OPS"] = [0.770, 0.640]

    atc = BASELINES["atc"](frame, "hitting")
    ytd = BASELINES["raw_ytd"](frame, "hitting")
    differing = [
        stat
        for stat in ("R", "HR", "RBI", "SB", "OPS")
        if not atc[f"pred_{stat}"].equals(ytd[f"pred_{stat}"])
    ]
    assert set(differing) >= {"R", "HR", "RBI", "SB", "OPS"}, (
        f"raw_ytd differs from atc only in {differing}. It must re-project "
        f"every counting stat from observed rates, not pass them through. "
        f"Check that assemble_backtest_frame still carries evid_R/HR/RBI/SB."
    )


def test_run_baselines_returns_one_row_per_baseline():
    result = run_baselines(_scored_frame(), _TOTALS, _GRADIENT, "hitting")
    assert set(result["baseline"]) >= {"atc", "raw_ytd", "flat_volume"}, (
        f"Missing baselines in result: {sorted(set(result['baseline']))}"
    )
    for col in ("baseline", "mae_mew", "rmse_mew", "n"):
        assert col in result.columns, f"Missing column {col}: {list(result.columns)}"
    assert (result["mae_mew"] >= 0).all(), "MAE cannot be negative"
