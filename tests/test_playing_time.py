"""Tests for the playing time adjustment module."""

import numpy as np
import pandas as pd
import pytest

from optimizer.playing_time import (
    _blend,
    _compute_age_factor,
    _compute_historical_ratio,
    _compute_talent_factor,
    adjust_projections,
    AGE_THRESHOLD_HITTER,
    AGE_THRESHOLD_PITCHER,
    FULL_SEASON_PA,
    TALENT_PENALTY,
)


# --------------------------------------------------------------------------
# Tests for _compute_historical_ratio
# --------------------------------------------------------------------------


def test_historical_ratio_both_seasons_available():
    ratio = _compute_historical_ratio(500, 500, FULL_SEASON_PA)
    assert ratio == pytest.approx(500 / 600, rel=1e-3)


def test_historical_ratio_only_current_season():
    ratio = _compute_historical_ratio(400, None, FULL_SEASON_PA)
    assert ratio == pytest.approx(400 / 600, rel=1e-3)


def test_historical_ratio_only_prior_season():
    ratio = _compute_historical_ratio(None, 300, FULL_SEASON_PA)
    assert ratio == pytest.approx(300 / 600, rel=1e-3)


def test_historical_ratio_no_data_returns_1():
    ratio = _compute_historical_ratio(None, None, FULL_SEASON_PA)
    assert ratio == 1.0


def test_historical_ratio_cap_at_1():
    ratio = _compute_historical_ratio(700, 700, FULL_SEASON_PA)
    assert ratio == 1.0


def test_historical_ratio_nan_values_treated_as_none():
    ratio = _compute_historical_ratio(np.nan, 400, FULL_SEASON_PA)
    assert ratio == pytest.approx(400 / 600, rel=1e-3)


# --------------------------------------------------------------------------
# Tests for _compute_age_factor
# --------------------------------------------------------------------------


def test_age_factor_young_hitter_no_penalty():
    assert _compute_age_factor(25, is_pitcher=False) == 1.0
    assert _compute_age_factor(31, is_pitcher=False) == 1.0


def test_age_factor_young_pitcher_no_penalty():
    assert _compute_age_factor(25, is_pitcher=True) == 1.0
    assert _compute_age_factor(32, is_pitcher=True) == 1.0


def test_age_factor_older_hitter_penalty():
    assert _compute_age_factor(32, is_pitcher=False) == pytest.approx(0.95)
    assert _compute_age_factor(35, is_pitcher=False) == pytest.approx(0.80)


def test_age_factor_older_pitcher_penalty():
    assert _compute_age_factor(33, is_pitcher=True) == pytest.approx(0.95)
    assert _compute_age_factor(38, is_pitcher=True) == pytest.approx(0.70)


def test_age_factor_floor_at_0_5():
    assert _compute_age_factor(45, is_pitcher=False) == 0.5


def test_age_factor_none_age_no_penalty():
    assert _compute_age_factor(None, is_pitcher=False) == 1.0


def test_age_factor_nan_age_no_penalty():
    assert _compute_age_factor(np.nan, is_pitcher=False) == 1.0


# --------------------------------------------------------------------------
# Tests for _compute_talent_factor
# --------------------------------------------------------------------------


def test_talent_factor_above_threshold_no_penalty():
    assert _compute_talent_factor(3.0, 2.0) == 1.0
    assert _compute_talent_factor(2.0, 2.0) == 1.0


def test_talent_factor_below_threshold_penalty():
    assert _compute_talent_factor(1.0, 2.0) == TALENT_PENALTY
    assert TALENT_PENALTY == pytest.approx(0.85)


def test_talent_factor_nan_values_no_penalty():
    assert _compute_talent_factor(np.nan, 2.0) == 1.0
    assert _compute_talent_factor(1.0, np.nan) == 1.0


# --------------------------------------------------------------------------
# Tests for _blend
# --------------------------------------------------------------------------


def test_blend_no_adjustment():
    result = _blend(600, 1.0, 1.0, 1.0)
    assert result == 600


def test_blend_half_historical_ratio():
    # projected=600, historical_ratio=0.5 → adjusted=300
    # blended = 0.65*600 + 0.35*300 = 390 + 105 = 495
    result = _blend(600, 0.5, 1.0, 1.0)
    assert result == pytest.approx(495)


def test_blend_all_factors():
    result = _blend(600, 0.8, 0.9, 0.85)
    expected = 0.65 * 600 + 0.35 * (600 * 0.8 * 0.9 * 0.85)
    assert result == pytest.approx(expected)


# --------------------------------------------------------------------------
# Integration tests using real data files
# --------------------------------------------------------------------------


@pytest.fixture
def skip_if_no_data():
    from pathlib import Path

    required = [
        "data/fangraphs-atc-projections-hitters.csv",
        "data/fangraphs-atc-projections-pitchers.csv",
    ]
    for f in required:
        if not Path(f).exists():
            pytest.skip(f"Data file {f} not found")


def test_adjust_projections(skip_if_no_data, tmp_path):
    """Full pipeline produces valid output."""
    hitters_out = tmp_path / "hitters.csv"
    pitchers_out = tmp_path / "pitchers.csv"

    result = adjust_projections(
        hitters_output=hitters_out,
        pitchers_output=pitchers_out,
    )

    assert result["hitters"] > 500
    assert result["pitchers"] > 500
    assert result["pa_reduction"] > 0.05  # At least 5% reduction
    assert result["ip_reduction"] > 0.05

    # Check output structure matches input
    orig_h = pd.read_csv("data/fangraphs-atc-projections-hitters.csv")
    adj_h = pd.read_csv(hitters_out)
    assert list(orig_h.columns) == list(adj_h.columns)
    assert len(orig_h) == len(adj_h)


def test_output_files_exist(skip_if_no_data):
    """Default output files are created."""
    from pathlib import Path

    adjust_projections()

    assert Path("data/fangraphs-atc-pt-adjusted-hitters.csv").exists()
    assert Path("data/fangraphs-atc-pt-adjusted-pitchers.csv").exists()
