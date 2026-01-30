"""Tests for the playing time adjustment module."""

import numpy as np
import pandas as pd
import pytest

from optimizer.playing_time.adjust import (
    adjust_playing_time,
    apply_adjustments,
    compute_age_factor,
    compute_historical_ratio,
    compute_talent_factor,
)
from optimizer.playing_time.config import (
    AGE_PENALTY_START_HITTER,
    AGE_PENALTY_START_PITCHER,
    FULL_SEASON_IP_RP,
    FULL_SEASON_IP_SP,
    FULL_SEASON_PA,
    TALENT_PENALTY_FACTOR,
)


class TestComputeHistoricalRatio:
    """Tests for compute_historical_ratio function."""

    def test_both_seasons_available(self):
        """Average of both seasons, capped at 1.0."""
        # Player with 500 PA in both seasons → avg 500, ratio 500/600 = 0.833
        ratio = compute_historical_ratio(500, 500, FULL_SEASON_PA)
        assert ratio == pytest.approx(500 / 600, rel=1e-3)

    def test_only_current_season(self):
        """Use current season only."""
        ratio = compute_historical_ratio(400, None, FULL_SEASON_PA)
        assert ratio == pytest.approx(400 / 600, rel=1e-3)

    def test_only_prior_season(self):
        """Use prior season only."""
        ratio = compute_historical_ratio(None, 300, FULL_SEASON_PA)
        assert ratio == pytest.approx(300 / 600, rel=1e-3)

    def test_no_historical_data_returns_1(self):
        """Rookies with no data get ratio of 1.0 (trust projection)."""
        ratio = compute_historical_ratio(None, None, FULL_SEASON_PA)
        assert ratio == 1.0

    def test_cap_at_1(self):
        """Ratio capped at 1.0 even for high-PT players."""
        # Player with 700 PA (above 600) should still get 1.0
        ratio = compute_historical_ratio(700, 700, FULL_SEASON_PA)
        assert ratio == 1.0

    def test_nan_values_treated_as_none(self):
        """NaN values treated as missing."""
        ratio = compute_historical_ratio(np.nan, 400, FULL_SEASON_PA)
        assert ratio == pytest.approx(400 / 600, rel=1e-3)


class TestComputeAgeFactor:
    """Tests for compute_age_factor function."""

    def test_young_hitter_no_penalty(self):
        """Hitters under 32 get no penalty."""
        assert compute_age_factor(25, is_pitcher=False) == 1.0
        assert compute_age_factor(31, is_pitcher=False) == 1.0

    def test_young_pitcher_no_penalty(self):
        """Pitchers under 33 get no penalty."""
        assert compute_age_factor(25, is_pitcher=True) == 1.0
        assert compute_age_factor(32, is_pitcher=True) == 1.0

    def test_older_hitter_penalty(self):
        """Hitters 32+ get age penalty."""
        # Age 32 → 1 year over → 5% penalty
        assert compute_age_factor(32, is_pitcher=False) == pytest.approx(0.95)
        # Age 35 → 4 years over → 20% penalty
        assert compute_age_factor(35, is_pitcher=False) == pytest.approx(0.80)

    def test_older_pitcher_penalty(self):
        """Pitchers 33+ get age penalty."""
        # Age 33 → 1 year over → 5% penalty
        assert compute_age_factor(33, is_pitcher=True) == pytest.approx(0.95)
        # Age 38 → 6 years over → 30% penalty
        assert compute_age_factor(38, is_pitcher=True) == pytest.approx(0.70)

    def test_floor_at_0_5(self):
        """Age factor floors at 0.5 (50% max reduction)."""
        # Very old player (45+) should hit floor
        assert compute_age_factor(45, is_pitcher=False) == 0.5

    def test_none_age_no_penalty(self):
        """None/unknown age gets no penalty."""
        assert compute_age_factor(None, is_pitcher=False) == 1.0

    def test_nan_age_no_penalty(self):
        """NaN age gets no penalty."""
        assert compute_age_factor(np.nan, is_pitcher=False) == 1.0


class TestComputeTalentFactor:
    """Tests for compute_talent_factor function."""

    def test_above_threshold_no_penalty(self):
        """Players above 25th percentile get no penalty."""
        assert compute_talent_factor(3.0, 2.0) == 1.0
        assert compute_talent_factor(2.0, 2.0) == 1.0  # Equal is not below

    def test_below_threshold_penalty(self):
        """Players below 25th percentile get 15% penalty."""
        assert compute_talent_factor(1.0, 2.0) == TALENT_PENALTY_FACTOR
        assert TALENT_PENALTY_FACTOR == pytest.approx(0.85)

    def test_nan_values_no_penalty(self):
        """NaN values get no penalty."""
        assert compute_talent_factor(np.nan, 2.0) == 1.0
        assert compute_talent_factor(1.0, np.nan) == 1.0


class TestAdjustPlayingTime:
    """Tests for adjust_playing_time function."""

    def test_blending_formula(self):
        """Verify 65/35 blending formula."""
        # projected=600, all factors=1.0 → adjusted=600, blended=600
        result = adjust_playing_time(600, 1.0, 1.0, 1.0)
        assert result == 600

        # projected=600, historical_ratio=0.5 → adjusted=300
        # blended = 0.65*600 + 0.35*300 = 390 + 105 = 495
        result = adjust_playing_time(600, 0.5, 1.0, 1.0)
        assert result == pytest.approx(495)

    def test_all_factors_applied(self):
        """All three factors multiply together."""
        # projected=600, hist=0.8, age=0.9, talent=0.85
        # adjusted = 600 * 0.8 * 0.9 * 0.85 = 367.2
        # blended = 0.65*600 + 0.35*367.2 = 390 + 128.52 = 518.52
        result = adjust_playing_time(600, 0.8, 0.9, 0.85)
        expected = 0.65 * 600 + 0.35 * (600 * 0.8 * 0.9 * 0.85)
        assert result == pytest.approx(expected)


class TestApplyAdjustments:
    """Integration tests for apply_adjustments function."""

    @pytest.fixture
    def sample_projections(self):
        """Create sample projections DataFrame."""
        return pd.DataFrame(
            {
                "Name": [
                    "Aaron Judge",
                    "Mike Trout",
                    "Shohei Ohtani-P",
                    "Jacob deGrom",
                ],
                "MLBAMID": [592450, 545361, 660271, 594798],
                "Team": ["NYY", "LAA", "LAD", "TEX"],
                "PA": [650, 500, 0, 0],
                "R": [100, 80, 0, 0],
                "HR": [40, 30, 0, 0],
                "RBI": [100, 80, 0, 0],
                "SB": [5, 10, 0, 0],
                "OPS": [0.950, 0.850, 0.0, 0.0],
                "IP": [0.0, 0.0, 180.0, 150.0],
                "W": [0, 0, 15, 12],
                "SV": [0, 0, 0, 0],
                "K": [0, 0, 220, 180],
                "ERA": [0.0, 0.0, 2.50, 2.80],
                "WHIP": [0.0, 0.0, 0.90, 1.00],
                "GS": [0, 0, 30, 25],
                "WAR": [8.0, 6.0, 6.0, 5.0],
                "player_type": ["hitter", "hitter", "pitcher", "pitcher"],
                "Position": ["OF", "OF", "SP", "SP"],
            }
        )

    @pytest.fixture
    def sample_historical(self):
        """Create sample historical data."""
        return {
            "hitters": pd.DataFrame(
                {
                    "mlbam_id": [592450, 592450, 545361, 545361],
                    "season": [2024, 2023, 2024, 2023],
                    "pa": [600, 570, 300, 400],  # Judge full season, Trout injured
                }
            ),
            "pitchers": pd.DataFrame(
                {
                    "mlbam_id": [660271, 660271, 594798],
                    "season": [2024, 2023, 2024],
                    "ip": [160.0, 130.0, 80.0],  # Ohtani pitched, deGrom injured
                    "gs": [28, 23, 15],
                }
            ),
        }

    @pytest.fixture
    def sample_ages(self):
        """Create sample ages data."""
        return pd.DataFrame(
            {
                "mlbam_id": [592450, 545361, 660271, 594798],
                "age": [32, 32, 30, 36],  # Judge/Trout at age threshold, deGrom old
            }
        )

    def test_preserves_originals(
        self, sample_projections, sample_historical, sample_ages
    ):
        """Original PA/IP preserved in _original columns."""
        result = apply_adjustments(sample_projections, sample_historical, sample_ages)

        assert "PA_original" in result.columns
        assert "IP_original" in result.columns

        # Check originals match input
        judge = result[result["Name"] == "Aaron Judge"].iloc[0]
        assert judge["PA_original"] == 650

    def test_hitter_historical_adjustment(
        self, sample_projections, sample_historical, sample_ages
    ):
        """Hitters with low historical PA get reduced projections."""
        result = apply_adjustments(sample_projections, sample_historical, sample_ages)

        # Mike Trout had 300+400=700/2=350 PA avg, ratio = 350/600 = 0.583
        trout = result[result["Name"] == "Mike Trout"].iloc[0]
        assert trout["PA"] < trout["PA_original"]

    def test_pitcher_historical_adjustment(
        self, sample_projections, sample_historical, sample_ages
    ):
        """Pitchers with low historical IP get reduced projections."""
        result = apply_adjustments(sample_projections, sample_historical, sample_ages)

        # deGrom only had 80 IP in 2024, should be heavily reduced
        degrom = result[result["Name"] == "Jacob deGrom"].iloc[0]
        assert degrom["IP"] < degrom["IP_original"]

    def test_age_added_to_output(
        self, sample_projections, sample_historical, sample_ages
    ):
        """Age column added from ages DataFrame."""
        result = apply_adjustments(sample_projections, sample_historical, sample_ages)

        assert "age" in result.columns
        judge = result[result["Name"] == "Aaron Judge"].iloc[0]
        assert judge["age"] == 32

    def test_empty_historical_data(self, sample_projections, sample_ages):
        """Works with empty historical data (all rookies)."""
        empty_historical = {
            "hitters": pd.DataFrame(columns=["mlbam_id", "season", "pa"]),
            "pitchers": pd.DataFrame(columns=["mlbam_id", "season", "ip", "gs"]),
        }

        result = apply_adjustments(sample_projections, empty_historical, sample_ages)
        assert len(result) == len(sample_projections)

    def test_empty_ages_data(self, sample_projections, sample_historical):
        """Works with empty ages data (no age adjustment)."""
        empty_ages = pd.DataFrame(columns=["mlbam_id", "age"])

        result = apply_adjustments(sample_projections, sample_historical, empty_ages)
        assert len(result) == len(sample_projections)
        # Age should be NaN for all
        assert result["age"].isna().all()


class TestIntegration:
    """Integration tests using real data files."""

    @pytest.fixture
    def skip_if_no_data(self):
        """Skip if data files don't exist."""
        from pathlib import Path

        data_dir = Path("data")
        required_files = [
            "fangraphs-atc-projections-hitters.csv",
            "fangraphs-atc-projections-pitchers.csv",
        ]
        for f in required_files:
            if not (data_dir / f).exists():
                pytest.skip(f"Data file {f} not found")

    def test_load_atc_projections(self, skip_if_no_data):
        """Can load ATC projections from real files."""
        from optimizer.playing_time.load import load_atc_projections

        df = load_atc_projections()

        assert len(df) > 1000, "Expected 1000+ total players"
        assert "Name" in df.columns
        assert "MLBAMID" in df.columns
        assert df["MLBAMID"].notna().all()

        hitters = df[df["player_type"] == "hitter"]
        pitchers = df[df["player_type"] == "pitcher"]

        assert len(hitters) > 500, "Expected 500+ hitters"
        assert len(pitchers) > 500, "Expected 500+ pitchers"
        assert (hitters["PA"] > 0).any(), "Some hitters must have PA > 0"
        assert (pitchers["IP"] > 0).any(), "Some pitchers must have IP > 0"

    def test_load_historical_actuals(self, skip_if_no_data):
        """Can load historical actuals from mlb_stats.db."""
        from pathlib import Path

        from optimizer.playing_time.load import load_historical_actuals

        mlb_stats_db = Path("../mlb_player_comps_dashboard/mlb_stats.db")
        if not mlb_stats_db.exists():
            pytest.skip("mlb_stats.db not found")

        historical = load_historical_actuals(mlb_stats_db, [2024, 2023])

        assert len(historical["hitters"]) > 500, "Expected 500+ hitter-seasons"
        assert len(historical["pitchers"]) > 800, "Expected 800+ pitcher-seasons"

    def test_full_pipeline(self, skip_if_no_data):
        """Full pipeline produces valid output."""
        from pathlib import Path

        from optimizer.playing_time.adjust import apply_adjustments
        from optimizer.playing_time.load import (
            load_ages,
            load_atc_projections,
            load_historical_actuals,
        )

        mlb_stats_db = Path("../mlb_player_comps_dashboard/mlb_stats.db")
        if not mlb_stats_db.exists():
            pytest.skip("mlb_stats.db not found")

        projections = load_atc_projections()
        historical = load_historical_actuals(mlb_stats_db, [2024, 2023])
        ages = load_ages()

        result = apply_adjustments(projections, historical, ages)

        # Validate output
        assert len(result) == len(projections)
        assert result["MLBAMID"].notna().all()
        assert (result["PA"] >= 0).all()
        assert (result["IP"] >= 0).all()

        # Total playing time should decrease (not increase significantly)
        hitters = result[result["player_type"] == "hitter"]
        pitchers = result[result["player_type"] == "pitcher"]

        pa_ratio = hitters["PA"].sum() / hitters["PA_original"].sum()
        ip_ratio = pitchers["IP"].sum() / pitchers["IP_original"].sum()

        assert pa_ratio <= 1.05, f"Total PA increased too much: {pa_ratio:.1%}"
        assert ip_ratio <= 1.05, f"Total IP increased too much: {ip_ratio:.1%}"
