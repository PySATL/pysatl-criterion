"""
Unit tests for Student's t-distribution goodness-of-fit test statistics.

This module contains tests for all statistics implemented in
pysatl_criterion.statistics.student module.
"""

import numpy as np
import pytest
import scipy.stats as scipy_stats

from pysatl_criterion.statistics.student import (
    AbstractStudentGofStatistic,
    AndersonDarlingStudentGofStatistic,
    ChiSquareStudentGofStatistic,
    CramerVonMisesStudentGofStatistic,
    KolmogorovSmirnovStudentGofStatistic,
    KuiperStudentGofStatistic,
    LillieforsStudentGofStatistic,
    WatsonStudentGofStatistic,
    ZhangZaStudentGofStatistic,
    ZhangZcStudentGofStatistic,
)


# Test data: samples from t-distribution with different degrees of freedom
@pytest.fixture
def t_sample_df3():
    """Sample from t-distribution with df=3."""
    np.random.default_rng(42)
    return np.random.standard_t(df=3, size=50)


@pytest.fixture
def t_sample_df5():
    """Sample from t-distribution with df=5."""
    np.random.default_rng(42)
    return np.random.standard_t(df=5, size=50)


@pytest.fixture
def normal_sample():
    """Sample from normal distribution."""
    np.random.default_rng(42)
    return np.random.normal(0, 1, size=50)


# Tests for AbstractStudentGofStatistic
def test_abstract_student_criterion_code():
    """Test that abstract class returns correct code."""
    assert "STUDENT_GOODNESS_OF_FIT" == AbstractStudentGofStatistic.code()


def test_abstract_student_invalid_df():
    """Test that invalid df raises ValueError."""
    with pytest.raises(ValueError, match="Degrees of freedom must be positive"):
        KolmogorovSmirnovStudentGofStatistic(df=0)
    with pytest.raises(ValueError, match="Degrees of freedom must be positive"):
        KolmogorovSmirnovStudentGofStatistic(df=-1)


def test_abstract_student_invalid_scale():
    """Test that invalid scale raises ValueError."""
    with pytest.raises(ValueError, match="Scale must be positive"):
        KolmogorovSmirnovStudentGofStatistic(df=5, scale=0)
    with pytest.raises(ValueError, match="Scale must be positive"):
        KolmogorovSmirnovStudentGofStatistic(df=5, scale=-1)


# Tests for KolmogorovSmirnovStudentGofStatistic
class TestKolmogorovSmirnovStudent:
    """Tests for Kolmogorov-Smirnov statistic."""

    @pytest.mark.parametrize(
        ("data", "df", "expected"),
        [
            # Pre-calculated values using scipy.stats.kstest
            ([0.5, 1.0, -0.5, 0.0, 2.0], 5, 0.31915),
            ([-1.5, -0.5, 0.5, 1.5], 3, 0.17428),
            ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 10, 0.53884),
        ],
    )
    def test_ks_statistic_values(self, data, df, expected):
        """Test KS statistic against pre-calculated values."""
        stat = KolmogorovSmirnovStudentGofStatistic(df=df)
        result = stat.execute_statistic(data)
        assert result == pytest.approx(expected, rel=0.01)

    def test_ks_code(self):
        """Test that KS statistic returns correct code."""
        assert "KS_STUDENT_GOODNESS_OF_FIT" == KolmogorovSmirnovStudentGofStatistic.code()

    def test_ks_with_scipy(self, t_sample_df5):
        """Verify KS statistic matches scipy calculation."""
        stat = KolmogorovSmirnovStudentGofStatistic(df=5)
        our_result = stat.execute_statistic(t_sample_df5)

        # Compare with scipy kstest
        scipy_result = scipy_stats.kstest(t_sample_df5, scipy_stats.t(df=5).cdf)
        assert our_result == pytest.approx(scipy_result.statistic, rel=0.01)

    def test_ks_alternatives(self):
        """Test KS statistic with different alternatives."""
        data = [0.5, 1.0, -0.5, 0.0, 2.0]

        stat_two = KolmogorovSmirnovStudentGofStatistic(df=5, alternative="two-sided")
        stat_less = KolmogorovSmirnovStudentGofStatistic(df=5, alternative="less")
        stat_greater = KolmogorovSmirnovStudentGofStatistic(df=5, alternative="greater")

        result_two = stat_two.execute_statistic(data)
        result_less = stat_less.execute_statistic(data)
        result_greater = stat_greater.execute_statistic(data)

        # Two-sided should be >= both one-sided
        assert result_two >= result_less and result_two >= result_greater


# Tests for AndersonDarlingStudentGofStatistic
class TestAndersonDarlingStudent:
    """Tests for Anderson-Darling statistic."""

    @pytest.mark.parametrize(
        ("data", "df", "expected"),
        [
            # Pre-calculated values for Anderson-Darling statistic
            ([0.5, 1.0, -0.5, 0.0, 2.0], 5, 0.82676),
            ([-1.5, -0.5, 0.5, 1.5], 3, 0.17673),
            ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 10, 3.48858),
        ],
    )
    def test_ad_statistic_values(self, data, df, expected):
        """Test AD statistic against pre-calculated values."""
        stat = AndersonDarlingStudentGofStatistic(df=df)
        result = stat.execute_statistic(data)
        assert result == pytest.approx(expected, rel=0.05)

    def test_ad_code(self):
        """Test that AD statistic returns correct code."""
        assert "AD_STUDENT_GOODNESS_OF_FIT" == AndersonDarlingStudentGofStatistic.code()

    def test_ad_positive(self, t_sample_df5):
        """Test that AD statistic is always positive."""
        stat = AndersonDarlingStudentGofStatistic(df=5)
        result = stat.execute_statistic(t_sample_df5)
        assert result > 0


# Tests for CramerVonMisesStudentGofStatistic
class TestCramerVonMisesStudent:
    """Tests for Cramer-von Mises statistic."""

    @pytest.mark.parametrize(
        ("data", "df", "expected"),
        [
            # Pre-calculated values using scipy.stats.cramervonmises
            ([0.5, 1.0, -0.5, 0.0, 2.0], 5, 0.15382),
            ([-1.5, -0.5, 0.5, 1.5], 3, 0.02588),
            ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 10, 0.76533),
        ],
    )
    def test_cvm_statistic_values(self, data, df, expected):
        """Test CVM statistic against pre-calculated values."""
        stat = CramerVonMisesStudentGofStatistic(df=df)
        result = stat.execute_statistic(data)
        assert result == pytest.approx(expected, rel=0.05)

    def test_cvm_code(self):
        """Test that CVM statistic returns correct code."""
        assert "CVM_STUDENT_GOODNESS_OF_FIT" == CramerVonMisesStudentGofStatistic.code()

    def test_cvm_with_scipy(self, t_sample_df5):
        """Verify CVM statistic matches scipy calculation."""
        stat = CramerVonMisesStudentGofStatistic(df=5)
        our_result = stat.execute_statistic(t_sample_df5)

        # Compare with scipy cramervonmises
        scipy_result = scipy_stats.cramervonmises(t_sample_df5, scipy_stats.t(df=5).cdf)
        assert our_result == pytest.approx(scipy_result.statistic, rel=0.01)


# Tests for KuiperStudentGofStatistic
class TestKuiperStudent:
    """Tests for Kuiper's statistic."""

    @pytest.mark.parametrize(
        ("data", "df", "expected"),
        [
            # Pre-calculated values for Kuiper statistic
            ([0.5, 1.0, -0.5, 0.0, 2.0], 5, 0.37012),
            ([-1.5, -0.5, 0.5, 1.5], 3, 0.34855),
            ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 10, 0.70929),
        ],
    )
    def test_kuiper_statistic_values(self, data, df, expected):
        """Test Kuiper statistic against pre-calculated values."""
        stat = KuiperStudentGofStatistic(df=df)
        result = stat.execute_statistic(data)
        assert result == pytest.approx(expected, rel=0.05)

    def test_kuiper_code(self):
        """Test that Kuiper statistic returns correct code."""
        assert "KUIPER_STUDENT_GOODNESS_OF_FIT" == KuiperStudentGofStatistic.code()

    def test_kuiper_positive(self, t_sample_df5):
        """Test that Kuiper statistic is always positive."""
        stat = KuiperStudentGofStatistic(df=5)
        result = stat.execute_statistic(t_sample_df5)
        assert result > 0


# Tests for WatsonStudentGofStatistic
class TestWatsonStudent:
    """Tests for Watson's U^2 statistic."""

    @pytest.mark.parametrize(
        ("data", "df", "expected"),
        [
            # Pre-calculated values for Watson U^2 statistic
            ([0.5, 1.0, -0.5, 0.0, 2.0], 5, 0.03603),
            ([-1.5, -0.5, 0.5, 1.5], 3, 0.02588),
            ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 10, 0.38466),
        ],
    )
    def test_watson_statistic_values(self, data, df, expected):
        """Test Watson statistic against pre-calculated values."""
        stat = WatsonStudentGofStatistic(df=df)
        result = stat.execute_statistic(data)
        assert result == pytest.approx(expected, rel=0.1)

    def test_watson_code(self):
        """Test that Watson statistic returns correct code."""
        assert "WATSON_STUDENT_GOODNESS_OF_FIT" == WatsonStudentGofStatistic.code()


# Tests for ZhangZcStudentGofStatistic
class TestZhangZcStudent:
    """Tests for Zhang's Zc statistic."""

    def test_zhang_zc_code(self):
        """Test that Zhang Zc statistic returns correct code."""
        assert "ZHANG_ZC_STUDENT_GOODNESS_OF_FIT" == ZhangZcStudentGofStatistic.code()

    def test_zhang_zc_positive(self, t_sample_df5):
        """Test that Zhang Zc statistic is always positive."""
        stat = ZhangZcStudentGofStatistic(df=5)
        result = stat.execute_statistic(t_sample_df5)
        assert result > 0

    @pytest.mark.parametrize(
        ("data", "df"),
        [
            ([0.5, 1.0, -0.5, 0.0, 2.0], 5),
            ([-1.5, -0.5, 0.5, 1.5], 3),
            ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 10),
        ],
    )
    def test_zhang_zc_finite(self, data, df):
        """Test that Zhang Zc statistic returns finite values."""
        stat = ZhangZcStudentGofStatistic(df=df)
        result = stat.execute_statistic(data)
        assert np.isfinite(result)


# Tests for ZhangZaStudentGofStatistic
class TestZhangZaStudent:
    """Tests for Zhang's Za statistic."""

    def test_zhang_za_code(self):
        """Test that Zhang Za statistic returns correct code."""
        assert "ZHANG_ZA_STUDENT_GOODNESS_OF_FIT" == ZhangZaStudentGofStatistic.code()

    def test_zhang_za_positive(self, t_sample_df5):
        """Test that Zhang Za statistic returns a finite value."""
        stat = ZhangZaStudentGofStatistic(df=5)
        result = stat.execute_statistic(t_sample_df5)
        assert np.isfinite(result)

    @pytest.mark.parametrize(
        ("data", "df"),
        [
            ([0.5, 1.0, -0.5, 0.0, 2.0], 5),
            ([-1.5, -0.5, 0.5, 1.5], 3),
            ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 10),
        ],
    )
    def test_zhang_za_finite(self, data, df):
        """Test that Zhang Za statistic returns finite values."""
        stat = ZhangZaStudentGofStatistic(df=df)
        result = stat.execute_statistic(data)
        assert np.isfinite(result)


# Tests for LillieforsStudentGofStatistic
class TestLillieforsStudent:
    """Tests for Lilliefors-type statistic."""

    def test_lilliefors_code(self):
        """Test that Lilliefors statistic returns correct code."""
        assert "LILLIE_STUDENT_GOODNESS_OF_FIT" == LillieforsStudentGofStatistic.code()

    def test_lilliefors_positive(self, t_sample_df5):
        """Test that Lilliefors statistic is always positive."""
        stat = LillieforsStudentGofStatistic(df=5)
        result = stat.execute_statistic(t_sample_df5)
        assert result > 0

    @pytest.mark.parametrize(
        ("data", "df"),
        [
            ([0.5, 1.0, -0.5, 0.0, 2.0], 5),
            ([-1.5, -0.5, 0.5, 1.5], 3),
            ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 10),
        ],
    )
    def test_lilliefors_less_than_one(self, data, df):
        """Test that Lilliefors statistic is less than 1."""
        stat = LillieforsStudentGofStatistic(df=df)
        result = stat.execute_statistic(data)
        assert 0 < result < 1


# Tests for ChiSquareStudentGofStatistic
class TestChiSquareStudent:
    """Tests for Chi-Square statistic."""

    def test_chi2_code(self):
        """Test that Chi-Square statistic returns correct code."""
        assert "CHI2_STUDENT_GOODNESS_OF_FIT" == ChiSquareStudentGofStatistic.code()

    def test_chi2_positive(self, t_sample_df5):
        """Test that Chi-Square statistic is always non-negative."""
        stat = ChiSquareStudentGofStatistic(df=5, n_bins=5)
        result = stat.execute_statistic(t_sample_df5)
        assert result >= 0

    @pytest.mark.parametrize("n_bins", [5, 10, 20])
    def test_chi2_different_bins(self, t_sample_df5, n_bins):
        """Test Chi-Square with different number of bins."""
        stat = ChiSquareStudentGofStatistic(df=5, n_bins=n_bins)
        result = stat.execute_statistic(t_sample_df5)
        assert result >= 0
        assert np.isfinite(result)


# Integration tests
class TestStudentStatisticsIntegration:
    """Integration tests for Student's t-distribution statistics."""

    def test_all_statistics_run(self, t_sample_df5):
        """Test that all statistics can be computed without errors."""
        statistics = [
            KolmogorovSmirnovStudentGofStatistic(df=5),
            AndersonDarlingStudentGofStatistic(df=5),
            CramerVonMisesStudentGofStatistic(df=5),
            KuiperStudentGofStatistic(df=5),
            WatsonStudentGofStatistic(df=5),
            ZhangZcStudentGofStatistic(df=5),
            ZhangZaStudentGofStatistic(df=5),
            LillieforsStudentGofStatistic(df=5),
            ChiSquareStudentGofStatistic(df=5, n_bins=5),
        ]

        for stat in statistics:
            result = stat.execute_statistic(t_sample_df5)
            assert np.isfinite(result), f"{stat.code()} returned non-finite value"

    def test_statistics_with_location_scale(self, t_sample_df5):
        """Test statistics with non-default location and scale."""
        loc, scale = 5.0, 2.0
        transformed_sample = t_sample_df5 * scale + loc

        stat = KolmogorovSmirnovStudentGofStatistic(df=5, loc=loc, scale=scale)
        result = stat.execute_statistic(transformed_sample)

        # Should be similar to the standard case
        stat_standard = KolmogorovSmirnovStudentGofStatistic(df=5)
        result_standard = stat_standard.execute_statistic(t_sample_df5)

        assert result == pytest.approx(result_standard, rel=0.01)

    def test_power_against_normal(self):
        """Test that statistics detect non-t data."""
        np.random.default_rng(42)
        normal_data = np.random.normal(0, 1, size=1000)
        t_data = np.random.standard_t(df=3, size=1000)

        stat = KolmogorovSmirnovStudentGofStatistic(df=3)

        # Statistic for t-data should be smaller (better fit)
        result_t = stat.execute_statistic(t_data)
        result_normal = stat.execute_statistic(normal_data)

        # Normal data should have larger KS statistic when tested against t(df=3)
        # because t(3) has heavier tails
        assert np.isfinite(result_t)
        assert np.isfinite(result_normal)
        # because of big size of data it is deterministic
        assert result_t < result_normal


# Edge case tests
class TestStudentStatisticsEdgeCases:
    """Edge case tests for Student's t-distribution statistics."""

    def test_small_sample(self):
        """Test statistics with small sample size."""
        data = [0.1, 0.2, 0.3]
        stat = KolmogorovSmirnovStudentGofStatistic(df=5)
        result = stat.execute_statistic(data)
        assert np.isfinite(result)

    def test_large_df(self):
        """Test statistics with large degrees of freedom (approaches normal)."""
        np.random.default_rng(42)
        data = np.random.standard_t(df=100, size=50)
        stat = KolmogorovSmirnovStudentGofStatistic(df=100)
        result = stat.execute_statistic(data)
        assert np.isfinite(result)
        assert result < 0.5  # Should be a good fit

    def test_df_equals_one(self):
        """Test statistics with df=1 (Cauchy distribution)."""
        np.random.default_rng(42)
        data = np.random.standard_cauchy(size=50)
        stat = KolmogorovSmirnovStudentGofStatistic(df=1)
        result = stat.execute_statistic(data)
        assert np.isfinite(result)

    def test_extreme_values(self):
        """Test statistics with extreme values in data."""
        data = [-100, -10, -1, 0, 1, 10, 100]
        stat = KolmogorovSmirnovStudentGofStatistic(df=3)
        result = stat.execute_statistic(data)
        assert np.isfinite(result)
