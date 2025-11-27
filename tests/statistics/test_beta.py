import numpy as np
import pytest
import scipy.stats as scipy_stats

from pysatl_criterion.statistics.beta import (
    AbstractBetaGofStatistic,
    AndersonDarlingBetaGofStatistic,
    Chi2PearsonBetaGofStatistic,
    CrammerVonMisesBetaGofStatistic,
    EntropyBetaGofStatistic,
    KolmogorovSmirnovBetaGofStatistic,
    KuiperBetaGofStatistic,
    LillieforsTestBetaGofStatistic,
    ModeBetaGofStatistic,
    MomentBasedBetaGofStatistic,
    RatioBetaGofStatistic,
    SkewnessKurtosisBetaGofStatistic,
    WatsonBetaGofStatistic,
)


def test_abstract_beta_criterion_code():
    """Test that the abstract Beta class returns correct code."""
    assert "BETA_GOODNESS_OF_FIT" == AbstractBetaGofStatistic.code()


def test_abstract_beta_criterion_parameters():
    """Test parameter validation for AbstractBetaGofStatistic."""
    # Test with concrete class since AbstractBetaGofStatistic is abstract
    # Valid parameters
    stat = KolmogorovSmirnovBetaGofStatistic(alpha=2, beta=3)
    assert stat.alpha == 2
    assert stat.beta == 3
    
    # Invalid alpha
    with pytest.raises(ValueError, match="alpha must be positive"):
        KolmogorovSmirnovBetaGofStatistic(alpha=-1, beta=2)
    
    with pytest.raises(ValueError, match="alpha must be positive"):
        KolmogorovSmirnovBetaGofStatistic(alpha=0, beta=2)
    
    # Invalid beta
    with pytest.raises(ValueError, match="beta must be positive"):
        KolmogorovSmirnovBetaGofStatistic(alpha=2, beta=-1)
    
    with pytest.raises(ValueError, match="beta must be positive"):
        KolmogorovSmirnovBetaGofStatistic(alpha=2, beta=0)


class TestKolmogorovSmirnovBetaGofStatistic:
    """Tests for Kolmogorov-Smirnov test statistic."""
    
    def test_code(self):
        """Test that the KS statistic returns correct code."""
        assert "KS_BETA_GOODNESS_OF_FIT" == KolmogorovSmirnovBetaGofStatistic.code()
    
    @pytest.mark.parametrize(
        ("alpha", "beta", "seed", "n"),
        [
            (2, 5, 42, 50),
            (1, 1, 123, 100),  # Uniform distribution
            (0.5, 0.5, 456, 75),
            (5, 2, 789, 200),
        ],
    )
    def test_ks_with_generated_data(self, alpha, beta, seed, n):
        """Test KS statistic with data generated from Beta distribution."""
        np.random.seed(seed)
        data = scipy_stats.beta.rvs(alpha, beta, size=n)
        
        stat = KolmogorovSmirnovBetaGofStatistic(alpha=alpha, beta=beta)
        statistic_value = stat.execute_statistic(data)
        
        # Statistic should be non-negative
        assert statistic_value >= 0
        
        # For data from the correct distribution, statistic should be relatively small
        # (though this is a probabilistic statement)
        assert statistic_value < 0.5  # Reasonable upper bound for good fit
    
    def test_ks_with_wrong_distribution(self):
        """Test KS statistic with data from wrong distribution."""
        np.random.seed(42)
        # Generate from Beta(5, 2) but test against Beta(2, 5)
        data = scipy_stats.beta.rvs(5, 2, size=100)
        
        stat = KolmogorovSmirnovBetaGofStatistic(alpha=2, beta=5)
        statistic_value = stat.execute_statistic(data)
        
        # Should detect the mismatch
        assert statistic_value > 0.1
    
    def test_ks_validation_errors(self):
        """Test that KS statistic validates input data."""
        stat = KolmogorovSmirnovBetaGofStatistic(alpha=2, beta=3)
        
        # Data outside [0, 1]
        with pytest.raises(ValueError, match="Beta distribution values must be in the interval"):
            stat.execute_statistic([0.5, 1.5, 0.3])
        
        with pytest.raises(ValueError, match="Beta distribution values must be in the interval"):
            stat.execute_statistic([-0.1, 0.5, 0.8])


class TestAndersonDarlingBetaGofStatistic:
    """Tests for Anderson-Darling test statistic."""
    
    def test_code(self):
        """Test that the AD statistic returns correct code."""
        assert "AD_BETA_GOODNESS_OF_FIT" == AndersonDarlingBetaGofStatistic.code()
    
    @pytest.mark.parametrize(
        ("alpha", "beta", "seed", "n"),
        [
            (2, 5, 42, 50),
            (1, 1, 123, 100),
            (3, 3, 456, 75),
        ],
    )
    def test_ad_with_generated_data(self, alpha, beta, seed, n):
        """Test AD statistic with data generated from Beta distribution."""
        np.random.seed(seed)
        data = scipy_stats.beta.rvs(alpha, beta, size=n)
        
        stat = AndersonDarlingBetaGofStatistic(alpha=alpha, beta=beta)
        statistic_value = stat.execute_statistic(data)
        
        # Statistic should be finite
        assert np.isfinite(statistic_value)
        
        # For data from the correct distribution, statistic should be reasonable
        assert statistic_value < 10  # Reasonable upper bound for good fit
    
    def test_ad_validation_errors(self):
        """Test that AD statistic validates input data."""
        stat = AndersonDarlingBetaGofStatistic(alpha=2, beta=3)
        
        with pytest.raises(ValueError, match="Beta distribution values must be in the interval"):
            stat.execute_statistic([0.5, 1.5, 0.3])


class TestCrammerVonMisesBetaGofStatistic:
    """Tests for Cramér-von Mises test statistic."""
    
    def test_code(self):
        """Test that the CVM statistic returns correct code."""
        assert "CVM_BETA_GOODNESS_OF_FIT" == CrammerVonMisesBetaGofStatistic.code()
    
    @pytest.mark.parametrize(
        ("alpha", "beta", "seed", "n"),
        [
            (2, 5, 42, 50),
            (1, 1, 123, 100),
            (3, 3, 456, 75),
        ],
    )
    def test_cvm_with_generated_data(self, alpha, beta, seed, n):
        """Test CVM statistic with data generated from Beta distribution."""
        np.random.seed(seed)
        data = scipy_stats.beta.rvs(alpha, beta, size=n)
        
        stat = CrammerVonMisesBetaGofStatistic(alpha=alpha, beta=beta)
        statistic_value = stat.execute_statistic(data)
        
        # Statistic should be non-negative
        assert statistic_value >= 0
        
        # For data from the correct distribution, statistic should be small
        assert statistic_value < 1  # Reasonable upper bound for good fit
    
    def test_cvm_validation_errors(self):
        """Test that CVM statistic validates input data."""
        stat = CrammerVonMisesBetaGofStatistic(alpha=2, beta=3)
        
        with pytest.raises(ValueError, match="Beta distribution values must be in the interval"):
            stat.execute_statistic([0.5, 1.5, 0.3])


class TestLillieforsTestBetaGofStatistic:
    """Tests for Lilliefors test statistic."""
    
    def test_code(self):
        """Test that the Lilliefors statistic returns correct code."""
        assert "LILLIE_BETA_GOODNESS_OF_FIT" == LillieforsTestBetaGofStatistic.code()
    
    @pytest.mark.parametrize(
        ("alpha", "beta", "seed", "n"),
        [
            (2, 5, 42, 50),
            (3, 3, 123, 100),
        ],
    )
    def test_lillie_with_generated_data(self, alpha, beta, seed, n):
        """Test Lilliefors statistic with data generated from Beta distribution."""
        np.random.seed(seed)
        data = scipy_stats.beta.rvs(alpha, beta, size=n)
        
        stat = LillieforsTestBetaGofStatistic(alpha=alpha, beta=beta)
        statistic_value = stat.execute_statistic(data)
        
        # Statistic should be non-negative
        assert statistic_value >= 0
        assert statistic_value < 0.5


class TestChi2PearsonBetaGofStatistic:
    """Tests for Pearson's Chi-squared test statistic."""
    
    def test_code(self):
        """Test that the Chi-squared statistic returns correct code."""
        assert "CHI2_PEARSON_BETA_GOODNESS_OF_FIT" == Chi2PearsonBetaGofStatistic.code()
    
    @pytest.mark.parametrize(
        ("alpha", "beta", "seed", "n"),
        [
            (2, 5, 42, 100),
            (3, 3, 123, 200),
        ],
    )
    def test_chi2_with_generated_data(self, alpha, beta, seed, n):
        """Test Chi-squared statistic with data generated from Beta distribution."""
        np.random.seed(seed)
        data = scipy_stats.beta.rvs(alpha, beta, size=n)
        
        stat = Chi2PearsonBetaGofStatistic(alpha=alpha, beta=beta)
        statistic_value = stat.execute_statistic(data)
        
        # Statistic should be non-negative
        assert statistic_value >= 0
        assert np.isfinite(statistic_value)


class TestWatsonBetaGofStatistic:
    """Tests for Watson test statistic."""
    
    def test_code(self):
        """Test that the Watson statistic returns correct code."""
        assert "W_BETA_GOODNESS_OF_FIT" == WatsonBetaGofStatistic.code()
    
    @pytest.mark.parametrize(
        ("alpha", "beta", "seed", "n"),
        [
            (2, 5, 42, 50),
            (3, 3, 123, 100),
        ],
    )
    def test_watson_with_generated_data(self, alpha, beta, seed, n):
        """Test Watson statistic with data generated from Beta distribution."""
        np.random.seed(seed)
        data = scipy_stats.beta.rvs(alpha, beta, size=n)
        
        stat = WatsonBetaGofStatistic(alpha=alpha, beta=beta)
        statistic_value = stat.execute_statistic(data)
        
        # Statistic should be finite
        assert np.isfinite(statistic_value)


class TestKuiperBetaGofStatistic:
    """Tests for Kuiper test statistic."""
    
    def test_code(self):
        """Test that the Kuiper statistic returns correct code."""
        assert "KUIPER_BETA_GOODNESS_OF_FIT" == KuiperBetaGofStatistic.code()
    
    @pytest.mark.parametrize(
        ("alpha", "beta", "seed", "n"),
        [
            (2, 5, 42, 50),
            (3, 3, 123, 100),
        ],
    )
    def test_kuiper_with_generated_data(self, alpha, beta, seed, n):
        """Test Kuiper statistic with data generated from Beta distribution."""
        np.random.seed(seed)
        data = scipy_stats.beta.rvs(alpha, beta, size=n)
        
        stat = KuiperBetaGofStatistic(alpha=alpha, beta=beta)
        statistic_value = stat.execute_statistic(data)
        
        # Statistic should be non-negative
        assert statistic_value >= 0
        assert statistic_value < 1  # Reasonable upper bound


class TestMomentBasedBetaGofStatistic:
    """Tests for Moment-based test statistic."""
    
    def test_code(self):
        """Test that the Moment-based statistic returns correct code."""
        assert "MB_BETA_GOODNESS_OF_FIT" == MomentBasedBetaGofStatistic.code()
    
    @pytest.mark.parametrize(
        ("alpha", "beta", "seed", "n"),
        [
            (2, 5, 42, 100),
            (3, 3, 123, 200),
        ],
    )
    def test_moment_with_generated_data(self, alpha, beta, seed, n):
        """Test Moment-based statistic with data generated from Beta distribution."""
        np.random.seed(seed)
        data = scipy_stats.beta.rvs(alpha, beta, size=n)
        
        stat = MomentBasedBetaGofStatistic(alpha=alpha, beta=beta)
        statistic_value = stat.execute_statistic(data)
        
        # Statistic should be non-negative
        assert statistic_value >= 0
        assert np.isfinite(statistic_value)
    
    def test_moment_expected_value(self):
        """Test that moment-based statistic is small for correct distribution."""
        np.random.seed(42)
        # Generate large sample for stable moments
        data = scipy_stats.beta.rvs(2, 5, size=1000)
        
        stat = MomentBasedBetaGofStatistic(alpha=2, beta=5)
        statistic_value = stat.execute_statistic(data)
        
        # For correct distribution and large sample, should be relatively small
        assert statistic_value < 10


class TestSkewnessKurtosisBetaGofStatistic:
    """Tests for Skewness-Kurtosis test statistic."""
    
    def test_code(self):
        """Test that the Skewness-Kurtosis statistic returns correct code."""
        assert "SK_BETA_GOODNESS_OF_FIT" == SkewnessKurtosisBetaGofStatistic.code()
    
    @pytest.mark.parametrize(
        ("alpha", "beta", "seed", "n"),
        [
            (2, 5, 42, 100),
            (3, 3, 123, 200),
        ],
    )
    def test_sk_with_generated_data(self, alpha, beta, seed, n):
        """Test Skewness-Kurtosis statistic with data generated from Beta distribution."""
        np.random.seed(seed)
        data = scipy_stats.beta.rvs(alpha, beta, size=n)
        
        stat = SkewnessKurtosisBetaGofStatistic(alpha=alpha, beta=beta)
        statistic_value = stat.execute_statistic(data)
        
        # Statistic should be non-negative
        assert statistic_value >= 0
        assert np.isfinite(statistic_value)


class TestRatioBetaGofStatistic:
    """Tests for Ratio test statistic."""
    
    def test_code(self):
        """Test that the Ratio statistic returns correct code."""
        assert "RT_BETA_GOODNESS_OF_FIT" == RatioBetaGofStatistic.code()
    
    @pytest.mark.parametrize(
        ("alpha", "beta", "seed", "n"),
        [
            (2, 5, 42, 100),
            (3, 3, 123, 200),
        ],
    )
    def test_ratio_with_generated_data(self, alpha, beta, seed, n):
        """Test Ratio statistic with data generated from Beta distribution."""
        np.random.seed(seed)
        # Add small noise to avoid exact 0 or 1
        data = scipy_stats.beta.rvs(alpha, beta, size=n)
        data = np.clip(data, 0.001, 0.999)
        
        stat = RatioBetaGofStatistic(alpha=alpha, beta=beta)
        statistic_value = stat.execute_statistic(data)
        
        # Statistic should be non-negative
        assert statistic_value >= 0
        assert np.isfinite(statistic_value)
    
    def test_ratio_validation_errors(self):
        """Test that Ratio statistic validates input data."""
        stat = RatioBetaGofStatistic(alpha=2, beta=3)
        
        # Data with exact 0 or 1 (for log calculation)
        with pytest.raises(ValueError, match="Beta distribution values must be in the open interval"):
            stat.execute_statistic([0.5, 1.0, 0.3])


class TestEntropyBetaGofStatistic:
    """Tests for Entropy-based test statistic."""
    
    def test_code(self):
        """Test that the Entropy statistic returns correct code."""
        assert "ENT_BETA_GOODNESS_OF_FIT" == EntropyBetaGofStatistic.code()
    
    @pytest.mark.parametrize(
        ("alpha", "beta", "seed", "n"),
        [
            (2, 5, 42, 100),
            (3, 3, 123, 200),
        ],
    )
    def test_entropy_with_generated_data(self, alpha, beta, seed, n):
        """Test Entropy statistic with data generated from Beta distribution."""
        np.random.seed(seed)
        data = scipy_stats.beta.rvs(alpha, beta, size=n)
        
        stat = EntropyBetaGofStatistic(alpha=alpha, beta=beta)
        statistic_value = stat.execute_statistic(data)
        
        # Statistic should be non-negative
        assert statistic_value >= 0
        assert np.isfinite(statistic_value)


class TestModeBetaGofStatistic:
    """Tests for Mode-based test statistic."""
    
    def test_code(self):
        """Test that the Mode statistic returns correct code."""
        assert "MODE_BETA_GOODNESS_OF_FIT" == ModeBetaGofStatistic.code()
    
    def test_mode_parameters_validation(self):
        """Test that Mode statistic validates alpha and beta > 1."""
        # Valid parameters
        stat = ModeBetaGofStatistic(alpha=2, beta=2)
        assert stat.alpha == 2
        assert stat.beta == 2
        
        # Invalid alpha
        with pytest.raises(ValueError, match="alpha must be greater than 1"):
            ModeBetaGofStatistic(alpha=1, beta=2)
        
        # Invalid beta
        with pytest.raises(ValueError, match="beta must be greater than 1"):
            ModeBetaGofStatistic(alpha=2, beta=1)
    
    @pytest.mark.parametrize(
        ("alpha", "beta", "seed", "n"),
        [
            (2, 5, 42, 200),
            (3, 3, 123, 300),
        ],
    )
    def test_mode_with_generated_data(self, alpha, beta, seed, n):
        """Test Mode statistic with data generated from Beta distribution."""
        np.random.seed(seed)
        data = scipy_stats.beta.rvs(alpha, beta, size=n)
        
        stat = ModeBetaGofStatistic(alpha=alpha, beta=beta)
        statistic_value = stat.execute_statistic(data)
        
        # Statistic should be non-negative
        assert statistic_value >= 0
        assert np.isfinite(statistic_value)


# Integration tests
class TestBetaIntegration:
    """Integration tests for Beta distribution statistics."""
    
    def test_uniform_distribution(self):
        """Test that Beta(1,1) equals uniform distribution."""
        np.random.seed(42)
        data = np.random.uniform(0, 1, 100)
        
        # Should fit well to Beta(1, 1)
        ks_stat = KolmogorovSmirnovBetaGofStatistic(alpha=1, beta=1)
        ks_value = ks_stat.execute_statistic(data)
        
        # Should be relatively small
        assert ks_value < 0.2
    
    def test_all_statistics_run(self):
        """Test that all statistics can be computed without errors."""
        np.random.seed(42)
        data = scipy_stats.beta.rvs(2, 5, size=100)
        # Clip for tests that need open interval
        data_clipped = np.clip(data, 0.001, 0.999)
        
        statistics = [
            KolmogorovSmirnovBetaGofStatistic(alpha=2, beta=5),
            AndersonDarlingBetaGofStatistic(alpha=2, beta=5),
            CrammerVonMisesBetaGofStatistic(alpha=2, beta=5),
            LillieforsTestBetaGofStatistic(alpha=2, beta=5),
            Chi2PearsonBetaGofStatistic(alpha=2, beta=5),
            WatsonBetaGofStatistic(alpha=2, beta=5),
            KuiperBetaGofStatistic(alpha=2, beta=5),
            MomentBasedBetaGofStatistic(alpha=2, beta=5),
            SkewnessKurtosisBetaGofStatistic(alpha=2, beta=5),
            EntropyBetaGofStatistic(alpha=2, beta=5),
            ModeBetaGofStatistic(alpha=2, beta=5),
        ]
        
        for stat in statistics:
            try:
                if isinstance(stat, RatioBetaGofStatistic):
                    value = stat.execute_statistic(data_clipped)
                else:
                    value = stat.execute_statistic(data)
                assert np.isfinite(value), f"Statistic {stat.code()} returned non-finite value"
            except Exception as e:
                pytest.fail(f"Statistic {stat.code()} raised an exception: {e}")
        
        # Also test Ratio with clipped data
        ratio_stat = RatioBetaGofStatistic(alpha=2, beta=5)
        ratio_value = ratio_stat.execute_statistic(data_clipped)
        assert np.isfinite(ratio_value)

