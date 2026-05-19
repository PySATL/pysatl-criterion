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
        ("data", "alpha", "beta", "result"),
        [
            (
                [
                    0.3536766572,
                    0.2485580661,
                    0.4159590873,
                    0.1599675758,
                    0.5502830781,
                    0.1109452876,
                    0.5098966418,
                    0.1772703795,
                    0.1982904719,
                    0.3762367882,
                ],
                2,
                5,
                0.1877694,
            ),
            (
                [
                    0.7087962001,
                    0.2915205607,
                    0.5508644650,
                    0.8802250282,
                    0.5098339187,
                    0.6131471891,
                    0.3176350910,
                    0.1751690171,
                    0.4660253747,
                    0.5769342726,
                    0.8436683688,
                    0.4333839830,
                ],
                1,
                1,
                0.2081872,
            ),
            (
                [
                    0.6994383163,
                    0.5174900183,
                    0.5925767197,
                    0.5611698399,
                    0.9395027265,
                    0.6820294051,
                    0.4379703453,
                    0.5303633637,
                ],
                0.5,
                0.5,
                0.4604087,
            ),
            (
                [
                    0.7482953444,
                    0.8762706351,
                    0.4603376651,
                    0.7471648301,
                    0.8904978771,
                    0.7206587486,
                    0.8250535532,
                    0.7494859221,
                    0.8777684524,
                    0.9167142296,
                    0.9568773740,
                ],
                5,
                2,
                0.3749592,
            ),
            (
                [
                    0.3366836981,
                    0.5643169242,
                    0.5625418098,
                    0.4038462174,
                    0.3593278603,
                    0.7521677033,
                    0.6460545301,
                    0.7051711915,
                    0.1494688017,
                    0.8260848760,
                ],
                3,
                3,
                0.2160485,
            ),
        ],
    )
    def test_ks_with_parametrized_data(self, data, alpha, beta, result):
        """Test KS statistic with precomputed expected values."""
        stat = KolmogorovSmirnovBetaGofStatistic(alpha=alpha, beta=beta)
        statistic_value = stat.execute_statistic(data)
        assert result == pytest.approx(statistic_value, 0.00001)

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
        ("data", "alpha", "beta", "result"),
        [
            (
                [
                    0.3536766572,
                    0.2485580661,
                    0.4159590873,
                    0.1599675758,
                    0.5502830781,
                    0.1109452876,
                    0.5098966418,
                    0.1772703795,
                    0.1982904719,
                    0.3762367882,
                ],
                2,
                5,
                0.3746031,
            ),
            (
                [
                    0.7087962001,
                    0.2915205607,
                    0.5508644650,
                    0.8802250282,
                    0.5098339187,
                    0.6131471891,
                    0.3176350910,
                    0.1751690171,
                    0.4660253747,
                    0.5769342726,
                    0.8436683688,
                    0.4333839830,
                ],
                1,
                1,
                0.7490749,
            ),
            (
                [
                    0.3366836981,
                    0.5643169242,
                    0.5625418098,
                    0.4038462174,
                    0.3593278603,
                    0.7521677033,
                    0.6460545301,
                    0.7051711915,
                    0.1494688017,
                    0.8260848760,
                ],
                3,
                3,
                0.3998264,
            ),
        ],
    )
    def test_ad_with_parametrized_data(self, data, alpha, beta, result):
        """Test AD statistic with precomputed expected values."""
        stat = AndersonDarlingBetaGofStatistic(alpha=alpha, beta=beta)
        statistic_value = stat.execute_statistic(data)
        assert result == pytest.approx(statistic_value, 0.00001)

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
        ("data", "alpha", "beta", "result"),
        [
            (
                [
                    0.3536766572,
                    0.2485580661,
                    0.4159590873,
                    0.1599675758,
                    0.5502830781,
                    0.1109452876,
                    0.5098966418,
                    0.1772703795,
                    0.1982904719,
                    0.3762367882,
                ],
                2,
                5,
                0.0565441,
            ),
            (
                [
                    0.7087962001,
                    0.2915205607,
                    0.5508644650,
                    0.8802250282,
                    0.5098339187,
                    0.6131471891,
                    0.3176350910,
                    0.1751690171,
                    0.4660253747,
                    0.5769342726,
                    0.8436683688,
                    0.4333839830,
                ],
                1,
                1,
                0.1208704,
            ),
            (
                [
                    0.3366836981,
                    0.5643169242,
                    0.5625418098,
                    0.4038462174,
                    0.3593278603,
                    0.7521677033,
                    0.6460545301,
                    0.7051711915,
                    0.1494688017,
                    0.8260848760,
                ],
                3,
                3,
                0.0692098,
            ),
        ],
    )
    def test_cvm_with_parametrized_data(self, data, alpha, beta, result):
        """Test CVM statistic with precomputed expected values."""
        stat = CrammerVonMisesBetaGofStatistic(alpha=alpha, beta=beta)
        statistic_value = stat.execute_statistic(data)
        assert result == pytest.approx(statistic_value, 0.00001)

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
        ("data", "alpha", "beta", "result"),
        [
            (
                [
                    0.3536766572,
                    0.2485580661,
                    0.4159590873,
                    0.1599675758,
                    0.5502830781,
                    0.1109452876,
                    0.5098966418,
                    0.1772703795,
                    0.1982904719,
                    0.3762367882,
                ],
                2,
                5,
                0.1877694,
            ),
            (
                [
                    0.3366836981,
                    0.5643169242,
                    0.5625418098,
                    0.4038462174,
                    0.3593278603,
                    0.7521677033,
                    0.6460545301,
                    0.7051711915,
                    0.1494688017,
                    0.8260848760,
                ],
                3,
                3,
                0.2160485,
            ),
        ],
    )
    def test_lillie_with_parametrized_data(self, data, alpha, beta, result):
        """Test Lilliefors statistic with precomputed expected values."""
        stat = LillieforsTestBetaGofStatistic(alpha=alpha, beta=beta)
        statistic_value = stat.execute_statistic(data)
        assert result == pytest.approx(statistic_value, 0.00001)


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

    @pytest.mark.parametrize(
        ("data", "alpha", "beta", "result"),
        [
            (
                [
                    0.3536766572,
                    0.2485580661,
                    0.4159590873,
                    0.1599675758,
                    0.5502830781,
                    0.1109452876,
                    0.5098966418,
                    0.1772703795,
                    0.1982904719,
                    0.3762367882,
                ],
                2,
                5,
                1.30301816,
            ),
            (
                [
                    0.7087962001,
                    0.2915205607,
                    0.5508644650,
                    0.8802250282,
                    0.5098339187,
                    0.6131471891,
                    0.3176350910,
                    0.1751690171,
                    0.4660253747,
                    0.5769342726,
                    0.8436683688,
                    0.4333839830,
                ],
                1,
                1,
                3.33333333,
            ),
            (
                [
                    0.3366836981,
                    0.5643169242,
                    0.5625418098,
                    0.4038462174,
                    0.3593278603,
                    0.7521677033,
                    0.6460545301,
                    0.7051711915,
                    0.1494688017,
                    0.8260848760,
                ],
                3,
                3,
                1.13560740,
            ),
        ],
    )
    def test_chi2_with_parametrized_data(self, data, alpha, beta, result):
        """Test Chi-squared statistic with precomputed expected values."""
        stat = Chi2PearsonBetaGofStatistic(alpha=alpha, beta=beta)
        statistic_value = stat.execute_statistic(data)
        assert result == pytest.approx(statistic_value, 0.00001)


class TestWatsonBetaGofStatistic:
    """Tests for Watson test statistic."""

    def test_code(self):
        """Test that the Watson statistic returns correct code."""
        assert "W_BETA_GOODNESS_OF_FIT" == WatsonBetaGofStatistic.code()

    @pytest.mark.parametrize(
        ("data", "alpha", "beta", "result"),
        [
            (
                [
                    0.3536766572,
                    0.2485580661,
                    0.4159590873,
                    0.1599675758,
                    0.5502830781,
                    0.1109452876,
                    0.5098966418,
                    0.1772703795,
                    0.1982904719,
                    0.3762367882,
                ],
                2,
                5,
                0.0302649,
            ),
            (
                [
                    0.7087962001,
                    0.2915205607,
                    0.5508644650,
                    0.8802250282,
                    0.5098339187,
                    0.6131471891,
                    0.3176350910,
                    0.1751690171,
                    0.4660253747,
                    0.5769342726,
                    0.8436683688,
                    0.4333839830,
                ],
                1,
                1,
                0.1096339,
            ),
            (
                [
                    0.3366836981,
                    0.5643169242,
                    0.5625418098,
                    0.4038462174,
                    0.3593278603,
                    0.7521677033,
                    0.6460545301,
                    0.7051711915,
                    0.1494688017,
                    0.8260848760,
                ],
                3,
                3,
                0.0430198,
            ),
        ],
    )
    def test_watson_with_parametrized_data(self, data, alpha, beta, result):
        """Test Watson statistic with precomputed expected values."""
        stat = WatsonBetaGofStatistic(alpha=alpha, beta=beta)
        statistic_value = stat.execute_statistic(data)
        assert result == pytest.approx(statistic_value, 0.00001)


class TestKuiperBetaGofStatistic:
    """Tests for Kuiper test statistic."""

    def test_code(self):
        """Test that the Kuiper statistic returns correct code."""
        assert "KUIPER_BETA_GOODNESS_OF_FIT" == KuiperBetaGofStatistic.code()

    @pytest.mark.parametrize(
        ("data", "alpha", "beta", "result"),
        [
            (
                [
                    0.3536766572,
                    0.2485580661,
                    0.4159590873,
                    0.1599675758,
                    0.5502830781,
                    0.1109452876,
                    0.5098966418,
                    0.1772703795,
                    0.1982904719,
                    0.3762367882,
                ],
                2,
                5,
                0.2567761,
            ),
            (
                [
                    0.7087962001,
                    0.2915205607,
                    0.5508644650,
                    0.8802250282,
                    0.5098339187,
                    0.6131471891,
                    0.3176350910,
                    0.1751690171,
                    0.4660253747,
                    0.5769342726,
                    0.8436683688,
                    0.4333839830,
                ],
                1,
                1,
                0.3450400,
            ),
            (
                [
                    0.3366836981,
                    0.5643169242,
                    0.5625418098,
                    0.4038462174,
                    0.3593278603,
                    0.7521677033,
                    0.6460545301,
                    0.7051711915,
                    0.1494688017,
                    0.8260848760,
                ],
                3,
                3,
                0.2919412,
            ),
        ],
    )
    def test_kuiper_with_parametrized_data(self, data, alpha, beta, result):
        """Test Kuiper statistic with precomputed expected values."""
        stat = KuiperBetaGofStatistic(alpha=alpha, beta=beta)
        statistic_value = stat.execute_statistic(data)
        assert result == pytest.approx(statistic_value, 0.00001)


class TestMomentBasedBetaGofStatistic:
    """Tests for Moment-based test statistic."""

    def test_code(self):
        """Test that the Moment-based statistic returns correct code."""
        assert "MB_BETA_GOODNESS_OF_FIT" == MomentBasedBetaGofStatistic.code()

    @pytest.mark.parametrize(
        ("data", "alpha", "beta", "result"),
        [
            (
                [
                    0.3536766572,
                    0.2485580661,
                    0.4159590873,
                    0.1599675758,
                    0.5502830781,
                    0.1109452876,
                    0.5098966418,
                    0.1772703795,
                    0.1982904719,
                    0.3762367882,
                ],
                2,
                5,
                0.2349020,
            ),
            (
                [
                    0.7087962001,
                    0.2915205607,
                    0.5508644650,
                    0.8802250282,
                    0.5098339187,
                    0.6131471891,
                    0.3176350910,
                    0.1751690171,
                    0.4660253747,
                    0.5769342726,
                    0.8436683688,
                    0.4333839830,
                ],
                1,
                1,
                0.3372358,
            ),
            (
                [
                    0.3366836981,
                    0.5643169242,
                    0.5625418098,
                    0.4038462174,
                    0.3593278603,
                    0.7521677033,
                    0.6460545301,
                    0.7051711915,
                    0.1494688017,
                    0.8260848760,
                ],
                3,
                3,
                0.2891104,
            ),
        ],
    )
    def test_moment_with_parametrized_data(self, data, alpha, beta, result):
        """Test Moment-based statistic with precomputed expected values."""
        stat = MomentBasedBetaGofStatistic(alpha=alpha, beta=beta)
        statistic_value = stat.execute_statistic(data)
        assert result == pytest.approx(statistic_value, 0.00001)


class TestSkewnessKurtosisBetaGofStatistic:
    """Tests for Skewness-Kurtosis test statistic."""

    def test_code(self):
        """Test that the Skewness-Kurtosis statistic returns correct code."""
        assert "SK_BETA_GOODNESS_OF_FIT" == SkewnessKurtosisBetaGofStatistic.code()

    @pytest.mark.parametrize(
        ("data", "alpha", "beta", "result"),
        [
            (
                [
                    0.3536766572,
                    0.2485580661,
                    0.4159590873,
                    0.1599675758,
                    0.5502830781,
                    0.1109452876,
                    0.5098966418,
                    0.1772703795,
                    0.1982904719,
                    0.3762367882,
                ],
                2,
                5,
                0.7277307,
            ),
            (
                [
                    0.7087962001,
                    0.2915205607,
                    0.5508644650,
                    0.8802250282,
                    0.5098339187,
                    0.6131471891,
                    0.3176350910,
                    0.1751690171,
                    0.4660253747,
                    0.5769342726,
                    0.8436683688,
                    0.4333839830,
                ],
                1,
                1,
                0.2648237,
            ),
            (
                [
                    0.3366836981,
                    0.5643169242,
                    0.5625418098,
                    0.4038462174,
                    0.3593278603,
                    0.7521677033,
                    0.6460545301,
                    0.7051711915,
                    0.1494688017,
                    0.8260848760,
                ],
                3,
                3,
                0.2303306,
            ),
        ],
    )
    def test_sk_with_parametrized_data(self, data, alpha, beta, result):
        """Test Skewness-Kurtosis statistic with precomputed expected values."""
        stat = SkewnessKurtosisBetaGofStatistic(alpha=alpha, beta=beta)
        statistic_value = stat.execute_statistic(data)
        assert result == pytest.approx(statistic_value, 0.00001)


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

    @pytest.mark.parametrize(
        ("data", "alpha", "beta", "result"),
        [
            (
                [
                    0.3536766572,
                    0.2485580661,
                    0.4159590873,
                    0.1599675758,
                    0.5502830781,
                    0.1109452876,
                    0.5098966418,
                    0.1772703795,
                    0.1982904719,
                    0.3762367882,
                ],
                2,
                5,
                0.20054649,
            ),
            (
                [
                    0.7087962001,
                    0.2915205607,
                    0.5508644650,
                    0.8802250282,
                    0.5098339187,
                    0.6131471891,
                    0.3176350910,
                    0.1751690171,
                    0.4660253747,
                    0.5769342726,
                    0.8436683688,
                    0.4333839830,
                ],
                1,
                1,
                0.62062110,
            ),
            (
                [
                    0.3366836981,
                    0.5643169242,
                    0.5625418098,
                    0.4038462174,
                    0.3593278603,
                    0.7521677033,
                    0.6460545301,
                    0.7051711915,
                    0.1494688017,
                    0.8260848760,
                ],
                3,
                3,
                0.02561022,
            ),
        ],
    )
    def test_ratio_with_parametrized_data(self, data, alpha, beta, result):
        """Test Ratio statistic with precomputed expected values."""
        stat = RatioBetaGofStatistic(alpha=alpha, beta=beta)
        statistic_value = stat.execute_statistic(data)
        assert result == pytest.approx(statistic_value, 0.00001)

    def test_ratio_validation_errors(self):
        """Test that Ratio statistic validates input data."""
        stat = RatioBetaGofStatistic(alpha=2, beta=3)

        # Data with exact 0 or 1 (for log calculation)
        with pytest.raises(
            ValueError, match="Beta distribution values must be in the open interval"
        ):
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

    @pytest.mark.parametrize(
        ("data", "alpha", "beta", "result"),
        [
            (
                [
                    0.3536766572,
                    0.2485580661,
                    0.4159590873,
                    0.1599675758,
                    0.5502830781,
                    0.1109452876,
                    0.5098966418,
                    0.1772703795,
                    0.1982904719,
                    0.3762367882,
                ],
                2,
                5,
                1.46457551,
            ),
            (
                [
                    0.7087962001,
                    0.2915205607,
                    0.5508644650,
                    0.8802250282,
                    0.5098339187,
                    0.6131471891,
                    0.3176350910,
                    0.1751690171,
                    0.4660253747,
                    0.5769342726,
                    0.8436683688,
                    0.4333839830,
                ],
                1,
                1,
                1.64485770,
            ),
            (
                [
                    0.3366836981,
                    0.5643169242,
                    0.5625418098,
                    0.4038462174,
                    0.3593278603,
                    0.7521677033,
                    0.6460545301,
                    0.7051711915,
                    0.1494688017,
                    0.8260848760,
                ],
                3,
                3,
                0.86392069,
            ),
        ],
    )
    def test_entropy_with_parametrized_data(self, data, alpha, beta, result):
        """Test Entropy statistic with precomputed expected values."""
        stat = EntropyBetaGofStatistic(alpha=alpha, beta=beta)
        statistic_value = stat.execute_statistic(data)
        assert result == pytest.approx(statistic_value, 0.00001)


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

    @pytest.mark.parametrize(
        ("data", "alpha", "beta", "result"),
        [
            (
                [
                    0.3536766572,
                    0.2485580661,
                    0.4159590873,
                    0.1599675758,
                    0.5502830781,
                    0.1109452876,
                    0.5098966418,
                    0.1772703795,
                    0.1982904719,
                    0.3762367882,
                ],
                2,
                5,
                0.03766130,
            ),
            (
                [
                    0.3366836981,
                    0.5643169242,
                    0.5625418098,
                    0.4038462174,
                    0.3593278603,
                    0.7521677033,
                    0.6460545301,
                    0.7051711915,
                    0.1494688017,
                    0.8260848760,
                ],
                3,
                3,
                0.39711215,
            ),
        ],
    )
    def test_mode_with_parametrized_data(self, data, alpha, beta, result):
        """Test Mode statistic with precomputed expected values."""
        stat = ModeBetaGofStatistic(alpha=alpha, beta=beta)
        statistic_value = stat.execute_statistic(data)
        assert result == pytest.approx(statistic_value, 0.00001)


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
