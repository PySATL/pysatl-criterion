import numpy as np
import pytest
import scipy.stats as scipy_stats

from pysatl_criterion.statistics.uniform import (
    AbstractUniformGofStatistic,
    KolmogorovSmirnovUniformGofStatistic,
    AndersonDarlingUniformGofStatistic,
    CrammerVonMisesUniformGofStatistic,
    LillieforsTestUniformGofStatistic,
    Chi2PearsonUniformGofStatistic,
    WatsonUniformGofStatistic,
    KuiperUniformGofStatistic,
    GreenwoodTestUniformGofStatistic,
    BickelRosenblattUniformGofStatistic,
    ZhangTestsUniformGofStatistic,
    SteinUniformGofStatistic,
    CensoredSteinUniformGofStatistic,
    NeymanSmoothTestUniformGofStatistic,
    ShermanUniformGofStatistic,
    QuesenberryMillerUniformGofStatistic,
)


def test_abstract_uniform_criterion_code():
    """Test that the abstract Uniform class returns correct code."""
    assert "UNIFORM_GOODNESS_OF_FIT" == AbstractUniformGofStatistic.code()


def test_abstract_uniform_criterion_parameters():
    """Test parameter validation for AbstractUniformGofStatistic."""
    stat = KolmogorovSmirnovUniformGofStatistic(a=0, b=1)
    assert stat.a == 0
    assert stat.b == 1

    stat = KolmogorovSmirnovUniformGofStatistic(a=2, b=5)
    assert stat.a == 2
    assert stat.b == 5

    with pytest.raises(ValueError, match="b must be greater than a"):
        KolmogorovSmirnovUniformGofStatistic(a=5, b=5)

    with pytest.raises(ValueError, match="b must be greater than a"):
        KolmogorovSmirnovUniformGofStatistic(a=5, b=4)


class TestKolmogorovSmirnovUniformGofStatistic:
    """Tests for Kolmogorov-Smirnov test statistic."""

    def test_code(self):
        """Test that the KS statistic returns correct code."""
        assert "KS_UNIFORM_GOODNESS_OF_FIT" == KolmogorovSmirnovUniformGofStatistic.code()

    @pytest.mark.parametrize(
        ("a", "b", "seed", "n"),
        [
            (0, 1, 42, 50),
            (0, 10, 123, 100),
            (2, 5, 456, 75),
            (-1, 1, 789, 200),
        ],
    )
    def test_ks_with_generated_data(self, a, b, seed, n):
        """Test KS statistic with data generated from Uniform distribution."""
        np.random.seed(seed)
        data = np.random.uniform(a, b, n)

        stat = KolmogorovSmirnovUniformGofStatistic(a=a, b=b)
        statistic_value = stat.execute_statistic(data)

        assert statistic_value >= 0

        assert statistic_value < 0.2

    def test_ks_with_wrong_distribution(self):
        """Test KS statistic with data from wrong distribution."""
        np.random.seed(42)
        data = scipy_stats.expon.rvs(size=100)
        data_normalized = data / np.max(data)

        stat = KolmogorovSmirnovUniformGofStatistic(a=0, b=1)
        statistic_value = stat.execute_statistic(data_normalized)

        assert statistic_value > 0.4

    def test_ks_validation_errors(self):
        """Test that KS statistic validates input data."""
        stat = KolmogorovSmirnovUniformGofStatistic(a=0, b=1)

        with pytest.raises(ValueError, match="Uniform distribution values must be in the interval"):
            stat.execute_statistic([0.5, 1.5, 0.3])

        with pytest.raises(ValueError, match="Uniform distribution values must be in the interval"):
            stat.execute_statistic([-0.1, 0.5, 0.8])


class TestAndersonDarlingUniformGofStatistic:
    """Tests for Anderson-Darling test statistic."""

    def test_code(self):
        """Test that the AD statistic returns correct code."""
        assert "AD_UNIFORM_GOODNESS_OF_FIT" == AndersonDarlingUniformGofStatistic.code()

    @pytest.mark.parametrize(
        ("a", "b", "seed", "n"),
        [
            (0, 1, 42, 50),
            (0, 10, 123, 100),
            (2, 5, 456, 75),
        ],
    )
    def test_ad_with_generated_data(self, a, b, seed, n):
        """Test AD statistic with data generated from Uniform distribution."""
        np.random.seed(seed)
        data = np.random.uniform(a, b, n)

        stat = AndersonDarlingUniformGofStatistic(a=a, b=b)
        statistic_value = stat.execute_statistic(data)

        assert np.isfinite(statistic_value)

        assert statistic_value < 10

    def test_ad_validation_errors(self):
        """Test that AD statistic validates input data."""
        stat = AndersonDarlingUniformGofStatistic(a=0, b=1)

        with pytest.raises(ValueError, match="Uniform distribution values must be in the interval"):
            stat.execute_statistic([0.5, 1.5, 0.3])


class TestCrammerVonMisesUniformGofStatistic:
    """Tests for Cramér-von Mises test statistic."""

    def test_code(self):
        """Test that the CVM statistic returns correct code."""
        assert "CVM_UNIFORM_GOODNESS_OF_FIT" == CrammerVonMisesUniformGofStatistic.code()

    @pytest.mark.parametrize(
        ("a", "b", "seed", "n"),
        [
            (0, 1, 42, 50),
            (0, 10, 123, 100),
            (2, 5, 456, 75),
        ],
    )
    def test_cvm_with_generated_data(self, a, b, seed, n):
        """Test CVM statistic with data generated from Uniform distribution."""
        np.random.seed(seed)
        data = np.random.uniform(a, b, n)

        stat = CrammerVonMisesUniformGofStatistic(a=a, b=b)
        statistic_value = stat.execute_statistic(data)

        assert statistic_value >= 0

        assert statistic_value < 1

    def test_cvm_validation_errors(self):
        """Test that CVM statistic validates input data."""
        stat = CrammerVonMisesUniformGofStatistic(a=0, b=1)

        with pytest.raises(ValueError, match="Uniform distribution values must be in the interval"):
            stat.execute_statistic([0.5, 1.5, 0.3])


class TestLillieforsTestUniformGofStatistic:
    """Tests for Lilliefors test statistic."""

    def test_code(self):
        """Test that the Lilliefors statistic returns correct code."""
        assert "LILLIE_UNIFORM_GOODNESS_OF_FIT" == LillieforsTestUniformGofStatistic.code()

    @pytest.mark.parametrize(
        ("a", "b", "seed", "n"),
        [
            (0, 1, 42, 50),
            (0, 10, 123, 100),
        ],
    )
    def test_lillie_with_generated_data(self, a, b, seed, n):
        """Test Lilliefors statistic with data generated from Uniform distribution."""
        np.random.seed(seed)
        data = np.random.uniform(a, b, n)

        stat = LillieforsTestUniformGofStatistic(a=a, b=b)
        statistic_value = stat.execute_statistic(data)

        assert statistic_value >= 0
        assert statistic_value < 0.5


class TestChi2PearsonUniformGofStatistic:
    """Tests for Pearson's Chi-squared test statistic."""

    def test_code(self):
        """Test that the Chi-squared statistic returns correct code."""
        assert "CHI2_PEARSON_UNIFORM_GOODNESS_OF_FIT" == Chi2PearsonUniformGofStatistic.code()

    @pytest.mark.parametrize(
        ("a", "b", "bins", "seed", "n"),
        [
            (0, 1, 'sturges', 42, 100),
            (0, 10, 'sqrt', 123, 200),
            (2, 5, 5, 456, 150),
        ],
    )
    def test_chi2_with_generated_data(self, a, b, bins, seed, n):
        """Test Chi-squared statistic with data generated from Uniform distribution."""
        np.random.seed(seed)
        data = np.random.uniform(a, b, n)

        stat = Chi2PearsonUniformGofStatistic(a=a, b=b, bins=bins)
        statistic_value = stat.execute_statistic(data)

        assert statistic_value >= 0
        assert np.isfinite(statistic_value)

    def test_chi2_bin_selection(self):
        """Test different bin selection methods."""
        np.random.seed(42)
        data = np.random.uniform(0, 1, 100)

        for bins in ['sturges', 'sqrt', 'auto', 10]:
            stat = Chi2PearsonUniformGofStatistic(bins=bins)
            statistic_value = stat.execute_statistic(data)
            assert statistic_value >= 0
            assert np.isfinite(statistic_value)


class TestWatsonUniformGofStatistic:
    """Tests for Watson test statistic."""

    def test_code(self):
        """Test that the Watson statistic returns correct code."""
        assert "WATSON_UNIFORM_GOODNESS_OF_FIT" == WatsonUniformGofStatistic.code()

    @pytest.mark.parametrize(
        ("a", "b", "seed", "n"),
        [
            (0, 1, 42, 50),
            (0, 10, 123, 100),
        ],
    )
    def test_watson_with_generated_data(self, a, b, seed, n):
        """Test Watson statistic with data generated from Uniform distribution."""
        np.random.seed(seed)
        data = np.random.uniform(a, b, n)

        stat = WatsonUniformGofStatistic(a=a, b=b)
        statistic_value = stat.execute_statistic(data)

        assert np.isfinite(statistic_value)

        assert statistic_value < 1


class TestKuiperUniformGofStatistic:
    """Tests for Kuiper test statistic."""

    def test_code(self):
        """Test that the Kuiper statistic returns correct code."""
        assert "KUIPER_UNIFORM_GOODNESS_OF_FIT" == KuiperUniformGofStatistic.code()

    @pytest.mark.parametrize(
        ("a", "b", "seed", "n"),
        [
            (0, 1, 42, 50),
            (0, 10, 123, 100),
        ],
    )
    def test_kuiper_with_generated_data(self, a, b, seed, n):
        """Test Kuiper statistic with data generated from Uniform distribution."""
        np.random.seed(seed)
        data = np.random.uniform(a, b, n)

        stat = KuiperUniformGofStatistic(a=a, b=b)
        statistic_value = stat.execute_statistic(data)

        assert statistic_value >= 0
        assert statistic_value < 1


class TestGreenwoodTestUniformGofStatistic:
    """Tests for Greenwood test statistic."""

    def test_code(self):
        """Test that the Greenwood statistic returns correct code."""
        assert "GREENWOOD_UNIFORM_GOODNESS_OF_FIT" == GreenwoodTestUniformGofStatistic.code()

    @pytest.mark.parametrize(
        ("a", "b", "seed", "n"),
        [
            (0, 1, 42, 50),
            (0, 10, 123, 100),
        ],
    )
    def test_greenwood_with_generated_data(self, a, b, seed, n):
        """Test Greenwood statistic with data generated from Uniform distribution."""
        np.random.seed(seed)
        data = np.random.uniform(a, b, n)

        stat = GreenwoodTestUniformGofStatistic(a=a, b=b)
        statistic_value = stat.execute_statistic(data)

        assert 0 <= statistic_value <= 1

        expected = 2 / (n + 1)
        assert abs(statistic_value - expected) < 0.2


class TestBickelRosenblattUniformGofStatistic:
    """Tests for Bickel-Rosenblatt test statistic."""

    def test_code(self):
        """Test that the Bickel-Rosenblatt statistic returns correct code."""
        assert "BICKEL_ROSENBLATT_UNIFORM_GOODNESS_OF_FIT" == BickelRosenblattUniformGofStatistic.code()

    @pytest.mark.parametrize(
        ("a", "b", "bandwidth", "seed", "n"),
        [
            (0, 1, 'auto', 42, 100),
            (0, 10, 0.1, 123, 200),
        ],
    )
    def test_bickel_rosenblatt_with_generated_data(self, a, b, bandwidth, seed, n):
        """Test Bickel-Rosenblatt statistic with data generated from Uniform distribution."""
        np.random.seed(seed)
        data = np.random.uniform(a, b, n)

        stat = BickelRosenblattUniformGofStatistic(a=a, b=b, bandwidth=bandwidth)
        statistic_value = stat.execute_statistic(data)

        assert statistic_value >= 0
        assert np.isfinite(statistic_value)

        assert statistic_value < 1


class TestZhangTestsUniformGofStatistic:
    """Tests for Zhang tests statistic."""

    def test_code(self):
        """Test that the Zhang statistic returns correct code."""
        assert "ZHANG_UNIFORM_GOODNESS_OF_FIT" == ZhangTestsUniformGofStatistic.code()

    @pytest.mark.parametrize(
        ("test_type", "a", "b", "seed", "n"),
        [
            ('A', 0, 1, 42, 50),
            ('C', 0, 10, 123, 100),
            ('K', 2, 5, 456, 75),
        ],
    )
    def test_zhang_with_generated_data(self, test_type, a, b, seed, n):
        """Test Zhang statistic with data generated from Uniform distribution."""
        np.random.seed(seed)
        data = np.random.uniform(a, b, n)

        stat = ZhangTestsUniformGofStatistic(a=a, b=b, test_type=test_type)
        statistic_value = stat.execute_statistic(data)

        assert np.isfinite(statistic_value)

    def test_zhang_invalid_test_type(self):
        """Test that Zhang statistic validates test_type parameter."""
        with pytest.raises(ValueError, match="test_type must be 'A', 'C', or 'K'"):
            ZhangTestsUniformGofStatistic(test_type='X')


class TestSteinUniformGofStatistic:
    """Tests for Stein test statistic."""

    def test_code(self):
        """Test that the Stein statistic returns correct code."""
        assert "STEIN_U_UNIFORM_GOODNESS_OF_FIT" == SteinUniformGofStatistic.code()

    @pytest.mark.parametrize(
        ("a", "b", "seed", "n"),
        [
            (0, 1, 42, 50),
            (0, 10, 123, 100),
        ],
    )
    def test_stein_with_generated_data(self, a, b, seed, n):
        """Test Stein statistic with data generated from Uniform distribution."""
        np.random.seed(seed)
        data = np.random.uniform(a, b, n)

        stat = SteinUniformGofStatistic(a=a, b=b)
        statistic_value = stat.execute_statistic(data)

        assert abs(statistic_value) < 0.5

    def test_stein_u_statistic_computation(self):
        """Test the U-statistic computation directly."""
        data = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        statistic = SteinUniformGofStatistic._compute_u_statistic(data)

        assert isinstance(statistic, (float, np.floating))
        assert np.isfinite(statistic)


class TestCensoredSteinUniformGofStatistic:
    """Tests for censored Stein test statistic."""

    def test_code(self):
        """Test that the censored Stein statistic returns correct code."""
        assert "CENSORED_STEIN_U_UNIFORM_GOODNESS_OF_FIT" == CensoredSteinUniformGofStatistic.code()

    def test_censored_stein_without_censoring(self):
        """Test censored Stein statistic without censoring."""
        np.random.seed(42)
        data = np.random.uniform(0, 1, 100)

        stat = CensoredSteinUniformGofStatistic()
        statistic_value = stat.execute_statistic(data)

        assert abs(statistic_value) < 0.5

    def test_censored_stein_with_censoring(self):
        """Test censored Stein statistic with censoring."""
        np.random.seed(42)
        n = 100
        data = np.random.uniform(0, 1, n)
        censoring = np.random.binomial(1, 0.2, n)

        stat = CensoredSteinUniformGofStatistic()
        statistic_value = stat.execute_statistic(data, censoring_indices=censoring)

        assert np.isfinite(statistic_value)

    def test_kaplan_meier_estimation(self):
        """Test Kaplan-Meier estimator."""
        times = np.array([1, 2, 3, 4, 5])
        delta = np.array([0, 1, 0, 1, 0])  # 1=censored, 0=uncensored

        km_func = CensoredSteinUniformGofStatistic._kaplan_meier(times, delta)

        assert km_func(0.5) == 1.0
        assert km_func(6.0) <= 1.0


class TestNeymanSmoothTestUniformGofStatistic:
    """Tests for Neyman smooth test statistic."""

    def test_code(self):
        """Test that the Neyman statistic returns correct code."""
        assert "NEYMAN_UNIFORM_GOODNESS_OF_FIT" == NeymanSmoothTestUniformGofStatistic.code()

    @pytest.mark.parametrize(
        ("k", "a", "b", "seed", "n"),
        [
            (2, 0, 1, 42, 50),
            (4, 0, 10, 123, 100),
            (3, 2, 5, 456, 75),
        ],
    )
    def test_neyman_with_generated_data(self, k, a, b, seed, n):
        """Test Neyman statistic with data generated from Uniform distribution."""
        np.random.seed(seed)
        data = np.random.uniform(a, b, n)

        stat = NeymanSmoothTestUniformGofStatistic(a=a, b=b, k=k)
        statistic_value = stat.execute_statistic(data)

        assert statistic_value >= 0
        assert np.isfinite(statistic_value)


class TestShermanUniformGofStatistic:
    """Tests for Sherman test statistic."""

    def test_code(self):
        """Test that the Sherman statistic returns correct code."""
        assert "SHERMAN_UNIFORM_GOODNESS_OF_FIT" == ShermanUniformGofStatistic.code()

    @pytest.mark.parametrize(
        ("a", "b", "seed", "n"),
        [
            (0, 1, 42, 50),
            (0, 10, 123, 100),
        ],
    )
    def test_sherman_with_generated_data(self, a, b, seed, n):
        """Test Sherman statistic with data generated from Uniform distribution."""
        np.random.seed(seed)
        data = np.random.uniform(a, b, n)

        stat = ShermanUniformGofStatistic(a=a, b=b)
        statistic_value = stat.execute_statistic(data)

        assert statistic_value >= 0
        assert np.isfinite(statistic_value)

        assert statistic_value < (b - a) / 2


class TestQuesenberryMillerUniformGofStatistic:
    """Tests for Quesenberry-Miller test statistic."""

    def test_code(self):
        """Test that the Quesenberry-Miller statistic returns correct code."""
        assert "QUESENBERRY_MILLER_UNIFORM_GOODNESS_OF_FIT" == QuesenberryMillerUniformGofStatistic.code()

    @pytest.mark.parametrize(
        ("a", "b", "seed", "n"),
        [
            (0, 1, 42, 50),
            (0, 10, 123, 100),
        ],
    )
    def test_quesenberry_miller_with_generated_data(self, a, b, seed, n):
        """Test Quesenberry-Miller statistic with data generated from Uniform distribution."""
        np.random.seed(seed)
        data = np.random.uniform(a, b, n)

        stat = QuesenberryMillerUniformGofStatistic(a=a, b=b)
        statistic_value = stat.execute_statistic(data)

        assert statistic_value >= 0
        assert np.isfinite(statistic_value)


class TestUniformIntegration:
    """Integration tests for Uniform distribution statistics."""

    def test_all_statistics_run(self):
        """Test that all statistics can be computed without errors."""
        np.random.seed(42)
        data = np.random.uniform(0, 1, 100)

        statistics = [
            KolmogorovSmirnovUniformGofStatistic(),
            AndersonDarlingUniformGofStatistic(),
            CrammerVonMisesUniformGofStatistic(),
            LillieforsTestUniformGofStatistic(),
            Chi2PearsonUniformGofStatistic(),
            WatsonUniformGofStatistic(),
            KuiperUniformGofStatistic(),
            GreenwoodTestUniformGofStatistic(),
            BickelRosenblattUniformGofStatistic(),
            ZhangTestsUniformGofStatistic(test_type='A'),
            ZhangTestsUniformGofStatistic(test_type='C'),
            ZhangTestsUniformGofStatistic(test_type='K'),
            SteinUniformGofStatistic(),
            CensoredSteinUniformGofStatistic(),
            NeymanSmoothTestUniformGofStatistic(k=4),
            ShermanUniformGofStatistic(),
            QuesenberryMillerUniformGofStatistic(),
        ]

        for stat in statistics:
            try:
                value = stat.execute_statistic(data)
                assert np.isfinite(value), f"Statistic {stat.code()} returned non-finite value"
            except Exception as e:
                pytest.fail(f"Statistic {stat.code()} raised an exception: {e}")

    def test_different_bounds(self):
        """Test that statistics work with different distribution bounds."""
        np.random.seed(42)
        data = np.random.uniform(2, 5, 100)

        statistics = [
            KolmogorovSmirnovUniformGofStatistic(a=2, b=5),
            AndersonDarlingUniformGofStatistic(a=2, b=5),
            Chi2PearsonUniformGofStatistic(a=2, b=5),
            SteinUniformGofStatistic(a=2, b=5),
        ]

        for stat in statistics:
            value = stat.execute_statistic(data)
            assert np.isfinite(value), f"Statistic {stat.code()} failed with different bounds"

    def test_edge_cases(self):
        """Test edge cases for the statistics."""
        small_data = np.random.uniform(0, 1, 5)

        stats_to_test = [
            KolmogorovSmirnovUniformGofStatistic(),
            Chi2PearsonUniformGofStatistic(bins=2),
            SteinUniformGofStatistic(),
        ]

        for stat in stats_to_test:
            value = stat.execute_statistic(small_data)
            assert np.isfinite(value), f"Statistic {stat.code()} failed with small sample"

        perfect_data = np.linspace(0.05, 0.95, 20)

        for stat in stats_to_test:
            value = stat.execute_statistic(perfect_data)
            assert np.isfinite(value), f"Statistic {stat.code()} failed with perfect uniform data"

    def test_consistency_across_runs(self):
        """Test that statistics give consistent results across multiple runs."""
        np.random.seed(42)
        data = np.random.uniform(0, 1, 100)

        stat = KolmogorovSmirnovUniformGofStatistic()
        value1 = stat.execute_statistic(data)
        value2 = stat.execute_statistic(data)

        assert value1 == value2

        data = np.random.exponential(0.5, 100)
        data = data / max(data)

        value1 = stat.execute_statistic(data)
        value2 = stat.execute_statistic(data)

        assert value1 == value2


@pytest.mark.slow
class TestUniformBenchmark:
    """Benchmark tests for Uniform distribution statistics."""

    @pytest.mark.parametrize("n", [100, 1000, 10000])
    def test_large_samples(self, n):
        """Test performance with large samples."""
        np.random.seed(42)
        data = np.random.uniform(0, 1, n)

        stats = [
            KolmogorovSmirnovUniformGofStatistic(),
            Chi2PearsonUniformGofStatistic(bins='sqrt'),
            SteinUniformGofStatistic(),
            GreenwoodTestUniformGofStatistic(),
        ]

        for stat in stats:
            import time
            start = time.time()
            value = stat.execute_statistic(data)
            elapsed = time.time() - start

            assert np.isfinite(value)
            assert elapsed < 20.0, f"Statistic {stat.code()} took too long: {elapsed:.2f}s"