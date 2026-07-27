import numpy as np
import pytest
from scipy import stats

from pysatl_criterion.distribution.distribution_type import DistributionType
from pysatl_criterion.estimation.base import EstimationMethod
from pysatl_criterion.estimation.quantile_matching.estimators import (
    CauchyQmeEstimator,
    ExponentialQmeEstimator,
    LogNormalQmeEstimator,
    NormalQmeEstimator,
    ParetoQmeEstimator,
    UniformQmeEstimator,
    WeibullQmeEstimator,
)


RNG = np.random.default_rng(42)
LARGE_SAMPLE_SIZE = 200_000

class TestNormalQmeEstimator:
    def test_metadata(self):
        estimator = NormalQmeEstimator()
        assert estimator.distribution_type() == DistributionType.NORMAL
        assert estimator.method() == EstimationMethod.QME

    @pytest.mark.parametrize(
        "true_mean, true_std",
        [
            (0.0, 1.0),
            (10.5, 2.5),
            (-15.0, 0.5),
        ],
    )
    def test_estimation_accuracy(self, true_mean: float, true_std: float):
        sample = RNG.normal(loc=true_mean, scale=true_std, size=LARGE_SAMPLE_SIZE)
        result = NormalQmeEstimator().estimate(sample)

        assert result["mean"] == pytest.approx(true_mean, abs=0.02)
        assert result["var"] == pytest.approx(true_std**2, abs=0.08)

    def test_input_validation(self):
        estimator = NormalQmeEstimator()
        with pytest.raises(ValueError, match="At least 2 elements are required"):
            estimator.estimate([1.0])
        with pytest.raises(ValueError, match="At least 2 elements are required"):
            estimator.estimate([])


class TestUniformQmeEstimator:
    def test_metadata(self):
        estimator = UniformQmeEstimator()
        assert estimator.distribution_type() == DistributionType.UNIFORM
        assert estimator.method() == EstimationMethod.QME

    @pytest.mark.parametrize(
        "true_a, true_b",
        [
            (0.0, 10.0),
            (-5.0, 5.0),
            (100.0, 101.0),
        ],
    )
    def test_estimation_accuracy(self, true_a: float, true_b: float):
        sample = RNG.uniform(low=true_a, high=true_b, size=LARGE_SAMPLE_SIZE)
        result = UniformQmeEstimator().estimate(sample)

        assert result["a"] == pytest.approx(true_a, abs=0.05)
        assert result["b"] == pytest.approx(true_b, abs=0.05)

    def test_support_boundary_guarantee(self):
        sample = [1.2, 2.5, 3.1, 4.8, 9.0]
        result = UniformQmeEstimator().estimate(sample)

        assert result["a"] <= min(sample)
        assert result["b"] >= max(sample)

    def test_equal_quantiles_raises_error(self):
        constant_sample = [5.0, 5.0, 5.0, 5.0]
        with pytest.raises(ValueError, match="Quantiles are equal"):
            UniformQmeEstimator().estimate(constant_sample)

    def test_input_validation(self):
        with pytest.raises(ValueError, match="At least 2 elements are required"):
            UniformQmeEstimator().estimate([1.0])

class TestLogNormalQmeEstimator:
    def test_metadata(self):
        estimator = LogNormalQmeEstimator()
        assert estimator.distribution_type() == DistributionType.LOG_NORMAL
        assert estimator.method() == EstimationMethod.QME

    @pytest.mark.parametrize(
        "true_mu, true_sigma",
        [
            (0.0, 0.25),
            (1.5, 0.5),
            (-0.5, 0.1),
        ],
    )
    def test_estimation_accuracy(self, true_mu: float, true_sigma: float):
        sample = RNG.lognormal(mean=true_mu, sigma=true_sigma, size=LARGE_SAMPLE_SIZE)
        result = LogNormalQmeEstimator().estimate(sample)

        assert result["mean"] == pytest.approx(true_mu, abs=0.02)
        assert result["var"] == pytest.approx(true_sigma**2, abs=0.03)

    def test_non_positive_data_raises_error(self):
        estimator = LogNormalQmeEstimator()
        with pytest.raises(ValueError, match="strictly positive"):
            estimator.estimate([1.0, 2.0, 0.0])
        with pytest.raises(ValueError, match="strictly positive"):
            estimator.estimate([1.0, 2.0, -1.5])

    def test_input_validation(self):
        with pytest.raises(ValueError, match="At least 2 elements are required"):
            LogNormalQmeEstimator().estimate([2.0])


class TestExponentialQmeEstimator:
    def test_metadata(self):
        estimator = ExponentialQmeEstimator()
        assert estimator.distribution_type() == DistributionType.EXPONENTIAL
        assert estimator.method() == EstimationMethod.QME

    @pytest.mark.parametrize("true_rate", [0.5, 1.0, 3.5])
    def test_estimation_accuracy(self, true_rate: float):
        sample = RNG.exponential(scale=1.0 / true_rate, size=LARGE_SAMPLE_SIZE)
        result = ExponentialQmeEstimator().estimate(sample)

        assert result["rate"] == pytest.approx(true_rate, abs=0.02)

    def test_negative_data_raises_error(self):
        with pytest.raises(ValueError, match="non-negative"):
            ExponentialQmeEstimator().estimate([1.0, -0.1, 2.0])

    def test_zero_median_raises_error(self):
        with pytest.raises(ValueError, match="strictly positive"):
            ExponentialQmeEstimator().estimate([0.0, 0.0, 0.0])

    def test_input_validation(self):
        with pytest.raises(ValueError, match="Sample size must be at least 1"):
            ExponentialQmeEstimator().estimate([])

class TestWeibullQmeEstimator:
    def test_metadata(self):
        estimator = WeibullQmeEstimator()
        assert estimator.distribution_type() == DistributionType.WEIBULL
        assert estimator.method() == EstimationMethod.QME

    @pytest.mark.parametrize(
        "true_k, true_lambda",
        [
            (1.5, 2.0),
            (2.5, 0.8),
            (0.8, 5.0),
        ],
    )
    def test_estimation_accuracy(self, true_k: float, true_lambda: float):
        sample = RNG.weibull(a=true_k, size=LARGE_SAMPLE_SIZE) * true_lambda
        result = WeibullQmeEstimator().estimate(sample)

        assert result["shape"] == pytest.approx(true_k, abs=0.03)
        assert result["scale"] == pytest.approx(true_lambda, abs=0.03)

    def test_negative_data_raises_error(self):
        with pytest.raises(ValueError, match="non-negative"):
            WeibullQmeEstimator().estimate([1.0, -2.0, 3.0])

    def test_invalid_quantiles_raises_error(self):
        with pytest.raises(ValueError, match="Invalid quantiles"):
            WeibullQmeEstimator().estimate([0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="Invalid quantiles"):
            WeibullQmeEstimator().estimate([2.0, 2.0, 2.0])

    def test_input_validation(self):
        with pytest.raises(ValueError, match="At least 2 elements are required"):
            WeibullQmeEstimator().estimate([1.0])

class TestCauchyQmeEstimator:
    def test_metadata(self):
        estimator = CauchyQmeEstimator()
        assert estimator.distribution_type() == DistributionType.CAUCHY
        assert estimator.method() == EstimationMethod.QME

    @pytest.mark.parametrize(
        "true_loc, true_scale",
        [
            (0.0, 1.0),
            (-10.0, 4.0),
            (25.0, 0.5),
        ],
    )
    def test_estimation_accuracy(self, true_loc: float, true_scale: float):
        sample = stats.cauchy.rvs(
            loc=true_loc, scale=true_scale,
            size=LARGE_SAMPLE_SIZE, random_state=42
        )
        result = CauchyQmeEstimator().estimate(sample)

        assert result["loc"] == pytest.approx(true_loc, abs=0.03)
        assert result["scale"] == pytest.approx(true_scale, abs=0.03)

    def test_zero_gamma_raises_error(self):
        constant_sample = [3.0, 3.0, 3.0, 3.0]
        with pytest.raises(ValueError, match="strictly positive"):
            CauchyQmeEstimator().estimate(constant_sample)

    def test_input_validation(self):
        with pytest.raises(ValueError, match="At least 2 elements are required"):
            CauchyQmeEstimator().estimate([1.0])

class TestParetoQmeEstimator:
    def test_metadata(self):
        estimator = ParetoQmeEstimator()
        assert estimator.distribution_type() == DistributionType.PARETO
        assert estimator.method() == EstimationMethod.QME

    @pytest.mark.parametrize(
        "true_alpha, true_xm",
        [
            (2.0, 1.0),
            (3.5, 5.0),
            (1.2, 10.0),
        ],
    )
    def test_estimation_accuracy(self, true_alpha: float, true_xm: float):
        sample = (RNG.pareto(a=true_alpha, size=LARGE_SAMPLE_SIZE) + 1.0) * true_xm
        result = ParetoQmeEstimator().estimate(sample)

        assert result["shape"] == pytest.approx(true_alpha, abs=0.05)
        assert result["scale"] == pytest.approx(true_xm, abs=0.05)

    def test_support_boundary_guarantee(self):
        sample = [2.5, 3.0, 4.1, 5.0, 10.0]
        result = ParetoQmeEstimator().estimate(sample)

        assert result["scale"] <= min(sample)

    def test_non_positive_data_raises_error(self):
        with pytest.raises(ValueError, match="strictly positive"):
            ParetoQmeEstimator().estimate([1.0, 2.0, 0.0])
        with pytest.raises(ValueError, match="strictly positive"):
            ParetoQmeEstimator().estimate([-1.0, 2.0, 3.0])

    def test_equal_quantiles_raises_error(self):
        constant_sample = [2.0, 2.0, 2.0, 2.0]
        with pytest.raises(ValueError, match="Quantiles are equal"):
            ParetoQmeEstimator().estimate(constant_sample)

    def test_input_validation(self):
        with pytest.raises(ValueError, match="At least 2 elements are required"):
            ParetoQmeEstimator().estimate([1.0])
