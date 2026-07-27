from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import semicircular

from pysatl_criterion.distribution.distribution_type import DistributionType
from pysatl_criterion.estimation.base import EstimationMethod
from pysatl_criterion.estimation.method_moments.estimators import (
    BetaMmEstimator,
    Chi2MmEstimator,
    ExponentialMmEstimator,
    FisherMmEstimator,
    GammaMmEstimator,
    LaplaceMmEstimator,
    LogNormalMmEstimator,
    NormalMmEstimator,
    ParetoMmEstimator,
    RayleighMmEstimator,
    StudentMmEstimator,
    UniformMmEstimator,
    WeibullMmEstimator,
    WignerMmEstimator,
)


RNG = np.random.default_rng(42)

class TestNormalMmEstimator:
    def test_metadata(self):
        assert NormalMmEstimator.distribution_type() == DistributionType.NORMAL
        assert NormalMmEstimator.method() == EstimationMethod.MM

    def test_estimate_success(self):
        data = RNG.normal(loc=5.0, scale=2.0, size=10_000)
        res = NormalMmEstimator().estimate(data)

        assert res["mean"] == pytest.approx(5.0, abs=0.1)
        assert res["var"] == pytest.approx(4.0, abs=0.2)

    def test_negative_data(self):
        data = [-10.0, -5.0, -1.0, 0.0]
        res = NormalMmEstimator().estimate(data)
        assert res["mean"] == pytest.approx(-4.0)

    def test_all_values_equal_to_mathExpect(self):
        data = [7.0, 7.0, 7.0, 7.0]
        res = NormalMmEstimator().estimate(data)
        assert res["mean"] == 7.0
        assert res["var"] == 0.0

    def test_big_and_small_distribution(self):
        res_big = NormalMmEstimator().estimate(RNG.normal(1e6, 1e3, 1000))
        assert res_big["mean"] == pytest.approx(1e6, rel=1e-2)

        res_small = NormalMmEstimator().estimate(RNG.normal(1e-6, 1e-7, 1000))
        assert res_small["mean"] == pytest.approx(1e-6, rel=1e-2)

    def test_a_lot_of_data(self):
        data = RNG.normal(loc=12.5, scale=np.sqrt(3.2), size=200_000)
        res = NormalMmEstimator().estimate(data)
        assert res["mean"] == pytest.approx(12.5, rel=1e-2)
        assert res["var"] == pytest.approx(3.2, rel=1e-2)

class TestUniformMmEstimator:
    def test_metadata(self):
        assert UniformMmEstimator.distribution_type() == DistributionType.UNIFORM
        assert UniformMmEstimator.method() == EstimationMethod.MM

    def test_estimate_success(self):
        data = RNG.uniform(low=2.0, high=8.0, size=10_000)
        res = UniformMmEstimator().estimate(data)

        assert res["a"] == pytest.approx(2.0, abs=0.1)
        assert res["b"] == pytest.approx(8.0, abs=0.1)

    def test_negative_data(self):
        data = RNG.uniform(low=-10.0, high=-2.0, size=2000)
        res = UniformMmEstimator().estimate(data)
        assert res["a"] == pytest.approx(-10.0, abs=0.2)
        assert res["b"] == pytest.approx(-2.0, abs=0.2)

    def test_all_values_equal_to_mathExpect(self):
        data = [3.0, 3.0, 3.0, 3.0]
        res = UniformMmEstimator().estimate(data)
        assert res["a"] == pytest.approx(3.0)
        assert res["b"] == pytest.approx(3.0)

    def test_a_lot_of_data(self):
        data = RNG.uniform(low=0.0, high=100.0, size=200_000)
        res = UniformMmEstimator().estimate(data)
        assert res["a"] == pytest.approx(0.0, abs=0.5)
        assert res["b"] == pytest.approx(100.0, abs=0.5)

    def test_invalid_sample_size(self):
        with pytest.raises(ValueError, match="At least 2 elements"):
            UniformMmEstimator().estimate([5.0])

class TestLogNormalMmEstimator:
    def test_metadata(self):
        assert LogNormalMmEstimator.distribution_type() == DistributionType.LOG_NORMAL
        assert LogNormalMmEstimator.method() == EstimationMethod.MM

    def test_estimate_success(self):
        data = RNG.lognormal(mean=0.5, sigma=0.5, size=20_000)
        res = LogNormalMmEstimator().estimate(data)

        assert res["mean"] == pytest.approx(0.5, abs=0.05)
        assert res["var"] == pytest.approx(0.25, abs=0.05)

    def test_negative_data(self):
        with pytest.raises(ValueError, match="strictly positive"):
            LogNormalMmEstimator().estimate([1.0, 2.0, -0.5])

    def test_all_values_equal_to_mathExpect(self):
        res = LogNormalMmEstimator().estimate([2.0, 2.0, 2.0])
        assert res["mean"] == pytest.approx(np.log(2.0))
        assert res["var"] == pytest.approx(0.0)

    def test_a_lot_of_data(self):
        data = RNG.lognormal(mean=1.2, sigma=0.4, size=100_000)
        res = LogNormalMmEstimator().estimate(data)
        assert res["mean"] == pytest.approx(1.2, rel=2e-2)
        assert res["var"] == pytest.approx(0.16, rel=2e-2)


class TestExponentialMmEstimator:
    def test_metadata(self):
        assert ExponentialMmEstimator.distribution_type() == DistributionType.EXPONENTIAL
        assert ExponentialMmEstimator.method() == EstimationMethod.MM

    def test_estimate_success(self):
        data = RNG.exponential(scale=0.5, size=20_000)
        res = ExponentialMmEstimator().estimate(data)

        assert res["rate"] == pytest.approx(2.0, rel=5e-2)

    def test_negative_data(self):
        with pytest.raises(ValueError, match="non-negative"):
            ExponentialMmEstimator().estimate([1.0, -2.0, 3.0])

    def test_all_values_equal_to_mathExpect(self):
        res = ExponentialMmEstimator().estimate([4.0, 4.0, 4.0])
        assert res["rate"] == pytest.approx(0.25)

class TestWeibullMmEstimator:
    def test_metadata(self):
        assert WeibullMmEstimator.distribution_type() == DistributionType.WEIBULL
        assert WeibullMmEstimator.method() == EstimationMethod.MM

    def test_estimate_success(self):
        shape, scale = 2.0, 3.0
        data = scale * RNG.weibull(a=shape, size=20_000)
        res = WeibullMmEstimator().estimate(data)

        assert res["shape"] == pytest.approx(shape, rel=5e-2)
        assert res["scale"] == pytest.approx(scale, rel=5e-2)

    def test_negative_data(self):
        with pytest.raises(ValueError, match="strictly positive"):
            WeibullMmEstimator().estimate([1.0, -1.0])

class TestGammaMmEstimator:
    def test_metadata(self):
        assert GammaMmEstimator.distribution_type() == DistributionType.GAMMA
        assert GammaMmEstimator.method() == EstimationMethod.MM

    def test_estimate_success(self):
        shape, scale = 3.0, 2.0
        data = RNG.gamma(shape=shape, scale=scale, size=20_000)
        res = GammaMmEstimator().estimate(data)

        assert res["shape"] == pytest.approx(shape, rel=5e-2)
        assert res["scale"] == pytest.approx(scale, rel=5e-2)

    def test_all_values_equal_to_mathExpect(self):
        with pytest.raises(ValueError, match="Variance"):
            GammaMmEstimator().estimate([4.0, 4.0, 4.0])

class TestBetaMmEstimator:
    def test_metadata(self):
        assert BetaMmEstimator.distribution_type() == DistributionType.BETA
        assert BetaMmEstimator.method() == EstimationMethod.MM

    def test_estimate_success(self):
        alpha, beta = 2.0, 5.0
        data = RNG.beta(a=alpha, b=beta, size=20_000)
        res = BetaMmEstimator().estimate(data)

        assert res["alpha"] == pytest.approx(alpha, rel=5e-2)
        assert res["beta"] == pytest.approx(beta, rel=5e-2)

    def test_out_of_bounds_data(self):
        with pytest.raises(ValueError, match=r"strictly in \(0, 1\)"):
            BetaMmEstimator().estimate([-0.1, 0.5, 1.2])

class TestChi2MmEstimator:
    def test_metadata(self):
        assert Chi2MmEstimator.distribution_type() == DistributionType.CHI_2
        assert Chi2MmEstimator.method() == EstimationMethod.MM

    def test_estimate_success(self):
        df = 4.0
        data = RNG.chisquare(df=df, size=20_000)
        res = Chi2MmEstimator().estimate(data)

        assert res["df"] == pytest.approx(df, rel=5e-2)

class TestStudentMmEstimator:
    def test_metadata(self):
        assert StudentMmEstimator.distribution_type() == DistributionType.STUDENT
        assert StudentMmEstimator.method() == EstimationMethod.MM

    def test_estimate_success(self):
        df = 6.0
        data = RNG.standard_t(df=df, size=100_000)
        res = StudentMmEstimator().estimate(data)

        assert res["df"] == pytest.approx(df, rel=0.15)


class TestFisherMmEstimator:
    def test_metadata(self):
        assert FisherMmEstimator.distribution_type() == DistributionType.FISHER
        assert FisherMmEstimator.method() == EstimationMethod.MM

    def test_estimate_success(self):
        df1, df2 = 10.0, 8.0
        data = RNG.f(dfnum=df1, dfden=df2, size=100_000)
        res = FisherMmEstimator().estimate(data)

        assert res["df1"] == pytest.approx(df1, rel=0.15)
        assert res["df2"] == pytest.approx(df2, rel=0.15)


class TestRayleighMmEstimator:
    def test_metadata(self):
        assert RayleighMmEstimator.distribution_type() == DistributionType.RAYLEIGH
        assert RayleighMmEstimator.method() == EstimationMethod.MM

    def test_estimate_success(self):
        scale = 3.0
        data = RNG.rayleigh(scale=scale, size=20_000)
        res = RayleighMmEstimator().estimate(data)

        assert res["scale"] == pytest.approx(scale, rel=5e-2)

class TestWignerMmEstimator:
    def test_metadata(self):
        assert WignerMmEstimator.distribution_type() == DistributionType.WIGNER
        assert WignerMmEstimator.method() == EstimationMethod.MM

    def test_estimate_success(self):
        radius = 4.0
        data = semicircular.rvs(scale=radius, size=20_000, random_state=RNG)
        res = WignerMmEstimator().estimate(data)

        assert res["radius"] == pytest.approx(radius, rel=5e-2)

class TestParetoMmEstimator:
    def test_metadata(self):
        assert ParetoMmEstimator.distribution_type() == DistributionType.PARETO
        assert ParetoMmEstimator.method() == EstimationMethod.MM

    def test_estimate_success(self):
        alpha, scale = 3.0, 5.0
        data = scale * (RNG.pareto(a=alpha, size=100_000) + 1.0)
        res = ParetoMmEstimator().estimate(data)

        assert res["shape"] == pytest.approx(alpha, rel=0.1)
        assert res["scale"] == pytest.approx(scale, rel=0.1)

class TestLaplaceMmEstimator:
    def test_metadata(self):
        assert LaplaceMmEstimator.distribution_type() == DistributionType.LAPLACE
        assert LaplaceMmEstimator.method() == EstimationMethod.MM

    def test_estimate_success(self):
        loc, scale = 2.0, 1.5
        data = RNG.laplace(loc=loc, scale=scale, size=20_000)
        res = LaplaceMmEstimator().estimate(data)

        assert res["t"] == pytest.approx(loc, abs=0.1)
        assert res["s"] == pytest.approx(scale, rel=5e-2)
