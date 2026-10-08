"""Formula, calibration and public scope of the two published Lilliefors tests."""

import numpy as np
import pytest
from scipy import stats

from pysatl_criterion.distribution.distributions import (
    ExponentialDistributionDescriptor as Exponential,
)
from pysatl_criterion.distribution.distributions import NormalDistributionDescriptor as Normal
from pysatl_criterion.statistics.alternative import RightAlternative
from pysatl_criterion.statistics.goodness_of_fit import (
    LillieforsExponentialityGofStatistic,
    LillieforsNormalityGofStatistic,
    common,
    log_normal,
)
from pysatl_criterion.utils.generator import get_hypothesis_generator


@pytest.mark.parametrize("sample", [[0.2, 0.5, 1.0, 2.0], [0, 0, 0.3, 5], [2, 2, 2]])
def test_exponential_formula_and_scale_invariance(sample):
    statistic = LillieforsExponentialityGofStatistic(Exponential.DEFAULT.parse({}))
    x = np.array(sample, dtype=float)
    reference = stats.kstest(x, stats.expon(scale=x.mean()).cdf).statistic
    for scale in [1e-200, 1, 1e200]:
        assert statistic.execute_statistic(x[::-1] * scale) == pytest.approx(reference)
    assert isinstance(statistic.alternative(), RightAlternative)
    assert statistic.hypothesis().parameters() == {}
    assert get_hypothesis_generator(statistic).parameters() == {"lam": 1}
    with pytest.raises(ValueError, match="Unsupported"):
        LillieforsExponentialityGofStatistic(Exponential.DEFAULT.parse({"lam": 1}))


def test_normal_uses_unbiased_variance_and_refits():
    statistic = LillieforsNormalityGofStatistic(Normal.DEFAULT.parse({}))
    for x in [np.array([-2, -1, 0, 0.5, 3.0]), np.array([0.1, 0.2, 0.6, 4])]:
        expected = stats.kstest(x, stats.norm(x.mean(), x.std(ddof=1)).cdf).statistic
        assert statistic.execute_statistic(x) == pytest.approx(expected)
        assert statistic.execute_statistic(3 * x[::-1] + 7) == pytest.approx(expected)
        assert statistic.execute_statistic(x) != pytest.approx(
            stats.kstest(x, stats.norm(x.mean(), x.std(ddof=0)).cdf).statistic
        )


@pytest.mark.parametrize(
    "sample",
    [
        [],
        [1],
        [0, 0],
        [-1, 2],
        [[1, 2]],
        [np.nan, 1],
        [np.inf, 1],
        [1j, 2],
        np.ma.array([1, 2], mask=[True, False]),
    ],
)
def test_invalid_exponential_samples(sample):
    statistic = LillieforsExponentialityGofStatistic(Exponential.DEFAULT.parse({}))
    with pytest.raises(ValueError):
        statistic.execute_statistic(sample)


def test_lilliefors_is_exported_only_for_normal_and_exponential():
    from pysatl_criterion.statistics import goodness_of_fit

    exported = {name for name in goodness_of_fit.__all__ if name.startswith("Lilliefors")}
    assert exported == {"LillieforsNormalityGofStatistic", "LillieforsExponentialityGofStatistic"}
    assert not hasattr(common, "LillieforsTest")
    assert not hasattr(log_normal, "LillieforsLogNormalGofStatistic")
