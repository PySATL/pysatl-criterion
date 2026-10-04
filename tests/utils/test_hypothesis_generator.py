import math

import numpy as np
import pytest
from scipy import stats

from pysatl_criterion.hypothesis_testing.limit_distribution.base import (
    MonteCarloLimitDistributionResolver,
)
from pysatl_criterion.statistics.goodness_of_fit.beta import KolmogorovSmirnovBetaGofStatistic
from pysatl_criterion.statistics.goodness_of_fit.exponent import (
    KolmogorovSmirnovExponentialityGofStatistic,
)
from pysatl_criterion.statistics.goodness_of_fit.gamma import KolmogorovSmirnovGammaGofStatistic
from pysatl_criterion.statistics.goodness_of_fit.laplace import KolmogorovSmirnovLaplaceGofStatistic
from pysatl_criterion.statistics.goodness_of_fit.log_normal import (
    KolmogorovSmirnovLogNormalGofStatistic,
)
from pysatl_criterion.statistics.goodness_of_fit.normal import (
    KolmogorovSmirnovNormalityGofStatistic,
)
from pysatl_criterion.statistics.goodness_of_fit.student import KolmogorovSmirnovStudentGofStatistic
from pysatl_criterion.statistics.goodness_of_fit.uniform import KolmogorovSmirnovUniformGofStatistic
from pysatl_criterion.statistics.goodness_of_fit.weibull import KolmogorovSmirnovWeibullGofStatistic
from pysatl_criterion.utils.generator import get_hypothesis_generator


@pytest.mark.parametrize(
    ("statistic", "parameters", "reference"),
    [
        (
            KolmogorovSmirnovBetaGofStatistic(alpha=2, beta=5),
            {"a": 2, "b": 5},
            stats.beta(a=2, b=5),
        ),
        (
            KolmogorovSmirnovGammaGofStatistic(alpha=3, beta=4),
            {"alfa": 3, "beta": 4},
            stats.gamma(a=3, scale=0.25),
        ),
        (
            KolmogorovSmirnovLogNormalGofStatistic(s=0.7, scale=5),
            {"s": 0.7, "mu": math.log(5)},
            stats.lognorm(s=0.7, scale=5),
        ),
        (
            KolmogorovSmirnovStudentGofStatistic(df=7, loc=3, scale=2),
            {"df": 7, "loc": 3, "scale": 2},
            stats.t(df=7, loc=3, scale=2),
        ),
        (
            KolmogorovSmirnovNormalityGofStatistic(mean=3, var=4),
            {"mean": 3, "var": 4},
            stats.norm(loc=3, scale=2),
        ),
        (KolmogorovSmirnovExponentialityGofStatistic(lam=4), {"lam": 4}, stats.expon(scale=0.25)),
        (
            KolmogorovSmirnovUniformGofStatistic(a=-2, b=3),
            {"a": -2, "b": 3},
            stats.uniform(loc=-2, scale=5),
        ),
        (
            KolmogorovSmirnovLaplaceGofStatistic(t=3, s=2),
            {"t": 3, "s": 2},
            stats.laplace(loc=3, scale=2),
        ),
        (
            KolmogorovSmirnovWeibullGofStatistic(a=2, k=3),
            {"a": 2, "k": 3},
            stats.exponweib(a=2, c=3),
        ),
    ],
)
def test_hypothesis_sampling_matches_reference(statistic, parameters, reference):
    hypothesis_before = statistic.hypothesis().parameters()
    generator = get_hypothesis_generator(statistic)

    assert generator.parameters() == parameters
    actual = generator.generate(40, random_state=np.random.default_rng(123))
    expected = reference.rvs(size=40, random_state=np.random.default_rng(123))
    np.testing.assert_allclose(actual, expected, rtol=1e-14)
    assert statistic.hypothesis().parameters() == hypothesis_before


def test_monte_carlo_passes_beta_hypothesis_to_sampler(mocker):
    samples = np.array([0.1, 0.2, 0.4])
    sampler = mocker.patch(
        "pysatl_criterion.generator.generators.generate_beta", return_value=samples
    )
    statistic = KolmogorovSmirnovBetaGofStatistic(alpha=2, beta=5)
    result = MonteCarloLimitDistributionResolver(2).resolve(statistic, sample_size=3)

    assert result == [statistic.execute_statistic(samples)] * 2
    assert sampler.call_args_list == [mocker.call(size=3, a=2, b=5, random_state=None)] * 2
