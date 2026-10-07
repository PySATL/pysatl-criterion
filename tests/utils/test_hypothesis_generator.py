import math

import numpy as np
import pytest
from scipy import stats

from pysatl_criterion.distribution.distributions import BetaDistributionDescriptor as Beta
from pysatl_criterion.distribution.distributions import (
    ExponentialDistributionDescriptor,
    GammaDistributionDescriptor,
    LaplaceDistributionDescriptor,
    LogNormalDistributionDescriptor,
    NormalDistributionDescriptor,
    StudentDistributionDescriptor,
    UniformDistributionDescriptor,
)
from pysatl_criterion.distribution.distributions import (
    ExponentiatedWeibullDistributionDescriptor as ExponentiatedWeibull,
)
from pysatl_criterion.hypothesis_testing.limit_distribution.base import (
    MonteCarloLimitDistributionResolver,
)
from pysatl_criterion.statistics.goodness_of_fit.beta import KolmogorovSmirnovBetaGofStatistic
from pysatl_criterion.statistics.goodness_of_fit.exponent import (
    KolmogorovSmirnovExponentialityGofStatistic,
)
from pysatl_criterion.statistics.goodness_of_fit.exponentiated_weibull import (
    KolmogorovSmirnovExponentiatedWeibullGofStatistic,
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
from pysatl_criterion.utils.generator import get_hypothesis_generator


@pytest.mark.parametrize(
    ("statistic", "parameters", "reference"),
    [
        (
            KolmogorovSmirnovBetaGofStatistic(Beta.DEFAULT.parse({"a": 2, "b": 5})),
            {"a": 2, "b": 5},
            stats.beta(a=2, b=5),
        ),
        (
            KolmogorovSmirnovGammaGofStatistic(
                GammaDistributionDescriptor.DEFAULT.parse({"alfa": 3, "beta": 4})
            ),
            {"alfa": 3, "beta": 4},
            stats.gamma(a=3, scale=0.25),
        ),
        (
            KolmogorovSmirnovLogNormalGofStatistic(
                LogNormalDistributionDescriptor.SHAPE_SCALE.parse({"s": 0.7, "scale": 5})
            ),
            {"s": 0.7, "mu": math.log(5)},
            stats.lognorm(s=0.7, scale=5),
        ),
        (
            KolmogorovSmirnovStudentGofStatistic(
                StudentDistributionDescriptor.DEFAULT.parse({"df": 7, "loc": 3, "scale": 2})
            ),
            {"df": 7, "loc": 3, "scale": 2},
            stats.t(df=7, loc=3, scale=2),
        ),
        (
            KolmogorovSmirnovNormalityGofStatistic(
                NormalDistributionDescriptor.DEFAULT.parse({"mean": 3, "var": 4})
            ),
            {"mean": 3, "var": 4},
            stats.norm(loc=3, scale=2),
        ),
        (
            KolmogorovSmirnovExponentialityGofStatistic(
                ExponentialDistributionDescriptor.DEFAULT.parse({"lam": 4})
            ),
            {"lam": 4},
            stats.expon(scale=0.25),
        ),
        (
            KolmogorovSmirnovUniformGofStatistic(
                UniformDistributionDescriptor.DEFAULT.parse({"a": -2, "b": 3})
            ),
            {"a": -2, "b": 3},
            stats.uniform(loc=-2, scale=5),
        ),
        (
            KolmogorovSmirnovLaplaceGofStatistic(
                LaplaceDistributionDescriptor.DEFAULT.parse({"t": 3, "s": 2})
            ),
            {"t": 3, "s": 2},
            stats.laplace(loc=3, scale=2),
        ),
        (
            KolmogorovSmirnovExponentiatedWeibullGofStatistic(
                ExponentiatedWeibull.DEFAULT.parse({"exponent": 2, "shape": 3, "scale": 1})
            ),
            {"exponent": 2, "shape": 3, "scale": 1},
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
    statistic = KolmogorovSmirnovBetaGofStatistic(Beta.DEFAULT.parse({"a": 2, "b": 5}))
    result = MonteCarloLimitDistributionResolver(2).resolve(statistic, sample_size=3)

    assert result == [statistic.execute_statistic(samples)] * 2
    assert sampler.call_args_list == [mocker.call(size=3, a=2, b=5, random_state=None)] * 2


@pytest.mark.parametrize(
    "criterion, fixed",
    [
        (KolmogorovSmirnovBetaGofStatistic, {"a": 2, "b": 5}),
        (KolmogorovSmirnovGammaGofStatistic, {"alfa": 3, "beta": 4}),
        (KolmogorovSmirnovLaplaceGofStatistic, {"t": 3, "s": 2}),
        (KolmogorovSmirnovUniformGofStatistic, {"a": -2, "b": 3}),
    ],
)
def test_sampler_reads_stable_identities_after_schema_rename(monkeypatch, criterion, fixed):
    from dataclasses import replace

    descriptor = criterion.distribution()
    original = descriptor.DEFAULT.parse(fixed)
    expected = get_hypothesis_generator(criterion(original)).parameters()
    renamed = tuple(replace(p, name="renamed_" + p.name) for p in descriptor.DEFAULT.parameters)
    schema = replace(descriptor.DEFAULT, parameters=renamed)
    monkeypatch.setattr(descriptor, "DEFAULT", schema)
    values = schema.parse({p: original[p] for p in renamed})
    statistic = criterion(values)
    assert get_hypothesis_generator(statistic).parameters() == expected
    assert statistic.hypothesis().parameter_values is values
    assert values.storage_identity() == original.storage_identity()
