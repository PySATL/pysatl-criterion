"""Distinct distribution identities and SciPy references for both Weibull families."""

from itertools import combinations

import numpy as np
import pytest
from scipy import stats

from pysatl_criterion import DistributionType
from pysatl_criterion.core.distributions.continues import exponentiated_weibull as exp_core
from pysatl_criterion.core.distributions.continues import weibull as core
from pysatl_criterion.distribution.distributions import (
    ExponentiatedWeibullDistributionDescriptor as Exp,
)
from pysatl_criterion.distribution.distributions import WeibullDistributionDescriptor as Weibull
from pysatl_criterion.generator.generators import ExponentiatedWeibullGenerator, WeibullGenerator
from pysatl_criterion.statistics.goodness_of_fit import exponentiated_weibull as exp
from pysatl_criterion.statistics.goodness_of_fit import weibull
from pysatl_criterion.statistics.hypothesis import GoodnessOfFitHypothesis
from pysatl_criterion.utils.distribution import get_available_distribution_descriptor
from pysatl_criterion.utils.generator import get_available_generator, get_hypothesis_generator
from pysatl_criterion.utils.statistic import get_available_criteria, get_criterion_by_code


@pytest.mark.parametrize("scale", [0.2, 1, 4])
def test_sampling_and_probability_functions_match_the_correct_family(scale):
    shape, exponent = 1.7, 2.3
    x = np.array([-1, 0, 0.1, 0.5, 2, 10]) * scale
    for name in ("cdf", "logcdf", "logsf"):
        np.testing.assert_allclose(
            getattr(core, f"generate_weibull_{name}")(x, shape=shape, scale=scale),
            getattr(stats.weibull_min(shape, scale=scale), name)(x),
        )
        np.testing.assert_allclose(
            getattr(exp_core, f"generate_exponentiated_weibull_{name}")(
                x, exponent=exponent, shape=shape, scale=scale
            ),
            getattr(stats.exponweib(exponent, shape, scale=scale), name)(x),
        )
        np.testing.assert_allclose(
            getattr(exp_core, f"generate_exponentiated_weibull_{name}")(
                x, exponent=1, shape=shape, scale=scale
            ),
            getattr(core, f"generate_weibull_{name}")(x, shape=shape, scale=scale),
        )
    for generator, reference in (
        (WeibullGenerator(shape=shape, scale=scale), stats.weibull_min(shape, scale=scale)),
        (
            ExponentiatedWeibullGenerator(exponent=exponent, shape=shape, scale=scale),
            stats.exponweib(exponent, shape, scale=scale),
        ),
    ):
        actual = generator.generate(100, random_state=np.random.default_rng(42))
        expected = reference.rvs(size=100, random_state=np.random.default_rng(42))
        np.testing.assert_array_equal(actual, expected)


def test_registries_do_not_mix_families_or_reuse_old_codes():
    assert Exp.type() is DistributionType.EXPONENTIATED_WEIBULL
    assert Weibull.type() is DistributionType.WEIBULL
    for descriptor, generator_class, parameters in (
        (Weibull, WeibullGenerator, {"shape": 2, "scale": 3}),
        (Exp, ExponentiatedWeibullGenerator, {"exponent": 4, "shape": 2, "scale": 3}),
    ):
        assert isinstance(get_available_distribution_descriptor(descriptor.type()), descriptor)
        generator = get_available_generator(descriptor.type(), parameters)
        assert isinstance(generator, generator_class)
        assert generator.parameters() == parameters
        assert generator.code().startswith(descriptor.type().value + "_")
        criteria = get_available_criteria(descriptor.type())
        assert criteria
        assert all(cls.distribution() is descriptor for cls in criteria)
    assert len(get_available_criteria(Exp.type())) == 5
    assert len(get_available_criteria(Weibull.type())) == 18
    with pytest.raises(ValueError):
        get_criterion_by_code("KS_WEIBULL_GOODNESS_OF_FIT")
    assert (
        get_criterion_by_code("KS_EXPONENTIATED_WEIBULL_GOODNESS_OF_FIT")
        is exp.KolmogorovSmirnovExponentiatedWeibullGofStatistic
    )
    with pytest.raises(TypeError):
        WeibullGenerator(a=2, k=3)
    with pytest.raises(ValueError):
        Exp.convert_parameters(Weibull.DEFAULT.parse({}), Exp.DEFAULT)


@pytest.mark.parametrize(
    "criterion", get_available_criteria(DistributionType.EXPONENTIATED_WEIBULL)
)
@pytest.mark.parametrize("scale", [1e-100, 0.3, 7, 1e100])
def test_specified_criteria_respect_scale_and_keep_input_unchanged(criterion, scale):
    sample = np.array([0.12, 0.23, 0.4, 0.61, 0.88, 1.05, 1.37, 1.7, 2.3])
    unit = Exp.DEFAULT.parse({"exponent": 1.4, "shape": 2.1, "scale": 1})
    scaled = Exp.DEFAULT.parse({"exponent": 1.4, "shape": 2.1, "scale": scale})
    expected = criterion(unit).execute_statistic(sample)
    observations = sample * scale
    original = observations.copy()
    statistic = criterion(scaled)
    assert statistic.execute_statistic(observations) == pytest.approx(expected, rel=1e-10)
    np.testing.assert_array_equal(observations, original)
    assert statistic.hypothesis().parameter_values is scaled
    assert get_hypothesis_generator(statistic).parameters() == scaled.as_dict()
    with pytest.raises(ValueError, match="Unsupported"):
        criterion(Weibull.DEFAULT.parse({"shape": 2, "scale": 1}))
    with pytest.raises(ValueError, match="Unsupported"):
        criterion(Exp.DEFAULT.parse({"exponent": 1, "shape": 2}))


def test_ordinary_composite_calibration_uses_ordinary_weibull_and_keeps_shapes_unknown():
    values = Weibull.DEFAULT.parse({})
    statistic = weibull.AndersonDarlingWeibullGofStatistic(values)
    generator = get_hypothesis_generator(statistic)
    assert isinstance(generator, WeibullGenerator)
    assert generator.parameters() == {"shape": 1, "scale": 1}
    assert statistic.hypothesis().parameter_values is values
    assert statistic.hypothesis().parameters() == {}
    record_statistic = weibull.MahdiDoostparastWeibullGofStatistic(values)
    with pytest.raises(ValueError, match="Record calibration"):
        get_hypothesis_generator(record_statistic)


@pytest.mark.parametrize("family", [Weibull, Exp])
def test_every_criterion_enforces_family_and_exact_fixed_parameter_set(family):
    for criterion in get_available_criteria(family.type()):
        for descriptor in (Weibull, Exp):
            names = [parameter.name for parameter in descriptor.DEFAULT.parameters]
            for count in range(len(names) + 1):
                for fixed in combinations(names, count):
                    values = descriptor.DEFAULT.parse(dict.fromkeys(fixed, 2))
                    supported = descriptor is family and (
                        count == 0 if family is Weibull else count == len(names)
                    )
                    assert (
                        criterion.supports_hypothesis(GoodnessOfFitHypothesis(values)) is supported
                    )
                    if supported:
                        assert criterion(values).hypothesis().parameter_values is values
                    else:
                        with pytest.raises(ValueError, match="Unsupported"):
                            criterion(values)


@pytest.mark.parametrize("exponent", [1, 2.3])
@pytest.mark.parametrize("scale", [0.2, 4])
def test_fixed_edf_statistics_match_scipy_with_explicit_scale(exponent, scale):
    sample = np.array([0, 0.12, 0.4, 0.88, 1.37, 2.3, 8])
    parameters = Exp.DEFAULT.parse({"exponent": exponent, "shape": 1.7, "scale": scale})
    reference = stats.exponweib(exponent, 1.7, scale=scale)
    assert exp.KolmogorovSmirnovExponentiatedWeibullGofStatistic(parameters).execute_statistic(
        sample
    ) == pytest.approx(stats.kstest(sample, reference.cdf).statistic)
    cvm = stats.cramervonmises(sample, reference.cdf).statistic
    assert exp.CrammerVonMisesExponentiatedWeibullGofStatistic(parameters).execute_statistic(
        sample
    ) == pytest.approx(cvm)
    assert exp.WatsonExponentiatedWeibullGofStatistic(parameters).execute_statistic(
        sample
    ) == pytest.approx(cvm - len(sample) * (reference.cdf(sample).mean() - 0.5) ** 2)
