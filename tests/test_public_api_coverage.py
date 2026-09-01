import inspect

import numpy as np
import pytest

from pysatl_criterion import DistributionType
from pysatl_criterion.api import distributions as distributions_api
from pysatl_criterion.api import testing as testing_api
from pysatl_criterion.core.distributions import (
    beta,
    cauchy,
    chi2,
    expon,
    gamma,
    gompertz,
    gumbel,
    invgauss,
    laplace,
    lo_con_norm,
    logistic,
    lognormal,
    mix_con_norm,
    norm,
    rice,
    sample,
    scale_con_norm,
    student,
    truncnormal,
    tukey,
    uniform,
    weibull,
)
from pysatl_criterion.distribution import distributions
from pysatl_criterion.distribution.distribution_type import DistributionParameterDescriptor
from pysatl_criterion.distribution.validator import (
    NonNegativeNumberValidator,
    PositiveNumberValidator,
    ProbabilityValidator,
    Validator,
)
from pysatl_criterion.generator import generators
from pysatl_criterion.generator.model import AbstractRVSGenerator
from pysatl_criterion.statistics.goodness_of_fit import normal as normal_statistics
from pysatl_criterion.statistics.goodness_of_fit import weibull as weibull_statistics


def test_convenience_api_exports():
    assert distributions_api.DistributionType is DistributionType
    assert testing_api.GoodnessOfFitTest is not None
    assert testing_api.BonferroniMultipleTesting is not None
    assert testing_api.Holm is not None
    assert testing_api.SidakMultipleTesting is not None
    assert testing_api.SidakHolm is not None
    assert testing_api.BenjaminiYekutieli is not None
    assert testing_api.PValueCalculator is not None
    assert testing_api.CriticalValueCalculator is not None


def test_all_distribution_descriptors_expose_complete_metadata():
    descriptor_classes = [
        cls
        for _, cls in inspect.getmembers(distributions, inspect.isclass)
        if issubclass(cls, distributions.DistributionDescriptor)
        and cls is not distributions.DistributionDescriptor
    ]

    assert len(descriptor_classes) == 21
    assert {cls.type() for cls in descriptor_classes} <= set(DistributionType)
    for descriptor_class in descriptor_classes:
        for parameter in descriptor_class.parameters():
            assert parameter.display_name
            assert parameter.name

    assert DistributionType.list() == [item.value for item in DistributionType]
    assert DistributionParameterDescriptor("x", "x").name == "x"


def test_parameter_validators_and_abstract_contract():
    assert PositiveNumberValidator()(1)
    assert not PositiveNumberValidator()(0)
    assert NonNegativeNumberValidator()(0)
    assert not NonNegativeNumberValidator()(-1)
    assert ProbabilityValidator()(0)
    assert ProbabilityValidator()(1)
    assert not ProbabilityValidator()(-0.1)
    assert not ProbabilityValidator()(1.1)
    assert Validator.validate(object(), 1) is None


@pytest.mark.parametrize(
    ("function", "kwargs"),
    [
        (beta.generate_beta, {"a": 2, "b": 3}),
        (cauchy.generate_cauchy, {"t": 1, "s": 2}),
        (chi2.generate_chi2, {"df": 3}),
        (gamma.generate_gamma, {"alfa": 2, "beta": 3}),
        (gompertz.generate_gompertz, {"eta": 2, "b": 3}),
        (gumbel.generate_gumbel, {"mu": 1, "beta": 2}),
        (invgauss.generate_invgauss, {"mu": 2, "lam": 3}),
        (laplace.generate_laplace, {"t": 1, "s": 2}),
        (logistic.generate_logistic, {"t": 1, "s": 2}),
        (lognormal.generate_lognorm, {"mu": 1, "s": 2}),
        (rice.generate_rice, {"nu": 1, "sigma": 2}),
        (student.generate_t, {"df": 3}),
        (truncnormal.generate_truncnorm, {"mean": 0, "var": 1, "a": -2, "b": 2}),
        (tukey.generate_tukey, {"lam": 2}),
        (uniform.generate_uniform, {"a": 1, "b": 2}),
    ],
)
def test_core_generators(function, kwargs):
    assert len(function(size=3, **kwargs)) == 3


def test_core_distribution_helpers_and_contaminated_generators():
    values = np.array([-1.0, 0.0, 1.0])
    assert expon.generate_expon(3, lam=2).shape == (3,)
    assert expon.cdf_expon(values, lam=2).shape == (3,)
    assert norm.generate_norm(3, mean=1, var=2).shape == (3,)
    assert norm.cdf_norm(values, mean=1, var=2).shape == (3,)
    assert norm.pdf_norm(values, mean=1, var=2).shape == (3,)
    assert len(weibull.generate_weibull(3, a=2, k=3)) == 3
    assert weibull.generate_weibull_cdf(values, a=2, k=3).shape == (3,)
    assert weibull.generate_weibull_logcdf(values, a=2, k=3).shape == (3,)
    assert weibull.generate_weibull_logsf(values, a=2, k=3).shape == (3,)

    for function, kwargs in (
        (lo_con_norm.generate_lo_con_norm, {"a": 2}),
        (scale_con_norm.generate_scale_con_norm, {"b": 2}),
        (mix_con_norm.generate_mix_con_norm, {"a": 2, "b": 3}),
    ):
        assert len(function(2, p=1, **kwargs)) == 2
        assert len(function(2, p=0, **kwargs)) == 2

    assert sample.moment(values, mom=2).shape == ()
    assert sample.central_moment(values, mom=2).shape == ()


def test_every_generator_can_generate_a_sample():
    instances = [
        generators.BetaRVSGenerator(a=2, b=3),
        generators.CauchyRVSGenerator(t=1, s=2),
        generators.LaplaceRVSGenerator(t=1, s=2),
        generators.LogisticRVSGenerator(t=1, s=2),
        generators.TRVSGenerator(df=3),
        generators.TukeyRVSGenerator(lam=2),
        generators.LognormGenerator(s=1, mu=0),
        generators.GammaGenerator(alfa=2, beta=3),
        generators.TruncnormGenerator(mean=0, var=1, a=-2, b=2),
        generators.Chi2Generator(df=3),
        generators.GumbelGenerator(mu=0, beta=1),
        generators.WeibullGenerator(a=2, k=3),
        generators.LoConNormGenerator(p=1, a=2),
        generators.ScConNormGenerator(p=1, b=2),
        generators.MixConNormGenerator(p=1, a=2, b=3),
        generators.ExponentialGenerator(lam=2),
        generators.InvGaussGenerator(mu=2, lam=3),
        generators.RiceGenerator(nu=1, sigma=2),
        generators.GompertzGenerator(eta=2, b=3),
        generators.NormalGenerator(mean=0, var=1),
        generators.UniformGenerator(a=0, b=1),
    ]

    for generator in instances:
        assert len(generator.generate(2)) == 2


def test_abstract_generator_contract_raises():
    for method, arguments in (
        (AbstractRVSGenerator.distribution_type, ()),
        (AbstractRVSGenerator.parameters, (object(),)),
        (AbstractRVSGenerator.code, (object(),)),
        (AbstractRVSGenerator.generate, (1,)),
    ):
        with pytest.raises(NotImplementedError):
            method(*arguments)


def test_all_normality_statistics_expose_metadata():
    classes = [
        cls
        for _, cls in inspect.getmembers(normal_statistics, inspect.isclass)
        if cls.__module__ == normal_statistics.__name__
        and issubclass(cls, normal_statistics.AbstractNormalityGofStatistic)
        and not inspect.isabstract(cls)
    ]

    assert len(classes) > 40
    for statistic_class in classes:
        statistic = statistic_class()
        assert statistic.code()
        assert statistic.short_code()
        assert statistic.alternative()
        assert statistic.distribution() is DistributionType.NORMAL


def test_all_normality_statistics_execute_on_a_representative_sample():
    sample_values = np.random.default_rng(42).normal(size=20).tolist()
    classes = [
        cls
        for _, cls in inspect.getmembers(normal_statistics, inspect.isclass)
        if cls.__module__ == normal_statistics.__name__
        and issubclass(cls, normal_statistics.AbstractNormalityGofStatistic)
        and not inspect.isabstract(cls)
        and cls is not normal_statistics.BHSNormalityGofStatistic
    ]

    for statistic_class in classes:
        assert np.isscalar(statistic_class().execute_statistic(sample_values))

    # Four observations exercise the BHS implementation without its expensive
    # large-sample weighted-median search.
    assert np.isscalar(
        normal_statistics.BHSNormalityGofStatistic().execute_statistic([1.0, 2.0, 3.0, 4.0])
    )


@pytest.mark.parametrize(
    "statistic_class",
    [
        weibull_statistics.AndersonDarlingWeibullGofStatistic,
        weibull_statistics.Chi2PearsonWeibullGofStatistic,
        weibull_statistics.MahdiDoostparastWeibullGofStatistic,
        weibull_statistics.WatsonWeibullGofStatistic,
        weibull_statistics.LiaoShimokawaWeibullGofStatistic,
        weibull_statistics.KullbackLeiblerWeibullGofStatistic,
        weibull_statistics.LaplaceTransform2WeibullGofStatistic,
        weibull_statistics.LaplaceTransform3WeibullGofStatistic,
        weibull_statistics.CabanaQuirozWeibullGofStatistic,
    ],
)
def test_previously_uncovered_weibull_statistics(statistic_class):
    sample_values = np.linspace(0.4, 3.0, 20)
    result = statistic_class().execute_statistic(sample_values)
    assert np.isscalar(result)


def test_all_weibull_statistics_expose_metadata():
    classes = [
        cls
        for _, cls in inspect.getmembers(weibull_statistics, inspect.isclass)
        if cls.__module__ == weibull_statistics.__name__
        and issubclass(cls, weibull_statistics.AbstractWeibullGofStatistic)
        and not inspect.isabstract(cls)
    ]

    assert len(classes) > 20
    for statistic_class in classes:
        statistic = statistic_class()
        assert statistic.code()
        assert statistic.short_code()
        assert statistic.alternative()
        assert statistic.distribution() is DistributionType.WEIBULL


def test_all_concrete_weibull_statistics_execute():
    sample_values = np.linspace(0.4, 3.0, 20)
    classes = [
        cls
        for _, cls in inspect.getmembers(weibull_statistics, inspect.isclass)
        if cls.__module__ == weibull_statistics.__name__
        and issubclass(cls, weibull_statistics.AbstractWeibullGofStatistic)
        and not inspect.isabstract(cls)
    ]

    for statistic_class in classes:
        statistic = statistic_class()
        parameters = inspect.signature(statistic.execute_statistic).parameters
        result = (
            statistic.execute_statistic(sample_values, 1)
            if "type_" in parameters and parameters["type_"].default is inspect.Parameter.empty
            else statistic.execute_statistic(sample_values)
        )
        assert np.isscalar(result)
