import numpy as np
import pytest
from scipy import stats

from pysatl_criterion import DistributionType
from pysatl_criterion.distribution import (
    HyperbolicDistributionDescriptor,
    InverseGammaDistributionDescriptor,
    LogLogisticDistributionDescriptor,
    ParetoDistributionDescriptor,
)
from pysatl_criterion.generator.generators import (
    BetaRVSGenerator,
    CauchyRVSGenerator,
    Chi2Generator,
    ExponentialGenerator,
    GammaGenerator,
    GompertzGenerator,
    GumbelGenerator,
    HyperbolicGenerator,
    InverseGammaGenerator,
    InvGaussGenerator,
    LaplaceRVSGenerator,
    LoConNormGenerator,
    LogisticRVSGenerator,
    LogLogisticGenerator,
    LognormGenerator,
    MixConNormGenerator,
    NormalGenerator,
    ParetoGenerator,
    RiceGenerator,
    ScConNormGenerator,
    TruncnormGenerator,
    TRVSGenerator,
    TukeyRVSGenerator,
    UniformGenerator,
    WeibullGenerator,
)


@pytest.mark.parametrize(
    ("generator", "distribution_type", "parameters"),
    [
        (BetaRVSGenerator(a=2, b=3), DistributionType.BETA, {"a": 2, "b": 3}),
        (CauchyRVSGenerator(t=1, s=2), DistributionType.CAUCHY, {"t": 1, "s": 2}),
        (LaplaceRVSGenerator(t=1, s=2), DistributionType.LAPLACE, {"t": 1, "s": 2}),
        (LogisticRVSGenerator(t=1, s=2), DistributionType.LOGISTIC, {"t": 1, "s": 2}),
        (TRVSGenerator(df=5), DistributionType.STUDENT, {"df": 5, "loc": 0, "scale": 1}),
        (TukeyRVSGenerator(lam=0.5), DistributionType.TUKEY, {"lam": 0.5}),
        (LognormGenerator(s=1.5, mu=2), DistributionType.LOG_NORMAL, {"s": 1.5, "mu": 2}),
        (GammaGenerator(alfa=2, beta=3), DistributionType.GAMMA, {"alfa": 2, "beta": 3}),
        (
            TruncnormGenerator(mean=1, var=2, a=-3, b=4),
            DistributionType.TRUNC_NORMAL,
            {"mean": 1, "var": 2, "a": -3, "b": 4},
        ),
        (Chi2Generator(df=6), DistributionType.CHI_2, {"df": 6}),
        (GumbelGenerator(mu=1, beta=2), DistributionType.GUMBEL, {"mu": 1, "beta": 2}),
        (WeibullGenerator(scale=2, shape=5), DistributionType.WEIBULL, {"shape": 5, "scale": 2}),
        (LoConNormGenerator(p=0.25, a=2), DistributionType.LO_CON_NORMAL, {"p": 0.25, "a": 2}),
        (ScConNormGenerator(p=0.25, b=2), DistributionType.SCALE_CON_NORMAL, {"p": 0.25, "b": 2}),
        (
            MixConNormGenerator(p=0.25, a=2, b=3),
            DistributionType.MIX_CON_NORMAL,
            {"p": 0.25, "a": 2, "b": 3},
        ),
        (ExponentialGenerator(lam=0.75), DistributionType.EXPONENTIAL, {"lam": 0.75}),
        (InvGaussGenerator(mu=2, lam=3), DistributionType.INV_GAUSS, {"mu": 2, "lam": 3}),
        (RiceGenerator(nu=2, sigma=3), DistributionType.RICE, {"nu": 2, "sigma": 3}),
        (GompertzGenerator(eta=2, b=3), DistributionType.GOMPERTZ, {"eta": 2, "b": 3}),
        (NormalGenerator(mean=2, var=3), DistributionType.NORMAL, {"mean": 2, "var": 3}),
        (UniformGenerator(a=2, b=3), DistributionType.UNIFORM, {"a": 2, "b": 3}),
        (
            LogLogisticGenerator(alpha=2, beta=3),
            DistributionType.LOG_LOGISTIC,
            {"alpha": 2, "beta": 3},
        ),
        (
            InverseGammaGenerator(alpha=2, beta=3),
            DistributionType.INVERSE_GAMMA,
            {"alpha": 2, "beta": 3},
        ),
        (ParetoGenerator(shape=2, scale=3), DistributionType.PARETO, {"shape": 2, "scale": 3}),
        (
            HyperbolicGenerator(alpha=3, beta=1, delta=2, mu=-1),
            DistributionType.HYPERBOLIC,
            {"alpha": 3, "beta": 1, "delta": 2, "mu": -1},
        ),
    ],
)
def test_generator_metadata(generator, distribution_type, parameters):
    assert generator.parameters() == parameters
    assert generator.distribution_type() == distribution_type
    code_parameters = {"df": 5} if distribution_type == DistributionType.STUDENT else parameters
    assert generator.code() == "_".join(
        str(item) for item in [distribution_type.value, *code_parameters.values()]
    )
    with pytest.raises(TypeError, match="Unknown generator parameters: unexpected"):
        type(generator)(**parameters, unexpected=1)


def test_student_generator_code_includes_location_and_scale():
    assert TRVSGenerator(df=5, loc=2, scale=3).code() == "student_5_2_3"


@pytest.mark.parametrize(
    "generator",
    [
        BetaRVSGenerator(a=2, b=3),
        CauchyRVSGenerator(t=1, s=2),
        LaplaceRVSGenerator(t=1, s=2),
        LogisticRVSGenerator(t=1, s=2),
        TRVSGenerator(df=5),
        TukeyRVSGenerator(lam=0.5),
        LognormGenerator(s=1.5, mu=2),
        GammaGenerator(alfa=2, beta=3),
        TruncnormGenerator(mean=1, var=2, a=-3, b=4),
        Chi2Generator(df=6),
        GumbelGenerator(mu=1, beta=2),
        WeibullGenerator(scale=2, shape=5),
        LoConNormGenerator(p=0.25, a=2),
        ScConNormGenerator(p=0.25, b=2),
        MixConNormGenerator(p=0.25, a=2, b=3),
        ExponentialGenerator(lam=0.75),
        InvGaussGenerator(mu=2, lam=3),
        RiceGenerator(nu=2, sigma=3),
        GompertzGenerator(eta=2, b=3),
        NormalGenerator(mean=2, var=3),
        UniformGenerator(a=2, b=3),
        LogLogisticGenerator(alpha=2, beta=3),
        InverseGammaGenerator(alpha=2, beta=3),
        ParetoGenerator(shape=2, scale=3),
        HyperbolicGenerator(alpha=3, beta=1, delta=2, mu=-1),
    ],
)
def test_generator_uses_caller_owned_random_state(generator):
    first = generator.generate(20, random_state=np.random.default_rng(123))
    second = generator.generate(20, random_state=np.random.default_rng(123))

    assert np.array_equal(first, second)


@pytest.mark.parametrize(
    ("generator", "descriptor", "params"),
    [
        (
            LogLogisticGenerator(alpha=2.0, beta=3.0),
            LogLogisticDistributionDescriptor,
            {"alpha": 2.0, "beta": 3.0},
        ),
        (
            InverseGammaGenerator(alpha=2.5, beta=1.5),
            InverseGammaDistributionDescriptor,
            {"alpha": 2.5, "beta": 1.5},
        ),
        (
            ParetoGenerator(shape=2.5, scale=1.5),
            ParetoDistributionDescriptor,
            {"shape": 2.5, "scale": 1.5},
        ),
        (
            HyperbolicGenerator(alpha=3.0, beta=1.0, delta=2.0, mu=-0.5),
            HyperbolicDistributionDescriptor,
            {"alpha": 3.0, "beta": 1.0, "delta": 2.0, "mu": -0.5},
        ),
    ],
)
def test_new_generators_match_descriptor_cdf(generator, descriptor, params):
    sample = generator.generate(5000, random_state=np.random.default_rng(42))
    bound = descriptor.bind(descriptor.DEFAULT.parse(params))
    ks_result = stats.kstest(sample, bound.cdf)
    assert ks_result.pvalue > 0.01


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": 1.0, "beta": 1.0},
        {"alpha": 1.0, "beta": -1.5},
        {"alpha": 2.0, "beta": 0.0, "delta": 0.0},
    ],
)
def test_hyperbolic_generator_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        HyperbolicGenerator(**kwargs)
