import pytest

from pysatl_criterion import DistributionType
from pysatl_criterion.generator.generators import (
    BetaRVSGenerator,
    CauchyRVSGenerator,
    Chi2Generator,
    ExponentialGenerator,
    GammaGenerator,
    GompertzGenerator,
    GumbelGenerator,
    InvGaussGenerator,
    LaplaceRVSGenerator,
    LoConNormGenerator,
    LogisticRVSGenerator,
    LognormGenerator,
    MixConNormGenerator,
    NormalGenerator,
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
        (TRVSGenerator(df=5), DistributionType.STUDENT, {"df": 5}),
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
        (WeibullGenerator(a=2, k=5), DistributionType.WEIBULL, {"a": 2, "k": 5}),
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
    ],
)
def test_generator_metadata(generator, distribution_type, parameters):
    assert generator.parameters() == parameters
    assert generator.distribution_type() == distribution_type
    assert generator.code() == "_".join(
        str(item) for item in [distribution_type, *parameters.values()]
    )
