"""Goodness-of-fit statistics for a fully specified exponentiated Weibull null."""

from pysatl_criterion.statistics.goodness_of_fit.continuous.exponentiated_weibull.exponentiated_weibull_statistics import (  # noqa: E501
    AbstractExponentiatedWeibullGofStatistic,
    AndersonDarlingExponentiatedWeibullGofStatistic,
    Chi2PearsonExponentiatedWeibullGofStatistic,
    CrammerVonMisesExponentiatedWeibullGofStatistic,
    CressieReadExponentiatedWeibullGofStatistic,
    FreemanTukeyExponentiatedWeibullGofStatistic,
    KolmogorovSmirnovDMinusExponentiatedWeibullGofStatistic,
    KolmogorovSmirnovDPlusExponentiatedWeibullGofStatistic,
    KolmogorovSmirnovExponentiatedWeibullGofStatistic,
    KuiperCorrectedExponentiatedWeibullGofStatistic,
    KuiperExponentiatedWeibullGofStatistic,
    KuiperScaledExponentiatedWeibullGofStatistic,
    MinToshiyukiExponentiatedWeibullGofStatistic,
    ModifiedLogLikelihoodExponentiatedWeibullGofStatistic,
    NeymanChi2ExponentiatedWeibullGofStatistic,
    WatsonExponentiatedWeibullGofStatistic,
    ZhangAExponentiatedWeibullGofStatistic,
    ZhangCExponentiatedWeibullGofStatistic,
    ZhangKExponentiatedWeibullGofStatistic,
)


__all__ = [
    "AbstractExponentiatedWeibullGofStatistic",
    "AndersonDarlingExponentiatedWeibullGofStatistic",
    "Chi2PearsonExponentiatedWeibullGofStatistic",
    "CrammerVonMisesExponentiatedWeibullGofStatistic",
    "CressieReadExponentiatedWeibullGofStatistic",
    "FreemanTukeyExponentiatedWeibullGofStatistic",
    "KolmogorovSmirnovDMinusExponentiatedWeibullGofStatistic",
    "KolmogorovSmirnovDPlusExponentiatedWeibullGofStatistic",
    "KolmogorovSmirnovExponentiatedWeibullGofStatistic",
    "KuiperCorrectedExponentiatedWeibullGofStatistic",
    "KuiperExponentiatedWeibullGofStatistic",
    "KuiperScaledExponentiatedWeibullGofStatistic",
    "MinToshiyukiExponentiatedWeibullGofStatistic",
    "ModifiedLogLikelihoodExponentiatedWeibullGofStatistic",
    "NeymanChi2ExponentiatedWeibullGofStatistic",
    "WatsonExponentiatedWeibullGofStatistic",
    "ZhangAExponentiatedWeibullGofStatistic",
    "ZhangCExponentiatedWeibullGofStatistic",
    "ZhangKExponentiatedWeibullGofStatistic",
]

# Preserve the public module path for discovery and existing pickles.
for _name in __all__:
    globals()[_name].__module__ = __name__
del _name
