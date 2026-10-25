"""Goodness-of-fit statistics on the fixed Beta support [0, 1]."""

from pysatl_criterion.statistics.goodness_of_fit.continuous.beta.beta_statistics import (
    AbstractBetaGofStatistic,
    AbstractSpecifiedBetaGofStatistic,
    AndersonDarlingBetaGofStatistic,
    Chi2PearsonBetaGofStatistic,
    CrammerVonMisesBetaGofStatistic,
    CressieReadBetaGofStatistic,
    EbnerLiebenbergBetaGofStatistic,
    FreemanTukeyBetaGofStatistic,
    KolmogorovSmirnovBetaGofStatistic,
    KolmogorovSmirnovDMinusBetaGofStatistic,
    KolmogorovSmirnovDPlusBetaGofStatistic,
    KuiperBetaGofStatistic,
    KuiperCorrectedBetaGofStatistic,
    KuiperScaledBetaGofStatistic,
    KullbackLeiblerBetaGofStatistic,
    LiaoShimokawaBetaGofStatistic,
    ModifiedLogLikelihoodBetaGofStatistic,
    NeymanChi2BetaGofStatistic,
    NeymanSmoothBetaGofStatistic,
    RaschkeBetaGofStatistic,
    RiadAbdElMaboodBetaGofStatistic,
    WatsonBetaGofStatistic,
    ZhangABetaGofStatistic,
    ZhangCBetaGofStatistic,
    ZhangKBetaGofStatistic,
)


__all__ = [
    "AbstractBetaGofStatistic",
    "AbstractSpecifiedBetaGofStatistic",
    "AndersonDarlingBetaGofStatistic",
    "Chi2PearsonBetaGofStatistic",
    "CrammerVonMisesBetaGofStatistic",
    "CressieReadBetaGofStatistic",
    "EbnerLiebenbergBetaGofStatistic",
    "FreemanTukeyBetaGofStatistic",
    "KolmogorovSmirnovBetaGofStatistic",
    "KolmogorovSmirnovDMinusBetaGofStatistic",
    "KolmogorovSmirnovDPlusBetaGofStatistic",
    "KuiperBetaGofStatistic",
    "KuiperCorrectedBetaGofStatistic",
    "KuiperScaledBetaGofStatistic",
    "KullbackLeiblerBetaGofStatistic",
    "LiaoShimokawaBetaGofStatistic",
    "ModifiedLogLikelihoodBetaGofStatistic",
    "NeymanChi2BetaGofStatistic",
    "NeymanSmoothBetaGofStatistic",
    "RaschkeBetaGofStatistic",
    "RiadAbdElMaboodBetaGofStatistic",
    "WatsonBetaGofStatistic",
    "ZhangABetaGofStatistic",
    "ZhangCBetaGofStatistic",
    "ZhangKBetaGofStatistic",
]

# Preserve the public module path for discovery and existing pickles.
for _name in __all__:
    globals()[_name].__module__ = __name__
del _name
