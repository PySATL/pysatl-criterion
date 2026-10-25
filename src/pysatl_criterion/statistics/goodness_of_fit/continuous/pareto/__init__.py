"""Goodness-of-fit statistics for the Pareto I distribution."""

from pysatl_criterion.statistics.goodness_of_fit.continuous.pareto.pareto_statistics import (
    AbstractParetoGofStatistic,
    AndersonDarlingParetoGofStatistic,
    Chi2PearsonParetoGofStatistic,
    CramerVonMisesParetoGofStatistic,
    CressieReadParetoGofStatistic,
    FreemanTukeyParetoGofStatistic,
    GreenwoodParetoGofStatistic,
    KolmogorovSmirnovDMinusParetoGofStatistic,
    KolmogorovSmirnovDPlusParetoGofStatistic,
    KolmogorovSmirnovParetoGofStatistic,
    KuiperCorrectedParetoGofStatistic,
    KuiperParetoGofStatistic,
    KuiperScaledParetoGofStatistic,
    LequesneKlParetoGofStatistic,
    MinToshiyukiParetoGofStatistic,
    ModifiedLogLikelihoodParetoGofStatistic,
    NeymanChi2ParetoGofStatistic,
    ObradovicParetoGofStatistic,
    WatsonParetoGofStatistic,
    ZhangAParetoGofStatistic,
    ZhangCParetoGofStatistic,
    ZhangKParetoGofStatistic,
)


__all__ = [
    "AbstractParetoGofStatistic",
    "AndersonDarlingParetoGofStatistic",
    "Chi2PearsonParetoGofStatistic",
    "CramerVonMisesParetoGofStatistic",
    "CressieReadParetoGofStatistic",
    "FreemanTukeyParetoGofStatistic",
    "GreenwoodParetoGofStatistic",
    "KolmogorovSmirnovDMinusParetoGofStatistic",
    "KolmogorovSmirnovDPlusParetoGofStatistic",
    "KolmogorovSmirnovParetoGofStatistic",
    "KuiperCorrectedParetoGofStatistic",
    "KuiperParetoGofStatistic",
    "KuiperScaledParetoGofStatistic",
    "LequesneKlParetoGofStatistic",
    "MinToshiyukiParetoGofStatistic",
    "ModifiedLogLikelihoodParetoGofStatistic",
    "NeymanChi2ParetoGofStatistic",
    "ObradovicParetoGofStatistic",
    "WatsonParetoGofStatistic",
    "ZhangAParetoGofStatistic",
    "ZhangCParetoGofStatistic",
    "ZhangKParetoGofStatistic",
]

# Preserve the public module path for discovery and existing pickles.
for _name in __all__:
    globals()[_name].__module__ = __name__
del _name
