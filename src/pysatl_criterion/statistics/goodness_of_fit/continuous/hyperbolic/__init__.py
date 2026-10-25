"""Goodness-of-fit tests for a fully specified hyperbolic distribution."""

from .hyperbolic_statistics import (
    AbstractHyperbolicGofStatistic,
    AndersonDarlingHyperbolicGofStatistic,
    Chi2PearsonHyperbolicGofStatistic,
    CramerVonMisesHyperbolicGofStatistic,
    CressieReadHyperbolicGofStatistic,
    FreemanTukeyHyperbolicGofStatistic,
    KolmogorovSmirnovHyperbolicGofStatistic,
    KuiperHyperbolicGofStatistic,
    ModifiedLogLikelihoodHyperbolicGofStatistic,
    NeymanChi2HyperbolicGofStatistic,
    WatsonHyperbolicGofStatistic,
)


__all__ = [
    "AbstractHyperbolicGofStatistic",
    "AndersonDarlingHyperbolicGofStatistic",
    "Chi2PearsonHyperbolicGofStatistic",
    "CramerVonMisesHyperbolicGofStatistic",
    "CressieReadHyperbolicGofStatistic",
    "FreemanTukeyHyperbolicGofStatistic",
    "KolmogorovSmirnovHyperbolicGofStatistic",
    "KuiperHyperbolicGofStatistic",
    "ModifiedLogLikelihoodHyperbolicGofStatistic",
    "NeymanChi2HyperbolicGofStatistic",
    "WatsonHyperbolicGofStatistic",
]

for _name in __all__:
    globals()[_name].__module__ = __name__
del _name
