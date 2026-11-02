"""Goodness-of-fit statistics for a fully specified inverse-gamma distribution."""

from .inverse_gamma_statistics import (
    AbstractInverseGammaGofStatistic,
    AndersonDarlingInverseGammaGofStatistic,
    Chi2PearsonInverseGammaGofStatistic,
    CramerVonMisesInverseGammaGofStatistic,
    CressieReadInverseGammaGofStatistic,
    FreemanTukeyInverseGammaGofStatistic,
    GreenwoodInverseGammaGofStatistic,
    KolmogorovSmirnovDMinusInverseGammaGofStatistic,
    KolmogorovSmirnovDPlusInverseGammaGofStatistic,
    KolmogorovSmirnovInverseGammaGofStatistic,
    KuiperCorrectedInverseGammaGofStatistic,
    KuiperInverseGammaGofStatistic,
    KuiperScaledInverseGammaGofStatistic,
    LiaoShimokawaInverseGammaGofStatistic,
    ModifiedLogLikelihoodInverseGammaGofStatistic,
    NeymanChi2InverseGammaGofStatistic,
    WatsonInverseGammaGofStatistic,
    ZhangAInverseGammaGofStatistic,
    ZhangCInverseGammaGofStatistic,
    ZhangKInverseGammaGofStatistic,
)


__all__ = [
    "AbstractInverseGammaGofStatistic",
    "AndersonDarlingInverseGammaGofStatistic",
    "Chi2PearsonInverseGammaGofStatistic",
    "CramerVonMisesInverseGammaGofStatistic",
    "CressieReadInverseGammaGofStatistic",
    "FreemanTukeyInverseGammaGofStatistic",
    "GreenwoodInverseGammaGofStatistic",
    "KolmogorovSmirnovDMinusInverseGammaGofStatistic",
    "KolmogorovSmirnovDPlusInverseGammaGofStatistic",
    "KolmogorovSmirnovInverseGammaGofStatistic",
    "KuiperCorrectedInverseGammaGofStatistic",
    "KuiperInverseGammaGofStatistic",
    "KuiperScaledInverseGammaGofStatistic",
    "LiaoShimokawaInverseGammaGofStatistic",
    "ModifiedLogLikelihoodInverseGammaGofStatistic",
    "NeymanChi2InverseGammaGofStatistic",
    "WatsonInverseGammaGofStatistic",
    "ZhangAInverseGammaGofStatistic",
    "ZhangCInverseGammaGofStatistic",
    "ZhangKInverseGammaGofStatistic",
]

# Preserve the public module path for discovery and existing pickles.
for _name in __all__:
    globals()[_name].__module__ = __name__
del _name
