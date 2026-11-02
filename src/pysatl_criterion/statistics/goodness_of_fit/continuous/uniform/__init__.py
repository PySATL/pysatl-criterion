"""Goodness-of-fit statistics for a fully specified uniform distribution."""

from pysatl_criterion.statistics.goodness_of_fit.continuous.uniform.uniform_statistics import (
    AbstractUniformGofStatistic,
    AndersonDarlingUniformGofStatistic,
    BickelRosenblattUniformGofStatistic,
    CensoredSteinUniformGofStatistic,
    Chi2PearsonUniformGofStatistic,
    CrammerVonMisesUniformGofStatistic,
    CressieReadUniformGofStatistic,
    FreemanTukeyUniformGofStatistic,
    GreenwoodTestUniformGofStatistic,
    KolmogorovSmirnovDMinusUniformGofStatistic,
    KolmogorovSmirnovDPlusUniformGofStatistic,
    KolmogorovSmirnovUniformGofStatistic,
    KuiperCorrectedUniformGofStatistic,
    KuiperScaledUniformGofStatistic,
    KuiperUniformGofStatistic,
    LiaoShimokawaUniformGofStatistic,
    ModifiedLogLikelihoodUniformGofStatistic,
    NeymanChi2UniformGofStatistic,
    NeymanSmoothTestUniformGofStatistic,
    QuesenberryMillerUniformGofStatistic,
    ShermanUniformGofStatistic,
    SteinUniformGofStatistic,
    WatsonUniformGofStatistic,
    ZhangAUniformGofStatistic,
    ZhangCUniformGofStatistic,
    ZhangKUniformGofStatistic,
)


__all__ = [
    "AbstractUniformGofStatistic",
    "AndersonDarlingUniformGofStatistic",
    "BickelRosenblattUniformGofStatistic",
    "CensoredSteinUniformGofStatistic",
    "Chi2PearsonUniformGofStatistic",
    "CrammerVonMisesUniformGofStatistic",
    "CressieReadUniformGofStatistic",
    "FreemanTukeyUniformGofStatistic",
    "GreenwoodTestUniformGofStatistic",
    "KolmogorovSmirnovDMinusUniformGofStatistic",
    "KolmogorovSmirnovDPlusUniformGofStatistic",
    "KolmogorovSmirnovUniformGofStatistic",
    "KuiperCorrectedUniformGofStatistic",
    "KuiperScaledUniformGofStatistic",
    "KuiperUniformGofStatistic",
    "LiaoShimokawaUniformGofStatistic",
    "ModifiedLogLikelihoodUniformGofStatistic",
    "NeymanChi2UniformGofStatistic",
    "NeymanSmoothTestUniformGofStatistic",
    "QuesenberryMillerUniformGofStatistic",
    "ShermanUniformGofStatistic",
    "SteinUniformGofStatistic",
    "WatsonUniformGofStatistic",
    "ZhangAUniformGofStatistic",
    "ZhangCUniformGofStatistic",
    "ZhangKUniformGofStatistic",
]

# Preserve the public module path for discovery and existing pickles.
for _name in __all__:
    globals()[_name].__module__ = __name__
del _name
