"""Goodness-of-fit statistics for the log-logistic distribution."""

from .log_logistic_statistics import (
    AbstractLogLogisticGofStatistic,
    AndersonDarlingLogLogisticGofStatistic,
    Chi2PearsonLogLogisticGofStatistic,
    CramerVonMisesLogLogisticGofStatistic,
    CressieReadLogLogisticGofStatistic,
    FreemanTukeyLogLogisticGofStatistic,
    KolmogorovSmirnovDMinusLogLogisticGofStatistic,
    KolmogorovSmirnovDPlusLogLogisticGofStatistic,
    KolmogorovSmirnovLogLogisticGofStatistic,
    KuiperCorrectedLogLogisticGofStatistic,
    KuiperLogLogisticGofStatistic,
    KuiperScaledLogLogisticGofStatistic,
    LiaoShimokawaLogLogisticGofStatistic,
    MirvalievLogLogisticGofStatistic,
    ModifiedLogLikelihoodLogLogisticGofStatistic,
    NeymanChi2LogLogisticGofStatistic,
    NikulinLogLogisticGofStatistic,
    WatsonLogLogisticGofStatistic,
    ZhangALogLogisticGofStatistic,
    ZhangCLogLogisticGofStatistic,
    ZhangKLogLogisticGofStatistic,
)


__all__ = [
    "AbstractLogLogisticGofStatistic",
    "AndersonDarlingLogLogisticGofStatistic",
    "Chi2PearsonLogLogisticGofStatistic",
    "CramerVonMisesLogLogisticGofStatistic",
    "CressieReadLogLogisticGofStatistic",
    "FreemanTukeyLogLogisticGofStatistic",
    "KolmogorovSmirnovDMinusLogLogisticGofStatistic",
    "KolmogorovSmirnovDPlusLogLogisticGofStatistic",
    "KolmogorovSmirnovLogLogisticGofStatistic",
    "KuiperCorrectedLogLogisticGofStatistic",
    "KuiperLogLogisticGofStatistic",
    "KuiperScaledLogLogisticGofStatistic",
    "LiaoShimokawaLogLogisticGofStatistic",
    "MirvalievLogLogisticGofStatistic",
    "ModifiedLogLikelihoodLogLogisticGofStatistic",
    "NeymanChi2LogLogisticGofStatistic",
    "NikulinLogLogisticGofStatistic",
    "WatsonLogLogisticGofStatistic",
    "ZhangALogLogisticGofStatistic",
    "ZhangCLogLogisticGofStatistic",
    "ZhangKLogLogisticGofStatistic",
]

# Preserve the public module path for discovery and existing pickles.
for _name in __all__:
    globals()[_name].__module__ = __name__
del _name
