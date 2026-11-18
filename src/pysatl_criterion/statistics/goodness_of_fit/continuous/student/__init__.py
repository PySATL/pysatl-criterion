"""Goodness-of-fit statistics for Student distributions."""

from pysatl_criterion.statistics.goodness_of_fit.continuous.student.student_statistics import (
    AbstractStudentGofStatistic,
    AndersonDarlingStudentGofStatistic,
    CadirciRenyiStudentGofStatistic,
    ChiSquareStudentGofStatistic,
    CramerVonMisesStudentGofStatistic,
    CressieReadStudentGofStatistic,
    FreemanTukeyStudentGofStatistic,
    KolmogorovSmirnovDMinusStudentGofStatistic,
    KolmogorovSmirnovDPlusStudentGofStatistic,
    KolmogorovSmirnovStudentGofStatistic,
    KuiperCorrectedStudentGofStatistic,
    KuiperScaledStudentGofStatistic,
    KuiperStudentGofStatistic,
    LequesneRenyiStudentGofStatistic,
    LiaoShimokawaStudentGofStatistic,
    MeintanisStudentGofStatistic,
    ModifiedLogLikelihoodStudentGofStatistic,
    NeymanChi2StudentGofStatistic,
    WatsonStudentGofStatistic,
    ZhangKStudentGofStatistic,
    ZhangZaStudentGofStatistic,
    ZhangZcStudentGofStatistic,
)


__all__ = [
    "AbstractStudentGofStatistic",
    "AndersonDarlingStudentGofStatistic",
    "CadirciRenyiStudentGofStatistic",
    "ChiSquareStudentGofStatistic",
    "CramerVonMisesStudentGofStatistic",
    "CressieReadStudentGofStatistic",
    "FreemanTukeyStudentGofStatistic",
    "KolmogorovSmirnovDMinusStudentGofStatistic",
    "KolmogorovSmirnovDPlusStudentGofStatistic",
    "KolmogorovSmirnovStudentGofStatistic",
    "KuiperCorrectedStudentGofStatistic",
    "KuiperScaledStudentGofStatistic",
    "KuiperStudentGofStatistic",
    "LequesneRenyiStudentGofStatistic",
    "LiaoShimokawaStudentGofStatistic",
    "MeintanisStudentGofStatistic",
    "ModifiedLogLikelihoodStudentGofStatistic",
    "NeymanChi2StudentGofStatistic",
    "WatsonStudentGofStatistic",
    "ZhangKStudentGofStatistic",
    "ZhangZaStudentGofStatistic",
    "ZhangZcStudentGofStatistic",
]

# Preserve the public module path for discovery and existing pickles.
for _name in __all__:
    globals()[_name].__module__ = __name__
del _name
