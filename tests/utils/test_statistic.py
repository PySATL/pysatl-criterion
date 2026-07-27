import inspect
from collections import Counter

import pytest

from pysatl_criterion import DistributionType
from pysatl_criterion.statistics import AbstractGoodnessOfFitStatistic
from pysatl_criterion.utils.statistic import get_available_criteria, get_available_criteria_codes


@pytest.mark.parametrize(
    ("distribution", "result"),
    [
        (
            "normal",
            [
                "GLB",
                "LG",
                "BS",
                "HOSKING1",
                "INDEPENDENCENUMBER",
                "AD",
                "ZQ",
                "SKEW",
                "GMG",
                "HOSKING3",
                "DH",
                "EDGESNUMBER",
                "RJ",
                "MI",
                "SW",
                "COIN",
                "AVGDEGREE",
                "HOSKING4",
                "KURTOSIS",
                "BHS",
                "RJB",
                "MAXDEGREE",
                "SF",
                "CC1",
                "CVM",
                "D",
                "ZWC",
                "DAP",
                "SH",
                "BM1",
                "EP",
                "CONNECTEDCOMPONENTS",
                "CC2",
                "LILLIE",
                "ZQS",
                "ZWA",
                "FILLI",
                "DLDMZEPD",
                "BM2",
                "HOSKING2",
                "CLIQUENUMBER",
                "KS",
                "CS",
                "JB",
                "SWRG",
            ],
        ),
        (
            "exponential",
            [
                "INDEPENDENCENUMBER",
                "HM",
                "WE",
                "AHS",
                "LZ",
                "MAXDEGREE",
                "EPS",
                "HG1",
                "WW",
                "ATK",
                "MN",
                "AVGDEGREE",
                "FZ",
                "HP",
                "HG2",
                "CO",
                "PT",
                "CONNECTEDCOMPONENTS",
                "GINI",
                "SW",
                "EP",
                "KM",
                "CVM",
                "CLIQUENUMBER",
                "GD",
                "RS",
                "KS",
                "KC",
                "EDGESNUMBER",
                "DSP",
            ],
        ),
        (
            "weibull",
            [
                "MT",
                "ST1",
                "MSF",
                "SPP",
                "AD",
                "CHI2_PEARSON",
                "CVM",
                "RSB",
                "MD",
                "LT2",
                "KS",
                "LILLIE",
                "W",
                "OK",
                "LT3",
                "SB",
                "TS",
                "LS",
                "SB",
                "CQ*",
                "ST2",
                "LOS",
                "KL",
                "REJG",
            ],
        ),
        (
            "uniform",
            [
                "STEIN_U",
                "AD",
                "KUIPER",
                "CENSORED_STEIN_U",
                "CVM",
                "GREENWOOD",
                "NEYMAN",
                "LILLIE",
                "BICKEL_ROSENBLATT",
                "SHERMAN",
                "CHI2_PEARSON",
                "ZHANG",
                "KS",
                "QUESENBERRY_MILLER",
                "WATSON",
            ],
        ),
        (
            "student",
            ["KUIPER", "CHI2", "WATSON", "KS", "ZHANG_ZC", "AD", "ZHANG_ZA", "CVM", "LILLIE"],
        ),
        (
            "gamma",
            [
                "G_TEST",
                "MAXDEGREE",
                "MOR",
                "CRESSIE_READ",
                "CVM",
                "LILLIE",
                "AVGDEGREE",
                "MT",
                "PPCC",
                "AD",
                "CONNECTEDCOMPONENTS",
                "WAT",
                "CLIQUENUMBER",
                "KUI",
                "CHI2_PEARSON",
                "EDGESNUMBER",
                "INDEPENDENCENUMBER",
                "GRW",
                "KS",
            ],
        ),
    ],
)
def test_get_available_criteria_all_criteria(distribution, result):
    criteria = get_available_criteria_codes(DistributionType(distribution))
    assert Counter(criteria) == Counter(result)


@pytest.mark.parametrize(
    "distribution",
    ["normal", "exponential", "weibull", "uniform", "student", "gamma"],
)
def test_get_available_criteria_all_criteria_classes(distribution):
    criteria = get_available_criteria(DistributionType(distribution))

    assert criteria
    assert all(issubclass(criterion, AbstractGoodnessOfFitStatistic) for criterion in criteria)
    assert all(not inspect.isabstract(criterion) for criterion in criteria)
    assert all(criterion.distribution() == DistributionType(distribution) for criterion in criteria)
