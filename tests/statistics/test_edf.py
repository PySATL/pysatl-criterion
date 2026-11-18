"""References for the shared, CDF-only EDF kernels."""

import numpy as np
import pytest
from scipy import integrate, stats

from pysatl_criterion.statistics.alternative import AlternativeType
from pysatl_criterion.statistics.goodness_of_fit import edf


CLASSES = [
    edf.AbstractKSDPlusStatistic,
    edf.AbstractKSDMinusStatistic,
    edf.AbstractKSStatistic,
    edf.AbstractKuiperStatistic,
    edf.AbstractCramerVonMisesStatistic,
    edf.AbstractWatsonStatistic,
    edf.AbstractAndersonDarlingStatistic,
    edf.AbstractZhangKStatistic,
    edf.AbstractZhangAStatistic,
    edf.AbstractZhangCStatistic,
]
ZHANG = CLASSES[-3:]


@pytest.mark.parametrize("cls", CLASSES)
def test_edf_inheritance_and_nopython_execution(cls):
    assert issubclass(cls, edf.AbstractEDFStatistic)
    kernel = cls.do_execute_statistic
    assert np.isfinite(kernel(np.array([0.1, 0.3, 0.8])))
    assert kernel.nopython_signatures
    assert kernel.targetoptions["nopython"]
    assert cls.alternative(object()).type() == AlternativeType.RIGHT


@pytest.mark.parametrize("sample", [[0.4], [0.03, 0.17, 0.29, 0.52, 0.81], [0.2, 0.2, 0.7]])
def test_classical_statistics_against_independent_references(sample):
    values = np.array(sample)
    n = len(values)
    d_plus = stats.kstest(values, "uniform", alternative="greater").statistic
    d_minus = stats.kstest(values, "uniform", alternative="less").statistic
    for cls, expected in [
        (edf.AbstractKSDPlusStatistic, d_plus),
        (edf.AbstractKSDMinusStatistic, d_minus),
        (edf.AbstractKSStatistic, max(d_plus, d_minus)),
        (edf.AbstractKuiperStatistic, d_plus + d_minus),
        (edf.AbstractKuiperScaledStatistic, np.sqrt(n) * (d_plus + d_minus)),
        (
            edf.AbstractKuiperCorrectedStatistic,
            (np.sqrt(n) + 0.155 + 0.24 / np.sqrt(n)) * (d_plus + d_minus),
        ),
    ]:
        assert cls.do_execute_statistic(values) == pytest.approx(expected)

    def discrepancy(x):
        return np.count_nonzero(values <= x) / n - x

    def integral(function):
        return integrate.quad(function, 0, 1, points=values, epsabs=1e-11)[0]

    mean = integral(discrepancy)
    for cls, expected in [
        (edf.AbstractCramerVonMisesStatistic, n * integral(lambda x: discrepancy(x) ** 2)),
        (edf.AbstractWatsonStatistic, n * integral(lambda x: (discrepancy(x) - mean) ** 2)),
        (
            edf.AbstractAndersonDarlingStatistic,
            n * integral(lambda x: discrepancy(x) ** 2 / (x * (1 - x))),
        ),
    ]:
        assert cls.do_execute_statistic(values) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    ("cls", "expected"),
    [
        (edf.AbstractZhangKStatistic, 0.4851757595712566718),
        (edf.AbstractZhangAStatistic, 3.5389423702359975878),
        (edf.AbstractZhangCStatistic, 4.2917566201014437410),
    ],
)
def test_zhang_high_precision_reference(cls, expected):
    # Zhang (2002), Eqs. (3.1)-(3.3), evaluated with 50-digit Decimal arithmetic.
    values = np.array([0.03, 0.17, 0.29, 0.52, 0.81])
    assert cls.do_execute_statistic(values) == pytest.approx(expected, rel=1e-13)
    assert cls.do_execute_statistic(1 - values[::-1]) == pytest.approx(expected, rel=1e-13)


@pytest.mark.parametrize("cls", ZHANG)
@pytest.mark.parametrize("values", [[0.0], [1.0], [0, 0.5, 1]])
def test_zhang_endpoint_limits(cls, values):
    assert cls.do_execute_statistic(np.array(values, dtype=float)) == np.inf


@pytest.mark.parametrize("cls", ZHANG)
def test_zhang_empty_input(cls):
    with pytest.raises(ValueError, match="must not be empty"):
        cls.do_execute_statistic(np.array([], dtype=float))


@pytest.mark.parametrize("cls", ZHANG)
def test_zhang_extreme_probabilities_remain_finite(cls):
    values = np.array([np.nextafter(0.0, 1.0), 0.5, np.nextafter(1.0, 0.0)])
    assert np.isfinite(cls.do_execute_statistic(values))


def test_zhang_single_midpoint():
    values = np.array([0.5])
    assert edf.AbstractZhangKStatistic.do_execute_statistic(values) == pytest.approx(0)
    assert edf.AbstractZhangAStatistic.do_execute_statistic(values) == pytest.approx(4 * np.log(2))
    assert edf.AbstractZhangCStatistic.do_execute_statistic(values) == pytest.approx(0)


@pytest.mark.parametrize(
    "family",
    [
        "beta",
        "exponent",
        "exponentiated_weibull",
        "gamma",
        "hyperbolic",
        "inverse_gamma",
        "laplace",
        "log_logistic",
        "log_normal",
        "normal",
        "pareto",
        "student",
        "uniform",
        "weibull",
    ],
)
def test_each_family_exports_one_factory_liao_shimokawa(family):
    from importlib import import_module

    from pysatl_criterion.statistics.goodness_of_fit import continuous
    from pysatl_criterion.statistics.goodness_of_fit.continuous import _edf_factory

    package = import_module(f"{continuous.__name__}.{family}")
    implementation = import_module(f"{package.__name__}.{family}_statistics")
    names = [
        name
        for name, spec in implementation._EDF_STATISTICS.items()
        if spec.algorithm is edf.AbstractLiaoShimokawaStatistic
    ]
    assert len(names) == 1
    name = names[0]
    assert name.startswith("LiaoShimokawa")
    assert name in package.__all__
    criterion = getattr(package, name)
    assert criterion is getattr(continuous, name)
    assert criterion.execute_statistic.__code__.co_filename == _edf_factory.__file__
    assert issubclass(criterion, edf.AbstractLiaoShimokawaStatistic)
    assert not any(name.startswith("MinToshiyuki") for name in vars(implementation))
    assert not any(name.startswith("MinToshiyuki") for name in vars(package))
    assert not any(name.startswith("MinToshiyuki") for name in vars(continuous))
