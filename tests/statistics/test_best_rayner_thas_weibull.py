"""Independent moment references and invariance checks for BRT components."""

import pickle
from math import fsum, log, sqrt

import numpy as np
import pytest

from pysatl_criterion.distribution import WeibullDistributionDescriptor as Weibull
from pysatl_criterion.statistics.alternative import RightAlternative
from pysatl_criterion.statistics.goodness_of_fit.continuous import weibull as w


CLASSES = [w.BestRaynerThasSkewnessWeibullGofStatistic, w.BestRaynerThasKurtosisWeibullGofStatistic]
SAMPLE = np.array([0.14, 0.27, 0.5, 0.66, 0.84, 1.1, 1.39, 1.72, 2.03, 2.65, 3.7])


def criterion(cls):
    return cls(Weibull.DEFAULT.parse({}))


def reference(sample):
    # Scalar arithmetic, no production helper or NumPy standardized moments.
    y = [-log(float(x)) for x in sample]
    n = len(y)
    mean = fsum(y) / n
    centered = [x - mean for x in y]
    variance = fsum(x * x for x in centered) / n
    skew = fsum(x**3 for x in centered) / (n * variance * sqrt(variance))
    kurt = fsum(x**4 for x in centered) / (n * variance**2)
    return n * (skew - 1.139547) ** 2 / 20, n * (kurt - 7.55 * skew + 3.21) ** 2 / 219.72


@pytest.mark.parametrize("sample", [SAMPLE, [0.3, 0.3, 1, 2], [1e-300, 1, 1e300]])
@pytest.mark.parametrize("index,cls", list(enumerate(CLASSES)))
def test_scalar_reference(index, cls, sample):
    assert criterion(cls).execute_statistic(sample) == pytest.approx(reference(sample)[index])


@pytest.mark.parametrize("cls", CLASSES)
def test_invariance_and_no_mutation_or_mle(cls, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Moment components must not fit by MLE")

    monkeypatch.setattr(w.weibull_statistics, "_fit_logs", forbidden)
    sample = SAMPLE.copy()
    test = criterion(cls)
    value = test.execute_statistic(sample)
    for changed in (sample[::-1], sample * 1e150, sample * 1e-150, sample**0.05, sample**5):
        assert test.execute_statistic(changed) == pytest.approx(value, rel=1e-10)
    np.testing.assert_array_equal(sample, SAMPLE)
    assert isinstance(test.alternative(), RightAlternative)
    assert test.execute_statistic([0.1, 0.4, 1.3, 4]) == criterion(cls).execute_statistic(
        [0.1, 0.4, 1.3, 4]
    )
    with pytest.raises(ValueError, match="regenerate"):
        test._validate_storage_calibration()


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize(
    "sample", [[], [1], [1, 1], [0, 1], [-1, 2], [np.nan, 1], [np.inf, 1], [[1, 2]], [1j, 2]]
)
def test_invalid_samples(cls, sample):
    with pytest.raises(ValueError):
        criterion(cls).execute_statistic(sample)


@pytest.mark.parametrize("cls", CLASSES)
def test_two_observation_degeneracy(cls):
    test = criterion(cls)
    assert test.execute_statistic([1, 2]) == pytest.approx(test.execute_statistic([0.001, 10]))


def test_compatibility_names_and_codes():
    for cls, alias, code in zip(
        CLASSES, [w.ST1WeibullGofStatistic, w.ST2WeibullGofStatistic], ["ST1", "ST2"], strict=True
    ):
        assert cls is alias
        assert cls.short_code() == code
        assert pickle.loads(pickle.dumps(cls)) is cls  # noqa: S301
        assert cls.__name__ in w.__all__
        assert cls.code() == f"{code}_WEIBULL_GOODNESS_OF_FIT"


def test_log_sign_and_kurtosis_correction_matter():
    actual = [criterion(cls).execute_statistic(SAMPLE) for cls in CLASSES]
    reversed_sign = reference(1 / SAMPLE)
    assert all(abs(a - b) > 0.1 for a, b in zip(actual, reversed_sign, strict=True))
    z = -np.log(SAMPLE)
    z = (z - np.mean(z)) / np.std(z)
    plain_kurtosis = len(z) * (np.mean(z**4) - 5.4) ** 2 / 219.72
    assert actual[1] != pytest.approx(plain_kurtosis)


def test_port_pirie_source_formula():
    # Best et al. (2007), Table 1: EV observations, not Weibull observations.
    y = np.array(
        [
            4.03,
            3.96,
            4.06,
            4.11,
            3.91,
            3.83,
            3.85,
            3.71,
            4.24,
            3.72,
            3.65,
            3.93,
            3.96,
            3.96,
            4.0,
            3.88,
            3.75,
            4.06,
            4.21,
            3.66,
            4.01,
            3.63,
            4.55,
            3.74,
            3.62,
            4.08,
            3.57,
            3.79,
            3.85,
            4.33,
            4.18,
            4.25,
            3.89,
            3.88,
            4.55,
            3.8,
            3.97,
            4.11,
            3.66,
            3.75,
            4.36,
            4.05,
            3.85,
            4.11,
            4.08,
            3.96,
            4.24,
            3.86,
            3.71,
            3.9,
            3.98,
            4.22,
            3.86,
            4.18,
            3.88,
            4.69,
            3.73,
            4.21,
            3.9,
            3.94,
            3.85,
            4.37,
            4.01,
            3.78,
            4.33,
        ]
    )
    sample = np.exp(-y)
    expected = reference(sample)[0]
    actual = criterion(w.ST1WeibullGofStatistic).execute_statistic(sample)
    assert len(y) == 65
    assert expected == pytest.approx(0.5966736536522815, abs=1e-12)
    assert actual == pytest.approx(expected, abs=1e-12)
    # Section 4 reports 0.71. Do not silently substitute the raw skewness.
    g = np.mean(((y - y.mean()) / y.std()) ** 3)
    assert g == pytest.approx(0.7110707539472332, abs=1e-12)
    assert actual != pytest.approx(g, abs=0.01)

    # The kurtosis component also differs from section 4's reported 0.36.
    kurtosis_component = criterion(w.ST2WeibullGofStatistic).execute_statistic(sample)
    assert kurtosis_component == pytest.approx(reference(sample)[1], abs=1e-12)
    assert kurtosis_component == pytest.approx(0.37580214526699485, abs=1e-12)
