"""Independent references and boundary checks for frequency kernels."""

import inspect

import numpy as np
import pytest
from scipy.stats import power_divergence

from pysatl_criterion.statistics.alternative import AlternativeType
from pysatl_criterion.statistics.goodness_of_fit.continuous import frequency


CRITERIA = [
    (frequency.AbstractPearsonChi2Statistic, 1),
    (frequency.AbstractFreemanTukeyStatistic, -0.5),
    (frequency.AbstractModifiedLogLikelihoodStatistic, -1),
    (frequency.AbstractNeymanChi2Statistic, -2),
    (frequency.AbstractCressieReadStatistic, 2 / 3),
]


@pytest.mark.parametrize(("cls", "power"), CRITERIA)
@pytest.mark.parametrize("stride", [1, 2])
def test_against_scipy(cls, power, stride):
    rng = np.random.default_rng(42)
    observed = rng.uniform(1, 100, 20)[::stride]
    expected = rng.uniform(1, 100, len(observed))
    expected *= observed.sum() / expected.sum()
    reference = power_divergence(observed, expected, lambda_=power).statistic
    assert cls.do_execute_statistic(observed, expected) == pytest.approx(reference, rel=1e-12)


@pytest.mark.parametrize(("cls", "power"), CRITERIA)
def test_interface_and_numba(cls, power):
    assert issubclass(cls, frequency.AbstractFrequencyStatistic)
    assert inspect.isabstract(cls)
    assert cls.alternative(object()).type() == AlternativeType.RIGHT
    values = np.array([2.0, 5.0, 13.0])
    assert cls.do_execute_statistic(values, values) == 0.0
    assert cls.do_execute_statistic.nopython_signatures
    assert cls.do_execute_statistic.targetoptions["nopython"]


@pytest.mark.parametrize(("cls", "power"), CRITERIA)
def test_zero_observed_counts(cls, power):
    observed = np.array([0.0, 10.0])
    expected = np.array([5.0, 5.0])
    # Analytic values for the continuous extensions at a zero count.
    references = {
        1: 10.0,
        -0.5: 40.0 * (2.0 - np.sqrt(2.0)),
        -1: np.inf,
        -2: np.inf,
        2 / 3: 18.0 * (2.0 ** (2 / 3) - 1.0),
    }
    assert cls.do_execute_statistic(observed, expected) == pytest.approx(references[power])


@pytest.mark.parametrize(("cls", "power"), CRITERIA)
def test_single_category(cls, power):
    assert cls.do_execute_statistic(np.array([7.0]), np.array([7.0])) == 0.0
