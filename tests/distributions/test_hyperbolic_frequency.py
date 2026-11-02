"""Independent checks of all generated hyperbolic frequency criteria."""

import pickle
from itertools import pairwise

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import k1
from scipy.stats import power_divergence

from pysatl_criterion.statistics.alternative import RightAlternative
from pysatl_criterion.statistics.goodness_of_fit.binning import EqualProbabilityBins, ExplicitBins
from pysatl_criterion.statistics.goodness_of_fit.continuous import hyperbolic
from tests.parameter_cases import parameters_for


CASES = [
    (hyperbolic.Chi2PearsonHyperbolicGofStatistic, 1),
    (hyperbolic.FreemanTukeyHyperbolicGofStatistic, -0.5),
    (hyperbolic.ModifiedLogLikelihoodHyperbolicGofStatistic, -1),
    (hyperbolic.NeymanChi2HyperbolicGofStatistic, -2),
    (hyperbolic.CressieReadHyperbolicGofStatistic, 2 / 3),
]


@pytest.mark.parametrize("cls, power", CASES)
@pytest.mark.parametrize("sample", [[-2, -1, -0.5, 1], [0.2, 0.2, 2], [-100, 100]])
def test_equal_probability_cells(cls, power, sample):
    # A symmetric continuous hyperbolic law has median zero, independently of SciPy.
    parameters = parameters_for(cls, alpha=1, beta=0, delta=1, mu=0)
    criterion = cls(parameters, binning=EqualProbabilityBins(count=2))
    observed = np.array([sum(x < 0 for x in sample), sum(x >= 0 for x in sample)], dtype=float)
    expected = np.full(2, len(sample) / 2)
    if np.any(observed == 0) and power in {-1, -2}:
        reference = np.inf
    elif power == -0.5:
        reference = 4 * np.sum((np.sqrt(observed) - np.sqrt(expected)) ** 2)
    else:
        reference = power_divergence(observed, expected, lambda_=power).statistic
    original = np.asarray(sample).copy()
    assert criterion.execute_statistic(original) == pytest.approx(reference)
    np.testing.assert_array_equal(original, sample)
    assert isinstance(criterion.alternative(), RightAlternative)
    kernel = hyperbolic.hyperbolic_statistics._FREQUENCY_STATISTICS[cls.__name__].algorithm
    assert cls.do_execute_statistic is kernel.do_execute_statistic
    assert kernel.do_execute_statistic.nopython_signatures


@pytest.mark.parametrize("cls, power", CASES)
def test_explicit_bins_against_integrated_density(cls, power):
    alpha, beta, delta, mu = 2, -0.4, 0.8, 0.2
    gamma = np.sqrt(alpha**2 - beta**2)
    normalizer = gamma / (2 * alpha * delta * k1(delta * gamma))

    def density(x):
        y = x - mu
        return normalizer * np.exp(-alpha * np.hypot(delta, y) + beta * y)

    edges = [-np.inf, -1, 0.5, np.inf]
    sample = [-2, -1.2, -1, 0, 0.5, 1, 2]
    observed = np.array([2, 2, 3], dtype=float)  # Internal edges belong to the cell on the right.
    expected = len(sample) * np.array(
        [quad(density, left, right, epsabs=1e-12)[0] for left, right in pairwise(edges)]
    )
    reference = power_divergence(observed, expected, lambda_=power).statistic
    criterion = cls(
        parameters_for(cls, alpha=alpha, beta=beta, delta=delta, mu=mu),
        binning=ExplicitBins(tuple(edges)),
    )
    assert criterion.execute_statistic(sample) == pytest.approx(reference, rel=1e-9, abs=1e-11)


@pytest.mark.parametrize("cls, power", CASES)
def test_configuration_pickle_and_default(cls, power):
    criterion = cls(parameters_for(cls), binning=EqualProbabilityBins(count=3))
    restored = pickle.loads(pickle.dumps(criterion))  # noqa: S301 - own roundtrip
    assert restored.configuration() == criterion.configuration()
    assert restored.execute_statistic([-2, -0.3, 0, 1, 2]) == pytest.approx(
        criterion.execute_statistic([-2, -0.3, 0, 1, 2])
    )
    assert (
        criterion.configuration()
        != cls(parameters_for(cls), binning=EqualProbabilityBins(count=2)).configuration()
    )
    assert isinstance(cls(parameters_for(cls)).binning, EqualProbabilityBins)
    assert "supported_hypotheses" in cls.__dict__


@pytest.mark.parametrize("cls, power", CASES)
@pytest.mark.parametrize("sample", [[], [[1, 2]], [np.nan], [np.inf], [1j]])
def test_invalid_samples(cls, power, sample):
    with pytest.raises(ValueError):
        cls(parameters_for(cls)).execute_statistic(sample)
