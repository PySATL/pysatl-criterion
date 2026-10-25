from abc import ABC, abstractmethod

import numpy as np
from numba import jit
from numpy.typing import NDArray
from typing_extensions import override

from pysatl_criterion.statistics.alternative import Alternative, RightAlternative
from pysatl_criterion.statistics.statistic import AbstractStatistic


class AbstractFrequencyStatistic(AbstractStatistic, ABC):
    """Base class for frequency-based goodness-of-fit statistics.

    Inputs must be one-dimensional arrays of equal, nonzero length.

    observed:
        Finite, nonnegative observed counts.
    expected:
        Finite, strictly positive expected counts.

    Both arrays must describe the same categories in the same order,
    with equal total counts.
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @abstractmethod
    def do_execute_statistic(
        observed: NDArray[np.float64],
        expected: NDArray[np.float64],
    ) -> float:
        """Compute a statistic from already validated frequencies."""
        raise NotImplementedError


class AbstractPearsonChi2Statistic(AbstractFrequencyStatistic, ABC):
    """Pearson chi-square: sum((observed - expected)**2 / expected).

    References
    ----------
    Pearson, K. (1900). On the criterion that a given system of deviations
    from the probable in the case of a correlated system of variables is
    such that it can be reasonably supposed to have arisen from random sampling.
    Philosophical Magazine, 50(302), 157-175.
    https://doi.org/10.1080/14786440009463897
    """

    @staticmethod
    @override
    def short_code() -> str:
        return "CHI2_PEARSON"

    @override
    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(observed: NDArray[np.float64], expected: NDArray[np.float64]) -> float:
        result = 0.0
        for i in range(len(observed)):
            difference = observed[i] - expected[i]
            result += difference * difference / expected[i]
        return result


class AbstractFreemanTukeyStatistic(AbstractFrequencyStatistic, ABC):
    """Freeman-Tukey: 4 * sum((sqrt(observed) - sqrt(expected))**2).

    Zero observed counts contribute 4 * expected to the statistic.

    References
    ----------
    Cressie, N. and Read, T. R. C. (1984). Multinomial Goodness-of-Fit Tests.
    JRSS B, 46(3), 440-464; power divergence with lambda = -1/2.
    https://doi.org/10.1111/j.2517-6161.1984.tb01318.x
    """

    @staticmethod
    @override
    def short_code() -> str:
        return "FREEMAN_TUKEY"

    @override
    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(observed: NDArray[np.float64], expected: NDArray[np.float64]) -> float:
        result = 0.0
        for i in range(len(observed)):
            difference = np.sqrt(observed[i]) - np.sqrt(expected[i])
            result += difference * difference
        return 4.0 * result


class AbstractModifiedLogLikelihoodStatistic(AbstractFrequencyStatistic, ABC):
    """Modified log likelihood: 2 * sum(expected * log(expected / observed)).

    Returns positive infinity if any observed count is zero.

    References
    ----------
    Cressie, N. and Read, T. R. C. (1984). Multinomial Goodness-of-Fit Tests.
    JRSS B, 46(3), 440-464; power divergence with lambda = -1.
    https://doi.org/10.1111/j.2517-6161.1984.tb01318.x
    """

    @staticmethod
    @override
    def short_code() -> str:
        return "MODIFIED_LOG_LIKELIHOOD"

    @override
    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(observed: NDArray[np.float64], expected: NDArray[np.float64]) -> float:
        result = 0.0
        for i in range(len(observed)):
            if observed[i] == 0.0:
                return np.inf
            result += expected[i] * (np.log(expected[i]) - np.log(observed[i]))
        return 2.0 * result


class AbstractNeymanChi2Statistic(AbstractFrequencyStatistic, ABC):
    """Neyman modified chi-square: sum((observed - expected)**2 / observed).

    Returns positive infinity if any observed count is zero.

    References
    ----------
    Cressie, N. and Read, T. R. C. (1984). Multinomial Goodness-of-Fit Tests.
    JRSS B, 46(3), 440-464; power divergence with lambda = -2.
    https://doi.org/10.1111/j.2517-6161.1984.tb01318.x
    """

    @staticmethod
    @override
    def short_code() -> str:
        return "CHI2_NEYMAN"

    @override
    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(observed: NDArray[np.float64], expected: NDArray[np.float64]) -> float:
        result = 0.0
        for i in range(len(observed)):
            if observed[i] == 0.0:
                return np.inf
            difference = observed[i] - expected[i]
            result += difference * difference / observed[i]
        return result


class AbstractCressieReadStatistic(AbstractFrequencyStatistic, ABC):
    """Cressie-Read power divergence with lambda = 2/3.

    Computes 9/5 * sum(observed * ((observed / expected)**(2/3) - 1)).
    Zero observed counts contribute zero, by continuity.

    References
    ----------
    Cressie, N. and Read, T. R. C. (1984). Multinomial Goodness-of-Fit Tests.
    JRSS B, 46(3), 440-464.
    https://doi.org/10.1111/j.2517-6161.1984.tb01318.x
    """

    @staticmethod
    @override
    def short_code() -> str:
        return "CRESSIE_READ"

    @override
    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(observed: NDArray[np.float64], expected: NDArray[np.float64]) -> float:
        result = 0.0
        for i in range(len(observed)):
            if observed[i] > 0.0:
                result += observed[i] * ((observed[i] / expected[i]) ** (2.0 / 3.0) - 1.0)
        return 1.8 * result
