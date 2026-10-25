from abc import ABC, abstractmethod

import numpy as np
from numba import jit
from numpy.typing import NDArray
from typing_extensions import override

from pysatl_criterion.statistics.alternative import Alternative, RightAlternative
from pysatl_criterion.statistics.statistic import AbstractStatistic


class AbstractEDFStatistic(AbstractStatistic, ABC):
    """EDF statistic computed from CDF values of a sorted sample."""

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @abstractmethod
    def do_execute_statistic(
        cdf_vals: NDArray[np.float64],
    ) -> float:
        """Compute the statistic from aligned, sorted CDF values."""
        raise NotImplementedError


class AbstractKSDPlusStatistic(AbstractEDFStatistic, ABC):
    """One-sided D+ deviation for sorted CDF values.

    References
    ----------
    Smirnov, N. (1948). Table for Estimating the Goodness of Fit of Empirical
    Distributions. Annals of Mathematical Statistics, 19(2), 279-281.
    https://doi.org/10.1214/aoms/1177730256
    """

    @staticmethod
    @override
    def short_code() -> str:
        return "KS_PLUS"

    @override
    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(cdf_vals: NDArray[np.float64]) -> float:
        n = len(cdf_vals)
        return np.max(np.arange(1.0, n + 1) / n - cdf_vals)


class AbstractKSDMinusStatistic(AbstractEDFStatistic, ABC):
    """One-sided D- deviation for sorted CDF values.

    References
    ----------
    Smirnov, N. (1948). Table for Estimating the Goodness of Fit of Empirical
    Distributions. Annals of Mathematical Statistics, 19(2), 279-281.
    https://doi.org/10.1214/aoms/1177730256
    """

    @staticmethod
    @override
    def short_code() -> str:
        return "KS_MINUS"

    @override
    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(cdf_vals: NDArray[np.float64]) -> float:
        n = len(cdf_vals)
        return np.max(cdf_vals - np.arange(0.0, n) / n)


class AbstractKSStatistic(AbstractEDFStatistic, ABC):
    """Two-sided Kolmogorov-Smirnov D statistic.

    Returns max(D+, D-), without squaring or sqrt(n) scaling.

    References
    ----------
    Zhang, J. (2002). Powerful goodness-of-fit tests based on the likelihood
    ratio. JRSS B, 64(2), 281-294, Section 2.1.
    https://doi.org/10.1111/1467-9868.00337
    """

    @staticmethod
    @override
    def short_code() -> str:
        return "KS"

    @override
    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(
        cdf_vals: NDArray[np.float64],
    ) -> float:
        """Compute the unscaled two-sided D from sorted CDF values."""
        n = len(cdf_vals)
        d_plus = np.max(np.arange(1.0, n + 1) / n - cdf_vals)
        d_minus = np.max(cdf_vals - np.arange(0.0, n) / n)
        return max(d_plus, d_minus)


class AbstractAndersonDarlingStatistic(AbstractEDFStatistic, ABC):
    """Anderson-Darling A² statistic.

    Computes -n-sum((2*i-1)*(log(u_i)+log(1-u_(n+1-i))))/n.
    Endpoints give positive infinity; no finite-sample correction is applied.

    References
    ----------
    Anderson, T. W., and Darling, D. A. (1952). Asymptotic Theory of Certain
    Goodness of Fit Criteria Based on Stochastic Processes. Annals of
    Mathematical Statistics, 23(2), 193-212.
    https://doi.org/10.1214/aoms/1177729437
    Computational form: Zhang (2002), Section 2.2.
    https://doi.org/10.1111/1467-9868.00337
    """

    @staticmethod
    @override
    def short_code() -> str:
        return "AD"

    @override
    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(
        cdf_vals: NDArray[np.float64],
    ) -> float:
        n = len(cdf_vals)
        weights = 2.0 * np.arange(1, n + 1) - 1.0

        return -n - np.sum(weights * (np.log(cdf_vals) + np.log1p(-cdf_vals[::-1]))) / n

    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_log_statistic(log_cdf, log_sf) -> float:
        """Evaluate A² without rounding upper-tail CDF probabilities to one."""
        n = len(log_cdf)
        total = 0.0
        for i in range(n):
            weight = (2.0 * i + 1.0) / n
            # Weight each tail before adding: the unweighted pair or sum can
            # overflow even when the final statistic is representable.
            total -= weight * log_cdf[i]
            total -= weight * log_sf[n - 1 - i]
        return total - n


class AbstractCramerVonMisesStatistic(AbstractEDFStatistic, ABC):
    """Cramér-von Mises W² statistic.

    Computes 1/(12*n)+sum((u_i-(i-0.5)/n)**2).

    References
    ----------
    Zhang, J. (2002). Powerful goodness-of-fit tests based on the likelihood
    ratio. JRSS B, 64(2), 281-294, Section 2.3.
    https://doi.org/10.1111/1467-9868.00337
    """

    @staticmethod
    @override
    def short_code() -> str:
        return "CVM"

    @override
    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(
        cdf_vals: NDArray[np.float64],
    ) -> float:
        n = len(cdf_vals)
        expected = (2.0 * np.arange(1, n + 1) - 1.0) / (2.0 * n)

        return 1.0 / (12.0 * n) + np.sum((cdf_vals - expected) ** 2)


class AbstractLiaoShimokawaStatistic(AbstractEDFStatistic, ABC):
    """Liao–Shimokawa L_n statistic for sorted sample CDF values.

    Computes sum(max(i/n-u_i, u_i-(i-1)/n)/sqrt(u_i*(1-u_i)))/sqrt(n).
    Endpoints give positive infinity. Applying this kernel to a specified
    uniform CDF requires uniform-null calibration; fitted Weibull/extreme-value
    critical values from the paper do not apply.

    References
    ----------
    Liao, M., & Shimokawa, T. (1999).
    A new goodness-of-fit test for type-I extreme-value and
    2-parameter Weibull distributions with estimated parameters.
    Journal of Statistical Computation and Simulation, 64(1), 23–48.
    https://doi.org/10.1080/00949659908811965
    """

    @staticmethod
    @override
    def short_code() -> str:
        return "LIAO_SHIMOKAWA"

    @override
    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(
        cdf_vals: NDArray[np.float64],
    ) -> float:
        n = len(cdf_vals)
        if n == 0:
            raise ValueError("CDF values must not be empty")

        indices = np.arange(1.0, n + 1)
        d_plus = indices / n - cdf_vals
        d_minus = cdf_vals - (indices - 1.0) / n
        deviations = np.maximum(d_plus, d_minus)

        weights = np.sqrt(cdf_vals * (1.0 - cdf_vals))

        return np.sum(deviations / weights) / np.sqrt(n)

    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_log_statistic(log_cdf, log_sf) -> float:
        """Evaluate L_n using logarithmic probabilities in both tails."""
        cdf = np.exp(log_cdf)
        n = len(cdf)
        indices = np.arange(1.0, n + 1)
        deviations = np.maximum(indices / n - cdf, cdf - (indices - 1) / n)
        return float(np.sum(np.exp(np.log(deviations) - (log_cdf + log_sf) / 2)) / np.sqrt(n))


class AbstractWatsonStatistic(AbstractEDFStatistic, ABC):
    """Watson U² statistic for CDF values of a sorted sample.

    Returns W²-n*(mean(u)-1/2)**2, evaluated as a centered sum of squares
    plus 1/(12*n) to avoid cancellation. No finite-sample correction is used.

    References
    ----------
    Lemeshko et al. (2014), Eq. (6).
    https://doi.org/10.3103/S875669901401004X
    Watson (1961), https://doi.org/10.1093/biomet/48.1-2.109; sorted CDF input.
    """

    @staticmethod
    @override
    def short_code() -> str:
        return "WATSON"

    @override
    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(
        cdf_vals: NDArray[np.float64],
    ) -> float:
        n = len(cdf_vals)
        if n == 0:
            raise ValueError("CDF values must not be empty")

        expected = (np.arange(1.0, n + 1) - 0.5) / n
        deviations = cdf_vals - expected
        centered = deviations - np.mean(deviations)

        return 1.0 / (12.0 * n) + np.sum(centered**2)


@jit(nopython=True, cache=True)
def _kuiper_statistic(cdf_vals: NDArray[np.float64]) -> float:
    """Compute unscaled D+ + D- from sorted sample CDF values."""
    n = len(cdf_vals)
    if n == 0:
        raise ValueError("CDF values must not be empty")

    indices = np.arange(1.0, n + 1)
    d_plus = np.max(indices / n - cdf_vals)
    d_minus = np.max(cdf_vals - (indices - 1.0) / n)

    return d_plus + d_minus


class AbstractKuiperStatistic(AbstractEDFStatistic, ABC):
    """Kuiper statistic computed from CDF values of a sorted sample.
    Kuiper (1960), https://doi.org/10.1016/S1385-7258(60)50006-0; unscaled D+ + D-."""

    @staticmethod
    @override
    def short_code() -> str:
        return "KUIPER"

    _kuiper_statistic = staticmethod(_kuiper_statistic)

    @override
    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(
        cdf_vals: NDArray[np.float64],
    ) -> float:
        return _kuiper_statistic(cdf_vals)


class AbstractKuiperScaledStatistic(AbstractKuiperStatistic, ABC):
    """Kuiper statistic scaled by sqrt(n).

    References
    ----------
    Lemeshko et al. (2014), section "Kuiper Test", before Eq. (2).
    https://doi.org/10.3103/S875669901401004X
    """

    @staticmethod
    @override
    def short_code() -> str:
        return "KUIPER_SCALED"

    @override
    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(
        cdf_vals: NDArray[np.float64],
    ) -> float:
        v = _kuiper_statistic(cdf_vals)
        return np.sqrt(len(cdf_vals)) * v


class AbstractKuiperCorrectedStatistic(AbstractKuiperStatistic, ABC):
    """Kuiper statistic with a finite-sample correction.

    Computes (sqrt(n) + 0.155 + 0.24 / sqrt(n)) * (D+ + D-).

    References
    ----------
    Lemeshko et al. (2014), Eq. (3).
    https://doi.org/10.3103/S875669901401004X

    Stephens (1970).
    https://doi.org/10.1111/j.2517-6161.1970.tb00821.x
    """

    @staticmethod
    @override
    def short_code() -> str:
        return "KUIPER_CORRECTED"

    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(
        cdf_vals: NDArray[np.float64],
    ) -> float:
        v = _kuiper_statistic(cdf_vals)
        sqrt_n = np.sqrt(len(cdf_vals))

        return (sqrt_n + 0.155 + 0.24 / sqrt_n) * v


class AbstractZhangKStatistic(AbstractEDFStatistic, ABC):
    """Zhang Z_K statistic for nonempty, sorted CDF values in [0, 1].

    Endpoints give positive infinity. Logarithms are evaluated separately
    to avoid overflow in tail probability ratios.

    References
    ----------
    Zhang, J. (2002). Powerful goodness-of-fit tests based on the likelihood
    ratio. JRSS B, 64(2), 281-294, Eq. (3.1).
    https://doi.org/10.1111/1467-9868.00337.
    """

    @staticmethod
    @override
    def short_code() -> str:
        return "ZHANG_ZK"

    @override
    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(cdf_vals: NDArray[np.float64]) -> float:
        n = len(cdf_vals)
        if n == 0:
            raise ValueError("CDF values must not be empty")

        ranks = np.arange(1.0, n + 1) - 0.5
        terms = ranks * (np.log(ranks / n) - np.log(cdf_vals))
        terms += (n - ranks) * (np.log((n - ranks) / n) - np.log1p(-cdf_vals))
        return np.max(terms)

    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_log_statistic(log_cdf, log_sf) -> float:
        """Evaluate Z_K without reconstructing rounded tail probabilities."""
        n = len(log_cdf)
        ranks = np.arange(1.0, n + 1) - 0.5
        terms = ranks * (np.log(ranks / n) - log_cdf)
        terms += (n - ranks) * (np.log((n - ranks) / n) - log_sf)
        return float(np.max(terms))


class AbstractZhangAStatistic(AbstractEDFStatistic, ABC):
    """Zhang Z_A statistic for nonempty, sorted CDF values in [0, 1].

    Endpoints give positive infinity. Logarithms are evaluated separately
    to avoid overflow in tail probability ratios.

    References
    ----------
    Zhang, J. (2002). Powerful goodness-of-fit tests based on the likelihood
    ratio. JRSS B, 64(2), 281-294, Eq. (3.2).
    https://doi.org/10.1111/1467-9868.00337.
    """

    @staticmethod
    @override
    def short_code() -> str:
        return "ZHANG_ZA"

    @override
    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(cdf_vals: NDArray[np.float64]) -> float:
        n = len(cdf_vals)
        if n == 0:
            raise ValueError("CDF values must not be empty")

        ranks = np.arange(1.0, n + 1) - 0.5
        return -np.sum(np.log(cdf_vals) / (n - ranks) + np.log1p(-cdf_vals) / ranks)

    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_log_statistic(log_cdf, log_sf) -> float:
        """Evaluate Z_A directly from logarithmic tail probabilities."""
        n = len(log_cdf)
        ranks = np.arange(1.0, n + 1) - 0.5
        return float(-np.sum(log_cdf / (n - ranks) + log_sf / ranks))


class AbstractZhangCStatistic(AbstractEDFStatistic, ABC):
    """Zhang Z_C statistic for nonempty, sorted CDF values in [0, 1].

    Endpoints give positive infinity. Logarithms are evaluated separately
    to avoid overflow in tail probability ratios.

    References
    ----------
    Zhang, J. (2002). Powerful goodness-of-fit tests based on the likelihood
    ratio. JRSS B, 64(2), 281-294, Eq. (3.3).
    https://doi.org/10.1111/1467-9868.00337.
    """

    @staticmethod
    @override
    def short_code() -> str:
        return "ZHANG_ZC"

    @override
    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(cdf_vals: NDArray[np.float64]) -> float:
        n = len(cdf_vals)
        if n == 0:
            raise ValueError("CDF values must not be empty")

        indices = np.arange(1.0, n + 1)
        expected_log_odds = np.log(n - indices + 0.25) - np.log(indices - 0.75)
        log_odds = np.log1p(-cdf_vals) - np.log(cdf_vals)
        return np.sum((log_odds - expected_log_odds) ** 2)

    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_log_statistic(log_cdf, log_sf) -> float:
        """Evaluate Z_C using the difference of logarithms for odds."""
        n = len(log_cdf)
        indices = np.arange(1.0, n + 1)
        expected_log_odds = np.log(n - indices + 0.25) - np.log(indices - 0.75)
        return float(np.sum((log_sf - log_cdf - expected_log_odds) ** 2))
