from __future__ import annotations

from abc import ABC

import numpy as np
import scipy.stats as scipy_stats
from numba import njit
from typing_extensions import override

from pysatl_criterion import DistributionType
from pysatl_criterion.statistics import AbstractGoodnessOfFitStatistic
from pysatl_criterion.statistics.alternative import Alternative, AlternativeType, RightAlternative
from pysatl_criterion.statistics.goodness_of_fit.common import (
    ADStatistic,
    CrammerVonMisesStatistic,
    KSStatistic,
)
from pysatl_criterion.statistics.hypothesis import GoodnessOfFitHypothesis


class AbstractHyperbolicGofStatistic(AbstractGoodnessOfFitStatistic, ABC):
    """Base class for hyperbolic distribution goodness-of-fit statistics.

    The classical hyperbolic distribution is the generalized hyperbolic
    distribution with tail parameter ``p = 1``. The parameterization follows
    Barndorff-Nielsen (1978), *Hyperbolic Distributions and Distributions on
    Hyperbolae*, Scandinavian Journal of Statistics, 5(3), 151--157.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.0,
        delta: float = 1.0,
        mu: float = 0.0,
    ) -> None:
        """Initialize a hyperbolic goodness-of-fit statistic.

        :param alpha: positive shape parameter satisfying ``|beta| < alpha``.
        :param beta: skewness parameter satisfying ``|beta| < alpha``.
        :param delta: positive scale parameter.
        :param mu: location parameter.
        :raises ValueError: if any parameter is non-finite or violates its constraint.
        """
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("Alpha must be finite and positive.")
        if not np.isfinite(beta) or abs(beta) >= alpha:
            raise ValueError("Beta must be finite and satisfy abs(beta) < alpha.")
        if not np.isfinite(delta) or delta <= 0:
            raise ValueError("Delta must be finite and positive.")
        if not np.isfinite(mu):
            raise ValueError("Mu must be finite.")

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.delta = float(delta)
        self.mu = float(mu)

    @override
    def hypothesis(self) -> GoodnessOfFitHypothesis:
        """Return the parameters of the reference hyperbolic distribution.

        :return: hypothesis containing ``alpha``, ``beta``, ``delta``, and ``mu``.
        """
        return GoodnessOfFitHypothesis(
            {
                "alpha": self.alpha,
                "beta": self.beta,
                "delta": self.delta,
                "mu": self.mu,
            }
        )

    @staticmethod
    @override
    def distribution() -> DistributionType:
        """Return the hyperbolic distribution type.

        :return: hyperbolic distribution enum member.
        """
        return DistributionType.HYPERBOLIC

    @staticmethod
    @override
    def code() -> str:
        """Return the base identifier for hyperbolic statistics.

        :return: ``HYPERBOLIC_GOODNESS_OF_FIT``.
        """
        return f"HYPERBOLIC_{AbstractGoodnessOfFitStatistic.code()}"

    @staticmethod
    def _prepare_sample(rvs) -> np.ndarray:
        """Convert, validate, and sort a sample.

        :param rvs: one-dimensional observations.
        :return: sorted ``float64`` NumPy array.
        :raises ValueError: if the sample is not one-dimensional or is empty.
        """
        sample = np.asarray(rvs, dtype=np.float64)
        if sample.ndim != 1:
            raise ValueError("Sample must be one-dimensional.")
        if sample.size == 0:
            raise ValueError("Sample must contain at least one observation.")
        return np.sort(sample)

    def _cdf(self, sorted_rvs: np.ndarray) -> np.ndarray:
        """Evaluate the hyperbolic CDF for a sorted sample.

        :param sorted_rvs: sample sorted in ascending order.
        :return: CDF values in sample order.
        """
        return np.asarray(
            scipy_stats.genhyperbolic.cdf(
                sorted_rvs,
                p=1.0,
                a=self.alpha * self.delta,
                b=self.beta * self.delta,
                loc=self.mu,
                scale=self.delta,
            ),
            dtype=np.float64,
        )

    def _log_probabilities(self, sorted_rvs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate stable log-CDF and log-survival probabilities.

        :param sorted_rvs: sample sorted in ascending order.
        :return: pair containing log-CDF and log-survival arrays.
        """
        parameters = {
            "p": 1.0,
            "a": self.alpha * self.delta,
            "b": self.beta * self.delta,
            "loc": self.mu,
            "scale": self.delta,
        }
        log_cdf = scipy_stats.genhyperbolic.logcdf(sorted_rvs, **parameters)
        log_sf = scipy_stats.genhyperbolic.logsf(sorted_rvs, **parameters)
        return (
            np.asarray(log_cdf, dtype=np.float64),
            np.asarray(log_sf, dtype=np.float64),
        )


class KolmogorovSmirnovHyperbolicGofStatistic(AbstractHyperbolicGofStatistic, KSStatistic):
    """Kolmogorov--Smirnov EDF statistic for a hyperbolic distribution.

    The statistic is the largest signed or two-sided deviation between the
    empirical and reference CDFs. See Kolmogorov (1933), *Sulla determinazione
    empirica di una legge di distribuzione*.
    """

    def __init__(
        self,
        alternative_type: AlternativeType = AlternativeType.TWO_TAILED,
        mode: str = "auto",
        alpha: float = 1.0,
        beta: float = 0.0,
        delta: float = 1.0,
        mu: float = 0.0,
    ) -> None:
        """Initialize the Kolmogorov--Smirnov statistic.

        :param alternative_type: left, right, or two-sided alternative.
        :param mode: calculation mode retained for compatibility with ``KSStatistic``.
        :param alpha: positive shape parameter satisfying ``|beta| < alpha``.
        :param beta: skewness parameter satisfying ``|beta| < alpha``.
        :param delta: positive scale parameter.
        :param mu: location parameter.
        """
        AbstractHyperbolicGofStatistic.__init__(self, alpha, beta, delta, mu)
        KSStatistic.__init__(self, alternative_type=alternative_type, mode=mode)

    @staticmethod
    @override
    def short_code() -> str:
        """Return the short statistic identifier.

        :return: ``KS``.
        """
        return "KS"

    @staticmethod
    @override
    def code() -> str:
        """Return the unique statistic identifier.

        :return: ``KS_HYPERBOLIC_GOODNESS_OF_FIT``.
        """
        short_code = KolmogorovSmirnovHyperbolicGofStatistic.short_code()
        return f"{short_code}_{AbstractHyperbolicGofStatistic.code()}"

    @staticmethod
    @njit
    def _calculate_statistic(
        cdf_values: np.ndarray,
        alternative_code: int,
    ) -> float:  # pragma: no cover
        n = len(cdf_values)
        d_plus = 0.0
        d_minus = 0.0

        for i in range(n):
            current_cdf = cdf_values[i]
            if np.isnan(current_cdf):
                return np.nan
            d_plus = max(d_plus, (i + 1.0) / n - current_cdf)
            d_minus = max(d_minus, current_cdf - i / n)

        if alternative_code > 0:
            return d_plus
        if alternative_code < 0:
            return d_minus
        return max(d_plus, d_minus)

    @override
    def execute_statistic(self, rvs, **kwargs) -> float:
        """Calculate the Kolmogorov--Smirnov statistic.

        :param rvs: one-dimensional sample.
        :return: selected one- or two-sided KS statistic.
        """
        sorted_rvs = self._prepare_sample(rvs)
        return self.do_execute_statistic(sorted_rvs, self._cdf(sorted_rvs))

    def do_execute_statistic(self, rvs, cdf_vals=None) -> float:
        """Calculate the statistic from precomputed CDF values.

        :param rvs: sample sorted in ascending order.
        :param cdf_vals: reference CDF values in sorted sample order.
        :return: selected KS statistic.
        :raises ValueError: if CDF values are not provided.
        """
        if cdf_vals is None:
            raise ValueError("CDF values are required.")
        alternative_codes = {
            AlternativeType.TWO_TAILED: 0,
            AlternativeType.RIGHT: 1,
            AlternativeType.LEFT: -1,
        }
        alternative_code = alternative_codes[self.alternative_type]
        return float(self._calculate_statistic(cdf_vals, alternative_code))


class CramerVonMisesHyperbolicGofStatistic(
    AbstractHyperbolicGofStatistic, CrammerVonMisesStatistic
):
    """Cramer--von Mises EDF statistic for a hyperbolic distribution.

    The statistic integrates the squared difference between empirical and
    reference CDFs. See Cramér (1928), *On the composition of elementary errors*.
    """

    @staticmethod
    @override
    def short_code() -> str:
        """Return the short statistic identifier.

        :return: ``CVM``.
        """
        return "CVM"

    @staticmethod
    @override
    def code() -> str:
        """Return the unique statistic identifier.

        :return: ``CVM_HYPERBOLIC_GOODNESS_OF_FIT``.
        """
        short_code = CramerVonMisesHyperbolicGofStatistic.short_code()
        return f"{short_code}_{AbstractHyperbolicGofStatistic.code()}"

    @staticmethod
    @njit
    def _calculate_statistic(cdf_values: np.ndarray) -> float:  # pragma: no cover
        n = len(cdf_values)
        total = 1.0 / (12.0 * n)

        for i in range(n):
            expected_cdf = (2.0 * i + 1.0) / (2.0 * n)
            difference = expected_cdf - cdf_values[i]
            total += difference * difference

        return total

    @override
    def execute_statistic(self, rvs, **kwargs) -> float:
        """Calculate the Cramer--von Mises statistic.

        :param rvs: one-dimensional sample.
        :return: Cramer--von Mises statistic.
        """
        sorted_rvs = self._prepare_sample(rvs)
        return self.do_execute_statistic(sorted_rvs, self._cdf(sorted_rvs))

    def do_execute_statistic(self, rvs, cdf_vals) -> float:
        """Calculate the statistic from precomputed CDF values.

        :param rvs: sample sorted in ascending order.
        :param cdf_vals: reference CDF values in sorted sample order.
        :return: Cramer--von Mises statistic.
        """
        return float(self._calculate_statistic(cdf_vals))


class AndersonDarlingHyperbolicGofStatistic(AbstractHyperbolicGofStatistic, ADStatistic):
    """Anderson--Darling EDF statistic for a hyperbolic distribution.

    Tail differences receive more weight than in the Cramer--von Mises test.
    See Anderson and Darling (1952), *Asymptotic theory of certain goodness of
    fit criteria based on stochastic processes*, Annals of Mathematical Statistics.
    """

    @staticmethod
    @override
    def short_code() -> str:
        """Return the short statistic identifier.

        :return: ``AD``.
        """
        return "AD"

    @staticmethod
    @override
    def code() -> str:
        """Return the unique statistic identifier.

        :return: ``AD_HYPERBOLIC_GOODNESS_OF_FIT``.
        """
        short_code = AndersonDarlingHyperbolicGofStatistic.short_code()
        return f"{short_code}_{AbstractHyperbolicGofStatistic.code()}"

    @staticmethod
    @njit
    def _calculate_statistic(
        log_cdf: np.ndarray,
        log_sf: np.ndarray,
    ) -> float:  # pragma: no cover
        n = len(log_cdf)
        total = 0.0

        for i in range(n):
            total += (2.0 * i + 1.0) / n * (log_cdf[i] + log_sf[n - i - 1])

        return -n - total

    @override
    def execute_statistic(self, rvs, **kwargs) -> float:
        """Calculate the Anderson--Darling statistic.

        :param rvs: one-dimensional sample.
        :return: Anderson--Darling statistic.
        """
        sorted_rvs = self._prepare_sample(rvs)
        log_cdf, log_sf = self._log_probabilities(sorted_rvs)
        return self.do_execute_statistic(sorted_rvs, log_cdf=log_cdf, log_sf=log_sf)

    def do_execute_statistic(self, rvs, log_cdf=None, log_sf=None) -> float:
        """Calculate the statistic from stable log probabilities.

        :param rvs: sample sorted in ascending order.
        :param log_cdf: log-CDF values in sorted sample order.
        :param log_sf: log-survival values in sorted sample order.
        :return: Anderson--Darling statistic.
        :raises ValueError: if either log-probability array is not provided.
        """
        if log_cdf is None or log_sf is None:
            raise ValueError("Log-CDF and log-survival values are required.")
        return float(self._calculate_statistic(log_cdf, log_sf))


class KuiperHyperbolicGofStatistic(AbstractHyperbolicGofStatistic):
    """Kuiper EDF statistic for a hyperbolic distribution.

    The statistic adds the largest positive and negative EDF deviations. See
    Kuiper (1960), *Tests concerning random points on a circle*.
    """

    @override
    def alternative(self) -> Alternative:
        """Return the right-tailed alternative used for the statistic.

        :return: right-tailed alternative.
        """
        return RightAlternative()

    @staticmethod
    @override
    def short_code() -> str:
        """Return the short statistic identifier.

        :return: ``KUI``.
        """
        return "KUI"

    @staticmethod
    @override
    def code() -> str:
        """Return the unique statistic identifier.

        :return: ``KUI_HYPERBOLIC_GOODNESS_OF_FIT``.
        """
        short_code = KuiperHyperbolicGofStatistic.short_code()
        return f"{short_code}_{AbstractHyperbolicGofStatistic.code()}"

    @staticmethod
    @njit
    def _calculate_statistic(cdf_values: np.ndarray) -> float:  # pragma: no cover
        n = len(cdf_values)
        d_plus = 0.0
        d_minus = 0.0

        for i in range(n):
            current_cdf = cdf_values[i]
            if np.isnan(current_cdf):
                return np.nan
            d_plus = max(d_plus, (i + 1.0) / n - current_cdf)
            d_minus = max(d_minus, current_cdf - i / n)

        return d_plus + d_minus

    @override
    def execute_statistic(self, rvs, **kwargs) -> float:
        """Calculate the Kuiper statistic.

        :param rvs: one-dimensional sample.
        :return: Kuiper statistic ``D+ + D-``.
        """
        sorted_rvs = self._prepare_sample(rvs)
        return self.do_execute_statistic(self._cdf(sorted_rvs))

    def do_execute_statistic(self, cdf_values: np.ndarray) -> float:
        """Calculate the statistic from precomputed CDF values.

        :param cdf_values: reference CDF values in sorted sample order.
        :return: Kuiper statistic.
        """
        return float(self._calculate_statistic(cdf_values))


class WatsonHyperbolicGofStatistic(AbstractHyperbolicGofStatistic):
    """Watson centered Cramer--von Mises statistic for a hyperbolic distribution.

    The squared mean probability deviation is removed from the Cramer--von
    Mises statistic. See Watson (1961), *Goodness-of-fit tests on a circle*.
    """

    @override
    def alternative(self) -> Alternative:
        """Return the right-tailed alternative used for the statistic.

        :return: right-tailed alternative.
        """
        return RightAlternative()

    @staticmethod
    @override
    def short_code() -> str:
        """Return the short statistic identifier.

        :return: ``WAT``.
        """
        return "WAT"

    @staticmethod
    @override
    def code() -> str:
        """Return the unique statistic identifier.

        :return: ``WAT_HYPERBOLIC_GOODNESS_OF_FIT``.
        """
        short_code = WatsonHyperbolicGofStatistic.short_code()
        return f"{short_code}_{AbstractHyperbolicGofStatistic.code()}"

    @staticmethod
    @njit
    def _calculate_statistic(cdf_values: np.ndarray) -> float:  # pragma: no cover
        n = len(cdf_values)
        total = 1.0 / (12.0 * n)
        cdf_sum = 0.0

        for i in range(n):
            current_cdf = cdf_values[i]
            expected_cdf = (2.0 * i + 1.0) / (2.0 * n)
            difference = current_cdf - expected_cdf
            total += difference * difference
            cdf_sum += current_cdf

        mean_adjustment = cdf_sum - n / 2.0
        return total - mean_adjustment * mean_adjustment / n

    @override
    def execute_statistic(self, rvs, **kwargs) -> float:
        """Calculate the Watson statistic.

        :param rvs: one-dimensional sample.
        :return: Watson ``U²`` statistic.
        """
        sorted_rvs = self._prepare_sample(rvs)
        return self.do_execute_statistic(self._cdf(sorted_rvs))

    def do_execute_statistic(self, cdf_values: np.ndarray) -> float:
        """Calculate the statistic from precomputed CDF values.

        :param cdf_values: reference CDF values in sorted sample order.
        :return: Watson statistic.
        """
        return float(self._calculate_statistic(cdf_values))
