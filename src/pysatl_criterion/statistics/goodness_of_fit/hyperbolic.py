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
        alpha, beta, delta, mu = (
            self._real_scalar(name, value)
            for name, value in zip(
                ("alpha", "beta", "delta", "mu"), (alpha, beta, delta, mu), strict=True
            )
        )
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
        a, b = self.alpha * self.delta, self.beta * self.delta
        if not np.isfinite(a) or not np.isfinite(b) or a <= abs(b):
            raise ValueError(
                "Scaled parameters must be finite and satisfy alpha*delta > |beta*delta|."
            )

    @staticmethod
    def _real_scalar(name, value) -> float:
        array = np.asarray(value)
        if array.ndim != 0 or array.dtype.kind not in "iuf":
            raise ValueError(f"{name} must be a real numeric scalar.")
        return float(array)

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
        :raises ValueError: if the sample is empty, nonreal, nonfinite, or not 1D.
        """
        raw = np.asarray(rvs)
        if np.iscomplexobj(raw):
            raise ValueError("Sample must contain real observations.")
        try:
            sample = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("Sample must contain real finite observations.") from error
        if sample.ndim != 1:
            raise ValueError("Sample must be one-dimensional.")
        if sample.size == 0:
            raise ValueError("Sample must contain at least one observation.")
        if not np.all(np.isfinite(sample)):
            raise ValueError("Sample must contain only finite observations.")
        return np.sort(sample)

    def _cdf(self, sorted_rvs: np.ndarray) -> np.ndarray:
        """Evaluate the hyperbolic CDF for a sorted sample.

        :param sorted_rvs: sample sorted in ascending order.
        :return: CDF values in sample order.
        """
        values = np.asarray(
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
        if (
            not np.all(np.isfinite(values))
            or np.any(values < 0)
            or np.any(values > 1)
            or np.any(np.diff(values) < 0)
        ):
            raise FloatingPointError("Hyperbolic CDF evaluation failed.")
        return values

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
        if (
            not np.all(np.isfinite(log_cdf))
            or not np.all(np.isfinite(log_sf))
            or np.any(log_cdf > 0)
            or np.any(log_sf > 0)
        ):
            raise FloatingPointError("Hyperbolic log-tail evaluation failed or underflowed.")
        return (
            np.asarray(log_cdf, dtype=np.float64),
            np.asarray(log_sf, dtype=np.float64),
        )


class KolmogorovSmirnovHyperbolicGofStatistic(AbstractHyperbolicGofStatistic, KSStatistic):
    """Kolmogorov--Smirnov distance for a fixed hyperbolic CDF.

    Parameters
    ----------
    alternative_type : AlternativeType, default: AlternativeType.TWO_TAILED
        CDF deviation: TWO_TAILED selects D, RIGHT D+, and LEFT D-.
    mode : str, default: "auto"
        Compatibility setting; does not calculate p-values or change the result.
    alpha : float, default: 1.0
        Fixed shape parameter, finite and strictly greater than ``abs(beta)``.
    beta : float, default: 0.0
        Fixed finite skewness parameter.
    delta : float, default: 1.0
        Fixed finite positive scale parameter.
    mu : float, default: 0.0
        Fixed finite location parameter.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute one scalar statistic without changing the input or fitting.
    hypothesis()
        Return all four fixed distribution parameters.
    alternative()
        Return the right critical tail.

    Notes
    -----
    The null is an iid sample on the real line from the fully specified
    classical hyperbolic law. No parameters are estimated. Its CDF is evaluated
    with ``genhyperbolic(p=1, a=alpha*delta, b=beta*delta, loc=mu, scale=delta)``.
    Scaled shape parameters must remain finite and satisfy ``a > abs(b)``.
    Write ``u_i = F(x_(i))`` for sorted observations, ``i=1,...,n``.
    Large statistics reject the null. Under the continuous null the probability
    integral transform makes the null law independent of all four parameters;
    it still depends on sample size. Ties and constant samples are accepted,
    but a discrete or rounded observation model needs its own calibration.
    The cited work concerns general EDF statistics, not a fitted hyperbolic test.
    Parameter fitting requires a different calibration and is not supported.
    The built-in Monte Carlo resolver currently has no hyperbolic generator;
    simulate externally and call ``execute_statistic`` for each sample.

    ``D+ = max(i/n-u_i)``, ``D- = max(u_i-(i-1)/n)``; return
    ``max(D+, D-)`` (TWO_TAILED), ``D+`` (RIGHT), or ``D-`` (LEFT).
    All choices use a right critical tail. Storage keys omit the CDF direction,
    so stored calibration is rejected for one-sided distances; calibrate those
    externally. ``mode`` does not affect this scalar statistic.

    References
    ----------
    .. [1] Smirnov, N. (1948). Table for Estimating the Goodness of Fit of
       Empirical Distributions. Ann. Math. Statist. 19, 279-281.
       https://doi.org/10.1214/aoms/1177730256

    Examples
    --------
    >>> statistic = KolmogorovSmirnovHyperbolicGofStatistic(alpha=1.5, beta=0.2)
    >>> result = statistic.execute_statistic([-1.0, 0.0, 0.5, 2.0])
    >>> isinstance(result, float) and result >= 0
    True
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
        if alternative_type not in tuple(AlternativeType):
            raise ValueError("alternative_type must be an AlternativeType member.")
        KSStatistic.__init__(self, alternative_type=alternative_type, mode=mode)

    def _validate_storage_calibration(self):
        if self.alternative_type != AlternativeType.TWO_TAILED:
            raise ValueError("Stored KS calibration does not encode the CDF direction")

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
        """Compute the Kolmogorov--Smirnov distance.

        Parameters
        ----------
        rvs : array_like of float, shape (n,)
            Finite real observations, n >= 1. Ties and constants are allowed.
            A sorted copy is used; the input is not modified.
        **kwargs : dict
            Unused compatibility arguments; no per-call settings are supported.

        Returns
        -------
        float
            Uncorrected KS statistic; large values reject the null.

        Raises
        ------
        ValueError
            If the sample is empty, nonreal, nonfinite, or not one-dimensional.
        FloatingPointError
            If SciPy returns invalid probabilities.

        Notes
        -----
        The statistic is mathematically finite for finite observations.
        Numerical tail accuracy is limited by SciPy's integration routines.
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
    """Cramer--von Mises statistic for a fixed hyperbolic CDF.

    Parameters
    ----------
    alpha : float, default: 1.0
        Fixed shape parameter, finite and strictly greater than ``abs(beta)``.
    beta : float, default: 0.0
        Fixed finite skewness parameter.
    delta : float, default: 1.0
        Fixed finite positive scale parameter.
    mu : float, default: 0.0
        Fixed finite location parameter.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute one scalar statistic without changing the input or fitting.
    hypothesis()
        Return all four fixed distribution parameters.
    alternative()
        Return the right critical tail.

    Notes
    -----
    The null is an iid sample on the real line from the fully specified
    classical hyperbolic law. No parameters are estimated. Its CDF is evaluated
    with ``genhyperbolic(p=1, a=alpha*delta, b=beta*delta, loc=mu, scale=delta)``.
    Scaled shape parameters must remain finite and satisfy ``a > abs(b)``.
    Write ``u_i = F(x_(i))`` for sorted observations, ``i=1,...,n``.
    Large statistics reject the null. Under the continuous null the probability
    integral transform makes the null law independent of all four parameters;
    it still depends on sample size. Ties and constant samples are accepted,
    but a discrete or rounded observation model needs its own calibration.
    The cited work concerns general EDF statistics, not a fitted hyperbolic test.
    Parameter fitting requires a different calibration and is not supported.
    The built-in Monte Carlo resolver currently has no hyperbolic generator;
    simulate externally and call ``execute_statistic`` for each sample.

    Return ``W² = 1/(12*n) + sum((u_i-(2*i-1)/(2*n))**2)``.
    There is no further multiplication by n or finite-sample correction.

    References
    ----------
    .. [1] Anderson, T. W. and Darling, D. A. (1952). Asymptotic Theory of
       Certain "Goodness of Fit" Criteria Based on Stochastic Processes.
       Ann. Math. Statist. 23, 193-212. https://doi.org/10.1214/aoms/1177729437

    Examples
    --------
    >>> statistic = CramerVonMisesHyperbolicGofStatistic(alpha=1.5, beta=0.2)
    >>> result = statistic.execute_statistic([-1.0, 0.0, 0.5, 2.0])
    >>> isinstance(result, float) and result >= 0
    True
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
        """Compute the Cramer--von Mises statistic.

        Parameters
        ----------
        rvs : array_like of float, shape (n,)
            Finite real observations, n >= 1. Ties and constants are allowed.
            A sorted copy is used; the input is not modified.
        **kwargs : dict
            Unused compatibility arguments; no per-call settings are supported.

        Returns
        -------
        float
            Uncorrected CVM statistic; large values reject the null.

        Raises
        ------
        ValueError
            If the sample is empty, nonreal, nonfinite, or not one-dimensional.
        FloatingPointError
            If SciPy returns invalid probabilities.

        Notes
        -----
        The statistic is mathematically finite for finite observations.
        Numerical tail accuracy is limited by SciPy's integration routines.
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
    """Anderson--Darling statistic for a fixed hyperbolic CDF.

    Parameters
    ----------
    alpha : float, default: 1.0
        Fixed shape parameter, finite and strictly greater than ``abs(beta)``.
    beta : float, default: 0.0
        Fixed finite skewness parameter.
    delta : float, default: 1.0
        Fixed finite positive scale parameter.
    mu : float, default: 0.0
        Fixed finite location parameter.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute one scalar statistic without changing the input or fitting.
    hypothesis()
        Return all four fixed distribution parameters.
    alternative()
        Return the right critical tail.

    Notes
    -----
    The null is an iid sample on the real line from the fully specified
    classical hyperbolic law. No parameters are estimated. Its CDF is evaluated
    with ``genhyperbolic(p=1, a=alpha*delta, b=beta*delta, loc=mu, scale=delta)``.
    Scaled shape parameters must remain finite and satisfy ``a > abs(b)``.
    Write ``u_i = F(x_(i))`` for sorted observations, ``i=1,...,n``.
    Large statistics reject the null. Under the continuous null the probability
    integral transform makes the null law independent of all four parameters;
    it still depends on sample size. Ties and constant samples are accepted,
    but a discrete or rounded observation model needs its own calibration.
    The cited work concerns general EDF statistics, not a fitted hyperbolic test.
    Parameter fitting requires a different calibration and is not supported.
    The built-in Monte Carlo resolver currently has no hyperbolic generator;
    simulate externally and call ``execute_statistic`` for each sample.

    Return ``A² = -n - sum((2*i-1)/n * (log(u_i)+log(1-u_(n+1-i))))``.
    Log-survival probabilities are evaluated directly, not by subtracting
    a rounded CDF from one. SciPy tail integration can still underflow;
    nonfinite log tails raise FloatingPointError. The mathematical statistic
    is finite for every finite sample in this model. No clipping is used.

    References
    ----------
    .. [1] Anderson, T. W. and Darling, D. A. (1952). Asymptotic Theory of
       Certain "Goodness of Fit" Criteria Based on Stochastic Processes.
       Ann. Math. Statist. 23, 193-212. https://doi.org/10.1214/aoms/1177729437

    Examples
    --------
    >>> statistic = AndersonDarlingHyperbolicGofStatistic(alpha=1.5, beta=0.2)
    >>> result = statistic.execute_statistic([-1.0, 0.0, 0.5, 2.0])
    >>> isinstance(result, float) and result >= 0
    True
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
        """Compute the Anderson--Darling statistic.

        Parameters
        ----------
        rvs : array_like of float, shape (n,)
            Finite real observations, n >= 1. Ties and constants are allowed.
            A sorted copy is used; the input is not modified.
        **kwargs : dict
            Unused compatibility arguments; no per-call settings are supported.

        Returns
        -------
        float
            Uncorrected AD statistic; large values reject the null.

        Raises
        ------
        ValueError
            If the sample is empty, nonreal, nonfinite, or not one-dimensional.
        FloatingPointError
            If SciPy returns invalid probabilities or log tails underflow.

        Notes
        -----
        The statistic is mathematically finite for finite observations.
        Numerical tail accuracy is limited by SciPy's integration routines.
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
    """Kuiper statistic for a fixed hyperbolic CDF.

    Parameters
    ----------
    alpha : float, default: 1.0
        Fixed shape parameter, finite and strictly greater than ``abs(beta)``.
    beta : float, default: 0.0
        Fixed finite skewness parameter.
    delta : float, default: 1.0
        Fixed finite positive scale parameter.
    mu : float, default: 0.0
        Fixed finite location parameter.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute one scalar statistic without changing the input or fitting.
    hypothesis()
        Return all four fixed distribution parameters.
    alternative()
        Return the right critical tail.

    Notes
    -----
    The null is an iid sample on the real line from the fully specified
    classical hyperbolic law. No parameters are estimated. Its CDF is evaluated
    with ``genhyperbolic(p=1, a=alpha*delta, b=beta*delta, loc=mu, scale=delta)``.
    Scaled shape parameters must remain finite and satisfy ``a > abs(b)``.
    Write ``u_i = F(x_(i))`` for sorted observations, ``i=1,...,n``.
    Large statistics reject the null. Under the continuous null the probability
    integral transform makes the null law independent of all four parameters;
    it still depends on sample size. Ties and constant samples are accepted,
    but a discrete or rounded observation model needs its own calibration.
    The cited work concerns general EDF statistics, not a fitted hyperbolic test.
    Parameter fitting requires a different calibration and is not supported.
    The built-in Monte Carlo resolver currently has no hyperbolic generator;
    simulate externally and call ``execute_statistic`` for each sample.

    Return ``V = D+ + D-``, where ``D+ = max(i/n-u_i)`` and
    ``D- = max(u_i-(i-1)/n)``. No sqrt(n) scaling or correction is applied.
    Apply the circular uniformity statistic to the transformed probabilities.

    References
    ----------
    .. [1] Kuiper, N. H. (1960). Tests concerning random points on a circle.
       Indagationes Mathematicae (Proceedings) 63, 38-47.
       https://doi.org/10.1016/S1385-7258(60)50006-0

    Examples
    --------
    >>> statistic = KuiperHyperbolicGofStatistic(alpha=1.5, beta=0.2)
    >>> result = statistic.execute_statistic([-1.0, 0.0, 0.5, 2.0])
    >>> isinstance(result, float) and result >= 0
    True
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
        """Compute the Kuiper statistic.

        Parameters
        ----------
        rvs : array_like of float, shape (n,)
            Finite real observations, n >= 1. Ties and constants are allowed.
            A sorted copy is used; the input is not modified.
        **kwargs : dict
            Unused compatibility arguments; no per-call settings are supported.

        Returns
        -------
        float
            Uncorrected KUI statistic; large values reject the null.

        Raises
        ------
        ValueError
            If the sample is empty, nonreal, nonfinite, or not one-dimensional.
        FloatingPointError
            If SciPy returns invalid probabilities.

        Notes
        -----
        The statistic is mathematically finite for finite observations.
        Numerical tail accuracy is limited by SciPy's integration routines.
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
    """Watson centered Cramer--von Mises statistic for a fixed hyperbolic CDF.

    Parameters
    ----------
    alpha : float, default: 1.0
        Fixed shape parameter, finite and strictly greater than ``abs(beta)``.
    beta : float, default: 0.0
        Fixed finite skewness parameter.
    delta : float, default: 1.0
        Fixed finite positive scale parameter.
    mu : float, default: 0.0
        Fixed finite location parameter.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute one scalar statistic without changing the input or fitting.
    hypothesis()
        Return all four fixed distribution parameters.
    alternative()
        Return the right critical tail.

    Notes
    -----
    The null is an iid sample on the real line from the fully specified
    classical hyperbolic law. No parameters are estimated. Its CDF is evaluated
    with ``genhyperbolic(p=1, a=alpha*delta, b=beta*delta, loc=mu, scale=delta)``.
    Scaled shape parameters must remain finite and satisfy ``a > abs(b)``.
    Write ``u_i = F(x_(i))`` for sorted observations, ``i=1,...,n``.
    Large statistics reject the null. Under the continuous null the probability
    integral transform makes the null law independent of all four parameters;
    it still depends on sample size. Ties and constant samples are accepted,
    but a discrete or rounded observation model needs its own calibration.
    The cited work concerns general EDF statistics, not a fitted hyperbolic test.
    Parameter fitting requires a different calibration and is not supported.
    The built-in Monte Carlo resolver currently has no hyperbolic generator;
    simulate externally and call ``execute_statistic`` for each sample.

    Return ``U² = W² - n*(mean(u)-1/2)**2``. Equivalently, with
    ``d_i = u_i-(2*i-1)/(2*n)``, use ``1/(12*n)+sum((d_i-mean(d))**2)``
    to avoid subtracting two large terms. No finite-sample correction is used.
    Apply the circular uniformity statistic to the transformed probabilities.

    References
    ----------
    .. [1] Watson, G. S. (1961). Goodness-of-fit tests on a circle.
       Biometrika 48, 109-114. https://doi.org/10.1093/biomet/48.1-2.109

    Examples
    --------
    >>> statistic = WatsonHyperbolicGofStatistic(alpha=1.5, beta=0.2)
    >>> result = statistic.execute_statistic([-1.0, 0.0, 0.5, 2.0])
    >>> isinstance(result, float) and result >= 0
    True
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
        mean_deviation = 0.0
        for i in range(n):
            mean_deviation += cdf_values[i] - (i + 0.5) / n
        mean_deviation /= n

        total = 1.0 / (12.0 * n)
        for i in range(n):
            centered = cdf_values[i] - (i + 0.5) / n - mean_deviation
            total += centered * centered
        return total

    @override
    def execute_statistic(self, rvs, **kwargs) -> float:
        """Compute the Watson centered Cramer--von Mises statistic.

        Parameters
        ----------
        rvs : array_like of float, shape (n,)
            Finite real observations, n >= 1. Ties and constants are allowed.
            A sorted copy is used; the input is not modified.
        **kwargs : dict
            Unused compatibility arguments; no per-call settings are supported.

        Returns
        -------
        float
            Uncorrected WAT statistic; large values reject the null.

        Raises
        ------
        ValueError
            If the sample is empty, nonreal, nonfinite, or not one-dimensional.
        FloatingPointError
            If SciPy returns invalid probabilities.

        Notes
        -----
        The statistic is mathematically finite for finite observations.
        Numerical tail accuracy is limited by SciPy's integration routines.
        """
        sorted_rvs = self._prepare_sample(rvs)
        return self.do_execute_statistic(self._cdf(sorted_rvs))

    def do_execute_statistic(self, cdf_values: np.ndarray) -> float:
        """Calculate the statistic from precomputed CDF values.

        :param cdf_values: reference CDF values in sorted sample order.
        :return: Watson statistic.
        """
        return float(self._calculate_statistic(cdf_values))
