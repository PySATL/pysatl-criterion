from __future__ import annotations

from abc import ABC

import numpy as np
from numba import njit
from typing_extensions import override

from pysatl_criterion.distribution.distributions import LaplaceDistributionDescriptor
from pysatl_criterion.distribution.distributions import (
    LaplaceDistributionDescriptor as Distribution,
)
from pysatl_criterion.distribution.parameters import HypothesisSupport, ParameterValues
from pysatl_criterion.statistics import AbstractGoodnessOfFitStatistic
from pysatl_criterion.statistics.alternative import Alternative, AlternativeType, RightAlternative
from pysatl_criterion.statistics.goodness_of_fit.common import (
    ADStatistic,
    CrammerVonMisesStatistic,
    KSStatistic,
)


@njit
def _standardize(x: float, t: float, s: float) -> float:
    """Avoid overflow of x - t when the standardized value is representable."""
    difference = x - t
    if np.isinf(difference):
        return x / s - t / s
    return difference / s


@njit
def _laplace_cdf(x: float, t: float, s: float) -> float:
    z = _standardize(x, t, s)
    if z < 0.0:
        return 0.5 * np.exp(z)
    return 1.0 - 0.5 * np.exp(-z)


@njit
def _laplace_cdf_values(sorted_rvs: np.ndarray, t: float, s: float) -> np.ndarray:
    values = np.empty(len(sorted_rvs))
    for i in range(len(sorted_rvs)):
        values[i] = _laplace_cdf(sorted_rvs[i], t, s)
    return values


class AbstractLaplaceGofStatistic(AbstractGoodnessOfFitStatistic, ABC):
    """Abstract base class for Laplace distribution goodness-of-fit statistics."""

    @property
    def t(self) -> float:
        """Read t by its stable parameter identity."""
        return self._parameters[Distribution.LOCATION]

    @property
    def s(self) -> float:
        """Read s by its stable parameter identity."""
        return self._parameters[Distribution.SCALE]

    @staticmethod
    def _validate_parameter(value, name):
        try:
            array = np.asarray(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a finite real scalar.") from exc
        if array.ndim != 0 or array.dtype.kind not in "iuf":
            raise ValueError(f"{name} must be a finite real scalar.")
        value = float(array)
        if not np.isfinite(value):
            raise ValueError(f"{name} must be a finite real scalar.")
        return value

    @staticmethod
    def _validate_sample(rvs):
        try:
            sample = np.asarray(rvs)
            if sample.dtype.kind not in "iuf":
                raise ValueError("Observations must be real numbers.")
            sample = np.asarray(sample, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Sample must contain finite real numbers.") from exc
        if sample.ndim != 1:
            raise ValueError("Sample must be one-dimensional.")
        if sample.size == 0:
            raise ValueError("At least one observation is required.")
        if not np.all(np.isfinite(sample)):
            raise ValueError("Sample must contain finite real numbers.")
        return np.sort(sample)

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.LOCATION, Distribution.SCALE})
            ),
        )

    @staticmethod
    @override
    def distribution() -> type[LaplaceDistributionDescriptor]:
        """Return the distribution descriptor class."""
        return LaplaceDistributionDescriptor

    @classmethod
    @override
    def code(cls) -> str:
        """Return the family identifier or the concrete statistic's full identifier."""
        family_code = f"LAPLACE_{AbstractGoodnessOfFitStatistic.code()}"
        if "short_code" in cls.__abstractmethods__:
            return family_code
        return f"{cls.short_code()}_{family_code}"


class KolmogorovSmirnovLaplaceGofStatistic(AbstractLaplaceGofStatistic, KSStatistic):
    """Kolmogorov--Smirnov statistic D for a fully specified Laplace CDF.

    Parameters
    ----------
    parameters : ParameterValues
        Values with t, s fixed; omitted parameters are unknown.
    alternative_type : AlternativeType, default TWO_TAILED
        CDF deviation direction: TWO_TAILED gives D, RIGHT D+, LEFT D-.
        This setting does not change the right tail used for rejection.
    mode : {'auto', 'exact', 'asymp', 'approx'}, default 'auto'
        Compatibility setting; has no effect on the statistic. No p-value
        is computed. Internally 'auto' is stored as 'exact'.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic without changing the sample or model.
    hypothesis()
        Return the fixed parameters ``{"t": t, "s": s}``.
    alternative()
        Return the right-tailed rejection rule.

    Notes
    -----
    H0: observations are independent Laplace(t, s) on the real line.
    Let u_i = F_(t,s)(x_(i)) for sorted observations and sample size n.
    D+ = max(i/n - u_i), D- = max(u_i - (i-1)/n), and
    D = max(D+, D-), for i = 1, ..., n. RIGHT selects D+; LEFT
    selects D-; TWO_TAILED selects D. All reject for large values.

    Large values reject H0. Under H0 the probability integral transform
    gives independent uniforms: the null law depends on n, not t or s.
    Calibration must use this unmodified statistic and fixed parameters.
    Parameters fitted to the tested sample require a different calibration.
    Reference [1]_ supplies the general statistic; applying it through
    the known Laplace CDF does not define a fitted Laplace-family test.
    CDF values may round to 0 or 1 in extreme tails; small spacings can
    lose precision. Observations and probabilities are never clipped.

    Stored calibration cannot distinguish KS directions and is rejected
    for LEFT/RIGHT. Use Monte Carlo calibration for those directions.

    References
    ----------
    .. [1] M. A. Stephens (1974), "EDF Statistics for Goodness of Fit and
        Some Comparisons", JASA 69, 730-737, Section 2 (definitions and
        computation), Case 0 (fully specified continuous distribution).
        https://doi.org/10.1080/01621459.1974.10480196
        Full text: https://www.math.utah.edu/~morris/Courses/6010/p1/writeup/ks.pdf

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'t': 0.0, 's': 1.0})
    >>> statistic = KolmogorovSmirnovLaplaceGofStatistic(parameters)
    >>> value = statistic.execute_statistic([-1.0, 0.0, 0.5, 2.0])
    >>> bool(np.isfinite(value))
    True
    """

    def __init__(
        self,
        parameters: ParameterValues,
        *,
        alternative_type: AlternativeType = AlternativeType.TWO_TAILED,
        mode="auto",
    ):
        AbstractLaplaceGofStatistic.__init__(self, parameters)
        if alternative_type not in tuple(AlternativeType):
            raise ValueError("Invalid CDF deviation direction.")
        if not isinstance(mode, str) or mode not in ("auto", "exact", "asymp", "approx"):
            raise ValueError("Invalid KS mode.")
        KSStatistic.__init__(self, alternative_type=alternative_type, mode=mode)

    def _validate_storage_calibration(self):
        if self.alternative_type != AlternativeType.TWO_TAILED:
            raise ValueError("Stored KS calibration does not encode the CDF direction")

    @staticmethod
    @override
    def short_code():
        """Get the short code identifier for this test.

        :return: short code string ``KS``.
        """
        return "KS"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the unscaled Kolmogorov--Smirnov statistic D.

        Parameters
        ----------
        rvs : array_like of real numbers, shape (n,)
            Finite one-dimensional sample with n >= 1. Ties and constant
            samples are allowed. Input is copied for sorting.
        **kwargs : dict
            Accepted for interface compatibility; ignored. There are no
            per-call parameter overrides or fitting options.

        Returns
        -------
        float
            D; larger values indicate departure from the fixed null.

        Raises
        ------
        ValueError
            If rvs is empty, non-real, non-finite, or not one-dimensional.

        Notes
        -----
        The mathematical result is finite for every valid finite sample.
        """
        sorted_rvs = self._validate_sample(rvs)
        cdf_vals = _laplace_cdf_values(sorted_rvs, self.t, self.s)
        return float(KSStatistic.do_execute_statistic(self, sorted_rvs, cdf_vals))


class CramerVonMisesLaplaceGofStatistic(AbstractLaplaceGofStatistic, CrammerVonMisesStatistic):
    """Cramer--von Mises statistic W² for a fully specified Laplace CDF.

    Parameters
    ----------
    parameters : ParameterValues
        Values with t, s fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic without changing the sample or model.
    hypothesis()
        Return the fixed parameters ``{"t": t, "s": s}``.
    alternative()
        Return the right-tailed rejection rule.

    Notes
    -----
    H0: observations are independent Laplace(t, s) on the real line.
    Let u_i = F_(t,s)(x_(i)) for sorted observations and sample size n.
    W² = 1/(12*n) + sum((u_i - (2*i-1)/(2*n))**2), i = 1, ..., n.
    No finite-sample or fitted-parameter correction is applied.

    Large values reject H0. Under H0 the probability integral transform
    gives independent uniforms: the null law depends on n, not t or s.
    Calibration must use this unmodified statistic and fixed parameters.
    Parameters fitted to the tested sample require a different calibration.
    Reference [1]_ supplies the general statistic; applying it through
    the known Laplace CDF does not define a fitted Laplace-family test.
    CDF values may round to 0 or 1 in extreme tails; small spacings can
    lose precision. Observations and probabilities are never clipped.

    References
    ----------
    .. [1] M. A. Stephens (1974), "EDF Statistics for Goodness of Fit and
        Some Comparisons", JASA 69, 730-737, Section 2 (definitions and
        computation), Case 0 (fully specified continuous distribution).
        https://doi.org/10.1080/01621459.1974.10480196
        Full text: https://www.math.utah.edu/~morris/Courses/6010/p1/writeup/ks.pdf

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'t': 0.0, 's': 1.0})
    >>> statistic = CramerVonMisesLaplaceGofStatistic(parameters)
    >>> value = statistic.execute_statistic([-1.0, 0.0, 0.5, 2.0])
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def short_code():
        """Get the short code identifier for this test.

        :return: short code string ``CVM``.
        """
        return "CVM"

    @staticmethod
    @njit
    def _calculate_statistic(
        sorted_rvs: np.ndarray, t: float, s: float
    ) -> float:  # pragma: no cover
        n = len(sorted_rvs)
        total = 1.0 / (12.0 * n)

        for i in range(n):
            x = sorted_rvs[i]

            current_cdf = _laplace_cdf(x, t, s)

            expected_cdf = (2.0 * i + 1.0) / (2.0 * n)
            difference = expected_cdf - current_cdf
            total += difference * difference

        return total

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the unscaled Cramer--von Mises statistic W².

        Parameters
        ----------
        rvs : array_like of real numbers, shape (n,)
            Finite one-dimensional sample with n >= 1. Ties and constant
            samples are allowed. Input is copied for sorting.
        **kwargs : dict
            Accepted for interface compatibility; ignored. There are no
            per-call parameter overrides or fitting options.

        Returns
        -------
        float
            W²; larger values indicate departure from the fixed null.

        Raises
        ------
        ValueError
            If rvs is empty, non-real, non-finite, or not one-dimensional.

        Notes
        -----
        The mathematical result is finite for every valid finite sample.
        """
        sorted_rvs = self._validate_sample(rvs)
        return self.do_execute_statistic(sorted_rvs)

    def do_execute_statistic(self, sorted_rvs):
        """Calculate the statistic for an already sorted sample.

        :param sorted_rvs: one-dimensional sample sorted in ascending order.
        :return: Cramer--von Mises statistic computed from the Laplace CDF.
        """
        return float(
            self._calculate_statistic(
                sorted_rvs,
                self.t,
                self.s,
            )
        )


class AndersonDarlingLaplaceGofStatistic(AbstractLaplaceGofStatistic, ADStatistic):
    """Anderson--Darling statistic A² for a fully specified Laplace CDF.

    Parameters
    ----------
    parameters : ParameterValues
        Values with t, s fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic without changing the sample or model.
    hypothesis()
        Return the fixed parameters ``{"t": t, "s": s}``.
    alternative()
        Return the right-tailed rejection rule.

    Notes
    -----
    H0: observations are independent Laplace(t, s) on the real line.
    Let u_i = F_(t,s)(x_(i)) for sorted observations and sample size n.
    A² = -n - sum((2*i-1)/n * (log(u_i) + log(1-u_(n+1-i)))),
    i = 1, ..., n. Log probabilities are evaluated analytically:
    log(F(x)) = z-log(2) for z < 0 and log1p(-exp(-z)/2) otherwise,
    where z = (x-t)/s; log(S(x)) follows by reflection.
    This avoids exponential underflow in the logarithmic tails.
    No finite-sample or fitted-parameter correction is applied.

    Large values reject H0. Under H0 the probability integral transform
    gives independent uniforms: the null law depends on n, not t or s.
    Calibration must use this unmodified statistic and fixed parameters.
    Parameters fitted to the tested sample require a different calibration.
    Reference [1]_ supplies the general statistic; applying it through
    the known Laplace CDF does not define a fitted Laplace-family test.
    CDF values may round to 0 or 1 in extreme tails; small spacings can
    lose precision. AD uses log probabilities directly. Values beyond
    float64 range can still overflow; observations are never clipped.

    References
    ----------
    .. [1] M. A. Stephens (1974), "EDF Statistics for Goodness of Fit and
        Some Comparisons", JASA 69, 730-737, Section 2 (definitions and
        computation), Case 0 (fully specified continuous distribution).
        https://doi.org/10.1080/01621459.1974.10480196
        Full text: https://www.math.utah.edu/~morris/Courses/6010/p1/writeup/ks.pdf

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'t': 0.0, 's': 1.0})
    >>> statistic = AndersonDarlingLaplaceGofStatistic(parameters)
    >>> value = statistic.execute_statistic([-1.0, 0.0, 0.5, 2.0])
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def short_code():
        """Get the short code identifier for this test.

        :return: short code string ``AD``.
        """
        return "AD"

    @staticmethod
    @njit
    def _calculate_statistic(
        sorted_rvs: np.ndarray, t: float, s: float
    ) -> float:  # pragma: no cover
        n = len(sorted_rvs)
        total = 0.0

        for i in range(n):
            lower = _standardize(sorted_rvs[i], t, s)
            upper = _standardize(sorted_rvs[n - i - 1], t, s)

            if lower < 0.0:
                log_cdf = lower - np.log(2.0)
            else:
                log_cdf = np.log1p(-0.5 * np.exp(-lower))

            if upper < 0.0:
                log_sf = np.log1p(-0.5 * np.exp(upper))
            else:
                log_sf = -upper - np.log(2.0)

            weight = (2.0 * i + 1.0) / n
            total += weight * log_cdf
            total += weight * log_sf

        return -n - total

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the unscaled Anderson--Darling statistic A².

        Parameters
        ----------
        rvs : array_like of real numbers, shape (n,)
            Finite one-dimensional sample with n >= 1. Ties and constant
            samples are allowed. Input is copied for sorting.
        **kwargs : dict
            Accepted for interface compatibility; ignored. There are no
            per-call parameter overrides or fitting options.

        Returns
        -------
        float
            A²; larger values indicate departure from the fixed null.

        Raises
        ------
        ValueError
            If rvs is empty, non-real, non-finite, or not one-dimensional.

        Notes
        -----
        The mathematical result is finite for every valid finite sample.
        Floating-point overflow remains possible for AD beyond float64
        range; an infinite result is not replaced with an epsilon.
        """
        sorted_rvs = self._validate_sample(rvs)
        return self.do_execute_statistic(sorted_rvs)

    def do_execute_statistic(self, sorted_rvs):
        """Calculate the statistic for an already sorted sample.

        :param sorted_rvs: one-dimensional sample sorted in ascending order.
        :return: Anderson--Darling statistic computed from Laplace log probabilities.
        """
        return float(
            self._calculate_statistic(
                sorted_rvs,
                self.t,
                self.s,
            )
        )


class KuiperLaplaceGofStatistic(AbstractLaplaceGofStatistic):
    """Kuiper statistic V for a fully specified Laplace CDF.

    Parameters
    ----------
    parameters : ParameterValues
        Values with t, s fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic without changing the sample or model.
    hypothesis()
        Return the fixed parameters ``{"t": t, "s": s}``.
    alternative()
        Return the right-tailed rejection rule.

    Notes
    -----
    H0: observations are independent Laplace(t, s) on the real line.
    Let u_i = F_(t,s)(x_(i)) for sorted observations and sample size n.
    V = max(i/n - u_i) + max(u_i - (i-1)/n), i = 1, ..., n.
    The statistic is unscaled: no sqrt(n) or finite-sample correction.

    Large values reject H0. Under H0 the probability integral transform
    gives independent uniforms: the null law depends on n, not t or s.
    Calibration must use this unmodified statistic and fixed parameters.
    Parameters fitted to the tested sample require a different calibration.
    Reference [1]_ supplies the general statistic; applying it through
    the known Laplace CDF does not define a fitted Laplace-family test.
    CDF values may round to 0 or 1 in extreme tails; small spacings can
    lose precision. Observations and probabilities are never clipped.

    References
    ----------
    .. [1] M. A. Stephens (1974), "EDF Statistics for Goodness of Fit and
        Some Comparisons", JASA 69, 730-737, Section 2 (definitions and
        computation), Case 0 (fully specified continuous distribution).
        https://doi.org/10.1080/01621459.1974.10480196
        Full text: https://www.math.utah.edu/~morris/Courses/6010/p1/writeup/ks.pdf

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'t': 0.0, 's': 1.0})
    >>> statistic = KuiperLaplaceGofStatistic(parameters)
    >>> value = statistic.execute_statistic([-1.0, 0.0, 0.5, 2.0])
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """
        Get the short code identifier for this test.

        :return: short code string ``KUI``.
        """
        return "KUI"

    @staticmethod
    @njit
    def _calculate_statistic(
        sorted_rvs: np.ndarray,
        t: float,
        s: float,
    ) -> float:  # pragma: no cover
        n = len(sorted_rvs)
        d_plus = 0.0
        d_minus = 0.0

        for i in range(n):
            x = sorted_rvs[i]

            current_cdf = _laplace_cdf(x, t, s)

            d_plus = max(d_plus, (i + 1.0) / n - current_cdf)
            d_minus = max(d_minus, current_cdf - i / n)

        return d_plus + d_minus

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the unscaled Kuiper statistic V.

        Parameters
        ----------
        rvs : array_like of real numbers, shape (n,)
            Finite one-dimensional sample with n >= 1. Ties and constant
            samples are allowed. Input is copied for sorting.
        **kwargs : dict
            Accepted for interface compatibility; ignored. There are no
            per-call parameter overrides or fitting options.

        Returns
        -------
        float
            V; larger values indicate departure from the fixed null.

        Raises
        ------
        ValueError
            If rvs is empty, non-real, non-finite, or not one-dimensional.

        Notes
        -----
        The mathematical result is finite for every valid finite sample.
        """

        sorted_rvs = self._validate_sample(rvs)
        return self.do_execute_statistic(sorted_rvs)

    def do_execute_statistic(self, sorted_rvs):
        """
        Calculate the statistic for an already sorted sample.

        :param sorted_rvs: one-dimensional sample sorted in ascending order.
        :return: Kuiper statistic computed from the Laplace CDF.
        """
        return float(
            self._calculate_statistic(
                sorted_rvs,
                self.t,
                self.s,
            )
        )


class WatsonLaplaceGofStatistic(AbstractLaplaceGofStatistic):
    """Watson statistic U² for a fully specified Laplace CDF.

    Parameters
    ----------
    parameters : ParameterValues
        Values with t, s fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic without changing the sample or model.
    hypothesis()
        Return the fixed parameters ``{"t": t, "s": s}``.
    alternative()
        Return the right-tailed rejection rule.

    Notes
    -----
    H0: observations are independent Laplace(t, s) on the real line.
    Let u_i = F_(t,s)(x_(i)) for sorted observations and sample size n.
    U² = W² - n*(mean(u)-1/2)**2, where
    W² = 1/(12*n) + sum((u_i - (2*i-1)/(2*n))**2).
    Computation uses the equivalent centered sum of squares of
    u_i - (2*i-1)/(2*n) to avoid subtracting two large totals.
    No finite-sample correction is applied. For n=1, U²=1/12
    for every observation, so the statistic cannot discriminate.

    Large values reject H0. Under H0 the probability integral transform
    gives independent uniforms: the null law depends on n, not t or s.
    Calibration must use this unmodified statistic and fixed parameters.
    Parameters fitted to the tested sample require a different calibration.
    Reference [1]_ supplies the general statistic; applying it through
    the known Laplace CDF does not define a fitted Laplace-family test.
    CDF values may round to 0 or 1 in extreme tails; small spacings can
    lose precision. Observations and probabilities are never clipped.

    References
    ----------
    .. [1] M. A. Stephens (1974), "EDF Statistics for Goodness of Fit and
        Some Comparisons", JASA 69, 730-737, Section 2 (definitions and
        computation), Case 0 (fully specified continuous distribution).
        https://doi.org/10.1080/01621459.1974.10480196
        Full text: https://www.math.utah.edu/~morris/Courses/6010/p1/writeup/ks.pdf

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'t': 0.0, 's': 1.0})
    >>> statistic = WatsonLaplaceGofStatistic(parameters)
    >>> value = statistic.execute_statistic([-1.0, 0.0, 0.5, 2.0])
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "WAT".
        """
        return "WAT"

    @staticmethod
    @njit
    def _calculate_statistic(
        sorted_rvs: np.ndarray,
        t: float,
        s: float,
    ) -> float:  # pragma: no cover
        n = len(sorted_rvs)
        mean_difference = 0.0
        centered_sum = 0.0
        for i in range(n):
            current_cdf = _laplace_cdf(sorted_rvs[i], t, s)
            difference = current_cdf - (2.0 * i + 1.0) / (2.0 * n)
            delta = difference - mean_difference
            mean_difference += delta / (i + 1.0)
            centered_sum += delta * (difference - mean_difference)
        return 1.0 / (12.0 * n) + centered_sum

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the unscaled Watson statistic U².

        Parameters
        ----------
        rvs : array_like of real numbers, shape (n,)
            Finite one-dimensional sample with n >= 1. Ties and constant
            samples are allowed. Input is copied for sorting.
        **kwargs : dict
            Accepted for interface compatibility; ignored. There are no
            per-call parameter overrides or fitting options.

        Returns
        -------
        float
            U²; larger values indicate departure from the fixed null.

        Raises
        ------
        ValueError
            If rvs is empty, non-real, non-finite, or not one-dimensional.

        Notes
        -----
        The mathematical result is finite for every valid finite sample.
        """

        sorted_rvs = self._validate_sample(rvs)

        return self.do_execute_statistic(sorted_rvs)

    def do_execute_statistic(self, sorted_rvs):
        """
        Calculate the Watson statistic for an already sorted sample.

        :param sorted_rvs: one-dimensional sample sorted in ascending order.
        :return: Watson statistic computed from the Laplace CDF values.
        """
        return float(
            self._calculate_statistic(
                sorted_rvs,
                self.t,
                self.s,
            )
        )


class GreenwoodLaplaceGofStatistic(AbstractLaplaceGofStatistic):
    """Greenwood statistic G for a fully specified Laplace CDF.

    Parameters
    ----------
    parameters : ParameterValues
        Values with t, s fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic without changing the sample or model.
    hypothesis()
        Return the fixed parameters ``{"t": t, "s": s}``.
    alternative()
        Return the right-tailed rejection rule.

    Notes
    -----
    H0: observations are independent Laplace(t, s) on the real line.
    Let u_i = F_(t,s)(x_(i)) for sorted observations and sample size n.
    G = sum((u_i-u_(i-1))**2), i = 1, ..., n+1,
    with u_0=0 and u_(n+1)=1. Both endpoint spacings are included;
    repeated observations yield zero spacings. The result is unscaled.
    This right-tail spacing test detects clustering; it is not asserted
    to be consistent against every nonuniform alternative.

    Large values reject H0. Under H0 the probability integral transform
    gives independent uniforms: the null law depends on n, not t or s.
    Calibration must use this unmodified statistic and fixed parameters.
    Parameters fitted to the tested sample require a different calibration.
    Reference [1]_ supplies the general statistic; applying it through
    the known Laplace CDF does not define a fitted Laplace-family test.
    CDF values may round to 0 or 1 in extreme tails; small spacings can
    lose precision. Observations and probabilities are never clipped.

    References
    ----------
    .. [1] R. J. M. M. Does, R. Helmers and C. A. J. Klaassen (1988),
        "Approximating the distribution of Greenwood's statistic",
        Statistica Neerlandica 42, 153-162, Section 1, equations (1)-(2).
        https://ir.cwi.nl/pub/1694/1694D.pdf

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'t': 0.0, 's': 1.0})
    >>> statistic = GreenwoodLaplaceGofStatistic(parameters)
    >>> value = statistic.execute_statistic([-1.0, 0.0, 0.5, 2.0])
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        """Return the right-tailed alternative for the Greenwood statistic.

        :return: right-tailed alternative.
        """
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """Get the short code identifier for this test.

        :return: short code string ``GRW``.
        """
        return "GRW"

    @staticmethod
    @njit
    def _calculate_statistic(
        sorted_rvs: np.ndarray, t: float, s: float
    ) -> float:  # pragma: no cover
        n = len(sorted_rvs)
        total = 0.0
        previous_cdf = 0.0

        for i in range(n):
            x = sorted_rvs[i]

            current_cdf = _laplace_cdf(x, t, s)

            spacing = current_cdf - previous_cdf
            total += spacing * spacing
            previous_cdf = current_cdf

        final_spacing = 1.0 - previous_cdf
        total += final_spacing * final_spacing

        return total

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the unscaled Greenwood statistic G.

        Parameters
        ----------
        rvs : array_like of real numbers, shape (n,)
            Finite one-dimensional sample with n >= 1. Ties and constant
            samples are allowed. Input is copied for sorting.
        **kwargs : dict
            Accepted for interface compatibility; ignored. There are no
            per-call parameter overrides or fitting options.

        Returns
        -------
        float
            G; larger values indicate departure from the fixed null.

        Raises
        ------
        ValueError
            If rvs is empty, non-real, non-finite, or not one-dimensional.

        Notes
        -----
        The mathematical result is finite for every valid finite sample.
        """

        sorted_rvs = self._validate_sample(rvs)

        return self.do_execute_statistic(sorted_rvs)

    def do_execute_statistic(self, sorted_rvs):
        """Calculate the statistic for an already sorted sample.

        Keeping sorting outside the compiled kernel avoids allocating a new
        array on every kernel call and makes it possible to benchmark only the
        statistic calculation.

        :param sorted_rvs: one-dimensional sample sorted in ascending order.
        :return: Greenwood statistic computed from the Laplace CDF spacings.
        """
        return float(
            self._calculate_statistic(
                sorted_rvs,
                self.t,
                self.s,
            )
        )
