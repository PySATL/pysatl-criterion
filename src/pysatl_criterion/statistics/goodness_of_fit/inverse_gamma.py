from __future__ import annotations

from abc import ABC

import numpy as np
import scipy.stats as scipy_stats
from typing_extensions import override

from pysatl_criterion.distribution.distributions import InverseGammaDistributionDescriptor
from pysatl_criterion.distribution.distributions import (
    InverseGammaDistributionDescriptor as Distribution,
)
from pysatl_criterion.distribution.parameters import HypothesisSupport, ParameterValues
from pysatl_criterion.statistics import AbstractGoodnessOfFitStatistic
from pysatl_criterion.statistics.alternative import (
    Alternative,
    AlternativeType,
    RightAlternative,
)
from pysatl_criterion.statistics.goodness_of_fit.common import (
    ADStatistic,
    Chi2Statistic,
    CrammerVonMisesStatistic,
    KSStatistic,
    MinToshiyukiStatistic,
)


class AbstractInverseGammaGofStatistic(AbstractGoodnessOfFitStatistic, ABC):
    """Shared shape/scale hypothesis and validation for Inverse Gamma statistics."""

    @property
    def alpha(self) -> float:
        """Read alpha by its stable parameter identity."""
        return self._parameters[Distribution.SHAPE]

    @property
    def beta(self) -> float:
        """Read beta by its stable parameter identity."""
        return self._parameters[Distribution.SCALE]

    @staticmethod
    def _prepare_sample(rvs, minimum=1):
        sample = np.asarray(rvs)
        if sample.ndim != 1 or sample.dtype.kind not in "iuf":
            raise ValueError("Sample must be a one-dimensional real numeric array")
        if sample.size < minimum:
            raise ValueError(f"At least {minimum} observations are required")
        sample = np.asarray(sample, dtype=float)
        if not np.all(np.isfinite(sample)) or np.any(sample <= 0):
            raise ValueError("Observations must be finite and strictly positive for Inverse Gamma")
        return np.sort(sample)

    def _cdf(self, sample):
        values = scipy_stats.invgamma.cdf(sample, a=self.alpha, scale=self.beta)
        if (
            not np.all(np.isfinite(values))
            or np.any(values < 0)
            or np.any(values > 1)
            or np.any(np.diff(values) < 0)
        ):
            raise FloatingPointError("Invalid Inverse Gamma CDF values")
        return values

    def _log_probabilities(self, sample):
        log_cdf = scipy_stats.invgamma.logcdf(sample, a=self.alpha, scale=self.beta)
        log_sf = scipy_stats.invgamma.logsf(sample, a=self.alpha, scale=self.beta)
        if (
            not np.all(np.isfinite(log_cdf))
            or not np.all(np.isfinite(log_sf))
            or np.any(log_cdf > 0)
            or np.any(log_sf > 0)
        ):
            raise FloatingPointError("Inverse Gamma log tails exceed numerical precision")
        return log_cdf, log_sf

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.SCALE})
            ),
        )

    @staticmethod
    @override
    def distribution() -> type[InverseGammaDistributionDescriptor]:
        """Return the distribution descriptor class."""
        return InverseGammaDistributionDescriptor

    @classmethod
    @override
    def code(cls) -> str:
        """Return the family identifier or the concrete statistic's full identifier."""
        family_code = f"INV_GAMMA_{AbstractGoodnessOfFitStatistic.code()}"
        if "short_code" in cls.__abstractmethods__:
            return family_code
        return f"{cls.short_code()}_{family_code}"


class KolmogorovSmirnovInverseGammaGofStatistic(AbstractInverseGammaGofStatistic, KSStatistic):
    """Kolmogorov--Smirnov distance to a fixed Inverse Gamma CDF.

    Parameters
    ----------
    parameters : ParameterValues
        Values with alpha, beta fixed; omitted parameters are unknown.
    alternative_type : AlternativeType, default: TWO_TAILED
        Direction of CDF deviation; use an enum member.
    mode : {"auto", "exact", "approx", "asymp"}, default: "auto"
        Compatibility option; no p-value is calculated.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar without modifying the input or retaining estimates.
    hypothesis()
        Return fixed parameters; omitted parameters are unknown.
    alternative()
        Return the right critical tail.

    Notes
    -----
    The iid null model has density beta**alpha/Gamma(alpha) *
    x**(-alpha-1)*exp(-beta/x), x > 0, with location fixed at zero.
    Write u_i=F(x_(i)), i=1,...,n, for sorted observations.
    D+ = max(i/n-u_i), D- = max(u_i-(i-1)/n). Return max(D+,D-)
    for TWO_TAILED, D+ for RIGHT, D- for LEFT. All reject for large values.
    Stored calibration is blocked for directional KS because its key omits
    direction. The mode argument does not affect the scalar statistic.

    Both parameters are fixed; no fitting is performed. Under the continuous
    null the probability transform removes alpha and beta from the null law,
    which still depends on n. Large values reject. Ties and constants are
    accepted; rounded data need separate calibration. The built-in Monte
    Carlo resolver has no Inverse Gamma generator; simulate externally.
    Finite positive observations give mathematically finite statistics.
    Numerical log-tail underflow or overflow raises FloatingPointError,
    rather than replacing probabilities with epsilon. Ordinary CDF values
    and spacings may still round in extreme tails.
    The reference concerns the general statistic, applied here through the
    fixed CDF or cell probabilities, not an Inverse Gamma fitted test.

    References
    ----------
    .. [1] Anderson, T. W. and Darling, D. A. (1952). Asymptotic Theory of
       Certain "Goodness of Fit" Criteria Based on Stochastic Processes.
       Ann. Math. Statist. 23, 193-212. https://doi.org/10.1214/aoms/1177729437

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'alpha': 3.0, 'beta': 2.0})
    >>> statistic = KolmogorovSmirnovInverseGammaGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.4, 0.7, 1.2, 2.0])
    >>> bool(np.isfinite(value))
    True
    """

    def _validate_storage_calibration(self):
        if self.alternative_type != AlternativeType.TWO_TAILED:
            raise ValueError("Stored KS calibration does not encode the CDF direction")

    def __init__(
        self,
        parameters: ParameterValues,
        *,
        alternative_type: AlternativeType = AlternativeType.TWO_TAILED,
        mode="auto",
    ):
        AbstractInverseGammaGofStatistic.__init__(self, parameters)
        if alternative_type not in tuple(AlternativeType):
            raise ValueError("alternative_type must be an AlternativeType member")
        if mode not in ("auto", "exact", "approx", "asymp"):
            raise ValueError("Invalid KS mode")
        KSStatistic.__init__(self, alternative_type=alternative_type, mode=mode)

    @staticmethod
    @override
    def short_code() -> str:
        return "KS"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic defined in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real strictly positive observations. At least one value;
            ties and constant samples are allowed.
        **kwargs : dict
            Unused compatibility arguments.

        Returns
        -------
        float or numpy.float64
            Scalar statistic; large values reject the null.

        Raises
        ------
        ValueError
            If sample dimension, size, finiteness or support is invalid.
        FloatingPointError
            If CDF evaluation, required log tails or quantiles are numerically
            invalid, or the statistic exceeds float64 range.

        Notes
        -----
        No mathematically infinite result occurs for admissible finite data.
        No p-value or finite-sample correction is returned.
        """
        sorted_rvs = self._prepare_sample(rvs)
        cdf_vals = self._cdf(sorted_rvs)
        return KSStatistic.do_execute_statistic(self, sorted_rvs, cdf_vals)


class AndersonDarlingInverseGammaGofStatistic(AbstractInverseGammaGofStatistic, ADStatistic):
    """Anderson--Darling statistic for a fixed Inverse Gamma CDF.

    Parameters
    ----------
    parameters : ParameterValues
        Values with alpha, beta fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar without modifying the input or retaining estimates.
    hypothesis()
        Return fixed parameters; omitted parameters are unknown.
    alternative()
        Return the right critical tail.

    Notes
    -----
    The iid null model has density beta**alpha/Gamma(alpha) *
    x**(-alpha-1)*exp(-beta/x), x > 0, with location fixed at zero.
    Write u_i=F(x_(i)), i=1,...,n, for sorted observations.
    A2 = -n - sum((2*i-1)/n * (log(u_i)+log(1-u_(n+1-i)))).
    Log-CDF and log-survival are evaluated directly without probability clipping.

    Both parameters are fixed; no fitting is performed. Under the continuous
    null the probability transform removes alpha and beta from the null law,
    which still depends on n. Large values reject. Ties and constants are
    accepted; rounded data need separate calibration. The built-in Monte
    Carlo resolver has no Inverse Gamma generator; simulate externally.
    Finite positive observations give mathematically finite statistics.
    Numerical log-tail underflow or overflow raises FloatingPointError,
    rather than replacing probabilities with epsilon. Ordinary CDF values
    and spacings may still round in extreme tails.
    The reference concerns the general statistic, applied here through the
    fixed CDF or cell probabilities, not an Inverse Gamma fitted test.

    References
    ----------
    .. [1] Anderson, T. W. and Darling, D. A. (1952). Asymptotic Theory of
       Certain "Goodness of Fit" Criteria Based on Stochastic Processes.
       Ann. Math. Statist. 23, 193-212. https://doi.org/10.1214/aoms/1177729437

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'alpha': 3.0, 'beta': 2.0})
    >>> statistic = AndersonDarlingInverseGammaGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.4, 0.7, 1.2, 2.0])
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def short_code() -> str:
        return "AD"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic defined in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real strictly positive observations. At least one value;
            ties and constant samples are allowed.
        **kwargs : dict
            Unused compatibility arguments.

        Returns
        -------
        float or numpy.float64
            Scalar statistic; large values reject the null.

        Raises
        ------
        ValueError
            If sample dimension, size, finiteness or support is invalid.
        FloatingPointError
            If CDF evaluation, required log tails or quantiles are numerically
            invalid, or the statistic exceeds float64 range.

        Notes
        -----
        No mathematically infinite result occurs for admissible finite data.
        No p-value or finite-sample correction is returned.
        """
        sorted_rvs = self._prepare_sample(rvs)

        log_cdf, log_sf = self._log_probabilities(sorted_rvs)

        return super().do_execute_statistic(sorted_rvs, log_cdf=log_cdf, log_sf=log_sf)


class CramerVonMisesInverseGammaGofStatistic(
    AbstractInverseGammaGofStatistic, CrammerVonMisesStatistic
):
    """Cramer--von Mises statistic for a fixed Inverse Gamma CDF.

    Parameters
    ----------
    parameters : ParameterValues
        Values with alpha, beta fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar without modifying the input or retaining estimates.
    hypothesis()
        Return fixed parameters; omitted parameters are unknown.
    alternative()
        Return the right critical tail.

    Notes
    -----
    The iid null model has density beta**alpha/Gamma(alpha) *
    x**(-alpha-1)*exp(-beta/x), x > 0, with location fixed at zero.
    Write u_i=F(x_(i)), i=1,...,n, for sorted observations.
    W2 = 1/(12*n) + sum((u_i-(2*i-1)/(2*n))**2).
    No finite-sample adjustment is applied.

    Both parameters are fixed; no fitting is performed. Under the continuous
    null the probability transform removes alpha and beta from the null law,
    which still depends on n. Large values reject. Ties and constants are
    accepted; rounded data need separate calibration. The built-in Monte
    Carlo resolver has no Inverse Gamma generator; simulate externally.
    Finite positive observations give mathematically finite statistics.
    Numerical log-tail underflow or overflow raises FloatingPointError,
    rather than replacing probabilities with epsilon. Ordinary CDF values
    and spacings may still round in extreme tails.
    The reference concerns the general statistic, applied here through the
    fixed CDF or cell probabilities, not an Inverse Gamma fitted test.

    References
    ----------
    .. [1] Anderson, T. W. and Darling, D. A. (1952). Asymptotic Theory of
       Certain "Goodness of Fit" Criteria Based on Stochastic Processes.
       Ann. Math. Statist. 23, 193-212. https://doi.org/10.1214/aoms/1177729437

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'alpha': 3.0, 'beta': 2.0})
    >>> statistic = CramerVonMisesInverseGammaGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.4, 0.7, 1.2, 2.0])
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def short_code() -> str:
        return "CVM"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic defined in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real strictly positive observations. At least one value;
            ties and constant samples are allowed.
        **kwargs : dict
            Unused compatibility arguments.

        Returns
        -------
        float or numpy.float64
            Scalar statistic; large values reject the null.

        Raises
        ------
        ValueError
            If sample dimension, size, finiteness or support is invalid.
        FloatingPointError
            If CDF evaluation, required log tails or quantiles are numerically
            invalid, or the statistic exceeds float64 range.

        Notes
        -----
        No mathematically infinite result occurs for admissible finite data.
        No p-value or finite-sample correction is returned.
        """
        sorted_rvs = self._prepare_sample(rvs)

        cdf_vals = self._cdf(sorted_rvs)

        return CrammerVonMisesStatistic.do_execute_statistic(self, sorted_rvs, cdf_vals)


class AbstractBinnedInverseGammaGofStatistic(AbstractInverseGammaGofStatistic, Chi2Statistic, ABC):
    lambda_value: float = 1.0

    def __init__(self, parameters: ParameterValues, *, bins: int = 8):
        if (
            isinstance(bins, (bool, np.bool_))
            or not isinstance(bins, (int, np.integer))
            or bins < 2
        ):
            raise ValueError("At least two bins are required for binned Inverse Gamma statistics.")
        self.bins = bins
        AbstractInverseGammaGofStatistic.__init__(self, parameters)
        self.lambda_value = getattr(self, "lambda_value", 1.0)

    def _counts_and_expected(self, rvs):

        sample = self._prepare_sample(rvs)
        n = sample.size
        quantiles = np.linspace(0.0, 1.0, self.bins + 1)
        edges = scipy_stats.invgamma.ppf(quantiles, a=self.alpha, scale=self.beta)
        if not np.all(np.isfinite(edges[1:-1])) or np.any(np.diff(edges) <= 0):
            raise FloatingPointError("Inverse Gamma quantile bins are numerically degenerate")
        edges[0] = 0.0
        edges[-1] = np.inf
        counts, _ = np.histogram(sample, bins=edges)
        expected = np.full(self.bins, n / self.bins)

        return counts, expected

    @override
    def execute_statistic(self, rvs, **kwargs):
        counts, expected = self._counts_and_expected(rvs)
        return float(
            Chi2Statistic.do_execute_statistic(self, counts, expected, lambda_=self.lambda_value)
        )


class Chi2PearsonInverseGammaGofStatistic(AbstractBinnedInverseGammaGofStatistic):
    """Pearson statistic using fixed equiprobable Inverse Gamma bins.

    Parameters
    ----------
    parameters : ParameterValues
        Values with alpha, beta fixed; omitted parameters are unknown.
    bins : int, default: 8
        Number of fixed equiprobable cells, at least two; booleans rejected.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar without modifying the input or retaining estimates.
    hypothesis()
        Return fixed parameters; omitted parameters are unknown.
    alternative()
        Return the right critical tail.

    Notes
    -----
    The iid null model has density beta**alpha/Gamma(alpha) *
    x**(-alpha-1)*exp(-beta/x), x > 0, with location fixed at zero.
    Write u_i=F(x_(i)), i=1,...,n, for sorted observations.
    X2 = sum((O_j-n/bins)**2/(n/bins)), where O_j are histogram counts
    between fixed null quantiles j/bins. Interior edges belong to the right
    cell. Zero counts are allowed. The asymptotic reference has bins-1 degrees
    of freedom when expected counts are sufficiently large; sparse samples
    require multinomial simulation. Storage calibration is blocked because
    its key omits bins. The finite-sample null also depends on bins.

    Both parameters are fixed; no fitting is performed. Under the continuous
    null the probability transform removes alpha and beta from the null law,
    which still depends on n. Large values reject. Ties and constants are
    accepted; rounded data need separate calibration. The built-in Monte
    Carlo resolver has no Inverse Gamma generator; simulate externally.
    Finite positive observations give mathematically finite statistics.
    Numerical log-tail underflow or overflow raises FloatingPointError,
    rather than replacing probabilities with epsilon. Ordinary CDF values
    and spacings may still round in extreme tails.
    The reference concerns the general statistic, applied here through the
    fixed CDF or cell probabilities, not an Inverse Gamma fitted test.

    References
    ----------
    .. [1] Pearson, K. (1900). On the criterion that a given system of
       deviations from the probable in the case of a correlated system of
       variables is such that it can be reasonably supposed to have arisen
       from random sampling. Philosophical Magazine 50, 157-175.
       https://doi.org/10.1080/14786440009463897

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'alpha': 3.0, 'beta': 2.0})
    >>> statistic = Chi2PearsonInverseGammaGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.4, 0.7, 1.2, 2.0])
    >>> bool(np.isfinite(value))
    True
    """

    lambda_value = 1.0

    def _validate_storage_calibration(self):
        raise ValueError("Stored Pearson calibration does not encode bins; calibrate externally")

    @staticmethod
    @override
    def short_code() -> str:

        return "CHI2_PEARSON"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic defined in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real strictly positive observations. At least one value;
            ties and constant samples are allowed.
        **kwargs : dict
            Unused compatibility arguments.

        Returns
        -------
        float or numpy.float64
            Scalar statistic; large values reject the null.

        Raises
        ------
        ValueError
            If sample dimension, size, finiteness or support is invalid.
        FloatingPointError
            If CDF evaluation, required log tails or quantiles are numerically
            invalid, or the statistic exceeds float64 range.

        Notes
        -----
        No mathematically infinite result occurs for admissible finite data.
        No p-value or finite-sample correction is returned.
        """
        return super().execute_statistic(rvs)


class WatsonInverseGammaGofStatistic(AbstractInverseGammaGofStatistic):
    """Watson centered Cramer--von Mises statistic after the fixed CDF.

    Parameters
    ----------
    parameters : ParameterValues
        Values with alpha, beta fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar without modifying the input or retaining estimates.
    hypothesis()
        Return fixed parameters; omitted parameters are unknown.
    alternative()
        Return the right critical tail.

    Notes
    -----
    The iid null model has density beta**alpha/Gamma(alpha) *
    x**(-alpha-1)*exp(-beta/x), x > 0, with location fixed at zero.
    Write u_i=F(x_(i)), i=1,...,n, for sorted observations.
    With d_i = u_i-(2*i-1)/(2*n), return
    U2 = 1/(12*n) + sum((d_i-mean(d))**2), equivalently
    W2-n*(mean(u)-1/2)**2. Centering avoids cancellation. No correction is used.

    Both parameters are fixed; no fitting is performed. Under the continuous
    null the probability transform removes alpha and beta from the null law,
    which still depends on n. Large values reject. Ties and constants are
    accepted; rounded data need separate calibration. The built-in Monte
    Carlo resolver has no Inverse Gamma generator; simulate externally.
    Finite positive observations give mathematically finite statistics.
    Numerical log-tail underflow or overflow raises FloatingPointError,
    rather than replacing probabilities with epsilon. Ordinary CDF values
    and spacings may still round in extreme tails.
    The reference concerns the general statistic, applied here through the
    fixed CDF or cell probabilities, not an Inverse Gamma fitted test.

    References
    ----------
    .. [1] Watson, G. S. (1961). Goodness-of-fit tests on a circle.
       Biometrika 48, 109-114. https://doi.org/10.1093/biomet/48.1-2.109

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'alpha': 3.0, 'beta': 2.0})
    >>> statistic = WatsonInverseGammaGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.4, 0.7, 1.2, 2.0])
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        return "WAT"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic defined in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real strictly positive observations. At least one value;
            ties and constant samples are allowed.
        **kwargs : dict
            Unused compatibility arguments.

        Returns
        -------
        float or numpy.float64
            Scalar statistic; large values reject the null.

        Raises
        ------
        ValueError
            If sample dimension, size, finiteness or support is invalid.
        FloatingPointError
            If CDF evaluation, required log tails or quantiles are numerically
            invalid, or the statistic exceeds float64 range.

        Notes
        -----
        No mathematically infinite result occurs for admissible finite data.
        No p-value or finite-sample correction is returned.
        """
        sorted_rvs = self._prepare_sample(rvs)
        n = len(sorted_rvs)

        cdf_vals = self._cdf(sorted_rvs)

        u = (2 * np.arange(1, n + 1) - 1) / (2 * n)
        diff = cdf_vals - u
        return float(1.0 / (12 * n) + np.sum((diff - np.mean(diff)) ** 2))


class KuiperInverseGammaGofStatistic(AbstractInverseGammaGofStatistic):
    """Kuiper statistic after the fixed Inverse Gamma CDF.

    Parameters
    ----------
    parameters : ParameterValues
        Values with alpha, beta fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar without modifying the input or retaining estimates.
    hypothesis()
        Return fixed parameters; omitted parameters are unknown.
    alternative()
        Return the right critical tail.

    Notes
    -----
    The iid null model has density beta**alpha/Gamma(alpha) *
    x**(-alpha-1)*exp(-beta/x), x > 0, with location fixed at zero.
    Write u_i=F(x_(i)), i=1,...,n, for sorted observations.
    V = max(i/n-u_i) + max(u_i-(i-1)/n).
    This applies the circular uniformity statistic to the probability transform.
    No sqrt(n) scaling or finite-sample correction is applied.

    Both parameters are fixed; no fitting is performed. Under the continuous
    null the probability transform removes alpha and beta from the null law,
    which still depends on n. Large values reject. Ties and constants are
    accepted; rounded data need separate calibration. The built-in Monte
    Carlo resolver has no Inverse Gamma generator; simulate externally.
    Finite positive observations give mathematically finite statistics.
    Numerical log-tail underflow or overflow raises FloatingPointError,
    rather than replacing probabilities with epsilon. Ordinary CDF values
    and spacings may still round in extreme tails.
    The reference concerns the general statistic, applied here through the
    fixed CDF or cell probabilities, not an Inverse Gamma fitted test.

    References
    ----------
    .. [1] Kuiper, N. H. (1960). Tests concerning random points on a circle.
       Indagationes Mathematicae (Proceedings) 63, 38-47.
       https://doi.org/10.1016/S1385-7258(60)50006-0

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'alpha': 3.0, 'beta': 2.0})
    >>> statistic = KuiperInverseGammaGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.4, 0.7, 1.2, 2.0])
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        return "KUI"

    @classmethod
    @override
    def code(cls) -> str:
        """Preserve the legacy identifier suffix while using the subclass short code."""
        return f"{cls.short_code()}_INVGAMMA_INV_GAMMA_GOODNESS_OF_FIT"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic defined in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real strictly positive observations. At least one value;
            ties and constant samples are allowed.
        **kwargs : dict
            Unused compatibility arguments.

        Returns
        -------
        float or numpy.float64
            Scalar statistic; large values reject the null.

        Raises
        ------
        ValueError
            If sample dimension, size, finiteness or support is invalid.
        FloatingPointError
            If CDF evaluation, required log tails or quantiles are numerically
            invalid, or the statistic exceeds float64 range.

        Notes
        -----
        No mathematically infinite result occurs for admissible finite data.
        No p-value or finite-sample correction is returned.
        """
        sorted_rvs = self._prepare_sample(rvs)
        n = len(sorted_rvs)

        cdf_vals = self._cdf(sorted_rvs)

        i = np.arange(1, n + 1)
        d_plus = np.max(i / n - cdf_vals)
        d_minus = np.max(cdf_vals - (i - 1) / n)
        return float(d_plus + d_minus)


class MinToshiyukiInverseGammaGofStatistic(AbstractInverseGammaGofStatistic, MinToshiyukiStatistic):
    """Locally defined tail-weighted EDF discrepancy for Inverse Gamma.

    Parameters
    ----------
    parameters : ParameterValues
        Values with alpha, beta fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar without modifying the input or retaining estimates.
    hypothesis()
        Return fixed parameters; omitted parameters are unknown.
    alternative()
        Return the right critical tail.

    Notes
    -----
    The iid null model has density beta**alpha/Gamma(alpha) *
    x**(-alpha-1)*exp(-beta/x), x > 0, with location fixed at zero.
    Write u_i=F(x_(i)), i=1,...,n, for sorted observations.
    T = sum(max(i/n-u_i, u_i-(i-1)/n)/sqrt(u_i*(1-u_i)))/sqrt(n).
    The weights use direct log-CDF and log-survival to avoid cancellation.
    No primary source verifying this exact formula under the legacy name
    Min--Toshiyuki was found. Work by Liao and Shimokawa on Weibull and
    extreme-value tests does not establish this Inverse Gamma procedure.
    Treat this as a local discrepancy requiring its own calibration.

    Both parameters are fixed; no fitting is performed. Under the continuous
    null the probability transform removes alpha and beta from the null law,
    which still depends on n. Large values reject. Ties and constants are
    accepted; rounded data need separate calibration. The built-in Monte
    Carlo resolver has no Inverse Gamma generator; simulate externally.
    Finite positive observations give mathematically finite statistics.
    Numerical log-tail underflow or overflow raises FloatingPointError,
    rather than replacing probabilities with epsilon. Ordinary CDF values
    and spacings may still round in extreme tails.

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'alpha': 3.0, 'beta': 2.0})
    >>> statistic = MinToshiyukiInverseGammaGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.4, 0.7, 1.2, 2.0])
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def short_code():
        return "MT"

    @classmethod
    @override
    def code(cls) -> str:
        """Preserve the legacy identifier suffix while using the subclass short code."""
        return f"{cls.short_code()}_INVGAMMA_INV_GAMMA_GOODNESS_OF_FIT"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic defined in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real strictly positive observations. At least one value;
            ties and constant samples are allowed.
        **kwargs : dict
            Unused compatibility arguments.

        Returns
        -------
        float or numpy.float64
            Scalar statistic; large values reject the null.

        Raises
        ------
        ValueError
            If sample dimension, size, finiteness or support is invalid.
        FloatingPointError
            If CDF evaluation, required log tails or quantiles are numerically
            invalid, or the statistic exceeds float64 range.

        Notes
        -----
        No mathematically infinite result occurs for admissible finite data.
        No p-value or finite-sample correction is returned.
        """
        sample = self._prepare_sample(rvs)
        cdf = self._cdf(sample)
        log_cdf, log_sf = self._log_probabilities(sample)
        n = sample.size
        i = np.arange(1, n + 1)
        deviations = np.maximum(i / n - cdf, cdf - (i - 1) / n)
        with np.errstate(over="ignore"):
            result = np.sum(deviations * np.exp(-0.5 * (log_cdf + log_sf))) / np.sqrt(n)
        if not np.isfinite(result):
            raise FloatingPointError("Weighted EDF statistic exceeds float64 range")
        return float(result)


class GreenwoodInverseGammaGofStatistic(AbstractInverseGammaGofStatistic):
    """Greenwood sum of squared spacings after the fixed CDF.

    Parameters
    ----------
    parameters : ParameterValues
        Values with alpha, beta fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar without modifying the input or retaining estimates.
    hypothesis()
        Return fixed parameters; omitted parameters are unknown.
    alternative()
        Return the right critical tail.

    Notes
    -----
    The iid null model has density beta**alpha/Gamma(alpha) *
    x**(-alpha-1)*exp(-beta/x), x > 0, with location fixed at zero.
    Write u_i=F(x_(i)), i=1,...,n, for sorted observations.
    G = sum((u_(i+1)-u_i)**2), i=0,...,n, with u_0=0 and u_(n+1)=1.
    Include both endpoint gaps. This is the unscaled sum, with null mean
    2/(n+2); large values detect clustering. No normal approximation is used.

    Both parameters are fixed; no fitting is performed. Under the continuous
    null the probability transform removes alpha and beta from the null law,
    which still depends on n. Large values reject. Ties and constants are
    accepted; rounded data need separate calibration. The built-in Monte
    Carlo resolver has no Inverse Gamma generator; simulate externally.
    Finite positive observations give mathematically finite statistics.
    Numerical log-tail underflow or overflow raises FloatingPointError,
    rather than replacing probabilities with epsilon. Ordinary CDF values
    and spacings may still round in extreme tails.
    The reference concerns the general statistic, applied here through the
    fixed CDF or cell probabilities, not an Inverse Gamma fitted test.

    References
    ----------
    .. [1] Greenwood, M. (1946). The statistical study of infectious diseases.
       J. R. Stat. Soc. 109, 85-110, discussion of random intervals, pp. 99-101.
       https://www.ime.usp.br/~abe/lista/pdfr62zHOGxek.pdf

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'alpha': 3.0, 'beta': 2.0})
    >>> statistic = GreenwoodInverseGammaGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.4, 0.7, 1.2, 2.0])
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        return "GRW"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic defined in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real strictly positive observations. At least one value;
            ties and constant samples are allowed.
        **kwargs : dict
            Unused compatibility arguments.

        Returns
        -------
        float or numpy.float64
            Scalar statistic; large values reject the null.

        Raises
        ------
        ValueError
            If sample dimension, size, finiteness or support is invalid.
        FloatingPointError
            If CDF evaluation, required log tails or quantiles are numerically
            invalid, or the statistic exceeds float64 range.

        Notes
        -----
        No mathematically infinite result occurs for admissible finite data.
        No p-value or finite-sample correction is returned.
        """

        rvs_array = self._prepare_sample(rvs)
        sorted_rvs = np.sort(rvs_array)
        cdf_vals = self._cdf(sorted_rvs)
        spacings = np.diff(np.concatenate(([0.0], cdf_vals, [1.0])))

        if np.any(spacings < 0):
            raise ValueError("Spacings must be non-negative; check input data ordering.")

        if np.any(spacings > 1):
            raise ValueError("Spacings must be <= 1; check CDF values.")

        return float(np.sum(spacings**2))


class ZhangAInverseGammaGofStatistic(AbstractInverseGammaGofStatistic):
    """Zhang ZA likelihood-ratio statistic after the fixed CDF.

    Parameters
    ----------
    parameters : ParameterValues
        Values with alpha, beta fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar without modifying the input or retaining estimates.
    hypothesis()
        Return fixed parameters; omitted parameters are unknown.
    alternative()
        Return the right critical tail.

    Notes
    -----
    The iid null model has density beta**alpha/Gamma(alpha) *
    x**(-alpha-1)*exp(-beta/x), x > 0, with location fixed at zero.
    Write u_i=F(x_(i)), i=1,...,n, for sorted observations.
    ZA = -sum(log(u_i)/(n-i+1/2) + log(1-u_i)/(i-1/2)).
    Large values reject. Log tails are evaluated directly; epsilon clipping
    has been removed and passing epsilon raises TypeError.

    Both parameters are fixed; no fitting is performed. Under the continuous
    null the probability transform removes alpha and beta from the null law,
    which still depends on n. Large values reject. Ties and constants are
    accepted; rounded data need separate calibration. The built-in Monte
    Carlo resolver has no Inverse Gamma generator; simulate externally.
    Finite positive observations give mathematically finite statistics.
    Numerical log-tail underflow or overflow raises FloatingPointError,
    rather than replacing probabilities with epsilon. Ordinary CDF values
    and spacings may still round in extreme tails.
    The reference concerns the general statistic, applied here through the
    fixed CDF or cell probabilities, not an Inverse Gamma fitted test.

    References
    ----------
    .. [1] Zhang, J. (2002). Powerful Goodness-of-fit Tests Based on the
       Likelihood Ratio. JRSS B 64, 281-294.
       https://doi.org/10.1111/1467-9868.00337

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'alpha': 3.0, 'beta': 2.0})
    >>> statistic = ZhangAInverseGammaGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.4, 0.7, 1.2, 2.0])
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        return "ZAA"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic defined in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real strictly positive observations. At least one value;
            ties and constant samples are allowed.
        **kwargs : dict
            Unused compatibility arguments. Passing epsilon raises TypeError.

        Returns
        -------
        float or numpy.float64
            Scalar statistic; large values reject the null.

        Raises
        ------
        ValueError
            If sample dimension, size, finiteness or support is invalid.
        FloatingPointError
            If CDF evaluation, required log tails or quantiles are numerically
            invalid, or the statistic exceeds float64 range.
        TypeError
            If the removed epsilon setting is supplied.

        Notes
        -----
        No mathematically infinite result occurs for admissible finite data.
        No p-value or finite-sample correction is returned.
        """
        if "epsilon" in kwargs:
            raise TypeError("epsilon clipping is no longer supported")
        sample = self._prepare_sample(rvs)
        log_cdf, log_sf = self._log_probabilities(sample)
        n = sample.size
        i = np.arange(1, n + 1)
        return float(-np.sum(log_cdf / (n - i + 0.5) + log_sf / (i - 0.5)))


class ZhangCInverseGammaGofStatistic(AbstractInverseGammaGofStatistic):
    """Zhang ZC likelihood-ratio statistic after the fixed CDF.

    Parameters
    ----------
    parameters : ParameterValues
        Values with alpha, beta fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar without modifying the input or retaining estimates.
    hypothesis()
        Return fixed parameters; omitted parameters are unknown.
    alternative()
        Return the right critical tail.

    Notes
    -----
    The iid null model has density beta**alpha/Gamma(alpha) *
    x**(-alpha-1)*exp(-beta/x), x > 0, with location fixed at zero.
    Write u_i=F(x_(i)), i=1,...,n, for sorted observations.
    ZC = sum((log((1-u_i)/u_i)-log((n-i+1/4)/(i-3/4)))**2).
    Large values reject. Direct log tails avoid subtracting a rounded CDF
    from one. Passing the removed epsilon setting raises TypeError.

    Both parameters are fixed; no fitting is performed. Under the continuous
    null the probability transform removes alpha and beta from the null law,
    which still depends on n. Large values reject. Ties and constants are
    accepted; rounded data need separate calibration. The built-in Monte
    Carlo resolver has no Inverse Gamma generator; simulate externally.
    Finite positive observations give mathematically finite statistics.
    Numerical log-tail underflow or overflow raises FloatingPointError,
    rather than replacing probabilities with epsilon. Ordinary CDF values
    and spacings may still round in extreme tails.
    The reference concerns the general statistic, applied here through the
    fixed CDF or cell probabilities, not an Inverse Gamma fitted test.

    References
    ----------
    .. [1] Zhang, J. (2002). Powerful Goodness-of-fit Tests Based on the
       Likelihood Ratio. JRSS B 64, 281-294.
       https://doi.org/10.1111/1467-9868.00337

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'alpha': 3.0, 'beta': 2.0})
    >>> statistic = ZhangCInverseGammaGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.4, 0.7, 1.2, 2.0])
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        return "ZAC"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic defined in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real strictly positive observations. At least one value;
            ties and constant samples are allowed.
        **kwargs : dict
            Unused compatibility arguments. Passing epsilon raises TypeError.

        Returns
        -------
        float or numpy.float64
            Scalar statistic; large values reject the null.

        Raises
        ------
        ValueError
            If sample dimension, size, finiteness or support is invalid.
        FloatingPointError
            If CDF evaluation, required log tails or quantiles are numerically
            invalid, or the statistic exceeds float64 range.
        TypeError
            If the removed epsilon setting is supplied.

        Notes
        -----
        No mathematically infinite result occurs for admissible finite data.
        No p-value or finite-sample correction is returned.
        """
        if "epsilon" in kwargs:
            raise TypeError("epsilon clipping is no longer supported")
        sample = self._prepare_sample(rvs)
        log_cdf, log_sf = self._log_probabilities(sample)
        n = sample.size
        i = np.arange(1, n + 1)
        log_odds = np.log(n - i + 0.25) - np.log(i - 0.75)
        with np.errstate(over="ignore"):
            result = np.sum((log_sf - log_cdf - log_odds) ** 2)
        if not np.isfinite(result):
            raise FloatingPointError("Zhang C statistic exceeds float64 range")
        return float(result)


class ZhangKInverseGammaGofStatistic(AbstractInverseGammaGofStatistic):
    """Zhang ZK likelihood-ratio statistic after the fixed CDF.

    Parameters
    ----------
    parameters : ParameterValues
        Values with alpha, beta fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar without modifying the input or retaining estimates.
    hypothesis()
        Return fixed parameters; omitted parameters are unknown.
    alternative()
        Return the right critical tail.

    Notes
    -----
    The iid null model has density beta**alpha/Gamma(alpha) *
    x**(-alpha-1)*exp(-beta/x), x > 0, with location fixed at zero.
    Write u_i=F(x_(i)), i=1,...,n, for sorted observations.
    With p_i=(i-1/2)/n, ZK = n*max(p_i*log(p_i/u_i)
    + (1-p_i)*log((1-p_i)/(1-u_i))). Large values reject.
    Direct log tails replace probability clipping. Passing epsilon raises TypeError.

    Both parameters are fixed; no fitting is performed. Under the continuous
    null the probability transform removes alpha and beta from the null law,
    which still depends on n. Large values reject. Ties and constants are
    accepted; rounded data need separate calibration. The built-in Monte
    Carlo resolver has no Inverse Gamma generator; simulate externally.
    Finite positive observations give mathematically finite statistics.
    Numerical log-tail underflow or overflow raises FloatingPointError,
    rather than replacing probabilities with epsilon. Ordinary CDF values
    and spacings may still round in extreme tails.
    The reference concerns the general statistic, applied here through the
    fixed CDF or cell probabilities, not an Inverse Gamma fitted test.

    References
    ----------
    .. [1] Zhang, J. (2002). Powerful Goodness-of-fit Tests Based on the
       Likelihood Ratio. JRSS B 64, 281-294.
       https://doi.org/10.1111/1467-9868.00337

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'alpha': 3.0, 'beta': 2.0})
    >>> statistic = ZhangKInverseGammaGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.4, 0.7, 1.2, 2.0])
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        return "ZAK"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic defined in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real strictly positive observations. At least one value;
            ties and constant samples are allowed.
        **kwargs : dict
            Unused compatibility arguments. Passing epsilon raises TypeError.

        Returns
        -------
        float or numpy.float64
            Scalar statistic; large values reject the null.

        Raises
        ------
        ValueError
            If sample dimension, size, finiteness or support is invalid.
        FloatingPointError
            If CDF evaluation, required log tails or quantiles are numerically
            invalid, or the statistic exceeds float64 range.
        TypeError
            If the removed epsilon setting is supplied.

        Notes
        -----
        No mathematically infinite result occurs for admissible finite data.
        No p-value or finite-sample correction is returned.
        """
        if "epsilon" in kwargs:
            raise TypeError("epsilon clipping is no longer supported")
        sample = self._prepare_sample(rvs)
        log_cdf, log_sf = self._log_probabilities(sample)
        n = sample.size
        i = np.arange(1, n + 1)
        p = (i - 0.5) / n
        return float(np.max(n * (p * (np.log(p) - log_cdf) + (1 - p) * (np.log1p(-p) - log_sf))))
