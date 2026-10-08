from __future__ import annotations

from abc import ABC

import numpy as np
import scipy.stats as scipy_stats
from typing_extensions import override

from pysatl_criterion.distribution.distributions import GammaDistributionDescriptor
from pysatl_criterion.distribution.distributions import GammaDistributionDescriptor as Distribution
from pysatl_criterion.distribution.parameters import HypothesisSupport, ParameterValues
from pysatl_criterion.statistics import AbstractGoodnessOfFitStatistic
from pysatl_criterion.statistics.alternative import (
    Alternative,
    AlternativeType,
    RightAlternative,
    TwoSidedAlternative,
)
from pysatl_criterion.statistics.goodness_of_fit.common import (
    ADStatistic,
    Chi2Statistic,
    CrammerVonMisesStatistic,
    KSStatistic,
    MinToshiyukiStatistic,
)
from pysatl_criterion.statistics.goodness_of_fit.graph_goodness_of_fit import (
    AbstractGraphTestStatistic,
    GraphAverageDegreeTestStatistic,
    GraphCliqueNumberTestStatistic,
    GraphConnectedComponentsTestStatistic,
    GraphEdgesNumberTestStatistic,
    GraphIndependenceNumberTestStatistic,
    GraphMaxDegreeTestStatistic,
)


def _scalar(value, name):
    """Return a finite real scalar setting."""
    if (
        isinstance(value, (bool, np.bool_, str, bytes))
        or np.ndim(value) != 0
        or np.iscomplexobj(value)
    ):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real scalar")
    return value


def _sample(rvs, minimum=1):
    """Validate the closed Gamma support, retaining zero as a boundary value."""
    if np.iscomplexobj(rvs) or (np.ma.isMaskedArray(rvs) and np.any(rvs.mask)):
        raise ValueError("Sample must be real and contain no masked observations")
    try:
        sample = np.array(rvs, dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Sample must be numeric") from exc
    if sample.ndim != 1:
        raise ValueError("Sample must be one-dimensional")
    if sample.size == 0:
        raise ValueError("At least one observation is required")
    if sample.size < minimum:
        raise ValueError("At least two observations are required")
    if not np.all(np.isfinite(sample)) or np.any(sample < 0):
        raise ValueError("Sample must contain finite nonnegative observations")
    return sample


class AbstractGammaGofStatistic(AbstractGoodnessOfFitStatistic, ABC):
    """
    Abstract base class for Gamma distribution goodness-of-fit statistics.
    """

    @property
    def alpha(self) -> float:
        """Read alpha by its stable parameter identity."""
        return self._parameters[Distribution.SHAPE]

    @property
    def beta(self) -> float:
        """Read beta by its stable parameter identity."""
        return self._parameters[Distribution.RATE]

    def _scaled(self, sample):
        with np.errstate(over="ignore", under="ignore"):
            return sample * self.beta

    def _cdf(self, sample):
        return scipy_stats.gamma.cdf(self._scaled(sample), a=self.alpha)

    def _logs(self, sample):
        scaled = self._scaled(sample)
        return (
            scipy_stats.gamma.logcdf(scaled, a=self.alpha),
            scipy_stats.gamma.logsf(scaled, a=self.alpha),
        )

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.RATE})
            ),
        )

    @staticmethod
    @override
    def distribution() -> type[GammaDistributionDescriptor]:
        """Return the distribution descriptor class."""
        return GammaDistributionDescriptor

    @classmethod
    @override
    def code(cls) -> str:
        """Return the family identifier or the concrete statistic's full identifier."""
        family_code = f"GAMMA_{AbstractGoodnessOfFitStatistic.code()}"
        if "short_code" in cls.__abstractmethods__:
            return family_code
        return f"{cls.short_code()}_{family_code}"


class KolmogorovSmirnovGammaGofStatistic(AbstractGammaGofStatistic, KSStatistic):
    """KS distance to a fixed Gamma CDF.

    Parameters
    ----------
    parameters : ParameterValues
        Values with alfa, beta fixed; omitted parameters are unknown.
    alternative_type : AlternativeType, optional
        TWO_TAILED (default), RIGHT or LEFT selects D, D+ or D-.
    mode : {"auto", "exact", "asymp", "approx"}, optional
        Compatibility setting; no effect on the scalar statistic.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute one scalar statistic without changing the sample.
    hypothesis()
        Return fixed null parameters; omitted keys mean unknown parameters.

    Notes
    -----
    For ordered observations x_(i), let u_i=F(x_(i); alpha, beta),
    where Gamma has shape alpha, rate beta and known origin zero.
    Parameters are fixed, not fitted. Samples must be finite, real,
    one-dimensional and nonempty, in the closed support [0, infinity).
    Zero is accepted as a boundary value. Ties and constant samples are
    allowed except where stated. Inputs and instance state are unchanged.
    Continuous-null calibration assumes iid observations; rounded data
    require calibration of the observation process.

    D+ = max(i/n-u_i), D- = max(u_i-(i-1)/n), i=1,...,n.
    Return max(D+,D-), D+, or D- for TWO_TAILED, RIGHT, or LEFT.
    All three reject for large values. No parameters are fitted.
    Stored calibration supports only the default two-sided CDF distance.

    The fixed-CDF probability transform gives a parameter-free null law
    in exact arithmetic. Reject for large values. The referenced general
    statistic is applied through the Gamma CDF, not a special Gamma table.

    References
    ----------
    .. [1] N. Smirnov (1948), "Table for Estimating the Goodness of Fit of Empirical
       Distributions", Ann. Math. Statist. 19, 279-281.
       https://doi.org/10.1214/aoms/1177730256

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'alfa': 1, 'beta': 1})
    >>> statistic = KolmogorovSmirnovGammaGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
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
        AbstractGammaGofStatistic.__init__(self, parameters)
        if alternative_type not in tuple(AlternativeType):
            raise ValueError("Invalid CDF deviation direction")
        if mode not in ("auto", "exact", "asymp", "approx"):
            raise ValueError("Invalid KS mode")
        KSStatistic.__init__(self, alternative_type=alternative_type, mode=mode)

    def _validate_storage_calibration(self):
        if self.alternative_type != AlternativeType.TWO_TAILED:
            raise ValueError("Stored KS calibration does not encode the CDF direction")

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "KS".
        """
        return "KS"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the scalar statistic defined in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real nonnegative observations. Zero is a support boundary.
            At least one observation; ties and constant samples are allowed.
        **kwargs : dict, optional
            Reserved for interface compatibility; ignored.

        Returns
        -------
        statistic : float or numpy.float64
            The statistic in the class Notes. No p-value is calculated.

        Raises
        ------
        ValueError
            If the sample violates the stated shape, support or size constraints,
            or the required estimates/quantiles are numerically degenerate.

        Notes
        -----
        See the class Notes for the exact formula, critical tail and
        calibration requirements.
        """

        sorted_rvs = np.sort(_sample(rvs))
        cdf_vals = self._cdf(sorted_rvs)
        return KSStatistic.do_execute_statistic(self, sorted_rvs, cdf_vals)


class AndersonDarlingGammaGofStatistic(AbstractGammaGofStatistic, ADStatistic):
    """Anderson-Darling statistic for a fixed Gamma CDF.

    Parameters
    ----------
    parameters : ParameterValues
        Values with alfa, beta fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute one scalar statistic without changing the sample.
    hypothesis()
        Return fixed null parameters; omitted keys mean unknown parameters.

    Notes
    -----
    For ordered observations x_(i), let u_i=F(x_(i); alpha, beta),
    where Gamma has shape alpha, rate beta and known origin zero.
    Parameters are fixed, not fitted. Samples must be finite, real,
    one-dimensional and nonempty, in the closed support [0, infinity).
    Zero is accepted as a boundary value. Ties and constant samples are
    allowed except where stated. Inputs and instance state are unchanged.
    Continuous-null calibration assumes iid observations; rounded data
    require calibration of the observation process.

    A2 = -n - sum((2*i-1)*(log(u_i)+log(1-u_(n+1-i))))/n.
    Direct log-CDF and log-survival evaluations retain upper-tail precision.
    Zero observations give positive infinity. Extreme tails may underflow
    in SciPy, also producing infinity; no epsilon clipping is applied.

    The fixed-CDF probability transform gives a parameter-free null law
    in exact arithmetic. Reject for large values. The referenced general
    statistic is applied through the Gamma CDF, not a special Gamma table.

    References
    ----------
    .. [1] T. W. Anderson and D. A. Darling (1952), "Asymptotic Theory of Certain
       Goodness of Fit Criteria Based on Stochastic Processes", Ann. Math.
       Statist. 23, 193-212. https://doi.org/10.1214/aoms/1177729437

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'alfa': 1, 'beta': 1})
    >>> statistic = AndersonDarlingGammaGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "AD".
        """
        return "AD"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the scalar statistic defined in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real nonnegative observations. Zero is a support boundary.
            At least one observation; ties and constant samples are allowed.
        **kwargs : dict, optional
            Reserved for interface compatibility; ignored.

        Returns
        -------
        statistic : float or numpy.float64
            The statistic in the class Notes. No p-value is calculated.

        Raises
        ------
        ValueError
            If the sample violates the stated shape, support or size constraints,
            or the required estimates/quantiles are numerically degenerate.

        Notes
        -----
        Zero observations return positive infinity. Extreme tails can
        underflow in SciPy log-probabilities; no epsilon clipping is applied.
        """

        sorted_rvs = np.sort(_sample(rvs))
        log_cdf, log_sf = self._logs(sorted_rvs)
        return super().do_execute_statistic(sorted_rvs, log_cdf=log_cdf, log_sf=log_sf)


class CramerVonMisesGammaGofStatistic(AbstractGammaGofStatistic, CrammerVonMisesStatistic):
    """Cramer-von Mises statistic for a fixed Gamma CDF.

    Parameters
    ----------
    parameters : ParameterValues
        Values with alfa, beta fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute one scalar statistic without changing the sample.
    hypothesis()
        Return fixed null parameters; omitted keys mean unknown parameters.

    Notes
    -----
    For ordered observations x_(i), let u_i=F(x_(i); alpha, beta),
    where Gamma has shape alpha, rate beta and known origin zero.
    Parameters are fixed, not fitted. Samples must be finite, real,
    one-dimensional and nonempty, in the closed support [0, infinity).
    Zero is accepted as a boundary value. Ties and constant samples are
    allowed except where stated. Inputs and instance state are unchanged.
    Continuous-null calibration assumes iid observations; rounded data
    require calibration of the observation process.

    W2 = 1/(12*n) + sum((u_i-(2*i-1)/(2*n))**2).
    No fitted-parameter correction or additional n scaling is applied.

    The fixed-CDF probability transform gives a parameter-free null law
    in exact arithmetic. Reject for large values. The referenced general
    statistic is applied through the Gamma CDF, not a special Gamma table.

    References
    ----------
    .. [1] T. W. Anderson and D. A. Darling (1952), "Asymptotic Theory of Certain
       Goodness of Fit Criteria Based on Stochastic Processes", Ann. Math.
       Statist. 23, 193-212. https://doi.org/10.1214/aoms/1177729437

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'alfa': 1, 'beta': 1})
    >>> statistic = CramerVonMisesGammaGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "CVM".
        """
        return "CVM"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the scalar statistic defined in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real nonnegative observations. Zero is a support boundary.
            At least one observation; ties and constant samples are allowed.
        **kwargs : dict, optional
            Reserved for interface compatibility; ignored.

        Returns
        -------
        statistic : float or numpy.float64
            The statistic in the class Notes. No p-value is calculated.

        Raises
        ------
        ValueError
            If the sample violates the stated shape, support or size constraints,
            or the required estimates/quantiles are numerically degenerate.

        Notes
        -----
        See the class Notes for the exact formula, critical tail and
        calibration requirements.
        """

        sorted_rvs = np.sort(_sample(rvs))
        cdf_vals = self._cdf(sorted_rvs)
        return CrammerVonMisesStatistic.do_execute_statistic(self, sorted_rvs, cdf_vals)


class WatsonGammaGofStatistic(AbstractGammaGofStatistic):
    """Watson centered EDF statistic after the Gamma transform.

    Parameters
    ----------
    parameters : ParameterValues
        Values with alfa, beta fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute one scalar statistic without changing the sample.
    hypothesis()
        Return fixed null parameters; omitted keys mean unknown parameters.

    Notes
    -----
    For ordered observations x_(i), let u_i=F(x_(i); alpha, beta),
    where Gamma has shape alpha, rate beta and known origin zero.
    Parameters are fixed, not fitted. Samples must be finite, real,
    one-dimensional and nonempty, in the closed support [0, infinity).
    Zero is accepted as a boundary value. Ties and constant samples are
    allowed except where stated. Inputs and instance state are unchanged.
    Continuous-null calibration assumes iid observations; rounded data
    require calibration of the observation process.

    U2 = W2 - n*(mean(u)-1/2)**2, where
    W2 = 1/(12*n) + sum((u_i-(2*i-1)/(2*n))**2).
    Circular rotation invariance concerns transformed uniform values,
    not translation of the original Gamma observations.

    The fixed-CDF probability transform gives a parameter-free null law
    in exact arithmetic. Reject for large values. The referenced general
    statistic is applied through the Gamma CDF, not a special Gamma table.

    References
    ----------
    .. [1] G. S. Watson (1961), "Goodness-of-fit tests on a circle",
       Biometrika 48, 109-114. https://doi.org/10.1093/biomet/48.1-2.109

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'alfa': 1, 'beta': 1})
    >>> statistic = WatsonGammaGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
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

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the scalar statistic defined in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real nonnegative observations. Zero is a support boundary.
            At least one observation; ties and constant samples are allowed.
        **kwargs : dict, optional
            Reserved for interface compatibility; ignored.

        Returns
        -------
        statistic : float or numpy.float64
            The statistic in the class Notes. No p-value is calculated.

        Raises
        ------
        ValueError
            If the sample violates the stated shape, support or size constraints,
            or the required estimates/quantiles are numerically degenerate.

        Notes
        -----
        See the class Notes for the exact formula, critical tail and
        calibration requirements.
        """

        sorted_rvs = np.sort(_sample(rvs))
        n = len(sorted_rvs)

        cdf_vals = self._cdf(sorted_rvs)
        u = (2 * np.arange(1, n + 1) - 1) / (2 * n)
        diff = cdf_vals - u
        w_squared = 1.0 / (12 * n) + np.sum(diff**2)
        mean_adj = np.sum(cdf_vals) - n / 2
        return float(w_squared - (mean_adj**2) / n)


class KuiperGammaGofStatistic(AbstractGammaGofStatistic):
    """Kuiper EDF range after the Gamma transform.

    Parameters
    ----------
    parameters : ParameterValues
        Values with alfa, beta fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute one scalar statistic without changing the sample.
    hypothesis()
        Return fixed null parameters; omitted keys mean unknown parameters.

    Notes
    -----
    For ordered observations x_(i), let u_i=F(x_(i); alpha, beta),
    where Gamma has shape alpha, rate beta and known origin zero.
    Parameters are fixed, not fitted. Samples must be finite, real,
    one-dimensional and nonempty, in the closed support [0, infinity).
    Zero is accepted as a boundary value. Ties and constant samples are
    allowed except where stated. Inputs and instance state are unchanged.
    Continuous-null calibration assumes iid observations; rounded data
    require calibration of the observation process.

    V = max(i/n-u_i) + max(u_i-(i-1)/n), i=1,...,n.
    No finite-sample correction is applied.

    The fixed-CDF probability transform gives a parameter-free null law
    in exact arithmetic. Reject for large values. The referenced general
    statistic is applied through the Gamma CDF, not a special Gamma table.

    References
    ----------
    .. [1] N. H. Kuiper (1960), "Tests concerning random points on a circle",
       Indagationes Mathematicae (Proceedings) 63, 38-47.
       https://doi.org/10.1016/S1385-7258(60)50006-0

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'alfa': 1, 'beta': 1})
    >>> statistic = KuiperGammaGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
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

        :return: short code string "KUI".
        """
        return "KUI"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the scalar statistic defined in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real nonnegative observations. Zero is a support boundary.
            At least one observation; ties and constant samples are allowed.
        **kwargs : dict, optional
            Reserved for interface compatibility; ignored.

        Returns
        -------
        statistic : float or numpy.float64
            The statistic in the class Notes. No p-value is calculated.

        Raises
        ------
        ValueError
            If the sample violates the stated shape, support or size constraints,
            or the required estimates/quantiles are numerically degenerate.

        Notes
        -----
        See the class Notes for the exact formula, critical tail and
        calibration requirements.
        """

        sorted_rvs = np.sort(_sample(rvs))
        cdf_vals = self._cdf(sorted_rvs)

        n = len(sorted_rvs)

        i = np.arange(1, n + 1)
        d_plus = np.max(i / n - cdf_vals)
        d_minus = np.max(cdf_vals - (i - 1) / n)
        return d_plus + d_minus


class GreenwoodGammaGofStatistic(AbstractGammaGofStatistic):
    """Sum of squared Gamma probability spacings.

    Parameters
    ----------
    parameters : ParameterValues
        Values with alfa, beta fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute one scalar statistic without changing the sample.
    hypothesis()
        Return fixed null parameters; omitted keys mean unknown parameters.

    Notes
    -----
    For ordered observations x_(i), let u_i=F(x_(i); alpha, beta),
    where Gamma has shape alpha, rate beta and known origin zero.
    Parameters are fixed, not fitted. Samples must be finite, real,
    one-dimensional and nonempty, in the closed support [0, infinity).
    Zero is accepted as a boundary value. Ties and constant samples are
    allowed except where stated. Inputs and instance state are unchanged.
    Continuous-null calibration assumes iid observations; rounded data
    require calibration of the observation process.

    G = sum(D_j**2), j=1,...,n+1, where D_j=u_j-u_(j-1),
    u_0=0 and u_(n+1)=1. Both endpoint spacings are included.
    Repeated observations give zero spacings and are permitted.
    The selected right tail detects unusually large squared spacings;
    it does not test against exceptionally regular spacing.

    The fixed-CDF probability transform gives a parameter-free null law
    in exact arithmetic. Reject for large values. The referenced general
    statistic is applied through the Gamma CDF, not a special Gamma table.

    References
    ----------
    .. [1] M. Greenwood (1946), "The Statistical Study of Infectious Diseases",
       J. R. Statist. Soc. 109, 85-103.
       https://doi.org/10.1111/j.2397-2335.1946.tb04649.x

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'alfa': 1, 'beta': 1})
    >>> statistic = GreenwoodGammaGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
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

        :return: short code string "GRW".
        """
        return "GRW"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the scalar statistic defined in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real nonnegative observations. Zero is a support boundary.
            At least one observation; ties and constant samples are allowed.
        **kwargs : dict, optional
            Reserved for interface compatibility; ignored.

        Returns
        -------
        statistic : float or numpy.float64
            The statistic in the class Notes. No p-value is calculated.

        Raises
        ------
        ValueError
            If the sample violates the stated shape, support or size constraints,
            or the required estimates/quantiles are numerically degenerate.

        Notes
        -----
        See the class Notes for the exact formula, critical tail and
        calibration requirements.
        """

        sorted_rvs = np.sort(_sample(rvs))
        cdf_vals = self._cdf(sorted_rvs)
        spacings = np.diff(np.concatenate(([0.0], cdf_vals, [1.0])))
        if np.any(spacings < 0):
            raise ValueError("Spacings must be non-negative; check input data ordering.")
        return float(np.sum(spacings**2))


class MoranGammaGofStatistic(AbstractGammaGofStatistic):
    """Negative log product of Gamma probability spacings.

    Parameters
    ----------
    parameters : ParameterValues
        Values with alfa, beta fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute one scalar statistic without changing the sample.
    hypothesis()
        Return fixed null parameters; omitted keys mean unknown parameters.

    Notes
    -----
    For ordered observations x_(i), let u_i=F(x_(i); alpha, beta),
    where Gamma has shape alpha, rate beta and known origin zero.
    Parameters are fixed, not fitted. Samples must be finite, real,
    one-dimensional and nonempty, in the closed support [0, infinity).
    Zero is accepted as a boundary value. Ties and constant samples are
    allowed except where stated. Inputs and instance state are unchanged.
    Continuous-null calibration assumes iid observations; rounded data
    require calibration of the observation process.

    M = -sum(log(n*D_j)), j=1,...,n+1, with D_j=u_j-u_(j-1),
    u_0=0 and u_(n+1)=1. The historical n scaling is retained: this
    is shifted upward by (n+1)*log((n+1)/n) relative to (n+1) scaling.
    Zero observations or repeated values give positive infinity.
    Log-CDF differences in the lower tail and log-survival differences
    in the upper tail avoid subtracting CDF values rounded to one.
    Unresolved spacings at float64 precision also produce infinity.
    No primary source for this exact normalization was verified in the
    source search; do not reuse tables with another normalization.

    Reject for large values. No p-value is computed by this class.

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'alfa': 1, 'beta': 1})
    >>> statistic = MoranGammaGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
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

        :return: short code string "MOR".
        """
        return "MOR"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the scalar statistic defined in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real nonnegative observations. Zero is a support boundary.
            At least one observation; ties and constant samples are allowed.
        **kwargs : dict, optional
            Reserved for interface compatibility; ignored.

        Returns
        -------
        statistic : float or numpy.float64
            The statistic in the class Notes. No p-value is calculated.

        Raises
        ------
        ValueError
            If the sample violates the stated shape, support or size constraints,
            or the required estimates/quantiles are numerically degenerate.

        Notes
        -----
        Zero spacings (ties or the zero boundary) return positive infinity.
        Unresolved spacings at float64 precision also return infinity.
        """

        sorted_rvs = np.sort(_sample(rvs))
        n = len(sorted_rvs)

        log_cdf, log_sf = self._logs(sorted_rvs)
        lower = log_cdf[1:] <= -np.log(2.0)
        high = np.where(lower, log_cdf[1:], log_sf[:-1])
        low = np.where(lower, log_cdf[:-1], log_sf[1:])
        with np.errstate(divide="ignore", invalid="ignore"):
            interior = high + np.log(-np.expm1(low - high))
        interior = np.where(high == low, -np.inf, interior)
        log_spacings = np.concatenate(([log_cdf[0]], interior, [log_sf[-1]]))
        return float(-np.sum(log_spacings) - (n + 1) * np.log(n))


class MinToshiyukiGammaGofStatistic(AbstractGammaGofStatistic, MinToshiyukiStatistic):
    """Locally defined tail-weighted Gamma EDF discrepancy.

    Parameters
    ----------
    parameters : ParameterValues
        Values with alfa, beta fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute one scalar statistic without changing the sample.
    hypothesis()
        Return fixed null parameters; omitted keys mean unknown parameters.

    Notes
    -----
    For ordered observations x_(i), let u_i=F(x_(i); alpha, beta),
    where Gamma has shape alpha, rate beta and known origin zero.
    Parameters are fixed, not fitted. Samples must be finite, real,
    one-dimensional and nonempty, in the closed support [0, infinity).
    Zero is accepted as a boundary value. Ties and constant samples are
    allowed except where stated. Inputs and instance state are unchanged.
    Continuous-null calibration assumes iid observations; rounded data
    require calibration of the observation process.

    T = sum(max(i/n-u_i, u_i-(i-1)/n)/sqrt(u_i*(1-u_i)))/sqrt(n).
    Weights use log-CDF and log-survival values for tail accuracy.
    Zero observations yield positive infinity; extreme weights may overflow.
    The public name is retained. No primary source establishing this exact
    formula as a Gamma test was found. Liao and Shimokawa (1999) studied
    other families and do not establish this Gamma calibration.
    Use simulation of this formula under the specified Gamma null.

    Reject for large values. No p-value is computed by this class.

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'alfa': 1, 'beta': 1})
    >>> statistic = MinToshiyukiGammaGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "MT".
        """
        return "MT"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the scalar statistic defined in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real nonnegative observations. Zero is a support boundary.
            At least one observation; ties and constant samples are allowed.
        **kwargs : dict, optional
            Reserved for interface compatibility; ignored.

        Returns
        -------
        statistic : float or numpy.float64
            The statistic in the class Notes. No p-value is calculated.

        Raises
        ------
        ValueError
            If the sample violates the stated shape, support or size constraints,
            or the required estimates/quantiles are numerically degenerate.

        Notes
        -----
        Zero observations return positive infinity. Extreme tail weights
        can overflow; no epsilon clipping is applied.
        """

        sorted_rvs = np.sort(_sample(rvs))
        cdf_vals = self._cdf(sorted_rvs)
        n = sorted_rvs.size
        d = np.maximum(np.arange(1, n + 1) / n - cdf_vals, cdf_vals - np.arange(n) / n)
        log_cdf, log_sf = self._logs(sorted_rvs)
        with np.errstate(over="ignore"):
            return float(np.sum(d * np.exp(-0.5 * (log_cdf + log_sf))) / np.sqrt(n))


class AbstractBinnedGammaGofStatistic(AbstractGammaGofStatistic, Chi2Statistic, ABC):
    """
    Base class for Gamma GOF tests built on equiprobable histogram bins.

    Provides common infrastructure for chi-squared type tests that bin data
    using Gamma quantile function to ensure equal theoretical probability per bin.
    """

    lambda_value: float = 1.0

    def __init__(self, parameters: ParameterValues, *, bins: int = 8):
        if (
            isinstance(bins, (bool, np.bool_))
            or not isinstance(bins, (int, np.integer))
            or bins < 2
        ):
            raise ValueError("At least two bins are required for binned Gamma statistics.")
        self.bins = bins
        AbstractGammaGofStatistic.__init__(self, parameters)
        self.lambda_value = getattr(self, "lambda_value", 1.0)

    def _validate_storage_calibration(self):
        raise ValueError(
            "Gamma histogram calibration must include bins and power; "
            "the storage key does not include these settings"
        )

    def _counts_and_expected(self, rvs):
        sample = _sample(rvs)
        n = sample.size

        quantiles = np.linspace(0.0, 1.0, self.bins + 1)
        edges = scipy_stats.gamma.ppf(quantiles, a=self.alpha)
        if not np.all(np.isfinite(edges[1:-1])) or np.any(np.diff(edges) <= 0):
            raise ValueError("Gamma quantile bins collapse at float64 precision")
        edges[0] = -np.inf
        edges[-1] = np.inf
        counts, _ = np.histogram(self._scaled(sample), bins=edges)
        expected = np.full(self.bins, n / self.bins)
        return counts, expected

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the scalar statistic defined in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real nonnegative observations. Zero is a support boundary.
            At least one observation; ties and constant samples are allowed.
        **kwargs : dict, optional
            Reserved for interface compatibility; ignored.

        Returns
        -------
        statistic : float or numpy.float64
            The statistic in the class Notes. No p-value is calculated.

        Raises
        ------
        ValueError
            If the sample violates the stated shape, support or size constraints,
            or the required estimates/quantiles are numerically degenerate.

        Notes
        -----
        Empty bins return positive infinity for power <= -1.
        For power > -1 their contribution is evaluated by continuity.
        """
        counts, expected = self._counts_and_expected(rvs)
        return float(
            Chi2Statistic.do_execute_statistic(self, counts, expected, lambda_=self.lambda_value)
        )


class Chi2PearsonGammaGofStatistic(AbstractBinnedGammaGofStatistic):
    """Pearson statistic on equiprobable Gamma quantile bins.

    Parameters
    ----------
    parameters : ParameterValues
        Values with alfa, beta fixed; omitted parameters are unknown.
    bins : int, optional
        Number of equal-probability bins, at least 2. Default is 8.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute one scalar statistic without changing the sample.
    hypothesis()
        Return fixed null parameters; omitted keys mean unknown parameters.

    Notes
    -----
    For ordered observations x_(i), let u_i=F(x_(i); alpha, beta),
    where Gamma has shape alpha, rate beta and known origin zero.
    Parameters are fixed, not fitted. Samples must be finite, real,
    one-dimensional and nonempty, in the closed support [0, infinity).
    Zero is accepted as a boundary value. Ties and constant samples are
    allowed except where stated. Inputs and instance state are unchanged.
    Continuous-null calibration assumes iid observations; rounded data
    require calibration of the observation process.

    X2 = sum((O_j-E_j)**2/E_j), with E_j=n/bins and O_j the bin counts.
    Bins are [q_j,q_(j+1)); the last includes its right endpoint.
    Here q_j=Gamma.ppf(j/bins, alpha, scale=1/beta).
    This is the power-divergence family at power=1.
    Reject for large values. Under the fixed null, counts are multinomial
    with probabilities 1/bins. Null calibration depends on n, bins and power.
    The chi-square(bins-1) approximation requires sufficiently large expected
    counts. Storage lookup is blocked because its key omits bins and power;
    use MonteCarloLimitDistributionResolver, which uses this instance.
    Unrepresentable quantile boundaries raise ValueError.

    References
    ----------
    .. [1] N. Cressie and T. R. C. Read (1984), "Multinomial Goodness-Of-Fit
       Tests", J. R. Statist. Soc. B 46, 440-464.
       https://doi.org/10.1111/j.2517-6161.1984.tb01318.x

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'alfa': 1, 'beta': 1})
    >>> statistic = Chi2PearsonGammaGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
    >>> bool(np.isfinite(value))
    True
    """

    lambda_value = 1.0

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "CHI2_PEARSON".
        """
        return "CHI2_PEARSON"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the scalar statistic defined in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real nonnegative observations. Zero is a support boundary.
            At least one observation; ties and constant samples are allowed.
        **kwargs : dict, optional
            Reserved for interface compatibility; ignored.

        Returns
        -------
        statistic : float or numpy.float64
            The statistic in the class Notes. No p-value is calculated.

        Raises
        ------
        ValueError
            If the sample violates the stated shape, support or size constraints,
            or the required estimates/quantiles are numerically degenerate.

        Notes
        -----
        See the class Notes for the exact formula, critical tail and
        calibration requirements.
        """

        return super().execute_statistic(rvs)


class LikelihoodRatioGammaGofStatistic(AbstractBinnedGammaGofStatistic):
    """Multinomial likelihood-ratio statistic on Gamma quantile bins.

    Parameters
    ----------
    parameters : ParameterValues
        Values with alfa, beta fixed; omitted parameters are unknown.
    bins : int, optional
        Number of equal-probability bins, at least 2. Default is 8.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute one scalar statistic without changing the sample.
    hypothesis()
        Return fixed null parameters; omitted keys mean unknown parameters.

    Notes
    -----
    For ordered observations x_(i), let u_i=F(x_(i); alpha, beta),
    where Gamma has shape alpha, rate beta and known origin zero.
    Parameters are fixed, not fitted. Samples must be finite, real,
    one-dimensional and nonempty, in the closed support [0, infinity).
    Zero is accepted as a boundary value. Ties and constant samples are
    allowed except where stated. Inputs and instance state are unchanged.
    Continuous-null calibration assumes iid observations; rounded data
    require calibration of the observation process.

    G2 = 2*sum(O_j*log(O_j/E_j)), with E_j=n/bins.
    Zero counts contribute zero by continuity. Bins use Gamma quantiles
    at j/bins, include the left boundary, and cover the entire support.
    This is the power-divergence limit at power=0.
    Reject for large values. Under the fixed null, counts are multinomial
    with probabilities 1/bins. Null calibration depends on n, bins and power.
    The chi-square(bins-1) approximation requires sufficiently large expected
    counts. Storage lookup is blocked because its key omits bins and power;
    use MonteCarloLimitDistributionResolver, which uses this instance.
    Unrepresentable quantile boundaries raise ValueError.

    References
    ----------
    .. [1] N. Cressie and T. R. C. Read (1984), "Multinomial Goodness-Of-Fit
       Tests", J. R. Statist. Soc. B 46, 440-464.
       https://doi.org/10.1111/j.2517-6161.1984.tb01318.x

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'alfa': 1, 'beta': 1})
    >>> statistic = LikelihoodRatioGammaGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
    >>> bool(np.isfinite(value))
    True
    """

    lambda_value = 0.0

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "G_TEST".
        """
        return "G_TEST"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the scalar statistic defined in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real nonnegative observations. Zero is a support boundary.
            At least one observation; ties and constant samples are allowed.
        **kwargs : dict, optional
            Reserved for interface compatibility; ignored.

        Returns
        -------
        statistic : float or numpy.float64
            The statistic in the class Notes. No p-value is calculated.

        Raises
        ------
        ValueError
            If the sample violates the stated shape, support or size constraints,
            or the required estimates/quantiles are numerically degenerate.

        Notes
        -----
        See the class Notes for the exact formula, critical tail and
        calibration requirements.
        """

        return super().execute_statistic(rvs)


class CressieReadGammaGofStatistic(AbstractBinnedGammaGofStatistic):
    """Cressie-Read power divergence on Gamma quantile bins.

    Parameters
    ----------
    parameters : ParameterValues
        Values with alfa, beta fixed; omitted parameters are unknown.
    bins : int, optional
        Number of equal-probability bins, at least 2. Default is 8.
    power : float, optional
        Finite real divergence power. Default is 2/3.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute one scalar statistic without changing the sample.
    hypothesis()
        Return fixed null parameters; omitted keys mean unknown parameters.

    Notes
    -----
    For ordered observations x_(i), let u_i=F(x_(i); alpha, beta),
    where Gamma has shape alpha, rate beta and known origin zero.
    Parameters are fixed, not fitted. Samples must be finite, real,
    one-dimensional and nonempty, in the closed support [0, infinity).
    Zero is accepted as a boundary value. Ties and constant samples are
    allowed except where stated. Inputs and instance state are unchanged.
    Continuous-null calibration assumes iid observations; rounded data
    require calibration of the observation process.

    T = 2*sum(O_j*((O_j/E_j)**power-1))/(power*(power+1)),
    where E_j=n/bins. Quantile bins include their left boundary.
    power=0 uses 2*sum(O_j*log(O_j/E_j)); power=-1 uses
    2*sum(E_j*log(E_j/O_j)). Zero counts contribute zero for
    power>-1 in the first formula and give infinity for power<=-1.
    Large powers may overflow; values very near 0 or -1 can lose precision.
    Reject for large values. Under the fixed null, counts are multinomial
    with probabilities 1/bins. Null calibration depends on n, bins and power.
    The chi-square(bins-1) approximation requires sufficiently large expected
    counts. Storage lookup is blocked because its key omits bins and power;
    use MonteCarloLimitDistributionResolver, which uses this instance.
    Unrepresentable quantile boundaries raise ValueError.

    References
    ----------
    .. [1] N. Cressie and T. R. C. Read (1984), "Multinomial Goodness-Of-Fit
       Tests", J. R. Statist. Soc. B 46, 440-464.
       https://doi.org/10.1111/j.2517-6161.1984.tb01318.x

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'alfa': 1, 'beta': 1})
    >>> statistic = CressieReadGammaGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
    >>> bool(np.isfinite(value))
    True
    """

    def __init__(self, parameters: ParameterValues, *, power: float = 2 / 3, bins: int = 8):
        self.lambda_value = _scalar(power, "power")
        super().__init__(parameters, bins=bins)

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "CRESSIE_READ".
        """
        return "CRESSIE_READ"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the scalar statistic defined in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real nonnegative observations. Zero is a support boundary.
            At least one observation; ties and constant samples are allowed.
        **kwargs : dict, optional
            Reserved for interface compatibility; ignored.

        Returns
        -------
        statistic : float or numpy.float64
            The statistic in the class Notes. No p-value is calculated.

        Raises
        ------
        ValueError
            If the sample violates the stated shape, support or size constraints,
            or the required estimates/quantiles are numerically degenerate.

        Notes
        -----
        Empty bins return positive infinity for power <= -1.
        For power > -1 their contribution is evaluated by continuity.
        """

        return super().execute_statistic(rvs)


class ProbabilityPlotCorrelationGammaGofStatistic(AbstractGammaGofStatistic):
    """Gamma quantile correlation discrepancy with Blom positions.

    Parameters
    ----------
    parameters : ParameterValues
        Values with alfa, beta fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute one scalar statistic without changing the sample.
    hypothesis()
        Return fixed null parameters; omitted keys mean unknown parameters.

    Notes
    -----
    For ordered observations x_(i), let u_i=F(x_(i); alpha, beta),
    where Gamma has shape alpha, rate beta and known origin zero.
    Parameters are fixed, not fitted. Samples must be finite, real,
    one-dimensional and nonempty, in the closed support [0, infinity).
    Zero is accepted as a boundary value. Ties and constant samples are
    allowed except where stated. Inputs and instance state are unchanged.
    Continuous-null calibration assumes iid observations; rounded data
    require calibration of the observation process.

    Return 1-r, where r is the Pearson correlation of ordered x with
    Gamma.ppf((i-0.375)/(n+0.25), alpha, scale=1). The rate cancels
    from correlation; beta remains fixed in the declared null hypothesis.
    The statistic cannot detect positive affine changes of the sample.
    The null law depends on alpha. A nonconstant sample with n>=2 is
    required; for n=2 the statistic is degenerate and has no useful power.
    Independent rescaling of data and quantiles avoids moment overflow.
    No primary source validating this exact Gamma/Blom test was found;
    Filliben normality tables do not calibrate it.

    Reject for large values. No p-value is computed by this class.

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'alfa': 1, 'beta': 1})
    >>> statistic = ProbabilityPlotCorrelationGammaGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
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

        :return: short code string "PPCC".
        """
        return "PPCC"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the scalar statistic defined in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real nonnegative observations. Zero is a support boundary.
            At least two observations; the sample must be nonconstant.
        **kwargs : dict, optional
            Reserved for interface compatibility; ignored.

        Returns
        -------
        statistic : float or numpy.float64
            The statistic in the class Notes. No p-value is calculated.

        Raises
        ------
        ValueError
            If the sample violates the stated shape, support or size constraints,
            or the required estimates/quantiles are numerically degenerate.

        Notes
        -----
        See the class Notes for the exact formula, critical tail and
        calibration requirements.
        """

        sample = np.sort(_sample(rvs, minimum=2))
        n = sample.size

        plotting_positions = (np.arange(1, n + 1) - 0.375) / (n + 0.25)
        expected = scipy_stats.gamma.ppf(plotting_positions, a=self.alpha)

        if np.max(sample) == 0 or not np.all(np.isfinite(expected)) or np.max(expected) == 0:
            raise ValueError("Degenerate data encountered while computing PPCC statistic.")
        sample /= np.max(sample)
        expected /= np.max(expected)
        sample_centered = sample - np.mean(sample)
        expected_centered = expected - np.mean(expected)
        numerator = np.sum(sample_centered * expected_centered)
        denominator = np.sqrt(np.sum(sample_centered**2) * np.sum(expected_centered**2))
        if denominator == 0:
            raise ValueError("Degenerate data encountered while computing PPCC statistic.")

        corr = numerator / denominator
        return float(1.0 - np.clip(corr, -1.0, 1.0))


class AbstractGraphGammaGofStatistic(AbstractGammaGofStatistic, AbstractGraphTestStatistic):
    """
    Base class for Gamma graph-based GOF statistics using EDF transforms.

    Combines Gamma distribution testing with graph-theoretic statistics by
    transforming data through Gamma CDF and analyzing resulting structure.
    """

    @override
    def alternative(self) -> Alternative:
        return TwoSidedAlternative()

    @classmethod
    @override
    def code(cls) -> str:
        """Return the family identifier or the concrete statistic's full identifier."""
        family_code = f"GRAPH_GAMMA_{AbstractGoodnessOfFitStatistic.code()}"
        if "short_code" in cls.__abstractmethods__:
            return family_code
        return f"{cls.short_code()}_{family_code}"

    def _transform_sample(self, rvs):
        sample = _sample(rvs)

        sorted_sample = np.sort(sample)
        uniformized = self._cdf(sorted_sample)
        return uniformized.tolist()

    def _evaluate_graph_statistic(self, transformed_sample, **kwargs):
        """Delegate graph statistic evaluation to the generic adjacency-based logic."""

        return AbstractGraphTestStatistic.execute_statistic(self, transformed_sample, **kwargs)

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the scalar statistic defined in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real nonnegative observations. Zero is a support boundary.
            At least one observation; ties and constant samples are allowed.
        **kwargs : dict, optional
            Reserved for interface compatibility; ignored.

        Returns
        -------
        statistic : float or numpy.float64
            The statistic in the class Notes. No p-value is calculated.

        Raises
        ------
        ValueError
            If the sample violates the stated shape, support or size constraints,
            or the required estimates/quantiles are numerically degenerate.

        Notes
        -----
        Uses the strict distance rule in the concrete class Notes.
        A nonempty constant sample yields an edgeless graph.
        """
        transformed_sample = self._transform_sample(rvs)
        return float(self._evaluate_graph_statistic(transformed_sample, **kwargs))


class GraphEdgesNumberGammaGofStatistic(
    AbstractGraphGammaGofStatistic, GraphEdgesNumberTestStatistic
):
    """Number of undirected edges, sum(degrees)/2.

    Parameters
    ----------
    parameters : ParameterValues
        Values with alfa, beta fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute one scalar statistic without changing the sample.
    hypothesis()
        Return fixed null parameters; omitted keys mean unknown parameters.

    Notes
    -----
    For ordered observations x_(i), let u_i=F(x_(i); alpha, beta),
    where Gamma has shape alpha, rate beta and known origin zero.
    Parameters are fixed, not fitted. Samples must be finite, real,
    one-dimensional and nonempty, in the closed support [0, infinity).
    Zero is accepted as a boundary value. Ties and constant samples are
    allowed except where stated. Inputs and instance state are unchanged.
    Continuous-null calibration assumes iid observations; rounded data
    require calibration of the observation process.

    Number of undirected edges, sum(degrees)/2.
    Vertices are observations transformed by the fixed Gamma CDF.
    An edge joins i and j exactly when abs(u_i-u_j) < h, with
    h=(max(u)-min(u))/10. This uses CDF positions, not their spacings.
    Ties are separate vertices; h=0 yields an edgeless graph.
    Both tails are selected as a local convention; calibrate this exact
    graph construction. No primary source establishing this Gamma-specific
    procedure or its power properties was found. Floating-point saturation
    of the CDF can merge distinct tail values.

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'alfa': 1, 'beta': 1})
    >>> statistic = GraphEdgesNumberGammaGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
    >>> bool(np.isfinite(value))
    True
    """


class GraphMaxDegreeGammaGofStatistic(AbstractGraphGammaGofStatistic, GraphMaxDegreeTestStatistic):
    """Maximum vertex degree.

    Parameters
    ----------
    parameters : ParameterValues
        Values with alfa, beta fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute one scalar statistic without changing the sample.
    hypothesis()
        Return fixed null parameters; omitted keys mean unknown parameters.

    Notes
    -----
    For ordered observations x_(i), let u_i=F(x_(i); alpha, beta),
    where Gamma has shape alpha, rate beta and known origin zero.
    Parameters are fixed, not fitted. Samples must be finite, real,
    one-dimensional and nonempty, in the closed support [0, infinity).
    Zero is accepted as a boundary value. Ties and constant samples are
    allowed except where stated. Inputs and instance state are unchanged.
    Continuous-null calibration assumes iid observations; rounded data
    require calibration of the observation process.

    Maximum vertex degree.
    Vertices are observations transformed by the fixed Gamma CDF.
    An edge joins i and j exactly when abs(u_i-u_j) < h, with
    h=(max(u)-min(u))/10. This uses CDF positions, not their spacings.
    Ties are separate vertices; h=0 yields an edgeless graph.
    Both tails are selected as a local convention; calibrate this exact
    graph construction. No primary source establishing this Gamma-specific
    procedure or its power properties was found. Floating-point saturation
    of the CDF can merge distinct tail values.

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'alfa': 1, 'beta': 1})
    >>> statistic = GraphMaxDegreeGammaGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
    >>> bool(np.isfinite(value))
    True
    """


class GraphAverageDegreeGammaGofStatistic(
    AbstractGraphGammaGofStatistic, GraphAverageDegreeTestStatistic
):
    """Mean vertex degree, 2*edges/n.

    Parameters
    ----------
    parameters : ParameterValues
        Values with alfa, beta fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute one scalar statistic without changing the sample.
    hypothesis()
        Return fixed null parameters; omitted keys mean unknown parameters.

    Notes
    -----
    For ordered observations x_(i), let u_i=F(x_(i); alpha, beta),
    where Gamma has shape alpha, rate beta and known origin zero.
    Parameters are fixed, not fitted. Samples must be finite, real,
    one-dimensional and nonempty, in the closed support [0, infinity).
    Zero is accepted as a boundary value. Ties and constant samples are
    allowed except where stated. Inputs and instance state are unchanged.
    Continuous-null calibration assumes iid observations; rounded data
    require calibration of the observation process.

    Mean vertex degree, 2*edges/n.
    Vertices are observations transformed by the fixed Gamma CDF.
    An edge joins i and j exactly when abs(u_i-u_j) < h, with
    h=(max(u)-min(u))/10. This uses CDF positions, not their spacings.
    Ties are separate vertices; h=0 yields an edgeless graph.
    Both tails are selected as a local convention; calibrate this exact
    graph construction. No primary source establishing this Gamma-specific
    procedure or its power properties was found. Floating-point saturation
    of the CDF can merge distinct tail values.

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'alfa': 1, 'beta': 1})
    >>> statistic = GraphAverageDegreeGammaGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
    >>> bool(np.isfinite(value))
    True
    """


class GraphConnectedComponentsGammaGofStatistic(
    AbstractGraphGammaGofStatistic, GraphConnectedComponentsTestStatistic
):
    """Number of connected components, including isolated vertices.

    Parameters
    ----------
    parameters : ParameterValues
        Values with alfa, beta fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute one scalar statistic without changing the sample.
    hypothesis()
        Return fixed null parameters; omitted keys mean unknown parameters.

    Notes
    -----
    For ordered observations x_(i), let u_i=F(x_(i); alpha, beta),
    where Gamma has shape alpha, rate beta and known origin zero.
    Parameters are fixed, not fitted. Samples must be finite, real,
    one-dimensional and nonempty, in the closed support [0, infinity).
    Zero is accepted as a boundary value. Ties and constant samples are
    allowed except where stated. Inputs and instance state are unchanged.
    Continuous-null calibration assumes iid observations; rounded data
    require calibration of the observation process.

    Number of connected components, including isolated vertices.
    Vertices are observations transformed by the fixed Gamma CDF.
    An edge joins i and j exactly when abs(u_i-u_j) < h, with
    h=(max(u)-min(u))/10. This uses CDF positions, not their spacings.
    Ties are separate vertices; h=0 yields an edgeless graph.
    Both tails are selected as a local convention; calibrate this exact
    graph construction. No primary source establishing this Gamma-specific
    procedure or its power properties was found. Floating-point saturation
    of the CDF can merge distinct tail values.

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'alfa': 1, 'beta': 1})
    >>> statistic = GraphConnectedComponentsGammaGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
    >>> bool(np.isfinite(value))
    True
    """


class GraphCliqueNumberGammaGofStatistic(
    AbstractGraphGammaGofStatistic, GraphCliqueNumberTestStatistic
):
    """Size of the largest clique (1 for a nonempty edgeless graph).

    Parameters
    ----------
    parameters : ParameterValues
        Values with alfa, beta fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute one scalar statistic without changing the sample.
    hypothesis()
        Return fixed null parameters; omitted keys mean unknown parameters.

    Notes
    -----
    For ordered observations x_(i), let u_i=F(x_(i); alpha, beta),
    where Gamma has shape alpha, rate beta and known origin zero.
    Parameters are fixed, not fitted. Samples must be finite, real,
    one-dimensional and nonempty, in the closed support [0, infinity).
    Zero is accepted as a boundary value. Ties and constant samples are
    allowed except where stated. Inputs and instance state are unchanged.
    Continuous-null calibration assumes iid observations; rounded data
    require calibration of the observation process.

    Size of the largest clique (1 for a nonempty edgeless graph).
    Vertices are observations transformed by the fixed Gamma CDF.
    An edge joins i and j exactly when abs(u_i-u_j) < h, with
    h=(max(u)-min(u))/10. This uses CDF positions, not their spacings.
    Ties are separate vertices; h=0 yields an edgeless graph.
    Both tails are selected as a local convention; calibrate this exact
    graph construction. No primary source establishing this Gamma-specific
    procedure or its power properties was found. Floating-point saturation
    of the CDF can merge distinct tail values.

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'alfa': 1, 'beta': 1})
    >>> statistic = GraphCliqueNumberGammaGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
    >>> bool(np.isfinite(value))
    True
    """

    def _evaluate_graph_statistic(self, transformed_sample, **kwargs):
        distance = self._compute_dist(transformed_sample)
        if distance == 0:
            return 1.0
        right = 0
        largest = 1
        for left, value in enumerate(transformed_sample):
            while right < len(transformed_sample) and transformed_sample[right] - value < distance:
                right += 1
            largest = max(largest, right - left)
        return float(largest)


class GraphIndependenceNumberGammaGofStatistic(
    AbstractGraphGammaGofStatistic, GraphIndependenceNumberTestStatistic
):
    """Size of the largest independent set (n for an edgeless graph).

    Parameters
    ----------
    parameters : ParameterValues
        Values with alfa, beta fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute one scalar statistic without changing the sample.
    hypothesis()
        Return fixed null parameters; omitted keys mean unknown parameters.

    Notes
    -----
    For ordered observations x_(i), let u_i=F(x_(i); alpha, beta),
    where Gamma has shape alpha, rate beta and known origin zero.
    Parameters are fixed, not fitted. Samples must be finite, real,
    one-dimensional and nonempty, in the closed support [0, infinity).
    Zero is accepted as a boundary value. Ties and constant samples are
    allowed except where stated. Inputs and instance state are unchanged.
    Continuous-null calibration assumes iid observations; rounded data
    require calibration of the observation process.

    Size of the largest independent set (n for an edgeless graph).
    Vertices are observations transformed by the fixed Gamma CDF.
    An edge joins i and j exactly when abs(u_i-u_j) < h, with
    h=(max(u)-min(u))/10. This uses CDF positions, not their spacings.
    Ties are separate vertices; h=0 yields an edgeless graph.
    Both tails are selected as a local convention; calibrate this exact
    graph construction. No primary source establishing this Gamma-specific
    procedure or its power properties was found. Floating-point saturation
    of the CDF can merge distinct tail values.

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'alfa': 1, 'beta': 1})
    >>> statistic = GraphIndependenceNumberGammaGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
    >>> bool(np.isfinite(value))
    True
    """

    def _evaluate_graph_statistic(self, transformed_sample, **kwargs):
        distance = self._compute_dist(transformed_sample)
        last = transformed_sample[0]
        count = 1
        for value in transformed_sample[1:]:
            if value - last >= distance:
                count += 1
                last = value
        return float(count)
