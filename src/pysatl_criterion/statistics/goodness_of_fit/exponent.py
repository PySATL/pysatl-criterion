"""Exponential goodness-of-fit statistics for observations with known origin zero.

See ``docs/exponent-statistics-audit.md`` for source verification, calibration
migration and limitations. Composite statistics are invariant under positive
rescaling; Monte Carlo calibration may therefore use Exp(1).
"""

import math
from abc import ABC
from itertools import combinations

import numpy as np
import scipy.special as scipy_special
from typing_extensions import override

from pysatl_criterion import DistributionType
from pysatl_criterion.statistics import AbstractGoodnessOfFitStatistic
from pysatl_criterion.statistics.alternative import (
    Alternative,
    AlternativeType,
    RightAlternative,
    TwoSidedAlternative,
)
from pysatl_criterion.statistics.goodness_of_fit.common import (
    CrammerVonMisesStatistic,
    KSStatistic,
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
from pysatl_criterion.statistics.hypothesis import GoodnessOfFitHypothesis


def _sample(rvs, minimum=2, *, rescale=True, positive_total=True):
    """Validate a sample and return an independent, optionally rescaled copy."""
    if np.iscomplexobj(rvs):
        raise ValueError("rvs must be real")
    try:
        x = np.array(rvs, dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("rvs must be a real numeric sample") from exc
    if x.ndim != 1 or x.size < minimum:
        raise ValueError(f"rvs must be one-dimensional with at least {minimum} observations")
    if not np.all(np.isfinite(x)) or np.any(x < 0):
        raise ValueError("rvs must contain finite, nonnegative observations")
    maximum = np.max(x)
    if positive_total and maximum == 0:
        raise ValueError("rvs must have a positive total")
    if rescale and maximum > 0:
        with np.errstate(under="ignore"):
            # A power of two preserves representable ties in strict comparisons.
            scaled = np.ldexp(x, -int(np.frexp(maximum)[1]))
        if np.any((x > 0) & (scaled == 0)):
            raise ValueError("Sample dynamic range exceeds float64 precision")
        x = scaled
    return x


def _scalar(value, name):
    """Validate a real finite scalar setting."""
    if (
        isinstance(value, (bool, np.bool_, str, bytes))
        or np.ndim(value) != 0
        or np.iscomplexobj(value)
    ):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        value = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real scalar")
    return value


def _split(r, n, groups):
    if r is None:
        r = max(1, round(n / (2 * groups)))
    if isinstance(r, bool) or not isinstance(r, (int, np.integer)) or not 1 <= groups * r < n:
        raise ValueError("r must be an integer with 1 <= groups*r < n")
    return r


def _spacings(x):
    return np.arange(len(x), 0, -1) * np.diff(np.r_[0.0, np.sort(x)])


def _ratio(numerator, denominator):
    if denominator == 0:
        if numerator == 0:
            raise ValueError("The ratio is undefined (0/0)")
        return float("inf")
    return float(numerator / denominator)


def _settings(kwargs):
    if kwargs:
        raise TypeError("Pass statistic settings to the constructor, not execute_statistic")


class AbstractExponentialityGofStatistic(AbstractGoodnessOfFitStatistic, ABC):
    """Base for the zero-origin exponential family with unknown positive rate."""

    @override
    def hypothesis(self):
        return GoodnessOfFitHypothesis({})

    def _validate_storage_calibration(self):
        """Reject settings omitted by the current distribution storage key."""
        defaults = {
            "p": 0.99 if self.short_code() == "ATK" else 0.5,
            "b": 0.44,
            "r": None,
            "alternative_type": AlternativeType.TWO_TAILED,
        }
        for name, default in defaults.items():
            if getattr(self, name, default) != default:
                raise ValueError(
                    "Stored exponential calibration does not identify nondefault settings; "
                    "use MonteCarloLimitDistributionResolver with this statistic"
                )

    @staticmethod
    @override
    def code():
        return f"EXPONENTIALITY_{AbstractGoodnessOfFitStatistic.code()}"

    @staticmethod
    @override
    def distribution():
        return DistributionType.EXPONENTIAL


class EppsPulleyExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """EppsPulley statistic for zero-origin exponential observations.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar without changing the sample or fitting state.
    hypothesis()
        Return the fixed parameters of the null.
    alternative()
        Return the critical tail.

    Notes
    -----
    H0 is Exp(lam), lam > 0 unknown, with known origin zero.
    ``hypothesis().parameters()`` is empty. Positive rescaling leaves
    the statistic unchanged, so Exp(1) simulation is valid.
    Write n for sample size, x_(i) for sorted observations, and
    y=x/mean(x). The implemented formula is::

        sqrt(48*n) * (mean(exp(-y)) - 1/2). This is a signed
        Laplace-transform contrast, not an integrated squared distance.

    Reject in the upper tail. This is a directional test; it is not
    claimed consistent against every nonexponential alternative.
    Use independent continuous, uncensored observations. Rounded or tied
    data require calibration of the observation process. No parameters
    are retained from a previous call. Rescaling uses a power of two
    to preserve exact binary ties; unrepresentable positive ratios raise
    ValueError. Strict comparisons near floating-point boundaries can
    still depend on roundoff.

    References
    ----------
    .. [1] Epps, T. W. and Pulley, L. B. (1986). A Test of Exponentiality
       Vs. Monotone-Hazard Alternatives Derived from the Empirical
       Characteristic Function. JRSS B 48, 206-213.
       https://doi.org/10.1111/j.2517-6161.1986.tb01403.x

    Examples
    --------
    >>> statistic = EppsPulleyExponentialityGofStatistic()
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier."""
        return "EP"

    @staticmethod
    @override
    def code():
        """Return the full statistic identifier."""
        short_code = EppsPulleyExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the EppsPulley scalar statistic.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite nonnegative observations; n >= 2. The total must be positive.
            A copy is used. Ties and positive constants are allowed except
            where they make the formula undefined (see Notes on the class).
        **kwargs : dict
            No execution settings are supported; configure the constructor.

        Returns
        -------
        statistic : float
            Value in the convention documented on the class. Infinite
            boundary values are retained where the formula has that limit.

        Raises
        ------
        ValueError
            Invalid sample, insufficient observations, invalid sample-dependent
            setting, undefined ratio, or rescaling underflow of positive data.
        TypeError
            Unsupported execution keyword arguments.
        """
        _settings(kwargs)
        rvs = _sample(rvs, 2)
        n = len(rvs)
        y = rvs / np.mean(rvs)
        ep = np.sqrt(48 * n) * np.sum(np.exp(-y) - 1 / 2) / n
        return float(ep)


class KolmogorovSmirnovExponentialityGofStatistic(AbstractExponentialityGofStatistic, KSStatistic):
    """KolmogorovSmirnov statistic for zero-origin exponential observations.

    Parameters
    ----------
    alternative_type : AlternativeType, optional
        TWO_TAILED (default), RIGHT (D+), or LEFT (D-). All reject
        for large statistic values.
    lam : float, optional
        Fixed, finite positive rate; default 1.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar without changing the sample or fitting state.
    hypothesis()
        Return the fixed parameters of the null.
    alternative()
        Return the critical tail.

    Notes
    -----
    H0 is Exp(lam) with known origin zero and fixed rate lam.
    Write n for sample size, x_(i) for sorted observations, and
    y=x/mean(x). The implemented formula is::

        D+ = max(i/n-F(x_(i))), D- = max(F(x_(i))-(i-1)/n).
        Return max(D+, D-), D+, or D- according to alternative_type.
        Here F(x) = 1-exp(-lam*x); no parameters are estimated.

    Reject in the upper tail. Calibrate the exact statistic returned here.
    Use independent continuous, uncensored observations. Rounded or tied
    data require calibration of the observation process. No parameters
    are retained from a previous call. Rescaling uses a power of two
    to preserve exact binary ties; unrepresentable positive ratios raise
    ValueError. Strict comparisons near floating-point boundaries can
    still depend on roundoff.
    This is the general EDF statistic applied through the known
    exponential CDF (probability integral transform); [1] is a general
    source, not a study of a special fitted exponential version.

    References
    ----------
    .. [1] Durbin, J. (1973). Distribution Theory for Tests Based on the
       Sample Distribution Function, chapter 1. SIAM.
       https://doi.org/10.1137/1.9781611970586.ch1

    Examples
    --------
    >>> statistic = KolmogorovSmirnovExponentialityGofStatistic()
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    def __init__(self, alternative_type=AlternativeType.TWO_TAILED, lam=1):
        self.lam = _scalar(lam, "lam")
        if self.lam <= 0:
            raise ValueError("lam must be positive")
        if not isinstance(alternative_type, AlternativeType):
            raise TypeError("alternative_type must be an AlternativeType")
        KSStatistic.__init__(self, alternative_type)

    @override
    def hypothesis(self):
        return GoodnessOfFitHypothesis({"lam": self.lam})

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier."""
        return "KS"

    @staticmethod
    @override
    def code():
        """Return the full statistic identifier."""
        short_code = KolmogorovSmirnovExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the KolmogorovSmirnov scalar statistic.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite nonnegative observations; n >= 1. All-zero samples are allowed.
            A copy is used. Ties and positive constants are allowed except
            where they make the formula undefined (see Notes on the class).
        **kwargs : dict
            No execution settings are supported; configure the constructor.

        Returns
        -------
        statistic : float
            Value in the convention documented on the class. Infinite
            boundary values are retained where the formula has that limit.

        Raises
        ------
        ValueError
            Invalid sample, insufficient observations, invalid sample-dependent
            setting, undefined ratio, or rescaling underflow of positive data.
        TypeError
            Unsupported execution keyword arguments.
        """
        _settings(kwargs)
        rvs = _sample(rvs, 1, rescale=False, positive_total=False)
        rvs = np.sort(rvs)
        with np.errstate(over="ignore", under="ignore"):
            cdf_vals = -np.expm1(-self.lam * rvs)
        return float(KSStatistic.do_execute_statistic(self, rvs, cdf_vals))


class AhsanullahExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """Ahsanullah statistic for zero-origin exponential observations.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar without changing the sample or fitting state.
    hypothesis()
        Return the fixed parameters of the null.
    alternative()
        Return the critical tail.

    Notes
    -----
    H0 is Exp(lam), lam > 0 unknown, with known origin zero.
    ``hypothesis().parameters()`` is empty. Positive rescaling leaves
    the statistic unchanged, so Exp(1) simulation is valid.
    Write n for sample size, x_(i) for sorted observations, and
    y=x/mean(x). The implemented formula is::

        n**(-3) * sum_{i,j,k} (1{|x_i-x_j| < x_k}
        - 1{2*min(x_i,x_j) < x_k}), including repeated indices.

    Reject in the upper tail. This is a directional test; it is not
    claimed consistent against every nonexponential alternative.
    Use independent continuous, uncensored observations. Rounded or tied
    data require calibration of the observation process. No parameters
    are retained from a previous call. Rescaling uses a power of two
    to preserve exact binary ties; unrepresentable positive ratios raise
    ValueError. Strict comparisons near floating-point boundaries can
    still depend on roundoff.
    A primary source establishing this exact implemented convention
    was not verified in the review; no published tables or asymptotic
    law are endorsed unless derived explicitly above. See the audit.

    Examples
    --------
    >>> statistic = AhsanullahExponentialityGofStatistic()
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier."""
        return "AHS"

    @staticmethod
    @override
    def code():
        """Return the full statistic identifier."""
        short_code = AhsanullahExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the Ahsanullah scalar statistic.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite nonnegative observations; n >= 2. The total must be positive.
            A copy is used. Ties and positive constants are allowed except
            where they make the formula undefined (see Notes on the class).
        **kwargs : dict
            No execution settings are supported; configure the constructor.

        Returns
        -------
        statistic : float
            Value in the convention documented on the class. Infinite
            boundary values are retained where the formula has that limit.

        Raises
        ------
        ValueError
            Invalid sample, insufficient observations, invalid sample-dependent
            setting, undefined ratio, or rescaling underflow of positive data.
        TypeError
            Unsupported execution keyword arguments.
        """
        _settings(kwargs)
        rvs = _sample(rvs, 2)
        n = len(rvs)
        h = 0
        g = 0
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if abs(rvs[i] - rvs[j]) < rvs[k]:
                        h += 1
                    if 2 * min(rvs[i], rvs[j]) < rvs[k]:
                        g += 1
        a = (h - g) / n**3
        return float(a)


class AtkinsonExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """Atkinson statistic for zero-origin exponential observations.

    Parameters
    ----------
    p : float, optional
        Finite power greater than -1, excluding 0 and 1; default 0.99.
        The reference gamma expression must be representable in float64.
        The exclusions avoid divergent moments and degenerate contrasts.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar without changing the sample or fitting state.
    hypothesis()
        Return the fixed parameters of the null.
    alternative()
        Return the critical tail.

    Notes
    -----
    H0 is Exp(lam), lam > 0 unknown, with known origin zero.
    ``hypothesis().parameters()`` is empty. Positive rescaling leaves
    the statistic unchanged, so Exp(1) simulation is valid.
    Write n for sample size, x_(i) for sorted observations, and
    y=x/mean(x). The implemented formula is::

        sqrt(n) * abs(mean(y**p)**(1/p) - Gamma(1+p)**(1/p)).

    This is an unstandardized absolute moment contrast. No universal
    normal or chi-square calibration is claimed. Negative powers at zero
    use the limiting power mean zero. Near p=0 a log-gamma series is used.
    For abs(p) < 1e-100 and strictly positive observations, the geometric
    mean limit is used: the power correction is below float64 precision.
    Near p=1 the contrast can lose relative precision through subtraction.

    Reject in the upper tail. Calibrate the exact statistic returned here.
    Use independent continuous, uncensored observations. Rounded or tied
    data require calibration of the observation process. No parameters
    are retained from a previous call. Rescaling uses a power of two
    to preserve exact binary ties; unrepresentable positive ratios raise
    ValueError. Strict comparisons near floating-point boundaries can
    still depend on roundoff.
    A primary source establishing this exact implemented convention
    was not verified in the review; no published tables or asymptotic
    law are endorsed unless derived explicitly above. See the audit.
    Settings belong in the constructor. The current storage key omits
    these settings; use fresh Monte Carlo calibration for this object,
    not StorageLimitDistributionResolver shared across settings.

    Examples
    --------
    >>> statistic = AtkinsonExponentialityGofStatistic()
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    def __init__(self, p=0.99):
        self.p = _scalar(p, "p")
        if self.p <= -1 or self.p in (0, 1):
            raise ValueError("p must exceed -1 and differ from 0 and 1")
        if not np.isfinite(scipy_special.gammaln(1 + self.p) / self.p):
            raise ValueError("p exceeds the supported numerical range")

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier."""
        return "ATK"

    @staticmethod
    @override
    def code():
        """Return the full statistic identifier."""
        short_code = AtkinsonExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the Atkinson scalar statistic.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite nonnegative observations; n >= 2. The total must be positive.
            A copy is used. Ties and positive constants are allowed except
            where they make the formula undefined (see Notes on the class).
        **kwargs : dict
            No execution settings are supported; configure the constructor.

        Returns
        -------
        statistic : float
            Value in the convention documented on the class. Infinite
            boundary values are retained where the formula has that limit.

        Raises
        ------
        ValueError
            Invalid sample, insufficient observations, invalid sample-dependent
            setting, undefined ratio, or rescaling underflow of positive data.
        TypeError
            Unsupported execution keyword arguments.
        """
        _settings(kwargs)
        rvs = _sample(rvs, 2)
        p = self.p
        if p < 0 and np.any(rvs == 0):
            empirical = 0.0
        else:
            with np.errstate(divide="ignore"):
                logs = np.log(rvs) - np.log(np.mean(rvs))
            # At this scale the power correction is below float64 precision.
            # Avoid multiplying by a subnormal p and then dividing by it.
            if abs(p) < 1e-100 and np.all(np.isfinite(logs)):
                empirical = np.exp(np.mean(logs))
                return float(np.sqrt(len(rvs)) * abs(empirical - np.exp(-np.euler_gamma)))
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                powers = p * logs
            if np.all(np.isfinite(powers)) and np.max(np.abs(powers)) < 0.5:
                # Avoid cancellation of log(n) terms when p is near zero.
                empirical = np.exp(np.log1p(np.mean(np.expm1(powers))) / p)
            else:
                pivot = np.max(logs) if p > 0 else np.min(logs)
                with np.errstate(over="ignore", under="ignore"):
                    log_mean = scipy_special.logsumexp(p * (logs - pivot)) - np.log(len(rvs))
                    empirical = np.exp(pivot + log_mean / p)
        if abs(p) < 1e-100:
            log_reference = -np.euler_gamma
        elif abs(p) < 1e-4:
            # Taylor series of log Gamma(1+p)/p, avoiding rounding 1+p to 1.
            log_reference = -np.euler_gamma + sum(
                (-1) ** k * scipy_special.zeta(k, 1) * p ** (k - 1) / k for k in range(2, 6)
            )
        else:
            log_reference = scipy_special.gammaln(1 + p) / p
        reference = np.exp(log_reference)
        return float(np.sqrt(len(rvs)) * abs(empirical - reference))


class CoxOakesExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """CoxOakes statistic for zero-origin exponential observations.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar without changing the sample or fitting state.
    hypothesis()
        Return the fixed parameters of the null.
    alternative()
        Return the critical tail.

    Notes
    -----
    H0 is Exp(lam), lam > 0 unknown, with known origin zero.
    ``hypothesis().parameters()`` is empty. Positive rescaling leaves
    the statistic unchanged, so Exp(1) simulation is valid.
    Write n for sample size, x_(i) for sorted observations, and
    y=x/mean(x). The implemented formula is::

        n + sum((1-y)*log(y)), the signed Weibull shape score.
        A zero observation gives negative infinity.

    Reject in both tails. Calibrate the exact statistic returned here.
    Use independent continuous, uncensored observations. Rounded or tied
    data require calibration of the observation process. No parameters
    are retained from a previous call. Rescaling uses a power of two
    to preserve exact binary ties; unrepresentable positive ratios raise
    ValueError. Strict comparisons near floating-point boundaries can
    still depend on roundoff.
    A primary source establishing this exact implemented convention
    was not verified in the review; no published tables or asymptotic
    law are endorsed unless derived explicitly above. See the audit.

    Examples
    --------
    >>> statistic = CoxOakesExponentialityGofStatistic()
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @override
    def alternative(self) -> Alternative:
        return TwoSidedAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier."""
        return "CO"

    @staticmethod
    @override
    def code():
        """Return the full statistic identifier."""
        short_code = CoxOakesExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the CoxOakes scalar statistic.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite nonnegative observations; n >= 2. The total must be positive.
            A copy is used. Ties and positive constants are allowed except
            where they make the formula undefined (see Notes on the class).
        **kwargs : dict
            No execution settings are supported; configure the constructor.

        Returns
        -------
        statistic : float
            Value in the convention documented on the class. Infinite
            boundary values are retained where the formula has that limit.

        Raises
        ------
        ValueError
            Invalid sample, insufficient observations, invalid sample-dependent
            setting, undefined ratio, or rescaling underflow of positive data.
        TypeError
            Unsupported execution keyword arguments.
        """
        _settings(kwargs)
        rvs = _sample(rvs, 2)
        if np.any(rvs == 0):
            return float("-inf")
        n = len(rvs)
        y = rvs / np.mean(rvs)
        y = np.log(y) * (1 - y)
        co = np.sum(y) + n
        return float(co)


class CramerVonMisesExponentialityGofStatistic(
    AbstractExponentialityGofStatistic, CrammerVonMisesStatistic
):
    """CramerVonMises statistic for zero-origin exponential observations.

    Parameters
    ----------
    lam : float, optional
        Fixed, finite positive rate; default 1.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar without changing the sample or fitting state.
    hypothesis()
        Return the fixed parameters of the null.
    alternative()
        Return the critical tail.

    Notes
    -----
    H0 is Exp(lam) with known origin zero and fixed rate lam.
    Write n for sample size, x_(i) for sorted observations, and
    y=x/mean(x). The implemented formula is::

        1/(12*n) + sum((F(x_(i))-(2*i-1)/(2*n))**2),
        where F(x) = 1-exp(-lam*x). No parameters are estimated.

    Reject in the upper tail. Calibrate the exact statistic returned here.
    Use independent continuous, uncensored observations. Rounded or tied
    data require calibration of the observation process. No parameters
    are retained from a previous call. Rescaling uses a power of two
    to preserve exact binary ties; unrepresentable positive ratios raise
    ValueError. Strict comparisons near floating-point boundaries can
    still depend on roundoff.
    This is the general EDF statistic applied through the known
    exponential CDF (probability integral transform); [1] is a general
    source, not a study of a special fitted exponential version.

    References
    ----------
    .. [1] Durbin, J. (1973). Distribution Theory for Tests Based on the
       Sample Distribution Function, chapter 1. SIAM.
       https://doi.org/10.1137/1.9781611970586.ch1

    Examples
    --------
    >>> statistic = CramerVonMisesExponentialityGofStatistic()
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    def __init__(self, lam=1):
        self.lam = _scalar(lam, "lam")
        if self.lam <= 0:
            raise ValueError("lam must be positive")

    @override
    def hypothesis(self):
        return GoodnessOfFitHypothesis({"lam": self.lam})

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier."""
        return "CVM"

    @staticmethod
    @override
    def code():
        """Return the full statistic identifier."""
        short_code = CramerVonMisesExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the CramerVonMises scalar statistic.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite nonnegative observations; n >= 1. All-zero samples are allowed.
            A copy is used. Ties and positive constants are allowed except
            where they make the formula undefined (see Notes on the class).
        **kwargs : dict
            No execution settings are supported; configure the constructor.

        Returns
        -------
        statistic : float
            Value in the convention documented on the class. Infinite
            boundary values are retained where the formula has that limit.

        Raises
        ------
        ValueError
            Invalid sample, insufficient observations, invalid sample-dependent
            setting, undefined ratio, or rescaling underflow of positive data.
        TypeError
            Unsupported execution keyword arguments.
        """
        _settings(kwargs)
        rvs = _sample(rvs, 1, rescale=False, positive_total=False)
        rvs = np.sort(rvs)
        with np.errstate(over="ignore", under="ignore"):
            cdf_vals = -np.expm1(-self.lam * rvs)
        return float(CrammerVonMisesStatistic.do_execute_statistic(self, rvs, cdf_vals))


class DeshpandeExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """Deshpande statistic for zero-origin exponential observations.

    Parameters
    ----------
    b : float, optional
        Finite threshold in (0, 1); default 0.44.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar without changing the sample or fitting state.
    hypothesis()
        Return the fixed parameters of the null.
    alternative()
        Return the critical tail.

    Notes
    -----
    H0 is Exp(lam), lam > 0 unknown, with known origin zero.
    ``hypothesis().parameters()`` is empty. Positive rescaling leaves
    the statistic unchanged, so Exp(1) simulation is valid.
    Write n for sample size, x_(i) for sorted observations, and
    y=x/mean(x). The implemented formula is::

        sum_{i != j} 1{x_i > b*x_j} / (n*(n-1)).
        Its exact expectation under the continuous null is 1/(1+b).

    Reject in the upper tail. This is a directional test; it is not
    claimed consistent against every nonexponential alternative.
    Use independent continuous, uncensored observations. Rounded or tied
    data require calibration of the observation process. No parameters
    are retained from a previous call. Rescaling uses a power of two
    to preserve exact binary ties; unrepresentable positive ratios raise
    ValueError. Strict comparisons near floating-point boundaries can
    still depend on roundoff.
    A primary source establishing this exact implemented convention
    was not verified in the review; no published tables or asymptotic
    law are endorsed unless derived explicitly above. See the audit.
    Settings belong in the constructor. The current storage key omits
    these settings; use fresh Monte Carlo calibration for this object,
    not StorageLimitDistributionResolver shared across settings.

    Examples
    --------
    >>> statistic = DeshpandeExponentialityGofStatistic()
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    def __init__(self, b=0.44):
        self.b = _scalar(b, "b")
        if not 0 < self.b < 1:
            raise ValueError("b must lie in (0, 1)")

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier."""
        return "DSP"

    @staticmethod
    @override
    def code():
        """Return the full statistic identifier."""
        short_code = DeshpandeExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the Deshpande scalar statistic.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite nonnegative observations; n >= 2. The total must be positive.
            A copy is used. Ties and positive constants are allowed except
            where they make the formula undefined (see Notes on the class).
        **kwargs : dict
            No execution settings are supported; configure the constructor.

        Returns
        -------
        statistic : float
            Value in the convention documented on the class. Infinite
            boundary values are retained where the formula has that limit.

        Raises
        ------
        ValueError
            Invalid sample, insufficient observations, invalid sample-dependent
            setting, undefined ratio, or rescaling underflow of positive data.
        TypeError
            Unsupported execution keyword arguments.
        """
        _settings(kwargs)
        rvs = _sample(rvs, 2)
        b = self.b
        n = len(rvs)
        des = 0
        for i in range(n):
            for k in range(n):
                if i != k and rvs[i] > b * rvs[k]:
                    des += 1
        des /= n * (n - 1)
        return float(des)


class EpsteinExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """Epstein statistic for zero-origin exponential observations.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar without changing the sample or fitting state.
    hypothesis()
        Return the fixed parameters of the null.
    alternative()
        Return the critical tail.

    Notes
    -----
    H0 is Exp(lam), lam > 0 unknown, with known origin zero.
    ``hypothesis().parameters()`` is empty. Positive rescaling leaves
    the statistic unchanged, so Exp(1) simulation is valid.
    Write n for sample size, x_(i) for sorted observations, and
    y=x/mean(x). The implemented formula is::

        2*n*(log(mean(d))-mean(log(d))) / (1+(n+1)/(6*n)),
        d_i=(n-i+1)*(x_(i)-x_(i-1)), x_(0)=0. The finite-n
        correction is retained; use simulation rather than an asserted
        chi-square law. Zero spacings give positive infinity.

    Reject in the upper tail. Calibrate the exact statistic returned here.
    Use independent continuous, uncensored observations. Rounded or tied
    data require calibration of the observation process. No parameters
    are retained from a previous call. Rescaling uses a power of two
    to preserve exact binary ties; unrepresentable positive ratios raise
    ValueError. Strict comparisons near floating-point boundaries can
    still depend on roundoff.
    A primary source establishing this exact implemented convention
    was not verified in the review; no published tables or asymptotic
    law are endorsed unless derived explicitly above. See the audit.

    Examples
    --------
    >>> statistic = EpsteinExponentialityGofStatistic()
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier."""
        return "EPS"

    @staticmethod
    @override
    def code():
        """Return the full statistic identifier."""
        short_code = EpsteinExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the Epstein scalar statistic.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite nonnegative observations; n >= 2. The total must be positive.
            A copy is used. Ties and positive constants are allowed except
            where they make the formula undefined (see Notes on the class).
        **kwargs : dict
            No execution settings are supported; configure the constructor.

        Returns
        -------
        statistic : float
            Value in the convention documented on the class. Infinite
            boundary values are retained where the formula has that limit.

        Raises
        ------
        ValueError
            Invalid sample, insufficient observations, invalid sample-dependent
            setting, undefined ratio, or rescaling underflow of positive data.
        TypeError
            Unsupported execution keyword arguments.
        """
        _settings(kwargs)
        rvs = _sample(rvs, 2)
        d = _spacings(rvs)
        if np.any(d == 0):
            return float("inf")
        n = len(rvs)
        return float(2 * n * (np.log(np.mean(d)) - np.mean(np.log(d))) / (1 + (n + 1) / (6 * n)))


class FroziniExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """Frozini statistic for zero-origin exponential observations.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar without changing the sample or fitting state.
    hypothesis()
        Return the fixed parameters of the null.
    alternative()
        Return the critical tail.

    Notes
    -----
    H0 is Exp(lam), lam > 0 unknown, with known origin zero.
    ``hypothesis().parameters()`` is empty. Positive rescaling leaves
    the statistic unchanged, so Exp(1) simulation is valid.
    Write n for sample size, x_(i) for sorted observations, and
    y=x/mean(x). The implemented formula is::

        sum(abs(1-exp(-y_(i))-(i-1/2)/n))/sqrt(n).
        The exponential mean is re-estimated for every sample.

    Reject in the upper tail. Calibrate the exact statistic returned here.
    Use independent continuous, uncensored observations. Rounded or tied
    data require calibration of the observation process. No parameters
    are retained from a previous call. Rescaling uses a power of two
    to preserve exact binary ties; unrepresentable positive ratios raise
    ValueError. Strict comparisons near floating-point boundaries can
    still depend on roundoff.
    A primary source establishing this exact implemented convention
    was not verified in the review; no published tables or asymptotic
    law are endorsed unless derived explicitly above. See the audit.

    Examples
    --------
    >>> statistic = FroziniExponentialityGofStatistic()
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier."""
        return "FZ"

    @staticmethod
    @override
    def code():
        """Return the full statistic identifier."""
        short_code = FroziniExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the Frozini scalar statistic.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite nonnegative observations; n >= 2. The total must be positive.
            A copy is used. Ties and positive constants are allowed except
            where they make the formula undefined (see Notes on the class).
        **kwargs : dict
            No execution settings are supported; configure the constructor.

        Returns
        -------
        statistic : float
            Value in the convention documented on the class. Infinite
            boundary values are retained where the formula has that limit.

        Raises
        ------
        ValueError
            Invalid sample, insufficient observations, invalid sample-dependent
            setting, undefined ratio, or rescaling underflow of positive data.
        TypeError
            Unsupported execution keyword arguments.
        """
        _settings(kwargs)
        rvs = _sample(rvs, 2)
        n = len(rvs)
        rvs.sort()
        rvs = np.array(rvs)
        y = np.mean(rvs)
        froz = (
            1 / np.sqrt(n) * np.sum(np.abs(-np.expm1(-rvs / y) - (np.arange(1, n + 1) - 0.5) / n))
        )
        return float(froz)


class GiniExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """Gini statistic for zero-origin exponential observations.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar without changing the sample or fitting state.
    hypothesis()
        Return the fixed parameters of the null.
    alternative()
        Return the critical tail.

    Notes
    -----
    H0 is Exp(lam), lam > 0 unknown, with known origin zero.
    ``hypothesis().parameters()`` is empty. Positive rescaling leaves
    the statistic unchanged, so Exp(1) simulation is valid.
    Write n for sample size, x_(i) for sorted observations, and
    y=x/mean(x). The implemented formula is::

        sum_{i<j} abs(x_i-x_j) / ((n-1)*sum(x)).
        This uses the n-1 convention; the exact null expectation is 1/2.

    Reject in both tails. Calibrate the exact statistic returned here.
    Use independent continuous, uncensored observations. Rounded or tied
    data require calibration of the observation process. No parameters
    are retained from a previous call. Rescaling uses a power of two
    to preserve exact binary ties; unrepresentable positive ratios raise
    ValueError. Strict comparisons near floating-point boundaries can
    still depend on roundoff.
    A primary source establishing this exact implemented convention
    was not verified in the review; no published tables or asymptotic
    law are endorsed unless derived explicitly above. See the audit.

    Examples
    --------
    >>> statistic = GiniExponentialityGofStatistic()
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @override
    def alternative(self) -> Alternative:
        return TwoSidedAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier."""
        return "GINI"

    @staticmethod
    @override
    def code():
        """Return the full statistic identifier."""
        short_code = GiniExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the Gini scalar statistic.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite nonnegative observations; n >= 2. The total must be positive.
            A copy is used. Ties and positive constants are allowed except
            where they make the formula undefined (see Notes on the class).
        **kwargs : dict
            No execution settings are supported; configure the constructor.

        Returns
        -------
        statistic : float
            Value in the convention documented on the class. Infinite
            boundary values are retained where the formula has that limit.

        Raises
        ------
        ValueError
            Invalid sample, insufficient observations, invalid sample-dependent
            setting, undefined ratio, or rescaling underflow of positive data.
        TypeError
            Unsupported execution keyword arguments.
        """
        _settings(kwargs)
        rvs = _sample(rvs, 2)
        n = len(rvs)
        a = np.arange(1, n)
        b = np.arange(n - 1, 0, -1)
        a = a * b
        x = np.sort(rvs)
        k = x[1:] - x[:-1]
        gini = np.sum(k * a) / ((n - 1) * np.sum(x))
        return float(gini)


class GnedenkoExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """Gnedenko statistic for zero-origin exponential observations.

    Parameters
    ----------
    r : int or None, optional
        Split count with 1 <= r < n. Default None uses max(1, round(n/2)).

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar without changing the sample or fitting state.
    hypothesis()
        Return the fixed parameters of the null.
    alternative()
        Return the critical tail.

    Notes
    -----
    H0 is Exp(lam), lam > 0 unknown, with known origin zero.
    ``hypothesis().parameters()`` is empty. Positive rescaling leaves
    the statistic unchanged, so Exp(1) simulation is valid.
    Write n for sample size, x_(i) for sorted observations, and
    y=x/mean(x). The implemented formula is::

        mean(d[:r])/mean(d[r:]), where
        d_i=(n-i+1)*(x_(i)-x_(i-1)), x_(0)=0. Under H0 the
        spacings are independent exponentials, giving F(2*r, 2*(n-r)).
        A zero denominator with positive numerator gives positive infinity.

    Reject in the upper tail. This is a directional test; it is not
    claimed consistent against every nonexponential alternative.
    Use independent continuous, uncensored observations. Rounded or tied
    data require calibration of the observation process. No parameters
    are retained from a previous call. Rescaling uses a power of two
    to preserve exact binary ties; unrepresentable positive ratios raise
    ValueError. Strict comparisons near floating-point boundaries can
    still depend on roundoff.
    A primary source establishing this exact implemented convention
    was not verified in the review; no published tables or asymptotic
    law are endorsed unless derived explicitly above. See the audit.
    Settings belong in the constructor. The current storage key omits
    these settings; use fresh Monte Carlo calibration for this object,
    not StorageLimitDistributionResolver shared across settings.

    Examples
    --------
    >>> statistic = GnedenkoExponentialityGofStatistic()
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    def __init__(self, r=None):
        if r is not None and (isinstance(r, bool) or not isinstance(r, (int, np.integer)) or r < 1):
            raise ValueError("r must be a positive integer or None")
        self.r = r

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier."""
        return "GD"

    @staticmethod
    @override
    def code():
        """Return the full statistic identifier."""
        short_code = GnedenkoExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the Gnedenko scalar statistic.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite nonnegative observations; n >= 2. The total must be positive.
            A copy is used. Ties and positive constants are allowed except
            where they make the formula undefined (see Notes on the class).
        **kwargs : dict
            No execution settings are supported; configure the constructor.

        Returns
        -------
        statistic : float
            Value in the convention documented on the class. Infinite
            boundary values are retained where the formula has that limit.

        Raises
        ------
        ValueError
            Invalid sample, insufficient observations, invalid sample-dependent
            setting, undefined ratio, or rescaling underflow of positive data.
        TypeError
            Unsupported execution keyword arguments.
        """
        _settings(kwargs)
        rvs = _sample(rvs, 2)
        r = _split(self.r, len(rvs), 1)
        d = _spacings(rvs)
        return _ratio(np.mean(d[:r]), np.mean(d[r:]))


class HarrisExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """Harris statistic for zero-origin exponential observations.

    Parameters
    ----------
    r : int or None, optional
        Tail count with 1 <= 2*r < n. Default None uses max(1, round(n/4)).

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar without changing the sample or fitting state.
    hypothesis()
        Return the fixed parameters of the null.
    alternative()
        Return the critical tail.

    Notes
    -----
    H0 is Exp(lam), lam > 0 unknown, with known origin zero.
    ``hypothesis().parameters()`` is empty. Positive rescaling leaves
    the statistic unchanged, so Exp(1) simulation is valid.
    Write n for sample size, x_(i) for sorted observations, and
    y=x/mean(x). The implemented formula is::

        mean(concatenate((d[:r], d[-r:])))/mean(d[r:-r]),
        d_i=(n-i+1)*(x_(i)-x_(i-1)), x_(0)=0. Independent
        exponential spacings give the exact F(4*r, 2*(n-2*r)) law.
        A zero denominator with positive numerator gives positive infinity.

    Reject in the upper tail. This is a directional test; it is not
    claimed consistent against every nonexponential alternative.
    Use independent continuous, uncensored observations. Rounded or tied
    data require calibration of the observation process. No parameters
    are retained from a previous call. Rescaling uses a power of two
    to preserve exact binary ties; unrepresentable positive ratios raise
    ValueError. Strict comparisons near floating-point boundaries can
    still depend on roundoff.
    A primary source establishing this exact implemented convention
    was not verified in the review; no published tables or asymptotic
    law are endorsed unless derived explicitly above. See the audit.
    Settings belong in the constructor. The current storage key omits
    these settings; use fresh Monte Carlo calibration for this object,
    not StorageLimitDistributionResolver shared across settings.

    Examples
    --------
    >>> statistic = HarrisExponentialityGofStatistic()
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    def __init__(self, r=None):
        if r is not None and (isinstance(r, bool) or not isinstance(r, (int, np.integer)) or r < 1):
            raise ValueError("r must be a positive integer or None")
        self.r = r

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier."""
        return "HM"

    @staticmethod
    @override
    def code():
        """Return the full statistic identifier."""
        short_code = HarrisExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the Harris scalar statistic.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite nonnegative observations; n >= 3. The total must be positive.
            A copy is used. Ties and positive constants are allowed except
            where they make the formula undefined (see Notes on the class).
        **kwargs : dict
            No execution settings are supported; configure the constructor.

        Returns
        -------
        statistic : float
            Value in the convention documented on the class. Infinite
            boundary values are retained where the formula has that limit.

        Raises
        ------
        ValueError
            Invalid sample, insufficient observations, invalid sample-dependent
            setting, undefined ratio, or rescaling underflow of positive data.
        TypeError
            Unsupported execution keyword arguments.
        """
        _settings(kwargs)
        rvs = _sample(rvs, 3)
        r = _split(self.r, len(rvs), 2)
        d = _spacings(rvs)
        return _ratio((np.sum(d[:r]) + np.sum(d[-r:])) / (2 * r), np.mean(d[r:-r]))


class HegazyGreen1ExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """HegazyGreen1 statistic for zero-origin exponential observations.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar without changing the sample or fitting state.
    hypothesis()
        Return the fixed parameters of the null.
    alternative()
        Return the critical tail.

    Notes
    -----
    H0 is Exp(lam), lam > 0 unknown, with known origin zero.
    ``hypothesis().parameters()`` is empty. Positive rescaling leaves
    the statistic unchanged, so Exp(1) simulation is valid.
    Write n for sample size, x_(i) for sorted observations, and
    y=x/mean(x). The implemented formula is::

        mean(abs(y_(i) + log(1-i/(n+1)))).
        This is a fitted exponential quantile discrepancy.

    Reject in the upper tail. Calibrate the exact statistic returned here.
    Use independent continuous, uncensored observations. Rounded or tied
    data require calibration of the observation process. No parameters
    are retained from a previous call. Rescaling uses a power of two
    to preserve exact binary ties; unrepresentable positive ratios raise
    ValueError. Strict comparisons near floating-point boundaries can
    still depend on roundoff.
    A primary source establishing this exact implemented convention
    was not verified in the review; no published tables or asymptotic
    law are endorsed unless derived explicitly above. See the audit.
    The original Hegazy-Green paper concerns uniform/normal models;
    it does not establish this fitted exponential adaptation.

    Examples
    --------
    >>> statistic = HegazyGreen1ExponentialityGofStatistic()
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier."""
        return "HG1"

    @staticmethod
    @override
    def code():
        """Return the full statistic identifier."""
        short_code = HegazyGreen1ExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the HegazyGreen1 scalar statistic.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite nonnegative observations; n >= 2. The total must be positive.
            A copy is used. Ties and positive constants are allowed except
            where they make the formula undefined (see Notes on the class).
        **kwargs : dict
            No execution settings are supported; configure the constructor.

        Returns
        -------
        statistic : float
            Value in the convention documented on the class. Infinite
            boundary values are retained where the formula has that limit.

        Raises
        ------
        ValueError
            Invalid sample, insufficient observations, invalid sample-dependent
            setting, undefined ratio, or rescaling underflow of positive data.
        TypeError
            Unsupported execution keyword arguments.
        """
        _settings(kwargs)
        rvs = _sample(rvs, 2)
        rvs = rvs / np.mean(rvs)
        n = len(rvs)
        x = np.sort(rvs)
        b = -np.log(1 - np.arange(1, n + 1) / (n + 1))
        hg = n ** (-1) * np.sum(np.abs(x - b))
        return float(hg)


class HollanderProshanExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """HollanderProshan statistic for zero-origin exponential observations.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar without changing the sample or fitting state.
    hypothesis()
        Return the fixed parameters of the null.
    alternative()
        Return the critical tail.

    Notes
    -----
    H0 is Exp(lam), lam > 0 unknown, with known origin zero.
    ``hypothesis().parameters()`` is empty. Positive rescaling leaves
    the statistic unchanged, so Exp(1) simulation is valid.
    Write n for sample size, x_(i) for sorted observations, and
    y=x/mean(x). The implemented formula is::

        2*sum_{i,j<k; i!=j,k} 1{x_i > x_j+x_k}
        / (n*(n-1)*(n-2)). This is the raw U-statistic, with null
        expectation 1/4, not a standardized normal statistic.

    Reject in the upper tail. This is a directional test; it is not
    claimed consistent against every nonexponential alternative.
    Use independent continuous, uncensored observations. Rounded or tied
    data require calibration of the observation process. No parameters
    are retained from a previous call. Rescaling uses a power of two
    to preserve exact binary ties; unrepresentable positive ratios raise
    ValueError. Strict comparisons near floating-point boundaries can
    still depend on roundoff.
    A primary source establishing this exact implemented convention
    was not verified in the review; no published tables or asymptotic
    law are endorsed unless derived explicitly above. See the audit.

    Examples
    --------
    >>> statistic = HollanderProshanExponentialityGofStatistic()
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier."""
        return "HP"

    @staticmethod
    @override
    def code():
        """Return the full statistic identifier."""
        short_code = HollanderProshanExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the HollanderProshan scalar statistic.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite nonnegative observations; n >= 3. The total must be positive.
            A copy is used. Ties and positive constants are allowed except
            where they make the formula undefined (see Notes on the class).
        **kwargs : dict
            No execution settings are supported; configure the constructor.

        Returns
        -------
        statistic : float
            Value in the convention documented on the class. Infinite
            boundary values are retained where the formula has that limit.

        Raises
        ------
        ValueError
            Invalid sample, insufficient observations, invalid sample-dependent
            setting, undefined ratio, or rescaling underflow of positive data.
        TypeError
            Unsupported execution keyword arguments.
        """
        _settings(kwargs)
        rvs = _sample(rvs, 3)
        n = len(rvs)
        t = 0
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if i != j and i != k and (j < k) and (rvs[i] > rvs[j] + rvs[k]):
                        t += 1
        hp = 2 / (n * (n - 1) * (n - 2)) * t
        return float(hp)


class KimberMichaelExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """KimberMichael statistic for zero-origin exponential observations.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar without changing the sample or fitting state.
    hypothesis()
        Return the fixed parameters of the null.
    alternative()
        Return the critical tail.

    Notes
    -----
    H0 is Exp(lam), lam > 0 unknown, with known origin zero.
    ``hypothesis().parameters()`` is empty. Positive rescaling leaves
    the statistic unchanged, so Exp(1) simulation is valid.
    Write n for sample size, x_(i) for sorted observations, and
    y=x/mean(x). The implemented formula is::

        max(abs(A((i-1/2)/n)-A(1-exp(-y_(i))))),
        where A(u)=2*arcsin(sqrt(u))/pi. The mean is fitted per sample.

    Reject in the upper tail. Calibrate the exact statistic returned here.
    Use independent continuous, uncensored observations. Rounded or tied
    data require calibration of the observation process. No parameters
    are retained from a previous call. Rescaling uses a power of two
    to preserve exact binary ties; unrepresentable positive ratios raise
    ValueError. Strict comparisons near floating-point boundaries can
    still depend on roundoff.
    A primary source establishing this exact implemented convention
    was not verified in the review; no published tables or asymptotic
    law are endorsed unless derived explicitly above. See the audit.

    Examples
    --------
    >>> statistic = KimberMichaelExponentialityGofStatistic()
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier."""
        return "KM"

    @staticmethod
    @override
    def code():
        """Return the full statistic identifier."""
        short_code = KimberMichaelExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the KimberMichael scalar statistic.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite nonnegative observations; n >= 2. The total must be positive.
            A copy is used. Ties and positive constants are allowed except
            where they make the formula undefined (see Notes on the class).
        **kwargs : dict
            No execution settings are supported; configure the constructor.

        Returns
        -------
        statistic : float
            Value in the convention documented on the class. Infinite
            boundary values are retained where the formula has that limit.

        Raises
        ------
        ValueError
            Invalid sample, insufficient observations, invalid sample-dependent
            setting, undefined ratio, or rescaling underflow of positive data.
        TypeError
            Unsupported execution keyword arguments.
        """
        _settings(kwargs)
        rvs = _sample(rvs, 2)
        n = len(rvs)
        rvs.sort()
        y = np.mean(rvs)
        s = 2 / np.pi * np.arcsin(np.sqrt(-np.expm1(-rvs / y)))
        r = 2 / np.pi * np.arcsin(np.sqrt((np.arange(1, n + 1) - 0.5) / n))
        km = max(abs(r - s))
        return float(km)


class KocharExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """Kochar statistic for zero-origin exponential observations.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar without changing the sample or fitting state.
    hypothesis()
        Return the fixed parameters of the null.
    alternative()
        Return the critical tail.

    Notes
    -----
    H0 is Exp(lam), lam > 0 unknown, with known origin zero.
    ``hypothesis().parameters()`` is empty. Positive rescaling leaves
    the statistic unchanged, so Exp(1) simulation is valid.
    Write n for sample size, x_(i) for sorted observations, and
    y=x/mean(x). The implemented formula is::

        sqrt(108*n/17) * sum(J(i/(n+1))*x_(i))/sum(x),
        J(u)=2*(1-u)*(1-log(1-u))-1. No exact normal law is claimed.

    Reject in the upper tail. This is a directional test; it is not
    claimed consistent against every nonexponential alternative.
    Use independent continuous, uncensored observations. Rounded or tied
    data require calibration of the observation process. No parameters
    are retained from a previous call. Rescaling uses a power of two
    to preserve exact binary ties; unrepresentable positive ratios raise
    ValueError. Strict comparisons near floating-point boundaries can
    still depend on roundoff.
    A primary source establishing this exact implemented convention
    was not verified in the review; no published tables or asymptotic
    law are endorsed unless derived explicitly above. See the audit.

    Examples
    --------
    >>> statistic = KocharExponentialityGofStatistic()
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier."""
        return "KC"

    @staticmethod
    @override
    def code():
        """Return the full statistic identifier."""
        short_code = KocharExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the Kochar scalar statistic.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite nonnegative observations; n >= 2. The total must be positive.
            A copy is used. Ties and positive constants are allowed except
            where they make the formula undefined (see Notes on the class).
        **kwargs : dict
            No execution settings are supported; configure the constructor.

        Returns
        -------
        statistic : float
            Value in the convention documented on the class. Infinite
            boundary values are retained where the formula has that limit.

        Raises
        ------
        ValueError
            Invalid sample, insufficient observations, invalid sample-dependent
            setting, undefined ratio, or rescaling underflow of positive data.
        TypeError
            Unsupported execution keyword arguments.
        """
        _settings(kwargs)
        rvs = _sample(rvs, 2)
        n = len(rvs)
        rvs.sort()
        u = np.array([(i + 1) / (n + 1) for i in range(n)])
        j = 2 * (1 - u) * (1 - np.log(1 - u)) - 1
        kc = np.sqrt(108 * n / 17) * np.sum(j * rvs) / np.sum(rvs)
        return float(kc)


class LorenzExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """Lorenz statistic for zero-origin exponential observations.

    Parameters
    ----------
    p : float, optional
        Finite fraction in (0, 1); default 0.5. The sample must satisfy
        1 <= floor(n*p) < n.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar without changing the sample or fitting state.
    hypothesis()
        Return the fixed parameters of the null.
    alternative()
        Return the critical tail.

    Notes
    -----
    H0 is Exp(lam), lam > 0 unknown, with known origin zero.
    ``hypothesis().parameters()`` is empty. Positive rescaling leaves
    the statistic unchanged, so Exp(1) simulation is valid.
    Write n for sample size, x_(i) for sorted observations, and
    y=x/mean(x). The implemented formula is::

        sum(x_(i), i=1,...,floor(n*p))/sum(x).
        This uses the step convention without fractional interpolation.
        The limiting null center is p+(1-p)*log(1-p).

    Reject in the upper tail. This is a directional test; it is not
    claimed consistent against every nonexponential alternative.
    Use independent continuous, uncensored observations. Rounded or tied
    data require calibration of the observation process. No parameters
    are retained from a previous call. Rescaling uses a power of two
    to preserve exact binary ties; unrepresentable positive ratios raise
    ValueError. Strict comparisons near floating-point boundaries can
    still depend on roundoff.
    A primary source establishing this exact implemented convention
    was not verified in the review; no published tables or asymptotic
    law are endorsed unless derived explicitly above. See the audit.
    Settings belong in the constructor. The current storage key omits
    these settings; use fresh Monte Carlo calibration for this object,
    not StorageLimitDistributionResolver shared across settings.

    Examples
    --------
    >>> statistic = LorenzExponentialityGofStatistic()
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    def __init__(self, p=0.5):
        self.p = _scalar(p, "p")
        if not 0 < self.p < 1:
            raise ValueError("p must lie in (0, 1)")

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier."""
        return "LZ"

    @staticmethod
    @override
    def code():
        """Return the full statistic identifier."""
        short_code = LorenzExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the Lorenz scalar statistic.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite nonnegative observations; n >= 2. The total must be positive.
            A copy is used. Ties and positive constants are allowed except
            where they make the formula undefined (see Notes on the class).
        **kwargs : dict
            No execution settings are supported; configure the constructor.

        Returns
        -------
        statistic : float
            Value in the convention documented on the class. Infinite
            boundary values are retained where the formula has that limit.

        Raises
        ------
        ValueError
            Invalid sample, insufficient observations, invalid sample-dependent
            setting, undefined ratio, or rescaling underflow of positive data.
        TypeError
            Unsupported execution keyword arguments.
        """
        _settings(kwargs)
        rvs = _sample(rvs, 2)
        k = int(len(rvs) * self.p)
        if not 1 <= k < len(rvs):
            raise ValueError("p must select at least one and fewer than n observations")
        return float(np.sum(np.sort(rvs)[:k]) / np.sum(rvs))


class MoranExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """Moran statistic for zero-origin exponential observations.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar without changing the sample or fitting state.
    hypothesis()
        Return the fixed parameters of the null.
    alternative()
        Return the critical tail.

    Notes
    -----
    H0 is Exp(lam), lam > 0 unknown, with known origin zero.
    ``hypothesis().parameters()`` is empty. Positive rescaling leaves
    the statistic unchanged, so Exp(1) simulation is valid.
    Write n for sample size, x_(i) for sorted observations, and
    y=x/mean(x). The implemented formula is::

        -digamma(1) + mean(log(y)), a signed centered log-mean
        statistic. It is not standardized to unit variance. Zero observations
        give negative infinity.

    Reject in the upper tail. This is a directional test; it is not
    claimed consistent against every nonexponential alternative.
    Use independent continuous, uncensored observations. Rounded or tied
    data require calibration of the observation process. No parameters
    are retained from a previous call. Rescaling uses a power of two
    to preserve exact binary ties; unrepresentable positive ratios raise
    ValueError. Strict comparisons near floating-point boundaries can
    still depend on roundoff.
    A primary source establishing this exact implemented convention
    was not verified in the review; no published tables or asymptotic
    law are endorsed unless derived explicitly above. See the audit.

    Examples
    --------
    >>> statistic = MoranExponentialityGofStatistic()
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier."""
        return "MN"

    @staticmethod
    @override
    def code():
        """Return the full statistic identifier."""
        short_code = MoranExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the Moran scalar statistic.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite nonnegative observations; n >= 2. The total must be positive.
            A copy is used. Ties and positive constants are allowed except
            where they make the formula undefined (see Notes on the class).
        **kwargs : dict
            No execution settings are supported; configure the constructor.

        Returns
        -------
        statistic : float
            Value in the convention documented on the class. Infinite
            boundary values are retained where the formula has that limit.

        Raises
        ------
        ValueError
            Invalid sample, insufficient observations, invalid sample-dependent
            setting, undefined ratio, or rescaling underflow of positive data.
        TypeError
            Unsupported execution keyword arguments.
        """
        _settings(kwargs)
        rvs = _sample(rvs, 2)
        if np.any(rvs == 0):
            return float("-inf")
        y = np.mean(rvs)
        mn = -scipy_special.digamma(1) + np.mean(np.log(rvs / y))
        return float(mn)


class PietraExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """Pietra statistic for zero-origin exponential observations.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar without changing the sample or fitting state.
    hypothesis()
        Return the fixed parameters of the null.
    alternative()
        Return the critical tail.

    Notes
    -----
    H0 is Exp(lam), lam > 0 unknown, with known origin zero.
    ``hypothesis().parameters()`` is empty. Positive rescaling leaves
    the statistic unchanged, so Exp(1) simulation is valid.
    Write n for sample size, x_(i) for sorted observations, and
    y=x/mean(x). The implemented formula is::

        mean(abs(y-1))/2. The limiting null center is 1/e.

    Reject in both tails. Calibrate the exact statistic returned here.
    Use independent continuous, uncensored observations. Rounded or tied
    data require calibration of the observation process. No parameters
    are retained from a previous call. Rescaling uses a power of two
    to preserve exact binary ties; unrepresentable positive ratios raise
    ValueError. Strict comparisons near floating-point boundaries can
    still depend on roundoff.
    A primary source establishing this exact implemented convention
    was not verified in the review; no published tables or asymptotic
    law are endorsed unless derived explicitly above. See the audit.

    Examples
    --------
    >>> statistic = PietraExponentialityGofStatistic()
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @override
    def alternative(self) -> Alternative:
        return TwoSidedAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier."""
        return "PT"

    @staticmethod
    @override
    def code():
        """Return the full statistic identifier."""
        short_code = PietraExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the Pietra scalar statistic.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite nonnegative observations; n >= 2. The total must be positive.
            A copy is used. Ties and positive constants are allowed except
            where they make the formula undefined (see Notes on the class).
        **kwargs : dict
            No execution settings are supported; configure the constructor.

        Returns
        -------
        statistic : float
            Value in the convention documented on the class. Infinite
            boundary values are retained where the formula has that limit.

        Raises
        ------
        ValueError
            Invalid sample, insufficient observations, invalid sample-dependent
            setting, undefined ratio, or rescaling underflow of positive data.
        TypeError
            Unsupported execution keyword arguments.
        """
        _settings(kwargs)
        rvs = _sample(rvs, 2)
        n = len(rvs)
        xm = np.mean(rvs)
        pt = np.sum(np.abs(rvs - xm)) / (2 * n * xm)
        return float(pt)


class ShapiroWilkExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """ShapiroWilk statistic for zero-origin exponential observations.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar without changing the sample or fitting state.
    hypothesis()
        Return the fixed parameters of the null.
    alternative()
        Return the critical tail.

    Notes
    -----
    H0 is Exp(lam), lam > 0 unknown, with known origin zero.
    ``hypothesis().parameters()`` is empty. Positive rescaling leaves
    the statistic unchanged, so Exp(1) simulation is valid.
    Write n for sample size, x_(i) for sorted observations, and
    y=x/mean(x). The implemented formula is::

        n*(mean(x)-min(x))**2 / ((n-1)*sum((x-mean(x))**2)).
        This W is also translation invariant, so it cannot detect a positive
        shift of an exponential law. Both tails are used. Constant samples
        are undefined; n >= 3 avoids the identically-one n=2 case.

    Computation uses z=(x-min(x))/(max(x)-min(x)) before evaluating W.
    Affine invariance preserves the formula while avoiding cancellation
    in mean(x)-min(x) for nearly constant observations.

    Reject in both tails. Calibrate the exact statistic returned here.
    Use independent continuous, uncensored observations. Rounded or tied
    data require calibration of the observation process. No parameters
    are retained from a previous call. Rescaling uses a power of two
    to preserve exact binary ties; unrepresentable positive ratios raise
    ValueError. Strict comparisons near floating-point boundaries can
    still depend on roundoff.

    References
    ----------
    .. [1] Shapiro, S. S. and Wilk, M. B. (1972). An Analysis of Variance
       Test for the Exponential Distribution (Complete Samples).
       Technometrics 14, 355-370.
       https://doi.org/10.1080/00401706.1972.10488921
    .. [2] Spinelli, J. J. and Stephens, M. A. (1987). Tests for
       Exponentiality When Origin and Scale Parameters Are Unknown.
       Technometrics 29, 471-476, Section 3, pp. 474-475.
       https://www.stat.cmu.edu/technometrics/80-89/VOL-29-04/v2904471.pdf

    Examples
    --------
    >>> statistic = ShapiroWilkExponentialityGofStatistic()
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @override
    def alternative(self) -> Alternative:
        return TwoSidedAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier."""
        return "SW"

    @staticmethod
    @override
    def code():
        """Return the full statistic identifier."""
        short_code = ShapiroWilkExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the ShapiroWilk scalar statistic.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite nonnegative observations; n >= 3. The total must be positive.
            A copy is used. Ties and positive constants are allowed except
            where they make the formula undefined (see Notes on the class).
        **kwargs : dict
            No execution settings are supported; configure the constructor.

        Returns
        -------
        statistic : float
            Value in the convention documented on the class. Infinite
            boundary values are retained where the formula has that limit.

        Raises
        ------
        ValueError
            Invalid sample, insufficient observations, invalid sample-dependent
            setting, undefined ratio, or rescaling underflow of positive data.
        TypeError
            Unsupported execution keyword arguments.
        """
        _settings(kwargs)
        rvs = _sample(rvs, 3)
        if np.ptp(rvs) == 0:
            raise ValueError("Shapiro-Wilk requires a nonconstant sample")
        n = len(rvs)
        # W is affine invariant. Center before taking the mean so a small
        # spread around a large common offset is not lost to cancellation.
        shifted = rvs - np.min(rvs)
        shifted /= np.max(shifted)
        y = np.mean(shifted)
        sw = n * y**2 / ((n - 1) * np.sum((shifted - y) ** 2))
        return float(sw)


class RossbergExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """Rossberg statistic for zero-origin exponential observations.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar without changing the sample or fitting state.
    hypothesis()
        Return the fixed parameters of the null.
    alternative()
        Return the critical tail.

    Notes
    -----
    H0 is Exp(lam), lam > 0 unknown, with known origin zero.
    ``hypothesis().parameters()`` is empty. Positive rescaling leaves
    the statistic unchanged, so Exp(1) simulation is valid.
    Write n for sample size, x_(i) for sorted observations, and
    y=x/mean(x). The implemented formula is::

        mean_k(H(x_k)-G(x_k)), where H averages
        1{x_(2:3)-x_(1:3) < t} over distinct unordered triples and G
        averages 1{min(x_i,x_j) < t} over distinct unordered pairs.
        This is a signed integral contrast with strict inequalities.

    Reject in the upper tail. This is a directional test; it is not
    claimed consistent against every nonexponential alternative.
    Use independent continuous, uncensored observations. Rounded or tied
    data require calibration of the observation process. No parameters
    are retained from a previous call. Rescaling uses a power of two
    to preserve exact binary ties; unrepresentable positive ratios raise
    ValueError. Strict comparisons near floating-point boundaries can
    still depend on roundoff.
    A primary source establishing this exact implemented convention
    was not verified in the review; no published tables or asymptotic
    law are endorsed unless derived explicitly above. See the audit.

    Examples
    --------
    >>> statistic = RossbergExponentialityGofStatistic()
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier."""
        return "RS"

    @staticmethod
    @override
    def code():
        """Return the full statistic identifier."""
        short_code = RossbergExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the Rossberg scalar statistic.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite nonnegative observations; n >= 3. The total must be positive.
            A copy is used. Ties and positive constants are allowed except
            where they make the formula undefined (see Notes on the class).
        **kwargs : dict
            No execution settings are supported; configure the constructor.

        Returns
        -------
        statistic : float
            Value in the convention documented on the class. Infinite
            boundary values are retained where the formula has that limit.

        Raises
        ------
        ValueError
            Invalid sample, insufficient observations, invalid sample-dependent
            setting, undefined ratio, or rescaling underflow of positive data.
        TypeError
            Unsupported execution keyword arguments.
        """
        _settings(kwargs)
        rvs = _sample(rvs, 3)
        n = len(rvs)
        x = np.sort(rvs)
        # Count x_k > gap using binary search; no cancellation of three observations.
        h = sum(
            n - np.searchsorted(x, b - a, side="right") for a, b, c in combinations(x, 3)
        ) / math.comb(n, 3)
        g = sum(n - np.searchsorted(x, a, side="right") for a, b in combinations(x, 2)) / math.comb(
            n, 2
        )
        return float((h - g) / n)


class WeExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """We statistic for zero-origin exponential observations.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar without changing the sample or fitting state.
    hypothesis()
        Return the fixed parameters of the null.
    alternative()
        Return the critical tail.

    Notes
    -----
    H0 is Exp(lam), lam > 0 unknown, with known origin zero.
    ``hypothesis().parameters()`` is empty. Positive rescaling leaves
    the statistic unchanged, so Exp(1) simulation is valid.
    Write n for sample size, x_(i) for sorted observations, and
    y=x/mean(x). The implemented formula is::

        (n-1)*var(y, ddof=0)/n**2. This retained local CV-squared
        functional is not the Shapiro-Wilk W statistic. Its historical class
        name does not establish equivalence to a published WE test.

    Reject in the upper tail. This is a directional test; it is not
    claimed consistent against every nonexponential alternative.
    Use independent continuous, uncensored observations. Rounded or tied
    data require calibration of the observation process. No parameters
    are retained from a previous call. Rescaling uses a power of two
    to preserve exact binary ties; unrepresentable positive ratios raise
    ValueError. Strict comparisons near floating-point boundaries can
    still depend on roundoff.
    A primary source establishing this exact implemented convention
    was not verified in the review; no published tables or asymptotic
    law are endorsed unless derived explicitly above. See the audit.

    Examples
    --------
    >>> statistic = WeExponentialityGofStatistic()
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier."""
        return "WE"

    @staticmethod
    @override
    def code():
        """Return the full statistic identifier."""
        short_code = WeExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the We scalar statistic.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite nonnegative observations; n >= 2. The total must be positive.
            A copy is used. Ties and positive constants are allowed except
            where they make the formula undefined (see Notes on the class).
        **kwargs : dict
            No execution settings are supported; configure the constructor.

        Returns
        -------
        statistic : float
            Value in the convention documented on the class. Infinite
            boundary values are retained where the formula has that limit.

        Raises
        ------
        ValueError
            Invalid sample, insufficient observations, invalid sample-dependent
            setting, undefined ratio, or rescaling underflow of positive data.
        TypeError
            Unsupported execution keyword arguments.
        """
        _settings(kwargs)
        rvs = _sample(rvs, 2)
        n = len(rvs)
        m = np.mean(rvs)
        v = np.var(rvs)
        we = (n - 1) * v / (n**2 * m**2)
        return float(we)


class WongWongExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """WongWong statistic for zero-origin exponential observations.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar without changing the sample or fitting state.
    hypothesis()
        Return the fixed parameters of the null.
    alternative()
        Return the critical tail.

    Notes
    -----
    H0 is Exp(lam), lam > 0 unknown, with known origin zero.
    ``hypothesis().parameters()`` is empty. Positive rescaling leaves
    the statistic unchanged, so Exp(1) simulation is valid.
    Write n for sample size, x_(i) for sorted observations, and
    y=x/mean(x). The implemented formula is::

        max(x)/min(x), the raw extremal quotient, without a log
        transform. A zero minimum and positive maximum give positive infinity.

    Reject in the upper tail. This is a directional test; it is not
    claimed consistent against every nonexponential alternative.
    Use independent continuous, uncensored observations. Rounded or tied
    data require calibration of the observation process. No parameters
    are retained from a previous call. Rescaling uses a power of two
    to preserve exact binary ties; unrepresentable positive ratios raise
    ValueError. Strict comparisons near floating-point boundaries can
    still depend on roundoff.
    A primary source establishing this exact implemented convention
    was not verified in the review; no published tables or asymptotic
    law are endorsed unless derived explicitly above. See the audit.

    Examples
    --------
    >>> statistic = WongWongExponentialityGofStatistic()
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier."""
        return "WW"

    @staticmethod
    @override
    def code():
        """Return the full statistic identifier."""
        short_code = WongWongExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the WongWong scalar statistic.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite nonnegative observations; n >= 2. The total must be positive.
            A copy is used. Ties and positive constants are allowed except
            where they make the formula undefined (see Notes on the class).
        **kwargs : dict
            No execution settings are supported; configure the constructor.

        Returns
        -------
        statistic : float
            Value in the convention documented on the class. Infinite
            boundary values are retained where the formula has that limit.

        Raises
        ------
        ValueError
            Invalid sample, insufficient observations, invalid sample-dependent
            setting, undefined ratio, or rescaling underflow of positive data.
        TypeError
            Unsupported execution keyword arguments.
        """
        _settings(kwargs)
        rvs = _sample(rvs, 2)
        return _ratio(np.max(rvs), np.min(rvs))


class HegazyGreen2ExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """HegazyGreen2 statistic for zero-origin exponential observations.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar without changing the sample or fitting state.
    hypothesis()
        Return the fixed parameters of the null.
    alternative()
        Return the critical tail.

    Notes
    -----
    H0 is Exp(lam), lam > 0 unknown, with known origin zero.
    ``hypothesis().parameters()`` is empty. Positive rescaling leaves
    the statistic unchanged, so Exp(1) simulation is valid.
    Write n for sample size, x_(i) for sorted observations, and
    y=x/mean(x). The implemented formula is::

        mean((y_(i) + log(1-i/(n+1)))**2).
        This is a fitted exponential quantile discrepancy.

    Reject in the upper tail. Calibrate the exact statistic returned here.
    Use independent continuous, uncensored observations. Rounded or tied
    data require calibration of the observation process. No parameters
    are retained from a previous call. Rescaling uses a power of two
    to preserve exact binary ties; unrepresentable positive ratios raise
    ValueError. Strict comparisons near floating-point boundaries can
    still depend on roundoff.
    A primary source establishing this exact implemented convention
    was not verified in the review; no published tables or asymptotic
    law are endorsed unless derived explicitly above. See the audit.
    The original Hegazy-Green paper concerns uniform/normal models;
    it does not establish this fitted exponential adaptation.

    Examples
    --------
    >>> statistic = HegazyGreen2ExponentialityGofStatistic()
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier."""
        return "HG2"

    @staticmethod
    @override
    def code():
        """Return the full statistic identifier."""
        short_code = HegazyGreen2ExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the HegazyGreen2 scalar statistic.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite nonnegative observations; n >= 2. The total must be positive.
            A copy is used. Ties and positive constants are allowed except
            where they make the formula undefined (see Notes on the class).
        **kwargs : dict
            No execution settings are supported; configure the constructor.

        Returns
        -------
        statistic : float
            Value in the convention documented on the class. Infinite
            boundary values are retained where the formula has that limit.

        Raises
        ------
        ValueError
            Invalid sample, insufficient observations, invalid sample-dependent
            setting, undefined ratio, or rescaling underflow of positive data.
        TypeError
            Unsupported execution keyword arguments.
        """
        _settings(kwargs)
        rvs = _sample(rvs, 2)
        rvs = rvs / np.mean(rvs)
        n = len(rvs)
        rvs.sort()
        b = -np.log(1 - np.arange(1, n + 1) / (n + 1))
        hg = n ** (-1) * np.sum((rvs - b) ** 2)
        return float(hg)


class AbstractGraphExponentialityGofStatistic(
    AbstractExponentialityGofStatistic, AbstractGraphTestStatistic
):
    """Scale-invariant proximity graphs with threshold range/10."""

    @staticmethod
    @override
    def code():
        return f"GRAPH_{AbstractExponentialityGofStatistic.code()}"

    @staticmethod
    @override
    def _compute_dist(rvs):
        return float((np.max(rvs) - np.min(rvs)) / 10)

    def _graph_statistic(self, rvs):
        distance = self._compute_dist(rvs)
        graph = self._make_adjacency_list(rvs, distance)
        return float(self.get_graph_stat(graph))


class GraphEdgesNumberExponentialityGofStatistic(
    AbstractGraphExponentialityGofStatistic, GraphEdgesNumberTestStatistic
):
    """GraphEdgesNumber statistic for zero-origin exponential observations.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar without changing the sample or fitting state.
    hypothesis()
        Return the fixed parameters of the null.
    alternative()
        Return the critical tail.

    Notes
    -----
    H0 is Exp(lam), lam > 0 unknown, with known origin zero.
    ``hypothesis().parameters()`` is empty. Positive rescaling leaves
    the statistic unchanged, so Exp(1) simulation is valid.
    Write n for sample size, x_(i) for sorted observations, and
    y=x/mean(x). The implemented formula is::

        Vertices are observations; an edge joins distinct vertices when
        abs(x_i-x_j) < (max(x)-min(x))/10. The statistic is the edge count.
        Constant samples have no edges. This range-based graph is also
        translation invariant and cannot identify a nonzero origin.

    Reject in both tails. Calibrate the exact statistic returned here.
    Use independent continuous, uncensored observations. Rounded or tied
    data require calibration of the observation process. No parameters
    are retained from a previous call. Rescaling uses a power of two
    to preserve exact binary ties; unrepresentable positive ratios raise
    ValueError. Strict comparisons near floating-point boundaries can
    still depend on roundoff.
    A primary source establishing this exact implemented convention
    was not verified in the review; no published tables or asymptotic
    law are endorsed unless derived explicitly above. See the audit.

    Examples
    --------
    >>> statistic = GraphEdgesNumberExponentialityGofStatistic()
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @override
    def alternative(self) -> Alternative:
        return TwoSidedAlternative()

    @staticmethod
    @override
    def code():
        """Return the full statistic identifier."""
        parent_code = AbstractGraphExponentialityGofStatistic.code()
        short_code = GraphEdgesNumberExponentialityGofStatistic.short_code()
        return f"{short_code}_{parent_code}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the GraphEdgesNumber scalar statistic.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite nonnegative observations; n >= 1. All-zero samples are allowed.
            A copy is used. Ties and positive constants are allowed except
            where they make the formula undefined (see Notes on the class).
        **kwargs : dict
            No execution settings are supported; configure the constructor.

        Returns
        -------
        statistic : float
            Value in the convention documented on the class. Infinite
            boundary values are retained where the formula has that limit.

        Raises
        ------
        ValueError
            Invalid sample, insufficient observations, invalid sample-dependent
            setting, undefined ratio, or rescaling underflow of positive data.
        TypeError
            Unsupported execution keyword arguments.
        """
        _settings(kwargs)
        rvs = _sample(rvs, 1, positive_total=False)
        return self._graph_statistic(rvs)


class GraphMaxDegreeExponentialityGofStatistic(
    AbstractGraphExponentialityGofStatistic, GraphMaxDegreeTestStatistic
):
    """GraphMaxDegree statistic for zero-origin exponential observations.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar without changing the sample or fitting state.
    hypothesis()
        Return the fixed parameters of the null.
    alternative()
        Return the critical tail.

    Notes
    -----
    H0 is Exp(lam), lam > 0 unknown, with known origin zero.
    ``hypothesis().parameters()`` is empty. Positive rescaling leaves
    the statistic unchanged, so Exp(1) simulation is valid.
    Write n for sample size, x_(i) for sorted observations, and
    y=x/mean(x). The implemented formula is::

        Vertices are observations; an edge joins distinct vertices when
        abs(x_i-x_j) < (max(x)-min(x))/10. The statistic is the maximum degree.
        Constant samples have no edges. This range-based graph is also
        translation invariant and cannot identify a nonzero origin.

    Reject in both tails. Calibrate the exact statistic returned here.
    Use independent continuous, uncensored observations. Rounded or tied
    data require calibration of the observation process. No parameters
    are retained from a previous call. Rescaling uses a power of two
    to preserve exact binary ties; unrepresentable positive ratios raise
    ValueError. Strict comparisons near floating-point boundaries can
    still depend on roundoff.
    A primary source establishing this exact implemented convention
    was not verified in the review; no published tables or asymptotic
    law are endorsed unless derived explicitly above. See the audit.

    Examples
    --------
    >>> statistic = GraphMaxDegreeExponentialityGofStatistic()
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @override
    def alternative(self) -> Alternative:
        return TwoSidedAlternative()

    @staticmethod
    @override
    def code():
        """Return the full statistic identifier."""
        parent_code = AbstractGraphExponentialityGofStatistic.code()
        short_code = GraphMaxDegreeExponentialityGofStatistic.short_code()
        return f"{short_code}_{parent_code}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the GraphMaxDegree scalar statistic.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite nonnegative observations; n >= 1. All-zero samples are allowed.
            A copy is used. Ties and positive constants are allowed except
            where they make the formula undefined (see Notes on the class).
        **kwargs : dict
            No execution settings are supported; configure the constructor.

        Returns
        -------
        statistic : float
            Value in the convention documented on the class. Infinite
            boundary values are retained where the formula has that limit.

        Raises
        ------
        ValueError
            Invalid sample, insufficient observations, invalid sample-dependent
            setting, undefined ratio, or rescaling underflow of positive data.
        TypeError
            Unsupported execution keyword arguments.
        """
        _settings(kwargs)
        rvs = _sample(rvs, 1, positive_total=False)
        return self._graph_statistic(rvs)


class GraphAverageDegreeExponentialityGofStatistic(
    AbstractGraphExponentialityGofStatistic, GraphAverageDegreeTestStatistic
):
    """GraphAverageDegree statistic for zero-origin exponential observations.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar without changing the sample or fitting state.
    hypothesis()
        Return the fixed parameters of the null.
    alternative()
        Return the critical tail.

    Notes
    -----
    H0 is Exp(lam), lam > 0 unknown, with known origin zero.
    ``hypothesis().parameters()`` is empty. Positive rescaling leaves
    the statistic unchanged, so Exp(1) simulation is valid.
    Write n for sample size, x_(i) for sorted observations, and
    y=x/mean(x). The implemented formula is::

        Vertices are observations; an edge joins distinct vertices when
        abs(x_i-x_j) < (max(x)-min(x))/10. The statistic is twice the edge count divided by n.
        Constant samples have no edges. This range-based graph is also
        translation invariant and cannot identify a nonzero origin.

    Reject in both tails. Calibrate the exact statistic returned here.
    Use independent continuous, uncensored observations. Rounded or tied
    data require calibration of the observation process. No parameters
    are retained from a previous call. Rescaling uses a power of two
    to preserve exact binary ties; unrepresentable positive ratios raise
    ValueError. Strict comparisons near floating-point boundaries can
    still depend on roundoff.
    A primary source establishing this exact implemented convention
    was not verified in the review; no published tables or asymptotic
    law are endorsed unless derived explicitly above. See the audit.

    Examples
    --------
    >>> statistic = GraphAverageDegreeExponentialityGofStatistic()
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @override
    def alternative(self) -> Alternative:
        return TwoSidedAlternative()

    @staticmethod
    @override
    def code():
        """Return the full statistic identifier."""
        parent_code = AbstractGraphExponentialityGofStatistic.code()
        short_code = GraphAverageDegreeExponentialityGofStatistic.short_code()
        return f"{short_code}_{parent_code}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the GraphAverageDegree scalar statistic.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite nonnegative observations; n >= 1. All-zero samples are allowed.
            A copy is used. Ties and positive constants are allowed except
            where they make the formula undefined (see Notes on the class).
        **kwargs : dict
            No execution settings are supported; configure the constructor.

        Returns
        -------
        statistic : float
            Value in the convention documented on the class. Infinite
            boundary values are retained where the formula has that limit.

        Raises
        ------
        ValueError
            Invalid sample, insufficient observations, invalid sample-dependent
            setting, undefined ratio, or rescaling underflow of positive data.
        TypeError
            Unsupported execution keyword arguments.
        """
        _settings(kwargs)
        rvs = _sample(rvs, 1, positive_total=False)
        return self._graph_statistic(rvs)


class GraphConnectedComponentsExponentialityGofStatistic(
    AbstractGraphExponentialityGofStatistic, GraphConnectedComponentsTestStatistic
):
    """GraphConnectedComponents statistic for zero-origin exponential observations.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar without changing the sample or fitting state.
    hypothesis()
        Return the fixed parameters of the null.
    alternative()
        Return the critical tail.

    Notes
    -----
    H0 is Exp(lam), lam > 0 unknown, with known origin zero.
    ``hypothesis().parameters()`` is empty. Positive rescaling leaves
    the statistic unchanged, so Exp(1) simulation is valid.
    Write n for sample size, x_(i) for sorted observations, and
    y=x/mean(x). The implemented formula is::

        Vertices are observations; an edge joins distinct vertices when
        abs(x_i-x_j) < (max(x)-min(x))/10. The statistic is the number of connected components.
        Constant samples have no edges. This range-based graph is also
        translation invariant and cannot identify a nonzero origin.

    Reject in both tails. Calibrate the exact statistic returned here.
    Use independent continuous, uncensored observations. Rounded or tied
    data require calibration of the observation process. No parameters
    are retained from a previous call. Rescaling uses a power of two
    to preserve exact binary ties; unrepresentable positive ratios raise
    ValueError. Strict comparisons near floating-point boundaries can
    still depend on roundoff.
    A primary source establishing this exact implemented convention
    was not verified in the review; no published tables or asymptotic
    law are endorsed unless derived explicitly above. See the audit.

    Examples
    --------
    >>> statistic = GraphConnectedComponentsExponentialityGofStatistic()
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @override
    def alternative(self) -> Alternative:
        return TwoSidedAlternative()

    @staticmethod
    @override
    def code():
        """Return the full statistic identifier."""
        parent_code = AbstractGraphExponentialityGofStatistic.code()
        short_code = GraphConnectedComponentsExponentialityGofStatistic.short_code()
        return f"{short_code}_{parent_code}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the GraphConnectedComponents scalar statistic.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite nonnegative observations; n >= 1. All-zero samples are allowed.
            A copy is used. Ties and positive constants are allowed except
            where they make the formula undefined (see Notes on the class).
        **kwargs : dict
            No execution settings are supported; configure the constructor.

        Returns
        -------
        statistic : float
            Value in the convention documented on the class. Infinite
            boundary values are retained where the formula has that limit.

        Raises
        ------
        ValueError
            Invalid sample, insufficient observations, invalid sample-dependent
            setting, undefined ratio, or rescaling underflow of positive data.
        TypeError
            Unsupported execution keyword arguments.
        """
        _settings(kwargs)
        rvs = _sample(rvs, 1, positive_total=False)
        return self._graph_statistic(rvs)


class GraphCliqueNumberExponentialityGofStatistic(
    AbstractGraphExponentialityGofStatistic, GraphCliqueNumberTestStatistic
):
    """GraphCliqueNumber statistic for zero-origin exponential observations.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar without changing the sample or fitting state.
    hypothesis()
        Return the fixed parameters of the null.
    alternative()
        Return the critical tail.

    Notes
    -----
    H0 is Exp(lam), lam > 0 unknown, with known origin zero.
    ``hypothesis().parameters()`` is empty. Positive rescaling leaves
    the statistic unchanged, so Exp(1) simulation is valid.
    Write n for sample size, x_(i) for sorted observations, and
    y=x/mean(x). The implemented formula is::

        Vertices are observations; an edge joins distinct vertices when
        abs(x_i-x_j) < (max(x)-min(x))/10. The statistic is the size of the largest clique.
        Constant samples have no edges. This range-based graph is also
        translation invariant and cannot identify a nonzero origin.

    Reject in both tails. Calibrate the exact statistic returned here.
    Use independent continuous, uncensored observations. Rounded or tied
    data require calibration of the observation process. No parameters
    are retained from a previous call. Rescaling uses a power of two
    to preserve exact binary ties; unrepresentable positive ratios raise
    ValueError. Strict comparisons near floating-point boundaries can
    still depend on roundoff.
    A primary source establishing this exact implemented convention
    was not verified in the review; no published tables or asymptotic
    law are endorsed unless derived explicitly above. See the audit.

    Examples
    --------
    >>> statistic = GraphCliqueNumberExponentialityGofStatistic()
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @override
    def alternative(self) -> Alternative:
        return TwoSidedAlternative()

    @staticmethod
    @override
    def code():
        """Return the full statistic identifier."""
        parent_code = AbstractGraphExponentialityGofStatistic.code()
        short_code = GraphCliqueNumberExponentialityGofStatistic.short_code()
        return f"{short_code}_{parent_code}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the GraphCliqueNumber scalar statistic.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite nonnegative observations; n >= 1. All-zero samples are allowed.
            A copy is used. Ties and positive constants are allowed except
            where they make the formula undefined (see Notes on the class).
        **kwargs : dict
            No execution settings are supported; configure the constructor.

        Returns
        -------
        statistic : float
            Value in the convention documented on the class. Infinite
            boundary values are retained where the formula has that limit.

        Raises
        ------
        ValueError
            Invalid sample, insufficient observations, invalid sample-dependent
            setting, undefined ratio, or rescaling underflow of positive data.
        TypeError
            Unsupported execution keyword arguments.
        """
        _settings(kwargs)
        rvs = _sample(rvs, 1, positive_total=False)
        x = np.sort(rvs)
        distance = self._compute_dist(x)
        if distance == 0:
            return 1.0
        right = 0
        largest = 1
        for left in range(len(x)):
            while right < len(x) and x[right] - x[left] < distance:
                right += 1
            largest = max(largest, right - left)
        return float(largest)


class GraphIndependenceNumberExponentialityGofStatistic(
    AbstractGraphExponentialityGofStatistic, GraphIndependenceNumberTestStatistic
):
    """GraphIndependenceNumber statistic for zero-origin exponential observations.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar without changing the sample or fitting state.
    hypothesis()
        Return the fixed parameters of the null.
    alternative()
        Return the critical tail.

    Notes
    -----
    H0 is Exp(lam), lam > 0 unknown, with known origin zero.
    ``hypothesis().parameters()`` is empty. Positive rescaling leaves
    the statistic unchanged, so Exp(1) simulation is valid.
    Write n for sample size, x_(i) for sorted observations, and
    y=x/mean(x). The implemented formula is::

        Vertices are observations; an edge joins distinct vertices when
        abs(x_i-x_j) < (max(x)-min(x))/10. The statistic is the size of the largest independent set.
        Constant samples have no edges. This range-based graph is also
        translation invariant and cannot identify a nonzero origin.

    Reject in both tails. Calibrate the exact statistic returned here.
    Use independent continuous, uncensored observations. Rounded or tied
    data require calibration of the observation process. No parameters
    are retained from a previous call. Rescaling uses a power of two
    to preserve exact binary ties; unrepresentable positive ratios raise
    ValueError. Strict comparisons near floating-point boundaries can
    still depend on roundoff.
    A primary source establishing this exact implemented convention
    was not verified in the review; no published tables or asymptotic
    law are endorsed unless derived explicitly above. See the audit.

    Examples
    --------
    >>> statistic = GraphIndependenceNumberExponentialityGofStatistic()
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @override
    def alternative(self) -> Alternative:
        return TwoSidedAlternative()

    @staticmethod
    @override
    def code():
        """Return the full statistic identifier."""
        parent_code = AbstractGraphExponentialityGofStatistic.code()
        short_code = GraphIndependenceNumberExponentialityGofStatistic.short_code()
        return f"{short_code}_{parent_code}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the GraphIndependenceNumber scalar statistic.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite nonnegative observations; n >= 1. All-zero samples are allowed.
            A copy is used. Ties and positive constants are allowed except
            where they make the formula undefined (see Notes on the class).
        **kwargs : dict
            No execution settings are supported; configure the constructor.

        Returns
        -------
        statistic : float
            Value in the convention documented on the class. Infinite
            boundary values are retained where the formula has that limit.

        Raises
        ------
        ValueError
            Invalid sample, insufficient observations, invalid sample-dependent
            setting, undefined ratio, or rescaling underflow of positive data.
        TypeError
            Unsupported execution keyword arguments.
        """
        _settings(kwargs)
        rvs = _sample(rvs, 1, positive_total=False)
        x = np.sort(rvs)
        distance = self._compute_dist(x)
        last = x[0]
        count = 1
        for value in x[1:]:
            if value - last >= distance:
                count += 1
                last = value
        return float(count)
