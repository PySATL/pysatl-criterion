"""Exponential goodness-of-fit statistics for observations with known origin zero.

See ``docs/criteria/goodness-of-fit/exponentiality/mathematical-validation.md``
for verification, calibration migration and unresolved source limitations.
Composite statistics are invariant under positive rescaling; Monte Carlo
calibration may therefore use Exp(1).
"""

from abc import ABC

import numpy as np
import scipy.special as scipy_special
from numba import jit
from typing_extensions import override

from pysatl_criterion.distribution import ExponentialDistributionDescriptor
from pysatl_criterion.distribution import (
    ExponentialDistributionDescriptor as Distribution,
)
from pysatl_criterion.distribution.parameters import HypothesisSupport, ParameterValues
from pysatl_criterion.statistics.alternative import (
    Alternative,
    AlternativeType,
    LeftAlternative,
    RightAlternative,
    TwoSidedAlternative,
)
from pysatl_criterion.statistics.goodness_of_fit.binning import EqualProbabilityBins
from pysatl_criterion.statistics.goodness_of_fit.continuous import edf, frequency
from pysatl_criterion.statistics.goodness_of_fit.continuous._edf_factory import (
    EDFSpec,
    export_edf_statistics,
)
from pysatl_criterion.statistics.goodness_of_fit.continuous._frequency_factory import (
    FrequencySpec,
    export_frequency_statistics,
)
from pysatl_criterion.statistics.goodness_of_fit.continuous.graph_goodness_of_fit import (
    AbstractGraphTestStatistic,
    GraphAverageDegreeTestStatistic,
    GraphCliqueNumberTestStatistic,
    GraphConnectedComponentsTestStatistic,
    GraphEdgesNumberTestStatistic,
    GraphIndependenceNumberTestStatistic,
    GraphMaxDegreeTestStatistic,
)
from pysatl_criterion.statistics.goodness_of_fit.goodness_of_fit_statistic import (
    AbstractGoodnessOfFitStatistic,
)


def _sample(rvs, minimum=2, *, rescale=True, positive_total=True):
    """Validate a sample and return an independent, optionally rescaled copy."""
    if np.ma.isMaskedArray(rvs) and np.any(np.ma.getmaskarray(rvs)):
        raise ValueError("rvs must not contain masked observations")
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

    @property
    def lam(self) -> float:
        """Read lam by its stable parameter identity."""
        return self._parameters[Distribution.RATE]

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

    @classmethod
    @override
    def code(cls) -> str:
        """Return the family identifier or the concrete statistic's full identifier."""
        family_code = f"EXPONENTIALITY_{AbstractGoodnessOfFitStatistic.code()}"
        if "short_code" in cls.__abstractmethods__:
            return family_code
        return f"{cls.short_code()}_{family_code}"

    @staticmethod
    @override
    def distribution() -> type[ExponentialDistributionDescriptor]:
        """Return the distribution descriptor class."""
        return ExponentialDistributionDescriptor


class EppsPulleyExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """EppsPulley statistic for zero-origin exponential observations.

    Parameters
    ----------
    parameters : ParameterValues
        An empty distribution schema; all distribution parameters are unknown.


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

    Reject in both tails. This signed contrast is not
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
    >>> parameters = Distribution.DEFAULT.parse({})
    >>> statistic = EppsPulleyExponentialityGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @override
    def alternative(self) -> Alternative:
        return TwoSidedAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier."""
        return "EP"

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
        return self._statistic(rvs)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _statistic(rvs):
        n = len(rvs)
        y = rvs / np.mean(rvs)
        ep = np.sqrt(48 * n) * np.sum(np.exp(-y) - 1 / 2) / n
        return float(ep)


class AhsanullahExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """Ahsanullah statistic for zero-origin exponential observations.

    Parameters
    ----------
    parameters : ParameterValues
        An empty distribution schema; all distribution parameters are unknown.


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
    >>> parameters = Distribution.DEFAULT.parse({})
    >>> statistic = AhsanullahExponentialityGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier."""
        return "AHS"

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
        return self._statistic(rvs)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _statistic(rvs):
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
    parameters : ParameterValues
        An empty distribution schema; all distribution parameters are unknown.
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
    >>> parameters = Distribution.DEFAULT.parse({})
    >>> statistic = AtkinsonExponentialityGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    def __init__(self, parameters: ParameterValues, *, p=0.99):
        AbstractGoodnessOfFitStatistic.__init__(self, parameters)
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
        return self._statistic(rvs, p, reference)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _statistic(rvs, p, reference):
        if p < 0 and np.any(rvs == 0):
            empirical = 0.0
        else:
            logs = np.log(rvs) - np.log(np.mean(rvs))
            # At this scale the power correction is below float64 precision.
            # Avoid multiplying by a subnormal p and then dividing by it.
            if abs(p) < 1e-100 and np.all(np.isfinite(logs)):
                empirical = np.exp(np.mean(logs))
                return float(np.sqrt(len(rvs)) * abs(empirical - np.exp(-np.euler_gamma)))
            powers = p * logs
            if np.all(np.isfinite(powers)) and np.max(np.abs(powers)) < 0.5:
                # Avoid cancellation of log(n) terms when p is near zero.
                empirical = np.exp(np.log1p(np.mean(np.expm1(powers))) / p)
            else:
                pivot = np.max(logs) if p > 0 else np.min(logs)
                log_mean = np.log(np.sum(np.exp(p * (logs - pivot)))) - np.log(len(rvs))
                empirical = np.exp(pivot + log_mean / p)
        return float(np.sqrt(len(rvs)) * abs(empirical - reference))


class CoxOakesExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """CoxOakes statistic for zero-origin exponential observations.

    Parameters
    ----------
    parameters : ParameterValues
        An empty distribution schema; all distribution parameters are unknown.


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
    >>> parameters = Distribution.DEFAULT.parse({})
    >>> statistic = CoxOakesExponentialityGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @override
    def alternative(self) -> Alternative:
        return TwoSidedAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier."""
        return "CO"

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
        return self._statistic(rvs)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _statistic(rvs):
        if np.any(rvs == 0):
            return float("-inf")
        n = len(rvs)
        y = rvs / np.mean(rvs)
        y = np.log(y) * (1 - y)
        co = np.sum(y) + n
        return float(co)


class DeshpandeExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """Deshpande statistic for zero-origin exponential observations.

    Parameters
    ----------
    parameters : ParameterValues
        An empty distribution schema; all distribution parameters are unknown.
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
    >>> parameters = Distribution.DEFAULT.parse({})
    >>> statistic = DeshpandeExponentialityGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    def __init__(self, parameters: ParameterValues, *, b=0.44):
        AbstractGoodnessOfFitStatistic.__init__(self, parameters)
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
        return self._statistic(rvs, self.b)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _statistic(rvs, b):
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

    Parameters
    ----------
    parameters : ParameterValues
        An empty distribution schema; all distribution parameters are unknown.


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
    >>> parameters = Distribution.DEFAULT.parse({})
    >>> statistic = EpsteinExponentialityGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier."""
        return "EPS"

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
        return self._statistic(_spacings(rvs))

    @staticmethod
    @jit(nopython=True, cache=True)
    def _statistic(d):
        if np.any(d == 0):
            return float("inf")
        n = len(d)
        return float(2 * n * (np.log(np.mean(d)) - np.mean(np.log(d))) / (1 + (n + 1) / (6 * n)))


class FroziniExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """Frozini statistic for zero-origin exponential observations.

    Parameters
    ----------
    parameters : ParameterValues
        An empty distribution schema; all distribution parameters are unknown.


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
    >>> parameters = Distribution.DEFAULT.parse({})
    >>> statistic = FroziniExponentialityGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier."""
        return "FZ"

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
        return self._statistic(rvs)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _statistic(rvs):
        n = len(rvs)
        rvs.sort()
        y = np.mean(rvs)
        froz = (
            1 / np.sqrt(n) * np.sum(np.abs(-np.expm1(-rvs / y) - (np.arange(1, n + 1) - 0.5) / n))
        )
        return float(froz)


class GiniExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """Gini statistic for zero-origin exponential observations.

    Parameters
    ----------
    parameters : ParameterValues
        An empty distribution schema; all distribution parameters are unknown.


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
    >>> parameters = Distribution.DEFAULT.parse({})
    >>> statistic = GiniExponentialityGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @override
    def alternative(self) -> Alternative:
        return TwoSidedAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier."""
        return "GINI"

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
        return self._statistic(rvs)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _statistic(rvs):
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
    parameters : ParameterValues
        An empty distribution schema; all distribution parameters are unknown.
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

    Reject in both tails, using equal tail probabilities under the null
    F distribution. Calibrate for the sample size and split count r.
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
    >>> parameters = Distribution.DEFAULT.parse({})
    >>> statistic = GnedenkoExponentialityGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    def __init__(self, parameters: ParameterValues, *, r=None):
        AbstractGoodnessOfFitStatistic.__init__(self, parameters)
        if r is not None and (isinstance(r, bool) or not isinstance(r, (int, np.integer)) or r < 1):
            raise ValueError("r must be a positive integer or None")
        self.r = r

    @override
    def alternative(self) -> Alternative:
        return TwoSidedAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier."""
        return "GD"

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
        return self._statistic(_spacings(rvs), r)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _statistic(d, r):
        numerator = np.mean(d[:r])
        denominator = np.mean(d[r:])
        if denominator == 0:
            if numerator == 0:
                raise ValueError("The ratio is undefined (0/0)")
            return float("inf")
        return float(numerator / denominator)


class HarrisExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """Harris statistic for zero-origin exponential observations.

    Parameters
    ----------
    parameters : ParameterValues
        An empty distribution schema; all distribution parameters are unknown.
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

    Reject in both tails, using equal tail probabilities under the null
    F distribution. Calibrate for the sample size and split count r.
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
    >>> parameters = Distribution.DEFAULT.parse({})
    >>> statistic = HarrisExponentialityGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    def __init__(self, parameters: ParameterValues, *, r=None):
        AbstractGoodnessOfFitStatistic.__init__(self, parameters)
        if r is not None and (isinstance(r, bool) or not isinstance(r, (int, np.integer)) or r < 1):
            raise ValueError("r must be a positive integer or None")
        self.r = r

    @override
    def alternative(self) -> Alternative:
        return TwoSidedAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier."""
        return "HM"

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
        return self._statistic(_spacings(rvs), r)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _statistic(d, r):
        numerator = (np.sum(d[:r]) + np.sum(d[-r:])) / (2 * r)
        denominator = np.mean(d[r:-r])
        if denominator == 0:
            if numerator == 0:
                raise ValueError("The ratio is undefined (0/0)")
            return float("inf")
        return float(numerator / denominator)


class HegazyGreen1ExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """HegazyGreen1 statistic for zero-origin exponential observations.

    Parameters
    ----------
    parameters : ParameterValues
        An empty distribution schema; all distribution parameters are unknown.


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
    Scientific reference for the exponential variants:
    N. G. Ushakov and V. G. Ushakov, "On sensitivity of exponentiality
    tests to data rounding: a Monte Carlo simulation study",
    https://doi.org/10.1080/03610918.2021.2009868.
    Historical reference: Hegazy and Green (1975), "Some new
    goodness-of-fit tests using order statistics",
    https://doi.org/10.2307/2347090.
    Scientific attribution is established. Historical publication priority
    is separate from verification of the formula and calibration.

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({})
    >>> statistic = HegazyGreen1ExponentialityGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier."""
        return "HG1"

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
        return self._statistic(rvs)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _statistic(rvs):
        rvs = rvs / np.mean(rvs)
        n = len(rvs)
        x = np.sort(rvs)
        b = -np.log(1 - np.arange(1, n + 1) / (n + 1))
        hg = (1.0 / n) * np.sum(np.abs(x - b))
        return float(hg)


class HollanderProshanExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """HollanderProshan statistic for zero-origin exponential observations.

    Parameters
    ----------
    parameters : ParameterValues
        An empty distribution schema; all distribution parameters are unknown.


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

    Reject in the lower tail against NBU alternatives: under NBU,
    P(X > Y+Z) <= 1/4, with equality under the exponential null.
    This is not claimed consistent against every nonexponential alternative.
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
    >>> parameters = Distribution.DEFAULT.parse({})
    >>> statistic = HollanderProshanExponentialityGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @override
    def alternative(self) -> Alternative:
        return LeftAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier."""
        return "HP"

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
        return self._statistic(rvs)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _statistic(rvs):
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

    Parameters
    ----------
    parameters : ParameterValues
        An empty distribution schema; all distribution parameters are unknown.


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
    >>> parameters = Distribution.DEFAULT.parse({})
    >>> statistic = KimberMichaelExponentialityGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier."""
        return "KM"

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
        return self._statistic(rvs)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _statistic(rvs):
        n = len(rvs)
        rvs.sort()
        y = np.mean(rvs)
        s = 2 / np.pi * np.arcsin(np.sqrt(-np.expm1(-rvs / y)))
        r = 2 / np.pi * np.arcsin(np.sqrt((np.arange(1, n + 1) - 0.5) / n))
        km = np.max(np.abs(r - s))
        return float(km)


class KocharExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """Kochar statistic for zero-origin exponential observations.

    Parameters
    ----------
    parameters : ParameterValues
        An empty distribution schema; all distribution parameters are unknown.


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
    >>> parameters = Distribution.DEFAULT.parse({})
    >>> statistic = KocharExponentialityGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier."""
        return "KC"

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
        return self._statistic(rvs)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _statistic(rvs):
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
    parameters : ParameterValues
        An empty distribution schema; all distribution parameters are unknown.
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
    >>> parameters = Distribution.DEFAULT.parse({})
    >>> statistic = LorenzExponentialityGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    def __init__(self, parameters: ParameterValues, *, p=0.5):
        AbstractGoodnessOfFitStatistic.__init__(self, parameters)
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
        return self._statistic(rvs, self.p)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _statistic(rvs, p):
        k = int(len(rvs) * p)
        if not 1 <= k < len(rvs):
            raise ValueError("p must select at least one and fewer than n observations")
        return float(np.sum(np.sort(rvs)[:k]) / np.sum(rvs))


class MoranExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """Moran statistic for zero-origin exponential observations.

    Parameters
    ----------
    parameters : ParameterValues
        An empty distribution schema; all distribution parameters are unknown.


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

    Reject in both tails. This signed contrast is not
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
    >>> parameters = Distribution.DEFAULT.parse({})
    >>> statistic = MoranExponentialityGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @override
    def alternative(self) -> Alternative:
        return TwoSidedAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier."""
        return "MN"

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
        return self._statistic(rvs)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _statistic(rvs):
        if np.any(rvs == 0):
            return float("-inf")
        y = np.mean(rvs)
        mn = np.euler_gamma + np.mean(np.log(rvs / y))
        return float(mn)


class PietraExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """Pietra statistic for zero-origin exponential observations.

    Parameters
    ----------
    parameters : ParameterValues
        An empty distribution schema; all distribution parameters are unknown.


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
    >>> parameters = Distribution.DEFAULT.parse({})
    >>> statistic = PietraExponentialityGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @override
    def alternative(self) -> Alternative:
        return TwoSidedAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier."""
        return "PT"

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
        return self._statistic(rvs)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _statistic(rvs):
        n = len(rvs)
        xm = np.mean(rvs)
        pt = np.sum(np.abs(rvs - xm)) / (2 * n * xm)
        return float(pt)


class ShapiroWilkExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """ShapiroWilk statistic for zero-origin exponential observations.

    Parameters
    ----------
    parameters : ParameterValues
        An empty distribution schema; all distribution parameters are unknown.


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
    >>> parameters = Distribution.DEFAULT.parse({})
    >>> statistic = ShapiroWilkExponentialityGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @override
    def alternative(self) -> Alternative:
        return TwoSidedAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier."""
        return "SW"

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
        return self._statistic(rvs)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _statistic(rvs):
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

    Parameters
    ----------
    parameters : ParameterValues
        An empty distribution schema; all distribution parameters are unknown.


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
    >>> parameters = Distribution.DEFAULT.parse({})
    >>> statistic = RossbergExponentialityGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier."""
        return "RS"

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
        return self._statistic(rvs)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _statistic(rvs):
        n = len(rvs)
        x = np.sort(rvs)
        h = 0.0
        g = 0.0
        for i in range(n - 1):
            g += (n - i - 1) * (n - np.searchsorted(x, x[i], side="right"))
            for j in range(i + 1, n - 1):
                # Every later third observation gives the same gap.
                h += (n - j - 1) * (n - np.searchsorted(x, x[j] - x[i], side="right"))
        h /= n * (n - 1) * (n - 2) / 6
        g /= n * (n - 1) / 2
        return float((h - g) / n)


class WeExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """We statistic for zero-origin exponential observations.

    Parameters
    ----------
    parameters : ParameterValues
        An empty distribution schema; all distribution parameters are unknown.


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

        var(y, ddof=0)/n
        = sum((x-mean(x))**2)/sum(x)**2 = Greenwood - 1/n.

    This is the WE normalization associated with Ascher (1990),
    DOI 10.1080/03610929008830292, as reproduced in the later
    rounding study, DOI 10.1080/03610918.2021.2009868. See the audit.

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
    >>> parameters = Distribution.DEFAULT.parse({})
    >>> statistic = WeExponentialityGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier."""
        return "WE"

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
        return self._statistic(rvs)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _statistic(rvs):
        n = len(rvs)
        m = np.mean(rvs)
        v = np.var(rvs)
        we = v / (n * m**2)
        return float(we)


class WongWongExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """WongWong statistic for zero-origin exponential observations.

    Parameters
    ----------
    parameters : ParameterValues
        An empty distribution schema; all distribution parameters are unknown.


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
    >>> parameters = Distribution.DEFAULT.parse({})
    >>> statistic = WongWongExponentialityGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier."""
        return "WW"

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
        return self._statistic(rvs)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _statistic(rvs):
        numerator = np.max(rvs)
        denominator = np.min(rvs)
        if denominator == 0:
            if numerator == 0:
                raise ValueError("The ratio is undefined (0/0)")
            return float("inf")
        return float(numerator / denominator)


class HegazyGreen2ExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """HegazyGreen2 statistic for zero-origin exponential observations.

    Parameters
    ----------
    parameters : ParameterValues
        An empty distribution schema; all distribution parameters are unknown.


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
    Scientific reference for the exponential variants:
    N. G. Ushakov and V. G. Ushakov, "On sensitivity of exponentiality
    tests to data rounding: a Monte Carlo simulation study",
    https://doi.org/10.1080/03610918.2021.2009868.
    Historical reference: Hegazy and Green (1975), "Some new
    goodness-of-fit tests using order statistics",
    https://doi.org/10.2307/2347090.
    Scientific attribution is established. Historical publication priority
    is separate from verification of the formula and calibration.

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({})
    >>> statistic = HegazyGreen2ExponentialityGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier."""
        return "HG2"

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
        return self._statistic(rvs)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _statistic(rvs):
        rvs = rvs / np.mean(rvs)
        n = len(rvs)
        rvs.sort()
        b = -np.log(1 - np.arange(1, n + 1) / (n + 1))
        hg = (1.0 / n) * np.sum((rvs - b) ** 2)
        return float(hg)


class AbstractGraphExponentialityGofStatistic(
    AbstractExponentialityGofStatistic, AbstractGraphTestStatistic
):
    """Scale-invariant proximity graphs with threshold range/10."""

    @classmethod
    @override
    def code(cls) -> str:
        """Return the family identifier or the concrete statistic's full identifier."""
        family_code = f"GRAPH_EXPONENTIALITY_{AbstractGoodnessOfFitStatistic.code()}"
        if "short_code" in cls.__abstractmethods__:
            return family_code
        return f"{cls.short_code()}_{family_code}"

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

    Parameters
    ----------
    parameters : ParameterValues
        An empty distribution schema; all distribution parameters are unknown.


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
    >>> parameters = Distribution.DEFAULT.parse({})
    >>> statistic = GraphEdgesNumberExponentialityGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @override
    def alternative(self) -> Alternative:
        return TwoSidedAlternative()

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
        return self._statistic(rvs)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _statistic(rvs):
        n = len(rvs)
        distance = (np.max(rvs) - np.min(rvs)) / 10
        degrees = np.zeros(n, dtype=np.int64)
        for i in range(n):
            for j in range(i):
                if abs(rvs[i] - rvs[j]) < distance:
                    degrees[i] += 1
                    degrees[j] += 1
        return float(np.sum(degrees) / 2)


class GraphMaxDegreeExponentialityGofStatistic(
    AbstractGraphExponentialityGofStatistic, GraphMaxDegreeTestStatistic
):
    """GraphMaxDegree statistic for zero-origin exponential observations.

    Parameters
    ----------
    parameters : ParameterValues
        An empty distribution schema; all distribution parameters are unknown.


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
    >>> parameters = Distribution.DEFAULT.parse({})
    >>> statistic = GraphMaxDegreeExponentialityGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @override
    def alternative(self) -> Alternative:
        return TwoSidedAlternative()

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
        return self._statistic(rvs)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _statistic(rvs):
        n = len(rvs)
        distance = (np.max(rvs) - np.min(rvs)) / 10
        degrees = np.zeros(n, dtype=np.int64)
        for i in range(n):
            for j in range(i):
                if abs(rvs[i] - rvs[j]) < distance:
                    degrees[i] += 1
                    degrees[j] += 1
        return float(np.max(degrees))


class GraphAverageDegreeExponentialityGofStatistic(
    AbstractGraphExponentialityGofStatistic, GraphAverageDegreeTestStatistic
):
    """GraphAverageDegree statistic for zero-origin exponential observations.

    Parameters
    ----------
    parameters : ParameterValues
        An empty distribution schema; all distribution parameters are unknown.


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
    >>> parameters = Distribution.DEFAULT.parse({})
    >>> statistic = GraphAverageDegreeExponentialityGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @override
    def alternative(self) -> Alternative:
        return TwoSidedAlternative()

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
        return self._statistic(rvs)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _statistic(rvs):
        n = len(rvs)
        distance = (np.max(rvs) - np.min(rvs)) / 10
        degrees = np.zeros(n, dtype=np.int64)
        for i in range(n):
            for j in range(i):
                if abs(rvs[i] - rvs[j]) < distance:
                    degrees[i] += 1
                    degrees[j] += 1
        return float(np.mean(degrees))


class GraphConnectedComponentsExponentialityGofStatistic(
    AbstractGraphExponentialityGofStatistic, GraphConnectedComponentsTestStatistic
):
    """GraphConnectedComponents statistic for zero-origin exponential observations.

    Parameters
    ----------
    parameters : ParameterValues
        An empty distribution schema; all distribution parameters are unknown.


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
    >>> parameters = Distribution.DEFAULT.parse({})
    >>> statistic = GraphConnectedComponentsExponentialityGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @override
    def alternative(self) -> Alternative:
        return TwoSidedAlternative()

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
        return self._statistic(rvs)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _statistic(rvs):
        x = np.sort(rvs)
        distance = (x[-1] - x[0]) / 10
        return float(1 + np.sum(np.diff(x) >= distance))


class GraphCliqueNumberExponentialityGofStatistic(
    AbstractGraphExponentialityGofStatistic, GraphCliqueNumberTestStatistic
):
    """GraphCliqueNumber statistic for zero-origin exponential observations.

    Parameters
    ----------
    parameters : ParameterValues
        An empty distribution schema; all distribution parameters are unknown.


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
    >>> parameters = Distribution.DEFAULT.parse({})
    >>> statistic = GraphCliqueNumberExponentialityGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @override
    def alternative(self) -> Alternative:
        return TwoSidedAlternative()

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
        return self._statistic(rvs)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _statistic(rvs):
        x = np.sort(rvs)
        distance = (np.max(x) - np.min(x)) / 10
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

    Parameters
    ----------
    parameters : ParameterValues
        An empty distribution schema; all distribution parameters are unknown.


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
    >>> parameters = Distribution.DEFAULT.parse({})
    >>> statistic = GraphIndependenceNumberExponentialityGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
    >>> isinstance(value, float)
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @override
    def alternative(self) -> Alternative:
        return TwoSidedAlternative()

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
        return self._statistic(rvs)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _statistic(rvs):
        x = np.sort(rvs)
        distance = (np.max(x) - np.min(x)) / 10
        last = x[0]
        count = 1
        for value in x[1:]:
            if value - last >= distance:
                count += 1
                last = value
        return float(count)


class GreenwoodExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """G=sum(x_i²)/sum(x_i)²; both tails, unknown rate.

    Greenwood (1946), The Statistical Study of Infectious Diseases, HMSO
    (no DOI assigned). See also Moran (1947), DOI 10.2307/2983572, and
    Arendarczyk et al. (2023), DOI 10.57805/revstat.v21i3.357, (1.4)-(1.5).
    Normalized exponential observations have the Dirichlet
    law of uniform spacings; do not replace observations by ordered spacings.
    E(G)=2/(n+1), Var(G)=4(n-1)/((n+1)²(n+2)(n+3)). This moment contrast is
    not universally consistent. sqrt(n)*(n*G-2)/2 tends to N(0,1).
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @staticmethod
    def short_code():
        return "GREENWOOD"

    def alternative(self):
        return TwoSidedAlternative()

    def execute_statistic(self, rvs, **kwargs):
        _settings(kwargs)
        x = _sample(rvs)
        return self._statistic(x)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _statistic(x):
        p = x / x.sum()
        return float(p @ p)


class AbstractTunedExponentialStatistic(AbstractExponentialityGofStatistic, ABC):
    """A fixed, prespecified tuning parameter is part of the calibration."""

    def __init__(self, parameters: ParameterValues, *, a=1.0):
        super().__init__(parameters)
        self.a = _scalar(a, "a")
        if not 0 < self.a <= 100:
            raise ValueError("Supported numerical range is 0 < a <= 100")

    def alternative(self):
        return RightAlternative()

    def _validate_storage_calibration(self):
        if self.a != 1:
            raise ValueError("Stored calibration omits a; use fresh Monte Carlo")


class BaringhausHenzeELTExponentialityGofStatistic(AbstractTunedExponentialStatistic):
    """Baringhaus & Henze (1991), pp. 552-553, empirical Laplace ODE test.

    DOI 10.1007/BF00053372. T=n*integral(((1+t)L'_n(t)+L_n(t))² exp(-a*t),
    t=0..inf), L_n(t)=mean(exp(-t*x/mean(x))). Theory allows a>0; this
    implementation restricts a<=100 to limit cancellation in the pair kernel.
    Weighted chi-square series null, not normal or a single chi-square law.
    Right tail; consistent for alternatives with positive finite mean.
    O(n²) time and O(n) auxiliary memory. No state retained from fitting.
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @staticmethod
    def short_code():
        return "BH_ELT"

    def execute_statistic(self, rvs, **kwargs):
        _settings(kwargs)
        x = _sample(rvs)
        y = x / x.mean()
        total = np.longdouble(0)
        for v in y:
            kernel = self._kernel(v, y, self.a)
            total += np.sum(kernel, dtype=np.longdouble)
        result = float(total / len(y))
        if not np.isfinite(result):
            raise ValueError("Statistic outside representable numerical range")
        return max(0.0, result)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _kernel(v, y, a):
        inv = 1 / (v + y + a)
        return ((1 - v) * (1 - y) + (2 * v * y - v - y) * inv + 2 * v * y * inv**2) * inv


class BaringhausHenzeBHCExponentialityGofStatistic(AbstractTunedExponentialStatistic):
    """Generalized mean residual life statistic, a>0 (a=1 is BH CvM).

    Baringhaus & Henze (2008), DOI 10.1016/j.spl.2007.09.060; independent
    primary derivation: Ebner (2023), arXiv:2209.05153v2, (4), Remark 1.1.
    T=n*integral(mean((Y-t-1)*1{Y>t})²*exp(-a*t), t=0..inf).
    Piecewise polynomial integration avoids a cancelling n-by-n sum.
    No separate class for the identical a=1 statistic. Positive finite mean
    alternatives; right tail; Gaussian quadratic functional null.
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @staticmethod
    def short_code():
        return "BH_BHC"

    def execute_statistic(self, rvs, **kwargs):
        _settings(kwargs)
        x = _sample(rvs)
        y = np.sort(x / x.mean())
        left = np.r_[0.0, y[:-1]]
        width = y - left
        count = np.arange(len(y), 0, -1)
        tail = np.cumsum(y[::-1])[::-1]
        q = tail - count * (left + 1)
        z = -self.a * width
        # Integral_0^h u^k exp(-a*u) du = h^(k+1)/(k+1)*1F1(k+1;k+2;-a*h).
        m0 = width * scipy_special.hyp1f1(1, 2, z)
        m1 = width**2 / 2 * scipy_special.hyp1f1(2, 3, z)
        m2 = width**3 / 3 * scipy_special.hyp1f1(3, 4, z)
        return self._statistic(left, q, count, m0, m1, m2, self.a)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _statistic(left, q, count, m0, m1, m2, a):
        terms = np.exp(-a * left) * (q * q * m0 - 2 * q * count * m1 + count * count * m2)
        return float(np.sum(np.maximum(terms, 0)) / len(left))


class AbstractExponentialEDFStatistic(AbstractExponentialityGofStatistic, ABC):
    """Base for generated exponential EDF criteria with fixed or fitted rate."""


class AbstractExponentialFrequencyStatistic(AbstractExponentialityGofStatistic, ABC):
    """Exponential frequency criteria with configurable bins and per-sample fitting."""

    def _validate_input(self, sample):
        fixed = Distribution.RATE in self._parameters
        return _sample(sample, 1 if fixed else 2, rescale=False, positive_total=not fixed)

    def _validate_monte_carlo_calibration(self):
        if (
            Distribution.RATE not in self._parameters
            and type(self.binning) is not EqualProbabilityBins
        ):
            raise ValueError(
                "Fitted exponential with this binning requires external calibration; "
                "only EqualProbabilityBins supports standard exponential simulation"
            )


def _exponential_log_probabilities(criterion, sample):
    fixed = Distribution.RATE in criterion._parameters
    x = _sample(sample, 1 if fixed else 2, rescale=not fixed, positive_total=not fixed)
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        y = x * criterion.lam if fixed else x / x.mean()
        return np.log(-np.expm1(-y)), -y


def _exponential_bound_distribution(criterion, sample):
    if Distribution.RATE in criterion._parameters:
        return Distribution.bind(criterion._parameters)
    # Compute the mean in scaled units to avoid overflowing the sum.
    maximum = np.max(sample)
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        rate = (1 / np.mean(sample / maximum)) / maximum
        scale = 1 / rate
    if not np.isfinite(rate) or rate <= 0 or not np.isfinite(scale):
        raise ValueError("Fitted exponential rate or scale cannot be represented")
    return Distribution.bind(Distribution.DEFAULT.parse({Distribution.RATE: rate}))


_EDF_STATISTICS = {
    "KolmogorovSmirnovDPlusExponentialityGofStatistic": EDFSpec(
        edf.AbstractKSDPlusStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset({Distribution.RATE})),
        ),
    ),
    "KolmogorovSmirnovDMinusExponentialityGofStatistic": EDFSpec(
        edf.AbstractKSDMinusStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset({Distribution.RATE})),
        ),
    ),
    "KolmogorovSmirnovExponentialityGofStatistic": EDFSpec(
        edf.AbstractKSStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset({Distribution.RATE})),
        ),
    ),
    "LillieforsExponentialityGofStatistic": EDFSpec(
        edf.AbstractKSStatistic,
        short_code="LILLIE",
        supported_hypotheses=(HypothesisSupport(Distribution.DEFAULT, frozenset()),),
    ),
    "AndersonDarlingExponentialityGofStatistic": EDFSpec(
        edf.AbstractAndersonDarlingStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset()),
            HypothesisSupport(Distribution.DEFAULT, frozenset({Distribution.RATE})),
        ),
    ),
    "CramerVonMisesExponentialityGofStatistic": EDFSpec(
        edf.AbstractCramerVonMisesStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset({Distribution.RATE})),
            HypothesisSupport(Distribution.DEFAULT, frozenset()),
        ),
    ),
    "LiaoShimokawaExponentialityGofStatistic": EDFSpec(
        edf.AbstractLiaoShimokawaStatistic,
        short_code="LS",
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset()),
            HypothesisSupport(Distribution.DEFAULT, frozenset({Distribution.RATE})),
        ),
    ),
    "WatsonExponentialityGofStatistic": EDFSpec(
        edf.AbstractWatsonStatistic,
        short_code="W",
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset()),
            HypothesisSupport(Distribution.DEFAULT, frozenset({Distribution.RATE})),
        ),
    ),
    "KuiperExponentialityGofStatistic": EDFSpec(
        edf.AbstractKuiperStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset()),
            HypothesisSupport(Distribution.DEFAULT, frozenset({Distribution.RATE})),
        ),
    ),
    "KuiperScaledExponentialityGofStatistic": EDFSpec(
        edf.AbstractKuiperScaledStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset()),
            HypothesisSupport(Distribution.DEFAULT, frozenset({Distribution.RATE})),
        ),
    ),
    "KuiperCorrectedExponentialityGofStatistic": EDFSpec(
        edf.AbstractKuiperCorrectedStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset()),
            HypothesisSupport(Distribution.DEFAULT, frozenset({Distribution.RATE})),
        ),
    ),
    "ZhangKExponentialityGofStatistic": EDFSpec(
        edf.AbstractZhangKStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset()),
            HypothesisSupport(Distribution.DEFAULT, frozenset({Distribution.RATE})),
        ),
    ),
    "ZhangAExponentialityGofStatistic": EDFSpec(
        edf.AbstractZhangAStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset()),
            HypothesisSupport(Distribution.DEFAULT, frozenset({Distribution.RATE})),
        ),
    ),
    "ZhangCExponentialityGofStatistic": EDFSpec(
        edf.AbstractZhangCStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset()),
            HypothesisSupport(Distribution.DEFAULT, frozenset({Distribution.RATE})),
        ),
    ),
}

_FREQUENCY_STATISTICS = {
    "Chi2PearsonExponentialityGofStatistic": FrequencySpec(
        frequency.AbstractPearsonChi2Statistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset()),
            HypothesisSupport(Distribution.DEFAULT, frozenset({Distribution.RATE})),
        ),
    ),
    "FreemanTukeyExponentialityGofStatistic": FrequencySpec(
        frequency.AbstractFreemanTukeyStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset()),
            HypothesisSupport(Distribution.DEFAULT, frozenset({Distribution.RATE})),
        ),
    ),
    "ModifiedLogLikelihoodExponentialityGofStatistic": FrequencySpec(
        frequency.AbstractModifiedLogLikelihoodStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset()),
            HypothesisSupport(Distribution.DEFAULT, frozenset({Distribution.RATE})),
        ),
    ),
    "NeymanChi2ExponentialityGofStatistic": FrequencySpec(
        frequency.AbstractNeymanChi2Statistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset()),
            HypothesisSupport(Distribution.DEFAULT, frozenset({Distribution.RATE})),
        ),
    ),
    "CressieReadExponentialityGofStatistic": FrequencySpec(
        frequency.AbstractCressieReadStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset()),
            HypothesisSupport(Distribution.DEFAULT, frozenset({Distribution.RATE})),
        ),
    ),
}

__all__ = [
    "AbstractExponentialEDFStatistic",
    "AbstractExponentialFrequencyStatistic",
    "AbstractExponentialityGofStatistic",
    "AbstractGraphExponentialityGofStatistic",
    "AbstractTunedExponentialStatistic",
    "AhsanullahExponentialityGofStatistic",
    "AtkinsonExponentialityGofStatistic",
    "BaringhausHenzeBHCExponentialityGofStatistic",
    "BaringhausHenzeELTExponentialityGofStatistic",
    "CoxOakesExponentialityGofStatistic",
    "DeshpandeExponentialityGofStatistic",
    "EppsPulleyExponentialityGofStatistic",
    "EpsteinExponentialityGofStatistic",
    "FroziniExponentialityGofStatistic",
    "GiniExponentialityGofStatistic",
    "GnedenkoExponentialityGofStatistic",
    "GraphAverageDegreeExponentialityGofStatistic",
    "GraphCliqueNumberExponentialityGofStatistic",
    "GraphConnectedComponentsExponentialityGofStatistic",
    "GraphEdgesNumberExponentialityGofStatistic",
    "GraphIndependenceNumberExponentialityGofStatistic",
    "GraphMaxDegreeExponentialityGofStatistic",
    "GreenwoodExponentialityGofStatistic",
    "HarrisExponentialityGofStatistic",
    "HegazyGreen1ExponentialityGofStatistic",
    "HegazyGreen2ExponentialityGofStatistic",
    "HollanderProshanExponentialityGofStatistic",
    "KimberMichaelExponentialityGofStatistic",
    "KocharExponentialityGofStatistic",
    "LorenzExponentialityGofStatistic",
    "MoranExponentialityGofStatistic",
    "PietraExponentialityGofStatistic",
    "RossbergExponentialityGofStatistic",
    "ShapiroWilkExponentialityGofStatistic",
    "WeExponentialityGofStatistic",
    "WongWongExponentialityGofStatistic",
]
__all__.extend(
    export_edf_statistics(
        globals(),
        AbstractExponentialEDFStatistic,
        _EDF_STATISTICS,
        log_probabilities=_exponential_log_probabilities,
    )
)
__all__.extend(
    export_frequency_statistics(
        globals(),
        AbstractExponentialFrequencyStatistic,
        _FREQUENCY_STATISTICS,
        bind_distribution=_exponential_bound_distribution,
    )
)

for _name in (*_EDF_STATISTICS, *_FREQUENCY_STATISTICS):
    globals()[_name].__doc__ += """

    Exponential hypothesis
    ----------------------
    The origin is fixed at zero. Observations must be finite and nonnegative.
    Use Distribution.DEFAULT.parse({'lam': rate}) for a specified positive rate,
    or parse({}) for an unknown rate, where supported_hypotheses permits it.
    Unknown rate is fitted as 1/mean(sample) on every call; at least two
    observations and a positive total are required. No fitted state is retained.
    Monte Carlo calibration must repeat the fit for every simulated sample.
    """
    if _name in _FREQUENCY_STATISTICS:
        globals()[_name].__doc__ += """
    The binning keyword accepts a BinningStrategy; the default is
    EqualProbabilityBins(). Calibration must use the same binning strategy.
    For unknown rate, built-in Monte Carlo requires EqualProbabilityBins;
    other strategies require external calibration at an explicit rate.
    """
del _name
