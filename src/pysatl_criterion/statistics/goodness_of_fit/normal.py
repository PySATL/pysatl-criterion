"""Goodness-of-fit statistics for normal distributions.

Hypothesis parameters contain only values fixed under the null. Family
statistics leave the mean and variance free and report an empty dictionary.
Kolmogorov-Smirnov and Cramer-von Mises fix both parameters. The graph
statistics fix only the variance because their construction is invariant
to location but depends on scale.

For location-scale-invariant family statistics, Monte Carlo calibration
can use standard normal samples. That sampling choice does not fix the
parameters of the hypothesis. Constructor options controlling an algorithm,
such as Ryan-Joiner plotting positions, are not distribution parameters.
"""

import math
from abc import ABC

import numpy as np
import pandas as pd
import scipy.stats as scipy_stats
from numpy.polynomial import Polynomial
from numpy.polynomial.hermite_e import HermiteE
from scipy.special import gammaln, logsumexp, xlogy
from typing_extensions import override

from pysatl_criterion import DistributionType
from pysatl_criterion.statistics import AbstractGoodnessOfFitStatistic
from pysatl_criterion.statistics.alternative import (
    Alternative,
    AlternativeType,
    LeftAlternative,
    RightAlternative,
    TwoSidedAlternative,
)
from pysatl_criterion.statistics.goodness_of_fit.common import (
    ADStatistic,
    CrammerVonMisesStatistic,
    KSStatistic,
    LillieforsTest,
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


def _validate_normal_parameters(mean: float, var: float) -> None:
    for name, value in (("mean", mean), ("var", var)):
        message = f"{name} must be " + ("finite" if name == "mean" else "positive and finite")
        if (
            isinstance(value, (bool, np.bool_, str, bytes))
            or np.ndim(value) != 0
            or np.iscomplexobj(value)
        ):
            raise ValueError(message + " (real scalar)")
        try:
            valid = math.isfinite(value) and (name == "mean" or value > 0)
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise ValueError(message)


def _normal_sample(rvs, minimum=4, *, standardize=True):
    """Return a validated copy, removing location and scale only when invariant."""
    if np.iscomplexobj(rvs):
        raise ValueError("rvs must be real")
    try:
        x = np.array(rvs, dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("rvs must be a real numeric sample") from exc
    if x.ndim != 1 or x.size < minimum:
        raise ValueError(f"rvs must be one-dimensional with at least {minimum} observations")
    if not np.all(np.isfinite(x)):
        raise ValueError("rvs must contain only finite observations")
    if standardize:
        # Power-of-two rescaling avoids overflow before subtracting the centre.
        largest = np.max(np.abs(x))
        if largest == 0:
            raise ValueError("rvs must have nonzero dispersion")
        with np.errstate(under="ignore"):
            x = np.ldexp(x, -int(np.frexp(largest)[1]))
        x -= np.median(x)
        spread = np.max(np.abs(x))
        if spread == 0:
            raise ValueError("rvs must have nonzero dispersion")
        x /= spread
        x -= np.mean(x)
    return x


def _positive_scale(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite for this sample")
    return value


def _cabana_supremum(x, *, kurtosis):
    """Supremum of the l=5 truncated process, including limits at infinity."""
    z = (x - np.mean(x)) / np.std(x, ddof=1)
    h = [
        HermiteE.basis(j).convert(kind=Polynomial) / math.sqrt(math.factorial(j)) for j in range(9)
    ]
    means = [np.sum(poly(z)) / np.sqrt(len(z)) for poly in h]
    identity = Polynomial([0, 1])
    if kurtosis:
        poly = sum((np.sqrt(j / (j - 1)) * h[j - 2] + h[j]) * means[j + 3] for j in range(2, 6))
        poly = poly + means[3] + identity * means[4]
        cdf_coefficient = means[4]
    else:
        poly = sum(h[j - 1] * means[j + 3] / np.sqrt(j) for j in range(1, 6))
        cdf_coefficient = means[3]
    derivative = cdf_coefficient + identity * poly - poly.deriv()
    roots = derivative.roots()
    candidates = roots.real[np.abs(roots.imag) <= 1e-9 * (1 + np.abs(roots.real))]
    values = cdf_coefficient * scipy_stats.norm.cdf(candidates) - scipy_stats.norm.pdf(
        candidates
    ) * poly(candidates)
    return float(max(abs(cdf_coefficient), np.max(np.abs(values), initial=0)))


class AbstractNormalityGofStatistic(AbstractGoodnessOfFitStatistic, ABC):
    """Base class for goodness-of-fit statistics for the normal family.

    Notes
    -----
    The default null hypothesis allows any finite mean and positive variance.
    Its parameter dictionary is empty. Subclasses testing a specified mean or
    variance override ``hypothesis`` to declare those fixed parameters.
    """

    @override
    def __init__(self):
        """Initialize a statistic with no fixed distribution parameters."""

    def _validate_storage_calibration(self):
        """Reject algorithm settings absent from the current storage key."""
        if (
            getattr(self, "alternative_type", AlternativeType.TWO_TAILED)
            != AlternativeType.TWO_TAILED
            or getattr(self, "weighted", False)
            or getattr(self, "cte_alpha", "3/8") != "3/8"
        ):
            raise ValueError(
                "Stored calibration does not identify these settings; "
                "use MonteCarloLimitDistributionResolver"
            )

    @override
    def hypothesis(self) -> GoodnessOfFitHypothesis:
        """Return the composite null hypothesis of normality.

        Returns
        -------
        GoodnessOfFitHypothesis
            Normal family with unknown mean and variance; ``parameters()`` is
            an empty dictionary.
        """
        return GoodnessOfFitHypothesis({})

    @staticmethod
    @override
    def distribution() -> DistributionType:
        """
        Return the distribution family of the null hypothesis.

        Returns
        -------
        DistributionType
            ``DistributionType.NORMAL``.
        """
        return DistributionType.NORMAL

    @staticmethod
    @override
    def code():
        return f"NORMALITY_{AbstractGoodnessOfFitStatistic.code()}"


class KolmogorovSmirnovNormalityGofStatistic(AbstractNormalityGofStatistic, KSStatistic):
    """One-sample Kolmogorov-Smirnov statistic for a specified normal law.

    Parameters
    ----------
    alternative_type : AlternativeType, optional
        ``TWO_TAILED`` computes ``max(D_plus, D_minus)``; ``RIGHT`` computes
        ``D_plus = sup(F_n - F_0)`` and ``LEFT`` computes
        ``D_minus = sup(F_0 - F_n)``. Defaults to ``TWO_TAILED``.
    mode : str, optional
        Setting retained by the shared KS implementation. ``"auto"`` is
        stored as ``"exact"``. It does not affect the statistic and this
        class does not calculate a p-value.
    mean : float, optional
        Finite mean fixed by the null hypothesis. Default is 0.
    var : float, optional
        Positive, finite variance fixed by the null hypothesis. Default is 1.
        The normal CDF uses ``scale=sqrt(var)``.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic.
    hypothesis()
        Report fixed null parameters only.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is N(mean,var) with both parameters fixed; hypothesis() reports both.
    Constant samples are valid. Simulate the specified normal law.
    The upper tail of the statistic defines rejection. Use at least 1
    finite real observations in one dimension.
    Ties are accepted unless a scale or contrast becomes undefined.
    Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
    the standard normal CDF and density, s0=std(x,ddof=0), and
    s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

    D+ = max(i/n - F0(x_(i))) and D- = max(F0(x_(i)) - (i-1)/n).
    Return D+, D-, or max(D+, D-) according to alternative_type.
    F0 is the normal CDF with the fixed mean and variance. All three
    distances use the upper critical tail; mode does not compute a p-value.

    References
    ----------
    .. [1] Smirnov, N. (1948). Table for Estimating the Goodness of Fit of
       Empirical Distributions. The Annals of Mathematical Statistics,
       19(2), 279-281. https://doi.org/10.1214/aoms/1177730256

    Examples
    --------
    >>> statistic = KolmogorovSmirnovNormalityGofStatistic()
    >>> sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
    >>> value = statistic.execute_statistic(sample)
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def __init__(
        self,
        alternative_type: AlternativeType = AlternativeType.TWO_TAILED,
        mode="auto",
        mean: float = 0,
        var: float = 1,
    ):
        _validate_normal_parameters(mean, var)
        self.mean = mean
        self.var = var
        if not isinstance(alternative_type, AlternativeType):
            raise TypeError("alternative_type must be an AlternativeType")
        if mode not in ("auto", "exact", "approx", "asymp"):
            raise ValueError("Unsupported KS mode")
        KSStatistic.__init__(self, alternative_type, mode)

    @override
    def hypothesis(self) -> GoodnessOfFitHypothesis:
        """Return the normal null hypothesis with specified parameters.

        Returns
        -------
        GoodnessOfFitHypothesis
            Fixed ``mean`` and ``var``; neither is estimated from the sample.
        """
        return GoodnessOfFitHypothesis({"mean": self.mean, "var": self.var})

    @staticmethod
    @override
    def short_code():
        return "KS"

    @staticmethod
    @override
    def code():
        short_code = KolmogorovSmirnovNormalityGofStatistic.short_code()
        return f"{short_code}_{AbstractNormalityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute KolmogorovSmirnov for this sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real one-dimensional observations, n >= 1.
            Constant samples are accepted.
        **kwargs : dict, optional
            Unused keywords accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Scalar statistic in the normalization stated in the class notes.
            No p-value or test decision is computed.

        Raises
        ------
        ValueError
            If the sample is nonfinite, nonreal, multidimensional, too short,
            or a required scale or transformation is undefined.
        """
        rvs = _normal_sample(rvs, 1, standardize=False)
        rvs = np.sort(rvs)
        cdf_vals = scipy_stats.norm.cdf(rvs, loc=self.mean, scale=np.sqrt(self.var))
        return KSStatistic.do_execute_statistic(self, rvs, cdf_vals)


class AndersonDarlingNormalityGofStatistic(AbstractNormalityGofStatistic, ADStatistic):
    """Anderson-Darling statistic for the normal location-scale family.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic.
    hypothesis()
        Report fixed null parameters only.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is N(mu,sigma**2), with unknown real mu and sigma>0; hypothesis() reports {}.
    Location and positive scale cancel, allowing N(0,1) null simulation.
    Every simulated sample must go through execute_statistic again.
    The upper tail of the statistic defines rejection. Use at least 2
    finite real observations in one dimension. Nonzero dispersion is required.
    Ties are accepted unless a scale or contrast becomes undefined.
    Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
    the standard normal CDF and density, s0=std(x,ddof=0), and
    s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

    A2 = -n - sum((2*i-1)*(log(Phi(z_(i))) +
    log(1-Phi(z_(n+1-i)))))/n, where z=(x-mean(x))/s1.
    This is the unadjusted fitted-normal statistic. Log-CDF and log-survival
    functions avoid rounding probabilities to zero or one.

    References
    ----------
    .. [1] Anderson, T. W. and Darling, D. A. (1954). A Test of Goodness
       of Fit. Journal of the American Statistical Association, 49(268),
       765-769. https://doi.org/10.1080/01621459.1954.10501232
    .. [2] Stephens, M. A. (1976). Asymptotic Results for Goodness-of-Fit
       Statistics with Unknown Parameters. The Annals of Statistics,
       4(2), 357-369. https://doi.org/10.1214/aos/1176343411

    Examples
    --------
    >>> statistic = AndersonDarlingNormalityGofStatistic()
    >>> sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
    >>> value = statistic.execute_statistic(sample)
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def short_code():
        return "AD"

    @staticmethod
    @override
    def code():
        short_code = AndersonDarlingNormalityGofStatistic.short_code()
        return f"{short_code}_{AbstractNormalityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute AndersonDarling for this sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real one-dimensional observations, n >= 2.
            Nonzero dispersion and the scales specified in Notes are required.
        **kwargs : dict, optional
            Unused keywords accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Scalar statistic in the normalization stated in the class notes.
            No p-value or test decision is computed.

        Raises
        ------
        ValueError
            If the sample is nonfinite, nonreal, multidimensional, too short,
            or a required scale or transformation is undefined.
        """
        rvs = _normal_sample(rvs, 2)
        s = np.std(rvs, ddof=1, axis=0)
        y = np.sort(rvs)
        xbar = np.mean(rvs, axis=0)
        w = (y - xbar) / s
        logcdf = scipy_stats.distributions.norm.logcdf(w)
        logsf = scipy_stats.distributions.norm.logsf(w)
        return super().do_execute_statistic(rvs, log_cdf=logcdf, log_sf=logsf)


class ShapiroWilkNormalityGofStatistic(AbstractNormalityGofStatistic):
    """Shapiro-Wilk ``W`` statistic for the normal location-scale family.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic.
    hypothesis()
        Report fixed null parameters only.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is N(mu,sigma**2), with unknown real mu and sigma>0; hypothesis() reports {}.
    Location and positive scale cancel, allowing N(0,1) null simulation.
    Every simulated sample must go through execute_statistic again.
    The lower tail of the statistic defines rejection. Use at least 3
    finite real observations in one dimension. Nonzero dispersion is required.
    Ties are accepted unless a scale or contrast becomes undefined.
    Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
    the standard normal CDF and density, s0=std(x,ddof=0), and
    s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

    W = (sum(a_i*x_(i)))**2 / sum((x-mean(x))**2).
    The normalized symmetric weights use the Royston polynomial
    approximation, with the exact three-observation weights. This is a
    weight approximation to Shapiro-Wilk, not exact covariance inversion.

    References
    ----------
    .. [1] Shapiro, S. S. and Wilk, M. B. (1965). An analysis of variance
       test for normality (complete samples). Biometrika, 52(3-4), 591-611.
       https://doi.org/10.1093/biomet/52.3-4.591
    .. [2] Royston, P. (1995). Remark AS R94: A Remark on Algorithm AS 181:
       The W-test for Normality. Applied Statistics, 44(4), 547-551.
       https://doi.org/10.2307/2986146

    Examples
    --------
    >>> statistic = ShapiroWilkNormalityGofStatistic()
    >>> sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
    >>> value = statistic.execute_statistic(sample)
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return LeftAlternative()

    @staticmethod
    @override
    def short_code():
        return "SW"

    @staticmethod
    @override
    def code():
        short_code = ShapiroWilkNormalityGofStatistic.short_code()
        return f"{short_code}_{AbstractNormalityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute ShapiroWilk for this sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real one-dimensional observations, n >= 3.
            Nonzero dispersion and the scales specified in Notes are required.
        **kwargs : dict, optional
            Unused keywords accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Scalar statistic in the normalization stated in the class notes.
            No p-value or test decision is computed.

        Raises
        ------
        ValueError
            If the sample is nonfinite, nonreal, multidimensional, too short,
            or a required scale or transformation is undefined.
        """
        rvs = _normal_sample(rvs, 3)
        f_obs = np.asanyarray(rvs)
        f_obs_sorted = np.sort(f_obs)
        x_mean = np.mean(f_obs)

        denominator = (f_obs - x_mean) ** 2
        denominator = denominator.sum()

        a = self.ordered_statistic(len(f_obs))
        terms = a * f_obs_sorted
        return (terms.sum() ** 2) / denominator

    @staticmethod
    def ordered_statistic(n):
        if n == 3:
            sqrt = np.sqrt(0.5)
            return np.array([sqrt, 0, -sqrt])

        m = np.array([scipy_stats.norm.ppf((i - 3 / 8) / (n + 0.25)) for i in range(1, n + 1)])

        m2 = m**2
        term = np.sqrt(m2.sum())
        cn = m[-1] / term
        cn1 = m[-2] / term

        p1 = [-2.706056, 4.434685, -2.071190, -0.147981, 0.221157, cn]
        u = 1 / np.sqrt(n)

        wn = np.polyval(p1, u)
        # wn = np.array([p1[0] * (u ** 5), p1[1] * (u ** 4), p1[2] * (u ** 3), p1[3] * (u ** 2),
        # p1[4] * (u ** 1), p1[5]]).sum()
        w1 = -wn

        if n == 4 or n == 5:
            phi = (m2.sum() - 2 * m[-1] ** 2) / (1 - 2 * wn**2)
            phi_sqrt = np.sqrt(phi)
            result = np.array([m[k] / phi_sqrt for k in range(1, n - 1)])
            return np.concatenate([[w1], result, [wn]])

        p2 = [-3.582633, 5.682633, -1.752461, -0.293762, 0.042981, cn1]

        if n > 5:
            wn1 = np.polyval(p2, u)
            w2 = -wn1
            phi = (m2.sum() - 2 * m[-1] ** 2 - 2 * m[-2] ** 2) / (1 - 2 * wn**2 - 2 * wn1**2)
            phi_sqrt = np.sqrt(phi)
            result = np.array([m[k] / phi_sqrt for k in range(2, n - 2)])
            return np.concatenate([[w1, w2], result, [wn1, wn]])


class CramerVonMiseNormalityGofStatistic(AbstractNormalityGofStatistic, CrammerVonMisesStatistic):
    """Cramer-von Mises statistic for a specified normal distribution.

    Parameters
    ----------
    mean : float, optional
        Finite mean fixed by the null hypothesis. Default is 0.
    var : float, optional
        Positive, finite variance fixed by the null hypothesis. Default is 1.
        This is a variance, not the standard deviation.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic.
    hypothesis()
        Report fixed null parameters only.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is N(mean,var) with both parameters fixed; hypothesis() reports both.
    Constant samples are valid. Simulate the specified normal law.
    The upper tail of the statistic defines rejection. Use at least 1
    finite real observations in one dimension.
    Ties are accepted unless a scale or contrast becomes undefined.
    Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
    the standard normal CDF and density, s0=std(x,ddof=0), and
    s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

    W2 = 1/(12*n) + sum((F0(x_(i)) - (2*i-1)/(2*n))**2).
    F0 is the normal CDF with the fixed mean and variance.

    References
    ----------
    .. [1] Cramer, H. (1928). On the composition of elementary errors.
       Second paper: Statistical applications. Scandinavian Actuarial
       Journal, 1928(1), 141-180.
       https://doi.org/10.1080/03461238.1928.10416872

    Examples
    --------
    >>> statistic = CramerVonMiseNormalityGofStatistic()
    >>> sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
    >>> value = statistic.execute_statistic(sample)
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def __init__(self, mean: float = 0, var: float = 1):
        _validate_normal_parameters(mean, var)
        self.mean = mean
        self.var = var

    @override
    def hypothesis(self) -> GoodnessOfFitHypothesis:
        """Return the normal null hypothesis with specified parameters.

        Returns
        -------
        GoodnessOfFitHypothesis
            Fixed ``mean`` and ``var``; neither is estimated from the sample.
        """
        return GoodnessOfFitHypothesis({"mean": self.mean, "var": self.var})

    @staticmethod
    @override
    def short_code():
        return "CVM"

    @staticmethod
    @override
    def code():
        short_code = CramerVonMiseNormalityGofStatistic.short_code()
        base_code = super(AbstractNormalityGofStatistic, AbstractNormalityGofStatistic).code()
        return f"{short_code}_{base_code}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute CramerVonMise for this sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real one-dimensional observations, n >= 1.
            Constant samples are accepted.
        **kwargs : dict, optional
            Unused keywords accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Scalar statistic in the normalization stated in the class notes.
            No p-value or test decision is computed.

        Raises
        ------
        ValueError
            If the sample is nonfinite, nonreal, multidimensional, too short,
            or a required scale or transformation is undefined.
        """
        rvs = _normal_sample(rvs, 1, standardize=False)
        sorted_rvs = np.sort(np.asarray(rvs))
        cdf_vals = scipy_stats.norm.cdf(sorted_rvs, loc=self.mean, scale=np.sqrt(self.var))

        return CrammerVonMisesStatistic.do_execute_statistic(self, sorted_rvs, cdf_vals)


class LillieforsNormalityGofStatistic(AbstractNormalityGofStatistic, LillieforsTest):
    """Lilliefors statistic for normality with unknown mean and variance.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic.
    hypothesis()
        Report fixed null parameters only.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is N(mu,sigma**2), with unknown real mu and sigma>0; hypothesis() reports {}.
    Location and positive scale cancel, allowing N(0,1) null simulation.
    Every simulated sample must go through execute_statistic again.
    The upper tail of the statistic defines rejection. Use at least 2
    finite real observations in one dimension. Nonzero dispersion is required.
    Ties are accepted unless a scale or contrast becomes undefined.
    Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
    the standard normal CDF and density, s0=std(x,ddof=0), and
    s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

    D = max_i(i/n-Phi(z_(i)), Phi(z_(i))-(i-1)/n),
    where z=(x-mean(x))/s1. Both parameters are re-estimated on every call.
    Ordinary fixed-CDF KS tables do not apply.

    References
    ----------
    .. [1] Lilliefors, H. W. (1967). On the Kolmogorov-Smirnov Test for
       Normality with Mean and Variance Unknown. Journal of the American
       Statistical Association, 62(318), 399-402.
       https://doi.org/10.1080/01621459.1967.10482916

    Examples
    --------
    >>> statistic = LillieforsNormalityGofStatistic()
    >>> sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
    >>> value = statistic.execute_statistic(sample)
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def short_code():
        return "LILLIE"

    @staticmethod
    @override
    def code():
        short_code = LillieforsNormalityGofStatistic.short_code()
        return f"{short_code}_{AbstractNormalityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute Lilliefors for this sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real one-dimensional observations, n >= 2.
            Nonzero dispersion and the scales specified in Notes are required.
        **kwargs : dict, optional
            Unused keywords accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Scalar statistic in the normalization stated in the class notes.
            No p-value or test decision is computed.

        Raises
        ------
        ValueError
            If the sample is nonfinite, nonreal, multidimensional, too short,
            or a required scale or transformation is undefined.
        """
        rvs = _normal_sample(rvs, 2)
        x = np.asarray(rvs)
        z = (x - x.mean()) / x.std(ddof=1)
        cdf_vals = scipy_stats.norm.cdf(np.sort(z))
        return super(LillieforsTest, self).do_execute_statistic(rvs, cdf_vals)


class JBNormalityGofStatistic(AbstractNormalityGofStatistic):
    """Jarque-Bera statistic based on sample skewness and excess kurtosis.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic.
    hypothesis()
        Report fixed null parameters only.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is N(mu,sigma**2), with unknown real mu and sigma>0; hypothesis() reports {}.
    Location and positive scale cancel, allowing N(0,1) null simulation.
    Every simulated sample must go through execute_statistic again.
    The upper tail of the statistic defines rejection. Use at least 2
    finite real observations in one dimension. Nonzero dispersion is required.
    Ties are accepted unless a scale or contrast becomes undefined.
    Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
    the standard normal CDF and density, s0=std(x,ddof=0), and
    s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

    JB = n*(g1**2/6 + (g2-3)**2/24), where
    g1=m3/m2**1.5, g2=m4/m2**2, and mk=mean((x-mean(x))**k).

    References
    ----------
    .. [1] Jarque, C. M. and Bera, A. K. (1987). A Test for Normality of
       Observations and Regression Residuals. International Statistical
       Review, 55(2), 163-172. https://doi.org/10.2307/1403192

    Examples
    --------
    >>> statistic = JBNormalityGofStatistic()
    >>> sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
    >>> value = statistic.execute_statistic(sample)
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        return "JB"

    @staticmethod
    @override
    def code():
        short_code = JBNormalityGofStatistic.short_code()
        return f"{short_code}_{AbstractNormalityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute JB for this sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real one-dimensional observations, n >= 2.
            Nonzero dispersion and the scales specified in Notes are required.
        **kwargs : dict, optional
            Unused keywords accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Scalar statistic in the normalization stated in the class notes.
            No p-value or test decision is computed.

        Raises
        ------
        ValueError
            If the sample is nonfinite, nonreal, multidimensional, too short,
            or a required scale or transformation is undefined.
        """
        rvs = _normal_sample(rvs, 2)
        x = np.asarray(rvs)
        x = x.ravel()
        axis = 0

        n = x.shape[axis]
        if n == 0:
            raise ValueError("At least one observation is required.")

        mu = x.mean(axis=axis, keepdims=True)
        diffx = x - mu
        s = scipy_stats.skew(diffx, axis=axis)
        k = scipy_stats.kurtosis(diffx, axis=axis)
        statistic = n / 6 * (s**2 + k**2 / 4)
        return statistic


class SkewNormalityGofStatistic(AbstractNormalityGofStatistic):
    """D'Agostino transformed-skewness statistic for the normal family.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic.
    hypothesis()
        Report fixed null parameters only.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is N(mu,sigma**2), with unknown real mu and sigma>0; hypothesis() reports {}.
    Location and positive scale cancel, allowing N(0,1) null simulation.
    Every simulated sample must go through execute_statistic again.
    Both tails of the statistic defines rejection. Use at least 8
    finite real observations in one dimension. Nonzero dispersion is required.
    Ties are accepted unless a scale or contrast becomes undefined.
    Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
    the standard normal CDF and density, s0=std(x,ddof=0), and
    s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

    Return the signed D'Agostino transform delta*asinh(y/alpha) of
    g1=m3/m2**1.5, with the sample-size coefficients in skew_test.
    In particular, exactly zero sample skewness maps to zero. This tests
    skewness departures in either direction; it is not an omnibus test.

    References
    ----------
    .. [1] D'Agostino, R. B. (1970). Transformation to normality of the null
       distribution of g1. Biometrika, 57(3), 679-681.
       https://doi.org/10.1093/biomet/57.3.679

    Examples
    --------
    >>> statistic = SkewNormalityGofStatistic()
    >>> sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
    >>> value = statistic.execute_statistic(sample)
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return TwoSidedAlternative()

    @staticmethod
    @override
    def short_code():
        return "SKEW"

    @staticmethod
    @override
    def code():
        short_code = SkewNormalityGofStatistic.short_code()
        return f"{short_code}_{AbstractNormalityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute Skew for this sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real one-dimensional observations, n >= 8.
            Nonzero dispersion and the scales specified in Notes are required.
        **kwargs : dict, optional
            Unused keywords accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Scalar statistic in the normalization stated in the class notes.
            No p-value or test decision is computed.

        Raises
        ------
        ValueError
            If the sample is nonfinite, nonreal, multidimensional, too short,
            or a required scale or transformation is undefined.
        """
        rvs = _normal_sample(rvs, 8)
        x = np.asanyarray(rvs)
        y = np.sort(x)

        return self.skew_test(y)

    @staticmethod
    def skew_test(a):
        n = len(a)
        if n < 8:
            raise ValueError(
                f"skew test is not valid with less than 8 samples; {int(n)} samples were given."
            )
        b2 = scipy_stats.skew(a, axis=0)
        y = b2 * math.sqrt(((n + 1) * (n + 3)) / (6.0 * (n - 2)))
        beta2 = (
            3.0
            * (n**2 + 27 * n - 70)
            * (n + 1)
            * (n + 3)
            / ((n - 2.0) * (n + 5) * (n + 7) * (n + 9))
        )
        w2 = -1 + math.sqrt(2 * (beta2 - 1))
        delta = 1 / math.sqrt(0.5 * math.log(w2))
        alpha = math.sqrt(2.0 / (w2 - 1))
        z = delta * np.arcsinh(y / alpha)

        return z


class KurtosisNormalityGofStatistic(AbstractNormalityGofStatistic):
    """Anscombe-Glynn transformed-kurtosis statistic for the normal family.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic.
    hypothesis()
        Report fixed null parameters only.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is N(mu,sigma**2), with unknown real mu and sigma>0; hypothesis() reports {}.
    Location and positive scale cancel, allowing N(0,1) null simulation.
    Every simulated sample must go through execute_statistic again.
    Both tails of the statistic defines rejection. Use at least 5
    finite real observations in one dimension. Nonzero dispersion is required.
    Ties are accepted unless a scale or contrast becomes undefined.
    Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
    the standard normal CDF and density, s0=std(x,ddof=0), and
    s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

    Return the signed Anscombe-Glynn transform of g2=m4/m2**2,
    using the finite-n expectation, variance and skewness of g2. A real
    cube root is used. A zero transformation denominator is undefined.
    Both light and heavy tails matter; the normal approximation is
    particularly inaccurate for small samples.

    References
    ----------
    .. [1] Anscombe, F. J. and Glynn, W. J. (1983). Distribution of the
       kurtosis statistic b2 for normal samples. Biometrika, 70(1), 227-234.
       https://doi.org/10.1093/biomet/70.1.227

    Examples
    --------
    >>> statistic = KurtosisNormalityGofStatistic()
    >>> sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
    >>> value = statistic.execute_statistic(sample)
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return TwoSidedAlternative()

    @staticmethod
    @override
    def short_code():
        return "KURTOSIS"

    @staticmethod
    @override
    def code():
        short_code = KurtosisNormalityGofStatistic.short_code()
        return f"{short_code}_{AbstractNormalityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute Kurtosis for this sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real one-dimensional observations, n >= 5.
            Nonzero dispersion and the scales specified in Notes are required.
        **kwargs : dict, optional
            Unused keywords accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Scalar statistic in the normalization stated in the class notes.
            No p-value or test decision is computed.

        Raises
        ------
        ValueError
            If the sample is nonfinite, nonreal, multidimensional, too short,
            or a required scale or transformation is undefined.
        """
        rvs = _normal_sample(rvs, 5)
        x = np.asanyarray(rvs)
        y = np.sort(x)

        return self.kurtosis_test(y)

    @staticmethod
    def kurtosis_test(a):
        n = len(a)
        if n < 5:
            raise ValueError(
                f"kurtosistest requires at least 5 observations; {int(n)} observations were given."
            )
        # if n < 20:
        #    warnings.warn("kurtosistest only valid for n>=20 ... continuing "
        #                  "anyway, n=%i" % int(n),
        #                  stacklevel=2)
        b2 = scipy_stats.kurtosis(a, axis=0, fisher=False)

        e = 3.0 * (n - 1) / (n + 1)
        var_b2 = (
            24.0 * n * (n - 2) * (n - 3) / ((n + 1) * (n + 1.0) * (n + 3) * (n + 5))
        )  # [1]_ Eq. 1
        x = (b2 - e) / np.sqrt(var_b2)  # [1]_ Eq. 4
        # [1]_ Eq. 2:
        sqrt_beta1 = (
            6.0
            * (n * n - 5 * n + 2)
            / ((n + 7) * (n + 9))
            * np.sqrt((6.0 * (n + 3) * (n + 5)) / (n * (n - 2) * (n - 3)))
        )
        # [1]_ Eq. 3:
        a = 6.0 + 8.0 / sqrt_beta1 * (2.0 / sqrt_beta1 + np.sqrt(1 + 4.0 / (sqrt_beta1**2)))
        term1 = 1 - 2 / (9.0 * a)
        denom = 1 + x * np.sqrt(2 / (a - 4.0))
        if denom == 0:
            raise ValueError("Kurtosis transformation has a zero denominator")
        term2 = np.cbrt((1 - 2.0 / a) / denom)
        # if np.any(denom == 0):
        #    msg = ("Test statistic not defined in some cases due to division by "
        #           "zero. Return nan in that case...")
        #    warnings.warn(msg, RuntimeWarning, stacklevel=2)

        z = (term1 - term2) / np.sqrt(2 / (9.0 * a))  # [1]_ Eq. 5

        return z


class DAPNormalityGofStatistic(SkewNormalityGofStatistic, KurtosisNormalityGofStatistic):
    """D'Agostino-Pearson omnibus ``K_squared`` statistic for normality.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic.
    hypothesis()
        Report fixed null parameters only.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is N(mu,sigma**2), with unknown real mu and sigma>0; hypothesis() reports {}.
    Location and positive scale cancel, allowing N(0,1) null simulation.
    Every simulated sample must go through execute_statistic again.
    The upper tail of the statistic defines rejection. Use at least 8
    finite real observations in one dimension. Nonzero dispersion is required.
    Ties are accepted unless a scale or contrast becomes undefined.
    Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
    the standard normal CDF and density, s0=std(x,ddof=0), and
    s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

    K2 = Z_skew**2 + Z_kurtosis**2 using the signed transforms
    implemented by SkewNormalityGofStatistic and KurtosisNormalityGofStatistic.
    The resulting nonnegative discrepancy uses the upper tail.

    References
    ----------
    .. [1] D'Agostino, R. B. and Pearson, E. S. (1973). Tests for departure
       from normality. Empirical results for the distributions of b2 and
       sqrt(b1). Biometrika, 60(3), 613-622.
       https://doi.org/10.1093/biomet/60.3.613

    Examples
    --------
    >>> statistic = DAPNormalityGofStatistic()
    >>> sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
    >>> value = statistic.execute_statistic(sample)
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        return "DAP"

    @staticmethod
    @override
    def code():
        short_code = DAPNormalityGofStatistic.short_code()
        return f"{short_code}_{AbstractNormalityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute DAP for this sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real one-dimensional observations, n >= 8.
            Nonzero dispersion and the scales specified in Notes are required.
        **kwargs : dict, optional
            Unused keywords accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Scalar statistic in the normalization stated in the class notes.
            No p-value or test decision is computed.

        Raises
        ------
        ValueError
            If the sample is nonfinite, nonreal, multidimensional, too short,
            or a required scale or transformation is undefined.
        """
        rvs = _normal_sample(rvs, 8)
        x = np.asanyarray(rvs)
        y = np.sort(x)

        s = self.skew_test(y)
        k = self.kurtosis_test(y)
        k2 = s * s + k * k
        return k2


# https://github.com/puzzle-in-a-mug/normtest
class FilliNormalityGofStatistic(AbstractNormalityGofStatistic):
    """Filliben probability-plot correlation statistic for normality.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic.
    hypothesis()
        Report fixed null parameters only.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is N(mu,sigma**2), with unknown real mu and sigma>0; hypothesis() reports {}.
    Location and positive scale cancel, allowing N(0,1) null simulation.
    Every simulated sample must go through execute_statistic again.
    The lower tail of the statistic defines rejection. Use at least 3
    finite real observations in one dimension. Nonzero dispersion is required.
    Ties are accepted unless a scale or contrast becomes undefined.
    Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
    the standard normal CDF and density, s0=std(x,ddof=0), and
    s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

    Return corr(x_(i), Phi**-1(p_i)). The interior plotting positions
    are (i-0.3175)/(n+0.365); endpoints are 1-0.5**(1/n) and 0.5**(1/n).

    References
    ----------
    .. [1] Filliben, J. J. (1975). The Probability Plot Correlation
       Coefficient Test for Normality. Technometrics, 17(1), 111-117.
       https://doi.org/10.1080/00401706.1975.10489279

    Examples
    --------
    >>> statistic = FilliNormalityGofStatistic()
    >>> sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
    >>> value = statistic.execute_statistic(sample)
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return LeftAlternative()

    @staticmethod
    @override
    def short_code():
        return "FILLI"

    @staticmethod
    @override
    def code():
        short_code = FilliNormalityGofStatistic.short_code()
        return f"{short_code}_{AbstractNormalityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute Filli for this sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real one-dimensional observations, n >= 3.
            Nonzero dispersion and the scales specified in Notes are required.
        **kwargs : dict, optional
            Unused keywords accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Scalar statistic in the normalization stated in the class notes.
            No p-value or test decision is computed.

        Raises
        ------
        ValueError
            If the sample is nonfinite, nonreal, multidimensional, too short,
            or a required scale or transformation is undefined.
        """
        rvs = _normal_sample(rvs, 3)
        uniform_order = self._uniform_order_medians(len(rvs))
        zi = self._normal_order_medians(uniform_order)
        x_data = np.sort(rvs)
        statistic = self._statistic(x_data=x_data, zi=zi)
        return statistic

    @staticmethod
    def _uniform_order_medians(sample_size):
        i = np.arange(1, sample_size + 1)
        mi = (i - 0.3175) / (sample_size + 0.365)
        mi[0] = 1 - 0.5 ** (1 / sample_size)
        mi[-1] = 0.5 ** (1 / sample_size)

        return mi

    @staticmethod
    def _normal_order_medians(mi):
        normal_ordered = scipy_stats.norm.ppf(mi)
        return normal_ordered

    @staticmethod
    def _statistic(x_data, zi):
        correl = scipy_stats.pearsonr(x_data, zi)[0]
        return correl


# https://github.com/puzzle-in-a-mug/normtest
class LooneyGulledgeNormalityGofStatistic(AbstractNormalityGofStatistic):
    """Looney-Gulledge probability-plot correlation statistic for normality.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic.
    hypothesis()
        Report fixed null parameters only.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is N(mu,sigma**2), with unknown real mu and sigma>0; hypothesis() reports {}.
    Location and positive scale cancel, allowing N(0,1) null simulation.
    Every simulated sample must go through execute_statistic again.
    The lower tail of the statistic defines rejection. Use at least 3
    finite real observations in one dimension. Nonzero dispersion is required.
    Ties are accepted unless a scale or contrast becomes undefined.
    Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
    the standard normal CDF and density, s0=std(x,ddof=0), and
    s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

    Return corr(x_(i), Phi**-1((i-3/8)/(n+1/4))).
    The Blom plotting-position version is used without averaging ties.

    References
    ----------
    .. [1] Looney, S. W. and Gulledge, T. R., Jr. (1985). Use of the
       Correlation Coefficient with Normal Probability Plots. The American
       Statistician, 39(1), 75-79.
       https://doi.org/10.1080/00031305.1985.10479395

    Examples
    --------
    >>> statistic = LooneyGulledgeNormalityGofStatistic()
    >>> sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
    >>> value = statistic.execute_statistic(sample)
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return LeftAlternative()

    @staticmethod
    @override
    def short_code():
        return "LG"

    @staticmethod
    @override
    def code():
        short_code = LooneyGulledgeNormalityGofStatistic.short_code()
        return f"{short_code}_{AbstractNormalityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        # ordering
        """Compute LooneyGulledge for this sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real one-dimensional observations, n >= 3.
            Nonzero dispersion and the scales specified in Notes are required.
        **kwargs : dict, optional
            Unused keywords accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Scalar statistic in the normalization stated in the class notes.
            No p-value or test decision is computed.

        Raises
        ------
        ValueError
            If the sample is nonfinite, nonreal, multidimensional, too short,
            or a required scale or transformation is undefined.
        """
        rvs = _normal_sample(rvs, 3)
        x_data = np.sort(rvs)

        # zi
        zi = self._normal_order_statistic(
            x_data=x_data,
            weighted=False,  # TODO: False or True
        )

        # calculating the stats
        statistic = self._statistic(x_data=x_data, zi=zi)
        return statistic

    @staticmethod
    def _normal_order_statistic(x_data, weighted=False):
        # ordering
        x_data = np.sort(x_data)
        if weighted:
            df = pd.DataFrame({"x_data": x_data})
            # getting mi values
            df["Rank"] = np.arange(1, df.shape[0] + 1)
            df["Ui"] = LooneyGulledgeNormalityGofStatistic._order_statistic(
                sample_size=x_data.size,
            )
            df["Mi"] = df.groupby(["x_data"])["Ui"].transform("mean")
            normal_ordered = scipy_stats.norm.ppf(df["Mi"])
        else:
            ordered = LooneyGulledgeNormalityGofStatistic._order_statistic(
                sample_size=x_data.size,
            )
            normal_ordered = scipy_stats.norm.ppf(ordered)

        return normal_ordered

    @staticmethod
    def _statistic(x_data, zi):
        correl = scipy_stats.pearsonr(zi, x_data)[0]
        return correl

    @staticmethod
    def _order_statistic(sample_size):
        i = np.arange(1, sample_size + 1)
        cte_alpha = 3 / 8
        return (i - cte_alpha) / (sample_size - 2 * cte_alpha + 1)


# https://github.com/puzzle-in-a-mug/normtest
class RyanJoinerNormalityGofStatistic(AbstractNormalityGofStatistic):
    """Ryan-Joiner probability-plot correlation statistic for normality.

    Parameters
    ----------
    weighted : bool, optional
        If True, average the plotting probabilities within each group of
        tied observations before applying the normal quantile function.
        Default is False.
    cte_alpha : {'3/8', '1/2', '0'}, optional
        Plotting-position constant ``a`` in ``(i-a)/(n-2*a+1)`` for one-based
        ranks. Default is ``'3/8'``. Unrecognized values currently fall back
        to ``3/8``.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic.
    hypothesis()
        Report fixed null parameters only.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is N(mu,sigma**2), with unknown real mu and sigma>0; hypothesis() reports {}.
    Location and positive scale cancel, allowing N(0,1) null simulation.
    Every simulated sample must go through execute_statistic again.
    The lower tail of the statistic defines rejection. Use at least 3
    finite real observations in one dimension. Nonzero dispersion is required.
    Ties are accepted unless a scale or contrast becomes undefined.
    Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
    the standard normal CDF and density, s0=std(x,ddof=0), and
    s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

    Return corr(x_(i), Phi**-1(p_i)), p_i=(i-alpha)/(n+1-2*alpha).
    cte_alpha selects alpha=0, 3/8, or 1/2. If weighted=True, replace
    positions of tied observations by their group mean before taking
    normal quantiles. This tie convention and nondefault positions need
    separate calibration; they are not fixed distribution parameters.

    References
    ----------
    .. [1] Ryan, T. A., Jr. and Joiner, B. L. (1976). Normal Probability
       Plots and Tests for Normality. Technical report, Statistics
       Department, The Pennsylvania State University. Original report:
       https://www.additive-net.de/en/component/jdownloads/send/70-support/236-normal-probability-plots-and-tests-for-normality-thomas-a-ryan-jr-bryan-l-joiner

    Examples
    --------
    >>> statistic = RyanJoinerNormalityGofStatistic()
    >>> sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
    >>> value = statistic.execute_statistic(sample)
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def __init__(self, weighted=False, cte_alpha="3/8"):
        super().__init__()
        if not isinstance(weighted, (bool, np.bool_)):
            raise TypeError("weighted must be boolean")
        if cte_alpha not in ("0", "3/8", "1/2"):
            raise ValueError("cte_alpha must be '0', '3/8', or '1/2'")
        self.weighted = weighted
        self.cte_alpha = cte_alpha

    @override
    def alternative(self) -> Alternative:
        return LeftAlternative()

    @staticmethod
    @override
    def short_code():
        return "RJ"

    @staticmethod
    @override
    def code():
        short_code = RyanJoinerNormalityGofStatistic.short_code()
        return f"{short_code}_{AbstractNormalityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        # ordering
        """Compute RyanJoiner for this sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real one-dimensional observations, n >= 3.
            Nonzero dispersion and the scales specified in Notes are required.
        **kwargs : dict, optional
            Unused keywords accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Scalar statistic in the normalization stated in the class notes.
            No p-value or test decision is computed.

        Raises
        ------
        ValueError
            If the sample is nonfinite, nonreal, multidimensional, too short,
            or a required scale or transformation is undefined.
        """
        rvs = _normal_sample(rvs, 3)
        x_data = np.sort(rvs)

        # zi
        zi = self._normal_order_statistic(
            x_data=x_data,
            weighted=self.weighted,
            cte_alpha=self.cte_alpha,
        )

        # calculating the stats
        statistic = self._statistic(x_data=x_data, zi=zi)
        return statistic

    def _normal_order_statistic(self, x_data, weighted=False, cte_alpha="3/8"):
        # ordering
        x_data = np.sort(x_data)
        if weighted:
            df = pd.DataFrame({"x_data": x_data})
            # getting mi values
            df["Rank"] = np.arange(1, df.shape[0] + 1)
            df["Ui"] = self._order_statistic(
                sample_size=x_data.size,
                cte_alpha=cte_alpha,
            )
            df["Mi"] = df.groupby(["x_data"])["Ui"].transform("mean")
            normal_ordered = scipy_stats.norm.ppf(df["Mi"])
        else:
            ordered = self._order_statistic(
                sample_size=x_data.size,
                cte_alpha=cte_alpha,
            )
            normal_ordered = scipy_stats.norm.ppf(ordered)

        return normal_ordered

    @staticmethod
    def _statistic(x_data, zi):
        return scipy_stats.pearsonr(zi, x_data)[0]

    @staticmethod
    def _order_statistic(sample_size, cte_alpha="3/8"):
        i = np.arange(1, sample_size + 1)
        if cte_alpha == "1/2":
            cte_alpha = 0.5
        elif cte_alpha == "0":
            cte_alpha = 0
        else:
            cte_alpha = 3 / 8

        return (i - cte_alpha) / (sample_size - 2 * cte_alpha + 1)


class SFNormalityGofStatistic(AbstractNormalityGofStatistic):
    """Shapiro-Francia ``W_prime`` statistic for the normal family.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic.
    hypothesis()
        Report fixed null parameters only.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is N(mu,sigma**2), with unknown real mu and sigma>0; hypothesis() reports {}.
    Location and positive scale cancel, allowing N(0,1) null simulation.
    Every simulated sample must go through execute_statistic again.
    The lower tail of the statistic defines rejection. Use at least 3
    finite real observations in one dimension. Nonzero dispersion is required.
    Ties are accepted unless a scale or contrast becomes undefined.
    Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
    the standard normal CDF and density, s0=std(x,ddof=0), and
    s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

    Return corr(x_(i), Phi**-1((i-3/8)/(n+1/4)))**2.
    This Shapiro-Francia approximation uses Blom scores in place of
    exact expected normal order statistics.

    References
    ----------
    .. [1] Shapiro, S. S. and Francia, R. S. (1972). An Approximate
       Analysis of Variance Test for Normality. Journal of the American
       Statistical Association, 67(337), 215-216.
       https://doi.org/10.1080/01621459.1972.10481232
    .. [2] Weisberg, S. and Bingham, C. (1975). An Approximate Analysis of
       Variance Test for Non-Normality Suitable for Machine Calculation.
       Technometrics, 17(1), 133-134.
       https://doi.org/10.1080/00401706.1975.10489283

    Examples
    --------
    >>> statistic = SFNormalityGofStatistic()
    >>> sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
    >>> value = statistic.execute_statistic(sample)
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return LeftAlternative()

    @staticmethod
    @override
    def short_code():
        return "SF"

    @staticmethod
    @override
    def code():
        short_code = SFNormalityGofStatistic.short_code()
        return f"{short_code}_{AbstractNormalityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute SF for this sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real one-dimensional observations, n >= 3.
            Nonzero dispersion and the scales specified in Notes are required.
        **kwargs : dict, optional
            Unused keywords accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Scalar statistic in the normalization stated in the class notes.
            No p-value or test decision is computed.

        Raises
        ------
        ValueError
            If the sample is nonfinite, nonreal, multidimensional, too short,
            or a required scale or transformation is undefined.
        """
        rvs = _normal_sample(rvs, 3)
        n = len(rvs)
        rvs = np.sort(rvs)

        x_mean = np.mean(rvs)
        alpha = 0.375
        terms = (np.arange(1, n + 1) - alpha) / (n - 2 * alpha + 1)
        e = -scipy_stats.norm.ppf(terms)

        w = np.sum(e * rvs) ** 2 / (np.sum((rvs - x_mean) ** 2) * np.sum(e**2))
        return w


# https://habr.com/ru/articles/685582/
class EppsPulleyNormalityGofStatistic(AbstractNormalityGofStatistic):
    """Epps-Pulley empirical-characteristic-function statistic for normality.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic.
    hypothesis()
        Report fixed null parameters only.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is N(mu,sigma**2), with unknown real mu and sigma>0; hypothesis() reports {}.
    Location and positive scale cancel, allowing N(0,1) null simulation.
    Every simulated sample must go through execute_statistic again.
    The upper tail of the statistic defines rejection. Use at least 2
    finite real observations in one dimension. Nonzero dispersion is required.
    Ties are accepted unless a scale or contrast becomes undefined.
    Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
    the standard normal CDF and density, s0=std(x,ddof=0), and
    s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

    For z=(x-mean(x))/s0, return
    n/sqrt(3) + sum_ij(exp(-(z_i-z_j)**2/2))/n
    - sqrt(2)*sum_i(exp(-z_i**2/4)).
    This equals n times a Gaussian-weighted squared characteristic-function
    distance. The double sum includes its diagonal; computation is O(n**2).

    References
    ----------
    .. [1] Epps, T. W. and Pulley, L. B. (1983). A test for normality based
       on the empirical characteristic function. Biometrika, 70(3), 723-726.
       https://doi.org/10.1093/biomet/70.3.723

    Examples
    --------
    >>> statistic = EppsPulleyNormalityGofStatistic()
    >>> sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
    >>> value = statistic.execute_statistic(sample)
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        return "EP"

    @staticmethod
    @override
    def code():
        short_code = EppsPulleyNormalityGofStatistic.short_code()
        return f"{short_code}_{AbstractNormalityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute EppsPulley for this sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real one-dimensional observations, n >= 2.
            Nonzero dispersion and the scales specified in Notes are required.
        **kwargs : dict, optional
            Unused keywords accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Scalar statistic in the normalization stated in the class notes.
            No p-value or test decision is computed.

        Raises
        ------
        ValueError
            If the sample is nonfinite, nonreal, multidimensional, too short,
            or a required scale or transformation is undefined.
        """
        rvs = _normal_sample(rvs, 2)
        n = len(rvs)
        x = np.sort(rvs)
        x_mean = np.mean(x)
        m2 = np.var(x, ddof=0)

        a = np.sqrt(2) * np.sum([np.exp(-((x[i] - x_mean) ** 2) / (4 * m2)) for i in range(n)])
        b = 0
        for k in range(1, n):
            b = b + np.sum(np.exp(-((x[:k] - x[k]) ** 2) / (2 * m2)))
        b = 2 / n * b
        t = 1 + n / np.sqrt(3) + b - a
        return t


class Hosking2NormalityGofStatistic(AbstractNormalityGofStatistic):
    """Hosking normality statistic with symmetric trimming level 1.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic.
    hypothesis()
        Report fixed null parameters only.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is N(mu,sigma**2), with unknown real mu and sigma>0; hypothesis() reports {}.
    Location and positive scale cancel, allowing N(0,1) null simulation.
    Every simulated sample must go through execute_statistic again.
    The upper tail of the statistic defines rejection. Use at least 6
    finite real observations in one dimension. Nonzero dispersion is required.
    Ties are accepted unless a scale or contrast becomes undefined.
    Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
    the standard normal CDF and density, s0=std(x,ddof=0), and
    s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

    Local quadratic discrepancy of L-moment ratios with trim t=1.
    For r=2,3,4, L_r=sum_i x_(i)*sum_{k=0..r-1}((-1)**k*
    C(r-1,k)*C(i-1,r+t-1-k)*C(n-i,t+k))/(r*C(n,r+2*t)).
    Return (L3/L2)**2/v3+(L4/L2-mu4)**2/v4. The constants
    (mu4,v3,v4) are the three historical rows in the implementation
    for n<=25, 25<n<=50 and n>50. They are not a consistent
    asymptotic covariance sequence. A primary source for this exact
    omnibus formula and these rows was not located; the L-moment
    papers alone do not validate this test. Treat it as a local
    statistic and simulate its null law at the actual n. L2 must be positive.

    Examples
    --------
    >>> statistic = Hosking2NormalityGofStatistic()
    >>> sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
    >>> value = statistic.execute_statistic(sample)
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        return "HOSKING2"

    @staticmethod
    @override
    def code():
        short_code = Hosking2NormalityGofStatistic.short_code()
        return f"{short_code}_{AbstractNormalityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute Hosking2 for this sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real one-dimensional observations, n >= 6.
            Nonzero dispersion and the scales specified in Notes are required.
        **kwargs : dict, optional
            Unused keywords accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Scalar statistic in the normalization stated in the class notes.
            No p-value or test decision is computed.

        Raises
        ------
        ValueError
            If the sample is nonfinite, nonreal, multidimensional, too short,
            or a required scale or transformation is undefined.
        """
        rvs = _normal_sample(rvs, 6)
        n = len(rvs)

        if n > 3:
            x_tmp = [0] * n
            l21, l31, l41 = 0.0, 0.0, 0.0
            mu_tau41, v_tau31, v_tau41 = 0.0, 0.0, 0.0
            for i in range(n):
                x_tmp[i] = rvs[i]
            x_tmp = np.sort(x_tmp)
            for i in range(2, n):
                l21 += x_tmp[i - 1] * self.pstarmod1(2, n, i)
                l31 += x_tmp[i - 1] * self.pstarmod1(3, n, i)
                l41 += x_tmp[i - 1] * self.pstarmod1(4, n, i)
            l21 = l21 / (2.0 * math.comb(n, 4))
            l31 = l31 / (3.0 * math.comb(n, 5))
            l41 = l41 / (4.0 * math.comb(n, 6))
            tau31 = l31 / _positive_scale(l21, "trimmed L-scale")
            tau41 = l41 / _positive_scale(l21, "trimmed L-scale")
            if 1 <= n <= 25:
                mu_tau41 = 0.067077
                v_tau31 = 0.0081391
                v_tau41 = 0.0042752
            if 25 < n <= 50:
                mu_tau41 = 0.064456
                v_tau31 = 0.0034657
                v_tau41 = 0.0015699
            if 50 < n:
                mu_tau41 = 0.063424
                v_tau31 = 0.0016064
                v_tau41 = 0.00068100
            return pow(tau31, 2.0) / v_tau31 + pow(tau41 - mu_tau41, 2.0) / v_tau41

        return 0

    @staticmethod
    def pstarmod1(r, n, i):
        res = 0.0
        for k in range(r):
            res = res + (-1.0) ** k * math.comb(r - 1, k) * math.comb(
                i - 1, r + 1 - 1 - k
            ) * math.comb(n - i, 1 + k)

        return res


class Hosking1NormalityGofStatistic(AbstractNormalityGofStatistic):
    """Hosking L-moment normality statistic without trimming.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic.
    hypothesis()
        Report fixed null parameters only.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is N(mu,sigma**2), with unknown real mu and sigma>0; hypothesis() reports {}.
    Location and positive scale cancel, allowing N(0,1) null simulation.
    Every simulated sample must go through execute_statistic again.
    The upper tail of the statistic defines rejection. Use at least 4
    finite real observations in one dimension. Nonzero dispersion is required.
    Ties are accepted unless a scale or contrast becomes undefined.
    Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
    the standard normal CDF and density, s0=std(x,ddof=0), and
    s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

    Local quadratic discrepancy of L-moment ratios with trim t=0.
    For r=2,3,4, L_r=sum_i x_(i)*sum_{k=0..r-1}((-1)**k*
    C(r-1,k)*C(i-1,r+t-1-k)*C(n-i,t+k))/(r*C(n,r+2*t)).
    Return (L3/L2)**2/v3+(L4/L2-mu4)**2/v4. The constants
    (mu4,v3,v4) are the three historical rows in the implementation
    for n<=25, 25<n<=50 and n>50. They are not a consistent
    asymptotic covariance sequence. A primary source for this exact
    omnibus formula and these rows was not located; the L-moment
    papers alone do not validate this test. Treat it as a local
    statistic and simulate its null law at the actual n. L2 must be positive.

    Examples
    --------
    >>> statistic = Hosking1NormalityGofStatistic()
    >>> sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
    >>> value = statistic.execute_statistic(sample)
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        return "HOSKING1"

    @staticmethod
    @override
    def code():
        short_code = Hosking1NormalityGofStatistic.short_code()
        return f"{short_code}_{AbstractNormalityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute Hosking1 for this sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real one-dimensional observations, n >= 4.
            Nonzero dispersion and the scales specified in Notes are required.
        **kwargs : dict, optional
            Unused keywords accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Scalar statistic in the normalization stated in the class notes.
            No p-value or test decision is computed.

        Raises
        ------
        ValueError
            If the sample is nonfinite, nonreal, multidimensional, too short,
            or a required scale or transformation is undefined.
        """
        rvs = _normal_sample(rvs, 4)
        return self.stat10(rvs)

    @staticmethod
    def stat10(x):
        n = len(x)

        if n > 3:
            x_tmp = x[:n].copy()
            x_tmp.sort()
            tmp1 = n * (n - 1)
            tmp2 = tmp1 * (n - 2)
            tmp3 = tmp2 * (n - 3)
            b0 = sum(x_tmp[:3]) + sum(x_tmp[3:])
            b1 = 1.0 * x_tmp[1] + 2.0 * x_tmp[2] + sum(i * x_tmp[i] for i in range(3, n))
            b2 = 2.0 * x_tmp[2] + sum((i * (i - 1)) * x_tmp[i] for i in range(3, n))
            b3 = sum((i * (i - 1) * (i - 2)) * x_tmp[i] for i in range(3, n))
            b0 /= n
            b1 /= tmp1
            b2 /= tmp2
            b3 /= tmp3
            l2 = 2.0 * b1 - b0
            l3 = 6.0 * b2 - 6.0 * b1 + b0
            l4 = 20.0 * b3 - 30.0 * b2 + 12.0 * b1 - b0
            tau3 = l3 / _positive_scale(l2, "trimmed L-scale")
            tau4 = l4 / _positive_scale(l2, "trimmed L-scale")

            if 1 <= n <= 25:
                mu_tau4 = 0.12383
                v_tau3 = 0.0088038
                v_tau4 = 0.0049295
            elif 25 < n <= 50:
                mu_tau4 = 0.12321
                v_tau3 = 0.0040493
                v_tau4 = 0.0020802
            else:
                mu_tau4 = 0.12291
                v_tau3 = 0.0019434
                v_tau4 = 0.00095785

            stat_tl_mom = (tau3**2) / v_tau3 + (tau4 - mu_tau4) ** 2 / v_tau4
            return stat_tl_mom


class Hosking3NormalityGofStatistic(AbstractNormalityGofStatistic):
    """Hosking normality statistic with symmetric trimming level 2.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic.
    hypothesis()
        Report fixed null parameters only.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is N(mu,sigma**2), with unknown real mu and sigma>0; hypothesis() reports {}.
    Location and positive scale cancel, allowing N(0,1) null simulation.
    Every simulated sample must go through execute_statistic again.
    The upper tail of the statistic defines rejection. Use at least 8
    finite real observations in one dimension. Nonzero dispersion is required.
    Ties are accepted unless a scale or contrast becomes undefined.
    Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
    the standard normal CDF and density, s0=std(x,ddof=0), and
    s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

    Local quadratic discrepancy of L-moment ratios with trim t=2.
    For r=2,3,4, L_r=sum_i x_(i)*sum_{k=0..r-1}((-1)**k*
    C(r-1,k)*C(i-1,r+t-1-k)*C(n-i,t+k))/(r*C(n,r+2*t)).
    Return (L3/L2)**2/v3+(L4/L2-mu4)**2/v4. The constants
    (mu4,v3,v4) are the three historical rows in the implementation
    for n<=25, 25<n<=50 and n>50. They are not a consistent
    asymptotic covariance sequence. A primary source for this exact
    omnibus formula and these rows was not located; the L-moment
    papers alone do not validate this test. Treat it as a local
    statistic and simulate its null law at the actual n. L2 must be positive.

    Examples
    --------
    >>> statistic = Hosking3NormalityGofStatistic()
    >>> sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
    >>> value = statistic.execute_statistic(sample)
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        return "HOSKING3"

    @staticmethod
    @override
    def code():
        short_code = Hosking3NormalityGofStatistic.short_code()
        return f"{short_code}_{AbstractNormalityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute Hosking3 for this sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real one-dimensional observations, n >= 8.
            Nonzero dispersion and the scales specified in Notes are required.
        **kwargs : dict, optional
            Unused keywords accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Scalar statistic in the normalization stated in the class notes.
            No p-value or test decision is computed.

        Raises
        ------
        ValueError
            If the sample is nonfinite, nonreal, multidimensional, too short,
            or a required scale or transformation is undefined.
        """
        rvs = _normal_sample(rvs, 8)
        return self.stat12(rvs)

    def stat12(self, x):
        n = len(x)

        if n > 3:
            x_tmp = x[:n].copy()
            x_tmp.sort()
            l22 = 0.0
            l32 = 0.0
            l42 = 0.0
            for i in range(2, n):
                l22 += x_tmp[i - 1] * self.pstarmod2(2, n, i)
                l32 += x_tmp[i - 1] * self.pstarmod2(3, n, i)
                l42 += x_tmp[i - 1] * self.pstarmod2(4, n, i)
            l22 /= 2.0 * math.comb(n, 6)
            l32 /= 3.0 * math.comb(n, 7)
            l42 /= 4.0 * math.comb(n, 8)
            tau32 = l32 / _positive_scale(l22, "trimmed L-scale")
            tau42 = l42 / _positive_scale(l22, "trimmed L-scale")

            if 1 <= n <= 25:
                mu_tau42 = 0.044174
                v_tau32 = 0.0086570
                v_tau42 = 0.0042066
            elif 25 < n <= 50:
                mu_tau42 = 0.040389
                v_tau32 = 0.0033818
                v_tau42 = 0.0013301
            else:
                mu_tau42 = 0.039030
                v_tau32 = 0.0015120
                v_tau42 = 0.00054207

            stat_tl_mom2 = (tau32**2) / v_tau32 + (tau42 - mu_tau42) ** 2 / v_tau42
            return stat_tl_mom2

    @staticmethod
    def pstarmod2(r, n, i):
        res = 0.0
        for k in range(r):
            res += (
                (-1) ** k
                * math.comb(r - 1, k)
                * math.comb(i - 1, r + 2 - 1 - k)
                * math.comb(n - i, 2 + k)
            )
        return res


class Hosking4NormalityGofStatistic(AbstractNormalityGofStatistic):
    """Hosking normality statistic with symmetric trimming level 3.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic.
    hypothesis()
        Report fixed null parameters only.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is N(mu,sigma**2), with unknown real mu and sigma>0; hypothesis() reports {}.
    Location and positive scale cancel, allowing N(0,1) null simulation.
    Every simulated sample must go through execute_statistic again.
    The upper tail of the statistic defines rejection. Use at least 10
    finite real observations in one dimension. Nonzero dispersion is required.
    Ties are accepted unless a scale or contrast becomes undefined.
    Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
    the standard normal CDF and density, s0=std(x,ddof=0), and
    s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

    Local quadratic discrepancy of L-moment ratios with trim t=3.
    For r=2,3,4, L_r=sum_i x_(i)*sum_{k=0..r-1}((-1)**k*
    C(r-1,k)*C(i-1,r+t-1-k)*C(n-i,t+k))/(r*C(n,r+2*t)).
    Return (L3/L2)**2/v3+(L4/L2-mu4)**2/v4. The constants
    (mu4,v3,v4) are the three historical rows in the implementation
    for n<=25, 25<n<=50 and n>50. They are not a consistent
    asymptotic covariance sequence. A primary source for this exact
    omnibus formula and these rows was not located; the L-moment
    papers alone do not validate this test. Treat it as a local
    statistic and simulate its null law at the actual n. L2 must be positive.

    Examples
    --------
    >>> statistic = Hosking4NormalityGofStatistic()
    >>> sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
    >>> value = statistic.execute_statistic(sample)
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        return "HOSKING4"

    @staticmethod
    @override
    def code():
        short_code = Hosking4NormalityGofStatistic.short_code()
        return f"{short_code}_{AbstractNormalityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute Hosking4 for this sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real one-dimensional observations, n >= 10.
            Nonzero dispersion and the scales specified in Notes are required.
        **kwargs : dict, optional
            Unused keywords accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Scalar statistic in the normalization stated in the class notes.
            No p-value or test decision is computed.

        Raises
        ------
        ValueError
            If the sample is nonfinite, nonreal, multidimensional, too short,
            or a required scale or transformation is undefined.
        """
        rvs = _normal_sample(rvs, 10)
        return self.stat13(rvs)

    def stat13(self, x):
        n = len(x)

        if n > 3:
            x_tmp = x[:n].copy()
            x_tmp.sort()
            l23 = 0.0
            l33 = 0.0
            l43 = 0.0
            for i in range(2, n):
                l23 += x_tmp[i - 1] * self.pstarmod3(2, n, i)
                l33 += x_tmp[i - 1] * self.pstarmod3(3, n, i)
                l43 += x_tmp[i - 1] * self.pstarmod3(4, n, i)
            l23 /= 2.0 * math.comb(n, 8)
            l33 /= 3.0 * math.comb(n, 9)
            l43 /= 4.0 * math.comb(n, 10)
            tau33 = l33 / _positive_scale(l23, "trimmed L-scale")
            tau43 = l43 / _positive_scale(l23, "trimmed L-scale")

            if 1 <= n <= 25:
                mu_tau43 = 0.033180
                v_tau33 = 0.0095765
                v_tau43 = 0.0044609
            elif 25 < n <= 50:
                mu_tau43 = 0.028224
                v_tau33 = 0.0033813
                v_tau43 = 0.0011823
            else:
                mu_tau43 = 0.026645
                v_tau33 = 0.0014547
                v_tau43 = 0.00045107

            stat_tl_mom3 = (tau33**2) / v_tau33 + (tau43 - mu_tau43) ** 2 / v_tau43
            return stat_tl_mom3

    @staticmethod
    def pstarmod3(r, n, i):
        res = 0.0
        for k in range(r):
            res += (
                (-1) ** k
                * math.comb(r - 1, k)
                * math.comb(i - 1, r + 3 - 1 - k)
                * math.comb(n - i, 3 + k)
            )
        return res


class ZhangWuCNormalityGofStatistic(AbstractNormalityGofStatistic):
    """Zhang-Wu likelihood-ratio normality statistic ``Z_C``.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic.
    hypothesis()
        Report fixed null parameters only.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is N(mu,sigma**2), with unknown real mu and sigma>0; hypothesis() reports {}.
    Location and positive scale cancel, allowing N(0,1) null simulation.
    Every simulated sample must go through execute_statistic again.
    The upper tail of the statistic defines rejection. Use at least 4
    finite real observations in one dimension. Nonzero dispersion is required.
    Ties are accepted unless a scale or contrast becomes undefined.
    Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
    the standard normal CDF and density, s0=std(x,ddof=0), and
    s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

    For z=(x-mean(x))/s1, return sum_i((log((1-Phi(z_(i)))/
    Phi(z_(i))) - log((n-i+1/4)/(i-3/4)))**2).
    Normal log-CDF/log-survival evaluations retain extreme tail information.

    References
    ----------
    .. [1] Zhang, J. and Wu, Y. (2005).
       Likelihood-ratio tests for normality.
       Computational Statistics & Data Analysis, 49(3), 709-721.
       https://doi.org/10.1016/j.csda.2004.05.034

    Examples
    --------
    >>> statistic = ZhangWuCNormalityGofStatistic()
    >>> sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
    >>> value = statistic.execute_statistic(sample)
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        return "ZWC"

    @staticmethod
    @override
    def code():
        short_code = ZhangWuCNormalityGofStatistic.short_code()
        return f"{short_code}_{AbstractNormalityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute ZhangWuC for this sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real one-dimensional observations, n >= 4.
            Nonzero dispersion and the scales specified in Notes are required.
        **kwargs : dict, optional
            Unused keywords accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Scalar statistic in the normalization stated in the class notes.
            No p-value or test decision is computed.

        Raises
        ------
        ValueError
            If the sample is nonfinite, nonreal, multidimensional, too short,
            or a required scale or transformation is undefined.
        """
        rvs = _normal_sample(rvs, 4)
        z = np.sort((rvs - np.mean(rvs)) / np.std(rvs, ddof=1))
        i = np.arange(1, len(z) + 1)
        log_odds = scipy_stats.norm.logsf(z) - scipy_stats.norm.logcdf(z)
        expected = np.log((len(z) - i + 0.25) / (i - 0.75))
        return float(np.sum((log_odds - expected) ** 2))


class ZhangWuANormalityGofStatistic(AbstractNormalityGofStatistic):
    """Zhang-Wu likelihood-ratio normality statistic ``Z_A``.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic.
    hypothesis()
        Report fixed null parameters only.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is N(mu,sigma**2), with unknown real mu and sigma>0; hypothesis() reports {}.
    Location and positive scale cancel, allowing N(0,1) null simulation.
    Every simulated sample must go through execute_statistic again.
    The upper tail of the statistic defines rejection. Use at least 4
    finite real observations in one dimension. Nonzero dispersion is required.
    Ties are accepted unless a scale or contrast becomes undefined.
    Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
    the standard normal CDF and density, s0=std(x,ddof=0), and
    s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

    For z=(x-mean(x))/s1, let A=-sum_i(log(Phi(z_(i)))/
    (n-i+1/2) + log(1-Phi(z_(i)))/(i-1/2)). Return 10*A-32.
    The historical affine scaling is retained and must also be used in
    calibration. Logs are evaluated without first rounding normal CDFs.

    References
    ----------
    .. [1] Zhang, J. and Wu, Y. (2005).
       Likelihood-ratio tests for normality.
       Computational Statistics & Data Analysis, 49(3), 709-721.
       https://doi.org/10.1016/j.csda.2004.05.034

    Examples
    --------
    >>> statistic = ZhangWuANormalityGofStatistic()
    >>> sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
    >>> value = statistic.execute_statistic(sample)
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        return "ZWA"

    @staticmethod
    @override
    def code():
        short_code = ZhangWuANormalityGofStatistic.short_code()
        return f"{short_code}_{AbstractNormalityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute ZhangWuA for this sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real one-dimensional observations, n >= 4.
            Nonzero dispersion and the scales specified in Notes are required.
        **kwargs : dict, optional
            Unused keywords accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Scalar statistic in the normalization stated in the class notes.
            No p-value or test decision is computed.

        Raises
        ------
        ValueError
            If the sample is nonfinite, nonreal, multidimensional, too short,
            or a required scale or transformation is undefined.
        """
        rvs = _normal_sample(rvs, 4)
        z = np.sort((rvs - np.mean(rvs)) / np.std(rvs, ddof=1))
        i = np.arange(1, len(z) + 1)
        total = np.sum(
            scipy_stats.norm.logcdf(z) / (len(z) - i + 0.5) + scipy_stats.norm.logsf(z) / (i - 0.5)
        )
        return float(-10 * total - 32)


class GlenLeemisBarrNormalityGofStatistic(AbstractNormalityGofStatistic):
    """Glen-Leemis-Barr order-statistic goodness-of-fit statistic.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic.
    hypothesis()
        Report fixed null parameters only.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is N(mu,sigma**2), with unknown real mu and sigma>0; hypothesis() reports {}.
    Location and positive scale cancel, allowing N(0,1) null simulation.
    Every simulated sample must go through execute_statistic again.
    The upper tail of the statistic defines rejection. Use at least 4
    finite real observations in one dimension. Nonzero dispersion is required.
    Ties are accepted unless a scale or contrast becomes undefined.
    Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
    the standard normal CDF and density, s0=std(x,ddof=0), and
    s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

    For z=(x-mean(x))/s1, form u_i=Phi(z_(i)), then
    v_i=BetaCDF(u_i; i,n+1-i), and sort the v_i again. Return
    -n-sum_i((2*n+1-2*i)*log(v_(i))+(2*i-1)*log(1-v_(i)))/n.
    These are reversed Anderson-Darling weights, not the ordinary AD
    statistic. A primary source confirming this exact fitted formula was
    not located in the review; no author-specific calibration is claimed.
    The beta transforms are dependent and are not uniform after fitting.
    Use simulation of this formula only. Log-binomial sums recover
    beta tails that underflow in direct probability arithmetic.

    Examples
    --------
    >>> statistic = GlenLeemisBarrNormalityGofStatistic()
    >>> sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
    >>> value = statistic.execute_statistic(sample)
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        return "GLB"

    @staticmethod
    @override
    def code():
        short_code = GlenLeemisBarrNormalityGofStatistic.short_code()
        return f"{short_code}_{AbstractNormalityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute GlenLeemisBarr for this sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real one-dimensional observations, n >= 4.
            Nonzero dispersion and the scales specified in Notes are required.
        **kwargs : dict, optional
            Unused keywords accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Scalar statistic in the normalization stated in the class notes.
            No p-value or test decision is computed.

        Raises
        ------
        ValueError
            If the sample is nonfinite, nonreal, multidimensional, too short,
            or a required scale or transformation is undefined.
        """
        rvs = _normal_sample(rvs, 4)
        n = len(rvs)
        z = np.sort((rvs - np.mean(rvs)) / np.std(rvs, ddof=1))
        i = np.arange(1, n + 1)
        # BetaCDF(Phi(z);i,n+1-i) is a binomial upper tail. Use the
        # complementary normal probability for the right half of the sample.
        p = scipy_stats.norm.cdf(-np.abs(z))
        low = scipy_stats.beta.logcdf(p, i, n + 1 - i)
        high = scipy_stats.beta.logsf(p, i, n + 1 - i)
        positive = z > 0
        low[positive] = scipy_stats.beta.logsf(p[positive], n + 1 - i[positive], i[positive])
        high[positive] = scipy_stats.beta.logcdf(p[positive], n + 1 - i[positive], i[positive])
        for j in np.flatnonzero(~np.isfinite(low) | ~np.isfinite(high)):
            k = np.arange(n + 1)
            terms = (
                gammaln(n + 1)
                - gammaln(k + 1)
                - gammaln(n - k + 1)
                + k * scipy_stats.norm.logcdf(z[j])
                + (n - k) * scipy_stats.norm.logsf(z[j])
            )
            low[j] = logsumexp(terms[j + 1 :])
            high[j] = logsumexp(terms[: j + 1])
        # Sorting by decreasing survival log also distinguishes CDFs rounded
        # to one. The pair (low, -high) supplies a stable lexicographic order.
        order = np.lexsort((-high, low))
        return float(-n - np.sum((2 * n + 1 - 2 * i) * low[order] + (2 * i - 1) * high[order]) / n)


class DoornikHansenNormalityGofStatistic(AbstractNormalityGofStatistic):
    """Univariate Doornik-Hansen normality statistic.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic.
    hypothesis()
        Report fixed null parameters only.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is N(mu,sigma**2), with unknown real mu and sigma>0; hypothesis() reports {}.
    Location and positive scale cancel, allowing N(0,1) null simulation.
    Every simulated sample must go through execute_statistic again.
    The upper tail of the statistic defines rejection. Use at least 8
    finite real observations in one dimension. Nonzero dispersion is required.
    Ties are accepted unless a scale or contrast becomes undefined.
    Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
    the standard normal CDF and density, s0=std(x,ddof=0), and
    s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

    Return Z1**2+Z2**2, with the Doornik-Hansen transformations of
    g1=m3/m2**1.5 and g2=m4/m2**2 (Appendix A of the author preprint).
    Z1 uses asinh; Z2 uses the real cube root and includes the g1**2
    adjustment to kurtosis. The chi-square law is asymptotic.

    References
    ----------
    .. [1] Doornik, J. A. and Hansen, H. (2008).
       An Omnibus Test for Univariate and Multivariate Normality.
       Oxford Bulletin of Economics and Statistics, 70(s1), 927-939.
       https://doi.org/10.1111/j.1468-0084.2008.00537.x

    Examples
    --------
    >>> statistic = DoornikHansenNormalityGofStatistic()
    >>> sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
    >>> value = statistic.execute_statistic(sample)
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        return "DH"

    @staticmethod
    @override
    def code():
        short_code = DoornikHansenNormalityGofStatistic.short_code()
        return f"{short_code}_{AbstractNormalityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute DoornikHansen for this sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real one-dimensional observations, n >= 8.
            Nonzero dispersion and the scales specified in Notes are required.
        **kwargs : dict, optional
            Unused keywords accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Scalar statistic in the normalization stated in the class notes.
            No p-value or test decision is computed.

        Raises
        ------
        ValueError
            If the sample is nonfinite, nonreal, multidimensional, too short,
            or a required scale or transformation is undefined.
        """
        rvs = _normal_sample(rvs, 8)
        return self.doornik_hansen(rvs)

    def doornik_hansen(self, x):
        n = len(x)
        m2 = scipy_stats.moment(x, moment=2)
        m3 = scipy_stats.moment(x, moment=3)
        m4 = scipy_stats.moment(x, moment=4)

        b1 = m3 / (m2**1.5)
        b2 = m4 / (m2**2)

        z1 = self.skewness_to_z1(b1, n)
        z2 = self.kurtosis_to_z2(b1, b2, n)

        stat = z1**2 + z2**2
        return stat

    @staticmethod
    def skewness_to_z1(skew, n):
        b = 3 * ((n**2) + 27 * n - 70) * (n + 1) * (n + 3) / ((n - 2) * (n + 5) * (n + 7) * (n + 9))
        w2 = -1 + math.sqrt(2 * (b - 1))
        d = 1 / math.sqrt(math.log(math.sqrt(w2)))
        y = skew * math.sqrt((n + 1) * (n + 3) / (6 * (n - 2)))
        a = math.sqrt(2 / (w2 - 1))
        z = d * math.log((y / a) + math.sqrt((y / a) ** 2 + 1))
        return z

    @staticmethod
    def kurtosis_to_z2(skew, kurt, n):
        n2 = n**2
        n3 = n**3
        p1 = n2 + 15 * n - 4
        p2 = n2 + 27 * n - 70
        p3 = n2 + 2 * n - 5
        p4 = n3 + 37 * n2 + 11 * n - 313
        d = (n - 3) * (n + 1) * p1
        a = (n - 2) * (n + 5) * (n + 7) * p2 / (6 * d)
        c = (n - 7) * (n + 5) * (n + 7) * p3 / (6 * d)
        k = (n + 5) * (n + 7) * p4 / (12 * d)
        alpha = a + skew**2 * c
        q = 2 * (kurt - 1 - skew**2) * k
        z = np.cbrt(0.5 * q / alpha) - 1 + 1 / (9 * alpha)
        z *= math.sqrt(9 * alpha)
        return z


class RobustJarqueBeraNormalityGofStatistic(AbstractNormalityGofStatistic):
    """Gel-Gastwirth robust Jarque-Bera normality statistic.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic.
    hypothesis()
        Report fixed null parameters only.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is N(mu,sigma**2), with unknown real mu and sigma>0; hypothesis() reports {}.
    Location and positive scale cancel, allowing N(0,1) null simulation.
    Every simulated sample must go through execute_statistic again.
    The upper tail of the statistic defines rejection. Use at least 4
    finite real observations in one dimension. Nonzero dispersion is required.
    Ties are accepted unless a scale or contrast becomes undefined.
    Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
    the standard normal CDF and density, s0=std(x,ddof=0), and
    s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

    Let J=sqrt(pi/2)*mean(abs(x-median(x))). Return
    n*(m3/J**3)**2/6 + n*(m4/J**4-3)**2/64.
    Moments are centered at the sample mean, whereas J uses the median.
    The reference identifies this robust-scale modification; small-sample
    chi-square calibration is not justified.

    References
    ----------
    .. [1] Gel, Y. R. and Gastwirth, J. L. (2008).
       A robust modification of the Jarque-Bera test of normality.
       Economics Letters, 99(1), 30-32.
       https://doi.org/10.1016/j.econlet.2007.05.022

    Examples
    --------
    >>> statistic = RobustJarqueBeraNormalityGofStatistic()
    >>> sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
    >>> value = statistic.execute_statistic(sample)
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        return "RJB"

    @staticmethod
    @override
    def code():
        short_code = RobustJarqueBeraNormalityGofStatistic.short_code()
        return f"{short_code}_{AbstractNormalityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute RobustJarqueBera for this sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real one-dimensional observations, n >= 4.
            Nonzero dispersion and the scales specified in Notes are required.
        **kwargs : dict, optional
            Unused keywords accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Scalar statistic in the normalization stated in the class notes.
            No p-value or test decision is computed.

        Raises
        ------
        ValueError
            If the sample is nonfinite, nonreal, multidimensional, too short,
            or a required scale or transformation is undefined.
        """
        rvs = _normal_sample(rvs, 4)
        y = np.sort(rvs)
        n = len(rvs)
        m = np.median(y)
        c = np.sqrt(math.pi / 2)
        j = (c / n) * np.sum(np.abs(rvs - m))
        m_3 = scipy_stats.moment(y, moment=3)
        m_4 = scipy_stats.moment(y, moment=4)
        rjb = (n / 6) * (m_3 / j**3) ** 2 + (n / 64) * (m_4 / j**4 - 3) ** 2
        return rjb


class BontempsMeddahi1NormalityGofStatistic(AbstractNormalityGofStatistic):
    """Bontemps-Meddahi normality statistic using Hermite orders 3 and 4.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic.
    hypothesis()
        Report fixed null parameters only.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is N(mu,sigma**2), with unknown real mu and sigma>0; hypothesis() reports {}.
    Location and positive scale cancel, allowing N(0,1) null simulation.
    Every simulated sample must go through execute_statistic again.
    The upper tail of the statistic defines rejection. Use at least 4
    finite real observations in one dimension. Nonzero dispersion is required.
    Ties are accepted unless a scale or contrast becomes undefined.
    Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
    the standard normal CDF and density, s0=std(x,ddof=0), and
    s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

    Let z=(x-mean(x))/s1 and h_j=He_j/sqrt(j!), where He_j
    are probabilists' Hermite polynomials. Return
    (sum(h_3(z))**2+sum(h_4(z))**2)/n.
    The sample-variance convention ddof=1 is part of this implementation.

    References
    ----------
    .. [1] Bontemps, C. and Meddahi, N. (2005).
       Testing normality: a GMM approach.
       Journal of Econometrics, 124(1), 149-186.
       https://doi.org/10.1016/j.jeconom.2004.02.014

    Examples
    --------
    >>> statistic = BontempsMeddahi1NormalityGofStatistic()
    >>> sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
    >>> value = statistic.execute_statistic(sample)
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        return "BM1"

    @staticmethod
    @override
    def code():
        short_code = BontempsMeddahi1NormalityGofStatistic.short_code()
        return f"{short_code}_{AbstractNormalityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute BontempsMeddahi1 for this sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real one-dimensional observations, n >= 4.
            Nonzero dispersion and the scales specified in Notes are required.
        **kwargs : dict, optional
            Unused keywords accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Scalar statistic in the normalization stated in the class notes.
            No p-value or test decision is computed.

        Raises
        ------
        ValueError
            If the sample is nonfinite, nonreal, multidimensional, too short,
            or a required scale or transformation is undefined.
        """
        rvs = _normal_sample(rvs, 4)
        n = len(rvs)

        if n > 3:
            z = [0.0] * n
            var_x = 0.0
            mean_x = 0.0
            tmp3 = 0.0
            tmp4 = 0.0

            for i in range(n):
                mean_x += rvs[i]
            mean_x /= n

            for i in range(n):
                var_x += rvs[i] ** 2
            var_x = np.var(rvs, ddof=1)
            sd_x = math.sqrt(var_x)

            for i in range(n):
                z[i] = (rvs[i] - mean_x) / sd_x

            for i in range(n):
                tmp3 += (z[i] ** 3 - 3 * z[i]) / math.sqrt(6)
                tmp4 += (z[i] ** 4 - 6 * z[i] ** 2 + 3) / (2 * math.sqrt(6))

            stat_bm34 = (tmp3**2 + tmp4**2) / n
            return stat_bm34


class BontempsMeddahi2NormalityGofStatistic(AbstractNormalityGofStatistic):
    """Bontemps-Meddahi normality statistic using Hermite orders 3 through 6.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic.
    hypothesis()
        Report fixed null parameters only.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is N(mu,sigma**2), with unknown real mu and sigma>0; hypothesis() reports {}.
    Location and positive scale cancel, allowing N(0,1) null simulation.
    Every simulated sample must go through execute_statistic again.
    The upper tail of the statistic defines rejection. Use at least 4
    finite real observations in one dimension. Nonzero dispersion is required.
    Ties are accepted unless a scale or contrast becomes undefined.
    Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
    the standard normal CDF and density, s0=std(x,ddof=0), and
    s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

    Let z=(x-mean(x))/s1 and h_j=He_j/sqrt(j!), where He_j
    are probabilists' Hermite polynomials. Return
    sum_j(sum_i(h_j(z_i))**2)/n for j=3,4,5,6.
    The sample-variance convention ddof=1 is part of this implementation.

    References
    ----------
    .. [1] Bontemps, C. and Meddahi, N. (2005).
       Testing normality: a GMM approach.
       Journal of Econometrics, 124(1), 149-186.
       https://doi.org/10.1016/j.jeconom.2004.02.014

    Examples
    --------
    >>> statistic = BontempsMeddahi2NormalityGofStatistic()
    >>> sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
    >>> value = statistic.execute_statistic(sample)
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        return "BM2"

    @staticmethod
    @override
    def code():
        short_code = BontempsMeddahi2NormalityGofStatistic.short_code()
        return f"{short_code}_{AbstractNormalityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute BontempsMeddahi2 for this sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real one-dimensional observations, n >= 4.
            Nonzero dispersion and the scales specified in Notes are required.
        **kwargs : dict, optional
            Unused keywords accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Scalar statistic in the normalization stated in the class notes.
            No p-value or test decision is computed.

        Raises
        ------
        ValueError
            If the sample is nonfinite, nonreal, multidimensional, too short,
            or a required scale or transformation is undefined.
        """
        rvs = _normal_sample(rvs, 4)
        return self.stat15(rvs)

    @staticmethod
    def stat15(x):
        n = len(x)

        if n > 3:
            z = np.zeros(n)
            mean_x = np.mean(x)
            var_x = np.var(x, ddof=1)
            sd_x = np.sqrt(var_x)
            for i in range(n):
                z[i] = (x[i] - mean_x) / sd_x
            tmp3 = np.sum((z**3 - 3 * z) / np.sqrt(6))
            tmp4 = np.sum((z**4 - 6 * z**2 + 3) / (2 * np.sqrt(6)))
            tmp5 = np.sum((z**5 - 10 * z**3 + 15 * z) / (2 * np.sqrt(30)))
            tmp6 = np.sum((z**6 - 15 * z**4 + 45 * z**2 - 15) / (12 * np.sqrt(5)))
            stat_bm36 = (tmp3**2 + tmp4**2 + tmp5**2 + tmp6**2) / n
            return stat_bm36


class BonettSeierNormalityGofStatistic(AbstractNormalityGofStatistic):
    """Bonett-Seier statistic based on a modified Geary kurtosis measure.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic.
    hypothesis()
        Report fixed null parameters only.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is N(mu,sigma**2), with unknown real mu and sigma>0; hypothesis() reports {}.
    Location and positive scale cancel, allowing N(0,1) null simulation.
    Every simulated sample must go through execute_statistic again.
    Both tails of the statistic defines rejection. Use at least 4
    finite real observations in one dimension. Nonzero dispersion is required.
    Ties are accepted unless a scale or contrast becomes undefined.
    Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
    the standard normal CDF and density, s0=std(x,ddof=0), and
    s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

    The publication was identified, but the exact finite-sample
    implementation is not fully certified against its primary text.

    Let d=mean(abs(x-mean(x))) and omega=13.29*log(s0/d).
    Return sqrt(n+2)*(omega-3)/3.54, a signed kurtosis transform.
    The constants are rounded approximations; use finite-sample simulation.

    References
    ----------
    .. [1] Bonett, D. G. and Seier, E. (2002).
       A test of normality with high uniform power.
       Computational Statistics & Data Analysis, 40(3), 435-445.
       https://doi.org/10.1016/S0167-9473(02)00074-9

    Examples
    --------
    >>> statistic = BonettSeierNormalityGofStatistic()
    >>> sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
    >>> value = statistic.execute_statistic(sample)
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return TwoSidedAlternative()

    @staticmethod
    @override
    def short_code():
        return "BS"

    @staticmethod
    @override
    def code():
        short_code = BonettSeierNormalityGofStatistic.short_code()
        return f"{short_code}_{AbstractNormalityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute BonettSeier for this sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real one-dimensional observations, n >= 4.
            Nonzero dispersion and the scales specified in Notes are required.
        **kwargs : dict, optional
            Unused keywords accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Scalar statistic in the normalization stated in the class notes.
            No p-value or test decision is computed.

        Raises
        ------
        ValueError
            If the sample is nonfinite, nonreal, multidimensional, too short,
            or a required scale or transformation is undefined.
        """
        rvs = _normal_sample(rvs, 4)
        return self.stat17(rvs)

    @staticmethod
    def stat17(x):
        n = len(x)

        if n > 3:
            m2 = 0.0
            mean_x = 0.0
            term = 0.0

            for i in range(n):
                mean_x += x[i]

            mean_x = mean_x / float(n)

            for i in range(n):
                m2 += (x[i] - mean_x) ** 2
                term += abs(x[i] - mean_x)

            m2 = m2 / float(n)
            term = term / float(n)
            omega = 13.29 * (math.log(math.sqrt(m2)) - math.log(term))
            stat_tw = math.sqrt(float(n + 2)) * (omega - 3.0) / 3.54
            return stat_tw


class MartinezIglewiczNormalityGofStatistic(AbstractNormalityGofStatistic):
    """Martinez-Iglewicz statistic comparing two estimates of dispersion.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic.
    hypothesis()
        Report fixed null parameters only.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is N(mu,sigma**2), with unknown real mu and sigma>0; hypothesis() reports {}.
    Location and positive scale cancel, allowing N(0,1) null simulation.
    Every simulated sample must go through execute_statistic again.
    The upper tail of the statistic defines rejection. Use at least 4
    finite real observations in one dimension. Nonzero dispersion is required.
    Ties are accepted unless a scale or contrast becomes undefined.
    Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
    the standard normal CDF and density, s0=std(x,ddof=0), and
    s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

    Let M=median(x), A=median(abs(x-M)), u=(x-M)/(9*A),
    and I={i: abs(u_i)<1}. Define B=n*sum_I((x_i-M)**2*
    (1-u_i**2)**4)/(sum_I((1-u_i**2)*(1-5*u_i**2)))**2.
    Return sum((x-M)**2)/((n-1)*B). Contributions with abs(u)>=1
    are zero in both biweight sums. Zero MAD or a zero biweight
    denominator makes the estimator undefined and raises ValueError.

    References
    ----------
    .. [1] Martinez, J. and Iglewicz, B. (1981).
       A test for departure from normality based on a biweight estimator of scale.
       Biometrika, 68(1), 331-333.
       https://doi.org/10.1093/biomet/68.1.331

    Examples
    --------
    >>> statistic = MartinezIglewiczNormalityGofStatistic()
    >>> sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
    >>> value = statistic.execute_statistic(sample)
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        return "MI"

    @staticmethod
    @override
    def code():
        short_code = MartinezIglewiczNormalityGofStatistic.short_code()
        return f"{short_code}_{AbstractNormalityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute MartinezIglewicz for this sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real one-dimensional observations, n >= 4.
            Nonzero dispersion and the scales specified in Notes are required.
        **kwargs : dict, optional
            Unused keywords accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Scalar statistic in the normalization stated in the class notes.
            No p-value or test decision is computed.

        Raises
        ------
        ValueError
            If the sample is nonfinite, nonreal, multidimensional, too short,
            or a required scale or transformation is undefined.
        """
        rvs = _normal_sample(rvs, 4)
        return self.stat32(rvs)

    @staticmethod
    def stat32(x):
        n = len(x)

        if n > 3:
            x_tmp = np.copy(x)
            x_tmp.sort()
            if n % 2 == 0:
                m = (x_tmp[n // 2] + x_tmp[n // 2 - 1]) / 2.0
            else:
                m = x_tmp[n // 2]

            aux1 = x - m
            x_tmp = np.abs(aux1)
            x_tmp.sort()
            if n % 2 == 0:
                a = (x_tmp[n // 2] + x_tmp[n // 2 - 1]) / 2.0
            else:
                a = x_tmp[n // 2]
            a = 9.0 * _positive_scale(a, "median absolute deviation")

            z = aux1 / a
            inside = np.abs(z) < 1
            zi = z[inside]
            term1 = np.sum(aux1[inside] ** 2 * (1 - zi**2) ** 4)
            term2 = np.sum((1 - zi**2) * (1 - 5 * zi**2))
            _positive_scale(term1, "biweight numerator")
            if term2 == 0:
                raise ValueError("Biweight scale denominator is zero")
            term3 = np.sum(aux1**2)

            sb2 = (n * term1) / term2**2
            stat_in = (term3 / (n - 1)) / sb2
            return stat_in


class CabanaCabana1NormalityGofStatistic(AbstractNormalityGofStatistic):
    """Cabaña-Cabaña normality statistic focused on skewness departures.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic.
    hypothesis()
        Report fixed null parameters only.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is N(mu,sigma**2), with unknown real mu and sigma>0; hypothesis() reports {}.
    Location and positive scale cancel, allowing N(0,1) null simulation.
    Every simulated sample must go through execute_statistic again.
    The upper tail of the statistic defines rejection. Use at least 4
    finite real observations in one dimension. Nonzero dispersion is required.
    Ties are accepted unless a scale or contrast becomes undefined.
    Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
    the standard normal CDF and density, s0=std(x,ddof=0), and
    s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

    Use the l=5 process in Cabana and Cabana, equation (7).
    For z=(x-mean(x))/s1, h_j=He_j/sqrt(j!), H_j=sum(h_j(z))/sqrt(n),
    P(t)=sum_{j=1..5}(h_(j-1)(t)*H_(j+3)/sqrt(j)). Return
    sup_t abs(Phi(t)*H_3-phi(t)*P(t)). Evaluate every real stationary
    point (roots of H_3+t*P-P') and both limits at infinity.
    This is the supremum of the truncated process, not the untruncated
    infinite series. Polynomial root solving is subject to float64 accuracy.

    References
    ----------
    .. [1] Cabaña, A. and Cabaña, E. M. (2003).
       Tests of Normality Based on Transformed Empirical Processes.
       Methodology and Computing in Applied Probability, 5, 309-335.
       https://doi.org/10.1023/A:1026235220018

    Examples
    --------
    >>> statistic = CabanaCabana1NormalityGofStatistic()
    >>> sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
    >>> value = statistic.execute_statistic(sample)
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        return "CC1"

    @staticmethod
    @override
    def code():
        short_code = CabanaCabana1NormalityGofStatistic.short_code()
        return f"{short_code}_{AbstractNormalityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute CabanaCabana1 for this sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real one-dimensional observations, n >= 4.
            Nonzero dispersion and the scales specified in Notes are required.
        **kwargs : dict, optional
            Unused keywords accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Scalar statistic in the normalization stated in the class notes.
            No p-value or test decision is computed.

        Raises
        ------
        ValueError
            If the sample is nonfinite, nonreal, multidimensional, too short,
            or a required scale or transformation is undefined.
        """
        rvs = _normal_sample(rvs, 4)
        return self.stat19(rvs)

    @staticmethod
    def stat19(x):
        return _cabana_supremum(x, kurtosis=False)


class CabanaCabana2NormalityGofStatistic(AbstractNormalityGofStatistic):
    """Cabaña-Cabaña-style normality statistic focused on kurtosis departures.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic.
    hypothesis()
        Report fixed null parameters only.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is N(mu,sigma**2), with unknown real mu and sigma>0; hypothesis() reports {}.
    Location and positive scale cancel, allowing N(0,1) null simulation.
    Every simulated sample must go through execute_statistic again.
    The upper tail of the statistic defines rejection. Use at least 4
    finite real observations in one dimension. Nonzero dispersion is required.
    Ties are accepted unless a scale or contrast becomes undefined.
    Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
    the standard normal CDF and density, s0=std(x,ddof=0), and
    s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

    Use the l=5 process in Cabana and Cabana, equation (8).
    For z=(x-mean(x))/s1, h_j=He_j/sqrt(j!), H_j=sum(h_j(z))/sqrt(n),
    Q(t)=sum_{j=2..5}((sqrt(j/(j-1))*h_(j-2)(t)+h_j(t))*H_(j+3)).
    Set P(t)=H_3+t*H_4+Q(t). Return
    sup_t abs(Phi(t)*H_4-phi(t)*P(t)). Evaluate every real stationary
    point (roots of H_4+t*P-P') and both limits at infinity.
    The H_8 term occurs once. This is the truncated-process supremum;
    polynomial root solving is subject to float64 accuracy.

    References
    ----------
    .. [1] Cabaña, A. and Cabaña, E. M. (2003).
       Tests of Normality Based on Transformed Empirical Processes.
       Methodology and Computing in Applied Probability, 5, 309-335.
       https://doi.org/10.1023/A:1026235220018

    Examples
    --------
    >>> statistic = CabanaCabana2NormalityGofStatistic()
    >>> sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
    >>> value = statistic.execute_statistic(sample)
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        return "CC2"

    @staticmethod
    @override
    def code():
        short_code = CabanaCabana2NormalityGofStatistic.short_code()
        return f"{short_code}_{AbstractNormalityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute CabanaCabana2 for this sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real one-dimensional observations, n >= 4.
            Nonzero dispersion and the scales specified in Notes are required.
        **kwargs : dict, optional
            Unused keywords accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Scalar statistic in the normalization stated in the class notes.
            No p-value or test decision is computed.

        Raises
        ------
        ValueError
            If the sample is nonfinite, nonreal, multidimensional, too short,
            or a required scale or transformation is undefined.
        """
        rvs = _normal_sample(rvs, 4)
        return self.stat20(rvs)

    @staticmethod
    def stat20(x):
        return _cabana_supremum(x, kurtosis=True)


class ChenShapiroNormalityGofStatistic(AbstractNormalityGofStatistic):
    """Chen-Shapiro normality statistic based on normalized spacings.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic.
    hypothesis()
        Report fixed null parameters only.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is N(mu,sigma**2), with unknown real mu and sigma>0; hypothesis() reports {}.
    Location and positive scale cancel, allowing N(0,1) null simulation.
    Every simulated sample must go through execute_statistic again.
    The upper tail of the statistic defines rejection. Use at least 4
    finite real observations in one dimension. Nonzero dispersion is required.
    Ties are accepted unless a scale or contrast becomes undefined.
    Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
    the standard normal CDF and density, s0=std(x,ddof=0), and
    s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

    For u_i=Phi**-1((i-3/8)/(n+1/4)), return
    sqrt(n)*(1-sum_i((x_(i+1)-x_(i))/(u_(i+1)-u_i))/((n-1)*s1)).
    This is the upper-tail QH* version, not the lower-tail raw QH.

    References
    ----------
    .. [1] Chen, L. and Shapiro, S. S. (1995).
       An alternative test for normality based on normalized spacings.
       Journal of Statistical Computation and Simulation, 53(3-4), 269-287.
       https://doi.org/10.1080/00949659508811711

    Examples
    --------
    >>> statistic = ChenShapiroNormalityGofStatistic()
    >>> sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
    >>> value = statistic.execute_statistic(sample)
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        return "CS"

    @staticmethod
    @override
    def code():
        short_code = ChenShapiroNormalityGofStatistic.short_code()
        return f"{short_code}_{AbstractNormalityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute ChenShapiro for this sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real one-dimensional observations, n >= 4.
            Nonzero dispersion and the scales specified in Notes are required.
        **kwargs : dict, optional
            Unused keywords accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Scalar statistic in the normalization stated in the class notes.
            No p-value or test decision is computed.

        Raises
        ------
        ValueError
            If the sample is nonfinite, nonreal, multidimensional, too short,
            or a required scale or transformation is undefined.
        """
        rvs = _normal_sample(rvs, 4)
        return self.stat26(rvs)

    @staticmethod
    def stat26(x):
        n = len(x)

        if n > 3:
            xs = np.sort(x)
            # mean_x = np.mean(x)
            var_x = np.var(x, ddof=1)
            m = scipy_stats.norm.ppf(np.arange(1, n + 1) / (n + 0.25) - 0.375 / (n + 0.25))
            stat_cs = np.sum((xs[1:] - xs[:-1]) / (m[1:] - m[:-1])) / ((n - 1) * np.sqrt(var_x))
            stat_cs = np.sqrt(n) * (1.0 - stat_cs)
            return stat_cs


class ZhangQNormalityGofStatistic(AbstractNormalityGofStatistic):
    """Zhang normality statistic based on a ratio of ordered-sample contrasts.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic.
    hypothesis()
        Report fixed null parameters only.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is N(mu,sigma**2), with unknown real mu and sigma>0; hypothesis() reports {}.
    Location and positive scale cancel, allowing N(0,1) null simulation.
    Every simulated sample must go through execute_statistic again.
    Both tails of the statistic defines rejection. Use at least 8
    finite real observations in one dimension. Nonzero dispersion is required.
    Ties are accepted unless a scale or contrast becomes undefined.
    Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
    the standard normal CDF and density, s0=std(x,ddof=0), and
    s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

    The publication was identified, but the exact finite-sample
    implementation is not fully certified against its primary text.

    For u_i=Phi**-1((i-3/8)/(n+1/4)), set
    q1=mean_{i=2..n}((x_(i)-x_(1))/(u_i-u_1)),
    q2=mean_{i=1..n-4}((x_(i+4)-x_(i))/(u_(i+4)-u_i)).
    Return log(q1)-log(q2). The four-spacing construction and Blom
    approximation are explicit implementation choices. Both contrasts
    must be positive. Both critical tails are used for the signed log ratio;
    this is the Q component, not a combined Q/Q* p-value procedure.

    References
    ----------
    .. [1] Zhang, P. (1999).
       Omnibus test of normality using the Q statistic.
       Journal of Applied Statistics, 26(4), 519-528.
       https://doi.org/10.1080/02664769922395

    Examples
    --------
    >>> statistic = ZhangQNormalityGofStatistic()
    >>> sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
    >>> value = statistic.execute_statistic(sample)
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return TwoSidedAlternative()

    @staticmethod
    @override
    def short_code():
        return "ZQ"

    @staticmethod
    @override
    def code():
        short_code = ZhangQNormalityGofStatistic.short_code()
        return f"{short_code}_{AbstractNormalityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute ZhangQ for this sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real one-dimensional observations, n >= 8.
            Nonzero dispersion and the scales specified in Notes are required.
        **kwargs : dict, optional
            Unused keywords accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Scalar statistic in the normalization stated in the class notes.
            No p-value or test decision is computed.

        Raises
        ------
        ValueError
            If the sample is nonfinite, nonreal, multidimensional, too short,
            or a required scale or transformation is undefined.
        """
        rvs = _normal_sample(rvs, 8)
        return self.stat27(rvs)

    @staticmethod
    def stat27(x):
        n = len(x)

        if n > 3:
            u = scipy_stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
            xs = np.sort(x)
            a = np.zeros(n)
            b = np.zeros(n)
            term = 0.0
            for i in range(2, n + 1):
                a[i - 1] = 1.0 / ((n - 1) * (u[i - 1] - u[0]))
                term += a[i - 1]
            a[0] = -term
            b[0] = 1.0 / ((n - 4) * (u[0] - u[4]))
            b[n - 1] = -b[0]
            b[1] = 1.0 / ((n - 4) * (u[1] - u[5]))
            b[n - 2] = -b[1]
            b[2] = 1.0 / ((n - 4) * (u[2] - u[6]))
            b[n - 3] = -b[2]
            b[3] = 1.0 / ((n - 4) * (u[3] - u[7]))
            b[n - 4] = -b[3]
            for i in range(5, n - 3):
                b[i - 1] = (1.0 / (u[i - 1] - u[i + 3]) - 1.0 / (u[i - 5] - u[i - 1])) / (n - 4)
            q1 = np.dot(a, xs)
            q2 = np.dot(b, xs)
            stat_q = np.log(_positive_scale(q1, "Q numerator")) - np.log(
                _positive_scale(q2, "Q denominator")
            )
            return stat_q


class CoinNormalityGofStatistic(AbstractNormalityGofStatistic):
    """Coin normality statistic based on polynomial regression.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic.
    hypothesis()
        Report fixed null parameters only.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is N(mu,sigma**2), with unknown real mu and sigma>0; hypothesis() reports {}.
    Location and positive scale cancel, allowing N(0,1) null simulation.
    Every simulated sample must go through execute_statistic again.
    The upper tail of the statistic defines rejection. Use at least 4
    finite real observations in one dimension. Nonzero dispersion is required.
    Ties are accepted unless a scale or contrast becomes undefined.
    Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
    the standard normal CDF and density, s0=std(x,ddof=0), and
    s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

    The publication was identified, but the exact finite-sample
    implementation is not fully certified against its primary text.

    Regress z_(i)=(x_(i)-mean(x))/s1 on a_i and a_i**3 without
    an intercept, where a_i are approximate expected normal order scores
    from nscor2. Return the square of the cubic coefficient. Normal-score
    approximations are only supported here for n<=2000. The reference
    identifies the regression procedure; its score approximation is
    retained and must be reproduced in calibration.

    References
    ----------
    .. [1] Coin, D. (2008).
       A goodness-of-fit test for normality based on polynomial regression.
       Computational Statistics & Data Analysis, 52(4), 2185-2198.
       https://doi.org/10.1016/j.csda.2007.07.012

    Examples
    --------
    >>> statistic = CoinNormalityGofStatistic()
    >>> sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
    >>> value = statistic.execute_statistic(sample)
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        return "COIN"

    @staticmethod
    @override
    def code():
        short_code = CoinNormalityGofStatistic.short_code()
        return f"{short_code}_{AbstractNormalityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute Coin for this sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real one-dimensional observations, n >= 4.
            Nonzero dispersion and the scales specified in Notes are required.
        **kwargs : dict, optional
            Unused keywords accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Scalar statistic in the normalization stated in the class notes.
            No p-value or test decision is computed.

        Raises
        ------
        ValueError
            If the sample is nonfinite, nonreal, multidimensional, too short,
            or a required scale or transformation is undefined.
        """
        rvs = _normal_sample(rvs, 4)
        if len(rvs) > 2000:
            raise ValueError("Coin normal-score approximation requires n <= 2000")
        return self.stat30(rvs)

    def stat30(self, x):
        n = len(x)

        if n > 3:
            z = [0] * n
            m = [n // 2]
            sp = [0] * m[0]
            a = [0] * n
            var_x = 0.0
            mean_x = 0.0
            term1 = 0.0
            term2 = 0.0
            term3 = 0.0
            term4 = 0.0
            term6 = 0.0

            for i in range(n):
                mean_x += x[i]
            mean_x /= n

            for i in range(n):
                var_x += x[i] ** 2
            var_x = np.var(x, ddof=1)
            sd_x = math.sqrt(var_x)

            for i in range(n):
                z[i] = (x[i] - mean_x) / sd_x

            z.sort()
            self.nscor2(sp, n, m)

            if n % 2 == 0:
                for i in range(n // 2):
                    a[i] = -sp[i]
                for i in range(n // 2, n):
                    a[i] = sp[n - i - 1]
            else:
                for i in range(n // 2):
                    a[i] = -sp[i]
                a[n // 2] = 0.0
                for i in range(n // 2 + 1, n):
                    a[i] = sp[n - i - 1]

            for i in range(n):
                term1 += a[i] ** 4
                term2 += a[i] * z[i]
                term3 += a[i] ** 2
                term4 += a[i] ** 3 * z[i]
                term6 += a[i] ** 6

            stat_beta32 = ((term1 * term2 - term3 * term4) / (term1 * term1 - term3 * term6)) ** 2
            return stat_beta32

    @staticmethod
    def correct(i, n):
        c1 = [9.5, 28.7, 1.9, 0.0, -7.0, -6.2, -1.6]
        c2 = [-6195.0, -9569.0, -6728.0, -17614.0, -8278.0, -3570.0, 1075.0]
        c3 = [93380.0, 175160.0, 410400.0, 2157600.0, 2.376e6, 2.065e6, 2.065e6]
        mic = 1e-6
        c14 = 1.9e-5

        if i * n == 4:
            return c14
        if i < 1 or i > 7:
            return 0
        if i != 4 and n > 20:
            return 0
        if i == 4 and n > 40:
            return 0

        an = 1.0 / (n * n)
        i -= 1
        return (c1[i] + an * (c2[i] + an * c3[i])) * mic

    def nscor2(self, s, n, n2):
        eps = [0.419885, 0.450536, 0.456936, 0.468488]
        dl1 = [0.112063, 0.12177, 0.239299, 0.215159]
        dl2 = [0.080122, 0.111348, -0.211867, -0.115049]
        gam = [0.474798, 0.469051, 0.208597, 0.259784]
        lam = [0.282765, 0.304856, 0.407708, 0.414093]
        bb = -0.283833
        d = -0.106136
        b1 = 0.5641896

        if n2[0] > n / 2:
            raise ValueError("n2>n")
        if n <= 1:
            raise ValueError("n<=1")
        if n > 2000:
            raise ValueError("Normal-score approximation requires n <= 2000")

        s[0] = b1
        if n == 2:
            return

        an = n
        k = 3
        k = min(k, n2[0])

        for i in range(k):
            ai = i + 1
            e1 = (ai - eps[i]) / (an + gam[i])
            e2 = e1 ** lam[i]
            s[i] = e1 + e2 * (dl1[i] + e2 * dl2[i]) / an - self.correct(i + 1, n)

        if n2[0] > k:
            for i in range(3, n2[0]):
                ai = i + 1
                e1 = (ai - eps[3]) / (an + gam[3])
                e2 = e1 ** (lam[3] + bb / (ai + d))
                s[i] = e1 + e2 * (dl1[3] + e2 * dl2[3]) / an - self.correct(i + 1, n)

        for i in range(n2[0]):
            s[i] = -scipy_stats.norm.ppf(s[i], 0.0, 1.0)

        return


class DagostinoNormalityGofStatistic(AbstractNormalityGofStatistic):
    """D'Agostino's normality statistic based on ordered observations.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic.
    hypothesis()
        Report fixed null parameters only.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is N(mu,sigma**2), with unknown real mu and sigma>0; hypothesis() reports {}.
    Location and positive scale cancel, allowing N(0,1) null simulation.
    Every simulated sample must go through execute_statistic again.
    Both tails of the statistic defines rejection. Use at least 4
    finite real observations in one dimension. Nonzero dispersion is required.
    Ties are accepted unless a scale or contrast becomes undefined.
    Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
    the standard normal CDF and density, s0=std(x,ddof=0), and
    s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

    D=sum_i((i-(n+1)/2)*x_(i))/(n**2*s0). Return
    sqrt(n)*(D-0.28209479)/0.02998598. This is the signed,
    asymptotically standardized D statistic, not D'Agostino-Pearson K2.

    References
    ----------
    .. [1] D'Agostino, R. B. (1971).
       An omnibus test of normality for moderate and large size samples.
       Biometrika, 58(2), 341-348.
       https://doi.org/10.1093/biomet/58.2.341

    Examples
    --------
    >>> statistic = DagostinoNormalityGofStatistic()
    >>> sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
    >>> value = statistic.execute_statistic(sample)
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return TwoSidedAlternative()

    @staticmethod
    @override
    def short_code():
        return "D"

    @staticmethod
    @override
    def code():
        short_code = DagostinoNormalityGofStatistic.short_code()
        return f"{short_code}_{AbstractNormalityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute Dagostino for this sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real one-dimensional observations, n >= 4.
            Nonzero dispersion and the scales specified in Notes are required.
        **kwargs : dict, optional
            Unused keywords accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Scalar statistic in the normalization stated in the class notes.
            No p-value or test decision is computed.

        Raises
        ------
        ValueError
            If the sample is nonfinite, nonreal, multidimensional, too short,
            or a required scale or transformation is undefined.
        """
        rvs = _normal_sample(rvs, 4)
        n = len(rvs)
        if n > 3:
            xs = np.sort(rvs)  # We sort the data
            var_x = np.var(xs, ddof=0)
            t = sum((i - 0.5 * (n + 1)) * xs[i - 1] for i in range(1, n + 1))
            d = t / ((n**2) * math.sqrt(var_x))
            stat_da = math.sqrt(n) * (d - 0.28209479) / 0.02998598

            return stat_da  # Here is the test statistic value


class ZhangQStarNormalityGofStatistic(AbstractNormalityGofStatistic):
    """Reflected Zhang ``Q*`` normality statistic.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic.
    hypothesis()
        Report fixed null parameters only.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is N(mu,sigma**2), with unknown real mu and sigma>0; hypothesis() reports {}.
    Location and positive scale cancel, allowing N(0,1) null simulation.
    Every simulated sample must go through execute_statistic again.
    Both tails of the statistic defines rejection. Use at least 8
    finite real observations in one dimension. Nonzero dispersion is required.
    Ties are accepted unless a scale or contrast becomes undefined.
    Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
    the standard normal CDF and density, s0=std(x,ddof=0), and
    s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

    The publication was identified, but the exact finite-sample
    implementation is not fully certified against its primary text.

    Return Zhang Q evaluated on the reflected sample -x, using
    u_i=Phi**-1((i-3/8)/(n+1/4)), the mean spacing from the minimum
    for q1 and mean four-spacing for q2; return log(q1)-log(q2).
    Both contrasts must be positive. Both tails of the signed log ratio
    are used. This is a reflected component, not combined Q/Q* inference.

    References
    ----------
    .. [1] Zhang, P. (1999).
       Omnibus test of normality using the Q statistic.
       Journal of Applied Statistics, 26(4), 519-528.
       https://doi.org/10.1080/02664769922395

    Examples
    --------
    >>> statistic = ZhangQStarNormalityGofStatistic()
    >>> sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
    >>> value = statistic.execute_statistic(sample)
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return TwoSidedAlternative()

    @staticmethod
    @override
    def short_code():
        return "ZQS"

    @staticmethod
    @override
    def code():
        short_code = ZhangQStarNormalityGofStatistic.short_code()
        return f"{short_code}_{AbstractNormalityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute ZhangQStar for this sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real one-dimensional observations, n >= 8.
            Nonzero dispersion and the scales specified in Notes are required.
        **kwargs : dict, optional
            Unused keywords accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Scalar statistic in the normalization stated in the class notes.
            No p-value or test decision is computed.

        Raises
        ------
        ValueError
            If the sample is nonfinite, nonreal, multidimensional, too short,
            or a required scale or transformation is undefined.
        """
        rvs = _normal_sample(rvs, 8)
        n = len(rvs)

        if n > 3:
            # Computation of the value of the test statistic
            xs = np.sort(rvs)
            u = scipy_stats.norm.ppf(np.arange(1, n + 1) / (n + 0.25) - 0.375 / (n + 0.25))

            a = np.zeros(n)
            a[1:] = 1 / ((n - 1) * (u[1:] - u[0]))
            a[0] = -a[1:].sum()

            b = np.zeros(n)
            b[0] = 1 / ((n - 4) * (u[0] - u[4]))
            b[-1] = -b[0]
            b[1] = 1 / ((n - 4) * (u[1] - u[5]))
            b[-2] = -b[1]
            b[2] = 1 / ((n - 4) * (u[2] - u[6]))
            b[-3] = -b[2]
            b[3] = 1 / ((n - 4) * (u[3] - u[7]))
            b[-4] = -b[3]
            for i in range(4, n - 4):
                b[i] = (1 / (u[i] - u[i + 4]) - 1 / (u[i - 4] - u[i])) / (n - 4)

            q1_star = -np.dot(a, xs[::-1])
            q2_star = -np.dot(b, xs[::-1])

            q_star = np.log(_positive_scale(q1_star, "Q* numerator")) - np.log(
                _positive_scale(q2_star, "Q* denominator")
            )
            return q_star


class SWRGNormalityGofStatistic(AbstractNormalityGofStatistic):
    """Rahman-Govindarajulu modification of the Shapiro-Wilk statistic.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic.
    hypothesis()
        Report fixed null parameters only.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is N(mu,sigma**2), with unknown real mu and sigma>0; hypothesis() reports {}.
    Location and positive scale cancel, allowing N(0,1) null simulation.
    Every simulated sample must go through execute_statistic again.
    The lower tail of the statistic defines rejection. Use at least 4
    finite real observations in one dimension. Nonzero dispersion is required.
    Ties are accepted unless a scale or contrast becomes undefined.
    Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
    the standard normal CDF and density, s0=std(x,ddof=0), and
    s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

    For m_i=Phi**-1(i/(n+1)), f_i=phi(m_i), form weights
    a*_i=-(n+1)*(n+2)*f_i*(m_(i-1)*f_(i-1)-2*m_i*f_i
    +m_(i+1)*f_(i+1)), taking missing endpoint terms as zero.
    Normalize a=a*/sqrt(sum(a* **2)); return
    (sum(a_i*x_(i)))**2/sum((x-mean(x))**2). Small W_RG rejects.

    References
    ----------
    .. [1] Rahman, M. M. and Govindarajulu, Z. (1997).
       A modification of the test of Shapiro and Wilk for normality.
       Journal of Applied Statistics, 24(2), 219-236.
       https://doi.org/10.1080/02664769723828

    Examples
    --------
    >>> statistic = SWRGNormalityGofStatistic()
    >>> sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
    >>> value = statistic.execute_statistic(sample)
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return LeftAlternative()

    @staticmethod
    @override
    def short_code():
        return "SWRG"

    @staticmethod
    @override
    def code():
        short_code = SWRGNormalityGofStatistic.short_code()
        return f"{short_code}_{AbstractNormalityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute SWRG for this sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real one-dimensional observations, n >= 4.
            Nonzero dispersion and the scales specified in Notes are required.
        **kwargs : dict, optional
            Unused keywords accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Scalar statistic in the normalization stated in the class notes.
            No p-value or test decision is computed.

        Raises
        ------
        ValueError
            If the sample is nonfinite, nonreal, multidimensional, too short,
            or a required scale or transformation is undefined.
        """
        rvs = _normal_sample(rvs, 4)
        n = len(rvs)

        if n > 3:
            # Computation of the value of the test statistic
            mi = scipy_stats.norm.ppf(np.arange(1, n + 1) / (n + 1))
            fi = scipy_stats.norm.pdf(mi)
            aux2 = 2 * mi * fi
            aux1 = np.concatenate(([0], mi[:-1] * fi[:-1]))
            aux3 = np.concatenate((mi[1:] * fi[1:], [0]))
            aux4 = aux1 - aux2 + aux3
            ai_star = -((n + 1) * (n + 2)) * fi * aux4
            norm2 = np.sum(ai_star**2)
            ai = ai_star / np.sqrt(norm2)

            xs = np.sort(rvs)
            mean_x = np.mean(xs)
            aux6 = np.sum((xs - mean_x) ** 2)
            stat_wrg = np.sum(ai * xs) ** 2 / aux6

            return stat_wrg  # Here is the test statistic value


class GMGNormalityGofStatistic(AbstractNormalityGofStatistic):
    """Gel-Miao-Gastwirth normality statistic directed at heavy tails.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic.
    hypothesis()
        Report fixed null parameters only.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is N(mu,sigma**2), with unknown real mu and sigma>0; hypothesis() reports {}.
    Location and positive scale cancel, allowing N(0,1) null simulation.
    Every simulated sample must go through execute_statistic again.
    The upper tail of the statistic defines rejection. Use at least 4
    finite real observations in one dimension. Nonzero dispersion is required.
    Ties are accepted unless a scale or contrast becomes undefined.
    Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
    the standard normal CDF and density, s0=std(x,ddof=0), and
    s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

    Return s0/J, where J=sqrt(pi/2)*mean(abs(x-median(x))).
    This is a directed upper-tail statistic for heavy-tailed departures.
    The ddof=0 convention is retained; its finite-sample law differs
    from versions using the unbiased sample variance.

    References
    ----------
    .. [1] Gel, Y. R., Miao, W. and Gastwirth, J. L. (2007).
       Robust directed tests of normality against heavy-tailed alternatives.
       Computational Statistics & Data Analysis, 51(5), 2734-2746.
       https://doi.org/10.1016/j.csda.2006.08.022

    Examples
    --------
    >>> statistic = GMGNormalityGofStatistic()
    >>> sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
    >>> value = statistic.execute_statistic(sample)
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        return "GMG"

    @staticmethod
    @override
    def code():
        short_code = GMGNormalityGofStatistic.short_code()
        return f"{short_code}_{AbstractNormalityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute GMG for this sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real one-dimensional observations, n >= 4.
            Nonzero dispersion and the scales specified in Notes are required.
        **kwargs : dict, optional
            Unused keywords accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Scalar statistic in the normalization stated in the class notes.
            No p-value or test decision is computed.

        Raises
        ------
        ValueError
            If the sample is nonfinite, nonreal, multidimensional, too short,
            or a required scale or transformation is undefined.
        """
        rvs = _normal_sample(rvs, 4)
        return self.stat33(rvs)

    @staticmethod
    def stat33(x):
        n = len(x)

        if n > 3:
            import math

            x_tmp = [0] * n
            var_x = 0.0
            mean_x = 0.0
            jn = 0.0
            pi = 4.0 * math.atan(1.0)  # or use pi = M_PI, where M_PI is defined in math.h

            # calculate sample mean
            for i in range(n):
                mean_x += x[i]
            mean_x = mean_x / n

            # calculate sample var and standard deviation
            for i in range(n):
                var_x += (x[i] - mean_x) ** 2
            var_x = var_x / n
            sd_x = math.sqrt(var_x)

            # calculate sample median
            for i in range(n):
                x_tmp[i] = x[i]

            x_tmp = np.sort(x_tmp)  # We sort the data

            if n % 2 == 0:
                m = (x_tmp[n // 2] + x_tmp[n // 2 - 1]) / 2.0
            else:
                m = x_tmp[n // 2]  # sample median

            # calculate statRsJ
            for i in range(n):
                jn += abs(x[i] - m)
            jn = math.sqrt(pi / 2.0) * jn / n

            stat_rsj = sd_x / jn

            return stat_rsj  # Here is the test statistic value


class BHSNormalityGofStatistic(AbstractNormalityGofStatistic):
    """Brys-Hubert-Struyf MC-LR statistic for the normal family.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic.
    hypothesis()
        Report fixed null parameters only.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is N(mu,sigma**2), with unknown real mu and sigma>0; hypothesis() reports {}.
    Location and positive scale cancel, allowing N(0,1) null simulation.
    Every simulated sample must go through execute_statistic again.
    The upper tail of the statistic defines rejection. Use at least 4
    finite real observations in one dimension. Nonzero dispersion is required.
    Ties are accepted unless a scale or contrast becomes undefined.
    Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
    the standard normal CDF and density, s0=std(x,ddof=0), and
    s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

    Return n*v.T*V**-1*v with v=(MC, LMC-0.198828, RMC-0.198828),
    V=[[1.24581,0.322918,-0.322918],[0.322918,2.62068,-0.0123455],
    [-0.322918,-0.0123455,2.62068]]. LMC=-MC(x<median(x));
    RMC=MC(x>median(x)); median observations belong to neither half.
    MC is the median of (upper+lower-2*median)/(upper-lower) over
    lower<=median<=upper, with antisymmetric rank values for zero ties.
    At least two observations must remain in each strict half. Constant
    halves have MC=0. The exact kernel median uses O(n**2) time and
    memory, unlike the fast selection algorithm in the paper. Constants
    agree with the rounded normal row in Table 1; simulate for finite n.

    References
    ----------
    .. [1] Brys, G., Hubert, M., and Struyf, A. (2008). "Goodness-of-fit tests
       based on a robust measure of skewness." Computational Statistics,
       23, 429-442. https://doi.org/10.1007/s00180-007-0083-7

    Examples
    --------
    >>> statistic = BHSNormalityGofStatistic()
    >>> sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
    >>> value = statistic.execute_statistic(sample)
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        return "BHS"

    @staticmethod
    @override
    def code():
        short_code = BHSNormalityGofStatistic.short_code()
        return f"{short_code}_{AbstractNormalityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute BHS for this sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real one-dimensional observations, n >= 4.
            Nonzero dispersion and the scales specified in Notes are required.
        **kwargs : dict, optional
            Unused keywords accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Scalar statistic in the normalization stated in the class notes.
            No p-value or test decision is computed.

        Raises
        ------
        ValueError
            If the sample is nonfinite, nonreal, multidimensional, too short,
            or a required scale or transformation is undefined.
        """
        rvs = _normal_sample(rvs, 4)
        return self.stat16(rvs)

    def stat16(self, x):
        n = len(x)

        if n > 3:
            # Computation of the value of the test statistic
            x_sorted = np.sort(x)
            median = np.median(x_sorted)
            x1 = x_sorted[x_sorted < median]
            x2 = x_sorted[x_sorted > median]
            if min(len(x1), len(x2)) < 2:
                raise ValueError(
                    "BHS requires at least two observations on each side of the median"
                )
            eps = [2.220446e-16, 2.225074e-308]
            w1 = self.mc_c_d(x, eps, [1000, 0])
            w2 = self.mc_c_d(x1, eps, [1000, 0])
            w3 = self.mc_c_d(x2, eps, [1000, 0])

            omega = [0.0, 0.198828, 0.198828]
            vec = [w1 - omega[0], -w2 - omega[1], w3 - omega[2]]

            inv_v = np.array(
                [
                    [0.8571890822945882, -0.1051268907484579, 0.1051268907484580],
                    [-0.1051268907484579, 0.3944817329840534, -0.01109532299714422],
                    [0.1051268907484579, -0.01109532299714422, 0.3944817329840535],
                ]
            )

            stat_tmclr = n * np.dot(vec, np.dot(inv_v, vec))
            return stat_tmclr  # Here is the test statistic value

    # TODO: refactor
    def mc_c_d(self, z, eps, iter_):
        """Compute the exact medcouple; compatibility arguments do not tune it.

        The kernel matrix uses O(n**2) memory and time. Median ties use the
        antisymmetric rank convention of Brys, Hubert and Struyf.
        """
        x = np.sort(np.asarray(z, dtype=float))
        if x.size < 2:
            raise ValueError("A medcouple requires at least two observations")
        x = x - np.median(x)
        lower = x[x <= 0]
        upper = x[x >= 0]
        denominator = upper[:, None] - lower[None, :]
        numerator = upper[:, None] + lower[None, :]
        kernel = np.divide(
            numerator, denominator, out=np.zeros_like(denominator), where=denominator != 0
        )
        ties = np.count_nonzero(x == 0)
        if ties:
            ranks = np.arange(ties)
            kernel[:ties, -ties:] = np.sign(ranks[:, None] + ranks[None, :] - ties + 1)
        iter_[:] = [0, True]
        return float(np.median(kernel))

    @staticmethod
    def h_kern(a, b, ai, bi, ab, eps):
        if abs(a - b) < 2 * eps or b > 0:
            return np.sign(ab - (ai + bi))
        return (a + b) / (a - b)

    @staticmethod
    def whi_med_i(a, w, n, a_cand, a_srt, w_cand):
        """Return the upper weighted median of the first n entries."""
        order = np.argsort(np.asarray(a)[:n])
        weights = np.asarray(w)[:n][order]
        if n == 0 or np.any(weights < 0) or np.sum(weights) <= 0:
            raise ValueError("Positive total weight is required")
        index = np.searchsorted(np.cumsum(weights), np.sum(weights) / 2, side="right")
        return float(np.asarray(a)[:n][order[index]])


class SpiegelhalterNormalityGofStatistic(AbstractNormalityGofStatistic):
    """Spiegelhalter statistic for the normal family.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic.
    hypothesis()
        Report fixed null parameters only.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is N(mu,sigma**2), with unknown real mu and sigma>0; hypothesis() reports {}.
    Location and positive scale cancel, allowing N(0,1) null simulation.
    Every simulated sample must go through execute_statistic again.
    Both tails of the statistic defines rejection. Use at least 4
    finite real observations in one dimension. Nonzero dispersion is required.
    Ties are accepted unless a scale or contrast becomes undefined.
    Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
    the standard normal CDF and density, s0=std(x,ddof=0), and
    s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

    The publication was identified, but the exact finite-sample
    implementation is not fully certified against its primary text.

    Let U=(max(x)-min(x))/s1, G=sum(abs(x-mean(x)))/
    (s1*sqrt(n*(n-1))), and c_n=Gamma(n+1)**(1/(n-1))/(2*n).
    Return ((c_n*U)**(-(n-1))+G**(-(n-1)))**(1/(n-1)).
    Log-gamma and logaddexp evaluate this expression without overflowing.
    Both critical tails are retained; the full primary formula and
    its tail convention could not be independently checked in this review.

    References
    ----------
    .. [1] Spiegelhalter, D. J. (1977). "A test for normality against
       symmetric alternatives." Biometrika, 64(2), 415-418.
       https://doi.org/10.1093/biomet/64.2.415

    Examples
    --------
    >>> statistic = SpiegelhalterNormalityGofStatistic()
    >>> sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
    >>> value = statistic.execute_statistic(sample)
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return TwoSidedAlternative()

    @staticmethod
    @override
    def short_code():
        return "SH"

    @staticmethod
    @override
    def code():
        short_code = SpiegelhalterNormalityGofStatistic.short_code()
        return f"{short_code}_{AbstractNormalityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute Spiegelhalter for this sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real one-dimensional observations, n >= 4.
            Nonzero dispersion and the scales specified in Notes are required.
        **kwargs : dict, optional
            Unused keywords accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Scalar statistic in the normalization stated in the class notes.
            No p-value or test decision is computed.

        Raises
        ------
        ValueError
            If the sample is nonfinite, nonreal, multidimensional, too short,
            or a required scale or transformation is undefined.
        """
        rvs = _normal_sample(rvs, 4)
        return self.stat41(rvs)

    @staticmethod
    def stat41(x):
        n = len(x)

        if n > 3:
            stat_sp, var_x, mean = 0.0, 0.0, 0.0
            max_val, min_val = x[0], x[0]
            for i in range(1, n):
                max_val = max(max_val, x[i])
                min_val = min(min_val, x[i])
            for i in range(n):
                mean += x[i]
            mean /= n
            for i in range(n):
                var_x += (x[i] - mean) ** 2
            var_x /= n - 1
            sd = math.sqrt(var_x)
            u = (max_val - min_val) / sd
            g = 0.0
            for i in range(n):
                g += abs(x[i] - mean)
            g /= sd * math.sqrt(n) * math.sqrt(n - 1)
            log_cn = gammaln(n + 1) / (n - 1) - np.log(2 * n)
            log_a = -(n - 1) * (log_cn + np.log(u))
            log_b = -(n - 1) * np.log(g)
            stat_sp = np.exp(np.logaddexp(log_a, log_b) / (n - 1))
            return stat_sp  # Here is the test statistic value


class DesgagneLafayeNormalityGofStatistic(AbstractNormalityGofStatistic):
    """Desgagne-Lafaye de Micheaux-Leblanc R_n normality statistic.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic.
    hypothesis()
        Report fixed null parameters only.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is N(mu,sigma**2), with unknown real mu and sigma>0; hypothesis() reports {}.
    Location and positive scale cancel, allowing N(0,1) null simulation.
    Every simulated sample must go through execute_statistic again.
    The upper tail of the statistic defines rejection. Use at least 4
    finite real observations in one dimension. Nonzero dispersion is required.
    Ties are accepted unless a scale or contrast becomes undefined.
    Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
    the standard normal CDF and density, s0=std(x,ddof=0), and
    s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

    For z=(x-mean(x))/s0, let r=(0.18240929-mean(z**2*log(abs(z)))/2,
    0.5348223-mean(log(1+abs(z))),
    0.20981558-mean(log(log(e+abs(z))))). Return n*r.T*A*r, where
    A is the rounded inverse fitted-score covariance in stat35.
    The first integrand is zero at z=0 by continuity. Theorem 1 of the
    author preprint accounts for fitting mean and variance. A is ill
    conditioned; rounding its constants limits precision. Small-sample
    chi-square calibration is inappropriate.

    References
    ----------
    .. [1] Desgagne, A., Lafaye de Micheaux, P., and Leblanc, A. (2013).
       "Test of Normality Against Generalized Exponential Power
       Alternatives." Communications in Statistics - Theory and Methods,
       42(1), 164-190. https://doi.org/10.1080/03610926.2011.577548

    Examples
    --------
    >>> statistic = DesgagneLafayeNormalityGofStatistic()
    >>> sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
    >>> value = statistic.execute_statistic(sample)
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        return "DLDMZEPD"

    @staticmethod
    @override
    def code():
        short_code = DesgagneLafayeNormalityGofStatistic.short_code()
        return f"{short_code}_{AbstractNormalityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute DesgagneLafaye for this sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real one-dimensional observations, n >= 4.
            Nonzero dispersion and the scales specified in Notes are required.
        **kwargs : dict, optional
            Unused keywords accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Scalar statistic in the normalization stated in the class notes.
            No p-value or test decision is computed.

        Raises
        ------
        ValueError
            If the sample is nonfinite, nonreal, multidimensional, too short,
            or a required scale or transformation is undefined.
        """
        rvs = _normal_sample(rvs, 4)
        return self.stat35(rvs)

    @staticmethod
    def stat35(x):
        n = len(x)

        if n > 3:
            # Computation of the value of the test statistic
            y = np.zeros(n)
            varpop_x = 0.0
            mean_x = np.mean(x)
            r1 = 0.0
            r2 = 0.0
            r3 = 0.0

            for i in range(n):
                varpop_x += x[i] ** 2
            varpop_x = np.var(x, ddof=0)
            sd_x = np.sqrt(varpop_x)
            for i in range(n):
                y[i] = (x[i] - mean_x) / sd_x

            # Formulas given in our paper p. 169
            for i in range(n):
                r1 += xlogy(y[i] ** 2, abs(y[i]))
                r2 += np.log1p(abs(y[i]))
                r3 += np.log(np.log(2.71828182846 + abs(y[i])))
            r1 = 0.18240929 - 0.5 * r1 / n
            r2 = 0.5348223 - r2 / n
            r3 = 0.20981558 - r3 / n

            # Formula given in our paper p. 170
            rn = n * (
                (r1 * 1259.04213344 - r2 * 32040.69569026 + r3 * 85065.77739473) * r1
                + (-r1 * 32040.6956903 + r2 * 918649.9005906 - r3 * 2425883.3443201) * r2
                + (r1 * 85065.7773947 - r2 * 2425883.3443201 + r3 * 6407749.8211208) * r3
            )

            return rn  # Here is the test statistic value


class AbstractGraphNormalityGofStatistic(
    AbstractNormalityGofStatistic, AbstractGraphTestStatistic, ABC
):
    """Base for local normal proximity graphs with fixed positive variance.

    Concrete class notes specify the radius, graph summary and calibration.
    The graph eliminates location only; its null law depends on variance.
    """

    @override
    def __init__(self, *, var: float = 1):
        _validate_normal_parameters(0, var)
        self.var = var

    @override
    def hypothesis(self) -> GoodnessOfFitHypothesis:
        """Return normality with specified variance and unknown mean.

        Returns
        -------
        GoodnessOfFitHypothesis
            Only ``var`` is fixed. Calibration may use zero mean because the
            graph construction is invariant under a common shift.
        """
        return GoodnessOfFitHypothesis({"var": self.var})

    @override
    def alternative(self) -> Alternative:
        return TwoSidedAlternative()

    @staticmethod
    @override
    def code():
        parent_code = AbstractNormalityGofStatistic.code()
        return f"GRAPH_{parent_code}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the local proximity-graph statistic without modifying input.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            At least two finite real observations with positive variance.
        **kwargs : dict, optional
            Unused compatibility keywords.

        Returns
        -------
        statistic : float
            Graph summary; both tails are used in calibration at fixed variance.

        Raises
        ------
        ValueError
            If the sample or the graph radius is undefined in float64.
        """
        x = _normal_sample(rvs, 2, standardize=False)
        dist = self._compute_dist(x)
        if self.short_code() == "CLIQUENUMBER":
            x.sort()
            left = 0
            result = 1
            for right in range(len(x)):
                while left < right and x[right] - x[left] >= dist:
                    left += 1
                result = max(result, right - left + 1)
            return float(result)
        if self.short_code() == "INDEPENDENCENUMBER":
            x.sort()
            last = x[0]
            result = 1
            for value in x[1:]:
                if value - last >= dist:
                    result += 1
                    last = value
            return float(result)
        return float(self.get_graph_stat(self._make_adjacency_list(x, dist)))

    @staticmethod
    @override
    def _compute_dist(rvs):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            variance = np.var(rvs)
            dist = (np.max(rvs) - np.min(rvs)) / (10 * variance)
        _positive_scale(variance, "sample variance")
        return _positive_scale(dist, "graph radius")


class GraphEdgesNumberNormalityGofStatistic(
    AbstractGraphNormalityGofStatistic, GraphEdgesNumberTestStatistic
):
    """Number of edges for a normal sample with known variance.

    Parameters
    ----------
    var : float, optional
        Positive finite scalar variance fixed under H0; default 1.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic.
    hypothesis()
        Report fixed null parameters only.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is N(mu,var), with unknown mu and fixed var; hypothesis() reports var.
    Calibrate at this variance; setting the simulation mean to zero is valid.
    Both tails of the statistic defines rejection. Use at least 2
    finite real observations in one dimension. Nonzero dispersion is required.
    Ties are accepted unless a scale or contrast becomes undefined.
    Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
    the standard normal CDF and density, s0=std(x,ddof=0), and
    s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

    Connect i and j when abs(x_i-x_j) < (max(x)-min(x))/(10*m2),
    where m2=mean((x-mean(x))**2). Return the number of edges.
    This local construction is translation invariant but not scale invariant.
    var selects the null law for calibration; it is not inserted in the radius.
    A primary source for this exact normality test was not located. No
    published null law or omnibus power claim is made. Extreme scales that
    make the radius or variance unrepresentable raise ValueError.

    Examples
    --------
    >>> statistic = GraphEdgesNumberNormalityGofStatistic()
    >>> sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
    >>> value = statistic.execute_statistic(sample)
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def code():
        parent_code = AbstractGraphNormalityGofStatistic.code()
        short_code = GraphEdgesNumberNormalityGofStatistic.short_code()
        return f"{short_code}_{parent_code}"


class GraphMaxDegreeNormalityGofStatistic(
    AbstractGraphNormalityGofStatistic, GraphMaxDegreeTestStatistic
):
    """Maximum vertex degree for a normal sample with known variance.

    Parameters
    ----------
    var : float, optional
        Positive finite scalar variance fixed under H0; default 1.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic.
    hypothesis()
        Report fixed null parameters only.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is N(mu,var), with unknown mu and fixed var; hypothesis() reports var.
    Calibrate at this variance; setting the simulation mean to zero is valid.
    Both tails of the statistic defines rejection. Use at least 2
    finite real observations in one dimension. Nonzero dispersion is required.
    Ties are accepted unless a scale or contrast becomes undefined.
    Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
    the standard normal CDF and density, s0=std(x,ddof=0), and
    s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

    Connect i and j when abs(x_i-x_j) < (max(x)-min(x))/(10*m2),
    where m2=mean((x-mean(x))**2). Return the maximum degree.
    This local construction is translation invariant but not scale invariant.
    var selects the null law for calibration; it is not inserted in the radius.
    A primary source for this exact normality test was not located. No
    published null law or omnibus power claim is made. Extreme scales that
    make the radius or variance unrepresentable raise ValueError.

    Examples
    --------
    >>> statistic = GraphMaxDegreeNormalityGofStatistic()
    >>> sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
    >>> value = statistic.execute_statistic(sample)
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def code():
        parent_code = AbstractGraphNormalityGofStatistic.code()
        short_code = GraphMaxDegreeNormalityGofStatistic.short_code()
        return f"{short_code}_{parent_code}"


class GraphAverageDegreeNormalityGofStatistic(
    AbstractGraphNormalityGofStatistic, GraphAverageDegreeTestStatistic
):
    """Average vertex degree for a normal sample with known variance.

    Parameters
    ----------
    var : float, optional
        Positive finite scalar variance fixed under H0; default 1.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic.
    hypothesis()
        Report fixed null parameters only.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is N(mu,var), with unknown mu and fixed var; hypothesis() reports var.
    Calibrate at this variance; setting the simulation mean to zero is valid.
    Both tails of the statistic defines rejection. Use at least 2
    finite real observations in one dimension. Nonzero dispersion is required.
    Ties are accepted unless a scale or contrast becomes undefined.
    Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
    the standard normal CDF and density, s0=std(x,ddof=0), and
    s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

    Connect i and j when abs(x_i-x_j) < (max(x)-min(x))/(10*m2),
    where m2=mean((x-mean(x))**2). Return the mean vertex degree.
    This local construction is translation invariant but not scale invariant.
    var selects the null law for calibration; it is not inserted in the radius.
    A primary source for this exact normality test was not located. No
    published null law or omnibus power claim is made. Extreme scales that
    make the radius or variance unrepresentable raise ValueError.

    Examples
    --------
    >>> statistic = GraphAverageDegreeNormalityGofStatistic()
    >>> sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
    >>> value = statistic.execute_statistic(sample)
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def code():
        parent_code = AbstractGraphNormalityGofStatistic.code()
        short_code = GraphAverageDegreeNormalityGofStatistic.short_code()
        return f"{short_code}_{parent_code}"


class GraphConnectedComponentsNormalityGofStatistic(
    AbstractGraphNormalityGofStatistic, GraphConnectedComponentsTestStatistic
):
    """Number of connected components for a normal sample with known variance.

    Parameters
    ----------
    var : float, optional
        Positive finite scalar variance fixed under H0; default 1.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic.
    hypothesis()
        Report fixed null parameters only.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is N(mu,var), with unknown mu and fixed var; hypothesis() reports var.
    Calibrate at this variance; setting the simulation mean to zero is valid.
    Both tails of the statistic defines rejection. Use at least 2
    finite real observations in one dimension. Nonzero dispersion is required.
    Ties are accepted unless a scale or contrast becomes undefined.
    Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
    the standard normal CDF and density, s0=std(x,ddof=0), and
    s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

    Connect i and j when abs(x_i-x_j) < (max(x)-min(x))/(10*m2),
    where m2=mean((x-mean(x))**2). Return the number of connected components.
    This local construction is translation invariant but not scale invariant.
    var selects the null law for calibration; it is not inserted in the radius.
    A primary source for this exact normality test was not located. No
    published null law or omnibus power claim is made. Extreme scales that
    make the radius or variance unrepresentable raise ValueError.

    Examples
    --------
    >>> statistic = GraphConnectedComponentsNormalityGofStatistic()
    >>> sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
    >>> value = statistic.execute_statistic(sample)
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def code():
        parent_code = AbstractGraphNormalityGofStatistic.code()
        short_code = GraphConnectedComponentsNormalityGofStatistic.short_code()
        return f"{short_code}_{parent_code}"


class GraphCliqueNumberNormalityGofStatistic(
    AbstractGraphNormalityGofStatistic, GraphCliqueNumberTestStatistic
):
    """Clique-number statistic for a normal sample with known variance.

    Parameters
    ----------
    var : float, optional
        Positive finite scalar variance fixed under H0; default 1.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic.
    hypothesis()
        Report fixed null parameters only.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is N(mu,var), with unknown mu and fixed var; hypothesis() reports var.
    Calibrate at this variance; setting the simulation mean to zero is valid.
    Both tails of the statistic defines rejection. Use at least 2
    finite real observations in one dimension. Nonzero dispersion is required.
    Ties are accepted unless a scale or contrast becomes undefined.
    Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
    the standard normal CDF and density, s0=std(x,ddof=0), and
    s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

    Connect i and j when abs(x_i-x_j) < (max(x)-min(x))/(10*m2),
    where m2=mean((x-mean(x))**2). Return the largest clique size
    (using sorted intervals).
    This local construction is translation invariant but not scale invariant.
    var selects the null law for calibration; it is not inserted in the radius.
    A primary source for this exact normality test was not located. No
    published null law or omnibus power claim is made. Extreme scales that
    make the radius or variance unrepresentable raise ValueError.

    Examples
    --------
    >>> statistic = GraphCliqueNumberNormalityGofStatistic()
    >>> sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
    >>> value = statistic.execute_statistic(sample)
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def code():
        parent_code = AbstractGraphNormalityGofStatistic.code()
        short_code = GraphCliqueNumberNormalityGofStatistic.short_code()
        return f"{short_code}_{parent_code}"


class GraphIndependenceNumberNormalityGofStatistic(
    AbstractGraphNormalityGofStatistic, GraphIndependenceNumberTestStatistic
):
    """Independence-number statistic for a normal sample with known variance.

    Parameters
    ----------
    var : float, optional
        Positive finite scalar variance fixed under H0; default 1.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic.
    hypothesis()
        Report fixed null parameters only.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is N(mu,var), with unknown mu and fixed var; hypothesis() reports var.
    Calibrate at this variance; setting the simulation mean to zero is valid.
    Both tails of the statistic defines rejection. Use at least 2
    finite real observations in one dimension. Nonzero dispersion is required.
    Ties are accepted unless a scale or contrast becomes undefined.
    Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
    the standard normal CDF and density, s0=std(x,ddof=0), and
    s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

    Connect i and j when abs(x_i-x_j) < (max(x)-min(x))/(10*m2),
    where m2=mean((x-mean(x))**2). Return the largest independent set
    size (using a sorted greedy scan).
    This local construction is translation invariant but not scale invariant.
    var selects the null law for calibration; it is not inserted in the radius.
    A primary source for this exact normality test was not located. No
    published null law or omnibus power claim is made. Extreme scales that
    make the radius or variance unrepresentable raise ValueError.

    Examples
    --------
    >>> statistic = GraphIndependenceNumberNormalityGofStatistic()
    >>> sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
    >>> value = statistic.execute_statistic(sample)
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def code():
        parent_code = AbstractGraphNormalityGofStatistic.code()
        short_code = GraphIndependenceNumberNormalityGofStatistic.short_code()
        return f"{short_code}_{parent_code}"
