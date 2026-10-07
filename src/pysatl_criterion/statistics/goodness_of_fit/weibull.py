"""Specified exponentiated-Weibull and fitted ordinary-Weibull statistics.

Fixed-CDF classes use F(x)=(1-exp(-x**k))**a, with location 0 and scale 1.
Composite classes use F(x)=1-exp(-(x/eta)**k), location 0, eta and k unknown.
These are different hypotheses. No class mutates observations or stores fits.
"""

from abc import ABC
from numbers import Integral, Real

import numpy as np
from scipy.optimize import brentq
from scipy.special import gamma, gammaln, logsumexp, zeta
from scipy.stats import gumbel_l
from typing_extensions import override

from pysatl_criterion import DistributionType
from pysatl_criterion.core.distributions.continues.weibull import generate_weibull_cdf
from pysatl_criterion.statistics import AbstractGoodnessOfFitStatistic
from pysatl_criterion.statistics.alternative import (
    AlternativeType,
    LeftAlternative,
    RightAlternative,
    TwoSidedAlternative,
)
from pysatl_criterion.statistics.goodness_of_fit.common import (
    ADStatistic,
    Chi2Statistic,
    CrammerVonMisesStatistic,
    KSStatistic,
    LillieforsTest,
    MinToshiyukiStatistic,
)
from pysatl_criterion.statistics.hypothesis import GoodnessOfFitHypothesis


def _scalar(value, name, *, positive=False):
    message = f"{name}: expected finite {'positive ' if positive else ''}scalars"
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(message)  # noqa: TRY004 - shared validation contract uses ValueError
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError(message) from None
    if not np.isfinite(result) or (positive and result <= 0):
        raise ValueError(message)
    return result


def _sample(rvs, *, min_size=1, positive=False, nonconstant=False):
    if np.iscomplexobj(rvs):
        raise ValueError("Sample must be real")
    x = np.asarray(rvs, dtype=float)
    if x.ndim != 1 or x.size < min_size:
        raise ValueError(f"Sample must be one-dimensional with at least {min_size} observations")
    if not np.all(np.isfinite(x)) or np.any(x < 0) or (positive and np.any(x == 0)):
        raise ValueError(
            "Expected a finite positive sample"
            if positive
            else "Sample must contain finite nonnegative observations"
        )
    if nonconstant and np.ptp(x) == 0:
        raise ValueError("A nonconstant sample is required")
    return x


def _logs(rvs, min_size=2):
    x = _sample(rvs, min_size=min_size, positive=True, nonconstant=True)
    y = np.sort(np.log(x))
    if np.ptp(y) == 0:
        raise ValueError("Logarithms are indistinguishable at float64 precision")
    return y


def _fit_logs(y, counts=None):
    """Profile likelihood root in a rescaled log domain; optional record counts."""
    spread = np.ptp(y)
    v = (y - y[-1]) / spread
    weights = np.ones(y.size) if counts is None else np.asarray(counts, dtype=float)
    log_weights = np.log(weights)

    def score(t):
        w = np.exp(log_weights + t * v - logsumexp(log_weights + t * v))
        return np.dot(w, v) - np.mean(v) - 1 / t

    upper = 1.0
    while score(upper) <= 0:
        upper *= 2
        if not np.isfinite(upper):
            raise ValueError("Weibull likelihood root cannot be represented")
    t = brentq(score, 1e-12, upper, xtol=1e-14)
    log_mean_power = logsumexp(log_weights + t * v) - np.log(y.size)
    standardized = t * v - log_mean_power
    shape = t / spread
    log_scale = y[-1] + spread * log_mean_power / t
    if not np.all(np.isfinite(standardized)) or not np.isfinite(shape):
        raise ValueError("Weibull fit cannot be represented")
    return standardized, shape, log_scale


def _fitted(rvs):
    return _fit_logs(_logs(rvs))[0]


def _log_probabilities(z, a=1):
    """Log probabilities of (1-exp(-exp(z)))**a, retaining both tails."""
    z = np.asarray(z)
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        t = np.exp(z)
        log_cdf = np.empty_like(z)
        small = z < -36
        large = t > 36
        middle = ~(small | large)
        log_cdf[small] = a * z[small]
        log_cdf[middle] = a * np.log(-np.expm1(-t[middle]))
        log_cdf[large] = -np.exp(np.log(a) - t[large])
        log_sf = np.log(-np.expm1(log_cdf))
        upper_tail = large & (np.log(a) - t < -36)
        log_sf[upper_tail] = np.log(a) - t[upper_tail]
    return log_cdf, log_sf


def _weighted_edf(log_cdf, log_sf):
    u = np.exp(log_cdf)
    n = len(u)
    i = np.arange(1, n + 1)
    d = np.maximum(i / n - u, u - (i - 1) / n)
    with np.errstate(over="ignore"):
        return float(np.sum(np.exp(np.log(d) - (log_cdf + log_sf) / 2)) / np.sqrt(n))


def _correlation_squared(y, scores):
    y = y - np.mean(y)
    scores = scores - np.mean(scores)
    y = y / np.linalg.norm(y)
    scores = scores / np.linalg.norm(scores)
    return float(np.clip(np.dot(y, scores) ** 2, 0, 1))


def _moment_components(rvs):
    y = -_logs(rvs)
    y = (y - np.mean(y)) / np.std(y)
    return len(y), np.mean(y**3), np.mean(y**4)


def _moment_null():
    """Delta-method covariance of empirical standardized third/fourth moments."""
    import math

    # Central moments of a standardized maximum Gumbel, from its cumulants.
    cumulants = np.zeros(9)
    for j in range(2, 9):
        cumulants[j] = math.factorial(j - 1) * zeta(j, 1) / (np.pi / np.sqrt(6)) ** j
    moments = np.zeros(9)
    moments[0] = 1
    for j in range(2, 9):
        moments[j] = sum(
            math.comb(j - 1, k - 1) * cumulants[k] * moments[j - k] for k in range(2, j + 1)
        )
    g, b = moments[3:5]
    # Influence functions include estimating the mean and variance.
    p3 = np.array([g / 2, -3, -1.5 * g, 1])
    p4 = np.array([b, -4 * g, -2 * b, 0, 1])

    def covariance(p, q):
        product = np.convolve(p, q)
        return np.dot(product, moments[: len(product)])

    return g, b, covariance(p3, p3), covariance(p3, p4), covariance(p4, p4)


class AbstractWeibullGofStatistic(AbstractGoodnessOfFitStatistic, ABC):
    """Base for a specified exponentiated Weibull with unit scale and zero location."""

    def __init__(self, a=1, k=1):
        self.a = _scalar(a, "a", positive=True)
        self.k = _scalar(k, "k", positive=True)

    def hypothesis(self):
        return GoodnessOfFitHypothesis({"a": self.a, "k": self.k})

    @staticmethod
    def distribution():
        return DistributionType.WEIBULL

    @staticmethod
    def code():
        return f"WEIBULL_{AbstractGoodnessOfFitStatistic.code()}"

    def _validate_storage_calibration(self):
        raise ValueError(
            "Weibull statistics were revised: regenerate calibration; "
            "unversioned stored critical values are not supported"
        )


class _CompositeWeibullStatistic(AbstractWeibullGofStatistic, ABC):
    """Ordinary Weibull with fixed location zero and unknown scale and shape."""

    def __init__(self):
        pass

    def hypothesis(self):
        # a=1 selects ordinary Weibull; k and scale are deliberately omitted.
        return GoodnessOfFitHypothesis({"a": 1, "loc": 0})


class MinToshiyukiWeibullGofStatistic(AbstractWeibullGofStatistic, MinToshiyukiStatistic):
    """Weighted EDF distance for a specified exponentiated Weibull.

    Parameters
    ----------
    a, k : float, optional
        Fixed positive finite shape parameters, both default to 1.
        The parameter a is an exponent, not a scale.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar statistic, fitting parameters internally when needed.
    hypothesis()
        Return the fixed parameters; omitted parameters are unknown.
    alternative()
        Return the critical-region tail.

    Notes
    -----
    F(x)=(1-exp(-x**k))**a on x >= 0, with fixed a, k, scale 1
    and location 0. Put u_i=F(x_(i)), i=1,...,n.

    L = sum(max(i/n-u_i, u_i-(i-1)/n)/sqrt(u_i*(1-u_i)))/sqrt(n).

    Reject for right tail values.
    Calibrate using the specified null and the same sample size.
    Previously stored unversioned Weibull calibrations must be regenerated.
    The historical name uses the given names of Liao and Shimokawa.
    This fixed-CDF variant is not their fitted-parameter procedure.
    A zero observation gives positive infinity. Log probabilities preserve
    finite penalties when a CDF rounds to one; overflow may still give infinity.
    A primary source establishing this exact implemented formula was not
    verified in the literature search. Treat it as the specified local
    discrepancy; no named-test tables or chi-square limit are asserted.

    Examples
    --------
    >>> test = MinToshiyukiWeibullGofStatistic()
    >>> value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
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

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "MT_WEIBULL_{parent_code}".
        """
        short_code = MinToshiyukiWeibullGofStatistic.short_code()
        return f"{short_code}_{AbstractWeibullGofStatistic.code()}"

    def alternative(self):
        return RightAlternative()

    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like
            One-dimensional finite nonnegative observations, n >= 1.
            Ties and constant samples are allowed.
        **kwargs : dict
            Unused common-interface keywords.

        Returns
        -------
        statistic : float
            Scalar discrepancy. Infinite boundary penalties are preserved.

        Raises
        ------
        ValueError
            Invalid sample, unsupported settings, or unrepresentable fit.

        Notes
        -----
        The input is never changed and estimated parameters are not retained.
        """
        x = np.sort(_sample(rvs))
        with np.errstate(divide="ignore", over="ignore"):
            z = self.k * np.log(x)
        return _weighted_edf(*_log_probabilities(z, self.a))


class Chi2PearsonWeibullGofStatistic(AbstractWeibullGofStatistic, Chi2Statistic):
    """Pearson counts statistic in equal-probability bins.

    Parameters
    ----------
    a, k : float, optional
        Fixed positive finite shape parameters, both default to 1.
        The parameter a is an exponent, not a scale.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar statistic, fitting parameters internally when needed.
    hypothesis()
        Return the fixed parameters; omitted parameters are unknown.
    alternative()
        Return the critical-region tail.

    Notes
    -----
    F(x)=(1-exp(-x**k))**a on x >= 0, with fixed a, k, scale 1
    and location 0. Put u_i=F(x_(i)), i=1,...,n.

    B=ceil(sqrt(n)); T=sum((O_j-n/B)**2/(n/B)), j=1,...,B.
    Bins in CDF space cover [0,1], including both tails.

    Reject for right tail values.
    Calibrate using the specified null and the same sample size.
    Previously stored unversioned Weibull calibrations must be regenerated.
    The source gives the general count statistic, applied here to known
    cell probabilities. For small expected counts use simulation, not
    chi-square tables; the number of bins depends on sample size.
    For n=1 there is one bin and the statistic is identically zero.

    References
    ----------
    .. [1] K. Pearson (1900), On the criterion that a given system of deviations
       from the probable in the case of a correlated system of variables is
       such that it can be reasonably supposed to have arisen from random
       sampling. https://doi.org/10.1080/14786440009463897

    Examples
    --------
    >>> test = Chi2PearsonWeibullGofStatistic()
    >>> value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "CHI2_PEARSON".
        """
        return "CHI2_PEARSON"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "CHI2_PEARSON_WEIBULL_{parent_code}".
        """
        short_code = Chi2PearsonWeibullGofStatistic.short_code()
        return f"{short_code}_{AbstractWeibullGofStatistic.code()}"

    def alternative(self):
        return RightAlternative()

    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like
            One-dimensional finite nonnegative observations, n >= 1.
            Ties and constant samples are allowed.
        **kwargs : dict
            Unused common-interface keywords.

        Returns
        -------
        statistic : float
            Scalar discrepancy. Infinite boundary penalties are preserved.

        Raises
        ------
        ValueError
            Invalid sample, unsupported settings, or unrepresentable fit.

        Notes
        -----
        The input is never changed and estimated parameters are not retained.
        """
        x = _sample(rvs)
        n = len(x)
        bins = int(np.ceil(np.sqrt(n)))
        u = generate_weibull_cdf(x, a=self.a, k=self.k)
        observed, _ = np.histogram(u, bins=np.linspace(0, 1, bins + 1))
        return float(self.do_execute_statistic(observed, np.full(bins, n / bins), 1))


class LillieforsWeibullGofStatistic(_CompositeWeibullStatistic, LillieforsTest):
    """KS distance to an ordinary Weibull fitted by maximum likelihood.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar statistic, fitting parameters internally when needed.
    hypothesis()
        Return the fixed parameters; omitted parameters are unknown.
    alternative()
        Return the critical-region tail.

    Notes
    -----
    F(x)=1-exp(-(x/eta)**k), x > 0, with location fixed at zero
    and unknown positive scale eta and shape k. No constructor arguments.
    Logarithmic location and scale are eliminated within every call.

    D=max(max(i/n-u_i), max(u_i-(i-1)/n)), where u_i=G(z_i),
    G(z)=1-exp(-exp(z)), z_i=k_hat*(log(x_(i))-log(eta_hat)).
    MLE solves the ordinary two-parameter Weibull profile score.

    Reject for right tail values.
    Calibrate with repeated execute_statistic calls on ordinary Weibull
    samples. Affine invariance in log(x) permits eta=k=1 for simulation.
    Ordinary fixed-parameter KS tables do not apply to fitted CDFs.
    Previously stored unversioned Weibull calibrations must be regenerated.
    Lilliefors-type describes the fitted KS procedure; the cited paper
    compares fitted KS for this model, rather than introducing a criterion
    with this class name. There is no preliminary fit call.

    References
    ----------
    .. [1] M. Liao and T. Shimokawa (1999), A new goodness-of-fit test for type-I
       extreme-value and 2-parameter Weibull distributions with estimated
       parameters. https://doi.org/10.1080/00949659908811965

    Examples
    --------
    >>> test = LillieforsWeibullGofStatistic()
    >>> value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "LILLIE".
        """
        return "LILLIE"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "LILLIE_WEIBULL_{parent_code}".
        """
        short_code = LillieforsWeibullGofStatistic.short_code()
        return f"{short_code}_{AbstractWeibullGofStatistic.code()}"

    def alternative(self):
        return RightAlternative()

    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like
            One-dimensional finite strictly positive observations, n >= 2.
            Constant samples are invalid; ties are allowed unless stated below.
        **kwargs : dict
            Unused common-interface keywords.

        Returns
        -------
        statistic : float
            Scalar discrepancy. Infinite boundary penalties are preserved.

        Raises
        ------
        ValueError
            Invalid sample, unsupported settings, or unrepresentable fit.

        Notes
        -----
        The input is never changed and estimated parameters are not retained.
        """
        z = _fitted(rvs)
        return float(self.do_execute_statistic(z, gumbel_l.cdf(z)))


class CrammerVonMisesWeibullGofStatistic(AbstractWeibullGofStatistic, CrammerVonMisesStatistic):
    """Cramer-von Mises distance for a specified exponentiated Weibull.

    Parameters
    ----------
    a, k : float, optional
        Fixed positive finite shape parameters, both default to 1.
        The parameter a is an exponent, not a scale.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar statistic, fitting parameters internally when needed.
    hypothesis()
        Return the fixed parameters; omitted parameters are unknown.
    alternative()
        Return the critical-region tail.

    Notes
    -----
    F(x)=(1-exp(-x**k))**a on x >= 0, with fixed a, k, scale 1
    and location 0. Put u_i=F(x_(i)), i=1,...,n.

    W2=1/(12*n)+sum((u_i-(2*i-1)/(2*n))**2).

    Reject for right tail values.
    Calibrate using the specified null and the same sample size.
    Previously stored unversioned Weibull calibrations must be regenerated.
    The reference treats the general EDF functional; here the known CDF
    transforms the null to uniformity.

    References
    ----------
    .. [1] T. W. Anderson and D. A. Darling (1952), Asymptotic theory of certain
       "goodness of fit" criteria based on stochastic processes.
       https://doi.org/10.1214/aoms/1177729437

    Examples
    --------
    >>> test = CrammerVonMisesWeibullGofStatistic()
    >>> value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
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

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "CVM_WEIBULL_{parent_code}".
        """
        short_code = CrammerVonMisesWeibullGofStatistic.short_code()
        return f"{short_code}_{AbstractWeibullGofStatistic.code()}"

    def alternative(self):
        return RightAlternative()

    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like
            One-dimensional finite nonnegative observations, n >= 1.
            Ties and constant samples are allowed.
        **kwargs : dict
            Unused common-interface keywords.

        Returns
        -------
        statistic : float
            Scalar discrepancy. Infinite boundary penalties are preserved.

        Raises
        ------
        ValueError
            Invalid sample, unsupported settings, or unrepresentable fit.

        Notes
        -----
        The input is never changed and estimated parameters are not retained.
        """
        x = np.sort(_sample(rvs))
        return float(self.do_execute_statistic(x, generate_weibull_cdf(x, a=self.a, k=self.k)))


class AndersonDarlingWeibullGofStatistic(_CompositeWeibullStatistic, ADStatistic):
    """Anderson-Darling distance to a maximum-likelihood fitted ordinary Weibull.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar statistic, fitting parameters internally when needed.
    hypothesis()
        Return the fixed parameters; omitted parameters are unknown.
    alternative()
        Return the critical-region tail.

    Notes
    -----
    F(x)=1-exp(-(x/eta)**k), x > 0, with location fixed at zero
    and unknown positive scale eta and shape k. No constructor arguments.
    Logarithmic location and scale are eliminated within every call.

    A2=-n-sum((2*i-1)*(log(u_i)+log(1-u_(n+1-i))))/n,
    where u_i=G(z_i) and z_i are MLE-standardized ordered log observations.
    No finite-sample correction factor is applied.

    Reject for right tail values.
    Calibrate with repeated execute_statistic calls on ordinary Weibull
    samples. Affine invariance in log(x) permits eta=k=1 for simulation.
    Ordinary fixed-parameter KS tables do not apply to fitted CDFs.
    Previously stored unversioned Weibull calibrations must be regenerated.
    The cited model-specific comparison includes fitted Anderson-Darling.
    Stable log-CDF and log-survival functions retain small tail probabilities;
    exp(z) beyond float64 range gives numerical infinity.

    References
    ----------
    .. [1] M. Liao and T. Shimokawa (1999), A new goodness-of-fit test for type-I
       extreme-value and 2-parameter Weibull distributions with estimated
       parameters. https://doi.org/10.1080/00949659908811965

    Examples
    --------
    >>> test = AndersonDarlingWeibullGofStatistic()
    >>> value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
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

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "AD_WEIBULL_{parent_code}".
        """
        short_code = AndersonDarlingWeibullGofStatistic.short_code()
        return f"{short_code}_{AbstractWeibullGofStatistic.code()}"

    def alternative(self):
        return RightAlternative()

    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like
            One-dimensional finite strictly positive observations, n >= 2.
            Constant samples are invalid; ties are allowed unless stated below.
        **kwargs : dict
            Unused common-interface keywords.

        Returns
        -------
        statistic : float
            Scalar discrepancy. Infinite boundary penalties are preserved.

        Raises
        ------
        ValueError
            Invalid sample, unsupported settings, or unrepresentable fit.

        Notes
        -----
        The input is never changed and estimated parameters are not retained.
        """
        z = _fitted(rvs)
        log_cdf, log_sf = _log_probabilities(z)
        return float(self.do_execute_statistic(z, log_cdf=log_cdf, log_sf=log_sf))


class KolmogorovSmirnovWeibullGofStatistic(AbstractWeibullGofStatistic, KSStatistic):
    """Kolmogorov-Smirnov distance for a specified exponentiated Weibull.

    Parameters
    ----------
    a, k : float, optional
        Fixed positive finite shape parameters, default to 1 and 5 respectively.
        The parameter a is an exponent, not a scale.

    alternative_type : AlternativeType, optional
        TWO_TAILED gives D; RIGHT gives D+; LEFT gives D-.
    mode : str, optional
        Compatibility setting, default "auto"; no effect on the statistic.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar statistic, fitting parameters internally when needed.
    hypothesis()
        Return the fixed parameters; omitted parameters are unknown.
    alternative()
        Return the critical-region tail.

    Notes
    -----
    F(x)=(1-exp(-x**k))**a on x >= 0, with fixed a, k, scale 1
    and location 0. Put u_i=F(x_(i)), i=1,...,n.

    D+=max(i/n-u_i); D-=max(u_i-(i-1)/n); D=max(D+,D-).
    Every CDF-deviation direction has a right-tail rejection region.

    Reject for right tail values.
    Calibrate using the specified null and the same sample size.
    Previously stored unversioned Weibull calibrations must be regenerated.
    The reference concerns general continuous CDFs, applied here via F.

    References
    ----------
    .. [1] N. Smirnov (1948), Table for estimating the goodness of fit of empirical
       distributions. https://doi.org/10.1214/aoms/1177730256

    Examples
    --------
    >>> test = KolmogorovSmirnovWeibullGofStatistic()
    >>> value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
    >>> bool(np.isfinite(value))
    True
    """

    def __init__(self, alternative_type=AlternativeType.TWO_TAILED, mode="auto", a=1, k=5):
        AbstractWeibullGofStatistic.__init__(self, a, k)
        if alternative_type not in (
            AlternativeType.TWO_TAILED,
            AlternativeType.RIGHT,
            AlternativeType.LEFT,
        ):
            raise ValueError("Invalid alternative_type")
        KSStatistic.__init__(self, alternative_type, mode)

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "KS".
        """
        return "KS"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "KS_WEIBULL_{parent_code}".
        """
        short_code = KolmogorovSmirnovWeibullGofStatistic.short_code()
        return f"{short_code}_{AbstractWeibullGofStatistic.code()}"

    def alternative(self):
        return RightAlternative()

    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like
            One-dimensional finite nonnegative observations, n >= 1.
            Ties and constant samples are allowed.
        **kwargs : dict
            Unused common-interface keywords.

        Returns
        -------
        statistic : float
            Scalar discrepancy. Infinite boundary penalties are preserved.

        Raises
        ------
        ValueError
            Invalid sample, unsupported settings, or unrepresentable fit.

        Notes
        -----
        The input is never changed and estimated parameters are not retained.
        """
        x = np.sort(_sample(rvs))
        return float(self.do_execute_statistic(x, generate_weibull_cdf(x, a=self.a, k=self.k)))


def _sb_components(y):
    n = len(y)
    y = y - np.mean(y)
    w = np.log((n + 1) / (n - np.arange(1, n) + 1))
    wi = np.r_[w, n - np.sum(w)]
    wn = w * (1 + np.log(w)) - 1
    wn = np.r_[wn, 0.4228 * n - np.sum(wn)]
    b = np.dot(0.6079 * wn - 0.2570 * wi, y) / n
    return y, b


class WPPWeibullGofStatistic(_CompositeWeibullStatistic, ABC):
    """Abstract probability-plot family; select a concrete statistic to execute."""

    def alternative(self):
        return RightAlternative()

    @staticmethod
    def MLEst(x):
        """Return ordinary-Weibull MLEs and ascending standardized log data.

        Parameters
        ----------
        x : array_like
            At least two finite positive, nonconstant observations.

        Returns
        -------
        estimates : dict
            ``eta`` (scale), ``beta`` (shape), and ``y`` (standardized logs).
            Internal statistics use log-scale fits to avoid scale overflow.

        Raises
        ------
        ValueError
            Invalid input or a scale not representable as a positive float.
        """
        z, shape, log_scale = _fit_logs(_logs(x))
        with np.errstate(over="ignore", under="ignore"):
            scale = np.exp(log_scale)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Fitted scale cannot be represented")
        return {"eta": float(scale), "beta": float(shape), "y": z}


class SbWeibullGofStatistic(WPPWeibullGofStatistic):
    """Centered log-order linear-estimate ratio (historical SB name).

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar statistic, fitting parameters internally when needed.
    hypothesis()
        Return the fixed parameters; omitted parameters are unknown.
    alternative()
        Return the critical-region tail.

    Notes
    -----
    F(x)=1-exp(-(x/eta)**k), x > 0, with location fixed at zero
    and unknown positive scale eta and shape k. No constructor arguments.
    Logarithmic location and scale are eliminated within every call.

    For centered y_i=log(x_(i))-mean(log(x)), let w_i=log((n+1)/(n-i+1)),
    i<n, w_n=n-sum(w_i). Let v_i=w_i*(1+log(w_i))-1 for i<n and
    v_n=0.4228*n-sum(v_i). Set b=sum((0.6079*v_i-0.2570*w_i)*y_i)/n.
    Return n*b**2/sum(y_i**2). Centering removes rounded-coefficient drift.

    Reject for left tail values.
    Calibrate with repeated execute_statistic calls on ordinary Weibull
    samples. Affine invariance in log(x) permits eta=k=1 for simulation.
    Ordinary fixed-parameter KS tables do not apply to fitted CDFs.
    Previously stored unversioned Weibull calibrations must be regenerated.
    Sb and SB retain the same historical code and formula. The rounded
    linear weights are an approximation, not exact Shapiro-Wilk weights.
    A primary source establishing this exact implemented formula was not
    verified in the literature search. Treat it as the specified local
    discrepancy; no named-test tables or chi-square limit are asserted.

    Examples
    --------
    >>> test = SbWeibullGofStatistic()
    >>> value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "SB".
        """
        return "SB"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "SB_WEIBULL_{parent_code}".
        """
        short_code = SbWeibullGofStatistic.short_code()
        return f"{short_code}_{AbstractWeibullGofStatistic.code()}"

    def alternative(self):
        return LeftAlternative()

    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like
            One-dimensional finite strictly positive observations, n >= 2.
            Constant samples are invalid; ties are allowed unless stated below.
        **kwargs : dict
            Unused common-interface keywords.

        Returns
        -------
        statistic : float
            Scalar discrepancy. Infinite boundary penalties are preserved.

        Raises
        ------
        ValueError
            Invalid sample, unsupported settings, or unrepresentable fit.

        Notes
        -----
        The input is never changed and estimated parameters are not retained.
        """
        y, b = _sb_components(_logs(rvs))
        return float(len(y) * b**2 / np.dot(y, y))


class SBWeibullGofStatistic(WPPWeibullGofStatistic):
    """Centered log-order linear-estimate ratio (historical SB name).

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar statistic, fitting parameters internally when needed.
    hypothesis()
        Return the fixed parameters; omitted parameters are unknown.
    alternative()
        Return the critical-region tail.

    Notes
    -----
    F(x)=1-exp(-(x/eta)**k), x > 0, with location fixed at zero
    and unknown positive scale eta and shape k. No constructor arguments.
    Logarithmic location and scale are eliminated within every call.

    For centered y_i=log(x_(i))-mean(log(x)), let w_i=log((n+1)/(n-i+1)),
    i<n, w_n=n-sum(w_i). Let v_i=w_i*(1+log(w_i))-1 for i<n and
    v_n=0.4228*n-sum(v_i). Set b=sum((0.6079*v_i-0.2570*w_i)*y_i)/n.
    Return n*b**2/sum(y_i**2). Centering removes rounded-coefficient drift.

    Reject for left tail values.
    Calibrate with repeated execute_statistic calls on ordinary Weibull
    samples. Affine invariance in log(x) permits eta=k=1 for simulation.
    Ordinary fixed-parameter KS tables do not apply to fitted CDFs.
    Previously stored unversioned Weibull calibrations must be regenerated.
    Sb and SB retain the same historical code and formula. The rounded
    linear weights are an approximation, not exact Shapiro-Wilk weights.
    A primary source establishing this exact implemented formula was not
    verified in the literature search. Treat it as the specified local
    discrepancy; no named-test tables or chi-square limit are asserted.

    Examples
    --------
    >>> test = SBWeibullGofStatistic()
    >>> value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "SB".
        """
        return "SB"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "SB_WEIBULL_{parent_code}".
        """
        short_code = SBWeibullGofStatistic.short_code()
        return f"{short_code}_{AbstractWeibullGofStatistic.code()}"

    def alternative(self):
        return LeftAlternative()

    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like
            One-dimensional finite strictly positive observations, n >= 2.
            Constant samples are invalid; ties are allowed unless stated below.
        **kwargs : dict
            Unused common-interface keywords.

        Returns
        -------
        statistic : float
            Scalar discrepancy. Infinite boundary penalties are preserved.

        Raises
        ------
        ValueError
            Invalid sample, unsupported settings, or unrepresentable fit.

        Notes
        -----
        The input is never changed and estimated parameters are not retained.
        """
        y, b = _sb_components(_logs(rvs))
        return float(len(y) * b**2 / np.dot(y, y))


class OKWeibullGofStatistic(WPPWeibullGofStatistic):
    """Standardized log-order scale ratio (historical OK name).

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar statistic, fitting parameters internally when needed.
    hypothesis()
        Return the fixed parameters; omitted parameters are unknown.
    alternative()
        Return the critical-region tail.

    Notes
    -----
    F(x)=1-exp(-(x/eta)**k), x > 0, with location fixed at zero
    and unknown positive scale eta and shape k. No constructor arguments.
    Logarithmic location and scale are eliminated within every call.

    For centered y_i=log(x_(i))-mean(log(x)), let w_i=log((n+1)/(n-i+1)),
    i<n, w_n=n-sum(w_i). Let v_i=w_i*(1+log(w_i))-1 for i<n and
    v_n=0.4228*n-sum(v_i). Set b=sum((0.6079*v_i-0.2570*w_i)*y_i)/n.
    Let S=sum((2*i-1-n)*y_i)/(log(2)*(n-1)*n).
    Return (b/S-1-0.13/sqrt(n)+1.18/n)/(0.49/sqrt(n)-0.36/n).

    Reject for both tail values.
    Calibrate with repeated execute_statistic calls on ordinary Weibull
    samples. Affine invariance in log(x) permits eta=k=1 for simulation.
    Ordinary fixed-parameter KS tables do not apply to fitted CDFs.
    Previously stored unversioned Weibull calibrations must be regenerated.
    This signed scale ratio uses both tails; no standard-normal calibration
    is asserted for its empirical finite-sample centering and scaling.
    A primary source establishing this exact implemented formula was not
    verified in the literature search. Treat it as the specified local
    discrepancy; no named-test tables or chi-square limit are asserted.

    Examples
    --------
    >>> test = OKWeibullGofStatistic()
    >>> value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "OK".
        """
        return "OK"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "OK_WEIBULL_{parent_code}".
        """
        short_code = OKWeibullGofStatistic.short_code()
        return f"{short_code}_{AbstractWeibullGofStatistic.code()}"

    def alternative(self):
        return TwoSidedAlternative()

    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like
            One-dimensional finite strictly positive observations, n >= 2.
            Constant samples are invalid; ties are allowed unless stated below.
        **kwargs : dict
            Unused common-interface keywords.

        Returns
        -------
        statistic : float
            Scalar discrepancy. Infinite boundary penalties are preserved.

        Raises
        ------
        ValueError
            Invalid sample, unsupported settings, or unrepresentable fit.

        Notes
        -----
        The input is never changed and estimated parameters are not retained.
        """
        y, b = _sb_components(_logs(rvs))
        n = len(y)
        s = np.dot(2 * np.arange(1, n + 1) - 1 - n, y) / (np.log(2) * (n - 1) * n)
        return float((b / s - 1 - 0.13 / np.sqrt(n) + 1.18 / n) / (0.49 / np.sqrt(n) - 0.36 / n))


class RSBWeibullGofStatistic(_CompositeWeibullStatistic):
    """Log-probability-plot correlation discrepancy (historical RSB name).

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar statistic, fitting parameters internally when needed.
    hypothesis()
        Return the fixed parameters; omitted parameters are unknown.
    alternative()
        Return the critical-region tail.

    Notes
    -----
    F(x)=1-exp(-(x/eta)**k), x > 0, with location fixed at zero
    and unknown positive scale eta and shape k. No constructor arguments.
    Logarithmic location and scale are eliminated within every call.

    T=n*(1-r**2), where r=corr(log(x_(i)), log(-log(1-i/(n+1)))).

    Reject for right tail values.
    Calibrate with repeated execute_statistic calls on ordinary Weibull
    samples. Affine invariance in log(x) permits eta=k=1 for simulation.
    Ordinary fixed-parameter KS tables do not apply to fitted CDFs.
    Previously stored unversioned Weibull calibrations must be regenerated.
    A primary source establishing this exact implemented formula was not
    verified in the literature search. Treat it as the specified local
    discrepancy; no named-test tables or chi-square limit are asserted.

    Examples
    --------
    >>> test = RSBWeibullGofStatistic()
    >>> value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "RSB".
        """
        return "RSB"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "RSB_WEIBULL_{parent_code}".
        """
        short_code = RSBWeibullGofStatistic.short_code()
        return f"{short_code}_{AbstractWeibullGofStatistic.code()}"

    def alternative(self):
        return RightAlternative()

    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like
            One-dimensional finite strictly positive observations, n >= 2.
            Constant samples are invalid; ties are allowed unless stated below.
        **kwargs : dict
            Unused common-interface keywords.

        Returns
        -------
        statistic : float
            Scalar discrepancy. Infinite boundary penalties are preserved.

        Raises
        ------
        ValueError
            Invalid sample, unsupported settings, or unrepresentable fit.

        Notes
        -----
        The input is never changed and estimated parameters are not retained.
        """
        y = _logs(rvs)
        n = len(y)
        scores = np.log(-np.log1p(-np.arange(1, n + 1) / (n + 1)))
        return float(n * (1 - _correlation_squared(y, scores)))


class REJGWeibullGofStatistic(WPPWeibullGofStatistic):
    """Squared log-probability-plot correlation (historical REJG name).

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar statistic, fitting parameters internally when needed.
    hypothesis()
        Return the fixed parameters; omitted parameters are unknown.
    alternative()
        Return the critical-region tail.

    Notes
    -----
    F(x)=1-exp(-(x/eta)**k), x > 0, with location fixed at zero
    and unknown positive scale eta and shape k. No constructor arguments.
    Logarithmic location and scale are eliminated within every call.

    R2=corr(log(x_(i)), log(-log(1-p_i)))**2,
    p_i=(i-0.3175)/(n+0.365). No fitted shape is needed: it cancels.

    Reject for left tail values.
    Calibrate with repeated execute_statistic calls on ordinary Weibull
    samples. Affine invariance in log(x) permits eta=k=1 for simulation.
    Ordinary fixed-parameter KS tables do not apply to fitted CDFs.
    Previously stored unversioned Weibull calibrations must be regenerated.
    The former implementation returned the fourth power of correlation.
    Exact attribution of these plotting positions to Evans-Johnson-Green
    was not verified; use simulation for this explicitly defined variant.
    A primary source establishing this exact implemented formula was not
    verified in the literature search. Treat it as the specified local
    discrepancy; no named-test tables or chi-square limit are asserted.

    Examples
    --------
    >>> test = REJGWeibullGofStatistic()
    >>> value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "REJG".
        """
        return "REJG"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "REJG_WEIBULL_{parent_code}".
        """
        short_code = REJGWeibullGofStatistic.short_code()
        return f"{short_code}_{AbstractWeibullGofStatistic.code()}"

    def alternative(self):
        return LeftAlternative()

    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like
            One-dimensional finite strictly positive observations, n >= 2.
            Constant samples are invalid; ties are allowed unless stated below.
        **kwargs : dict
            Unused common-interface keywords.

        Returns
        -------
        statistic : float
            Scalar discrepancy. Infinite boundary penalties are preserved.

        Raises
        ------
        ValueError
            Invalid sample, unsupported settings, or unrepresentable fit.

        Notes
        -----
        The input is never changed and estimated parameters are not retained.
        """
        y = _logs(rvs)
        n = len(y)
        p = (np.arange(1, n + 1) - 0.3175) / (n + 0.365)
        return _correlation_squared(y, np.log(-np.log1p(-p)))


class SPPWeibullGofStatistic(WPPWeibullGofStatistic):
    """Maximum stabilized probability-plot distance with Weibull MLEs.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar statistic, fitting parameters internally when needed.
    hypothesis()
        Return the fixed parameters; omitted parameters are unknown.
    alternative()
        Return the critical-region tail.

    Notes
    -----
    F(x)=1-exp(-(x/eta)**k), x > 0, with location fixed at zero
    and unknown positive scale eta and shape k. No constructor arguments.
    Logarithmic location and scale are eliminated within every call.

    T=max(abs(2/pi*asin(sqrt((i-0.5)/n))-2/pi*asin(sqrt(G(z_i))))),
    where z_i are ascending MLE-standardized logs, G(z)=1-exp(-exp(z)).

    Reject for right tail values.
    Calibrate with repeated execute_statistic calls on ordinary Weibull
    samples. Affine invariance in log(x) permits eta=k=1 for simulation.
    Ordinary fixed-parameter KS tables do not apply to fitted CDFs.
    Previously stored unversioned Weibull calibrations must be regenerated.
    A primary source establishing this exact implemented formula was not
    verified in the literature search. Treat it as the specified local
    discrepancy; no named-test tables or chi-square limit are asserted.

    Examples
    --------
    >>> test = SPPWeibullGofStatistic()
    >>> value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "SPP".
        """
        return "SPP"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "SPP_WEIBULL_{parent_code}".
        """
        short_code = SPPWeibullGofStatistic.short_code()
        return f"{short_code}_{AbstractWeibullGofStatistic.code()}"

    def alternative(self):
        return RightAlternative()

    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like
            One-dimensional finite strictly positive observations, n >= 2.
            Constant samples are invalid; ties are allowed unless stated below.
        **kwargs : dict
            Unused common-interface keywords.

        Returns
        -------
        statistic : float
            Scalar discrepancy. Infinite boundary penalties are preserved.

        Raises
        ------
        ValueError
            Invalid sample, unsupported settings, or unrepresentable fit.

        Notes
        -----
        The input is never changed and estimated parameters are not retained.
        """
        z = _fitted(rvs)
        r = 2 / np.pi * np.arcsin(np.sqrt((np.arange(1, len(z) + 1) - 0.5) / len(z)))
        s = 2 / np.pi * np.arcsin(np.sqrt(gumbel_l.cdf(z)))
        return float(np.max(np.abs(r - s)))


class ST1WeibullGofStatistic(_CompositeWeibullStatistic):
    """Delta-method moment discrepancy for negative log Weibull observations.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar statistic, fitting parameters internally when needed.
    hypothesis()
        Return the fixed parameters; omitted parameters are unknown.
    alternative()
        Return the critical-region tail.

    Notes
    -----
    F(x)=1-exp(-(x/eta)**k), x > 0, with location fixed at zero
    and unknown positive scale eta and shape k. No constructor arguments.
    Logarithmic location and scale are eliminated within every call.

    Let Z be the standardized maximum Gumbel. Set g=E(Z**3), b=E(Z**4).
    For sample standardized -log(x), let g_n and b_n use divisor n.
    Influence functions are P3(z)=z**3-3*z-1.5*g*z**2+g/2 and
    P4(z)=z**4-4*g*z-2*b*z**2+b. Put V_ij=E(Pi(Z)*Pj(Z)).
    Return n*(g_n-g)**2/V_33.

    Reject for right tail values.
    Calibrate with repeated execute_statistic calls on ordinary Weibull
    samples. Affine invariance in log(x) permits eta=k=1 for simulation.
    Ordinary fixed-parameter KS tables do not apply to fitted CDFs.
    Previously stored unversioned Weibull calibrations must be regenerated.
    These are locally defined moment statistics under the historical
    ST names. Exact cumulants (j-1)!*zeta(j), j>=2, determine moments
    through order eight. The delta method gives an asymptotic chi-square(1)
    limit under the null, not an exact finite-sample law or the published
    smooth-test normalization. Use simulation for finite samples.
    A primary source establishing this exact implemented formula was not
    verified in the literature search. Treat it as the specified local
    discrepancy; no named-test tables or chi-square limit are asserted.

    Examples
    --------
    >>> test = ST1WeibullGofStatistic()
    >>> value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "ST1".
        """
        return "ST1"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "ST1_WEIBULL_{parent_code}".
        """
        short_code = ST1WeibullGofStatistic.short_code()
        return f"{short_code}_{AbstractWeibullGofStatistic.code()}"

    def alternative(self):
        return RightAlternative()

    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like
            One-dimensional finite strictly positive observations, n >= 2.
            Constant samples are invalid; ties are allowed unless stated below.
        **kwargs : dict
            Unused common-interface keywords.

        Returns
        -------
        statistic : float
            Scalar discrepancy. Infinite boundary penalties are preserved.

        Raises
        ------
        ValueError
            Invalid sample, unsupported settings, or unrepresentable fit.

        Notes
        -----
        The input is never changed and estimated parameters are not retained.
        """
        n, skew, _kurt = _moment_components(rvs)
        g, _b, v33, _v34, _v44 = _moment_null()
        return float(n * (skew - g) ** 2 / v33)


class ST2WeibullGofStatistic(_CompositeWeibullStatistic):
    """Delta-method moment discrepancy for negative log Weibull observations.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar statistic, fitting parameters internally when needed.
    hypothesis()
        Return the fixed parameters; omitted parameters are unknown.
    alternative()
        Return the critical-region tail.

    Notes
    -----
    F(x)=1-exp(-(x/eta)**k), x > 0, with location fixed at zero
    and unknown positive scale eta and shape k. No constructor arguments.
    Logarithmic location and scale are eliminated within every call.

    Let Z be the standardized maximum Gumbel. Set g=E(Z**3), b=E(Z**4).
    For sample standardized -log(x), let g_n and b_n use divisor n.
    Influence functions are P3(z)=z**3-3*z-1.5*g*z**2+g/2 and
    P4(z)=z**4-4*g*z-2*b*z**2+b. Put V_ij=E(Pi(Z)*Pj(Z)).
    Return n*((b_n-b)-V_34/V_33*(g_n-g))**2/(V_44-V_34**2/V_33).

    Reject for right tail values.
    Calibrate with repeated execute_statistic calls on ordinary Weibull
    samples. Affine invariance in log(x) permits eta=k=1 for simulation.
    Ordinary fixed-parameter KS tables do not apply to fitted CDFs.
    Previously stored unversioned Weibull calibrations must be regenerated.
    These are locally defined moment statistics under the historical
    ST names. Exact cumulants (j-1)!*zeta(j), j>=2, determine moments
    through order eight. The delta method gives an asymptotic chi-square(1)
    limit under the null, not an exact finite-sample law or the published
    smooth-test normalization. Use simulation for finite samples.
    A primary source establishing this exact implemented formula was not
    verified in the literature search. Treat it as the specified local
    discrepancy; no named-test tables or chi-square limit are asserted.

    Examples
    --------
    >>> test = ST2WeibullGofStatistic()
    >>> value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "ST2".
        """
        return "ST2"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "ST2_WEIBULL_{parent_code}".
        """
        short_code = ST2WeibullGofStatistic.short_code()
        return f"{short_code}_{AbstractWeibullGofStatistic.code()}"

    def alternative(self):
        return RightAlternative()

    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like
            One-dimensional finite strictly positive observations, n >= 2.
            Constant samples are invalid; ties are allowed unless stated below.
        **kwargs : dict
            Unused common-interface keywords.

        Returns
        -------
        statistic : float
            Scalar discrepancy. Infinite boundary penalties are preserved.

        Raises
        ------
        ValueError
            Invalid sample, unsupported settings, or unrepresentable fit.

        Notes
        -----
        The input is never changed and estimated parameters are not retained.
        """
        n, skew, kurt = _moment_components(rvs)
        g, b, v33, v34, v44 = _moment_null()
        return float(n * ((kurt - b) - v34 / v33 * (skew - g)) ** 2 / (v44 - v34**2 / v33))


class NormalizeSpaceWeibullGofStatistic(_CompositeWeibullStatistic, ABC):
    """Abstract complete-sample normalized-spacing family (no censoring API)."""

    @staticmethod
    def GoFNS(t, n, m):
        """
        Compute normalized spacing coefficients.

        :param t: starting index offset.
        :param n: total sample size including censoring.
        :param m: number of observed failures.
        :return: array of normalized spacing coefficients.
        """
        res = np.zeros(m)
        for i in range(m):
            p_r = (t + i) / (n + 1)
            q_r = 1 - p_r
            Q_r = np.log(-np.log(1 - p_r))
            d_Q_r = -1 / ((1 - p_r) * np.log(1 - p_r))
            d2_Q_r = d_Q_r / q_r - d_Q_r * d_Q_r
            d3_Q_r = d2_Q_r / q_r + d_Q_r / (q_r * q_r) - 2 * (d_Q_r * d2_Q_r)
            d4_Q_r = d3_Q_r / q_r + 2 * d2_Q_r / (q_r * q_r)
            d4_Q_r += 2 * d_Q_r / (q_r * q_r * q_r) - 2 * (d2_Q_r * d2_Q_r + d_Q_r * d3_Q_r)
            res[i] = Q_r + (p_r * q_r / (2 * (n + 2))) * d2_Q_r
            res[i] += (
                q_r
                * p_r
                / ((n + 2) * (n + 2))
                * (1 / 3 * (q_r - p_r) * d3_Q_r + 1 / 8 * p_r * q_r * d4_Q_r)
            )
        return res

    def alternative(self):
        return RightAlternative()

    def _spacings(self, rvs):
        y = _logs(rvs, min_size=3)
        means = self.GoFNS(1, len(y), len(y))
        intervals = np.diff(means)
        if not np.all(np.isfinite(intervals)) or np.any(intervals <= 0):
            raise ValueError("Approximate expected log spacings must be positive")
        gaps = np.diff(y) / intervals
        return gaps / np.sum(gaps)


class TikuSinghWeibullGofStatistic(NormalizeSpaceWeibullGofStatistic):
    """Normalized log-spacing statistic (complete observations).

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar statistic, fitting parameters internally when needed.
    hypothesis()
        Return the fixed parameters; omitted parameters are unknown.
    alternative()
        Return the critical-region tail.

    Notes
    -----
    F(x)=1-exp(-(x/eta)**k), x > 0, with location fixed at zero
    and unknown positive scale eta and shape k. No constructor arguments.
    Logarithmic location and scale are eliminated within every call.

    Let y_i=log(x_(i)), mu_i approximate E(log(E_(i))) for unit exponential
    order statistics. GoFNS uses a fourth-order Taylor approximation to
    Q(U_(i)), Q(p)=log(-log(1-p)), p=i/(n+1).
    Let h_i=(y_(i+1)-y_i)/(mu_(i+1)-mu_i), g_i=h_i/sum(h_i).
    T=2*sum((n-1-i)*g_i, i=1,...,n-2)/(n-2).

    Reject for both tail values.
    Calibrate with repeated execute_statistic calls on ordinary Weibull
    samples. Affine invariance in log(x) permits eta=k=1 for simulation.
    Ordinary fixed-parameter KS tables do not apply to fitted CDFs.
    Previously stored unversioned Weibull calibrations must be regenerated.
    The expectation approximation is retained explicitly; it is not an
    exact order-statistic expectation. No censored-data calibration is
    supported. LOS is infinite when a terminal normalized spacing is zero.
    A primary source establishing this exact implemented formula was not
    verified in the literature search. Treat it as the specified local
    discrepancy; no named-test tables or chi-square limit are asserted.

    Examples
    --------
    >>> test = TikuSinghWeibullGofStatistic()
    >>> value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "TS".
        """
        return "TS"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "TS_WEIBULL_{parent_code}".
        """
        short_code = TikuSinghWeibullGofStatistic.short_code()
        return f"{short_code}_{AbstractWeibullGofStatistic.code()}"

    def alternative(self):
        return TwoSidedAlternative()

    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like
            One-dimensional finite strictly positive observations, n >= 3.
            Constant samples are invalid; ties are allowed unless stated below.
        **kwargs : dict
            Unused common-interface keywords.

        Returns
        -------
        statistic : float
            Scalar discrepancy. Infinite boundary penalties are preserved.

        Raises
        ------
        ValueError
            Invalid sample, unsupported settings, or unrepresentable fit.

        Notes
        -----
        The input is never changed and estimated parameters are not retained.
        """
        g = self._spacings(rvs)
        n = len(g) + 1
        return float(2 * np.dot(np.arange(n - 2, 0, -1), g[:-1]) / (n - 2))


class LOSWeibullGofStatistic(NormalizeSpaceWeibullGofStatistic):
    """Normalized log-spacing statistic (complete observations).

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar statistic, fitting parameters internally when needed.
    hypothesis()
        Return the fixed parameters; omitted parameters are unknown.
    alternative()
        Return the critical-region tail.

    Notes
    -----
    F(x)=1-exp(-(x/eta)**k), x > 0, with location fixed at zero
    and unknown positive scale eta and shape k. No constructor arguments.
    Logarithmic location and scale are eliminated within every call.

    Let y_i=log(x_(i)), mu_i approximate E(log(E_(i))) for unit exponential
    order statistics. GoFNS uses a fourth-order Taylor approximation to
    Q(U_(i)), Q(p)=log(-log(1-p)), p=i/(n+1).
    Let h_i=(y_(i+1)-y_i)/(mu_(i+1)-mu_i), g_i=h_i/sum(h_i).
    For z_i=sum(g_j,j<=i), i=1,...,n-2, return the AD functional
    -(n-2)-sum((2*i-1)*(log(z_i)+log(1-z_(n-1-i))))/(n-2).

    Reject for right tail values.
    Calibrate with repeated execute_statistic calls on ordinary Weibull
    samples. Affine invariance in log(x) permits eta=k=1 for simulation.
    Ordinary fixed-parameter KS tables do not apply to fitted CDFs.
    Previously stored unversioned Weibull calibrations must be regenerated.
    The expectation approximation is retained explicitly; it is not an
    exact order-statistic expectation. No censored-data calibration is
    supported. LOS is infinite when a terminal normalized spacing is zero.
    A primary source establishing this exact implemented formula was not
    verified in the literature search. Treat it as the specified local
    discrepancy; no named-test tables or chi-square limit are asserted.

    Examples
    --------
    >>> test = LOSWeibullGofStatistic()
    >>> value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "LOS".
        """
        return "LOS"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "LOS_WEIBULL_{parent_code}".
        """
        short_code = LOSWeibullGofStatistic.short_code()
        return f"{short_code}_{AbstractWeibullGofStatistic.code()}"

    def alternative(self):
        return RightAlternative()

    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like
            One-dimensional finite strictly positive observations, n >= 3.
            Constant samples are invalid; ties are allowed unless stated below.
        **kwargs : dict
            Unused common-interface keywords.

        Returns
        -------
        statistic : float
            Scalar discrepancy. Infinite boundary penalties are preserved.

        Raises
        ------
        ValueError
            Invalid sample, unsupported settings, or unrepresentable fit.

        Notes
        -----
        The input is never changed and estimated parameters are not retained.
        """
        g = self._spacings(rvs)
        z = np.cumsum(g)[:-1]
        sf = np.cumsum(g[::-1])[:-1][::-1]
        if np.any(z == 0) or np.any(sf == 0):
            return float("inf")
        q = len(z)
        return float(-q - np.dot(2 * np.arange(1, q + 1) - 1, np.log(z) + np.log(sf[::-1])) / q)


class MSFWeibullGofStatistic(NormalizeSpaceWeibullGofStatistic):
    """Normalized log-spacing statistic (complete observations).

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar statistic, fitting parameters internally when needed.
    hypothesis()
        Return the fixed parameters; omitted parameters are unknown.
    alternative()
        Return the critical-region tail.

    Notes
    -----
    F(x)=1-exp(-(x/eta)**k), x > 0, with location fixed at zero
    and unknown positive scale eta and shape k. No constructor arguments.
    Logarithmic location and scale are eliminated within every call.

    Let y_i=log(x_(i)), mu_i approximate E(log(E_(i))) for unit exponential
    order statistics. GoFNS uses a fourth-order Taylor approximation to
    Q(U_(i)), Q(p)=log(-log(1-p)), p=i/(n+1).
    Let h_i=(y_(i+1)-y_i)/(mu_(i+1)-mu_i), g_i=h_i/sum(h_i).
    Return sum(g_i, i=floor(n/2)+1,...,n-1).

    Reject for right tail values.
    Calibrate with repeated execute_statistic calls on ordinary Weibull
    samples. Affine invariance in log(x) permits eta=k=1 for simulation.
    Ordinary fixed-parameter KS tables do not apply to fitted CDFs.
    Previously stored unversioned Weibull calibrations must be regenerated.
    The expectation approximation is retained explicitly; it is not an
    exact order-statistic expectation. No censored-data calibration is
    supported. LOS is infinite when a terminal normalized spacing is zero.
    A primary source establishing this exact implemented formula was not
    verified in the literature search. Treat it as the specified local
    discrepancy; no named-test tables or chi-square limit are asserted.

    Examples
    --------
    >>> test = MSFWeibullGofStatistic()
    >>> value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "MSF".
        """
        return "MSF"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "MSF_WEIBULL_{parent_code}".
        """
        short_code = MSFWeibullGofStatistic.short_code()
        return f"{short_code}_{AbstractWeibullGofStatistic.code()}"

    def alternative(self):
        return RightAlternative()

    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like
            One-dimensional finite strictly positive observations, n >= 3.
            Constant samples are invalid; ties are allowed unless stated below.
        **kwargs : dict
            Unused common-interface keywords.

        Returns
        -------
        statistic : float
            Scalar discrepancy. Infinite boundary penalties are preserved.

        Raises
        ------
        ValueError
            Invalid sample, unsupported settings, or unrepresentable fit.

        Notes
        -----
        The input is never changed and estimated parameters are not retained.
        """
        g = self._spacings(rvs)
        return float(np.sum(g[(len(g) + 1) // 2 :]))


class LiaoShimokawaWeibullGofStatistic(_CompositeWeibullStatistic):
    """Liao-Shimokawa weighted EDF statistic with Weibull MLEs.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar statistic, fitting parameters internally when needed.
    hypothesis()
        Return the fixed parameters; omitted parameters are unknown.
    alternative()
        Return the critical-region tail.

    Notes
    -----
    F(x)=1-exp(-(x/eta)**k), x > 0, with location fixed at zero
    and unknown positive scale eta and shape k. No constructor arguments.
    Logarithmic location and scale are eliminated within every call.

    L=sum(max(i/n-u_i,u_i-(i-1)/n)/sqrt(u_i*(1-u_i)))/sqrt(n),
    where u_i=G(z_i) and z_i are MLE-standardized ordered logs.

    Reject for right tail values.
    Calibrate with repeated execute_statistic calls on ordinary Weibull
    samples. Affine invariance in log(x) permits eta=k=1 for simulation.
    Ordinary fixed-parameter KS tables do not apply to fitted CDFs.
    Previously stored unversioned Weibull calibrations must be regenerated.
    This selects the maximum-likelihood variant, not graphical estimates.
    Log probabilities retain tail precision; there is no epsilon clipping.
    Extremely large finite penalties may overflow float64 to infinity.

    References
    ----------
    .. [1] M. Liao and T. Shimokawa (1999), A new goodness-of-fit test for type-I
       extreme-value and 2-parameter Weibull distributions with estimated
       parameters. https://doi.org/10.1080/00949659908811965

    Examples
    --------
    >>> test = LiaoShimokawaWeibullGofStatistic()
    >>> value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "LS".
        """
        return "LS"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "LS_WEIBULL_{parent_code}".
        """
        short_code = LiaoShimokawaWeibullGofStatistic.short_code()
        return f"{short_code}_{AbstractWeibullGofStatistic.code()}"

    def alternative(self):
        return RightAlternative()

    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like
            One-dimensional finite strictly positive observations, n >= 2.
            Constant samples are invalid; ties are allowed unless stated below.
        **kwargs : dict
            Unused common-interface keywords.

        Returns
        -------
        statistic : float
            Scalar discrepancy. Infinite boundary penalties are preserved.

        Raises
        ------
        ValueError
            Invalid sample, unsupported settings, or unrepresentable fit.

        Notes
        -----
        The input is never changed and estimated parameters are not retained.
        """
        return _weighted_edf(*_log_probabilities(_fitted(rvs)))


class WatsonWeibullGofStatistic(CrammerVonMisesWeibullGofStatistic):
    """Watson centered EDF statistic for a specified exponentiated Weibull.

    Parameters
    ----------
    a, k : float, optional
        Fixed positive finite shape parameters, both default to 1.
        The parameter a is an exponent, not a scale.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar statistic, fitting parameters internally when needed.
    hypothesis()
        Return the fixed parameters; omitted parameters are unknown.
    alternative()
        Return the critical-region tail.

    Notes
    -----
    F(x)=(1-exp(-x**k))**a on x >= 0, with fixed a, k, scale 1
    and location 0. Put u_i=F(x_(i)), i=1,...,n.

    U2=W2-n*(mean(u_i)-1/2)**2; W2 is the Cramer-von Mises statistic.

    Reject for right tail values.
    Calibrate using the specified null and the same sample size.
    Previously stored unversioned Weibull calibrations must be regenerated.
    The general centered EDF functional is applied to the known CDF.
    Centering residuals before squaring avoids subtracting two large sums.

    References
    ----------
    .. [1] G. S. Watson (1961), Goodness-of-fit tests on a circle.
       https://doi.org/10.1093/biomet/48.1-2.109

    Examples
    --------
    >>> test = WatsonWeibullGofStatistic()
    >>> value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "W".
        """
        return "W"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "W_WEIBULL_{parent_code}".
        """
        short_code = WatsonWeibullGofStatistic.short_code()
        return f"{short_code}_{AbstractWeibullGofStatistic.code()}"

    def alternative(self):
        return RightAlternative()

    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like
            One-dimensional finite nonnegative observations, n >= 1.
            Ties and constant samples are allowed.
        **kwargs : dict
            Unused common-interface keywords.

        Returns
        -------
        statistic : float
            Scalar discrepancy. Infinite boundary penalties are preserved.

        Raises
        ------
        ValueError
            Invalid sample, unsupported settings, or unrepresentable fit.

        Notes
        -----
        The input is never changed and estimated parameters are not retained.
        """
        x = np.sort(_sample(rvs))
        u = generate_weibull_cdf(x, a=self.a, k=self.k)
        target = (np.arange(1, len(x) + 1) - 0.5) / len(x)
        delta = u - target
        return float(1 / (12 * len(x)) + np.sum((delta - np.mean(delta)) ** 2))


class KullbackLeiblerWeibullGofStatistic(_CompositeWeibullStatistic):
    """Spacing-entropy discrepancy from a fitted log-Weibull density.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar statistic, fitting parameters internally when needed.
    hypothesis()
        Return the fixed parameters; omitted parameters are unknown.
    alternative()
        Return the critical-region tail.

    Notes
    -----
    F(x)=1-exp(-(x/eta)**k), x > 0, with location fixed at zero
    and unknown positive scale eta and shape k. No constructor arguments.
    Logarithmic location and scale are eliminated within every call.

    For MLE-standardized ordered logs z_i, define
    H=mean(log(n*(z_(min(n,i+m))-z_(max(1,i-m)))/(2*m))).
    Return -H-mean(z)+mean(exp(z)), the entropy-based estimate of
    KL divergence to g(z)=exp(z-exp(z)). The finite-sample estimate may
    be negative. Default m=min(floor(sqrt(n)),floor((n-1)/2)), at least 1.

    Reject for right tail values.
    Calibrate with repeated execute_statistic calls on ordinary Weibull
    samples. Affine invariance in log(x) permits eta=k=1 for simulation.
    Ordinary fixed-parameter KS tables do not apply to fitted CDFs.
    Previously stored unversioned Weibull calibrations must be regenerated.
    Zero entropy-window spacings give positive infinity. Ties with
    positive window spacings are allowed. Preserve m in every simulation;
    the generic resolver supports only the default execution settings.
    A primary source establishing this exact implemented formula was not
    verified in the literature search. Treat it as the specified local
    discrepancy; no named-test tables or chi-square limit are asserted.

    Examples
    --------
    >>> test = KullbackLeiblerWeibullGofStatistic()
    >>> value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "KL".
        """
        return "KL"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "KL_WEIBULL_{parent_code}".
        """
        short_code = KullbackLeiblerWeibullGofStatistic.short_code()
        return f"{short_code}_{AbstractWeibullGofStatistic.code()}"

    def alternative(self):
        return RightAlternative()

    def execute_statistic(self, rvs, m=None, **kwargs):
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like
            One-dimensional finite strictly positive observations, n >= 3.
            Constant samples are invalid; ties are allowed unless stated below.
        m : int, optional
            Entropy window, 1 <= m < n/2. Default grows as sqrt(n).
        **kwargs : dict
            Unused common-interface keywords.

        Returns
        -------
        statistic : float
            Scalar discrepancy. Infinite boundary penalties are preserved.

        Raises
        ------
        ValueError
            Invalid sample, unsupported settings, or unrepresentable fit.

        Notes
        -----
        The input is never changed and estimated parameters are not retained.
        """
        z = _fitted(_sample(rvs, min_size=3, positive=True, nonconstant=True))
        n = len(z)
        if m is None:
            m = min(int(np.sqrt(n)), (n - 1) // 2)
        if isinstance(m, (bool, np.bool_)) or not isinstance(m, Integral) or not 1 <= m < n / 2:
            raise ValueError("m must be an integer with 1 <= m < n/2")
        i = np.arange(n)
        gaps = z[np.minimum(n - 1, i + m)] - z[np.maximum(0, i - m)]
        if np.any(gaps == 0):
            return float("inf")
        h = np.mean(np.log(gaps) + np.log(n / (2 * m)))
        return float(-h - np.mean(z) + np.mean(np.exp(z)))


class LaplaceTransformWeibullGofStatistic(_CompositeWeibullStatistic, ABC):
    """Abstract fitted-Weibull Laplace family; choose LT2 or LT3."""

    def alternative(self):
        return RightAlternative()

    def _laplace(self, rvs, m, a, kind):
        if isinstance(m, (bool, np.bool_)) or not isinstance(m, Integral) or m < 1:
            raise ValueError("m must be a positive integer")
        a = _scalar(a, "a")
        z = _fitted(rvs)
        if kind == 2:
            t = np.arange(-m, 0) / m
        else:
            # Equation (23): step 1/m, NOT m points on the interval.
            t = np.arange(int(np.ceil(-2.5 * m)), int(np.floor(0.49 * m)) + 1) / m
        log_empirical = logsumexp(-np.outer(z, t), axis=0) - np.log(len(z))
        log_theoretical = gammaln(1 - t)
        # Evaluate squared transform differences with their weights in log
        # space, avoiding overflow followed by multiplication by zero.
        with np.errstate(over="ignore", under="ignore", divide="ignore"):
            log_difference = np.maximum(log_empirical, log_theoretical) + np.log(
                -np.expm1(-np.abs(log_empirical - log_theoretical))
            )
            at = a * t
            log_weight = np.full_like(at, -np.inf)
            finite = np.isfinite(at)
            log_weight[finite] = at[finite] - np.exp(at[finite])
            terms = np.exp(log_weight + 2 * log_difference)
        return float(len(z) * np.sum(terms))


class LaplaceTransform2WeibullGofStatistic(LaplaceTransformWeibullGofStatistic):
    """Krit LT2 discrete Laplace statistic with maximum-likelihood fits.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar statistic, fitting parameters internally when needed.
    hypothesis()
        Return the fixed parameters; omitted parameters are unknown.
    alternative()
        Return the critical-region tail.

    Notes
    -----
    F(x)=1-exp(-(x/eta)**k), x > 0, with location fixed at zero
    and unknown positive scale eta and shape k. No constructor arguments.
    Logarithmic location and scale are eliminated within every call.

    T=n*sum(exp(a*t-exp(a*t))*(mean(exp(-t*z))-Gamma(1-t))**2).
    Here z are MLE-standardized logs and t=-1,-1+1/m,...,-1/m.
    This is a discrete sum, with no integration-step multiplier.

    Reject for right tail values.
    Calibrate with repeated execute_statistic calls on ordinary Weibull
    samples. Affine invariance in log(x) permits eta=k=1 for simulation.
    Ordinary fixed-parameter KS tables do not apply to fitted CDFs.
    Previously stored unversioned Weibull calibrations must be regenerated.
    Every simulation must preserve m and a. The generic resolver uses
    the default execution settings only. LT3 at m=100 has 300 grid points.
    Transform differences are evaluated in log space; statistics beyond the
    float64 range return positive infinity.

    References
    ----------
    .. [1] M. Krit (2014), Goodness-of-fit tests for the Weibull distribution based
       on the Laplace transform, sections 2, 4, 5.
       https://numdam.org/item/JSFS_2014__155_3_135_0.pdf

    Examples
    --------
    >>> test = LaplaceTransform2WeibullGofStatistic()
    >>> value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "LT2".
        """
        return "LT2"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "LT2_WEIBULL_{parent_code}".
        """
        short_code = LaplaceTransform2WeibullGofStatistic.short_code()
        return f"{short_code}_{AbstractWeibullGofStatistic.code()}"

    def alternative(self):
        return RightAlternative()

    def execute_statistic(self, rvs, m=100, a=-5, **kwargs):
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like
            One-dimensional finite strictly positive observations, n >= 2.
            Constant samples are invalid; ties are allowed unless stated below.
        m : int, optional
            Positive grid resolution, default 100.
        a : float, optional
            Finite weight parameter, default -5; not a distribution parameter.
        **kwargs : dict
            Unused common-interface keywords.

        Returns
        -------
        statistic : float
            Scalar discrepancy. Infinite boundary penalties are preserved.

        Raises
        ------
        ValueError
            Invalid sample, unsupported settings, or unrepresentable fit.

        Notes
        -----
        The input is never changed and estimated parameters are not retained.
        """
        return self._laplace(rvs, m, a, 2)


class LaplaceTransform3WeibullGofStatistic(LaplaceTransformWeibullGofStatistic):
    """Krit LT3 discrete Laplace statistic with maximum-likelihood fits.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar statistic, fitting parameters internally when needed.
    hypothesis()
        Return the fixed parameters; omitted parameters are unknown.
    alternative()
        Return the critical-region tail.

    Notes
    -----
    F(x)=1-exp(-(x/eta)**k), x > 0, with location fixed at zero
    and unknown positive scale eta and shape k. No constructor arguments.
    Logarithmic location and scale are eliminated within every call.

    T=n*sum(exp(a*t-exp(a*t))*(mean(exp(-t*z))-Gamma(1-t))**2).
    Here z are MLE-standardized logs and t=j/m, ceil(-2.5*m)<=j<=floor(0.49*m).
    This is a discrete sum, with no integration-step multiplier.

    Reject for right tail values.
    Calibrate with repeated execute_statistic calls on ordinary Weibull
    samples. Affine invariance in log(x) permits eta=k=1 for simulation.
    Ordinary fixed-parameter KS tables do not apply to fitted CDFs.
    Previously stored unversioned Weibull calibrations must be regenerated.
    Every simulation must preserve m and a. The generic resolver uses
    the default execution settings only. LT3 at m=100 has 300 grid points.
    Transform differences are evaluated in log space; statistics beyond the
    float64 range return positive infinity.

    References
    ----------
    .. [1] M. Krit (2014), Goodness-of-fit tests for the Weibull distribution based
       on the Laplace transform, sections 2, 4, 5.
       https://numdam.org/item/JSFS_2014__155_3_135_0.pdf

    Examples
    --------
    >>> test = LaplaceTransform3WeibullGofStatistic()
    >>> value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "LT3".
        """
        return "LT3"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "LT3_WEIBULL_{parent_code}".
        """
        short_code = LaplaceTransform3WeibullGofStatistic.short_code()
        return f"{short_code}_{AbstractWeibullGofStatistic.code()}"

    def alternative(self):
        return RightAlternative()

    def execute_statistic(self, rvs, m=100, a=-5, **kwargs):
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like
            One-dimensional finite strictly positive observations, n >= 2.
            Constant samples are invalid; ties are allowed unless stated below.
        m : int, optional
            Positive grid resolution, default 100.
        a : float, optional
            Finite weight parameter, default -5; not a distribution parameter.
        **kwargs : dict
            Unused common-interface keywords.

        Returns
        -------
        statistic : float
            Scalar discrepancy. Infinite boundary penalties are preserved.

        Raises
        ------
        ValueError
            Invalid sample, unsupported settings, or unrepresentable fit.

        Notes
        -----
        The input is never changed and estimated parameters are not retained.
        """
        return self._laplace(rvs, m, a, 3)


class CabanaQuirozWeibullGofStatistic(_CompositeWeibullStatistic):
    """Krit maximum-likelihood Cabana-Quiroz CQ* quadratic statistic.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar statistic, fitting parameters internally when needed.
    hypothesis()
        Return the fixed parameters; omitted parameters are unknown.
    alternative()
        Return the critical-region tail.

    Notes
    -----
    F(x)=1-exp(-(x/eta)**k), x > 0, with location fixed at zero
    and unknown positive scale eta and shape k. No constructor arguments.
    Logarithmic location and scale are eliminated within every call.

    v=sqrt(n)*(mean(exp(-s*z))-Gamma(1-s)), s=(-0.1,0.02).
    Return v.T @ inverse(A) @ v, A=[[1.59,0.91],[0.91,0.53]],
    where z are MLE-standardized log observations.

    Reject for right tail values.
    Calibrate with repeated execute_statistic calls on ordinary Weibull
    samples. Affine invariance in log(x) permits eta=k=1 for simulation.
    Ordinary fixed-parameter KS tables do not apply to fitted CDFs.
    Previously stored unversioned Weibull calibrations must be regenerated.
    This uses the specified nonsingular weighting matrix, not an estimated
    covariance matrix. A chi-square reference law is not asserted.
    Transform values beyond the float64 range give numerical infinity.

    References
    ----------
    .. [1] M. Krit (2014), Goodness-of-fit tests for the Weibull distribution based
       on the Laplace transform, sections 2, 4, 5.
       https://numdam.org/item/JSFS_2014__155_3_135_0.pdf

    Examples
    --------
    >>> test = CabanaQuirozWeibullGofStatistic()
    >>> value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "CQ*".
        """
        return "CQ*"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "CQ*_WEIBULL_{parent_code}".
        """
        return (
            f"{CabanaQuirozWeibullGofStatistic.short_code()}_{AbstractWeibullGofStatistic.code()}"
        )

    def alternative(self):
        return RightAlternative()

    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like
            One-dimensional finite strictly positive observations, n >= 2.
            Constant samples are invalid; ties are allowed unless stated below.
        **kwargs : dict
            Unused common-interface keywords.

        Returns
        -------
        statistic : float
            Scalar discrepancy. Infinite boundary penalties are preserved.

        Raises
        ------
        ValueError
            Invalid sample, unsupported settings, or unrepresentable fit.

        Notes
        -----
        The input is never changed and estimated parameters are not retained.
        """
        z = _fitted(rvs)
        s = np.array([-0.1, 0.02])
        with np.errstate(over="ignore"):
            v = np.sqrt(len(z)) * (np.mean(np.exp(-np.outer(z, s)), axis=0) - gamma(1 - s))
        if not np.all(np.isfinite(v)):
            return float("inf")
        # A Cholesky norm keeps the positive quadratic form numerically positive.
        chol = np.linalg.cholesky(np.array([[1.59, 0.91], [0.91, 0.53]]))
        transformed = np.linalg.solve(chol, v)
        with np.errstate(over="ignore"):
            return float(np.dot(transformed, transformed))


class MahdiDoostparastWeibullGofStatistic(_CompositeWeibullStatistic):
    """Doostparast weighted record EDF distance with record-likelihood fitting.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar statistic, fitting parameters internally when needed.
    hypothesis()
        Return the fixed parameters; omitted parameters are unknown.
    alternative()
        Return the critical-region tail.

    Notes
    -----
    F(x)=1-exp(-(x/eta)**k), x > 0, with location fixed at zero
    and unknown positive scale eta and shape k. No constructor arguments.
    Logarithmic location and scale are eliminated within every call.

    The input is a complete sequence in acquisition order. Extract lower
    records R_j and the counts K_j of observations until the next record,
    including the record itself. Order records increasingly, carrying counts.
    Fit eta,k by the record likelihood product f(R_j)*S(R_j)**(K_j-1).
    At ordered records, the survival estimate is the product of
    1-1/sum(K_l,l>=j). Return n*integral((S_hat-S_fit)**2/F_fit dF_fit)
    over the entire support, including the first and last intervals.

    Reject for right tail values.
    Calibration must reproduce record extraction and condition on at least
    two records. The generic unconditional resolver is blocked.
    Previously stored unversioned Weibull calibrations must be regenerated.
    Order matters for full acquisition sequences. At least two records
    are required. Precompressed records require counts; do not sort an
    ordinary sample before calling. Inverse-sampling calibration is external.

    References
    ----------
    .. [1] M. Doostparast (2011), Goodness-of-fit tests for Weibull populations
       on the basis of records, equations (9), (10), (13), (14), (17).
       https://arxiv.org/abs/1110.5509

    Examples
    --------
    >>> test = MahdiDoostparastWeibullGofStatistic()
    >>> value = test.execute_statistic([3.0, 1.2, 2.1, 0.4, 0.8, 0.2])
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "MD".
        """
        return "MD"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "MD_WEIBULL_{parent_code}".
        """
        short_code = MahdiDoostparastWeibullGofStatistic.short_code()
        return f"{short_code}_{AbstractWeibullGofStatistic.code()}"

    def alternative(self):
        return RightAlternative()

    def execute_statistic(self, rvs, record_counts=None, **kwargs):
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like
            One-dimensional finite strictly positive observations, n >= 2.
            Constant samples are invalid; ties are allowed unless stated below.
        record_counts : array_like of int, optional
            If given, rvs must be strictly decreasing lower records and these
            positive counts include each record. Otherwise extract from rvs.
        **kwargs : dict
            Unused common-interface keywords.

        Returns
        -------
        statistic : float
            Scalar discrepancy. Infinite boundary penalties are preserved.

        Raises
        ------
        ValueError
            Invalid sample, unsupported settings, or unrepresentable fit.

        Notes
        -----
        The input is never changed and estimated parameters are not retained.
        """
        x = _sample(rvs, min_size=2, positive=True)
        if record_counts is None:
            indices = np.flatnonzero(np.r_[True, x[1:] < np.minimum.accumulate(x)[:-1]])
            records = x[indices][::-1]
            counts = np.diff(np.r_[indices, len(x)])[::-1].astype(float)
        else:
            raw_counts = np.asarray(record_counts)
            if (
                raw_counts.shape != x.shape
                or raw_counts.dtype.kind not in "iu"
                or np.any(raw_counts <= 0)
                or np.any(np.diff(x) >= 0)
            ):
                raise ValueError(
                    "Records must decrease strictly and counts must be positive integers"
                )
            records, counts = x[::-1], raw_counts[::-1].astype(float)
        if len(records) < 2:
            raise ValueError("At least two lower records are required for fitting")
        y = _logs(records)
        z = _fit_logs(y, counts)[0]
        u = gumbel_l.cdf(z)
        log_u = _log_probabilities(z)[0]
        risk = np.cumsum(counts[::-1])[::-1]
        survival = np.r_[1.0, np.cumprod(1 - 1 / risk)]
        c = survival - 1
        # The first c is zero, so its logarithmic term is exactly zero.
        log_widths = np.diff(np.r_[log_u, 0.0])
        if np.any(~np.isfinite(log_widths)):
            raise ValueError("Record CDF intervals cannot be represented")
        widths = np.diff(np.r_[0.0, u, 1.0])
        value = np.dot(c[1:] ** 2, log_widths) + 2 * np.dot(c, widths) + 0.5
        return float(max(0.0, np.sum(counts) * value))

    def _validate_monte_carlo_calibration(self):
        raise ValueError(
            "Record calibration must reproduce the sampling scheme and "
            "condition on at least two records; use external calibration"
        )
