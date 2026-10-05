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
    if not math.isfinite(mean):
        raise ValueError("mean must be finite")
    if not math.isfinite(var) or var <= 0:
        raise ValueError("var must be positive and finite")


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

    Raises
    ------
    ValueError
        If the mean is nonfinite or the variance is nonpositive or nonfinite.

    Notes
    -----
    The null is ``N(mean, var)`` with both parameters specified independently
    of the observations. ``hypothesis().parameters()`` contains both values.
    The empirical CDF of the unstandardized sample is compared with this
    fixed CDF. Use ``LillieforsNormalityGofStatistic`` when both parameters
    are estimated from the sample. Supply a nonempty, finite sample.

    References
    ----------
    .. [1] Smirnov, N. (1948). Table for Estimating the Goodness of Fit of
       Empirical Distributions. The Annals of Mathematical Statistics,
       19(2), 279-281. https://doi.org/10.1214/aoms/1177730256

    Examples
    --------
    >>> statistic = KolmogorovSmirnovNormalityGofStatistic(mean=3, var=4)
    >>> statistic.hypothesis().parameters()
    {'mean': 3, 'var': 4}
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
        """Compute the statistic for the observed sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional sample of finite observations. Sample-size and
            dispersion requirements are described in the class notes.
        **kwargs : dict, optional
            Unused keyword arguments accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Value of the statistic for a supported sample. This method does
            not return a p-value or a hypothesis-test decision.
        """
        rvs = np.sort(rvs)
        cdf_vals = scipy_stats.norm.cdf(rvs, loc=self.mean, scale=np.sqrt(self.var))
        return KSStatistic.do_execute_statistic(self, rvs, cdf_vals)


"""""
class ChiSquareTest(AbstractNormalityTestStatistic):  # TODO: check test correctness

    @staticmethod
    @override
    def code():
        return 'CHI2' + '_' + super(AbstractNormalityTestStatistic, AbstractNormalityTestStatistic)
        .code()

    @override
    def execute_statistic(self, rvs, **kwargs):
        rvs = np.sort(rvs)

        f_obs = np.asanyarray(rvs)
        f_obs_float = f_obs.astype(np.float64)
        f_exp = pdf_norm(rvs)  # TODO: remove link to ext package
        scipy_stats.chi2_contingency()  # TODO: fix warning!!
        terms = (f_obs_float - f_exp) ** 2 / f_exp
        return terms.sum(axis=0)
""" ""


class AndersonDarlingNormalityGofStatistic(AbstractNormalityGofStatistic, ADStatistic):
    """Anderson-Darling statistic for the normal location-scale family.

    Notes
    -----
    The null allows any finite mean and positive variance;
    ``hypothesis().parameters()`` is empty. Each sample is standardized
    using its mean and standard deviation with ``ddof=1`` before computing
    the tail-weighted empirical-CDF discrepancy ``A_squared``.

    The returned value is the unadjusted statistic, without a finite-sample
    correction or a p-value. Larger values indicate poorer fit, and
    ``alternative()`` returns ``RightAlternative``. Supply at least two
    finite observations with nonzero sample variance. Calibration must
    account for estimating both normal parameters from every sample.

    References
    ----------
    .. [1] Anderson, T. W. and Darling, D. A. (1954). A Test of Goodness
       of Fit. Journal of the American Statistical Association, 49(268),
       765-769. https://doi.org/10.1080/01621459.1954.10501232
    .. [2] Stephens, M. A. (1976). Asymptotic Results for Goodness-of-Fit
       Statistics with Unknown Parameters. The Annals of Statistics,
       4(2), 357-369. https://doi.org/10.1214/aos/1176343411
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
        """Compute the statistic for the observed sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional sample of finite observations. Sample-size and
            dispersion requirements are described in the class notes.
        **kwargs : dict, optional
            Unused keyword arguments accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Value of the statistic for a supported sample. This method does
            not return a p-value or a hypothesis-test decision.
        """
        s = np.std(rvs, ddof=1, axis=0)
        y = np.sort(rvs)
        xbar = np.mean(rvs, axis=0)
        w = (y - xbar) / s
        logcdf = scipy_stats.distributions.norm.logcdf(w)
        logsf = scipy_stats.distributions.norm.logsf(w)
        return super().do_execute_statistic(rvs, log_cdf=logcdf, log_sf=logsf)


class ShapiroWilkNormalityGofStatistic(AbstractNormalityGofStatistic):
    """Shapiro-Wilk ``W`` statistic for the normal location-scale family.

    Notes
    -----
    The null leaves the mean and variance unspecified;
    ``hypothesis().parameters()`` is empty. The statistic is a squared
    weighted sum of ordered observations divided by their centered sum of
    squares, and is invariant to changes of location and scale.

    Order-statistic weights use a normal-quantile approximation with
    polynomial adjustments, following the algorithmic approach in [2]_.
    Smaller ``W`` values indicate departures from normality;
    ``alternative()`` returns ``LeftAlternative``. Supply at least three
    finite observations with nonzero sample variance. No p-value is returned.

    References
    ----------
    .. [1] Shapiro, S. S. and Wilk, M. B. (1965). An analysis of variance
       test for normality (complete samples). Biometrika, 52(3-4), 591-611.
       https://doi.org/10.1093/biomet/52.3-4.591
    .. [2] Royston, P. (1995). Remark AS R94: A Remark on Algorithm AS 181:
       The W-test for Normality. Applied Statistics, 44(4), 547-551.
       https://doi.org/10.2307/2986146
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
    def execute_statistic(self, rvs):
        """Compute the statistic for the observed sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional sample of finite observations. Sample-size and
            dispersion requirements are described in the class notes.

        Returns
        -------
        statistic : float
            Value of the statistic for a supported sample. This method does
            not return a p-value or a hypothesis-test decision.
        """
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

    Raises
    ------
    ValueError
        If the mean is nonfinite or the variance is nonpositive or nonfinite.

    Notes
    -----
    The null is ``N(mean, var)`` and ``hypothesis().parameters()`` contains
    both fixed values. The statistic is ``1/(12*n)`` plus the sum of squared
    differences between ``F_0(x_sorted[i])`` and ``(2*i-1)/(2*n)``, with
    one-based ranks. ``F_0`` uses ``scale=sqrt(var)`` without fitting the data.

    Larger values indicate poorer fit; ``alternative()`` returns
    ``RightAlternative``. Supply a nonempty, finite sample. Estimating the
    constructor parameters from that sample changes the calibration and
    does not turn this class into a composite-normality test.

    References
    ----------
    .. [1] Cramer, H. (1928). On the composition of elementary errors.
       Second paper: Statistical applications. Scandinavian Actuarial
       Journal, 1928(1), 141-180.
       https://doi.org/10.1080/03461238.1928.10416872
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
    def execute_statistic(self, rvs):
        """Compute the statistic for the observed sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional sample of finite observations. Sample-size and
            dispersion requirements are described in the class notes.

        Returns
        -------
        statistic : float
            Value of the statistic for a supported sample. This method does
            not return a p-value or a hypothesis-test decision.
        """
        sorted_rvs = np.sort(np.asarray(rvs))
        cdf_vals = scipy_stats.norm.cdf(sorted_rvs, loc=self.mean, scale=np.sqrt(self.var))

        return CrammerVonMisesStatistic.do_execute_statistic(self, sorted_rvs, cdf_vals)


class LillieforsNormalityGofStatistic(AbstractNormalityGofStatistic, LillieforsTest):
    """Lilliefors statistic for normality with unknown mean and variance.

    Notes
    -----
    The null is the entire normal location-scale family;
    ``hypothesis().parameters()`` is empty. The mean and standard deviation
    with ``ddof=1`` are estimated from each sample. The maximum absolute
    empirical-CDF discrepancy is then evaluated on standardized observations.

    The statistic is invariant to location and scale. It requires a
    nonconstant finite sample of at least two observations. Its null
    distribution differs from that of KS with specified parameters [1]_.
    Larger discrepancies indicate poorer fit. The current shared Lilliefors
    interface reports ``TwoSidedAlternative``; this method returns only
    the discrepancy, without a p-value.

    References
    ----------
    .. [1] Lilliefors, H. W. (1967). On the Kolmogorov-Smirnov Test for
       Normality with Mean and Variance Unknown. Journal of the American
       Statistical Association, 62(318), 399-402.
       https://doi.org/10.1080/01621459.1967.10482916
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
        """Compute the statistic for the observed sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional sample of finite observations. Sample-size and
            dispersion requirements are described in the class notes.
        **kwargs : dict, optional
            Unused keyword arguments accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Value of the statistic for a supported sample. This method does
            not return a p-value or a hypothesis-test decision.
        """
        x = np.asarray(rvs)
        z = (x - x.mean()) / x.std(ddof=1)
        cdf_vals = scipy_stats.norm.cdf(np.sort(z))
        return super(LillieforsTest, self).do_execute_statistic(rvs, cdf_vals)


"""
class DANormalityTest(AbstractNormalityTestStatistic):  # TODO: check for correctness

    @staticmethod
    @override
    def code():
        return 'DA' + '_' + super(AbstractNormalityTestStatistic, AbstractNormalityTestStatistic)
        .code()

    @override
    def execute_statistic(self, rvs, **kwargs):
        x = np.asanyarray(rvs)
        y = np.sort(x)
        n = len(x)

        x_mean = np.mean(x)
        m2 = np.sum((x - x_mean) ** 2) / n
        i = np.arange(1, n + 1)
        c = (n + 1) / 2
        terms = (i - c) * y
        stat = terms.sum() / (n ** 2 * np.sqrt(m2))
        return stat
"""


class JBNormalityGofStatistic(AbstractNormalityGofStatistic):
    """Jarque-Bera statistic based on sample skewness and excess kurtosis.

    Notes
    -----
    The null allows every normal mean and positive variance;
    ``hypothesis().parameters()`` is empty. Centered, standardized moments
    remove location and scale. The implementation returns
    ``n/6 * (skewness**2 + excess_kurtosis**2/4)`` using the biased moment
    estimators in ``scipy.stats.skew`` and ``scipy.stats.kurtosis``.

    Larger values indicate departures from normality;
    ``alternative()`` returns ``RightAlternative``. The input is flattened.
    Supply finite observations with nonzero variance. The asymptotic
    chi-squared calibration described in [1]_ is not a finite-sample
    identity, and this class computes no p-value.

    References
    ----------
    .. [1] Jarque, C. M. and Bera, A. K. (1987). A Test for Normality of
       Observations and Regression Residuals. International Statistical
       Review, 55(2), 163-172. https://doi.org/10.2307/1403192
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
        """Compute the statistic for the observed sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional sample of finite observations. Sample-size and
            dispersion requirements are described in the class notes.
        **kwargs : dict, optional
            Unused keyword arguments accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Value of the statistic for a supported sample. This method does
            not return a p-value or a hypothesis-test decision.
        """
        x = np.asarray(rvs)
        x = x.ravel()
        axis = 0

        n = x.shape[axis]
        if n == 0:
            raise ValueError("At least one observation is required.")

        mu = x.mean(axis=axis, keepdims=True)
        diffx = x - mu
        s = scipy_stats.skew(diffx, axis=axis, _no_deco=True)
        k = scipy_stats.kurtosis(diffx, axis=axis, _no_deco=True)
        statistic = n / 6 * (s**2 + k**2 / 4)
        return statistic


class SkewNormalityGofStatistic(AbstractNormalityGofStatistic):
    """D'Agostino transformed-skewness statistic for the normal family.

    Notes
    -----
    ``hypothesis().parameters()`` is empty: the mean and variance are
    unrestricted. The standardized third central moment is transformed to
    a signed ``Z`` score using the sample-size-dependent approximation
    in [1]_. Location shifts and positive scale changes leave it unchanged.

    At least eight finite observations and nonzero variance are required.
    Positive and negative values correspond to opposite skewness directions.
    The current ``alternative()`` is ``RightAlternative``, so its configured
    rejection direction targets large positive scores. It is not an omnibus
    normality statistic; ``DAPNormalityGofStatistic`` combines squared
    skewness and kurtosis scores. This class returns no p-value.

    References
    ----------
    .. [1] D'Agostino, R. B. (1970). Transformation to normality of the null
       distribution of g1. Biometrika, 57(3), 679-681.
       https://doi.org/10.1093/biomet/57.3.679
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

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
        """Compute the statistic for the observed sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional sample of finite observations. Sample-size and
            dispersion requirements are described in the class notes.
        **kwargs : dict, optional
            Unused keyword arguments accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Value of the statistic for a supported sample. This method does
            not return a p-value or a hypothesis-test decision.
        """
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
        y = np.where(y == 0, 1, y)
        z = delta * np.log(y / alpha + np.sqrt((y / alpha) ** 2 + 1))

        return z


class KurtosisNormalityGofStatistic(AbstractNormalityGofStatistic):
    """Anscombe-Glynn transformed-kurtosis statistic for the normal family.

    Notes
    -----
    The normal mean and variance are unrestricted;
    ``hypothesis().parameters()`` is empty. The implementation transforms
    the standardized fourth central moment to a signed ``Z`` score using
    the finite-sample moment approximations in [1]_.

    At least five finite observations with nonzero variance are required;
    the normal approximation can be unreliable for small samples. Negative
    and positive scores describe opposite kurtosis departures. The current
    ``alternative()`` returns ``RightAlternative`` and therefore targets
    large positive scores. Use ``DAPNormalityGofStatistic`` for the joint
    squared skewness and kurtosis statistic. No p-value is returned.

    References
    ----------
    .. [1] Anscombe, F. J. and Glynn, W. J. (1983). Distribution of the
       kurtosis statistic b2 for normal samples. Biometrika, 70(1), 227-234.
       https://doi.org/10.1093/biomet/70.1.227
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

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
        """Compute the statistic for the observed sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional sample of finite observations. Sample-size and
            dispersion requirements are described in the class notes.
        **kwargs : dict, optional
            Unused keyword arguments accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Value of the statistic for a supported sample. This method does
            not return a p-value or a hypothesis-test decision.
        """
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
        term2 = np.sign(denom) * np.where(
            denom == 0.0, np.nan, np.power((1 - 2.0 / a) / np.abs(denom), 1 / 3.0)
        )
        # if np.any(denom == 0):
        #    msg = ("Test statistic not defined in some cases due to division by "
        #           "zero. Return nan in that case...")
        #    warnings.warn(msg, RuntimeWarning, stacklevel=2)

        z = (term1 - term2) / np.sqrt(2 / (9.0 * a))  # [1]_ Eq. 5

        return z


class DAPNormalityGofStatistic(SkewNormalityGofStatistic, KurtosisNormalityGofStatistic):
    """D'Agostino-Pearson omnibus ``K_squared`` statistic for normality.

    Notes
    -----
    The null is the normal location-scale family, so
    ``hypothesis().parameters()`` is empty. The implementation adds the
    squared transformed skewness and kurtosis scores returned by
    ``SkewNormalityGofStatistic`` and ``KurtosisNormalityGofStatistic``.
    Centered standardized moments eliminate the unknown mean and variance.

    At least eight finite observations with nonzero variance are required.
    Larger scores indicate departures from normality;
    ``alternative()`` returns ``RightAlternative``. The chi-squared
    approximation is asymptotic; this class returns a statistic only.

    References
    ----------
    .. [1] D'Agostino, R. B. and Pearson, E. S. (1973). Tests for departure
       from normality. Empirical results for the distributions of b2 and
       sqrt(b1). Biometrika, 60(3), 613-622.
       https://doi.org/10.1093/biomet/60.3.613
    """

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
        """Compute the statistic for the observed sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional sample of finite observations. Sample-size and
            dispersion requirements are described in the class notes.
        **kwargs : dict, optional
            Unused keyword arguments accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Value of the statistic for a supported sample. This method does
            not return a p-value or a hypothesis-test decision.
        """
        x = np.asanyarray(rvs)
        y = np.sort(x)

        s = self.skew_test(y)
        k = self.kurtosis_test(y)
        k2 = s * s + k * k
        return k2


# https://github.com/puzzle-in-a-mug/normtest
class FilliNormalityGofStatistic(AbstractNormalityGofStatistic):
    """Filliben probability-plot correlation statistic for normality.

    Notes
    -----
    The null leaves normal location and scale unrestricted;
    ``hypothesis().parameters()`` is empty. The statistic is the Pearson
    correlation between sorted observations and normal quantiles obtained
    from Filliben's approximations to uniform order-statistic medians.
    Endpoint plotting positions use the separate formulas in [1]_.

    Correlation eliminates the unknown mean and scale. Smaller values
    indicate departures from normality, and ``alternative()`` returns
    ``LeftAlternative``. Supply finite, nonconstant observations; two
    observations always give a trivial correlation, so use at least three.
    The returned value is a correlation coefficient, not a p-value.

    References
    ----------
    .. [1] Filliben, J. J. (1975). The Probability Plot Correlation
       Coefficient Test for Normality. Technometrics, 17(1), 111-117.
       https://doi.org/10.1080/00401706.1975.10489279
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
        """Compute the statistic for the observed sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional sample of finite observations. Sample-size and
            dispersion requirements are described in the class notes.
        **kwargs : dict, optional
            Unused keyword arguments accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Value of the statistic for a supported sample. This method does
            not return a p-value or a hypothesis-test decision.
        """
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

    Notes
    -----
    The normal mean and variance are unrestricted, and
    ``hypothesis().parameters()`` is empty. Sorted observations are
    correlated with normal quantiles at ``(i - 3/8)/(n + 1/4)`` for
    one-based ranks. This class uses the unweighted plotting positions,
    including when observations are tied.

    Correlation removes location and scale. Smaller coefficients indicate
    departures from normality; ``alternative()`` returns ``LeftAlternative``.
    Supply finite, nonconstant observations. At least three observations
    are needed for a nontrivial correlation-based normality assessment.
    The returned coefficient does not include a p-value.

    References
    ----------
    .. [1] Looney, S. W. and Gulledge, T. R., Jr. (1985). Use of the
       Correlation Coefficient with Normal Probability Plots. The American
       Statistician, 39(1), 75-79.
       https://doi.org/10.1080/00031305.1985.10479395
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
        """Compute the statistic for the observed sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional sample of finite observations. Sample-size and
            dispersion requirements are described in the class notes.
        **kwargs : dict, optional
            Unused keyword arguments accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Value of the statistic for a supported sample. This method does
            not return a p-value or a hypothesis-test decision.
        """
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

    Notes
    -----
    The null allows arbitrary normal mean and positive variance;
    ``hypothesis().parameters()`` is empty. The two constructor arguments
    configure the statistic, not the distribution under the null.

    The statistic is the Pearson correlation of sorted observations with
    their normal scores. Smaller values indicate poorer fit, and
    ``alternative()`` returns ``LeftAlternative``. Supply finite,
    nonconstant data with at least three observations for a nontrivial
    result. Calibration must use the same plotting-position and tie options.

    References
    ----------
    .. [1] Ryan, T. A., Jr. and Joiner, B. L. (1976). Normal Probability
       Plots and Tests for Normality. Technical report, Statistics
       Department, The Pennsylvania State University. Original report:
       https://www.additive-net.de/de/component/jdownloads/send/70-support/236-normal-probability-plots-and-tests-for-normality-thomas-a-ryan-jr-bryan-l-joiner
    """

    @override
    def __init__(self, weighted=False, cte_alpha="3/8"):
        super().__init__()
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
        """Compute the statistic for the observed sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional sample of finite observations. Sample-size and
            dispersion requirements are described in the class notes.
        **kwargs : dict, optional
            Unused keyword arguments accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Value of the statistic for a supported sample. This method does
            not return a p-value or a hypothesis-test decision.
        """
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

    Notes
    -----
    The null leaves mean and variance unrestricted;
    ``hypothesis().parameters()`` is empty. The statistic is the squared
    correlation between sorted observations and approximate expected
    normal order statistics, using ``(i - 3/8)/(n + 1/4)`` plotting
    probabilities. This quantile approximation is discussed in [2]_.

    Location and scale cancel from the ratio. Smaller values indicate
    nonnormality; ``alternative()`` returns ``LeftAlternative``. Supply
    finite, nonconstant data and at least three observations for a
    nontrivial assessment. No p-value is calculated.

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
        """Compute the statistic for the observed sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional sample of finite observations. Sample-size and
            dispersion requirements are described in the class notes.
        **kwargs : dict, optional
            Unused keyword arguments accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Value of the statistic for a supported sample. This method does
            not return a p-value or a hypothesis-test decision.
        """
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

    Notes
    -----
    The null is the whole normal location-scale family;
    ``hypothesis().parameters()`` is empty. The implementation uses the
    sample mean and variance with ``ddof=0`` to remove location and scale.
    Gaussian kernels compare centered observations and all observation
    pairs, yielding the closed-form weighted characteristic-function
    discrepancy with the smoothing constant fixed by this implementation.

    Larger values indicate departures from normality;
    ``alternative()`` returns ``RightAlternative``. Supply at least two
    finite observations with positive sample variance. Pairwise terms
    require quadratic computation in the sample size. No p-value is returned.

    References
    ----------
    .. [1] Epps, T. W. and Pulley, L. B. (1983). A test for normality based
       on the empirical characteristic function. Biometrika, 70(3), 723-726.
       https://doi.org/10.1093/biomet/70.3.723
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
        """Compute the statistic for the observed sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional sample of finite observations. Sample-size and
            dispersion requirements are described in the class notes.
        **kwargs : dict, optional
            Unused keyword arguments accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Value of the statistic for a supported sample. This method does
            not return a p-value or a hypothesis-test decision.
        """
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

    Notes
    -----
    The null is the normal family with unknown mean and variance;
    ``hypothesis().parameters()`` is empty. The statistic combines squared
    deviations of trimmed L-skewness and L-kurtosis, using order-statistic
    weights with trimming level ``t=1`` from the construction in [1]_.
    The ratios remove location and positive scale.

    Supply at least 6 finite, nonconstant observations; estimating the
    fourth trimmed L-moment requires ``n >= 4 + 2*t``. Reference constants
    are selected from three sample-size ranges: up to 25, 26-50, and above
    50. The current ``alternative()`` reports ``TwoSidedAlternative`` for
    this nonnegative quadratic discrepancy. Reference [1]_ supplies the
    trimmed-moment foundation; it does not establish the provenance of
    the numerical constants or the exact omnibus implementation here.

    References
    ----------
    .. [1] Elamir, E. A. H. and Seheult, A. H. (2003). Trimmed L-moments.
       Computational Statistics & Data Analysis, 43(3), 299-314.
       https://doi.org/10.1016/S0167-9473(02)00250-5
    """

    @override
    def alternative(self) -> Alternative:
        return TwoSidedAlternative()

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
        """Compute the statistic for the observed sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional sample of finite observations. Sample-size and
            dispersion requirements are described in the class notes.
        **kwargs : dict, optional
            Unused keyword arguments accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Value of the statistic for a supported sample. This method does
            not return a p-value or a hypothesis-test decision.
        """
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
            tau31 = l31 / l21
            tau41 = l41 / l21
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

    Notes
    -----
    The normal mean and variance are unrestricted;
    ``hypothesis().parameters()`` is empty. Sample L-skewness and L-kurtosis
    are ratios of linear combinations of order statistics [1]_. These
    ratios remove location and positive scale. Their squared deviations
    from normal reference values are divided by tabulated variances.

    This implementation selects fixed numerical reference constants for
    ``n <= 25``, ``25 < n <= 50``, and ``n > 50``. Supply at least four
    finite, nonconstant observations. The current ``alternative()`` returns
    ``TwoSidedAlternative`` although the statistic is a nonnegative
    quadratic discrepancy. Reference [1]_ describes the L-moment foundation,
    not the provenance of these implementation-specific numerical constants.

    References
    ----------
    .. [1] Hosking, J. R. M. (1990). L-Moments: Analysis and Estimation of
       Distributions Using Linear Combinations of Order Statistics. Journal
       of the Royal Statistical Society, Series B, 52(1), 105-124.
       https://doi.org/10.1111/j.2517-6161.1990.tb01775.x
    """

    @override
    def alternative(self) -> Alternative:
        return TwoSidedAlternative()

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
        """Compute the statistic for the observed sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional sample of finite observations. Sample-size and
            dispersion requirements are described in the class notes.
        **kwargs : dict, optional
            Unused keyword arguments accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Value of the statistic for a supported sample. This method does
            not return a p-value or a hypothesis-test decision.
        """
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
            tau3 = l3 / l2
            tau4 = l4 / l2

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

    Notes
    -----
    The null is the normal family with unknown mean and variance;
    ``hypothesis().parameters()`` is empty. The statistic combines squared
    deviations of trimmed L-skewness and L-kurtosis, using order-statistic
    weights with trimming level ``t=2`` from the construction in [1]_.
    The ratios remove location and positive scale.

    Supply at least 8 finite, nonconstant observations; estimating the
    fourth trimmed L-moment requires ``n >= 4 + 2*t``. Reference constants
    are selected from three sample-size ranges: up to 25, 26-50, and above
    50. The current ``alternative()`` reports ``TwoSidedAlternative`` for
    this nonnegative quadratic discrepancy. Reference [1]_ supplies the
    trimmed-moment foundation; it does not establish the provenance of
    the numerical constants or the exact omnibus implementation here.

    References
    ----------
    .. [1] Elamir, E. A. H. and Seheult, A. H. (2003). Trimmed L-moments.
       Computational Statistics & Data Analysis, 43(3), 299-314.
       https://doi.org/10.1016/S0167-9473(02)00250-5
    """

    @override
    def alternative(self) -> Alternative:
        return TwoSidedAlternative()

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
        """Compute the statistic for the observed sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional sample of finite observations. Sample-size and
            dispersion requirements are described in the class notes.
        **kwargs : dict, optional
            Unused keyword arguments accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Value of the statistic for a supported sample. This method does
            not return a p-value or a hypothesis-test decision.
        """
        return self.stat12(rvs)

    def stat12(self, x):
        n = len(x)

        if n > 3:
            x_tmp = x[:n]
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
            tau32 = l32 / l22
            tau42 = l42 / l22

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

    Notes
    -----
    The null is the normal family with unknown mean and variance;
    ``hypothesis().parameters()`` is empty. The statistic combines squared
    deviations of trimmed L-skewness and L-kurtosis, using order-statistic
    weights with trimming level ``t=3`` from the construction in [1]_.
    The ratios remove location and positive scale.

    Supply at least 10 finite, nonconstant observations; estimating the
    fourth trimmed L-moment requires ``n >= 4 + 2*t``. Reference constants
    are selected from three sample-size ranges: up to 25, 26-50, and above
    50. The current ``alternative()`` reports ``TwoSidedAlternative`` for
    this nonnegative quadratic discrepancy. Reference [1]_ supplies the
    trimmed-moment foundation; it does not establish the provenance of
    the numerical constants or the exact omnibus implementation here.

    References
    ----------
    .. [1] Elamir, E. A. H. and Seheult, A. H. (2003). Trimmed L-moments.
       Computational Statistics & Data Analysis, 43(3), 299-314.
       https://doi.org/10.1016/S0167-9473(02)00250-5
    """

    @override
    def alternative(self) -> Alternative:
        return TwoSidedAlternative()

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
        """Compute the statistic for the observed sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional sample of finite observations. Sample-size and
            dispersion requirements are described in the class notes.
        **kwargs : dict, optional
            Unused keyword arguments accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Value of the statistic for a supported sample. This method does
            not return a p-value or a hypothesis-test decision.
        """
        return self.stat13(rvs)

    def stat13(self, x):
        n = len(x)

        if n > 3:
            x_tmp = x[:n]
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
            tau33 = l33 / l23
            tau43 = l43 / l23

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

    The null hypothesis is the normal family with unknown mean and variance.
    No distribution parameters are accepted; ``hypothesis().parameters()``
    returns ``{}``.

    Notes
    -----
    Observations are centered by the sample mean and scaled by the sample
    standard deviation with ``ddof=1``. Their fitted normal probabilities are
    compared with order-dependent probability scores through squared log odds.
    Large values correspond to the right-tail alternative.

    Use at least four finite observations with nonzero sample variance.
    Probabilities rounded to zero or one can produce an infinite statistic.

    References
    ----------
    .. [1] Zhang, J. and Wu, Y. (2005).
       Likelihood-ratio tests for normality.
       Computational Statistics & Data Analysis, 49(3), 709-721.
       https://doi.org/10.1016/j.csda.2004.05.034
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
        """Compute the statistic for the observed sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional sample of finite observations. Sample-size and
            dispersion requirements are described in the class notes.
        **kwargs : dict, optional
            Unused keyword arguments accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Value of the statistic for a supported sample. This method does
            not return a p-value or a hypothesis-test decision.
        """
        n = len(rvs)

        if n > 3:
            phiz = np.zeros(n)
            mean_x = np.mean(rvs)
            var_x = np.var(rvs, ddof=1)
            sd_x = np.sqrt(var_x)
            for i in range(n):
                phiz[i] = scipy_stats.norm.cdf((rvs[i] - mean_x) / sd_x)
            phiz.sort()
            stat_zc = 0.0
            for i in range(1, n + 1):
                stat_zc += np.log((1.0 / phiz[i - 1] - 1.0) / ((n - 0.5) / (i - 0.75) - 1.0)) ** 2
            return stat_zc


class ZhangWuANormalityGofStatistic(AbstractNormalityGofStatistic):
    """Zhang-Wu likelihood-ratio normality statistic ``Z_A``.

    The null hypothesis is the normal family with unknown mean and variance.
    No distribution parameters are accepted; ``hypothesis().parameters()``
    returns ``{}``.

    Notes
    -----
    The calculation uses sorted normal probabilities after centering by the
    sample mean and scaling by the sample standard deviation with ``ddof=1``.
    This implementation returns ``10 * Z_A - 32``. This increasing affine
    transformation preserves the right-tail rejection ordering, but critical
    values must use the same transformation.

    Use at least four finite observations with nonzero sample variance.
    Probabilities rounded to zero or one can produce an infinite statistic.

    References
    ----------
    .. [1] Zhang, J. and Wu, Y. (2005).
       Likelihood-ratio tests for normality.
       Computational Statistics & Data Analysis, 49(3), 709-721.
       https://doi.org/10.1016/j.csda.2004.05.034
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
        """Compute the statistic for the observed sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional sample of finite observations. Sample-size and
            dispersion requirements are described in the class notes.
        **kwargs : dict, optional
            Unused keyword arguments accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Value of the statistic for a supported sample. This method does
            not return a p-value or a hypothesis-test decision.
        """
        n = len(rvs)

        if n > 3:
            phiz = np.zeros(n)
            mean_x = np.mean(rvs)
            var_x = np.var(rvs, ddof=1)
            sd_x = np.sqrt(var_x)
            for i in range(n):
                phiz[i] = scipy_stats.norm.cdf((rvs[i] - mean_x) / sd_x)
            phiz.sort()
            stat_za = 0.0
            for i in range(1, n + 1):
                stat_za += np.log(phiz[i - 1]) / ((n - i) + 0.5) + np.log(1.0 - phiz[i - 1]) / (
                    i - 0.5
                )
            stat_za = -stat_za
            stat_za = 10.0 * stat_za - 32.0
            return stat_za


class GlenLeemisBarrNormalityGofStatistic(AbstractNormalityGofStatistic):
    """Glen-Leemis-Barr order-statistic goodness-of-fit statistic.

    The null hypothesis is the normal family with unknown mean and variance.
    No distribution parameters are accepted; ``hypothesis().parameters()``
    returns ``{}``.

    Notes
    -----
    The sample mean and standard deviation with ``ddof=1`` define a fitted
    normal CDF. The ordered probabilities are transformed by their respective
    beta CDFs, sorted again, and combined in an Anderson-Darling-style sum.
    Large values correspond to the right-tail alternative.

    Use at least four finite observations with nonzero sample variance.
    The reference describes the order-statistic construction; this class uses
    its fitted-normal version, whose null calibration includes estimation.

    References
    ----------
    .. [1] Glen, A. G., Leemis, L. M. and Barr, D. R. (2001).
       Order statistics in goodness-of-fit testing.
       IEEE Transactions on Reliability, 50(2), 209-213.
       https://doi.org/10.1109/24.963129
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
        """Compute the statistic for the observed sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional sample of finite observations. Sample-size and
            dispersion requirements are described in the class notes.
        **kwargs : dict, optional
            Unused keyword arguments accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Value of the statistic for a supported sample. This method does
            not return a p-value or a hypothesis-test decision.
        """
        n = len(rvs)

        if n > 3:
            phiz = np.zeros(n)
            mean_x = np.mean(rvs)
            var_x = np.var(rvs, ddof=1)
            sd_x = np.sqrt(var_x)
            for i in range(n):
                phiz[i] = scipy_stats.norm.cdf((rvs[i] - mean_x) / sd_x)
            phiz.sort()
            for i in range(1, n + 1):
                phiz[i - 1] = scipy_stats.beta.cdf(phiz[i - 1], i, n - i + 1)
            phiz.sort()
            stat_ps = 0
            for i in range(1, n + 1):
                stat_ps += (2 * n + 1 - 2 * i) * np.log(phiz[i - 1]) + (2 * i - 1) * np.log(
                    1 - phiz[i - 1]
                )
            return -n - stat_ps / n


class DoornikHansenNormalityGofStatistic(AbstractNormalityGofStatistic):
    """Univariate Doornik-Hansen normality statistic.

    The null hypothesis is the normal family with unknown mean and variance.
    No distribution parameters are accepted; ``hypothesis().parameters()``
    returns ``{}``.

    Notes
    -----
    The statistic is the sum of squares of transformed sample skewness and
    kurtosis. Both moments are computed about the sample mean, so neither a
    population mean nor a population variance is supplied. Large values
    correspond to the right-tail alternative.

    The formulas require a nonconstant sample and can be undefined at very
    small sample sizes. The reference studies the small-sample approximation
    from ten observations. This class computes the statistic only; it does not
    apply the paper's asymptotic chi-square calibration.

    References
    ----------
    .. [1] Doornik, J. A. and Hansen, H. (2008).
       An Omnibus Test for Univariate and Multivariate Normality.
       Oxford Bulletin of Economics and Statistics, 70(s1), 927-939.
       https://doi.org/10.1111/j.1468-0084.2008.00537.x
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
        """Compute the statistic for the observed sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional sample of finite observations. Sample-size and
            dispersion requirements are described in the class notes.
        **kwargs : dict, optional
            Unused keyword arguments accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Value of the statistic for a supported sample. This method does
            not return a p-value or a hypothesis-test decision.
        """
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
        z = (0.5 * q / alpha) ** (1 / 3) - 1 + 1 / (9 * alpha)
        z *= math.sqrt(9 * alpha)
        return z


class RobustJarqueBeraNormalityGofStatistic(AbstractNormalityGofStatistic):
    """Gel-Gastwirth robust Jarque-Bera normality statistic.

    The null hypothesis is the normal family with unknown mean and variance.
    No distribution parameters are accepted; ``hypothesis().parameters()``
    returns ``{}``.

    Notes
    -----
    The scale estimate is ``sqrt(pi / 2)`` times the average absolute
    deviation from the sample median. Third and fourth moments are still
    centered at the sample mean. The statistic combines their standardized
    squares with the constants 6 and 64 used in this implementation.
    Large values correspond to the right-tail alternative.

    Use finite observations with positive scale. A scalar statistic is
    returned; no asymptotic chi-square p-value is computed here.

    References
    ----------
    .. [1] Gel, Y. R. and Gastwirth, J. L. (2008).
       A robust modification of the Jarque-Bera test of normality.
       Economics Letters, 99(1), 30-32.
       https://doi.org/10.1016/j.econlet.2007.05.022
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
        """Compute the statistic for the observed sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional sample of finite observations. Sample-size and
            dispersion requirements are described in the class notes.
        **kwargs : dict, optional
            Unused keyword arguments accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Value of the statistic for a supported sample. This method does
            not return a p-value or a hypothesis-test decision.
        """
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

    The null hypothesis is the normal family with unknown mean and variance.
    No distribution parameters are accepted; ``hypothesis().parameters()``
    returns ``{}``.

    Notes
    -----
    The sample is centered and scaled using its mean and standard deviation
    with ``ddof=1``. Sums of the normalized third and fourth Hermite
    polynomials are squared and divided by the sample size. Large values
    correspond to the right-tail alternative.

    Use at least four finite observations with nonzero sample variance.
    This is the independent-observation version; the covariance correction
    for dependent time series discussed in the article is not implemented.

    References
    ----------
    .. [1] Bontemps, C. and Meddahi, N. (2005).
       Testing normality: a GMM approach.
       Journal of Econometrics, 124(1), 149-186.
       https://doi.org/10.1016/j.jeconom.2004.02.014
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
        """Compute the statistic for the observed sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional sample of finite observations. Sample-size and
            dispersion requirements are described in the class notes.
        **kwargs : dict, optional
            Unused keyword arguments accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Value of the statistic for a supported sample. This method does
            not return a p-value or a hypothesis-test decision.
        """
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
            var_x = (n * (var_x / n - mean_x**2)) / (n - 1)
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

    The null hypothesis is the normal family with unknown mean and variance.
    No distribution parameters are accepted; ``hypothesis().parameters()``
    returns ``{}``.

    Notes
    -----
    After centering and scaling with the sample mean and standard deviation
    with ``ddof=1``, the calculation combines squared sums of four normalized
    Hermite polynomials. It extends the third-and-fourth-order statistic with
    orders five and six. Large values correspond to the right-tail alternative.

    Use at least four finite observations with nonzero sample variance.
    The implementation assumes independent observations and does not estimate
    a covariance matrix for serially dependent data.

    References
    ----------
    .. [1] Bontemps, C. and Meddahi, N. (2005).
       Testing normality: a GMM approach.
       Journal of Econometrics, 124(1), 149-186.
       https://doi.org/10.1016/j.jeconom.2004.02.014
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
        """Compute the statistic for the observed sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional sample of finite observations. Sample-size and
            dispersion requirements are described in the class notes.
        **kwargs : dict, optional
            Unused keyword arguments accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Value of the statistic for a supported sample. This method does
            not return a p-value or a hypothesis-test decision.
        """
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

    The null hypothesis is the normal family with unknown mean and variance.
    No distribution parameters are accepted; ``hypothesis().parameters()``
    returns ``{}``.

    Notes
    -----
    The statistic compares the sample standard deviation with the average
    absolute deviation about the sample mean. The ratio is log-transformed,
    scaled by 13.29, centered at 3, and multiplied by ``sqrt(n + 2) / 3.54``.
    Departures in either direction correspond to the two-sided alternative.

    Use at least four finite observations with nonzero dispersion. This class
    implements the signed Geary-based component described in the article,
    not its joint procedure combining two different kurtosis measures.

    References
    ----------
    .. [1] Bonett, D. G. and Seier, E. (2002).
       A test of normality with high uniform power.
       Computational Statistics & Data Analysis, 40(3), 435-445.
       https://doi.org/10.1016/S0167-9473(02)00074-9
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
        """Compute the statistic for the observed sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional sample of finite observations. Sample-size and
            dispersion requirements are described in the class notes.
        **kwargs : dict, optional
            Unused keyword arguments accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Value of the statistic for a supported sample. This method does
            not return a p-value or a hypothesis-test decision.
        """
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

    The null hypothesis is the normal family with unknown mean and variance.
    No distribution parameters are accepted; ``hypothesis().parameters()``
    returns ``{}``.

    Notes
    -----
    The numerator uses squared deviations from the sample median. The
    denominator is a biweight-style scale estimate using nine times the
    median absolute deviation as its tuning scale. Large values correspond
    to the right-tail alternative.

    Use at least four finite observations, a positive median absolute
    deviation, and a nonzero biweight denominator. The current implementation
    evaluates the biweight polynomials for all observations without truncating
    terms whose standardized absolute deviation is at least one.

    References
    ----------
    .. [1] Martinez, J. and Iglewicz, B. (1981).
       A test for departure from normality based on a biweight estimator of scale.
       Biometrika, 68(1), 331-333.
       https://doi.org/10.1093/biomet/68.1.331
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
        """Compute the statistic for the observed sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional sample of finite observations. Sample-size and
            dispersion requirements are described in the class notes.
        **kwargs : dict, optional
            Unused keyword arguments accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Value of the statistic for a supported sample. This method does
            not return a p-value or a hypothesis-test decision.
        """
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
            a = 9.0 * a

            z = aux1 / a
            term1 = np.sum(aux1**2 * (1 - z**2) ** 4)
            term2 = np.sum((1 - z**2) * (1 - 5 * z**2))
            term3 = np.sum(aux1**2)

            sb2 = (n * term1) / term2**2
            stat_in = (term3 / (n - 1)) / sb2
            return stat_in


class CabanaCabana1NormalityGofStatistic(AbstractNormalityGofStatistic):
    """Cabaña-Cabaña normality statistic focused on skewness departures.

    The null hypothesis is the normal family with unknown mean and variance.
    No distribution parameters are accepted; ``hypothesis().parameters()``
    returns ``{}``.

    Notes
    -----
    The calculation evaluates a transformed empirical process at the sample
    points after standardization by the sample mean and standard deviation
    with ``ddof=1``. It uses normalized Hermite polynomials through degree
    eight, corresponding to the fixed truncation order ``l=5``.
    The maximum absolute process value gives a right-tail statistic.

    Use at least four finite observations with nonzero sample variance.
    The normality construction is described in the 2003 paper cited below.

    References
    ----------
    .. [1] Cabaña, A. and Cabaña, E. M. (2003).
       Tests of Normality Based on Transformed Empirical Processes.
       Methodology and Computing in Applied Probability, 5, 309-335.
       https://doi.org/10.1023/A:1026235220018
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
        """Compute the statistic for the observed sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional sample of finite observations. Sample-size and
            dispersion requirements are described in the class notes.
        **kwargs : dict, optional
            Unused keyword arguments accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Value of the statistic for a supported sample. This method does
            not return a p-value or a hypothesis-test decision.
        """
        return self.stat19(rvs)

    @staticmethod
    def stat19(x):
        n = len(x)

        if n > 3:
            z_data = (x - np.mean(x)) / np.std(x, ddof=1)
            mean_h3 = np.sum(z_data**3 - 3 * z_data) / (np.sqrt(6) * np.sqrt(n))
            mean_h4 = np.sum(z_data**4 - 6 * z_data**2 + 3) / (2 * np.sqrt(6) * np.sqrt(n))
            mean_h5 = np.sum(z_data**5 - 10 * z_data**3 + 15 * z_data) / (
                2 * np.sqrt(30) * np.sqrt(n)
            )
            mean_h6 = np.sum(z_data**6 - 15 * z_data**4 + 45 * z_data**2 - 15) / (
                12 * np.sqrt(5) * np.sqrt(n)
            )
            mean_h7 = np.sum(z_data**7 - 21 * z_data**5 + 105 * z_data**3 - 105 * z_data) / (
                12 * np.sqrt(35) * np.sqrt(n)
            )
            mean_h8 = np.sum(
                z_data**8 - 28 * z_data**6 + 210 * z_data**4 - 420 * z_data**2 + 105
            ) / (24 * np.sqrt(70) * np.sqrt(n))
            vector_aux1 = (
                mean_h4
                + mean_h5 * z_data / np.sqrt(2)
                + mean_h6 * (z_data**2 - 1) / np.sqrt(6)
                + mean_h7 * (z_data**3 - 3 * z_data) / (2 * np.sqrt(6))
                + mean_h8 * (z_data**4 - 6 * z_data**2 + 3) / (2 * np.sqrt(30))
            )
            stat_tsl = np.max(
                np.abs(
                    scipy_stats.norm.cdf(z_data) * mean_h3
                    - scipy_stats.norm.pdf(z_data) * vector_aux1
                )
            )
            return stat_tsl


class CabanaCabana2NormalityGofStatistic(AbstractNormalityGofStatistic):
    """Cabaña-Cabaña-style normality statistic focused on kurtosis departures.

    The null hypothesis is the normal family with unknown mean and variance.
    No distribution parameters are accepted; ``hypothesis().parameters()``
    returns ``{}``.

    Notes
    -----
    The sample is standardized by its mean and standard deviation with
    ``ddof=1``. A transformed empirical process built from Hermite
    polynomials through degree eight is evaluated at these sample points.
    The largest absolute value gives a right-tail statistic.

    Use at least four finite observations with nonzero sample variance.
    The current formula repeats its highest-order contribution. Consequently,
    critical values for the published ``l=5`` formula cannot be assumed to
    apply directly to this implementation.

    References
    ----------
    .. [1] Cabaña, A. and Cabaña, E. M. (2003).
       Tests of Normality Based on Transformed Empirical Processes.
       Methodology and Computing in Applied Probability, 5, 309-335.
       https://doi.org/10.1023/A:1026235220018
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
        """Compute the statistic for the observed sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional sample of finite observations. Sample-size and
            dispersion requirements are described in the class notes.
        **kwargs : dict, optional
            Unused keyword arguments accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Value of the statistic for a supported sample. This method does
            not return a p-value or a hypothesis-test decision.
        """
        return self.stat20(rvs)

    @staticmethod
    def stat20(x):
        n = len(x)

        if n > 3:
            # TODO: Move variance calculation

            var_x = n * np.var(x) / (n - 1)
            sd_x = np.sqrt(var_x)
            z = (x - np.mean(x)) / sd_x
            h0 = np.zeros(n)
            h1 = np.zeros(n)
            h2 = np.zeros(n)
            h3 = np.zeros(n)
            h4 = np.zeros(n)
            h5 = np.zeros(n)
            h6 = np.zeros(n)
            h7 = np.zeros(n)
            h8 = np.zeros(n)

            h3_tilde = 0
            h4_tilde = 0
            h5_tilde = 0
            h6_tilde = 0
            h7_tilde = 0
            h8_tilde = 0

            for i in range(n):
                h0[i] = 1
                h1[i] = z[i]
                h2[i] = (math.pow(z[i], 2.0) - 1.0) / np.sqrt(2.0)
                h3[i] = (math.pow(z[i], 3.0) - 3.0 * z[i]) / np.sqrt(6.0)
                h4[i] = (math.pow(z[i], 4.0) - 6.0 * math.pow(z[i], 2.0) + 3.0) / (
                    2.0 * np.sqrt(6.0)
                )
                h5[i] = (math.pow(z[i], 5.0) - 10.0 * math.pow(z[i], 3.0) + 15.0 * z[i]) / (
                    2.0 * np.sqrt(30.0)
                )
                h6[i] = (
                    math.pow(z[i], 6.0)
                    - 15.0 * math.pow(z[i], 4.0)
                    + 45.0 * math.pow(z[i], 2.0)
                    - 15.0
                ) / (12.0 * np.sqrt(5.0))
                h7[i] = (
                    math.pow(z[i], 7.0)
                    - 21.0 * math.pow(z[i], 5.0)
                    + 105.0 * math.pow(z[i], 3.0)
                    - 105.0 * z[i]
                ) / (12.0 * np.sqrt(35.0))
                h8[i] = (
                    math.pow(z[i], 8.0)
                    - 28.0 * math.pow(z[i], 6.0)
                    + 210.0 * math.pow(z[i], 4.0)
                    - 420.0 * math.pow(z[i], 2.0)
                    + 105.0
                ) / (24.0 * np.sqrt(70.0))

                h3_tilde = h3_tilde + h3[i]
                h4_tilde = h4_tilde + h4[i]
                h5_tilde = h5_tilde + h5[i]
                h6_tilde = h6_tilde + h6[i]
                h7_tilde = h7_tilde + h7[i]
                h8_tilde = h8_tilde + h8[i]

            h3_tilde = h3_tilde / np.sqrt(n)
            h4_tilde = h4_tilde / np.sqrt(n)
            h5_tilde = h5_tilde / np.sqrt(n)
            h6_tilde = h6_tilde / np.sqrt(n)
            h7_tilde = h7_tilde / np.sqrt(n)
            h8_tilde = h8_tilde / np.sqrt(n)

            vector_aux2 = (
                (np.sqrt(2) * h0 + h2) * h5_tilde
                + (np.sqrt(3 / 2) * h1 + h3) * h6_tilde
                + (np.sqrt(4 / 3) * h2 + h4) * h7_tilde
                + (np.sqrt(5 / 4) * h3 + h5) * h8_tilde
                + (np.sqrt(5 / 4) * h3 + h5) * h8_tilde
            )
            stat_tkl = np.max(
                np.abs(
                    -scipy_stats.norm.pdf(z) * h3_tilde
                    + (scipy_stats.norm.cdf(z) - z * scipy_stats.norm.pdf(z)) * h4_tilde
                    - scipy_stats.norm.pdf(z) * vector_aux2
                )
            )
            return stat_tkl


class ChenShapiroNormalityGofStatistic(AbstractNormalityGofStatistic):
    """Chen-Shapiro normality statistic based on normalized spacings.

    The null hypothesis is the normal family with unknown mean and variance.
    No distribution parameters are accepted; ``hypothesis().parameters()``
    returns ``{}``.

    Notes
    -----
    Successive sample spacings are divided by spacings of the normal scores
    ``norm.ppf((i - 0.375) / (n + 0.25))``. Their average is standardized by
    the sample standard deviation with ``ddof=1`` to form ``QH``.
    The returned statistic is ``sqrt(n) * (1 - QH)``; large values correspond
    to the right-tail alternative.

    Use at least four finite observations with nonzero sample variance.
    The calculation estimates location and scale implicitly from the sample.

    References
    ----------
    .. [1] Chen, L. and Shapiro, S. S. (1995).
       An alternative test for normality based on normalized spacings.
       Journal of Statistical Computation and Simulation, 53(3-4), 269-287.
       https://doi.org/10.1080/00949659508811711
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
        """Compute the statistic for the observed sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional sample of finite observations. Sample-size and
            dispersion requirements are described in the class notes.
        **kwargs : dict, optional
            Unused keyword arguments accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Value of the statistic for a supported sample. This method does
            not return a p-value or a hypothesis-test decision.
        """
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

    The null hypothesis is the normal family with unknown mean and variance.
    No distribution parameters are accepted; ``hypothesis().parameters()``
    returns ``{}``.

    Notes
    -----
    Two linear combinations of the ordered observations use coefficients
    constructed from approximate expected normal scores. The statistic is
    ``log(q1 / q2)``. Location cancels from each contrast and scale cancels
    from their ratio. The implementation declares a right-tail alternative.

    At least eight finite observations are needed because the coefficients
    access the first eight normal scores. The contrasts must yield a positive,
    finite ratio. Smaller samples are not supported by the current formula.

    References
    ----------
    .. [1] Zhang, P. (1999).
       Omnibus test of normality using the Q statistic.
       Journal of Applied Statistics, 26(4), 519-528.
       https://doi.org/10.1080/02664769922395
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

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
        """Compute the statistic for the observed sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional sample of finite observations. Sample-size and
            dispersion requirements are described in the class notes.
        **kwargs : dict, optional
            Unused keyword arguments accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Value of the statistic for a supported sample. This method does
            not return a p-value or a hypothesis-test decision.
        """
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
            stat_q = np.log(q1 / q2)
            return stat_q


class CoinNormalityGofStatistic(AbstractNormalityGofStatistic):
    """Coin normality statistic based on polynomial regression.

    The null hypothesis is the normal family with unknown mean and variance.
    No distribution parameters are accepted; ``hypothesis().parameters()``
    returns ``{}``.

    Notes
    -----
    The ordered sample is centered and scaled using its sample mean and
    standard deviation with ``ddof=1``. A regression on approximate expected
    normal order statistics and their cubes produces a cubic coefficient.
    The returned statistic is the square of that coefficient, with large
    values corresponding to the right-tail alternative.

    Use at least four finite observations with nonzero sample variance.
    The normal-score approximation used here reports possible inaccuracy
    when the sample contains more than 2000 observations.

    References
    ----------
    .. [1] Coin, D. (2008).
       A goodness-of-fit test for normality based on polynomial regression.
       Computational Statistics & Data Analysis, 52(4), 2185-2198.
       https://doi.org/10.1016/j.csda.2007.07.012
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
        """Compute the statistic for the observed sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional sample of finite observations. Sample-size and
            dispersion requirements are described in the class notes.
        **kwargs : dict, optional
            Unused keyword arguments accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Value of the statistic for a supported sample. This method does
            not return a p-value or a hypothesis-test decision.
        """
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
            var_x = (n * (var_x / n - mean_x**2)) / (n - 1)
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
            print("Values may be inaccurate because of the size of N")

        s[0] = b1
        if n == 2:
            return

        an = n
        k = 3
        if n2[0] < k:
            k = n2[0]

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

    The null hypothesis is the normal family with unknown mean and variance.
    No distribution parameters are accepted; ``hypothesis().parameters()``
    returns ``{}``.

    Notes
    -----
    A linear contrast of the order statistics is divided by the sample
    standard deviation computed with ``ddof=0`` and by ``n**2`` to form ``D``.
    This implementation returns
    ``sqrt(n) * (D - 0.28209479) / 0.02998598``.
    Both low and high values correspond to the two-sided alternative.

    Use at least four finite observations with nonzero sample variance.
    This order-statistic construction is distinct from the skewness-and-
    kurtosis D'Agostino-Pearson statistic.

    References
    ----------
    .. [1] D'Agostino, R. B. (1971).
       An omnibus test of normality for moderate and large size samples.
       Biometrika, 58(2), 341-348.
       https://doi.org/10.1093/biomet/58.2.341
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
        """Compute the statistic for the observed sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional sample of finite observations. Sample-size and
            dispersion requirements are described in the class notes.
        **kwargs : dict, optional
            Unused keyword arguments accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Value of the statistic for a supported sample. This method does
            not return a p-value or a hypothesis-test decision.
        """
        n = len(rvs)
        if n > 3:
            xs = np.sort(rvs)  # We sort the data
            mean_x = sum(xs) / n
            var_x = sum(x_i**2 for x_i in xs) / n - mean_x**2
            t = sum((i - 0.5 * (n + 1)) * xs[i - 1] for i in range(1, n + 1))
            d = t / ((n**2) * math.sqrt(var_x))
            stat_da = math.sqrt(n) * (d - 0.28209479) / 0.02998598

            return stat_da  # Here is the test statistic value


class ZhangQStarNormalityGofStatistic(AbstractNormalityGofStatistic):
    """Reflected Zhang ``Q*`` normality statistic.

    The null hypothesis is the normal family with unknown mean and variance.
    No distribution parameters are accepted; ``hypothesis().parameters()``
    returns ``{}``.

    Notes
    -----
    This variant applies the Zhang ``Q`` construction to the negated,
    reverse-ordered sample. It returns the logarithm of the ratio of two
    linear contrasts. Shifts and positive rescaling cancel from the result.
    The implementation declares a right-tail alternative.

    At least eight finite observations are needed because the coefficients
    access the first eight normal scores. A positive, finite contrast ratio
    is required. This is the reflected component alone, not a joint
    combination of the ``Q`` and ``Q*`` p-values.

    References
    ----------
    .. [1] Zhang, P. (1999).
       Omnibus test of normality using the Q statistic.
       Journal of Applied Statistics, 26(4), 519-528.
       https://doi.org/10.1080/02664769922395
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

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
        """Compute the statistic for the observed sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional sample of finite observations. Sample-size and
            dispersion requirements are described in the class notes.
        **kwargs : dict, optional
            Unused keyword arguments accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Value of the statistic for a supported sample. This method does
            not return a p-value or a hypothesis-test decision.
        """
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

            q_star = np.log(q1_star / q2_star)
            return q_star


"""
class ZhangQQStarNormalityTest(AbstractNormalityTestStatistic):  # TODO: check for correctness

    @staticmethod
    @override
    def code():
        return 'ZQQ' + '_' + super(AbstractNormalityTestStatistic, AbstractNormalityTestStatistic)
        .code()

    @override
    def execute_statistic(self, rvs, **kwargs):
        return self.stat28(rvs)

    @staticmethod
    def stat28(x):
        n = len(x)

        if n > 3:
            # Computation of the value of the test statistic
            def stat27(x):
                pass

            def stat34(x):
                pass

            p_value27 = [1.0]
            p_value34 = [1.0]

            stat27(x)  # stat Q de Zhang

            if p_value27[0] > 0.5:
                p_val1 = 1.0 - p_value27[0]
            else:
                p_val1 = p_value27[0]

            stat34(x)  # stat Q* de Zhang

            if p_value34[0] > 0.5:
                p_val2 = 1.0 - p_value34[0]
            else:
                p_val2 = p_value34[0]

            # Combinaison des valeurs-p (Fisher, 1932)
            stat = -2.0 * (np.log(p_val1) + np.log(p_val2))

            return stat  # Here is the test statistic value
"""


class SWRGNormalityGofStatistic(AbstractNormalityGofStatistic):
    """Rahman-Govindarajulu modification of the Shapiro-Wilk statistic.

    The null hypothesis is the normal family with unknown mean and variance.
    No distribution parameters are accepted; ``hypothesis().parameters()``
    returns ``{}``.

    Notes
    -----
    Weights are formed from second differences of normal-score density
    products and normalized to unit length. The squared weighted sum of the
    ordered sample is divided by its centered sum of squares. Use at least
    four finite observations with nonzero sample variance.

    The returned value is the raw ``W_RG`` ratio, not ``1 - W_RG``. The
    published Shapiro-Wilk-type test uses small values as evidence against
    normality, while this implementation currently declares a right-tail
    alternative. Account for that mismatch when using it for decisions.

    References
    ----------
    .. [1] Rahman, M. M. and Govindarajulu, Z. (1997).
       A modification of the test of Shapiro and Wilk for normality.
       Journal of Applied Statistics, 24(2), 219-236.
       https://doi.org/10.1080/02664769723828
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

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
        """Compute the statistic for the observed sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional sample of finite observations. Sample-size and
            dispersion requirements are described in the class notes.
        **kwargs : dict, optional
            Unused keyword arguments accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Value of the statistic for a supported sample. This method does
            not return a p-value or a hypothesis-test decision.
        """
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

    The null hypothesis is the normal family with unknown mean and variance.
    No distribution parameters are accepted; ``hypothesis().parameters()``
    returns ``{}``.

    Notes
    -----
    The statistic divides the sample standard deviation with ``ddof=0`` by
    ``sqrt(pi / 2)`` times the average absolute deviation from the sample
    median. This ratio is close to one under normality, and large values
    correspond to the right-tail alternative aimed at heavy-tailed departures.

    Use at least four finite observations with positive dispersion.
    The returned value is the raw ratio; no large-sample centering, scaling,
    or p-value approximation from the article is applied.

    References
    ----------
    .. [1] Gel, Y. R., Miao, W. and Gastwirth, J. L. (2007).
       Robust directed tests of normality against heavy-tailed alternatives.
       Computational Statistics & Data Analysis, 51(5), 2734-2746.
       https://doi.org/10.1016/j.csda.2006.08.022
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
        """Compute the statistic for the observed sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional sample of finite observations. Sample-size and
            dispersion requirements are described in the class notes.
        **kwargs : dict, optional
            Unused keyword arguments accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Value of the statistic for a supported sample. This method does
            not return a p-value or a hypothesis-test decision.
        """
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


""" Title: Statistique de test de Brys-Hubert-Struyf MC-LR
Ref. (book or article): Brys, G., Hubert, M. and Struyf, A. (2008), Goodness-of-fit tests based on
a robust measure of skewness, Computational Statistics, Vol. 23, Issue 3, pp. 429-442.
"""


class BHSNormalityGofStatistic(AbstractNormalityGofStatistic):
    """Brys-Hubert-Struyf MC-LR statistic for the normal family.

    Notes
    -----
    The null hypothesis is normality with unknown mean and variance;
    `hypothesis().parameters()` is an empty dictionary. No distribution
    parameters are accepted by the constructor.

    The statistic combines the medcouple with left and right medcouples in
    a quadratic form [1]_. These measure skewness and tail weight without
    fixing the location or scale. Large values provide evidence against
    normality; the class uses a right-sided alternative.

    Warnings
    --------
    The current medcouple implementation is incomplete. It may fail to
    terminate for samples with five or more observations and is not suitable
    for reliable inference until repaired. Existing regression tests for
    these sample sizes are disabled for this reason.

    References
    ----------
    .. [1] Brys, G., Hubert, M., and Struyf, A. (2008). "Goodness-of-fit tests
       based on a robust measure of skewness." Computational Statistics,
       23, 429-442. https://doi.org/10.1007/s00180-007-0083-7
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
        """Compute the statistic for the observed sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional sample of finite observations. Sample-size and
            dispersion requirements are described in the class notes.
        **kwargs : dict, optional
            Unused keyword arguments accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Value of the statistic for a supported sample. This method does
            not return a p-value or a hypothesis-test decision.
        """
        return self.stat16(rvs)

    def stat16(self, x):
        n = len(x)

        if n > 3:
            # Computation of the value of the test statistic
            x_sorted = np.sort(x)
            if n % 2 == 0:
                x1 = x_sorted[: n // 2]
                x2 = x_sorted[n // 2 :]
            else:
                x1 = x_sorted[: (n // 2) + 1]
                x2 = x_sorted[(n // 2) + 1 :]

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
    # flake8: noqa: C901
    def mc_c_d(self, z, eps, iter_):
        """
        NOTE:
            eps = [eps1, eps2]
            iter = [max_it, trace_lev] as input
                  = [it, converged] as output
        """
        trace_lev = iter_[1]
        it = 0
        converged = True
        med_c = None  # "the" result
        # dbl_max = 1.7976931348623158e+308
        dbl_max = 1.7976931348623158e308
        large = dbl_max / 4.0

        n = len(z)
        if n < 3:
            med_c = 0.0
            iter_[0] = it  # to return
            iter_[1] = converged
            return med_c

        # copy data before sort()ing in place, also reflecting it -- dealing with +-Inf.
        # NOTE: x[0] "empty" so we can use 1-indexing below
        x = [0.0] * (n + 1)
        for i in range(n):
            zi = z[i]
            x[i + 1] = -(large if zi == float("inf") else (-large if zi == -float("inf") else zi))

        x[1:] = sorted(x[1:])  # full sort

        # x_med := median(x[1:n]) = -median(z[0:(n-1)])
        if n % 2:  # n even
            x_med = x[(n // 2) + 1]
        else:  # n odd
            ind = n // 2
            x_med = (x[ind] + x[ind + 1]) / 2.0

        if abs(x[1] - x_med) < eps[0] * (eps[0] + abs(x_med)):
            med_c = -1.0
            iter_[0] = it  # to return
            iter_[1] = converged
            return med_c
        elif abs(x[n] - x_med) < eps[0] * (eps[0] + abs(x_med)):
            med_c = 1.0
            iter_[0] = it  # to return
            iter_[1] = converged
            return med_c
        # else: median is not at the border
        if trace_lev:
            print(f"mc_C_d(z[1:{n}], trace_lev={trace_lev}): Median = {-x_med} (not at the border)")

        # center x[] wrt median --> such that then median(x[1:n]) == 0
        for i in range(1, n + 1):
            x[i] -= x_med

        # Now scale to inside [-0.5, 0.5] and flip sign such that afterwards
        # x[1] >= x[2] >= ... >= x[n]
        x_den = -2 * max(-x[1], x[n])
        for i in range(1, n + 1):
            x[i] /= x_den
        x_med /= x_den
        if trace_lev >= 2:
            print(f" x[] has been rescaled (* 1/s) with s = {-x_den}")

        j = 1
        x_eps = eps[0] * (eps[0] + abs(x_med))
        while j <= n and x[j] > x_eps:  # test relative to x_med
            j += 1
        if trace_lev >= 2:
            print(f"   x1[] := {{x | x_j > x_eps = {x_eps}}}    has {j - 1} (='j-1') entries")
        i = 1
        x2 = x[j - 1 :]  # pointer -- corresponding to x2[i] = x[j]
        while j <= n and x[j] > -x_eps:  # test relative to x_med
            j += 1
            i += 1
        # now x1[] := {x | x_j > -eps} also includes the median (0)
        if trace_lev >= 2:
            print(f"'median-x' {{x | -eps < x_i <= eps}} has {i - 1} (= 'k') entries")
        h1 = j - 1  # == size of x1[] == the sum of those two sizes above
        # conceptually, x2[] := {x | x_j <= eps} (which includes the median 0)
        h2 = i + (n - j)  # == size of x2[] == maximal size of whi_med() arrays

        if trace_lev:
            print(f"  now allocating 2+5 work arrays of size (1+) h2={h2} each:")
        # work arrays for whi_med_i()
        a_cand = [0.0] * h2
        a_srt = [0.0] * h2
        iw_cand = [0] * h2
        # work arrays for the fast-median-of-table algorithm: currently still with 1-indexing
        left = [1] * (h2 + 1)
        right = [h1] * (h2 + 1)
        p = [0] * (h2 + 1)
        q = [0] * (h2 + 1)

        nr = h1 * h2  # <-- careful to *NOT* overflow
        knew = nr // 2 + 1
        if trace_lev >= 2:
            print(f" (h1,h2, nr, knew) = ({h1},{h2}, {nr}, {knew})")

        trial = -2.0  # -Wall
        work = [0.0] * n
        iwt = [0] * n
        is_found = False
        nl = 0
        neq = 0
        # MK: 'neq' counts the number of observations in the inside the tolerance range,
        # i.e., where left > right + 1, since we would miss those when just using 'nl-nr'.
        # This is to prevent index overflow in work[] later on.
        # left might be larger than right + 1 since we are only testing with accuracy eps_trial
        # and therefore there might be more than one observation in the `tolerance range`
        # between < and <=.
        while not is_found and (nr - nl + neq > n) and it < iter_[0]:
            if trace_lev >= 2:
                print(f"  it={it}")
            it += 1
            j = 0
            for i in range(1, h2 + 1):
                if left[i] <= right[i]:
                    iwt[j] = right[i] - left[i] + 1
                    k = left[i] + (iwt[j] // 2)
                    work[j] = self.h_kern(x[k], x2[i - 1], k, i, h1 + 1, eps[1])
                    j += 1
            if trace_lev >= 4:
                # print(" before whi_med(): work and iwt, each [0:({})]".format(j - 1))
                if j >= 100:
                    for i in range(90):
                        print(f" {work[i]:8g}", end="")
                    print("\n  ... ", end="")
                    for i in range(j - 4, j):
                        print(f" {work[i]:8g}", end="")
                    print("\n", end="")
                    for i in range(90):
                        print(f" {iwt[i]:8d}", end="")
                    print("\n  ... ", end="")
                    for i in range(j - 4, j):
                        print(f" {iwt[i]:8d}", end="")
                    print("\n", end="")
                else:  # j <= 99
                    for i in range(j):
                        print(f" {work[i]:8g}", end="")
                    print("\n", end="")
                    for i in range(j):
                        print(f" {iwt[i]:8d}", end="")
                    print("\n", end="")
            trial = self.whi_med_i(work, iwt, j, a_cand, a_srt, iw_cand)
            eps_trial = eps[0] * (eps[0] + abs(trial))
            if trace_lev >= 3:
                print(f"{' ':2s} it={it:2d}, whi_med(*, n={j:6d})= {trial:8g} ", end="")

            j = 1
            for i in range(h2, 0, -1):
                while (
                    j <= h1
                    and self.h_kern(x[j], x2[i - 1], j, i, h1 + 1, eps[1]) - trial > eps_trial
                ):
                    j += 1
                p[i] = j - 1

            j = h1
            sum_p = 0
            sum_q = 0
            for i in range(1, h2 + 1):
                while (
                    j >= 1
                    and trial - self.h_kern(x[j], x2[i - 1], j, i, h1 + 1, eps[1]) > eps_trial
                ):
                    j -= 1
                q[i] = j + 1

                sum_p += p[i]
                sum_q += j  # = q[i]-1

            if trace_lev >= 3:
                if trace_lev == 3:
                    print(f"sum_(p,q)= ({sum_p},{sum_q})", end="")
                else:  # trace_lev >= 4
                    print(f"\n{' ':3s} p[1:{h2}]:", end="")
                    lrg = h2 >= 100
                    i_m = 95 if lrg else h2
                    for i in range(1, i_m + 1):
                        print(f" {p[i]:2d}", end="")
                    if lrg:
                        print(" ...", end="")
                    print(f" sum={sum_p:4.0f}")
                    print(f"{' ':3s} q[1:{h2}]:", end="")
                    for i in range(1, i_m + 1):
                        print(f" {q[i]:2d}", end="")
                    if lrg:
                        print(" ...", end="")
                    print(f" sum={sum_q:4.0f}")

            if knew <= sum_p:
                if trace_lev >= 3:
                    print("; sum_p >= kn")
                for i in range(1, h2 + 1):
                    right[i] = p[i]
                    if left[i] > right[i] + 1:
                        neq += left[i] - right[i] - 1
                nr = sum_p
            else:  # knew > sum_p
                is_found = knew <= sum_q  # i.e. sum_p < knew <= sum_q

                if trace_lev >= 3:
                    print("; s_p < kn ?<=? s_q: {}".format("TRUE" if is_found else "no"))
                if is_found:
                    med_c = trial
                else:  # knew > sum_q
                    for i in range(1, h2 + 1):
                        left[i] = q[i]
                        if left[i] > right[i] + 1:
                            neq += left[i] - right[i] - 1
                    nl = sum_q

        converged = is_found or (nr - nl + neq <= n)
        if not converged:
            if trace_lev:
                print(f"maximal number of iterations ({iter_[0]} =? {it}) reached prematurely")
            # still:
            med_c = trial

        if converged and not is_found:  # e.g., for mc(1:4)
            j = 0
            for i in range(1, h2 + 1):
                if left[i] <= right[i]:
                    for k in range(left[i], right[i] + 1):
                        work[j] = -self.h_kern(x[k], x2[i - 1], k, i, h1 + 1, eps[1])
                        j += 1
            if trace_lev:
                print(
                    f"  not found [it={it},  (nr,nl) = ({nr},{nl})], -> (knew-nl, j) ="
                    f" ({knew - nl},{j})"
                )
            # using rPsort(work, n,k), since we don't need work[] anymore
            work[: (knew - nl)] = sorted(work[: (knew - nl)])
            med_c = -work[knew - nl - 1]

        if converged and trace_lev >= 2:
            print(f"converged in {it} iterations")

        iter_[0] = it  # to return
        iter_[1] = converged

        return med_c

    @staticmethod
    def h_kern(a, b, ai, bi, ab, eps):
        if np.abs(a - b) < 2.0 * eps or b > 0:
            return np.sign(ab - (ai + bi))
        else:
            return (a + b) / (a - b)

    @staticmethod
    def whi_med_i(a, w, n, a_cand, a_srt, w_cand):
        w_tot = sum(w)
        w_rest = 0

        while True:
            a_srt[:] = sorted(a)
            n2 = n // 2
            trial = a_srt[n2]

            w_left = sum(w[i] for i in range(n) if a[i] < trial)
            w_mid = sum(w[i] for i in range(n) if a[i] == trial)
            # w_right = sum(w[i] for i in range(n) if a[i] > trial)

            k_cand = 0
            if 2 * (w_rest + w_left) > w_tot:
                for i in range(n):
                    if a[i] < trial:
                        a_cand[k_cand] = a[i]
                        w_cand[k_cand] = w[i]
                        k_cand += 1
            elif 2 * (w_rest + w_left + w_mid) <= w_tot:
                for i in range(n):
                    if a[i] > trial:
                        a_cand[k_cand] = a[i]
                        w_cand[k_cand] = w[i]
                        k_cand += 1
                w_rest += w_left + w_mid
            else:
                return trial

            n = k_cand
            for i in range(n):
                a[i] = a_cand[i]
                w[i] = w_cand[i]


class SpiegelhalterNormalityGofStatistic(AbstractNormalityGofStatistic):
    """Spiegelhalter statistic for the normal family.

    Notes
    -----
    The null hypothesis is normality with unknown mean and variance;
    `hypothesis().parameters()` is an empty dictionary. No distribution
    parameters are accepted by the constructor.

    The statistic combines the standardized sample range and mean absolute
    deviation [1]_. Location and scale cancel in the calculation. This class
    uses a two-sided alternative and requires at least four observations
    with nonzero sample variance.

    References
    ----------
    .. [1] Spiegelhalter, D. J. (1977). "A test for normality against
       symmetric alternatives." Biometrika, 64(2), 415-418.
       https://doi.org/10.1093/biomet/64.2.415
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
        """Compute the statistic for the observed sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional sample of finite observations. Sample-size and
            dispersion requirements are described in the class notes.
        **kwargs : dict, optional
            Unused keyword arguments accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Value of the statistic for a supported sample. This method does
            not return a p-value or a hypothesis-test decision.
        """
        return self.stat41(rvs)

    @staticmethod
    def stat41(x):
        n = len(x)

        if n > 3:
            stat_sp, var_x, mean = 0.0, 0.0, 0.0
            max_val, min_val = x[0], x[0]
            for i in range(1, n):
                if x[i] > max_val:
                    max_val = x[i]
                if x[i] < min_val:
                    min_val = x[i]
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
            if n < 150:
                cn = 0.5 * math.gamma(n + 1) ** (1 / (n - 1)) / n
            else:
                cn = (
                    (2 * math.pi) ** (1 / (2 * (n - 1)))
                    * ((n * math.sqrt(n)) / math.e) ** (1 / (n - 1))
                    / (2 * math.e)
                )  # Stirling approximation

            stat_sp = ((cn * u) ** (-(n - 1)) + g ** (-(n - 1))) ** (1 / (n - 1))

            return stat_sp  # Here is the test statistic value


class DesgagneLafayeNormalityGofStatistic(AbstractNormalityGofStatistic):
    """Desgagne-Lafaye de Micheaux-Leblanc R_n normality statistic.

    Notes
    -----
    The null hypothesis is normality with unknown mean and variance;
    `hypothesis().parameters()` is an empty dictionary. No distribution
    parameters are accepted by the constructor.

    The sample is centered and divided by its standard deviation computed
    with `ddof=0`. The statistic combines three logarithmic measures of
    tail thickness in a quadratic form, as in the Rao score test against
    generalized exponential power alternatives [1]_. The implementation
    evaluates `R_n`, although its historical short code is `DLDMZEPD`.

    Large values provide evidence against normality; the class uses a
    right-sided alternative. At least four observations and nonzero sample
    variance are required. A centered observation equal to zero can produce
    a nonfinite result because the current implementation evaluates its
    logarithm directly.

    References
    ----------
    .. [1] Desgagne, A., Lafaye de Micheaux, P., and Leblanc, A. (2013).
       "Test of Normality Against Generalized Exponential Power
       Alternatives." Communications in Statistics - Theory and Methods,
       42(1), 164-190. https://doi.org/10.1080/03610926.2011.577548
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
        """Compute the statistic for the observed sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional sample of finite observations. Sample-size and
            dispersion requirements are described in the class notes.
        **kwargs : dict, optional
            Unused keyword arguments accepted for interface compatibility.

        Returns
        -------
        statistic : float
            Value of the statistic for a supported sample. This method does
            not return a p-value or a hypothesis-test decision.
        """
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
            varpop_x = varpop_x / n - mean_x**2
            sd_x = np.sqrt(varpop_x)
            for i in range(n):
                y[i] = (x[i] - mean_x) / sd_x

            # Formulas given in our paper p. 169
            for i in range(n):
                r1 += y[i] ** 2 * np.log(abs(y[i]))
                r2 += np.log(1.0 + abs(y[i]))
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
    """Base class for normal proximity-graph statistics with known variance.

    Parameters
    ----------
    var : float, optional
        Finite positive variance fixed by the null hypothesis. Default is
        1. This is a variance, not a standard deviation. Keyword-only.

    Raises
    ------
    ValueError
        If the variance is nonpositive or nonfinite.

    Notes
    -----
    The null hypothesis is normality with variance `var` and an arbitrary
    mean; `hypothesis().parameters()` returns `{"var": var}`.

    Observations are graph vertices. Two vertices are joined when their
    absolute difference is strictly smaller than
    `(max(rvs) - min(rvs)) / (10 * np.var(rvs, ddof=0))`.
    For zero sample variance, the denominator involving variance is omitted.
    The threshold uses the empirical variance, not the hypothesized `var`.

    The construction is invariant to shifts but not to changes of scale.
    Consequently, `var` determines the null distribution used for
    calibration without directly entering the statistic. Subclasses use
    a two-sided alternative.

    This threshold rule is documented from the PySATL implementation [1]_.
    An originating research article has not been identified.

    References
    ----------
    .. [1] PySATL. "pysatl-criterion", normal proximity-graph implementation.
       https://github.com/PySATL/pysatl-criterion
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

    @staticmethod
    @override
    def _compute_dist(rvs):
        base_dist = AbstractGraphTestStatistic._compute_dist(rvs)
        variance = np.var(rvs)
        return base_dist / variance if variance != 0 else base_dist


class GraphEdgesNumberNormalityGofStatistic(
    AbstractGraphNormalityGofStatistic, GraphEdgesNumberTestStatistic
):
    """Number of edges for a normal sample with known variance.

    Parameters
    ----------
    var : float, optional
        Finite positive null variance, specified as a keyword argument.
        Default is 1. The null mean is unspecified.

    See Also
    --------
    AbstractGraphNormalityGofStatistic : Graph construction and calibration.

    Notes
    -----
    Counts the undirected edges in the sample proximity graph.

    The null hypothesis is normality with arbitrary mean and variance
    `var`; `hypothesis().parameters()` returns `{"var": var}`.
    The alternative is two-sided. The graph threshold and its dependence
    on sample scale are described in `AbstractGraphNormalityGofStatistic`.

    This is the PySATL implementation [1]_; an originating research article
    has not been identified.

    References
    ----------
    .. [1] PySATL. "pysatl-criterion", normal proximity-graph implementation.
       https://github.com/PySATL/pysatl-criterion
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
        Finite positive null variance, specified as a keyword argument.
        Default is 1. The null mean is unspecified.

    See Also
    --------
    AbstractGraphNormalityGofStatistic : Graph construction and calibration.

    Notes
    -----
    Returns the largest number of neighbors of any sample vertex.

    The null hypothesis is normality with arbitrary mean and variance
    `var`; `hypothesis().parameters()` returns `{"var": var}`.
    The alternative is two-sided. The graph threshold and its dependence
    on sample scale are described in `AbstractGraphNormalityGofStatistic`.

    This is the PySATL implementation [1]_; an originating research article
    has not been identified.

    References
    ----------
    .. [1] PySATL. "pysatl-criterion", normal proximity-graph implementation.
       https://github.com/PySATL/pysatl-criterion
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
        Finite positive null variance, specified as a keyword argument.
        Default is 1. The null mean is unspecified.

    See Also
    --------
    AbstractGraphNormalityGofStatistic : Graph construction and calibration.

    Notes
    -----
    Returns the mean number of neighbors per sample vertex, equal to twice
    the edge count divided by the sample size.

    The null hypothesis is normality with arbitrary mean and variance
    `var`; `hypothesis().parameters()` returns `{"var": var}`.
    The alternative is two-sided. The graph threshold and its dependence
    on sample scale are described in `AbstractGraphNormalityGofStatistic`.

    This is the PySATL implementation [1]_; an originating research article
    has not been identified.

    References
    ----------
    .. [1] PySATL. "pysatl-criterion", normal proximity-graph implementation.
       https://github.com/PySATL/pysatl-criterion
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
        Finite positive null variance, specified as a keyword argument.
        Default is 1. The null mean is unspecified.

    See Also
    --------
    AbstractGraphNormalityGofStatistic : Graph construction and calibration.

    Notes
    -----
    Counts the connected components of the sample proximity graph using
    depth-first traversal.

    The null hypothesis is normality with arbitrary mean and variance
    `var`; `hypothesis().parameters()` returns `{"var": var}`.
    The alternative is two-sided. The graph threshold and its dependence
    on sample scale are described in `AbstractGraphNormalityGofStatistic`.

    This is the PySATL implementation [1]_; an originating research article
    has not been identified.

    References
    ----------
    .. [1] PySATL. "pysatl-criterion", normal proximity-graph implementation.
       https://github.com/PySATL/pysatl-criterion
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
        Finite positive null variance, specified as a keyword argument.
        Default is 1. The null mean is unspecified.

    See Also
    --------
    AbstractGraphNormalityGofStatistic : Graph construction and calibration.

    Notes
    -----
    Uses a sorted sliding window to estimate the maximum clique size in
    the sample proximity graph. The current implementation sorts the input
    sample in place.

    The null hypothesis is normality with arbitrary mean and variance
    `var`; `hypothesis().parameters()` returns `{"var": var}`.
    The alternative is two-sided. The graph threshold and its dependence
    on sample scale are described in `AbstractGraphNormalityGofStatistic`.

    This is the PySATL implementation [1]_; an originating research article
    has not been identified.

    References
    ----------
    .. [1] PySATL. "pysatl-criterion", normal proximity-graph implementation.
       https://github.com/PySATL/pysatl-criterion
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
        Finite positive null variance, specified as a keyword argument.
        Default is 1. The null mean is unspecified.

    See Also
    --------
    AbstractGraphNormalityGofStatistic : Graph construction and calibration.

    Notes
    -----
    Greedily selects separated observations in sorted order to compute the
    maximum independent-set size in the sample proximity graph. The
    implementation sorts a copy and preserves the input sample.

    The null hypothesis is normality with arbitrary mean and variance
    `var`; `hypothesis().parameters()` returns `{"var": var}`.
    The alternative is two-sided. The graph threshold and its dependence
    on sample scale are described in `AbstractGraphNormalityGofStatistic`.

    This is the PySATL implementation [1]_; an originating research article
    has not been identified.

    References
    ----------
    .. [1] PySATL. "pysatl-criterion", normal proximity-graph implementation.
       https://github.com/PySATL/pysatl-criterion
    """

    @staticmethod
    @override
    def code():
        parent_code = AbstractGraphNormalityGofStatistic.code()
        short_code = GraphIndependenceNumberNormalityGofStatistic.short_code()
        return f"{short_code}_{parent_code}"
