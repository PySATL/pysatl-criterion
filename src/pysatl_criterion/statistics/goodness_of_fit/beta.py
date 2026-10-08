"""Goodness-of-fit statistics on the fixed Beta support [0, 1].

Samples must be real-valued; masked observations are rejected, not discarded.

Numerical reductions use cached Numba kernels without fast-math. SciPy handles
special functions and fitting; Decimal retains guard digits for the theoretical
skewness/kurtosis covariance near singular shape limits.

KS, AD, CvM, Watson, Kuiper, Pearson and Neyman use specified shape parameters.
Ebner-Liebenberg and Raschke fit both shapes inside execute_statistic.
MB, SK, Ratio, Entropy and Mode are locally defined discrepancies: no scientific
source for their exact formulas was identified. Their docstrings state their
mathematical meaning without attributing them to unrelated published tests.
"""

from abc import ABC
from decimal import Decimal, localcontext

import numpy as np
import scipy.stats as scipy_stats
from numba import jit
from scipy.optimize import minimize_scalar
from scipy.special import betainc, betaln, hyp2f1
from typing_extensions import override

from pysatl_criterion.distribution.distributions import BetaDistributionDescriptor as Beta
from pysatl_criterion.distribution.distributions import UniformDistributionDescriptor as Uniform
from pysatl_criterion.distribution.parameters import HypothesisSupport, ParameterValues
from pysatl_criterion.statistics import AbstractGoodnessOfFitStatistic
from pysatl_criterion.statistics.alternative import Alternative, AlternativeType, RightAlternative
from pysatl_criterion.statistics.goodness_of_fit.common import (
    ADStatistic,
    Chi2Statistic,
    CrammerVonMisesStatistic,
    KSStatistic,
)
from pysatl_criterion.statistics.goodness_of_fit.uniform import NeymanSmoothTestUniformGofStatistic


class AbstractBetaGofStatistic(AbstractGoodnessOfFitStatistic, ABC):
    """
    Abstract base class for Beta distribution goodness-of-fit statistics.

    The Beta distribution is a continuous probability distribution defined on the interval [0, 1]
    parameterized by two positive shape parameters, denoted by α (alpha) and β (beta).
    Subclasses declare which parameters are fixed by their hypothesis.
    """

    @staticmethod
    def _beta_logtails(x, a, b):
        """Beta log tails, retaining interior probabilities below float64 range.

        On underflow use DLMF 8.17.8 in log form. Its hypergeometric series
        has positive terms, avoiding the cancellation in DLMF 8.17.7.
        Exact endpoints retain their mathematically infinite log tails.
        """
        logcdf = scipy_stats.beta.logcdf(x, a, b)
        logsf = scipy_stats.beta.logsf(x, a, b)
        for logs, points, first, second in ((logcdf, x, a, b), (logsf, 1 - x, b, a)):
            missing = np.isneginf(logs) & (points > 0) & (points < 1)
            t = points[missing]
            if t.size:
                factor = hyp2f1(first + second, 1, first + 1, t)
                if np.any(~np.isfinite(factor)) or np.any(factor <= 0):
                    raise ValueError("Beta log tail could not be evaluated with finite precision")
                logs[missing] = (
                    first * np.log(t)
                    + second * np.log1p(-t)
                    - np.log(first)
                    - betaln(first, second)
                    + np.log(factor)
                )
                if np.any(~np.isfinite(logs[missing])) or np.any(logs[missing] > 0):
                    raise ValueError("Beta log tail could not be evaluated with finite precision")
        if np.any(np.isnan(logcdf)) or np.any(np.isnan(logsf)):
            raise ValueError("Beta log tails could not be evaluated with finite precision")
        return logcdf, logsf

    @staticmethod
    @jit(nopython=True, cache=True)
    def _ad_kernel(logcdf, logsf):
        n = len(logcdf)
        total = 0.0
        for i in range(n):
            total += (2 * i + 1.0) / n * (logcdf[i] + logsf[n - 1 - i])
        return -n - total

    def _validate_storage_calibration(self):
        """Reject shape-dependent composite laws and settings missing from storage keys."""
        if not self.hypothesis().parameters():
            raise ValueError(
                "Composite Beta calibration depends on fitted shapes; "
                "use parametric_bootstrap_beta with the observed sample"
            )
        if (
            getattr(self, "alternative_type", AlternativeType.TWO_TAILED)
            != AlternativeType.TWO_TAILED
            or getattr(self, "lambda_", 1) != 1
            or getattr(self, "k", 4) != 4
        ):
            raise ValueError(
                "Stored Beta calibration does not identify these settings; "
                "use MonteCarloLimitDistributionResolver"
            )

    @staticmethod
    def _validate_rvs(rvs, min_size=1):
        if np.ma.isMaskedArray(rvs) and np.any(np.ma.getmaskarray(rvs)):
            raise ValueError("Sample must not contain masked observations")
        rvs = np.asarray(rvs)
        if np.iscomplexobj(rvs):
            raise ValueError("Sample values must be real")
        rvs = np.asarray(rvs, dtype=float)
        if rvs.ndim != 1 or rvs.size < min_size:
            raise ValueError(f"Sample must be one-dimensional with at least {min_size} values")
        if not np.all(np.isfinite(rvs)):
            raise ValueError("Sample values must be finite")
        # All values in [0, 1]
        if np.any((rvs < 0) | (rvs > 1)):
            raise ValueError("Beta distribution values must be in the interval [0, 1]")
        return rvs

    @classmethod
    def _fit(cls, rvs):
        sample = cls._validate_rvs(rvs, min_size=2)
        if np.any((sample <= 0) | (sample >= 1)):
            raise ValueError("Beta maximum likelihood requires observations strictly in (0, 1)")
        if np.ptp(sample) == 0:
            raise ValueError("Beta maximum likelihood requires a nonconstant sample")
        alpha, beta, _, _ = scipy_stats.beta.fit(sample, floc=0, fscale=1)
        if not np.isfinite(alpha) or not np.isfinite(beta) or alpha <= 0 or beta <= 0:
            raise ValueError("Beta maximum likelihood did not produce finite positive shapes")
        return sample, float(alpha), float(beta)

    @staticmethod
    @override
    def distribution() -> type[Beta]:
        """Return the distribution descriptor class."""
        return Beta

    @classmethod
    @override
    def code(cls) -> str:
        """Return the identifier using the concrete statistic's short code."""
        family_code = f"BETA_{AbstractGoodnessOfFitStatistic.code()}"
        if "short_code" in cls.__abstractmethods__:
            return family_code
        return f"{cls.short_code()}_{family_code}"


class AbstractSpecifiedBetaGofStatistic(AbstractBetaGofStatistic, ABC):
    """Base for Beta statistics with both shape parameters fixed by the hypothesis."""

    @property
    def alpha(self) -> float:
        """Read the first shape by its stable parameter identity."""
        return self._parameters[Beta.ALPHA]

    @property
    def beta(self) -> float:
        """Read the second shape by its stable parameter identity."""
        return self._parameters[Beta.BETA]

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Beta.DEFAULT, frozenset(Beta.DEFAULT.parameters)),)


class KolmogorovSmirnovBetaGofStatistic(AbstractSpecifiedBetaGofStatistic, KSStatistic):
    """Kolmogorov-Smirnov distance to a fully specified Beta distribution.

    Parameters
    ----------
    parameters : ParameterValues
        Beta.DEFAULT values with both positive shape parameters fixed.
    alternative_type : AlternativeType, optional
        TWO_TAILED (default) selects D, RIGHT selects D+, LEFT selects D-.
    mode : str, optional
        Legacy option, default 'auto'. Does not change the distance or
        compute a p-value.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar discrepancy; reject for large values.
    hypothesis()
        Return the null hypothesis parameter specification.

    Notes
    -----
    For ordered observations let u_i = F_{alpha,beta}(x_(i)). Then
    D+ = max_i(i/n-u_i), D- = max_i(u_i-(i-1)/n), and D = max(D+,D-).
    ``alternative_type`` selects D, D+, or D-; ``alternative()`` always
    selects the right rejection tail. Neither shape is estimated.

    This is the classical continuous-null statistic [1]_ applied through
    the specified Beta CDF. Its finite-sample null law is shape-free by
    the probability integral transform. It is not the fitted-Beta test;
    use EbnerLiebenbergBetaGofStatistic when both shapes are unknown.

    References
    ----------
    .. [1] N. Smirnov (1948), "Table for Estimating the Goodness of Fit
       of Empirical Distributions", Ann. Math. Statist. 19, 279-281.
       https://doi.org/10.1214/aoms/1177730256

    Examples
    --------
    >>> statistic = KolmogorovSmirnovBetaGofStatistic(Beta.DEFAULT.parse({"a": 2, "b": 5}))
    >>> value = statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
    >>> bool(value >= 0)
    True
    """

    def __init__(
        self,
        parameters: ParameterValues,
        *,
        alternative_type: AlternativeType = AlternativeType.TWO_TAILED,
        mode="auto",
    ):
        AbstractSpecifiedBetaGofStatistic.__init__(self, parameters)
        if not isinstance(alternative_type, AlternativeType):
            raise TypeError("alternative_type must be an AlternativeType")
        KSStatistic.__init__(self, alternative_type, mode)

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier.

        Returns
        -------
        code : str
            Stable identifier for this statistic.
        """
        return "KS"

    @override
    def execute_statistic(self, rvs, **kwargs) -> float | np.float64:
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional finite real sample in [0, 1], with at least one value.
        **kwargs : dict, optional
            Reserved for interface compatibility; ignored.

        Returns
        -------
        statistic : float
            Nonnegative discrepancy; larger values oppose the null hypothesis.

        Raises
        ------
        ValueError
            If the sample violates the class constraints.

        Notes
        -----
        See the class Notes for the formula, null hypothesis and calibration.
        """
        rvs = self._validate_rvs(rvs)
        cdf = scipy_stats.beta.cdf(np.sort(rvs), self.alpha, self.beta)
        direction = 1 if self.alternative_type == AlternativeType.RIGHT else -1
        if self.alternative_type == AlternativeType.TWO_TAILED:
            direction = 0
        return self._statistic_kernel(cdf, direction)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _statistic_kernel(cdf, direction):
        n = len(cdf)
        plus, minus = 0.0, 0.0
        for i in range(n):
            plus = max(plus, (i + 1) / n - cdf[i])
            minus = max(minus, cdf[i] - i / n)
        if direction == 1:
            return plus
        if direction == -1:
            return minus
        return max(plus, minus)


class AndersonDarlingBetaGofStatistic(AbstractSpecifiedBetaGofStatistic, ADStatistic):
    """Anderson-Darling statistic for a fully specified Beta distribution.

    Parameters
    ----------
    parameters : ParameterValues
        Beta.DEFAULT values with both positive shape parameters fixed.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar discrepancy; reject for large values.
    hypothesis()
        Return the null hypothesis parameter specification.

    Notes
    -----
    With u_i = F_{alpha,beta}(x_(i)), compute
    A2 = -n - sum_i (2*i-1)/n * (log(u_i) + log(1-u_(n+1-i))).
    Log-CDF and log-survival evaluations are used directly. Observations
    at 0 or 1 yield positive infinity, as the mathematical formula requires.

    This is the statistic in [1]_ applied to the specified Beta CDF,
    without finite-sample corrections for estimated parameters. Both
    shapes are known, and the null law is shape-free after transformation.

    References
    ----------
    .. [1] T. W. Anderson and D. A. Darling (1954), "A Test of Goodness
       of Fit", J. Amer. Statist. Assoc. 49, 765-769.
       https://doi.org/10.1080/01621459.1954.10501232

    Examples
    --------
    >>> statistic = AndersonDarlingBetaGofStatistic(Beta.DEFAULT.parse({"a": 2, "b": 5}))
    >>> value = statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
    >>> bool(value >= 0)
    True
    """

    @staticmethod
    @override
    def short_code():
        return "AD"

    @override
    def execute_statistic(self, rvs, **kwargs) -> float | np.float64:
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional finite real sample in [0, 1], with at least one value.
        **kwargs : dict, optional
            Reserved for interface compatibility; ignored.

        Returns
        -------
        statistic : float
            Nonnegative discrepancy; larger values oppose the null hypothesis. May be +inf.

        Raises
        ------
        ValueError
            If the sample violates the class constraints.

        Notes
        -----
        See the class Notes for the formula, null hypothesis and calibration.
        """
        rvs = self._validate_rvs(rvs)
        logcdf, logsf = self._beta_logtails(np.sort(rvs), self.alpha, self.beta)
        return self._ad_kernel(logcdf, logsf)


class CrammerVonMisesBetaGofStatistic(AbstractSpecifiedBetaGofStatistic, CrammerVonMisesStatistic):
    """Cramer-von Mises statistic for a fully specified Beta distribution.

    Parameters
    ----------
    parameters : ParameterValues
        Beta.DEFAULT values with both positive shape parameters fixed.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar discrepancy; reject for large values.
    hypothesis()
        Return the null hypothesis parameter specification.

    Notes
    -----
    For u_i = F_{alpha,beta}(x_(i)), compute
    W2 = 1/(12*n) + sum_i (u_i-(2*i-1)/(2*n))**2.
    This is the one-sample statistic studied in [1]_, evaluated after the
    specified Beta CDF transform. Both shapes are known; neither is fitted.
    The null law is shape-free. The legacy spelling ``Crammer`` is retained
    in the class name for compatibility.

    References
    ----------
    .. [1] S. Csorgo and J. J. Faraway (1996), "The Exact and Asymptotic
       Distributions of Cramer-von Mises Statistics", JRSS B 58, 221-234.
       https://doi.org/10.1111/j.2517-6161.1996.tb02077.x

    Examples
    --------
    >>> statistic = CrammerVonMisesBetaGofStatistic(Beta.DEFAULT.parse({"a": 2, "b": 5}))
    >>> value = statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
    >>> bool(value >= 0)
    True
    """

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier.

        Returns
        -------
        code : str
            Stable identifier for this statistic.
        """
        return "CVM"

    @override
    def execute_statistic(self, rvs, **kwargs) -> float | np.float64:
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional finite real sample in [0, 1], with at least one value.
        **kwargs : dict, optional
            Reserved for interface compatibility; ignored.

        Returns
        -------
        statistic : float
            Nonnegative discrepancy; larger values oppose the null hypothesis.

        Raises
        ------
        ValueError
            If the sample violates the class constraints.

        Notes
        -----
        See the class Notes for the formula, null hypothesis and calibration.
        """
        rvs = self._validate_rvs(rvs)
        return self._statistic_kernel(scipy_stats.beta.cdf(np.sort(rvs), self.alpha, self.beta))

    @staticmethod
    @jit(nopython=True, cache=True)
    def _statistic_kernel(cdf):
        n = len(cdf)
        value = 1.0 / (12 * n)
        for i in range(n):
            value += ((2 * i + 1) / (2 * n) - cdf[i]) ** 2
        return value


class Chi2PearsonBetaGofStatistic(AbstractSpecifiedBetaGofStatistic, Chi2Statistic):
    """Pearson statistic for binned observations under a specified Beta null.

    Parameters
    ----------
    parameters : ParameterValues
        Beta.DEFAULT values with both positive shape parameters fixed.
    lambda_ : float, optional
        Finite power-divergence parameter. Default 1 gives Pearson's test.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar discrepancy; reject for large values.
    hypothesis()
        Return the null hypothesis parameter specification.

    Notes
    -----
    Use k=ceil(sqrt(n)) equal-width bins covering [0, 1], counts O_j,
    and E_j=n*(F(b_j)-F(a_j)). For lambda_=1 return sum((O_j-E_j)**2/E_j),
    the Pearson statistic [1]_. For other lambda_ values return the
    Cressie-Read power divergence [2]_, with its limits at 0 and -1.

    Both shapes are known. Upper-tail probabilities use survival-function
    differences. Bins with small expected counts are not pooled. Since k
    grows with n, chi-square(k-1) is not asserted as an accurate automatic
    calibration. Simulate with the same n, shapes and lambda_. A numerically
    zero expected probability is rejected explicitly by the common routine.
    The references concern multinomial tests; Beta determines their cell
    probabilities here, not a separate Beta-specific Pearson formula.

    References
    ----------
    .. [1] K. Pearson (1900), "On the criterion that a given system of
       deviations from the probable in the case of a correlated system of
       variables is such that it can be reasonably supposed to have arisen
       from random sampling", Philosophical Magazine 50, 157-175.
       https://doi.org/10.1080/14786440009463897
    .. [2] N. Cressie and T. R. C. Read (1984), "Multinomial Goodness-of-Fit
       Tests", JRSS B 46, 440-464.
       https://doi.org/10.1111/j.2517-6161.1984.tb01318.x

    Examples
    --------
    >>> statistic = Chi2PearsonBetaGofStatistic(Beta.DEFAULT.parse({"a": 2, "b": 5}))
    >>> value = statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
    >>> bool(value >= 0)
    True
    """

    def __init__(self, parameters: ParameterValues, *, lambda_=1):
        AbstractSpecifiedBetaGofStatistic.__init__(self, parameters)
        Chi2Statistic.__init__(self)
        if np.ndim(lambda_) != 0 or np.iscomplexobj(lambda_) or not np.isfinite(lambda_):
            raise ValueError("lambda_ must be a finite real scalar")
        self.lambda_ = lambda_

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier.

        Returns
        -------
        code : str
            Stable identifier for this statistic.
        """
        return "CHI2_PEARSON"

    @override
    def execute_statistic(self, rvs, **kwargs) -> float | np.float64:
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional finite real sample in [0, 1], with at least one value.
        **kwargs : dict, optional
            Reserved for interface compatibility; ignored.

        Returns
        -------
        statistic : float
            Nonnegative discrepancy; larger values oppose the null hypothesis. May be +inf.

        Raises
        ------
        ValueError
            If the sample violates the class constraints.

        Notes
        -----
        See the class Notes for the formula, null hypothesis and calibration.
        """
        rvs = self._validate_rvs(rvs)

        rvs_sorted = np.sort(rvs)
        n = len(rvs_sorted)

        # Number of bins using the square-root rule
        num_bins = int(np.ceil(np.sqrt(n)))

        # Create histogram
        observed, bin_edges = np.histogram(rvs_sorted, bins=num_bins, range=(0, 1))

        # Calculate expected frequencies
        expected_cdf = scipy_stats.beta.cdf(bin_edges, self.alpha, self.beta)
        expected_sf = scipy_stats.beta.sf(bin_edges, self.alpha, self.beta)
        probabilities = np.where(
            expected_cdf[:-1] < 0.5, np.diff(expected_cdf), -np.diff(expected_sf)
        )
        expected = probabilities * n

        observed, expected = self._validate_frequencies(observed, expected)
        return self._statistic_kernel(observed, expected, float(self.lambda_))

    @staticmethod
    @jit(nopython=True, cache=True, error_model="numpy")
    def _statistic_kernel(observed, expected, power):
        total = 0.0
        for i in range(len(observed)):
            obs, exp = observed[i], expected[i]
            if obs == 0 and power <= -1:
                return np.inf
            if power == 1:
                total += (obs - exp) ** 2 / exp
            elif obs > 0:
                if power == 0:
                    total += 2 * obs * np.log(obs / exp)
                elif power == -1:
                    total += 2 * exp * np.log(exp / obs)
                else:
                    total += obs * ((obs / exp) ** power - 1) / (0.5 * power * (power + 1))
        return total


class WatsonBetaGofStatistic(AbstractSpecifiedBetaGofStatistic):
    """Watson U-squared statistic after a specified Beta CDF transform.

    Parameters
    ----------
    parameters : ParameterValues
        Beta.DEFAULT values with both positive shape parameters fixed.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar discrepancy; reject for large values.
    hypothesis()
        Return the null hypothesis parameter specification.

    Notes
    -----
    For u_i=F_{alpha,beta}(x_(i)), compute
    W2=1/(12*n)+sum_i(u_i-(2*i-1)/(2*n))**2 and
    U2=W2-n*(mean(u)-1/2)**2.
    This is Watson's statistic [1]_ applied to transformed uniform data.
    Its circular rotation invariance concerns u modulo 1, not arbitrary
    location shifts of observations on the original Beta support.
    Both shapes are specified, and the null law is shape-free. The cited
    paper is about the classical circular statistic, not a fitted-Beta test.

    References
    ----------
    .. [1] G. S. Watson (1961), "Goodness-of-fit tests on a circle",
       Biometrika 48, 109-114.
       https://doi.org/10.1093/biomet/48.1-2.109

    Examples
    --------
    >>> statistic = WatsonBetaGofStatistic(Beta.DEFAULT.parse({"a": 2, "b": 5}))
    >>> value = statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
    >>> bool(value >= 0)
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier.

        Returns
        -------
        code : str
            Stable identifier for this statistic.
        """
        return "W"

    @override
    def execute_statistic(self, rvs, **kwargs) -> float | np.float64:
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional finite real sample in [0, 1], with at least one value.
        **kwargs : dict, optional
            Reserved for interface compatibility; ignored.

        Returns
        -------
        statistic : float
            Nonnegative discrepancy; larger values oppose the null hypothesis.

        Raises
        ------
        ValueError
            If the sample violates the class constraints.

        Notes
        -----
        See the class Notes for the formula, null hypothesis and calibration.
        """
        rvs = self._validate_rvs(rvs)
        return self._statistic_kernel(scipy_stats.beta.cdf(np.sort(rvs), self.alpha, self.beta))

    @staticmethod
    @jit(nopython=True, cache=True)
    def _statistic_kernel(cdf):
        n = len(cdf)
        value = 1.0 / (12 * n)
        for i in range(n):
            value += ((2 * i + 1) / (2 * n) - cdf[i]) ** 2
        return value - n * (np.mean(cdf) - 0.5) ** 2


class KuiperBetaGofStatistic(AbstractSpecifiedBetaGofStatistic):
    """Kuiper statistic after a specified Beta CDF transform.

    Parameters
    ----------
    parameters : ParameterValues
        Beta.DEFAULT values with both positive shape parameters fixed.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar discrepancy; reject for large values.
    hypothesis()
        Return the null hypothesis parameter specification.

    Notes
    -----
    For u_i=F_{alpha,beta}(x_(i)), compute
    V=max_i(i/n-u_i)+max_i(u_i-(i-1)/n).
    This is the unscaled Kuiper statistic [1]_; no sqrt(n) factor or
    finite-sample correction is included. Both shapes are specified.
    The probability integral transform gives its shape-free null law.
    Rotation invariance concerns transformed u modulo 1. The source
    studies the circular uniformity statistic, not estimated Beta shapes.

    References
    ----------
    .. [1] N. H. Kuiper (1960), "Tests concerning random points on a
       circle", Proc. K. Ned. Akad. Wet. A 63, 38-47.
       https://doi.org/10.1016/S1385-7258(60)50006-0

    Examples
    --------
    >>> statistic = KuiperBetaGofStatistic(Beta.DEFAULT.parse({"a": 2, "b": 5}))
    >>> value = statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
    >>> bool(value >= 0)
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier.

        Returns
        -------
        code : str
            Stable identifier for this statistic.
        """
        return "KUIPER"

    @override
    def execute_statistic(self, rvs, **kwargs) -> float | np.float64:
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional finite real sample in [0, 1], with at least one value.
        **kwargs : dict, optional
            Reserved for interface compatibility; ignored.

        Returns
        -------
        statistic : float
            Nonnegative discrepancy; larger values oppose the null hypothesis.

        Raises
        ------
        ValueError
            If the sample violates the class constraints.

        Notes
        -----
        See the class Notes for the formula, null hypothesis and calibration.
        """
        rvs = self._validate_rvs(rvs)
        return self._statistic_kernel(scipy_stats.beta.cdf(np.sort(rvs), self.alpha, self.beta))

    @staticmethod
    @jit(nopython=True, cache=True)
    def _statistic_kernel(cdf):
        n = len(cdf)
        plus, minus = 0.0, 0.0
        for i in range(n):
            plus = max(plus, (i + 1) / n - cdf[i])
            minus = max(minus, cdf[i] - i / n)
        return plus + minus


class MomentBasedBetaGofStatistic(AbstractSpecifiedBetaGofStatistic):
    """Covariance-standardized mean/variance discrepancy for a specified Beta null.

    Parameters
    ----------
    parameters : ParameterValues
        Beta.DEFAULT values with both positive shape parameters fixed.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar discrepancy; reject for large values.
    hypothesis()
        Return the null hypothesis parameter specification.

    Notes
    -----
    For d=(mean(X)-mu, S2-sigma2), with S2 the unbiased sample variance,
    return T=n*d.T @ solve(Sigma,d), where
    Sigma=[[sigma2, mu3], [mu3, mu4-sigma2**2]] and mu_r are Beta central
    moments. Computation uses standardized moments to reduce cancellation.
    Both shapes are fixed. For fixed positive shapes T has an asymptotic
    chi-square(2) law; finite-sample calibration still depends on shapes.

    No publication describing this exact Beta statistic was identified.
    This is a locally defined moment discrepancy, not a claimed named
    published Beta criterion. It need not detect alternatives with matching
    mean and variance. At least two observations are required.

    Examples
    --------
    >>> statistic = MomentBasedBetaGofStatistic(Beta.DEFAULT.parse({"a": 2, "b": 5}))
    >>> value = statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
    >>> bool(value >= 0)
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier.

        Returns
        -------
        code : str
            Stable identifier for this statistic.
        """
        return "MB"

    @override
    def execute_statistic(self, rvs, **kwargs) -> float | np.float64:
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional finite sample in [0, 1], with at least 2 values.
        **kwargs : dict, optional
            Reserved for interface compatibility; ignored.

        Returns
        -------
        statistic : float
            Nonnegative discrepancy; larger values oppose the null hypothesis.

        Raises
        ------
        ValueError
            If the sample violates the class constraints.

        Notes
        -----
        See the class Notes for the formula, null hypothesis and calibration.
        """
        rvs = self._validate_rvs(rvs, min_size=2)
        mean, variance = scipy_stats.beta.stats(self.alpha, self.beta, moments="mv")
        skewness = scipy_stats.beta.stats(self.alpha, self.beta, moments="s")
        total = self.alpha + self.beta
        residual = 2 * (total / (total + 3)) * (1 + (skewness / 2) ** 2)
        value = self._statistic_kernel(rvs, float(mean), float(variance), float(skewness), residual)
        if not np.isfinite(value) or residual <= 0:
            raise ValueError("Beta moment statistic could not be evaluated with finite precision")
        return float(value)

    @staticmethod
    @jit(nopython=True, cache=True, error_model="numpy")
    def _statistic_kernel(rvs, mean, variance, skewness, residual):
        n = len(rvs)
        sample_mean = np.mean(rvs)
        sample_variance = np.sum((rvs - sample_mean) ** 2) / (n - 1)
        d0 = (sample_mean - mean) / np.sqrt(variance)
        d1 = sample_variance / variance - 1
        return n * (d0**2 + ((d1 - skewness * d0) / np.sqrt(residual)) ** 2)


class SkewnessKurtosisBetaGofStatistic(AbstractSpecifiedBetaGofStatistic):
    """Covariance-standardized skewness/excess discrepancy for a specified Beta null.

    Parameters
    ----------
    parameters : ParameterValues
        Beta.DEFAULT values with both positive shape parameters fixed.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar discrepancy; reject for large values.
    hypothesis()
        Return the null hypothesis parameter specification.

    Notes
    -----
    Let Z=(X-mu)/sigma, g=E[Z**3], k=E[Z**4]. The influence functions
    are p3=Z**3-3*Z-1.5*g*Z**2+g/2 and
    p4=Z**4-4*g*Z-2*k*Z**2+k. Set Sigma_ij=E[p_i*p_j] using Beta
    moments through order eight. Return n*d.T @ solve(Sigma,d), where d
    contains sample skewness minus g and sample excess minus (k-3).
    Sample estimates use bias=False after affine rescaling for numerical
    stability. This is not the normal-model
    Jarque-Bera statistic. Both shapes are fixed; the asymptotic null law
    is chi-square(2), not an exact finite-sample law.

    No publication describing this exact Beta statistic was identified.
    It is a locally defined discrepancy and need not detect distributions
    with matching skewness and excess. At least four observations and a
    nonconstant sample are required.

    Examples
    --------
    >>> statistic = SkewnessKurtosisBetaGofStatistic(Beta.DEFAULT.parse({"a": 2, "b": 5}))
    >>> value = statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
    >>> bool(value >= 0)
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier.

        Returns
        -------
        code : str
            Stable identifier for this statistic.
        """
        return "SK"

    @override
    def execute_statistic(self, rvs, **kwargs) -> float | np.float64:
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional finite sample in [0, 1], with at least 4 values.
        **kwargs : dict, optional
            Reserved for interface compatibility; ignored.

        Returns
        -------
        statistic : float
            Nonnegative discrepancy; larger values oppose the null hypothesis.

        Raises
        ------
        ValueError
            If the sample violates the class constraints.

        Notes
        -----
        See the class Notes for the formula, null hypothesis and calibration.
        """
        rvs = self._validate_rvs(rvs, min_size=4)
        if np.ptp(rvs) == 0:
            raise ValueError("Skewness and kurtosis require a nonconstant sample")
        skewness, excess = self._sample_moments(rvs)
        value = len(rvs) * self._skewness_kurtosis_quadratic(
            self.alpha, self.beta, skewness, excess
        )
        if not np.isfinite(value):
            raise ValueError("Beta moment statistic could not be evaluated with finite precision")
        return float(value)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _sample_moments(rvs):
        n = len(rvs)
        scaled = (rvs - np.min(rvs)) / (np.max(rvs) - np.min(rvs))
        centered = scaled - np.mean(scaled)
        m2 = np.mean(centered**2)
        m3 = np.mean(centered**3)
        m4 = np.mean(centered**4)
        skewness = np.sqrt(n * (n - 1)) / (n - 2) * m3 / m2**1.5
        excess = ((n * n - 1) * m4 / m2**2 - 3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
        return skewness, excess

    @staticmethod
    def _skewness_kurtosis_quadratic(a, b, sample_skewness, sample_excess):
        """Evaluate the same influence-function quadratic with guard digits.

        Near the Bernoulli limit, covariance terms cancel to O(a+b). Float64
        arithmetic can turn this positive covariance indefinite. Decimal is from
        the standard library; precision scales with the shape exponents. All
        operations through the Schur complement stay in that precision.
        """
        a, b = Decimal(float(a)), Decimal(float(b))
        with localcontext() as context:
            context.prec = 80 + 2 * max(abs(a.adjusted()), abs(b.adjusted()))
            total = a + b
            mean = a / total
            variance = a * b / (total**2 * (total + 1))
            asymmetry = (1 - 2 * mean) / variance.sqrt()
            moments = [Decimal(1), Decimal(0)]
            for k in range(1, 8):
                moments.append(
                    Decimal(k)
                    / (total + k)
                    * ((total + 1) * moments[k - 1] + asymmetry * moments[k])
                )
            g, kurtosis = moments[3:5]
            polynomials = [
                [g / 2, Decimal(-3), -3 * g / 2, Decimal(1)],
                [kurtosis, -4 * g, -2 * kurtosis, Decimal(0), Decimal(1)],
            ]
            covariance = [
                [
                    sum(
                        c * d * moments[i + j]
                        for i, c in enumerate(left)
                        for j, d in enumerate(right)
                    )
                    for right in polynomials
                ]
                for left in polynomials
            ]
            v00, v01 = covariance[0]
            if v00 <= 0:
                raise ValueError("Beta moment covariance lost numerical precision")
            residual = covariance[1][1] - v01**2 / v00
            if residual <= 0:
                raise ValueError("Beta moment covariance lost numerical precision")
            d0 = Decimal(float(sample_skewness)) - g
            d1 = Decimal(float(sample_excess)) - (kurtosis - 3)
            return float(d0**2 / v00 + (d1 - v01 / v00 * d0) ** 2 / residual)


class RatioBetaGofStatistic(AbstractSpecifiedBetaGofStatistic):
    """Geometric/arithmetic mean discrepancy for a specified Beta null.

    Parameters
    ----------
    parameters : ParameterValues
        Beta.DEFAULT values with both positive shape parameters fixed.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar discrepancy; reject for large values.
    hypothesis()
        Return the null hypothesis parameter specification.

    Notes
    -----
    Let R=exp(mean(log(X)))/mean(X) and
    r=exp(psi(alpha)-psi(alpha+beta))/(alpha/(alpha+beta)).
    Return sqrt(n)*abs(R-r). Here r is a probability limit, not the
    finite-sample expectation of R. Both shapes are fixed. No variance
    normalization or universal null law is supplied; simulate calibration
    for both shapes and n. A zero observation gives geometric mean zero;
    an all-zero sample is rejected because the ratio is undefined. The sample
    ratio is evaluated after scaling by the maximum, using log differences.

    No publication describing this exact Beta statistic was identified.
    It is a locally defined discrepancy, not an established named Beta
    criterion, and need not detect alternatives with the same ratio.

    Examples
    --------
    >>> statistic = RatioBetaGofStatistic(Beta.DEFAULT.parse({"a": 2, "b": 5}))
    >>> value = statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
    >>> bool(value >= 0)
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier.

        Returns
        -------
        code : str
            Stable identifier for this statistic.
        """
        return "RT"

    @override
    def execute_statistic(self, rvs, **kwargs) -> float | np.float64:
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional finite real sample in [0, 1], with at least one value.
        **kwargs : dict, optional
            Reserved for interface compatibility; ignored.

        Returns
        -------
        statistic : float
            Nonnegative discrepancy; larger values oppose the null hypothesis.

        Raises
        ------
        ValueError
            If the sample violates the class constraints.

        Notes
        -----
        See the class Notes for the formula, null hypothesis and calibration.
        """
        rvs = self._validate_rvs(rvs)
        from scipy.special import psi

        theoretical_mean = self.alpha / (self.alpha + self.beta)
        theoretical_log_mean = psi(self.alpha) - psi(self.alpha + self.beta)
        theoretical_ratio = np.exp(theoretical_log_mean) / theoretical_mean
        return self._statistic_kernel(rvs, theoretical_ratio)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _statistic_kernel(rvs, theoretical_ratio):
        scale = np.max(rvs)
        if scale == 0:
            raise ValueError("Arithmetic mean is zero, cannot compute ratio")
        arithmetic_mean = np.mean(rvs / scale)
        log_geometric_mean = np.mean(np.log(rvs) - np.log(scale))
        sample_ratio = np.exp(log_geometric_mean - np.log(arithmetic_mean))
        return np.sqrt(len(rvs)) * np.abs(sample_ratio - theoretical_ratio)


class EntropyBetaGofStatistic(AbstractSpecifiedBetaGofStatistic):
    """Vasicek-entropy discrepancy for a specified Beta null.

    Parameters
    ----------
    parameters : ParameterValues
        Beta.DEFAULT values with both positive shape parameters fixed.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar discrepancy; reject for large values.
    hypothesis()
        Return the null hypothesis parameter specification.

    Notes
    -----
    For ordered x and endpoint-replicated indices, estimate entropy as
    H_m=mean_i(log(n*(x_(i+m)-x_(i-m))/(2*m))). Return
    sqrt(n)*abs(H_m-H_Beta), where H_Beta is the theoretical differential
    entropy. Zero spacings give H_m=-infinity and T=+infinity.
    The default m=min(int(sqrt(n)+0.5),(n-1)//2) grows while m/n tends to
    zero. Require n>=3 and an integer 1<=m<n/2. Both shapes are fixed.
    Calibrate for n, both shapes and the same window setting; sqrt(n)
    is not a claim of a universal asymptotic law.

    No publication describing this exact Beta discrepancy was identified.
    Using the Vasicek entropy estimator does not make this the published
    Vasicek normality test. No normality-test article is cited as evidence
    for this Beta formula. Equal entropy does not characterize a Beta law.

    Examples
    --------
    >>> statistic = EntropyBetaGofStatistic(Beta.DEFAULT.parse({"a": 2, "b": 5}))
    >>> value = statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
    >>> bool(value >= 0)
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier.

        Returns
        -------
        code : str
            Stable identifier for this statistic.
        """
        return "ENT"

    @override
    def execute_statistic(self, rvs, **kwargs) -> float | np.float64:
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional finite sample in [0, 1], with at least 3 values.
        m : int, optional
            Window passed through **kwargs. Default
            min(int(sqrt(n)+0.5),(n-1)//2); require 1 <= m < n/2.
        **kwargs : dict, optional
            Other keyword arguments are ignored.

        Returns
        -------
        statistic : float
            Nonnegative discrepancy; larger values oppose the null hypothesis. May be +inf.

        Raises
        ------
        ValueError
            If the sample violates the class constraints.
        TypeError
            If m is not an integer.

        Notes
        -----
        See the class Notes for the formula, null hypothesis and calibration.
        """
        rvs = self._validate_rvs(rvs, min_size=3)
        n = len(rvs)
        m = kwargs.get("m", min(int(np.sqrt(n) + 0.5), (n - 1) // 2))
        if isinstance(m, (bool, np.bool_)) or not isinstance(m, (int, np.integer)):
            raise TypeError("m must be an integer with 1 <= m < n/2")
        if not 1 <= m < n / 2:
            raise ValueError("m must be an integer with 1 <= m < n/2")
        # Theoretical entropy
        from scipy.special import betaln, psi

        theoretical_entropy = (
            betaln(self.alpha, self.beta)
            - (self.alpha - 1) * psi(self.alpha)
            - (self.beta - 1) * psi(self.beta)
            + (self.alpha + self.beta - 2) * psi(self.alpha + self.beta)
        )

        # Test statistic
        statistic = self._statistic_kernel(rvs, int(m), theoretical_entropy)

        return statistic

    @staticmethod
    @jit(nopython=True, cache=True)
    def _statistic_kernel(rvs, m, theoretical_entropy):
        x = np.sort(rvs)
        n = len(x)
        total = 0.0
        for i in range(n):
            spacing = x[min(n - 1, i + m)] - x[max(0, i - m)]
            if spacing == 0:
                return np.inf
            total += np.log(n * spacing / (2 * m))
        return np.sqrt(n) * np.abs(total / n - theoretical_entropy)


class ModeBetaGofStatistic(AbstractSpecifiedBetaGofStatistic):
    """KDE-mode discrepancy for a specified unimodal Beta null.

    Parameters
    ----------
    parameters : ParameterValues
        Beta.DEFAULT values with both positive shape parameters fixed.
        Both shapes must be greater than 1 for a unique interior mode.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar discrepancy; reject for large values.
    hypothesis()
        Return the null hypothesis parameter specification.

    Notes
    -----
    The reference mode is (alpha-1)/(alpha+beta-2), alpha,beta>1.
    Estimate a sample mode using Gaussian KDE with Scott bandwidth,
    a grid over the sample range of size max(257,int(sqrt(n))+1), and
    bounded optimization of every detected local peak. Computation uses
    coordinates scaled to [0,1] within the sample range, including both
    sample extrema. A Gaussian KDE mode lies within that range. Return
    sqrt(n)*abs(mode_hat-mode_Beta).
    Both shapes are fixed. The numerical search has no general guarantee
    of finding every peak. KDE boundary bias remains. A constant sample
    or fewer than two observations is rejected.

    No publication describing this exact Beta statistic was identified.
    This is a locally defined discrepancy; sqrt(n) is only a scale
    convention, not a justified universal limit normalization. Simulate
    calibration for n and both shapes. Matching modes do not characterize
    the Beta family.

    Examples
    --------
    >>> statistic = ModeBetaGofStatistic(Beta.DEFAULT.parse({"a": 2, "b": 5}))
    >>> value = statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
    >>> bool(value >= 0)
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    def __init__(self, parameters: ParameterValues):
        super().__init__(parameters)
        if self.alpha <= 1:
            raise ValueError("alpha must be greater than 1 for mode to be well-defined")
        if self.beta <= 1:
            raise ValueError("beta must be greater than 1 for mode to be well-defined")

    @staticmethod
    @override
    def short_code():
        """Return the short statistic identifier.

        Returns
        -------
        code : str
            Stable identifier for this statistic.
        """
        return "MODE"

    @override
    def execute_statistic(self, rvs, **kwargs) -> float | np.float64:
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional finite sample in [0, 1], with at least 2 values.
        **kwargs : dict, optional
            Reserved for interface compatibility; ignored.

        Returns
        -------
        statistic : float
            Nonnegative discrepancy; larger values oppose the null hypothesis.

        Raises
        ------
        ValueError
            If the sample violates the class constraints.

        Notes
        -----
        See the class Notes for the formula, null hypothesis and calibration.
        """
        rvs = self._validate_rvs(rvs, min_size=2)
        if np.ptp(rvs) == 0:
            raise ValueError("KDE mode requires a nonconstant sample")
        n = len(rvs)
        # Scott-bandwidth KDE transforms affinely, including its modes.
        lower, width = rvs.min(), np.ptp(rvs)
        scaled = (rvs - lower) / width
        kde = scipy_stats.gaussian_kde(scaled)
        # Gaussian mixture modes lie inside the sample range. Refine every
        # grid-local maximum, including candidates near either support boundary.
        x_grid = np.linspace(0, 1, max(257, int(np.sqrt(n)) + 1))
        bandwidth = float(np.sqrt(kde.covariance[0, 0]))
        density = self._density_kernel(x_grid, scaled, bandwidth)
        peaks = np.flatnonzero((density[1:-1] >= density[:-2]) & (density[1:-1] >= density[2:])) + 1
        candidates = [0.0, 1.0]
        for i in peaks:
            result = minimize_scalar(
                lambda x: -self._density_kernel(np.array([x]), scaled, bandwidth)[0],
                bounds=(x_grid[i - 1], x_grid[i + 1]),
                method="bounded",
                options={"xatol": 1e-12},
            )
            candidates.append(result.x)
        candidate_density = self._density_kernel(np.asarray(candidates), scaled, bandwidth)
        sample_mode = lower + width * candidates[np.argmax(candidate_density)]

        # Theoretical mode
        theoretical_mode = (self.alpha - 1) / (self.alpha + self.beta - 2)

        # Test statistic
        statistic = np.sqrt(n) * np.abs(sample_mode - theoretical_mode)

        return statistic

    @staticmethod
    @jit(nopython=True, cache=True)
    def _density_kernel(points, sample, bandwidth):
        density = np.empty(len(points))
        normalizer = len(sample) * bandwidth * np.sqrt(2 * np.pi)
        for i in range(len(points)):
            total = 0.0
            for x in sample:
                total += np.exp(-0.5 * ((points[i] - x) / bandwidth) ** 2)
            density[i] = total / normalizer
        return density


class EbnerLiebenbergBetaGofStatistic(AbstractBetaGofStatistic):
    """Conditional-moment Beta goodness-of-fit statistic of Ebner and Liebenberg.

    Parameters
    ----------
    parameters : ParameterValues
        Beta.DEFAULT.parse({}); both shapes unknown, support fixed at [0, 1].

    Notes
    -----
    Fit a and b by maximum likelihood on each call. For c_j=(a+b)*x_j-a,
    compute T_n=n*integral_0^1 (mean_j(c_j*1{x_j>=t})
    - t**a*(1-t)**b/B(a,b))**2 dt, using equation (3) of [1]_.
    Sorting and cumulative sums evaluate the pairwise minimum term in
    O(n log n) time and O(n) space. Beta ratios use log beta functions.

    Reject for large values. The null law depends on the unknown shapes.
    Calibrate by parametric bootstrap from the fitted Beta distribution,
    refitting both shapes for every replicate (Section 2 of [1]_).
    This class returns only the statistic, without p-values or fitted state.

    References
    ----------
    .. [1] B. Ebner and S. C. Liebenberg, "On a new test of fit to the beta
       distribution" (2020), equation (3). https://arxiv.org/abs/2009.13995
       Published in Stat (2021): https://doi.org/10.1002/sta4.341

    Examples
    --------
    >>> statistic = EbnerLiebenbergBetaGofStatistic(Beta.DEFAULT.parse({}))
    >>> value = statistic.execute_statistic([0.1, 0.2, 0.3, 0.5])
    >>> value >= 0
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Beta.DEFAULT, frozenset()),)

    @staticmethod
    @override
    def short_code():
        return "EL"

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @override
    def execute_statistic(self, rvs, **kwargs) -> float:
        """Return T_n for a finite, nonconstant sample in (0, 1), n >= 2.

        Invalid samples raise ValueError; MLE solver failures propagate.
        Additional keyword arguments are ignored for interface compatibility.

        Returns
        -------
        statistic : float
            Nonnegative discrepancy; reject for large values.
        """
        sample, a, b = self._fit(rvs)
        x = np.sort(sample)
        integrals = betainc(a + 1, b + 1, x)
        third = len(x) * np.exp(betaln(2 * a + 1, 2 * b + 1) - 2 * betaln(a, b))
        return self._statistic_kernel(x, a, b, integrals, third)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _statistic_kernel(x, a, b, integrals, third):
        n = len(x)
        tail, first, cross = 0.0, 0.0, 0.0
        for i in range(n - 1, -1, -1):
            c = (a + b) * x[i] - a
            tail += c
            width = x[i] - x[i - 1] if i else x[i]
            first += width * tail**2
            cross += c * integrals[i]
        first /= n
        ratio = (a / (a + b)) * (b / (a + b + 1))
        second = 2 * ratio * cross
        value = first - second + third
        if not np.isfinite(value):
            raise ValueError("Beta statistic could not be evaluated with finite precision")
        tolerance = 64 * np.finfo(np.float64).eps * (abs(first) + abs(second) + abs(third))
        if value < -tolerance:
            raise ValueError("Beta statistic lost numerical precision")
        return max(0.0, value)


class NeymanSmoothBetaGofStatistic(AbstractSpecifiedBetaGofStatistic):
    """Neyman's smooth test applied to the specified Beta probability transform.

    Parameters
    ----------
    parameters : ParameterValues
        Both positive Beta shapes must be fixed independently of the sample.
    k : int, optional
        Number of shifted Legendre components, positive and fixed; default 4.

    Notes
    -----
    With u_i=F_{a,b}(x_i), return sum_{j=1}^k (sum_i phi_j(u_i))**2/n,
    phi_j(u)=sqrt(2*j+1)*P_j(2*u-1), evaluated by the Legendre recurrence.
    Reject for large values. For fixed k the null limit is chi-square(k)
    and the finite-sample null is shape-free. This is not omnibus for fixed k.
    Fitted shapes require a different calibration and are not supported here.

    References
    ----------
    J. Neyman (1937), Smooth test for goodness of fit, 149-199.
    https://doi.org/10.1080/03461238.1937.10404821
    """

    def __init__(self, parameters: ParameterValues, *, k=4):
        super().__init__(parameters)
        self._uniform = NeymanSmoothTestUniformGofStatistic(
            Uniform.DEFAULT.parse({"a": 0, "b": 1}), k=k
        )
        self.k = self._uniform.k

    @staticmethod
    @override
    def short_code():
        return "NEYMAN"

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @override
    def execute_statistic(self, rvs, **kwargs) -> float:
        """Return the smooth statistic for a nonempty finite sample in [0, 1]."""
        sample = self._validate_rvs(rvs)
        return self._statistic_kernel(scipy_stats.beta.cdf(sample, self.alpha, self.beta), self.k)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _statistic_kernel(cdf, k):
        sums = np.zeros(k)
        for u in cdf:
            x = 2 * u - 1
            previous, current = 1.0, x
            for j in range(1, k + 1):
                sums[j - 1] += np.sqrt(2 * j + 1) * current
                previous, current = current, ((2 * j + 1) * x * current - j * previous) / (j + 1)
        return np.sum(sums**2) / len(cdf)


class RaschkeBetaGofStatistic(AbstractBetaGofStatistic, ADStatistic):
    """Biased-transformation Beta test, Raschke (2009), Sections 4-5.

    Parameters
    ----------
    parameters : ParameterValues
        Beta.DEFAULT.parse({}); both shapes are estimated by MLE.

    Notes
    -----
    Transform y_i=Phi^{-1}(F_{a_hat,b_hat}(x_i)), fit normal mean and
    standard deviation by MLE (ddof=0), then compute fitted-normal AD.
    Return A**2*(1+0.75/n+2.25/n**2), the paper's corrected statistic.
    Uses the shared AD kernel and both Beta tails to avoid 1-CDF cancellation.
    Requires at least three nonconstant observations strictly in (0,1).
    Use parametric_bootstrap_beta in the hypothesis_testing layer to refit
    the whole procedure. The paper's normal critical values are approximate;
    no universal null law or omnibus consistency is asserted here.

    References
    ----------
    M. Raschke (2009), The Biased Transformation and Its Application in
    Goodness-of-Fit Tests for the Beta and Gamma Distribution, 1870-1890.
    https://doi.org/10.1080/03610910903152631
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Beta.DEFAULT, frozenset()),)

    @staticmethod
    @override
    def short_code():
        return "RASCHKE"

    @override
    def execute_statistic(self, rvs, **kwargs) -> float:
        """Return corrected AD after Beta and normal MLE; reject large values.

        Invalid samples or unrepresentable transformations raise ValueError.
        MLE solver errors propagate; observations are never clipped.
        """
        sample = self._validate_rvs(rvs, min_size=3)
        sample, a, b = self._fit(sample)
        x = np.sort(sample)
        lower = scipy_stats.beta.cdf(x, a, b)
        upper = scipy_stats.beta.sf(x, a, b)
        y = np.empty_like(x)
        left = lower <= 0.5
        y[left] = scipy_stats.norm.ppf(lower[left])
        y[~left] = scipy_stats.norm.isf(upper[~left])
        if not np.all(np.isfinite(y)):
            raise ValueError("Beta normal transformation lost numerical precision")
        scale = np.std(y, ddof=0)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Normal maximum likelihood requires positive finite scale")
        z = (y - np.mean(y)) / scale
        value = self._ad_kernel(scipy_stats.norm.logcdf(z), scipy_stats.norm.logsf(z))
        n = len(x)
        return float(value * (1 + 0.75 / n + 2.25 / n**2))
