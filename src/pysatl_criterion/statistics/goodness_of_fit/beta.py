"""Goodness-of-fit statistics on the fixed Beta support [0, 1].

KS, AD, CvM, Watson, Kuiper and Pearson use specified shape parameters.
Lilliefors fits both shapes inside execute_statistic, returning a scalar KS distance.
MB, SK, Ratio, Entropy and Mode are locally defined discrepancies: no scientific
source for their exact formulas was identified. Their docstrings state their
mathematical meaning without attributing them to unrelated published tests.
"""

from abc import ABC

import numpy as np
import scipy.stats as scipy_stats
from scipy.optimize import minimize_scalar
from typing_extensions import override

from pysatl_criterion import DistributionType
from pysatl_criterion.statistics import AbstractGoodnessOfFitStatistic
from pysatl_criterion.statistics.alternative import Alternative, AlternativeType, RightAlternative
from pysatl_criterion.statistics.goodness_of_fit.common import (
    ADStatistic,
    Chi2Statistic,
    CrammerVonMisesStatistic,
    KSStatistic,
    LillieforsTest,
)
from pysatl_criterion.statistics.hypothesis import GoodnessOfFitHypothesis


class AbstractBetaGofStatistic(AbstractGoodnessOfFitStatistic, ABC):
    """
    Abstract base class for Beta distribution goodness-of-fit statistics.

    The Beta distribution is a continuous probability distribution defined on the interval [0, 1]
    parameterized by two positive shape parameters, denoted by α (alpha) and β (beta).
    """

    def __init__(self, alpha=1, beta=1):
        if np.ndim(alpha) != 0 or not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be positive and finite")
        if np.ndim(beta) != 0 or not np.isfinite(beta) or beta <= 0:
            raise ValueError("beta must be positive and finite")
        self.alpha = alpha
        self.beta = beta

    @override
    def hypothesis(self) -> GoodnessOfFitHypothesis:
        return GoodnessOfFitHypothesis({"alpha": self.alpha, "beta": self.beta})

    @staticmethod
    def _validate_rvs(rvs, min_size=1):
        rvs = np.asarray(rvs, dtype=float)
        if rvs.ndim != 1 or rvs.size < min_size:
            raise ValueError(f"Sample must be one-dimensional with at least {min_size} values")
        if not np.all(np.isfinite(rvs)):
            raise ValueError("Sample values must be finite")
        # All values in [0, 1]
        if np.any((rvs < 0) | (rvs > 1)):
            raise ValueError("Beta distribution values must be in the interval [0, 1]")
        return rvs

    def _standardized_moments(self, order):
        """Central moments of Z=(X-E[X])/sd(X), using the Beta Stein identity."""
        total = self.alpha + self.beta
        mean, variance = scipy_stats.beta.stats(self.alpha, self.beta, moments="mv")
        moments = np.zeros(order + 1)
        moments[0] = 1
        for k in range(1, order):
            moments[k + 1] = (
                k
                / (total + k)
                * ((total + 1) * moments[k - 1] + (1 - 2 * mean) / np.sqrt(variance) * moments[k])
            )
        return moments

    @staticmethod
    @override
    def distribution() -> DistributionType:
        """
        Get distribution type.

        :return: DistributionType.
        """
        return DistributionType.BETA

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for Beta distribution statistics.

        :return: string code in format "BETA_{parent_code}".
        """
        return f"BETA_{AbstractGoodnessOfFitStatistic.code()}"


class KolmogorovSmirnovBetaGofStatistic(AbstractBetaGofStatistic, KSStatistic):
    """Kolmogorov-Smirnov distance to a fully specified Beta distribution.

    Parameters
    ----------
    alpha, beta : float, optional
        Finite positive shape parameters fixed by the null hypothesis.
        Both default to 1. No shape parameters are estimated.
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
    use LillieforsTestBetaGofStatistic when both shapes are unknown.

    References
    ----------
    .. [1] N. Smirnov (1948), "Table for Estimating the Goodness of Fit
       of Empirical Distributions", Ann. Math. Statist. 19, 279-281.
       https://doi.org/10.1214/aoms/1177730256

    Examples
    --------
    >>> statistic = KolmogorovSmirnovBetaGofStatistic(alpha=2, beta=5)
    >>> value = statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
    >>> bool(value >= 0)
    True
    """

    def __init__(
        self,
        alpha=1,
        beta=1,
        alternative_type: AlternativeType = AlternativeType.TWO_TAILED,
        mode="auto",
    ):
        AbstractBetaGofStatistic.__init__(self, alpha, beta)
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

    @staticmethod
    @override
    def code():
        """Return the full statistic identifier.

        Returns
        -------
        code : str
            Stable identifier for this statistic.
        """
        short_code = KolmogorovSmirnovBetaGofStatistic.short_code()
        return f"{short_code}_{AbstractBetaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs) -> float | np.float64:
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional finite sample in [0, 1], with at least 1 values.
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

        rvs_sorted = np.sort(rvs)
        cdf_vals = scipy_stats.beta.cdf(rvs_sorted, self.alpha, self.beta)
        return KSStatistic.do_execute_statistic(self, rvs_sorted, cdf_vals)


class AndersonDarlingBetaGofStatistic(AbstractBetaGofStatistic, ADStatistic):
    """Anderson-Darling statistic for a fully specified Beta distribution.

    Parameters
    ----------
    alpha, beta : float, optional
        Finite positive shape parameters fixed by the null hypothesis.
        Both default to 1. No shape parameters are estimated.

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
    >>> statistic = AndersonDarlingBetaGofStatistic(alpha=2, beta=5)
    >>> value = statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
    >>> bool(value >= 0)
    True
    """

    @staticmethod
    @override
    def short_code():
        return "AD"

    @staticmethod
    @override
    def code():
        """Return the full statistic identifier.

        Returns
        -------
        code : str
            Stable identifier for this statistic.
        """
        short_code = AndersonDarlingBetaGofStatistic.short_code()
        return f"{short_code}_{AbstractBetaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs) -> float | np.float64:
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional finite sample in [0, 1], with at least 1 values.
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

        # Compute log CDF and log SF (survival function)
        logcdf = scipy_stats.beta.logcdf(rvs_sorted, self.alpha, self.beta)
        logsf = scipy_stats.beta.logsf(rvs_sorted, self.alpha, self.beta)

        i = np.arange(1, n + 1)
        A2 = -n - np.sum((2 * i - 1.0) / n * (logcdf + logsf[::-1]))

        return A2


class CrammerVonMisesBetaGofStatistic(AbstractBetaGofStatistic, CrammerVonMisesStatistic):
    """Cramer-von Mises statistic for a fully specified Beta distribution.

    Parameters
    ----------
    alpha, beta : float, optional
        Finite positive shape parameters fixed by the null hypothesis.
        Both default to 1. No shape parameters are estimated.

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
    >>> statistic = CrammerVonMisesBetaGofStatistic(alpha=2, beta=5)
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

    @staticmethod
    @override
    def code():
        """Return the full statistic identifier.

        Returns
        -------
        code : str
            Stable identifier for this statistic.
        """
        short_code = CrammerVonMisesBetaGofStatistic.short_code()
        return f"{short_code}_{AbstractBetaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs) -> float | np.float64:
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional finite sample in [0, 1], with at least 1 values.
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

        rvs_sorted = np.sort(rvs)
        cdf_vals = scipy_stats.beta.cdf(rvs_sorted, self.alpha, self.beta)
        return CrammerVonMisesStatistic.do_execute_statistic(self, rvs_sorted, cdf_vals)


class LillieforsTestBetaGofStatistic(AbstractBetaGofStatistic, LillieforsTest):
    """Lilliefors-type KS statistic with both Beta shapes fitted by MLE.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar discrepancy; reject for large values.
    hypothesis()
        Return the null hypothesis parameter specification.

    Notes
    -----
    The null is the entire two-shape Beta family on the fixed support
    [0, 1]. Fit alpha_hat and beta_hat by maximum likelihood with location
    0 and scale 1. Compute the two-sided KS distance to the fitted CDF:
    D = max_i(i/n-u_i, u_i-(i-1)/n), u_i = F_hat(x_(i)).
    The fit solves psi(alpha_hat)-psi(alpha_hat+beta_hat)=mean(log(X))
    and the analogous equation for beta_hat and log(1-X).

    Reference [1]_, Section 4, studies this fitted-Beta KS procedure as a
    competitor to its proposed test. The paper calls it KS, not a separately
    named "Beta Lilliefors" test. The historical class name denotes the
    Lilliefors-type principle of fitting before computing the distance.
    This class does NOT implement the paper's new conditional-moment test.

    ``hypothesis().parameters()`` is empty because both shapes are unknown.
    The null law need not be shape-free. Calibration belongs to the
    hypothesis-testing layer, which must choose simulation shapes and call
    ``execute_statistic`` on every replicate so that both shapes are refitted.
    Ordinary KS critical values and a default Beta(1, 1) simulation are
    invalid here. This class computes only the scalar statistic, not p-values.
    Construction no longer accepts fixed ``alpha`` or ``beta`` arguments.
    Fitted values are local to a call; executing does not mutate the object.

    References
    ----------
    .. [1] B. Ebner and S. C. Liebenberg (2021), "On a new test of fit to
       the beta distribution", Stat 10, e341, Sections 2 and 4.
       https://doi.org/10.1002/sta4.341
       https://arxiv.org/pdf/2009.13995

    Examples
    --------
    >>> statistic = LillieforsTestBetaGofStatistic()
    >>> value = statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
    >>> bool(value >= 0)
    True
    """

    def __init__(self):
        """Create a fitted-Beta KS statistic with no fixed shape parameters."""

    @override
    def hypothesis(self) -> GoodnessOfFitHypothesis:
        """Return the Beta family with both shape parameters unknown."""
        return GoodnessOfFitHypothesis({})

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
    def short_code():
        """Return the short statistic identifier.

        Returns
        -------
        code : str
            Stable identifier for this statistic.
        """
        return "LILLIE"

    @staticmethod
    @override
    def code():
        """Return the full statistic identifier.

        Returns
        -------
        code : str
            Stable identifier for this statistic.
        """
        short_code = LillieforsTestBetaGofStatistic.short_code()
        return f"{short_code}_{AbstractBetaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs) -> float | np.float64:
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional, finite, nonconstant sample strictly in (0, 1), n >= 2.
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
        scipy.stats.FitError
            If the maximum-likelihood solver fails to converge.

        Notes
        -----
        See the class Notes for the formula, null hypothesis and calibration.
        """
        sample, alpha, beta = self._fit(rvs)
        ordered = np.sort(sample)
        cdf_vals = scipy_stats.beta.cdf(ordered, alpha, beta)
        return LillieforsTest.do_execute_statistic(self, ordered, cdf_vals)


class Chi2PearsonBetaGofStatistic(AbstractBetaGofStatistic, Chi2Statistic):
    """Pearson statistic for binned observations under a specified Beta null.

    Parameters
    ----------
    alpha, beta : float, optional
        Finite positive shape parameters fixed by the null hypothesis.
        Both default to 1. No shape parameters are estimated.
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
    >>> statistic = Chi2PearsonBetaGofStatistic(alpha=2, beta=5)
    >>> value = statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
    >>> bool(value >= 0)
    True
    """

    def __init__(self, alpha=1, beta=1, lambda_=1):
        AbstractBetaGofStatistic.__init__(self, alpha, beta)
        Chi2Statistic.__init__(self)
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

    @staticmethod
    @override
    def code():
        """Return the full statistic identifier.

        Returns
        -------
        code : str
            Stable identifier for this statistic.
        """
        short_code = Chi2PearsonBetaGofStatistic.short_code()
        return f"{short_code}_{AbstractBetaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs) -> float | np.float64:
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional finite sample in [0, 1], with at least 1 values.
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

        return Chi2Statistic.do_execute_statistic(self, observed, expected, self.lambda_)


class WatsonBetaGofStatistic(AbstractBetaGofStatistic):
    """Watson U-squared statistic after a specified Beta CDF transform.

    Parameters
    ----------
    alpha, beta : float, optional
        Finite positive shape parameters fixed by the null hypothesis.
        Both default to 1. No shape parameters are estimated.

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
    >>> statistic = WatsonBetaGofStatistic(alpha=2, beta=5)
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

    @staticmethod
    @override
    def code():
        """Return the full statistic identifier.

        Returns
        -------
        code : str
            Stable identifier for this statistic.
        """
        short_code = WatsonBetaGofStatistic.short_code()
        return f"{short_code}_{AbstractBetaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs) -> float | np.float64:
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional finite sample in [0, 1], with at least 1 values.
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

        rvs_sorted = np.sort(rvs)
        n = len(rvs_sorted)
        cdf_vals = scipy_stats.beta.cdf(rvs_sorted, self.alpha, self.beta)

        # Cramér-von Mises statistic
        u = (2 * np.arange(1, n + 1) - 1) / (2 * n)
        cvm = 1 / (12 * n) + np.sum((u - cdf_vals) ** 2)

        # Watson correction
        correction_term = n * (np.mean(cdf_vals) - 0.5) ** 2
        watson_statistic = cvm - correction_term

        return watson_statistic


class KuiperBetaGofStatistic(AbstractBetaGofStatistic):
    """Kuiper statistic after a specified Beta CDF transform.

    Parameters
    ----------
    alpha, beta : float, optional
        Finite positive shape parameters fixed by the null hypothesis.
        Both default to 1. No shape parameters are estimated.

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
    >>> statistic = KuiperBetaGofStatistic(alpha=2, beta=5)
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

    @staticmethod
    @override
    def code():
        """Return the full statistic identifier.

        Returns
        -------
        code : str
            Stable identifier for this statistic.
        """
        short_code = KuiperBetaGofStatistic.short_code()
        return f"{short_code}_{AbstractBetaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs) -> float | np.float64:
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional finite sample in [0, 1], with at least 1 values.
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

        rvs_sorted = np.sort(rvs)
        n = len(rvs_sorted)
        cdf_vals = scipy_stats.beta.cdf(rvs_sorted, self.alpha, self.beta)

        # D+ = max(i/n - F(x_i))
        d_plus = np.max(np.arange(1, n + 1) / n - cdf_vals)

        # D- = max(F(x_i) - (i-1)/n)
        d_minus = np.max(cdf_vals - np.arange(0, n) / n)

        # Kuiper statistic is D+ + D-
        return d_plus + d_minus


class MomentBasedBetaGofStatistic(AbstractBetaGofStatistic):
    """Covariance-standardized mean/variance discrepancy for a specified Beta null.

    Parameters
    ----------
    alpha, beta : float, optional
        Finite positive shape parameters fixed by the null hypothesis.
        Both default to 1. No shape parameters are estimated.

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
    >>> statistic = MomentBasedBetaGofStatistic(alpha=2, beta=5)
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

    @staticmethod
    @override
    def code():
        """Return the full statistic identifier.

        Returns
        -------
        code : str
            Stable identifier for this statistic.
        """
        short_code = MomentBasedBetaGofStatistic.short_code()
        return f"{short_code}_{AbstractBetaGofStatistic.code()}"

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
        moments = self._standardized_moments(4)
        covariance = np.array([[1, moments[3]], [moments[3], moments[4] - 1]])
        difference = np.array(
            [
                (np.mean(rvs) - mean) / np.sqrt(variance),
                np.var(rvs, ddof=1) / variance - 1,
            ]
        )
        return len(rvs) * difference @ np.linalg.solve(covariance, difference)


class SkewnessKurtosisBetaGofStatistic(AbstractBetaGofStatistic):
    """Covariance-standardized skewness/excess discrepancy for a specified Beta null.

    Parameters
    ----------
    alpha, beta : float, optional
        Finite positive shape parameters fixed by the null hypothesis.
        Both default to 1. No shape parameters are estimated.

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
    Sample estimates use bias=False. This is not the normal-model
    Jarque-Bera statistic. Both shapes are fixed; the asymptotic null law
    is chi-square(2), not an exact finite-sample law.

    No publication describing this exact Beta statistic was identified.
    It is a locally defined discrepancy and need not detect distributions
    with matching skewness and excess. At least four observations and a
    nonconstant sample are required.

    Examples
    --------
    >>> statistic = SkewnessKurtosisBetaGofStatistic(alpha=2, beta=5)
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

    @staticmethod
    @override
    def code():
        """Return the full statistic identifier.

        Returns
        -------
        code : str
            Stable identifier for this statistic.
        """
        short_code = SkewnessKurtosisBetaGofStatistic.short_code()
        return f"{short_code}_{AbstractBetaGofStatistic.code()}"

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
        moments = self._standardized_moments(8)
        skewness, pearson_kurtosis = moments[3:5]
        # Influence polynomials in Z=(X-mu)/sigma, coefficients in ascending order.
        polynomials = [
            [skewness / 2, -3, -1.5 * skewness, 1],
            [pearson_kurtosis, -4 * skewness, -2 * pearson_kurtosis, 0, 1],
        ]
        covariance = np.empty((2, 2))
        for i in range(2):
            for j in range(2):
                product = np.polynomial.polynomial.polymul(polynomials[i], polynomials[j])
                covariance[i, j] = product @ moments[: len(product)]
        difference = np.array(
            [
                scipy_stats.skew(rvs, bias=False) - skewness,
                scipy_stats.kurtosis(rvs, bias=False) - (pearson_kurtosis - 3),
            ]
        )
        return len(rvs) * difference @ np.linalg.solve(covariance, difference)


class RatioBetaGofStatistic(AbstractBetaGofStatistic):
    """Geometric/arithmetic mean discrepancy for a specified Beta null.

    Parameters
    ----------
    alpha, beta : float, optional
        Finite positive shape parameters fixed by the null hypothesis.
        Both default to 1. No shape parameters are estimated.

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
    an all-zero sample is rejected because the ratio is undefined.

    No publication describing this exact Beta statistic was identified.
    It is a locally defined discrepancy, not an established named Beta
    criterion, and need not detect alternatives with the same ratio.

    Examples
    --------
    >>> statistic = RatioBetaGofStatistic(alpha=2, beta=5)
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

    @staticmethod
    @override
    def code():
        """Return the full statistic identifier.

        Returns
        -------
        code : str
            Stable identifier for this statistic.
        """
        short_code = RatioBetaGofStatistic.short_code()
        return f"{short_code}_{AbstractBetaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs) -> float | np.float64:
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            One-dimensional finite sample in [0, 1], with at least 1 values.
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

        n = len(rvs)

        # Sample statistics
        arithmetic_mean = np.mean(rvs)
        with np.errstate(divide="ignore"):
            geometric_mean = np.exp(np.mean(np.log(rvs)))

        # Avoid division by zero
        if arithmetic_mean == 0:
            raise ValueError("Arithmetic mean is zero, cannot compute ratio")

        sample_ratio = geometric_mean / arithmetic_mean

        # Theoretical ratio for Beta distribution
        # E[X] = α/(α+β)
        # E[log X] = ψ(α) - ψ(α+β) where ψ is digamma function
        from scipy.special import psi

        theoretical_mean = self.alpha / (self.alpha + self.beta)
        theoretical_log_mean = psi(self.alpha) - psi(self.alpha + self.beta)
        theoretical_ratio = np.exp(theoretical_log_mean) / theoretical_mean

        # Test statistic
        statistic = np.sqrt(n) * np.abs(sample_ratio - theoretical_ratio)

        return statistic


class EntropyBetaGofStatistic(AbstractBetaGofStatistic):
    """Vasicek-entropy discrepancy for a specified Beta null.

    Parameters
    ----------
    alpha, beta : float, optional
        Finite positive shape parameters fixed by the null hypothesis.
        Both default to 1. No shape parameters are estimated.

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
    >>> statistic = EntropyBetaGofStatistic(alpha=2, beta=5)
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

    @staticmethod
    @override
    def code():
        """Return the full statistic identifier.

        Returns
        -------
        code : str
            Stable identifier for this statistic.
        """
        short_code = EntropyBetaGofStatistic.short_code()
        return f"{short_code}_{AbstractBetaGofStatistic.code()}"

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
        # Zero spacings imply entropy -infinity, hence discrepancy +infinity.
        with np.errstate(divide="ignore"):
            sample_entropy = scipy_stats.differential_entropy(
                rvs, window_length=m, method="vasicek"
            )

        # Theoretical entropy
        from scipy.special import betaln, psi

        theoretical_entropy = (
            betaln(self.alpha, self.beta)
            - (self.alpha - 1) * psi(self.alpha)
            - (self.beta - 1) * psi(self.beta)
            + (self.alpha + self.beta - 2) * psi(self.alpha + self.beta)
        )

        # Test statistic
        statistic = np.sqrt(n) * np.abs(sample_entropy - theoretical_entropy)

        return statistic


class ModeBetaGofStatistic(AbstractBetaGofStatistic):
    """KDE-mode discrepancy for a specified unimodal Beta null.

    Parameters
    ----------
    alpha, beta : float, optional
        Finite greater than 1 shape parameters fixed by the null hypothesis.
        Both default to 2. No shape parameters are estimated.

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
    bounded optimization of every detected local peak. Include endpoints
    of [0,1] among candidates. Return sqrt(n)*abs(mode_hat-mode_Beta).
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
    >>> statistic = ModeBetaGofStatistic(alpha=2, beta=5)
    >>> value = statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
    >>> bool(value >= 0)
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    def __init__(self, alpha=2, beta=2):
        super().__init__(alpha, beta)
        if alpha <= 1:
            raise ValueError("alpha must be greater than 1 for mode to be well-defined")
        if beta <= 1:
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

    @staticmethod
    @override
    def code():
        """Return the full statistic identifier.

        Returns
        -------
        code : str
            Stable identifier for this statistic.
        """
        short_code = ModeBetaGofStatistic.short_code()
        return f"{short_code}_{AbstractBetaGofStatistic.code()}"

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
        kde = scipy_stats.gaussian_kde(rvs)
        # Gaussian mixture modes lie inside the sample range. Refine every
        # grid-local maximum, including candidates near either support boundary.
        x_grid = np.linspace(rvs.min(), rvs.max(), max(257, int(np.sqrt(n)) + 1))
        density = kde(x_grid)
        peaks = np.flatnonzero((density[1:-1] >= density[:-2]) & (density[1:-1] >= density[2:])) + 1
        candidates = [0.0, 1.0, x_grid[0], x_grid[-1]]
        for i in peaks:
            result = minimize_scalar(
                lambda x: -kde([x])[0],
                bounds=(x_grid[i - 1], x_grid[i + 1]),
                method="bounded",
                options={"xatol": 1e-12},
            )
            candidates.append(result.x)
        sample_mode = candidates[np.argmax(kde(candidates))]

        # Theoretical mode
        theoretical_mode = (self.alpha - 1) / (self.alpha + self.beta - 2)

        # Test statistic
        statistic = np.sqrt(n) * np.abs(sample_mode - theoretical_mode)

        return statistic
