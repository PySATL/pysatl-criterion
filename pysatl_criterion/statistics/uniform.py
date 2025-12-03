from abc import ABC

import numpy as np
import scipy.stats as scipy_stats
from typing_extensions import override

from pysatl_criterion.statistics.common import (
    ADStatistic,
    Chi2Statistic,
    CrammerVonMisesStatistic,
    KSStatistic,
    LillieforsTest,
)
from pysatl_criterion.statistics.goodness_of_fit import AbstractGoodnessOfFitStatistic


class AbstractUniformGofStatistic(AbstractGoodnessOfFitStatistic, ABC):
    """
    Abstract base class for Uniform distribution goodness-of-fit statistics.

    The Uniform distribution is a continuous probability distribution where
    all values in the interval [a, b] are equally likely.

    Parameters
    ----------
    a : float, default=0
        Lower bound of the distribution
    b : float, default=1
        Upper bound of the distribution
    """

    def __init__(self, a=0, b=1):
        if b <= a:
            raise ValueError("b must be greater than a")
        self.a = a
        self.b = b

    @staticmethod
    @override
    def code():
        return f"UNIFORM_{AbstractGoodnessOfFitStatistic.code()}"


class KolmogorovSmirnovUniformGofStatistic(AbstractUniformGofStatistic, KSStatistic):
    """
    Kolmogorov-Smirnov test statistic for Uniform distribution.

    The Kolmogorov-Smirnov test compares the empirical distribution function
    with the theoretical Uniform distribution function.

    Parameters
    ----------
    a : float, default=0
        Lower bound of the distribution
    b : float, default=1
        Upper bound of the distribution
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis. Default is 'two-sided'.
    mode : {'auto', 'exact', 'approx', 'asymp'}, optional
        Defines the distribution used for calculating the p-value. Default is 'auto'.

    Returns
    -------
    statistic : float
        The Kolmogorov-Smirnov test statistic

    References
    ----------
    .. [1] Massey, F. J. (1951). The Kolmogorov-Smirnov test for goodness of fit.
           Journal of the American Statistical Association, 46(253), 68-78.
    """

    def __init__(self, a=0, b=1, alternative="two-sided", mode="auto"):
        AbstractUniformGofStatistic.__init__(self, a, b)
        KSStatistic.__init__(self, alternative, mode)

    @staticmethod
    @override
    def code():
        return f"KS_{AbstractUniformGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Kolmogorov-Smirnov test statistic.

        Parameters
        ----------
        rvs : array_like
            Array of sample data from Uniform distribution. Values should be in [a, b].

        Returns
        -------
        statistic : float
            The test statistic value
        """
        rvs = np.asarray(rvs)
        if np.any((rvs < self.a) | (rvs > self.b)):
            raise ValueError(f"Uniform distribution values must be in the interval"
                             f"[{self.a}, {self.b}]")

        rvs_sorted = np.sort(rvs)
        cdf_vals = scipy_stats.uniform.cdf(rvs_sorted, loc=self.a, scale=self.b - self.a)
        return KSStatistic.execute_statistic(self, rvs_sorted, cdf_vals)


class AndersonDarlingUniformGofStatistic(AbstractUniformGofStatistic, ADStatistic):
    """
    Anderson-Darling test statistic for Uniform distribution.

    The Anderson-Darling test is a modification of the Kolmogorov-Smirnov test
    that gives more weight to the tails of the distribution.

    Parameters
    ----------
    a : float, default=0
        Lower bound of the distribution
    b : float, default=1
        Upper bound of the distribution

    Returns
    -------
    statistic : float
        The Anderson-Darling test statistic

    References
    ----------
    .. [1] Anderson, T. W., & Darling, D. A. (1952). Asymptotic theory of certain
           "goodness of fit" criteria based on stochastic processes.
           The Annals of Mathematical Statistics, 23(2), 193-212.
    """

    @staticmethod
    @override
    def code():
        return f"AD_{AbstractUniformGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Anderson-Darling test statistic.

        Parameters
        ----------
        rvs : array_like
            Array of sample data from Uniform distribution. Values should be in [a, b].

        Returns
        -------
        statistic : float
            The test statistic value
        """
        rvs = np.asarray(rvs)
        if np.any((rvs < self.a) | (rvs > self.b)):
            raise ValueError(f"Uniform distribution values must be in the interval"
                             f"[{self.a}, {self.b}]")

        rvs_sorted = np.sort(rvs)
        n = len(rvs_sorted)

        logcdf = scipy_stats.uniform.logcdf(rvs_sorted, loc=self.a, scale=self.b - self.a)
        logsf = scipy_stats.uniform.logsf(rvs_sorted, loc=self.a, scale=self.b - self.a)

        i = np.arange(1, n + 1)
        a2 = -n - np.sum((2 * i - 1.0) / n * (logcdf + logsf[::-1]))

        return a2


class CrammerVonMisesUniformGofStatistic(AbstractUniformGofStatistic, CrammerVonMisesStatistic):
    """
    Cramér-von Mises test statistic for Uniform distribution.

    The Cramér-von Mises test is a goodness-of-fit test that measures the
    discrepancy between the empirical distribution function and the theoretical
    cumulative distribution function.

    Parameters
    ----------
    a : float, default=0
        Lower bound of the distribution
    b : float, default=1
        Upper bound of the distribution

    Returns
    -------
    statistic : float
        The Cramér-von Mises test statistic

    References
    ----------
    .. [1] Cramér, H. (1928). On the composition of elementary errors.
           Scandinavian Actuarial Journal, 1928(1), 13-74.
    .. [2] von Mises, R. (1928). Wahrscheinlichkeit, Statistik und Wahrheit.
           Julius Springer.
    """

    @staticmethod
    @override
    def code():
        return f"CVM_{AbstractUniformGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Cramér-von Mises test statistic.

        Parameters
        ----------
        rvs : array_like
            Array of sample data from Uniform distribution. Values should be in [a, b].

        Returns
        -------
        statistic : float
            The test statistic value
        """
        rvs = np.asarray(rvs)
        if np.any((rvs < self.a) | (rvs > self.b)):
            raise ValueError(f"Uniform distribution values must be in the interval"
                             f"[{self.a}, {self.b}]")

        rvs_sorted = np.sort(rvs)
        cdf_vals = scipy_stats.uniform.cdf(rvs_sorted, loc=self.a, scale=self.b - self.a)
        return CrammerVonMisesStatistic.execute_statistic(self, rvs_sorted, cdf_vals)


class LillieforsTestUniformGofStatistic(AbstractUniformGofStatistic, LillieforsTest):
    """
    Lilliefors test statistic for Uniform distribution.

    The Lilliefors test is a modification of the Kolmogorov-Smirnov test for
    the case when parameters are estimated from the data.

    Parameters
    ----------
    a : float, default=0
        Lower bound of the distribution
    b : float, default=1
        Upper bound of the distribution

    Returns
    -------
    statistic : float
        The Lilliefors test statistic

    References
    ----------
    .. [1] Lilliefors, H. W. (1967). On the Kolmogorov-Smirnov test for normality
           with mean and variance unknown. Journal of the American Statistical
           Association, 62(318), 399-402.
    """

    @staticmethod
    @override
    def code():
        return f"LILLIE_{AbstractUniformGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Lilliefors test statistic.

        Parameters
        ----------
        rvs : array_like
            Array of sample data from Uniform distribution. Values should be in [a, b].

        Returns
        -------
        statistic : float
            The test statistic value
        """
        rvs = np.asarray(rvs)
        if np.any((rvs < self.a) | (rvs > self.b)):
            raise ValueError(f"Uniform distribution values must be in the interval"
                             f"[{self.a}, {self.b}]")

        rvs_sorted = np.sort(rvs)
        cdf_vals = scipy_stats.uniform.cdf(rvs_sorted, loc=self.a, scale=self.b - self.a)
        return LillieforsTest.execute_statistic(self, rvs_sorted, cdf_vals)


class Chi2PearsonUniformGofStatistic(AbstractUniformGofStatistic, Chi2Statistic):
    """
    Pearson's Chi-squared test statistic for Uniform distribution.

    The chi-squared test compares observed frequencies with expected frequencies
    based on the Uniform distribution.

    Parameters
    ----------
    a : float, default=0
        Lower bound of the distribution
    b : float, default=1
        Upper bound of the distribution
    lambda_ : float, default=1
        Power divergence parameter. Lambda=1 gives Pearson's chi-squared statistic.
    bins : int or str, optional
        Number of bins for the histogram. Can be an integer or string like 'auto',
        'sqrt', 'sturges', etc. Default is 'sturges'.

    Returns
    -------
    statistic : float
        The chi-squared test statistic

    References
    ----------
    .. [1] Pearson, K. (1900). On the criterion that a given system of deviations from
           the probable in the case of a correlated system of variables is such that
           it can be reasonably supposed to have arisen from random sampling.
           The London, Edinburgh, and Dublin Philosophical Magazine and Journal of
           Science, 50(302), 157-175.
    """

    def __init__(self, a=0, b=1, lambda_=1, bins='sturges'):
        AbstractUniformGofStatistic.__init__(self, a, b)
        Chi2Statistic.__init__(self)
        self.lambda_ = lambda_
        self.bins = bins

    @staticmethod
    @override
    def code():
        return f"CHI2_PEARSON_{AbstractUniformGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Pearson's chi-squared test statistic.

        Parameters
        ----------
        rvs : array_like
            Array of sample data from Uniform distribution. Values should be in [a, b].

        Returns
        -------
        statistic : float
            The test statistic value
        """
        rvs = np.asarray(rvs)
        if np.any((rvs < self.a) | (rvs > self.b)):
            raise ValueError(f"Uniform distribution values must be in the interval"
                             f"[{self.a}, {self.b}]")

        n = len(rvs)
        if isinstance(self.bins, str):
            if self.bins == 'sturges':
                num_bins = int(np.ceil(np.log2(n) + 1))
            elif self.bins == 'sqrt':
                num_bins = int(np.ceil(np.sqrt(n)))
            elif self.bins == 'auto':
                # Scott's rule for bin width
                h = 3.5 * np.std(rvs) / (n ** (1 / 3))
                num_bins = int(np.ceil((self.b - self.a) / h))
            else:
                num_bins = 10  # default
        else:
            num_bins = int(self.bins)
        num_bins = max(2, num_bins)

        observed, bin_edges = np.histogram(rvs, bins=num_bins, range=(self.a, self.b))
        expected = np.full(num_bins, n / num_bins)

        return Chi2Statistic.execute_statistic(self, observed, expected, self.lambda_)


class WatsonUniformGofStatistic(AbstractUniformGofStatistic):
    """
    Watson's U² test statistic for Uniform distribution.

    Watson's test is a modification of the Cramér-von Mises test that is invariant
    to cyclic transformations, making it particularly suitable for circular data.

    Parameters
    ----------
    a : float, default=0
        Lower bound of the distribution
    b : float, default=1
        Upper bound of the distribution

    Returns
    -------
    statistic : float
        The Watson's U² test statistic

    References
    ----------
    .. [1] Watson, G. S. (1961). Goodness-of-fit tests on a circle.
           Biometrika, 48(1/2), 109-114.
    """

    def __init__(self, a=0, b=1):
        AbstractUniformGofStatistic.__init__(self, a, b)

    @staticmethod
    @override
    def code():
        return f"WATSON_{AbstractUniformGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Watson's U² test statistic.

        Parameters
        ----------
        rvs : array_like
            Array of sample data from Uniform distribution. Values should be in [a, b].

        Returns
        -------
        statistic : float
            The test statistic value
        """
        rvs = np.asarray(rvs)
        if np.any((rvs < self.a) | (rvs > self.b)):
            raise ValueError(f"Uniform distribution values must be in the interval"
                             f"[{self.a}, {self.b}]")

        n = len(rvs)
        rvs_sorted = np.sort(rvs)
        rvs_standardized = (rvs_sorted - self.a) / (self.b - self.a)

        i = np.arange(1, n + 1)
        fn = i / n
        mean_f = np.mean(rvs_standardized)
        u2 = np.sum((rvs_standardized - fn + 0.5 / n - mean_f) ** 2) / n + 1 / (12 * n ** 2)

        return u2


class KuiperUniformGofStatistic(AbstractUniformGofStatistic):
    """
    Kuiper test statistic for Uniform distribution.

    The Kuiper test is similar to the Kolmogorov-Smirnov test but is particularly
    sensitive to deviations at the ends of the distribution.

    Parameters
    ----------
    a : float, default=0
        Lower bound of the distribution
    b : float, default=1
        Upper bound of the distribution

    Returns
    -------
    statistic : float
        The Kuiper test statistic

    References
    ----------
    .. [1] Kuiper, N. H. (1960). Tests concerning random points on a circle.
           Indagationes Mathematicae (Proceedings), 63, 38-47.
    """

    def __init__(self, a=0, b=1):
        AbstractUniformGofStatistic.__init__(self, a, b)

    @staticmethod
    @override
    def code():
        return f"KUIPER_{AbstractUniformGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Kuiper test statistic.

        Parameters
        ----------
        rvs : array_like
            Array of sample data from Uniform distribution. Values should be in [a, b].

        Returns
        -------
        statistic : float
            The test statistic value
        """
        rvs = np.asarray(rvs, dtype=np.float64)
        if np.any((rvs < self.a) | (rvs > self.b)):
            raise ValueError(f"Uniform distribution values must be in the interval"
                             f"[{self.a}, {self.b}]")

        n = len(rvs)
        rvs_sorted = np.sort(rvs)
        if self.a != 0 or self.b != 1:
            rvs_standardized = (rvs_sorted - self.a) / (self.b - self.a)
        else:
            rvs_standardized = rvs_sorted.copy()

        i = np.arange(1, n + 1)
        fn = i / n

        d_plus = np.max(fn - rvs_standardized)
        d_minus = np.max(rvs_standardized - (i - 1) / n)
        v = float(d_plus) + float(d_minus)

        return v


class GreenwoodTestUniformGofStatistic(AbstractUniformGofStatistic):
    """
    Greenwood's test for Uniform distribution.

    Based on the sum of squares of spacings between order statistics.

    Parameters
    ----------
    a : float, default=0
        Lower bound of the distribution
    b : float, default=1
        Upper bound of the distribution

    Returns
    -------
    statistic : float
        The Greenwood statistic

    References
    ----------
    .. [1] Greenwood, M. (1946). The statistical study of infectious diseases.
           Journal of the Royal Statistical Society, 109(2), 85-110.
    """

    @staticmethod
    @override
    def code():
        return f"GREENWOOD_{AbstractUniformGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        rvs = np.asarray(rvs)
        if np.any((rvs < self.a) | (rvs > self.b)):
            raise ValueError(f"Values must be in [{self.a}, {self.b}]")

        rvs_sorted = np.sort(rvs)
        rvs_std = (rvs_sorted - self.a) / (self.b - self.a)
        rvs_with_boundaries = np.concatenate([[0], rvs_std, [1]])
        spacings = np.diff(rvs_with_boundaries)

        g = np.sum(spacings ** 2)

        return g


class BickelRosenblattUniformGofStatistic(AbstractUniformGofStatistic):
    """
    Bickel-Rosenblatt test for Uniform distribution.

    Kernel-based goodness-of-fit test.

    Parameters
    ----------
    a : float, default=0
        Lower bound of the distribution
    b : float, default=1
        Upper bound of the distribution
    bandwidth : float or str, default='auto'
        Bandwidth for kernel density estimation

    References
    ----------
    .. [1] Bickel, P. J., & Rosenblatt, M. (1973). On some global measures of
           the deviations of density function estimates.
           The Annals of Statistics, 1(6), 1071-1095.
    """

    def __init__(self, a=0, b=1, bandwidth='auto'):
        AbstractUniformGofStatistic.__init__(self, a, b)
        self.bandwidth = bandwidth

    @staticmethod
    @override
    def code():
        return f"BICKEL_ROSENBLATT_{AbstractUniformGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        rvs = np.asarray(rvs)
        if np.any((rvs < self.a) | (rvs > self.b)):
            raise ValueError(f"Values must be in [{self.a}, {self.b}]")

        n = len(rvs)
        rvs_std = (rvs - self.a) / (self.b - self.a)
        if self.bandwidth == 'auto':
            h = 1.06 * np.std(rvs_std) * (n ** (-1 / 5))
        else:
            h = self.bandwidth

        x_grid = np.linspace(0, 1, 1000)
        kde_vals = np.zeros_like(x_grid)

        for i, x in enumerate(x_grid):
            kde_vals[i] = np.mean(scipy_stats.norm.pdf((x - rvs_std) / h)) / h

        uniform_density = np.ones_like(x_grid)

        dx = x_grid[1] - x_grid[0]
        statistic = float(np.sum((kde_vals - uniform_density) ** 2)) * dx

        return statistic


class ZhangTestsUniformGofStatistic(AbstractUniformGofStatistic):
    """
    Zhang's tests (Z_A, Z_C, Z_K) for Uniform distribution.

    Powerful class of tests based on likelihood ratios.

    Parameters
    ----------
    a : float, default=0
        Lower bound of the distribution
    b : float, default=1
        Upper bound of the distribution
    test_type : {'A', 'C', 'K'}, default='A'
        Type of Zhang test

    References
    ----------
    .. [1] Zhang, J. (2002). Powerful goodness-of-fit tests based on the
           likelihood ratio. Journal of the Royal Statistical Society:
           Series B (Statistical Methodology), 64(2), 281-294.
    """

    def __init__(self, a=0, b=1, test_type='A'):
        AbstractUniformGofStatistic.__init__(self, a, b)
        self.test_type = test_type.upper()
        if self.test_type not in ['A', 'C', 'K']:
            raise ValueError("test_type must be 'A', 'C', or 'K'")

    @staticmethod
    @override
    def code():
        return f"ZHANG_{AbstractUniformGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        rvs = np.asarray(rvs)
        if np.any((rvs < self.a) | (rvs > self.b)):
            raise ValueError(f"Values must be in [{self.a}, {self.b}]")

        n = len(rvs)
        rvs_sorted = np.sort(rvs)

        rvs_std = (rvs_sorted - self.a) / (self.b - self.a)

        i = np.arange(1, n + 1)

        if self.test_type == 'A':
            term1 = np.sum(np.log(rvs_std) / (n - i + 0.5))
            term2 = np.sum(np.log(1 - rvs_std) / (i - 0.5))
            statistic = -term1 - term2

        elif self.test_type == 'C':
            term1 = np.sum((np.log(rvs_std) / (n - i + 0.5)) ** 2)
            term2 = np.sum((np.log(1 - rvs_std) / (i - 0.5)) ** 2)
            statistic = term1 + term2

        else:
            term1 = np.sum(np.log(rvs_std / (1 - rvs_std)) / (n - i + 0.5))
            term2 = np.sum(np.log((1 - rvs_std) / rvs_std) / (i - 0.5))
            statistic = max(np.abs(term1), np.abs(term2))

        return statistic


class SteinUniformGofStatistic(AbstractUniformGofStatistic):
    """
    Stein-type test statistic for Uniform distribution based on U-statistics.

    Uses Stein's fixed point characterization:
    2E(XI(X>t)) = E(I(X>t)) + t(1-t) for all t ∈ [0,1]

    Parameters
    ----------
    a : float, default=0
        Lower bound of the distribution
    b : float, default=1
        Upper bound of the distribution

    Returns
    -------
    statistic : float
        The Stein-type test statistic

    References
    ----------
    .. [1] Kattumannil, S. K., & Sreedevi, E. P. (2021).
           A new goodness of fit test for uniform distribution with censored observations.
           arXiv preprint arXiv:2106.06368.
    .. [2] Ebner, B., & Liebenberg, S. C. (2020). On a new test of fit to the beta distribution.
           Stat, e341.
    """

    def __init__(self, a=0, b=1):
        AbstractUniformGofStatistic.__init__(self, a, b)

    @staticmethod
    @override
    def code():
        return f"STEIN_U_{AbstractUniformGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Stein-type test statistic.

        Parameters
        ----------
        rvs : array_like
            Array of sample data from Uniform distribution. Values should be in [a, b].

        Returns
        -------
        statistic : float
            The test statistic value
        """
        rvs = np.asarray(rvs)
        if np.any((rvs < self.a) | (rvs > self.b)):
            raise ValueError(f"Uniform distribution values must be in the interval"
                             f"[{self.a}, {self.b}]")
        if self.a != 0 or self.b != 1:
            rvs_std = (rvs - self.a) / (self.b - self.a)
        else:
            rvs_std = rvs.copy()

        statistic = self._compute_u_statistic(rvs_std)

        return statistic

    @staticmethod
    def _compute_u_statistic(rvs_std):
        """Compute U-statistic directly using double sum."""
        n = len(rvs_std)

        def h1(x, y):
            return 0.5 * (2 * max(x, y) - 2 * x - 2 * y + x ** 2 + y ** 2)

        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += h1(rvs_std[i], rvs_std[j])

        statistic = 2 * total / (n * (n - 1)) if n > 1 else 0

        return statistic


class CensoredSteinUniformGofStatistic(AbstractUniformGofStatistic):
    """
    Stein-type test statistic for Uniform distribution with right censoring.

    Extension of the Stein test to handle right-censored data using
    inverse probability of censoring weighting (IPCW).

    Parameters
    ----------
    a : float, default=0
        Lower bound of the distribution
    b : float, default=1
        Upper bound of the distribution

    Returns
    -------
    statistic : float
        The censored Stein-type test statistic

    References
    ----------
    .. [1] Kattumannil, S. K., & Sreedevi, E. P. (2021).
           A new goodness of fit test for uniform distribution with censored observations.
           arXiv preprint arXiv:2106.06368.
    .. [2] Datta, S., Bandyopadhyay, D., & Satten, G. A. (2010).
           Inverse probability of censoring weighted U-statistics for right-censored data.
           Scandinavian Journal of Statistics, 37, 680-700.
    """

    def __init__(self, a=0, b=1):
        AbstractUniformGofStatistic.__init__(self, a, b)

    @staticmethod
    @override
    def code():
        return f"CENSORED_STEIN_U_{AbstractUniformGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, censoring_indices=None, **kwargs):
        """
        Execute the censored Stein-type test statistic.

        Parameters
        ----------
        rvs : array_like
            Array of sample data (observed times)
        censoring_indices : array_like, optional
            Binary array where 1 indicates censored observation, 0 indicates uncensored.
            If None, assumes all observations are uncensored.

        Returns
        -------
        statistic : float
            The test statistic value
        """
        rvs = np.asarray(rvs)
        if np.any((rvs < self.a) | (rvs > self.b)):
            raise ValueError(f"Values must be in [{self.a}, {self.b}]")

        if self.a != 0 or self.b != 1:
            rvs_std = (rvs - self.a) / (self.b - self.a)
        else:
            rvs_std = rvs.copy()

        if censoring_indices is None or np.all(censoring_indices == 0):
            return SteinUniformGofStatistic(self.a, self.b).execute_statistic(rvs)

        censoring_indices = np.asarray(censoring_indices)

        km_estimator = self._kaplan_meier(rvs_std, censoring_indices)

        statistic = self._compute_weighted_u_statistic(rvs_std, censoring_indices, km_estimator)

        return statistic

    @staticmethod
    def _kaplan_meier(times, delta):
        """
        Compute Kaplan-Meier estimator for censoring distribution.

        Parameters
        ----------
        times : array_like
            Observed times
        delta : array_like
            Censoring indicators (1 = censored, 0 = uncensored)

        Returns
        -------
        km_survival : function
            Function that returns survival probability at given time
        """
        sort_idx = np.argsort(times)
        times_sorted = times[sort_idx]
        delta_sorted = delta[sort_idx]

        n = len(times)
        at_risk = np.arange(n, 0, -1)
        km_survival = np.ones(n + 1)

        for i in range(n):
            if delta_sorted[i] == 1:
                km_survival[i + 1] = km_survival[i] * (1 - 1 / at_risk[i])
            else:
                km_survival[i + 1] = km_survival[i]

        def survival_func(t):
            idx = np.searchsorted(times_sorted, t, side='right')
            return km_survival[idx]

        return survival_func

    @staticmethod
    def _compute_weighted_u_statistic(rvs, delta, km_func):
        """
        Compute IPCW-weighted U-statistic.

        Parameters
        ----------
        rvs : array_like
            Standardized observations
        delta : array_like
            Censoring indicators
        km_func : function
            Kaplan-Meier survival function for censoring

        Returns
        -------
        statistic : float
            Weighted test statistic
        """
        n = len(rvs)

        def h1(x, y):
            return 0.5 * (2 * max(x, y) - 2 * x - 2 * y + x ** 2 + y ** 2)

        weights = np.zeros(n)
        for i in range(n):
            if delta[i] == 0:
                weights[i] = 1.0 / max(km_func(rvs[i]), 1e-10)
            else:
                weights[i] = 0

        total = 0
        count = 0

        for i in range(n):
            if delta[i] == 0:
                for j in range(i + 1, n):
                    if delta[j] == 0:
                        weight_ij = weights[i] * weights[j]
                        total += weight_ij * h1(rvs[i], rvs[j])
                        count += 1

        if count > 0:
            statistic = 2 * total / count
        else:
            statistic = 0

        return statistic


class NeymanSmoothTestUniformGofStatistic(AbstractUniformGofStatistic):
    """
    Neyman's smooth test for Uniform distribution.

    Powerful against smooth alternatives.

    Parameters
    ----------
    a : float, default=0
        Lower bound of the distribution
    b : float, default=1
        Upper bound of the distribution
    k : int, default=4
        Number of components to use

    References
    ----------
    .. [1] Neyman, J. (1937). Smooth test for goodness of fit.
           Scandinavian Actuarial Journal, 1937(1), 149-199.
    """

    def __init__(self, a=0, b=1, k=4):
        AbstractUniformGofStatistic.__init__(self, a, b)
        self.k = k

    @staticmethod
    @override
    def code():
        return f"NEYMAN_{AbstractUniformGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        rvs = np.asarray(rvs)
        if np.any((rvs < self.a) | (rvs > self.b)):
            raise ValueError(f"Values must be in [{self.a}, {self.b}]")

        n = len(rvs)

        rvs_std = (rvs - self.a) / (self.b - self.a)

        from scipy.special import legendre

        statistic = 0

        def phi(j, x):
            if j == 1:
                return np.sqrt(12) * (x - 0.5)
            elif j == 2:
                return np.sqrt(5) * (6 * (x - 0.5) ** 2 - 0.5)
            elif j == 3:
                return np.sqrt(7) * (20 * (x - 0.5) ** 3 - 3 * (x - 0.5))
            else:
                pj = legendre(j)
                return pj(2 * x - 1) * np.sqrt(2 * j + 1)

        for j in range(1, self.k + 1):
            vj = float(np.sum(phi(j, rvs_std))) / np.sqrt(n)
            statistic += vj ** 2

        return statistic


class ShermanUniformGofStatistic(AbstractUniformGofStatistic):
    """
    Sherman's test for Uniform distribution.

    Based on spacing between order statistics.

    References
    ----------
    .. [1] Sherman, B. (1950). A random variable related to the spacing of sample values.
           The Annals of Mathematical Statistics, 21, 339-361.
    """

    @staticmethod
    @override
    def code():
        return f"SHERMAN_{AbstractUniformGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        rvs = np.asarray(rvs)
        if np.any((rvs < self.a) | (rvs > self.b)):
            raise ValueError(f"Values must be in [{self.a}, {self.b}]")

        n = len(rvs)
        x_sorted = np.sort(rvs)

        x_with_boundaries = np.concatenate([[self.a], x_sorted, [self.b]])

        spacings = np.diff(x_with_boundaries)

        expected_spacing = (self.b - self.a) / (n + 1)
        s = 0.5 * np.sum(np.abs(spacings - expected_spacing))

        return s


class QuesenberryMillerUniformGofStatistic(AbstractUniformGofStatistic):
    """
    Quesenberry and Miller's Q-test for Uniform distribution.

    References
    ----------
    .. [1] Quesenberry, C. P., & Miller Jr, F. L. (1977).
           Power studies of some tests for uniformity.
           Journal of Statistical Computation and Simulation, 5, 169-191.
    """

    @staticmethod
    @override
    def code():
        return f"QUESENBERRY_MILLER_{AbstractUniformGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        rvs = np.asarray(rvs)
        if np.any((rvs < self.a) | (rvs > self.b)):
            raise ValueError(f"Values must be in [{self.a}, {self.b}]")

        x_sorted = np.sort(rvs)

        x_with_boundaries = np.concatenate([[self.a], x_sorted, [self.b]])

        spacings = np.diff(x_with_boundaries)

        sum_squares = np.sum(spacings ** 2)

        sum_consecutive_products = np.sum(spacings[:-1] * spacings[1:])

        q = float(sum_squares) + float(sum_consecutive_products)

        return q
