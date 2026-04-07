from abc import ABC

import numpy as np
import scipy.stats as scipy_stats
from scipy import special
from typing_extensions import override

from pysatl_criterion.statistics.models import AbstractStatistic


class KSStatistic(AbstractStatistic, ABC):
    """
    Kolmogorov-Smirnov statistic base class.

    Implements the KS test for comparing empirical distribution function
    with theoretical CDF. Supports one-sided and two-sided alternatives.
    """

    def __init__(self, alternative="two-sided", mode="auto"):
        """
        Initialize Kolmogorov-Smirnov statistic.

        :param alternative: defines the alternative hypothesis ('two-sided', 'less', 'greater').
        :param mode: defines method for p-value calculation ('auto', 'exact', 'approx', 'asymp').
        """
        self.alternative = alternative
        if mode == "auto":  # Always select exact
            mode = "exact"
        self.mode = mode

    @override
    def execute_statistic(self, rvs, cdf_vals=None):
        """
        Execute the Kolmogorov-Smirnov test statistic.

        :param rvs: unsorted vector of observed data samples.
        :param cdf_vals: theoretical CDF values evaluated at sorted rvs.
        :return: KS test statistic value (D+, D-, or max(D+, D-) depending on alternative).
        """

        d_minus, _ = KSStatistic.__compute_dminus(cdf_vals, rvs)

        if self.alternative == "greater":
            d_plus, d_location = KSStatistic.__compute_dplus(cdf_vals, rvs)
            return d_plus
        if self.alternative == "less":
            d_minus, d_location = KSStatistic.__compute_dminus(cdf_vals, rvs)
            return d_minus

        # alternative == 'two-sided':
        d_plus, d_plus_location = KSStatistic.__compute_dplus(cdf_vals, rvs)
        d_minus, d_minus_location = KSStatistic.__compute_dminus(cdf_vals, rvs)
        if d_plus > d_minus:
            D = d_plus
            # d_location = d_plus_location
            # d_sign = 1
        else:
            D = d_minus
            # d_location = d_minus_location
            # d_sign = -1
        return D

    @override
    def calculate_critical_value(self, rvs_size, sl):
        """
        Calculate critical value for Kolmogorov-Smirnov test.

        :param rvs_size: sample size.
        :param sl: significance level.
        :return: critical value from Kolmogorov distribution.
        """
        return scipy_stats.distributions.kstwo.ppf(1 - sl, rvs_size)

    @staticmethod
    def __compute_dplus(cdf_vals, rvs):
        """
        Compute D+ statistic (maximum positive deviation).

        :param cdf_vals: theoretical CDF values at sorted data points.
        :param rvs: sorted array of observed data samples.
        :return: tuple of (D+ value, location where maximum occurs).
        """
        n = len(cdf_vals)
        d_plus = np.arange(1.0, n + 1) / n - cdf_vals
        a_max = d_plus.argmax()
        loc_max = rvs[a_max]
        return d_plus[a_max], loc_max

    @staticmethod
    def __compute_dminus(cdf_vals, rvs):
        """
        Compute D- statistic (maximum negative deviation).

        :param cdf_vals: theoretical CDF values at sorted data points.
        :param rvs: sorted array of observed data samples.
        :return: tuple of (D- value, location where maximum occurs).
        """
        n = len(cdf_vals)
        d_minus = cdf_vals - np.arange(0.0, n) / n
        a_max = d_minus.argmax()
        loc_max = rvs[a_max]
        return d_minus[a_max], loc_max


class ADStatistic(AbstractStatistic):
    """
    Anderson-Darling statistic base class.

    Implements the AD test which gives more weight to the tails of the
    distribution compared to Kolmogorov-Smirnov test.
    """

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for Anderson-Darling statistic.

        :return: string code identifier.
        :raises NotImplementedError: method must be implemented by subclass.
        """
        raise NotImplementedError("Method is not implemented")

    @override
    def execute_statistic(self, rvs, log_cdf=None, log_sf=None, w=None):
        """
        Execute the Anderson-Darling test statistic.

        :param rvs: array of observed data samples.
        :param log_cdf: log of theoretical CDF values at sorted data points.
        :param log_sf: log of survival function (1 - CDF) values.
        :param w: optional weights for weighted AD statistic.
        :return: Anderson-Darling test statistic value.
        """
        n = len(rvs)

        i = np.arange(1, n + 1)
        A2 = -n - np.sum((2 * i - 1.0) / n * (log_cdf + log_sf[::-1]), axis=0)
        return A2


class LillieforsTest(KSStatistic, ABC):
    """
    Lilliefors test base class.

    Modification of Kolmogorov-Smirnov test for the case when distribution
    parameters are estimated from the data rather than specified a priori.
    """

    alternative = "two-sided"
    mode = "auto"

    @override
    def execute_statistic(self, z, cdf_vals=None):
        """
        Execute the Lilliefors test statistic.

        :param z: array of observed data samples.
        :param cdf_vals: theoretical CDF values evaluated at sorted data.
        :return: Lilliefors test statistic value.
        """
        return super().execute_statistic(z, cdf_vals)


class CrammerVonMisesStatistic(AbstractStatistic, ABC):
    """
    Cramér-von Mises statistic base class.

    Goodness-of-fit test that measures the integrated squared difference
    between empirical and theoretical cumulative distribution functions.
    """

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for Cramér-von Mises statistic.

        :return: string code "CVM".
        """
        return "CVM"

    @override
    def execute_statistic(self, rvs, cdf_vals):
        """
        Execute the Cramér-von Mises test statistic.

        :param rvs: array of observed data samples.
        :param cdf_vals: theoretical CDF values at sorted data points.
        :return: Cramér-von Mises test statistic value.
        """
        n = len(rvs)

        u = (2 * np.arange(1, n + 1) - 1) / (2 * n)
        w = 1 / (12 * n) + np.sum((u - cdf_vals) ** 2)

        return w


class Chi2Statistic(AbstractStatistic, ABC):
    """
    Chi-squared statistic base class.

    Implements Pearson's chi-squared test and generalizations via the
    Cressie-Read power divergence family.
    """

    @staticmethod
    def _m_sum(a, *, axis, preserve_mask, xp):
        """
        Compute sum with support for masked arrays.

        :param a: array to sum.
        :param axis: axis along which to sum.
        :param preserve_mask: whether to preserve mask in result.
        :param xp: array module (numpy or compatible).
        :return: sum of array elements along specified axis.
        """
        if np.ma.isMaskedArray(a):
            s = a.sum(axis)
            return s if preserve_mask else np.asarray(s)
        return xp.sum(a, axis=axis)

    @override
    def execute_statistic(self, f_obs, f_exp, lambda_):
        """
        Execute the chi-squared test statistic using power divergence.

        :param f_obs: observed frequencies.
        :param f_exp: expected frequencies under null hypothesis.
        :param lambda_: power divergence parameter:
            lambda_=1 for Pearson's chi-squared,
            lambda_=0 for log-likelihood ratio (G-test),
            lambda_=-1 for modified log-likelihood ratio.

        :return: chi-squared test statistic value.
        """
        # `terms` is the array of terms that are summed along `axis` to create
        # the test statistic.  We use some specialized code for a few special
        # cases of lambda_.
        f_obs = np.array(f_obs)
        if lambda_ == 1:
            # Pearson's chi-squared statistic
            terms = (f_obs - f_exp) ** 2 / f_exp
        elif lambda_ == 0:
            # Log-likelihood ratio (i.e. G-test)
            terms = 2.0 * special.xlogy(f_obs, f_obs / f_exp)
        elif lambda_ == -1:
            # Modified log-likelihood ratio
            terms = 2.0 * special.xlogy(f_exp, f_exp / f_obs)
        else:
            # General Cressie-Read power divergence.
            terms = f_obs * ((f_obs / f_exp) ** lambda_ - 1)
            terms /= 0.5 * lambda_ * (lambda_ + 1)

        return terms.sum()

    @override
    def calculate_critical_value(self, rvs_size, sl):
        """
        Calculate critical value for chi-squared test.

        :param rvs_size: sample size (used to compute degrees of freedom).
        :param sl: significance level.
        :return: critical value from chi-squared distribution.
        """
        return scipy_stats.distributions.chi2.ppf(1 - sl, rvs_size - 1)


class MinToshiyukiStatistic(AbstractStatistic, ABC):
    """
    Min-Toshiyuki statistic base class.

    Goodness-of-fit test based on weighted maximum deviation between
    empirical and theoretical distribution functions.
    """

    @override
    def execute_statistic(self, cdf_vals):
        """
        Execute the Min-Toshiyuki test statistic.

        :param cdf_vals: theoretical CDF values at sorted data points.
        :return: Min-Toshiyuki test statistic value.
        """
        n = len(cdf_vals)
        d_plus = np.arange(1.0, n + 1) / n - cdf_vals
        d_minus = cdf_vals - np.arange(0.0, n) / n
        d = np.maximum.reduce([d_plus, d_minus])

        fi = 1 / (cdf_vals * (1 - cdf_vals))

        s = np.sum(d * np.sqrt(fi))
        return s / np.sqrt(n)


# TODO: fix signatures