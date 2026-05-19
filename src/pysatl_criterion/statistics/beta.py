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


class AbstractBetaGofStatistic(AbstractGoodnessOfFitStatistic, ABC):
    """
    Abstract base class for Beta distribution goodness-of-fit statistics.

    The Beta distribution is a continuous probability distribution defined on the interval [0, 1]
    parameterized by two positive shape parameters, denoted by α (alpha) and β (beta).
    """

    def __init__(self, alpha=1, beta=1):
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if beta <= 0:
            raise ValueError("beta must be positive")
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def _validate_rvs(rvs, open_interval=False):
        rvs = np.asarray(rvs)
        if open_interval:
            # All values strictly between 0 and 1
            if np.any((rvs <= 0) | (rvs >= 1)):
                raise ValueError(
                    "Beta distribution values must be in the open interval (0, 1) for this test"
                )
        else:
            # All values in [0, 1]
            if np.any((rvs < 0) | (rvs > 1)):
                raise ValueError("Beta distribution values must be in the interval [0, 1]")
        return rvs

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for Beta distribution statistics.

        :return: string code in format "BETA_{parent_code}".
        """
        return f"BETA_{AbstractGoodnessOfFitStatistic.code()}"


class KolmogorovSmirnovBetaGofStatistic(AbstractBetaGofStatistic, KSStatistic):
    """
    Kolmogorov-Smirnov test statistic for Beta distribution.

    The Kolmogorov-Smirnov test compares the empirical distribution function
    with the theoretical Beta distribution function.
    """

    def __init__(self, alpha=1, beta=1, alternative="two-sided", mode="auto"):
        AbstractBetaGofStatistic.__init__(self, alpha, beta)
        KSStatistic.__init__(self, alternative, mode)

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

        :return: string code in format "KS_BETA_{parent_code}".
        """
        short_code = KolmogorovSmirnovBetaGofStatistic.short_code()
        return f"{short_code}_{AbstractBetaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Kolmogorov-Smirnov test statistic.

        :param rvs: array of sample data from Beta distribution (values in [0, 1]).
        :return: Kolmogorov-Smirnov test statistic value.
        """
        rvs = self._validate_rvs(rvs)

        rvs_sorted = np.sort(rvs)
        cdf_vals = scipy_stats.beta.cdf(rvs_sorted, self.alpha, self.beta)
        return KSStatistic.execute_statistic(self, rvs_sorted, cdf_vals)


class AndersonDarlingBetaGofStatistic(AbstractBetaGofStatistic, ADStatistic):
    """
    Anderson-Darling test statistic for Beta distribution.

    The Anderson-Darling test is a modification of the Kolmogorov-Smirnov test
    that gives more weight to the tails of the distribution.
    """

    @staticmethod
    @override
    def short_code():
        return "AD"

    @staticmethod
    @override
    def code():
        """
        Get short code identifier for this test.

        :return: short code string "AD".
        """
        short_code = AndersonDarlingBetaGofStatistic.short_code()
        return f"{short_code}_{AbstractBetaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Anderson-Darling test statistic.

        :param rvs: array of sample data from Beta distribution (values in [0, 1]).
        :return: Anderson-Darling test statistic value.
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
    """
    Cramér-von Mises test statistic for Beta distribution.

    Goodness-of-fit test that measures the integrated squared difference between
    the empirical and theoretical cumulative distribution functions.
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

        :return: string code in format "CVM_BETA_{parent_code}".
        """
        short_code = CrammerVonMisesBetaGofStatistic.short_code()
        return f"{short_code}_{AbstractBetaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Cramér-von Mises test statistic.

        :param rvs: array of sample data from Beta distribution (values in [0, 1]).
        :return: Cramér-von Mises test statistic value.
        """
        rvs = self._validate_rvs(rvs)

        rvs_sorted = np.sort(rvs)
        cdf_vals = scipy_stats.beta.cdf(rvs_sorted, self.alpha, self.beta)
        return CrammerVonMisesStatistic.execute_statistic(self, rvs_sorted, cdf_vals)


class LillieforsTestBetaGofStatistic(AbstractBetaGofStatistic, LillieforsTest):
    """
    Lilliefors test statistic for Beta distribution.

    The Lilliefors test is a modification of the Kolmogorov-Smirnov test for
    the case when parameters are estimated from the data.
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

        :return: string code in format "LILLIE_BETA_{parent_code}".
        """
        short_code = LillieforsTestBetaGofStatistic.short_code()
        return f"{short_code}_{AbstractBetaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Lilliefors test statistic.

        :param rvs: array of sample data from Beta distribution (values in [0, 1]).
        :return: Lilliefors test statistic value.
        """
        rvs = self._validate_rvs(rvs)

        rvs_sorted = np.sort(rvs)
        cdf_vals = scipy_stats.beta.cdf(rvs_sorted, self.alpha, self.beta)
        return LillieforsTest.execute_statistic(self, rvs_sorted, cdf_vals)


class Chi2PearsonBetaGofStatistic(AbstractBetaGofStatistic, Chi2Statistic):
    """
    Pearson's Chi-squared test statistic for Beta distribution.

    The chi-squared test compares observed frequencies with expected frequencies
    based on the Beta distribution.
    """

    def __init__(self, alpha=1, beta=1, lambda_=1):
        AbstractBetaGofStatistic.__init__(self, alpha, beta)
        Chi2Statistic.__init__(self)
        self.lambda_ = lambda_

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

        :return: string code in format "CHI2_PEARSON_BETA_{parent_code}".
        """
        short_code = Chi2PearsonBetaGofStatistic.short_code()
        return f"{short_code}_{AbstractBetaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute Pearson's Chi-squared test statistic.

        :param rvs: array of sample data from Beta distribution (values in [0, 1]).
        :return: Chi-squared test statistic value.
        """
        rvs = self._validate_rvs(rvs)

        rvs_sorted = np.sort(rvs)
        n = len(rvs_sorted)

        # Number of bins using Sturges' rule
        num_bins = int(np.ceil(np.sqrt(n)))

        # Create histogram
        observed, bin_edges = np.histogram(rvs_sorted, bins=num_bins, range=(0, 1))

        # Calculate expected frequencies
        expected_cdf = scipy_stats.beta.cdf(bin_edges, self.alpha, self.beta)
        expected = np.diff(expected_cdf) * n

        return Chi2Statistic.execute_statistic(self, observed, expected, self.lambda_)


class WatsonBetaGofStatistic(AbstractBetaGofStatistic):
    """
    Watson test statistic for Beta distribution.

    The Watson test is a modification of the Cramér-von Mises test that is
    invariant under location changes.
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

        :return: string code in format "W_BETA_{parent_code}".
        """
        short_code = WatsonBetaGofStatistic.short_code()
        return f"{short_code}_{AbstractBetaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Watson test statistic.

        :param rvs: array of sample data from Beta distribution (values in [0, 1]).
        :return: Watson test statistic value.
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
    """
    Kuiper test statistic for Beta distribution.

    The Kuiper test is a variant of the Kolmogorov-Smirnov test that is
    more sensitive to deviations in the tails of the distribution.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "KUIPER".
        """
        return "KUIPER"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "KUIPER_BETA_{parent_code}".
        """
        short_code = KuiperBetaGofStatistic.short_code()
        return f"{short_code}_{AbstractBetaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Kuiper test statistic.

        :param rvs: array of sample data from Beta distribution (values in [0, 1]).
        :return: Kuiper test statistic value.
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
    """
    Moment-based test statistic for Beta distribution.

    This test compares the sample moments with the theoretical moments of
    the Beta distribution. It uses the first two moments (mean and variance).
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "MB".
        """
        return "MB"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "MB_BETA_{parent_code}".
        """
        short_code = MomentBasedBetaGofStatistic.short_code()
        return f"{short_code}_{AbstractBetaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the moment-based test statistic.

        :param rvs: array of sample data from Beta distribution (values in [0, 1]).
        :return: moment-based test statistic value.
        """
        rvs = self._validate_rvs(rvs)

        n = len(rvs)

        # Sample moments
        sample_mean = np.mean(rvs)
        sample_var = np.var(rvs, ddof=1)

        # Theoretical moments
        alpha_beta_sum = self.alpha + self.beta
        theoretical_mean = self.alpha / alpha_beta_sum
        theoretical_var = (self.alpha * self.beta) / (alpha_beta_sum**2 * (alpha_beta_sum + 1))

        # Test statistic based on standardized differences
        mean_diff = np.sqrt(n) * (sample_mean - theoretical_mean) / np.sqrt(theoretical_var)
        var_diff = np.sqrt(n) * (sample_var - theoretical_var) / np.sqrt(theoretical_var)

        # Combined statistic
        statistic = mean_diff**2 + var_diff**2

        return statistic


class SkewnessKurtosisBetaGofStatistic(AbstractBetaGofStatistic):
    """
    Skewness-Kurtosis test statistic for Beta distribution.

    This test compares the sample skewness and kurtosis with the theoretical
    values of the Beta distribution.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "SK".
        """
        return "SK"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "SK_BETA_{parent_code}".
        """
        short_code = SkewnessKurtosisBetaGofStatistic.short_code()
        return f"{short_code}_{AbstractBetaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the skewness-kurtosis test statistic.

        :param rvs: array of sample data from Beta distribution (values in [0, 1]).
        :return: skewness-kurtosis test statistic value.
        """
        rvs = self._validate_rvs(rvs)

        n = len(rvs)

        # Sample skewness and kurtosis
        from scipy.stats import kurtosis, skew

        sample_skewness = skew(rvs, bias=False)
        sample_kurtosis = kurtosis(rvs, bias=False)

        # Theoretical skewness and kurtosis
        alpha_beta_sum = self.alpha + self.beta
        theoretical_skewness = (
            2
            * (self.beta - self.alpha)
            * np.sqrt(alpha_beta_sum + 1)
            / ((alpha_beta_sum + 2) * np.sqrt(self.alpha * self.beta))
        )

        numerator = 6 * (
            (self.alpha - self.beta) ** 2 * (alpha_beta_sum + 1)
            - self.alpha * self.beta * (alpha_beta_sum + 2)
        )
        denominator = self.alpha * self.beta * (alpha_beta_sum + 2) * (alpha_beta_sum + 3)
        theoretical_kurtosis = numerator / denominator

        # Test statistic (similar to Jarque-Bera)
        skew_diff = (sample_skewness - theoretical_skewness) ** 2
        kurt_diff = (sample_kurtosis - theoretical_kurtosis) ** 2

        statistic = (n / 6) * skew_diff + (n / 24) * kurt_diff

        return statistic


class RatioBetaGofStatistic(AbstractBetaGofStatistic):
    """
    Ratio test statistic for Beta distribution.

    This test is based on the ratio of geometric mean to arithmetic mean,
    which has a known relationship for the Beta distribution.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "RT".
        """
        return "RT"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "RT_BETA_{parent_code}".
        """
        short_code = RatioBetaGofStatistic.short_code()
        return f"{short_code}_{AbstractBetaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the ratio test statistic.

        :param rvs: array of sample data from Beta distribution (values in (0, 1)).
        :return: ratio test statistic value.
        :raises ValueError: if arithmetic mean is zero.
        """
        rvs = self._validate_rvs(rvs, open_interval=True)

        n = len(rvs)

        # Sample statistics
        arithmetic_mean = np.mean(rvs)
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
    """
    Entropy-based test statistic for Beta distribution.

    This test compares the sample entropy (estimated using kernel density)
    with the theoretical entropy of the Beta distribution.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "ENT".
        """
        return "ENT"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "ENT_BETA_{parent_code}".
        """
        short_code = EntropyBetaGofStatistic.short_code()
        return f"{short_code}_{AbstractBetaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the entropy-based test statistic.

        :param rvs: array of sample data from Beta distribution (values in [0, 1]).
        :return: entropy-based test statistic value.
        """
        rvs = self._validate_rvs(rvs)

        n = len(rvs)
        m = kwargs.get("m", n // 4)

        # Sort the data
        rvs_sorted = np.sort(rvs)

        # Vasicek's entropy estimator
        sample_entropy = 0
        for i in range(n):
            left = max(0, i - m)
            right = min(n - 1, i + m)
            window_range = rvs_sorted[right] - rvs_sorted[left]
            if window_range > 0:
                sample_entropy += np.log(window_range * n / (2 * m))

        sample_entropy /= n

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
    """
    Mode-based test statistic for Beta distribution.

    For Beta(α, β) with α, β > 1, the mode is (α-1)/(α+β-2).
    This test compares the sample mode (estimated) with the theoretical mode.
    """

    def __init__(self, alpha=2, beta=2):
        if alpha <= 1:
            raise ValueError("alpha must be greater than 1 for mode to be well-defined")
        if beta <= 1:
            raise ValueError("beta must be greater than 1 for mode to be well-defined")
        super().__init__(alpha, beta)

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "MODE".
        """
        return "MODE"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "MODE_BETA_{parent_code}".
        """
        short_code = ModeBetaGofStatistic.short_code()
        return f"{short_code}_{AbstractBetaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the mode-based test statistic.

        :param rvs: array of sample data from Beta distribution (values in [0, 1]).
        :return: mode-based test statistic value.
        """
        rvs = self._validate_rvs(rvs)

        n = len(rvs)

        # Estimate sample mode using kernel density estimation
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(rvs)
        x_grid = np.linspace(0.01, 0.99, 200)
        density = kde(x_grid)
        sample_mode = x_grid[np.argmax(density)]

        # Theoretical mode
        theoretical_mode = (self.alpha - 1) / (self.alpha + self.beta - 2)

        # Test statistic
        statistic = np.sqrt(n) * np.abs(sample_mode - theoretical_mode)

        return statistic
