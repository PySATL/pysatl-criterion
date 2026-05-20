from abc import ABC

import numpy as np
import scipy.stats as scipy_stats
from typing_extensions import override

from pysatl_criterion.statistics.common import ADStatistic, CrammerVonMisesStatistic, KSStatistic
from pysatl_criterion.statistics.goodness_of_fit import AbstractGoodnessOfFitStatistic


class AbstractStudentGofStatistic(AbstractGoodnessOfFitStatistic, ABC):
    """
    Abstract base class for Student's t-distribution goodness-of-fit statistics.
    """

    def __init__(self, df: float = 1, loc: float = 0, scale: float = 1):
        if df <= 0:
            raise ValueError("Degrees of freedom must be positive")
        if scale <= 0:
            raise ValueError("Scale must be positive")
        self.df = df
        self.loc = loc
        self.scale = scale

    @staticmethod
    @override
    def code():
        """
        Return the unique identifier code for Student's t-distribution GoF statistics.

        :return: string code "STUDENT_{parent_code}".
        """
        return f"STUDENT_{AbstractGoodnessOfFitStatistic.code()}"

    def _cdf_clipped(self, standardized: np.ndarray, eps: float = 1e-10) -> np.ndarray:
        """
        Compute the Student t CDF for standardized values and clip to avoid log(0) issues.

        :param standardized: array of standardized sample values (x - loc) / scale.
        :param eps: small clipping epsilon to keep CDF in [eps, 1-eps] (default is 1e-10).
        :return: array of clipped CDF values.
        """
        cdf_vals = scipy_stats.t.cdf(standardized, self.df)
        return np.clip(cdf_vals, eps, 1 - eps)


class KolmogorovSmirnovStudentGofStatistic(AbstractStudentGofStatistic, KSStatistic):
    """
    Kolmogorov-Smirnov test statistic for the Student's t-distribution.
    """

    def __init__(
        self,
        df: float = 1,
        loc: float = 0,
        scale: float = 1,
        alternative: str = "two-sided",
    ):
        AbstractStudentGofStatistic.__init__(self, df, loc, scale)
        KSStatistic.__init__(self, alternative)

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
        Return the unique identifier code for this statistic.

        :return: string code "KS_STUDENT_{parent_code}".
        """
        short_code = KolmogorovSmirnovStudentGofStatistic.short_code()
        return f"{short_code}_{AbstractStudentGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Calculate the Kolmogorov-Smirnov statistic for testing fit to Student's t-distribution.

        :param rvs: array of sample data.
        :return: Kolmogorov-Smirnov test statistic value.
        """
        rvs = np.sort(rvs)
        # Standardize the data
        standardized = (rvs - self.loc) / self.scale
        cdf_vals = scipy_stats.t.cdf(standardized, self.df)
        return KSStatistic.execute_statistic(self, rvs, cdf_vals)


class AndersonDarlingStudentGofStatistic(AbstractStudentGofStatistic, ADStatistic):
    """
    Anderson-Darling test statistic for the Student's t-distribution.
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
        Return the unique identifier code for this statistic.

        :return: string code "AD_STUDENT_{parent_code}".
        """
        short_code = AndersonDarlingStudentGofStatistic.short_code()
        return f"{short_code}_{AbstractStudentGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Calculate the Anderson-Darling statistic for testing fit to Student's t-distribution.

        :param rvs: array of sample data.
        :return: Anderson-Darling test statistic value.
        """
        y = np.sort(rvs)
        # Standardize the data
        standardized = (y - self.loc) / self.scale
        logcdf = scipy_stats.t.logcdf(standardized, self.df)
        logsf = scipy_stats.t.logsf(standardized, self.df)
        return ADStatistic.execute_statistic(self, y, log_cdf=logcdf, log_sf=logsf)


class CramerVonMisesStudentGofStatistic(AbstractStudentGofStatistic, CrammerVonMisesStatistic):
    """
    Cramer-von Mises test statistic for the Student's t-distribution.
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
        Return the unique identifier code for this statistic.

        :return: string code "CVM_STUDENT_{parent_code}".
        """
        short_code = CramerVonMisesStudentGofStatistic.short_code()
        return f"{short_code}_{AbstractStudentGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Calculate the Cramer-von Mises statistic for testing fit to Student's t-distribution.

        :param rvs: array of sample data.
        :return: Cramer-von Mises test statistic value.
        """
        rvs = np.sort(rvs)
        # Standardize the data
        standardized = (rvs - self.loc) / self.scale
        cdf_vals = scipy_stats.t.cdf(standardized, self.df)
        return CrammerVonMisesStatistic.execute_statistic(self, rvs, cdf_vals)


class KuiperStudentGofStatistic(AbstractStudentGofStatistic):
    """
    Kuiper's test statistic for the Student's t-distribution.
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
        Return the unique identifier code for this statistic.

        :return: string code "KUIPER_STUDENT_{parent_code}".
        """
        short_code = KuiperStudentGofStatistic.short_code()
        return f"{short_code}_{AbstractStudentGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Calculate Kuiper's statistic for testing fit to Student's t-distribution.

        :param rvs: array of sample data.
        :return: Kuiper's test statistic value (D+ + D-).
        """
        n = len(rvs)
        rvs = np.sort(rvs)
        # Standardize the data
        standardized = (rvs - self.loc) / self.scale
        cdf_vals = scipy_stats.t.cdf(standardized, self.df)

        # D+ and D-
        d_plus = np.max(np.arange(1.0, n + 1) / n - cdf_vals)
        d_minus = np.max(cdf_vals - np.arange(0.0, n) / n)

        return d_plus + d_minus


class WatsonStudentGofStatistic(AbstractStudentGofStatistic):
    """
    Watson's U^2 test statistic for the Student's t-distribution.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "WATSON".
        """
        return "WATSON"

    @staticmethod
    @override
    def code():
        """
        Return the unique identifier code for this statistic.

        :return: string code "WATSON_STUDENT_{parent_code}".
        """
        short_code = WatsonStudentGofStatistic.short_code()
        return f"{short_code}_{AbstractStudentGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Calculate Watson's U^2 statistic for testing fit to Student's t-distribution.

        :param rvs: array of sample data.
        :return: Watson's U^2 test statistic value.
        """
        n = len(rvs)
        rvs = np.sort(rvs)
        # Standardize the data
        standardized = (rvs - self.loc) / self.scale
        cdf_vals = scipy_stats.t.cdf(standardized, self.df)

        # Cramer-von Mises statistic
        u = (2 * np.arange(1, n + 1) - 1) / (2 * n)
        w2 = 1 / (12 * n) + np.sum((u - cdf_vals) ** 2)

        # Mean of CDF values
        f_bar = np.mean(cdf_vals)

        # Watson's U^2
        u2 = w2 - n * (f_bar - 0.5) ** 2

        return u2


class ZhangZcStudentGofStatistic(AbstractStudentGofStatistic):
    """
    Zhang's Zc test statistic for the Student's t-distribution.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "ZHANG_ZC".
        """
        return "ZHANG_ZC"

    @staticmethod
    @override
    def code():
        """
        Return the unique identifier code for this statistic.

        :return: string code "ZHANG_ZC_STUDENT_{parent_code}".
        """
        short_code = ZhangZcStudentGofStatistic.short_code()
        return f"{short_code}_{AbstractStudentGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Calculate Zhang's Zc statistic for testing fit to Student's t-distribution.

        :param rvs: array of sample data.
        :return: Zhang's Zc test statistic value.
        """
        n = len(rvs)
        rvs = np.sort(rvs)
        # Standardize the data
        standardized = (rvs - self.loc) / self.scale
        cdf_vals = self._cdf_clipped(standardized)

        i = np.arange(1, n + 1)
        # Zhang's Zc statistic
        term1 = 1 / cdf_vals - 1
        term2 = (n - 0.5) / i - 1
        # Avoid division by zero and log of negative numbers
        term2 = np.where(term2 <= 0, 1e-10, term2)
        zc = np.sum(np.log(term1 / term2) ** 2)

        return zc


class ZhangZaStudentGofStatistic(AbstractStudentGofStatistic):
    """
    Zhang's Za test statistic for the Student's t-distribution.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "ZHANG_ZA".
        """
        return "ZHANG_ZA"

    @staticmethod
    @override
    def code():
        """
        Return the unique identifier code for this statistic.

        :return: string code "ZHANG_ZA_STUDENT_{parent_code}".
        """
        short_code = ZhangZaStudentGofStatistic.short_code()
        return f"{short_code}_{AbstractStudentGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Calculate Zhang's Za statistic for testing fit to Student's t-distribution.

        :param rvs: array of sample data.
        :return: Zhang's Za test statistic value.
        """
        n = len(rvs)
        rvs = np.sort(rvs)
        # Standardize the data
        standardized = (rvs - self.loc) / self.scale
        cdf_vals = self._cdf_clipped(standardized)

        i = np.arange(1, n + 1)
        # Zhang's Za statistic
        za = -np.sum(np.log(cdf_vals) / (n - i + 0.5)) - np.sum(np.log(1 - cdf_vals) / (i - 0.5))

        return za


class LillieforsStudentGofStatistic(AbstractStudentGofStatistic, KSStatistic):
    """
    Lilliefors-type test statistic for the Student's t-distribution.
    """

    def __init__(self, df: float = 1):
        AbstractStudentGofStatistic.__init__(self, df, 0, 1)
        KSStatistic.__init__(self, "two-sided")

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
        Return the unique identifier code for this statistic.

        :return: string code "LILLIE_STUDENT_{parent_code}".
        """
        short_code = LillieforsStudentGofStatistic.short_code()
        return f"{short_code}_{AbstractStudentGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Calculate the Lilliefors statistic for testing fit to Student's t-distribution.

        Location and scale parameters are estimated from the sample data.

        :param rvs: array of sample data.
        :return: Lilliefors test statistic value.
        :raises ValueError: if sample standard deviation is zero.
        """
        x = np.asarray(rvs)
        # Estimate location and scale from data
        loc = np.mean(x)
        scale = np.std(x, ddof=1)
        if scale == 0:
            raise ValueError("Sample standard deviation is zero; Lilliefors undefined")
        # Standardize
        z = (x - loc) / scale
        cdf_vals = scipy_stats.t.cdf(z, self.df)
        return KSStatistic.execute_statistic(self, z, cdf_vals)


class ChiSquareStudentGofStatistic(AbstractStudentGofStatistic):
    """
    Chi-square goodness-of-fit test statistic for the Student's t-distribution.
    """

    def __init__(
        self,
        df: float = 1,
        loc: float = 0,
        scale: float = 1,
        n_bins: int = 10,
    ):
        super().__init__(df, loc, scale)
        self.n_bins = n_bins

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "CHI2".
        """
        return "CHI2"

    @staticmethod
    @override
    def code():
        """
        Return the unique identifier code for this statistic.

        :return: string code "CHI2_STUDENT_{parent_code}".
        """
        short_code = ChiSquareStudentGofStatistic.short_code()
        return f"{short_code}_{AbstractStudentGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Calculate the Chi-square statistic for testing fit to Student's t-distribution.

        :param rvs: array of sample data.
        :return: Chi-square test statistic value.
        """
        n = len(rvs)
        # Standardize the data
        standardized = (np.array(rvs) - self.loc) / self.scale

        # Create bin edges based on quantiles of the t-distribution
        bin_edges = scipy_stats.t.ppf(np.linspace(0, 1, self.n_bins + 1), self.df)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        # Observed frequencies
        observed, _ = np.histogram(standardized, bins=bin_edges)

        # Expected frequencies (uniform for equiprobable bins)
        expected = np.ones(self.n_bins) * n / self.n_bins

        # Chi-square statistic
        chi2 = np.sum((observed - expected) ** 2 / expected)

        return chi2
