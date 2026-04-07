import inspect
import sys
import warnings
from abc import ABC

import numpy as np
import scipy.stats as scipy_stats
from scipy import integrate
from typing_extensions import override

from pysatl_criterion.statistics import normal
from pysatl_criterion.statistics.common import CrammerVonMisesStatistic, KSStatistic
from pysatl_criterion.statistics.goodness_of_fit import AbstractGoodnessOfFitStatistic
from pysatl_criterion.statistics.normal import AbstractNormalityGofStatistic


class AbstractLogNormalGofStatistic(AbstractGoodnessOfFitStatistic, ABC):
    """
    Base class for Log-Normal distribution Goodness-of-Fit statistics.
    """

    def __init__(self, s=1, scale=1):
        if s <= 0:
            raise ValueError("Shape parameter s must be positive")
        if scale <= 0:
            raise ValueError("Scale parameter must be positive")

        self.s = s
        self.scale = scale

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for Log-Normal statistics.

        :return: string code in format "LOGNORMAL_{parent_code}".
        """
        return f"LOGNORMAL_{AbstractGoodnessOfFitStatistic.code()}"


# =================================================================================================
# EXPLICIT IMPLEMENTATIONS
# =================================================================================================


class KolmogorovSmirnovLogNormalGofStatistic(AbstractLogNormalGofStatistic, KSStatistic):
    """
    Kolmogorov-Smirnov test statistic for Log-Normal distribution.
    """

    @override
    def __init__(self, alternative="two-sided", mode="auto", s=1, scale=1):
        AbstractLogNormalGofStatistic.__init__(self, s=s, scale=scale)
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

        :return: string code in format "KS_LOGNORMAL_{parent_code}".
        """
        short_code = KolmogorovSmirnovLogNormalGofStatistic.short_code()
        return f"{short_code}_{AbstractLogNormalGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Kolmogorov-Smirnov test statistic for Log-Normal distribution.

        :param rvs: array of observed data samples.
        :return: Kolmogorov-Smirnov test statistic value.
        """
        rvs = np.sort(rvs)
        cdf_vals = scipy_stats.lognorm.cdf(rvs, s=self.s, scale=self.scale)
        return KSStatistic.execute_statistic(self, rvs, cdf_vals)


class CramerVonMiseLogNormalGofStatistic(AbstractLogNormalGofStatistic, CrammerVonMisesStatistic):
    """
    Cramér-von Mises test statistic for Log-Normal distribution.
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

        :return: string code in format "CVM_LOGNORMAL_{parent_code}".
        """
        short_code = CramerVonMiseLogNormalGofStatistic.short_code()
        return f"{short_code}_{AbstractLogNormalGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Cramér-von Mises test statistic for Log-Normal distribution.

        :param rvs: array of observed data samples.
        :return: Cramér-von Mises test statistic value.
        """
        rvs_sorted = np.sort(rvs)
        cdf_vals = scipy_stats.lognorm.cdf(rvs_sorted, s=self.s, scale=self.scale)
        return super().execute_statistic(rvs, cdf_vals)


class QuesenberryMillerLogNormalGofStatistic(AbstractLogNormalGofStatistic):
    """
    Quesenberry and Miller's Q-test for Log-Normal distribution.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "QUESENBERRY_MILLER".
        """
        return "QUESENBERRY_MILLER"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "QUESENBERRY_MILLER_LOGNORMAL_{parent_code}".
        """
        short_code = QuesenberryMillerLogNormalGofStatistic.short_code()
        return f"{short_code}_{AbstractLogNormalGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Quesenberry-Miller Q-test statistic for Log-Normal distribution.

        :param rvs: array of observed data samples.
        :return: Quesenberry-Miller test statistic value.
        """
        rvs = np.asarray(rvs)

        # Transform data to [0, 1] using the CDF of the Log-Normal distribution
        cdf_vals = scipy_stats.lognorm.cdf(rvs, s=self.s, scale=self.scale)

        x_sorted = np.sort(cdf_vals)

        # Add boundaries 0 and 1 because we are in the probability space [0, 1]
        x_with_boundaries = np.concatenate([[0.0], x_sorted, [1.0]])

        spacings = np.diff(x_with_boundaries)

        sum_squares = np.sum(spacings**2)

        sum_consecutive_products = np.sum(spacings[:-1] * spacings[1:])

        q = float(sum_squares) + float(sum_consecutive_products)

        return q


class KLSupremumLogNormalGoFStatistic(AbstractGoodnessOfFitStatistic):
    """
    Supremum test for lognormality based on Kullback-Leibler divergences.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "KL_SUP".
        """
        return "KL_SUP"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "KL_SUP_LOGNORMAL_{parent_code}".
        """
        short_code = KLSupremumLogNormalGoFStatistic.short_code()
        return f"{short_code}_{AbstractLogNormalGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the KL Supremum test statistic for Log-Normal distribution.

        :param rvs: array of observed data samples.
        :return: KL Supremum test statistic value (inf if data contains non-positive values).
        """
        rvs = np.array(rvs)

        if np.any(rvs <= 0):
            return float("inf")

        n = len(rvs)

        # Estimate parameters
        log_rvs = np.log(rvs)
        sigma_hat = np.std(log_rvs, ddof=0)  # MLE

        # Transform data
        Y = rvs ** (1 / sigma_hat)
        logY = np.log(Y)

        mean_logY = np.mean(logY)
        mean_logY2 = np.mean(logY**2)
        mean_logY3 = np.mean(logY**3)

        # D_{n,r} function
        def D_n_r(r):
            if abs(r) < 1e-8:
                C = -mean_logY3 + 3 * mean_logY2 * mean_logY - 2 * (mean_logY**3)
                return (np.sqrt(6) / 6) * C * (r**3)

            Yr = Y**r
            mean_Yr = np.mean(Yr)
            mean_Yr_logY = np.mean(Yr * logY)

            return 2 * np.log(mean_Yr) - r * mean_logY - r * mean_Yr_logY / mean_Yr

        # sigma_r
        def sigma_r(r):
            if abs(r) < 1e-8:
                return (abs(r) ** 3) / np.sqrt(6)

            expr = np.exp(r**2) * (r**4 - 3 * (r**2) + 4) - (4 + r**2)

            if expr < 1e-16:
                return 1e-8
            return np.sqrt(expr)

        # Calculate supremum over r in [-5, 5]
        r_grid = np.linspace(-5, 5, 1000)
        values = []
        for r in r_grid:
            D = D_n_r(r)
            sigma = sigma_r(r)
            if sigma > 1e-10 and not np.isnan(D):  # Avoid division by zero
                values.append(abs(D / sigma))

        if not values:
            return np.inf

        T1 = np.sqrt(n) * np.max(values)
        return T1

    def calculate_critical_value(self, n, alpha=0.05):
        """
        Calculate critical value for the KL Supremum test.

        :param n: sample size.
        :param alpha: significance level (default is 0.05). Supported: 0.01, 0.05,
        or range [0.001, 0.1].
        :return: critical value.
        :raises ValueError: if alpha is outside the supported range [0.001, 0.1].
        """
        if alpha == 0.05:
            # Formula (27)
            return 2.67076 - 3.90704 / np.sqrt(n) + 1.07162 / n
        elif alpha == 0.01:
            # Formula (29)
            return 4.10183 - 7.48001 / np.sqrt(n) + 0.638394 * np.log(n) / np.sqrt(n)
        else:
            # Formula (31) for general alpha in [0.001, 0.1]
            if alpha < 0.001 or alpha > 0.1:
                raise ValueError("alpha must be between 0.001 and 0.1")

            return (
                0.48632
                - 0.00148063 / alpha
                - 0.0000699 * n
                + 0.217752 / np.sqrt(alpha)
                - 2.39217 * np.sqrt(n / alpha)
                - 0.532872 / (np.sqrt(alpha) * n)
                + 0.179786 * np.log(n / alpha)
                - 0.13083 * np.log(alpha * n)
            )


class KLIntegralLogNormalGoFStatistic(AbstractLogNormalGofStatistic):
    """
    Integral test for lognormality based on Kullback-Leibler divergences.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "KL_INT".
        """
        return "KL_INT"

    @staticmethod
    @override
    def code():
        """
        Get short code identifier for this test.

        :return: short code string "KL_INT".
        """
        short_code = KLIntegralLogNormalGoFStatistic.short_code()
        return f"{short_code}_{AbstractLogNormalGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
         Execute the KL Integral test statistic for Log-Normal distribution.

         :param rvs: array of observed data samples.
         :return: KL Integral test statistic value (inf if data contains non-positive values).
         """
        rvs = np.array(rvs)

        if np.any(rvs <= 0):
            return float("inf")

        n = len(rvs)

        log_rvs = np.log(rvs)
        sigma_hat = np.std(log_rvs, ddof=0)

        Y = rvs ** (1 / sigma_hat)
        logY = np.log(Y)

        mean_logY = np.mean(logY)
        mean_logY2 = np.mean(logY**2)
        mean_logY3 = np.mean(logY**3)

        # Define D_n,r
        def D_n_r(r):
            if abs(r) < 1e-8:
                C = -mean_logY3 + 3 * mean_logY2 * mean_logY - 2 * (mean_logY**3)
                return (np.sqrt(6) / 6) * C * (r**3)

            Yr = Y**r
            mean_Yr = np.mean(Yr)
            mean_Yr_logY = np.mean(Yr * logY)

            return 2 * np.log(mean_Yr) - r * mean_logY - r * mean_Yr_logY / mean_Yr

        # sigma_r
        def sigma_r(r):
            if abs(r) < 1e-8:
                return (abs(r) ** 3) / np.sqrt(6)

            expr = np.exp(r**2) * (r**4 - 3 * (r**2) + 4) - (4 + r**2)

            if expr < 1e-16:
                return 1e-8
            return np.sqrt(expr)

        # Integrand function
        def integrand(r):
            D = D_n_r(r)
            sigma = sigma_r(r)

            if abs(r) < 1e-8:
                C = -mean_logY3 + 3 * mean_logY2 * mean_logY - 2 * (mean_logY**3)
                return (C / 6) ** 2

            ratio = D / sigma
            return ratio * ratio

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            result, error = integrate.quad(integrand, -5, 5, limit=100)

        T2 = n * result

        return T2 if not np.isnan(T2) else float("inf")

    def calculate_critical_value(self, n, alpha=0.05):
        """
        Calculate critical value for the KL Integral test.

        :param n: sample size.
        :param alpha: significance level (default is 0.05). Supported: 0.01, 0.05,
        or range [0.001, 0.1].
        :return: critical value.
        :raises ValueError: if alpha is outside the supported range [0.001, 0.1].
        """
        if alpha == 0.05:
            # Formula (28)
            return -2.88971 - 0.130082 * (np.log(n) ** 2) + 2.71403 * np.log(n)
        elif alpha == 0.01:
            # Formula (30)
            return 24.9099 - 31.9302 / np.sqrt(n) - 13.2562 * np.log(n) / np.sqrt(n)
        else:
            # Formula (32) for general alpha in [0.001, 0.1]
            if alpha < 0.001 or alpha > 0.1:
                raise ValueError("alpha must be between 0.001 and 0.1")

            return (
                0.880415
                + 0.00680135 / alpha
                - 35.179 * alpha
                - 0.18045 / (alpha * n)
                - 0.107479 * n
                + 1.67972 * np.sqrt(alpha * n)
                + 0.0119889 * n * np.log(n)
                + 0.372023 * np.sqrt(n / alpha)
                - 0.0213414 * np.sqrt(n / alpha) * np.log(n / alpha)
            )


# =================================================================================================
# DYNAMIC GENERATION
# =================================================================================================


def _create_lognormal_class(normal_cls):
    new_class_name = normal_cls.__name__.replace("Normality", "LogNormal")
    class_code = normal_cls.code().replace("NORMALITY", "LOGNORMAL")
    class_short_code = normal_cls.short_code()

    class LogNormalClass(AbstractLogNormalGofStatistic):
        @override
        def execute_statistic(self, rvs, **kwargs):
            """
            Execute the wrapped normality test on log-transformed data.

            :param rvs: array of observed data samples.
            :return: test statistic value.
            """
            rvs_arr = np.array(rvs)

            if np.any(rvs_arr <= 0):
                return float("inf")

            log_rvs = np.log(rvs_arr)
            standardized_log_rvs = (log_rvs - np.log(self.scale)) / self.s

            return normal_cls().execute_statistic(standardized_log_rvs, **kwargs)

        @staticmethod
        @override
        def short_code():
            """
            Get short code identifier for this dynamically generated test.

            :return: short code inherited from the parent Normality test.
            """
            return class_short_code

        @staticmethod
        @override
        def code():
            """
            Get unique code identifier for this dynamically generated test.

            :return: string code in format "{short_code}_LOGNORMAL_{parent_code}".
            """
            return class_code

    LogNormalClass.__name__ = new_class_name
    LogNormalClass.__qualname__ = new_class_name
    return LogNormalClass


# List of Normal statistics that we have explicitly implemented for LogNormal above.
# We should NOT generate dynamic wrappers for these.
EXPLICITLY_IMPLEMENTED_NORMAL_STATS = [
    "KolmogorovSmirnovNormalityGofStatistic",
    "CramerVonMiseNormalityGofStatistic",
    "QuesenberryMillerNormalityGofStatistic",
    "KLSupremumLogNormalGoFStatistic",
    "KLIntegralLogNormalGoFStatistic",
    "BHSNormalityGofStatistic",
]

current_module = sys.modules[__name__]
__all__ = [
    "AbstractLogNormalGofStatistic",
    "KolmogorovSmirnovLogNormalGofStatistic",
    "CramerVonMiseLogNormalGofStatistic",
    "QuesenberryMillerLogNormalGofStatistic",
    "KLSupremumLogNormalGoFStatistic",
    "KLIntegralLogNormalGoFStatistic",
]

for name, obj in inspect.getmembers(normal):
    if (
        inspect.isclass(obj)
        and issubclass(obj, AbstractNormalityGofStatistic)
        and obj is not AbstractNormalityGofStatistic
        and name not in EXPLICITLY_IMPLEMENTED_NORMAL_STATS
        and not name.startswith("Abstract")
        and not name.startswith("Graph")
    ):
        ln_class = _create_lognormal_class(obj)
        setattr(current_module, ln_class.__name__, ln_class)
        __all__.append(ln_class.__name__)
