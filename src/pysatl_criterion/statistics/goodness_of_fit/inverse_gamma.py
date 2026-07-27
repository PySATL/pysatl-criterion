from __future__ import annotations

from abc import ABC

import numpy as np
import scipy.stats as scipy_stats
from typing_extensions import override

from pysatl_criterion import DistributionType
from pysatl_criterion.statistics import AbstractGoodnessOfFitStatistic
from pysatl_criterion.statistics.alternative import (
    Alternative,
    AlternativeType,
    LeftAlternative,
    RightAlternative,
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


class AbstractInverseGammaGofStatistic(AbstractGoodnessOfFitStatistic, ABC):
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):

        if alpha <= 0:
            raise ValueError("Shape must be positive.")
        if beta <= 0:
            raise ValueError("Scale must be positive.")
        self.alpha = alpha
        self.beta = beta

    @override
    def hypothesis(self) -> GoodnessOfFitHypothesis:
        return GoodnessOfFitHypothesis({"alpha": self.alpha, "beta": self.beta})

    @staticmethod
    @override
    def distribution() -> DistributionType:
        return DistributionType.INVERSE_GAMMA

    @staticmethod
    @override
    def code() -> str:
        return f"INV_GAMMA_{AbstractGoodnessOfFitStatistic.code()}"


class KolmogorovSmirnovInverseGammaGofStatistic(AbstractInverseGammaGofStatistic, KSStatistic):
    @override
    def __init__(
        self,
        alternative_type: AlternativeType = AlternativeType.TWO_TAILED,
        mode="auto",
        alpha: float = 1.0,
        beta: float = 1.0,
    ):
        AbstractInverseGammaGofStatistic.__init__(self, alpha=alpha, beta=beta)
        KSStatistic.__init__(self, alternative_type=alternative_type, mode=mode)

    @staticmethod
    @override
    def short_code() -> str:
        return "KS"

    @staticmethod
    @override
    def code() -> str:
        short_code = KolmogorovSmirnovInverseGammaGofStatistic.short_code()
        return f"{short_code}_{AbstractInverseGammaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Kolmogorov-Smirnov test statistic for Inverse Gamma distribution.

        :param rvs: array of observations assumed to follow InvGamma(shape, scale).
        :return: Kolmogorov–Smirnov D statistic computed with the InvGamma CDF.
        """
        sorted_rvs = np.sort(np.asarray(rvs))
        cdf_vals = scipy_stats.invgamma.cdf(sorted_rvs, a=self.alpha, scale=self.beta)
        return KSStatistic.do_execute_statistic(self, sorted_rvs, cdf_vals)


class LillieforsInverseGammaGofStatistic(AbstractInverseGammaGofStatistic, LillieforsTest):
    @staticmethod
    @override
    def short_code() -> str:
        return "LILLIE"

    @staticmethod
    @override
    def code() -> str:
        short_code = LillieforsInverseGammaGofStatistic.short_code()
        return f"{short_code}_{AbstractInverseGammaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs):
        """
        Execute the Lilliefors test statistic for Inverse Gamma distribution.

        :param rvs: array of observations assumed to follow InvGamma(shape, scale).
        :return: Lilliefors-adjusted Kolmogorov–Smirnov statistic with estimated
                 Inverse Gamma parameters.
        :raises ValueError: if sample is empty or mean/variance is not valid for
                             Inverse Gamma estimation.
        """
        sample = np.asarray(rvs, dtype=float)
        n = sample.size
        if n == 0:
            raise ValueError("At least one observation is required for the Lilliefors statistic.")
        mean = np.mean(sample)
        var = np.var(sample, ddof=1)

        if mean <= 0 or var <= 0:
            raise ValueError(
                "Sample mean and variance must be positive for Inverse Gamma parameter estimation."
            )

        alpha_hat = 2.0 + (mean**2) / var
        beta_hat = mean * (alpha_hat - 1.0)

        if alpha_hat <= 2.0 or beta_hat <= 0.0:
            raise ValueError("Estimated parameters are out of valid range for Inverse Gamma ")

        sorted_sample = np.sort(sample)
        cdf_vals = scipy_stats.invgamma.cdf(sorted_sample, a=alpha_hat, scale=beta_hat)

        return super(LillieforsTest, self).do_execute_statistic(sorted_sample, cdf_vals)


class AndersonDarlingInverseGammaGofStatistic(AbstractInverseGammaGofStatistic, ADStatistic):
    @staticmethod
    @override
    def short_code() -> str:
        return "AD"

    @staticmethod
    @override
    def code() -> str:
        short_code = AndersonDarlingInverseGammaGofStatistic.short_code()
        return f"{short_code}_{AbstractInverseGammaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        sorted_rvs = np.sort(np.asarray(rvs))

        log_cdf = scipy_stats.invgamma.logcdf(sorted_rvs, a=self.alpha, scale=self.beta)
        log_sf = scipy_stats.invgamma.logsf(sorted_rvs, a=self.alpha, scale=self.beta)

        return super().do_execute_statistic(sorted_rvs, log_cdf=log_cdf, log_sf=log_sf)


class CramerVonMisesInverseGammaGofStatistic(
    AbstractInverseGammaGofStatistic, CrammerVonMisesStatistic
):
    @staticmethod
    @override
    def short_code() -> str:
        return "CVM"

    @staticmethod
    @override
    def code() -> str:
        short_code = CramerVonMisesInverseGammaGofStatistic.short_code()
        return f"{short_code}_{AbstractInverseGammaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        sorted_rvs = np.sort(np.asarray(rvs))

        cdf_vals = scipy_stats.invgamma.cdf(sorted_rvs, a=self.alpha, scale=self.beta)

        return CrammerVonMisesStatistic.do_execute_statistic(self, sorted_rvs, cdf_vals)


class AbstractBinnedInverseGammaGofStatistic(AbstractInverseGammaGofStatistic, Chi2Statistic, ABC):
    lambda_value: float = 1.0

    def __init__(self, bins: int = 8, alpha: float = 1.0, beta: float = 1.0):
        if bins < 2:
            raise ValueError("At least two bins are required for binned Inverse Gamma statistics.")
        self.bins = bins
        AbstractInverseGammaGofStatistic.__init__(self, alpha=alpha, beta=beta)
        self.lambda_value = getattr(self, "lambda_value", 1.0)

    def _counts_and_expected(self, rvs):

        sample = np.asarray(rvs)
        n = sample.size
        if n == 0:
            raise ValueError(
                "At least one observation is required for binned Inverse Gamma statistics."
            )
        if np.any(sample <= 0):
            raise ValueError(
                "All observations must be strictly positive for Inverse Gamma distribution."
            )
        quantiles = np.linspace(0.0, 1.0, self.bins + 1)
        edges = scipy_stats.invgamma.ppf(quantiles, a=self.alpha, scale=1.0 / self.beta)
        edges[0] = -np.inf
        edges[-1] = np.inf
        counts, _ = np.histogram(sample, bins=edges)
        expected = np.full(self.bins, n / self.bins)

        return counts, expected

    @override
    def execute_statistic(self, rvs):
        counts, expected = self._counts_and_expected(rvs)
        return float(
            Chi2Statistic.do_execute_statistic(self, counts, expected, lambda_=self.lambda_value)
        )


class Chi2PearsonInverseGammaGofStatistic(AbstractBinnedInverseGammaGofStatistic):
    lambda_value = 1.0

    @staticmethod
    @override
    def short_code() -> str:

        return "CHI2_PEARSON"

    @staticmethod
    @override
    def code() -> str:
        short_code = Chi2PearsonInverseGammaGofStatistic.short_code()
        return f"{short_code}_{AbstractInverseGammaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs):
        return super().execute_statistic(rvs)


class WatsonInverseGammaGofStatistic(AbstractInverseGammaGofStatistic):
    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        return "WAT"

    @staticmethod
    @override
    def code():
        short_code = WatsonInverseGammaGofStatistic.short_code()
        return f"{short_code}_{AbstractInverseGammaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):

        sorted_rvs = np.sort(np.asarray(rvs))
        n = len(sorted_rvs)

        if n == 0:
            raise ValueError("Sample cannot be empty.")

        if np.any(sorted_rvs <= 0):
            raise ValueError("Inverse Gamma requires strictly positive observations.")
        cdf_vals = scipy_stats.invgamma.cdf(sorted_rvs, a=self.alpha, scale=1.0 / self.beta)

        u = (2 * np.arange(1, n + 1) - 1) / (2 * n)
        diff = cdf_vals - u
        w_squared = 1.0 / (12 * n) + np.sum(diff**2)
        mean_adj = np.sum(cdf_vals) - n / 2.0
        return float(w_squared - (mean_adj**2) / n)


class KuiperInverseGammaGofStatistic(AbstractInverseGammaGofStatistic):
    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        return "KUI"

    @staticmethod
    @override
    def code():
        short_code = KuiperInverseGammaGofStatistic.short_code()
        return f"{short_code}_INVGAMMA_{AbstractInverseGammaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs):

        sorted_rvs = np.sort(np.asarray(rvs))
        n = len(sorted_rvs)

        if n == 0:
            raise ValueError(
                "At least one observation is required to compute the Kuiper statistic."
            )

        if np.any(sorted_rvs <= 0):
            raise ValueError(
                "All observations must be strictly positive for Inverse Gamma distribution."
            )

        cdf_vals = scipy_stats.invgamma.cdf(sorted_rvs, a=self.alpha, scale=1.0 / self.beta)

        i = np.arange(1, n + 1)
        d_plus = np.max(i / n - cdf_vals)
        d_minus = np.max(cdf_vals - (i - 1) / n)
        return float(d_plus + d_minus)


class MinToshiyukiInverseGammaGofStatistic(AbstractInverseGammaGofStatistic, MinToshiyukiStatistic):
    @staticmethod
    @override
    def short_code():
        return "MT"

    @staticmethod
    @override
    def code():
        short_code = MinToshiyukiInverseGammaGofStatistic.short_code()
        return f"{short_code}_INVGAMMA_{AbstractInverseGammaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        sorted_rvs = np.sort(np.asarray(rvs))
        n = len(sorted_rvs)

        if n == 0:
            raise ValueError(
                "At least one observation is required to compute the Min-Toshiyuki statistic."
            )

        if np.any(sorted_rvs <= 0):
            raise ValueError(
                "All observations must be strictly positive for Inverse Gamma distribution."
            )

        cdf_vals = scipy_stats.invgamma.cdf(sorted_rvs, a=self.alpha, scale=1.0 / self.beta)
        return MinToshiyukiStatistic.do_execute_statistic(self, cdf_vals)


class GreenwoodInverseGammaGofStatistic(AbstractInverseGammaGofStatistic):
    """
    Greenwood spacing statistic measuring uniformized Inverse Gamma gaps.

    Test based on sum of squared spacings between consecutive Inverse Gamma CDF values.
    Sensitive to clustering of observations.
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        return "GRW"

    @staticmethod
    @override
    def code():
        short_code = GreenwoodInverseGammaGofStatistic.short_code()
        return f"{short_code}_{AbstractInverseGammaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Greenwood test statistic for Inverse Gamma distribution.

        :param rvs: array of observations assumed to follow Inv-Gamma(alpha, beta).
        :return: Greenwood spacing statistic G = sum(D_i^2) where spacings D_i are computed
        from Inverse Gamma CDF values.
        :raises ValueError: if spacings are negative.
        :raises ValueError: if any observation is <= 0 (Inverse Gamma domain is (0, ∞)).
        """

        rvs_array = np.asarray(rvs)
        if np.any(rvs_array <= 0):
            raise ValueError("All observations must be positive for Inverse Gamma distribution.")
        sorted_rvs = np.sort(rvs_array)
        cdf_vals = scipy_stats.invgamma.cdf(sorted_rvs, a=self.alpha, scale=1 / self.beta)
        spacings = np.diff(np.concatenate(([0.0], cdf_vals, [1.0])))

        if np.any(spacings < 0):
            raise ValueError("Spacings must be non-negative; check input data ordering.")

        if np.any(spacings > 1):
            raise ValueError("Spacings must be <= 1; check CDF values.")

        return float(np.sum(spacings**2))


class ZhangAInverseGammaGofStatistic(AbstractInverseGammaGofStatistic):
    """
    Zhang ZA statistic for Inverse Gamma distribution.

    Test based on weighted sum of log-likelihood ratios.
    Analog of Anderson-Darling test. Sensitive to deviations in the tails.
    Rejects for small values of the statistic (left-tailed).
    """

    @override
    def alternative(self) -> Alternative:
        return LeftAlternative()

    @staticmethod
    @override
    def short_code():
        return "ZAA"

    @staticmethod
    @override
    def code():
        short_code = ZhangAInverseGammaGofStatistic.short_code()
        return f"{short_code}_{AbstractInverseGammaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        ZA = -sum_{i=1}^{n} [ ln(U_i) / (n - i + 0.5) + ln(1 - U_i) / (i - 0.5) ]
        where U_i = F(X_(i)) is the Inverse Gamma CDF value.
        """
        epsilon = kwargs.get("epsilon", 1e-10)

        rvs_array = np.asarray(rvs)
        if np.any(rvs_array <= 0):
            raise ValueError("All observations must be positive for Inverse Gamma distribution.")

        sorted_rvs = np.sort(rvs_array)
        n = len(sorted_rvs)

        cdf_vals = scipy_stats.invgamma.cdf(sorted_rvs, a=self.alpha, scale=1.0 / self.beta)
        cdf_vals = np.clip(cdf_vals, epsilon, 1.0 - epsilon)

        if np.any((cdf_vals < 0) | (cdf_vals > 1)):
            raise ValueError("CDF values must be in [0, 1]; check parameters.")
        za = 0.0
        for i, F_val in enumerate(cdf_vals, start=1):
            term = np.log(F_val) / (n - i + 0.5) + np.log(1.0 - F_val) / (i - 0.5)
            za -= term

        return float(za)


class ZhangCInverseGammaGofStatistic(AbstractInverseGammaGofStatistic):
    """
    Zhang ZC statistic for Inverse Gamma distribution.

    Test based on sum of squared log-likelihood ratios.
    Analog of Cramér-von Mises test. Sensitive to general deviations.
    Rejects for small values of the statistic (left-tailed).

    ZC = sum_{i=1}^{n} [ ln( (1/U_i - 1) / ((n - 0.5)/(i - 0.75) - 1) ) ]^2
    where U_i = F(X_(i)) is the Inverse Gamma CDF value.
    """

    @override
    def alternative(self) -> Alternative:
        return LeftAlternative()

    @staticmethod
    @override
    def short_code():
        return "ZAC"

    @staticmethod
    @override
    def code():
        short_code = ZhangCInverseGammaGofStatistic.short_code()
        return f"{short_code}_{AbstractInverseGammaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        epsilon = kwargs.get("epsilon", 1e-10)

        rvs_array = np.asarray(rvs)
        if np.any(rvs_array <= 0):
            raise ValueError("All observations must be positive for Inverse Gamma distribution.")

        sorted_rvs = np.sort(rvs_array)
        n = len(sorted_rvs)

        cdf_vals = scipy_stats.invgamma.cdf(sorted_rvs, a=self.alpha, scale=1.0 / self.beta)
        cdf_vals = np.clip(cdf_vals, epsilon, 1.0 - epsilon)

        if np.any((cdf_vals < 0) | (cdf_vals > 1)):
            raise ValueError("CDF values must be in [0, 1]; check parameters.")

        zc = 0.0
        for i, F_val in enumerate(cdf_vals, start=1):
            odds_theor = (1.0 - F_val) / F_val
            odds_emp = (n - 0.5) / (i - 0.75) - 1.0
            if odds_emp <= 0:
                odds_emp = epsilon

            log_ratio = np.log(odds_theor / odds_emp)
            zc += log_ratio**2

        return float(zc)


class ZhangKInverseGammaGofStatistic(AbstractInverseGammaGofStatistic):
    """
    Zhang ZK statistic for Inverse Gamma distribution.

    Test based on maximum of log-likelihood ratios.
    Analog of Kolmogorov-Smirnov test. Sensitive to the maximum deviation.
    Rejects for large values of the statistic (right-tailed).

    ZK = max_{i} [ (i - 0.5) * ln((i - 0.5) / (n * U_i)) +
                   (n - i + 0.5) * ln((n - i + 0.5) / (n * (1 - U_i))) ]
    where U_i = F(X_(i)) is the Inverse Gamma CDF value.
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        return "ZAK"

    @staticmethod
    @override
    def code():
        short_code = ZhangKInverseGammaGofStatistic.short_code()
        return f"{short_code}_{AbstractInverseGammaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        epsilon = kwargs.get("epsilon", 1e-10)

        rvs_array = np.asarray(rvs)
        if np.any(rvs_array <= 0):
            raise ValueError("All observations must be positive for Inverse Gamma distribution.")

        sorted_rvs = np.sort(rvs_array)
        n = len(sorted_rvs)

        cdf_vals = scipy_stats.invgamma.cdf(sorted_rvs, a=self.alpha, scale=1.0 / self.beta)
        cdf_vals = np.clip(cdf_vals, epsilon, 1.0 - epsilon)

        if np.any((cdf_vals < 0) | (cdf_vals > 1)):
            raise ValueError("CDF values must be in [0, 1]; check parameters.")
        zk = 0.0
        for i, F_val in enumerate(cdf_vals, start=1):
            term1 = (i - 0.5) * np.log((i - 0.5) / (n * F_val))
            term2 = (n - i + 0.5) * np.log((n - i + 0.5) / (n * (1.0 - F_val)))
            term = term1 + term2

            if term > zk:
                zk = term

        return float(zk)
