from __future__ import annotations

from abc import ABC

import numpy as np
import scipy.stats as scipy_stats
from typing_extensions import override

from pysatl_criterion import DistributionType
from pysatl_criterion.statistics import AbstractGoodnessOfFitStatistic
from pysatl_criterion.statistics.alternative import AlternativeType
from pysatl_criterion.statistics.goodness_of_fit.common import (
    ADStatistic,
    CrammerVonMisesStatistic,
    KSStatistic,
    LillieforsTest,
    MinToshiyukiStatistic,
)
from pysatl_criterion.statistics.hypothesis import GoodnessOfFitHypothesis


class AbstractParetoGofStatistic(AbstractGoodnessOfFitStatistic, ABC):
    def __init__(self, shape: float = 1.0, scale: float = 1.0):
        """
        Initialize Pareto distribution goodness-of-fit statistic.

        :param shape: shape parameter (shape) > 0.
        :param scale: scale parameter (scale) > 0.
        :raises ValueError: if shape or scale is not positive.
        """
        if shape <= 0:
            raise ValueError("Shape must be positive.")
        if scale <= 0:
            raise ValueError("Scale must be positive.")
        self.shape = shape
        self.scale = scale

    @override
    def hypothesis(self) -> GoodnessOfFitHypothesis:
        return GoodnessOfFitHypothesis({"shape": self.shape, "scale": self.scale})

    @staticmethod
    @override
    def distribution() -> DistributionType:
        return DistributionType.PARETO

    @staticmethod
    @override
    def code():
        return f"PARETO_{AbstractGoodnessOfFitStatistic.code()}"


class KolmogorovSmirnovParetoGofStatistic(AbstractParetoGofStatistic, KSStatistic):

    def __init__(
        self,
        alternative_type: AlternativeType = AlternativeType.TWO_TAILED,
        mode="auto",
        shape: float = 1.0,
        scale: float = 1.0,
    ):
        AbstractParetoGofStatistic.__init__(self, shape=shape, scale=scale)
        KSStatistic.__init__(self, alternative_type=alternative_type, mode=mode)

    @staticmethod
    @override
    def short_code():
        return "KS"

    @staticmethod
    @override
    def code():
        short_code = KolmogorovSmirnovParetoGofStatistic.short_code()
        return f"{short_code}_{AbstractParetoGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        sorted_rvs = np.sort(np.asarray(rvs, dtype=float))
        cdf_vals = scipy_stats.pareto.cdf(sorted_rvs, b=self.shape, scale=self.scale)
        return KSStatistic.do_execute_statistic(self, sorted_rvs, cdf_vals)


class AndersonDarlingParetoGofStatistic(AbstractParetoGofStatistic, ADStatistic):
    @staticmethod
    @override
    def short_code():
        return "AD"

    @staticmethod
    @override
    def code():
        short_code = AndersonDarlingParetoGofStatistic.short_code()
        return f"{short_code}_{AbstractParetoGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        sorted_rvs = np.sort(np.asarray(rvs, dtype=float))
        log_cdf = scipy_stats.pareto.logcdf(sorted_rvs, b=self.shape, scale=self.scale)
        log_sf = scipy_stats.pareto.logsf(sorted_rvs, b=self.shape, scale=self.scale)
        return super().do_execute_statistic(sorted_rvs, log_cdf=log_cdf, log_sf=log_sf)


class CramerVonMisesParetoGofStatistic(AbstractParetoGofStatistic, CrammerVonMisesStatistic):

    @staticmethod
    @override
    def short_code():
        return "CVM"

    @staticmethod
    @override
    def code():
        short_code = CramerVonMisesParetoGofStatistic.short_code()
        return f"{short_code}_{AbstractParetoGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        sorted_rvs = np.sort(np.asarray(rvs, dtype=float))
        cdf_vals = scipy_stats.pareto.cdf(sorted_rvs, b=self.shape, scale=self.scale)
        return CrammerVonMisesStatistic.do_execute_statistic(self, sorted_rvs, cdf_vals)

class LillieforsParetoGofStatistic(AbstractParetoGofStatistic, LillieforsTest):

    def __init__(self, shape: float = 1.0, scale: float = 1.0):
        AbstractParetoGofStatistic.__init__(self, shape=shape, scale=scale)

    @staticmethod
    @override
    def short_code():
        return "Lilliefors"

    @staticmethod
    @override
    def code():
        short_code = LillieforsParetoGofStatistic.short_code()
        return f"{short_code}_{AbstractParetoGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        sample = np.asarray(rvs, dtype=float)
        n = sample.size
        if n == 0:
            raise ValueError("At least one observation is required for the Lilliefors statistic.")
        
        if np.any(sample <= 0):
            raise ValueError("Sample values must be strictly positive for Pareto distribution.")
        scale_hat = float(np.min(sample))
        shape_hat = float(n / np.sum(np.log(sample / scale_hat)))

        self.shape = shape_hat
        self.scale = scale_hat

        sorted_sample = np.sort(sample)
        cdf_vals = scipy_stats.pareto.cdf(sorted_sample, b=shape_hat, scale=scale_hat)
        
        return super().do_execute_statistic(sorted_sample, cdf_vals)


class MinToshiyukiParetoGofStatistic(AbstractParetoGofStatistic, MinToshiyukiStatistic):
    @staticmethod
    @override
    def short_code():
        return "MT"

    @staticmethod
    @override
    def code():
        short_code = MinToshiyukiParetoGofStatistic.short_code()
        return f"{short_code}_{AbstractParetoGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        sorted_rvs = np.sort(np.asarray(rvs, dtype=float))
        cdf_vals = scipy_stats.pareto.cdf(sorted_rvs, b=self.shape, scale=self.scale)
        
        cdf_vals = np.clip(cdf_vals, 1e-15, 1.0 - 1e-15)
        
        return super().do_execute_statistic(cdf_vals)
    
class ObradovicParetoGofStatistic(AbstractParetoGofStatistic):
    """
    Obradovic-Jovanovic-Milosevic U-statistic goodness-of-fit test for Pareto distribution.
    
    This statistic is based on a new characterization: X and max(X/Y, Y/X) are identically 
    distributed if and only if X follows a Pareto distribution.
    
    References:
        - Obradovic M., Jovanovic M., Milosevic B. (2014). Goodness of Fit Tests 
          for Pareto Distribution Based on a Characterization and their Asymptotics.
          arXiv:1310.5510v2
    """

    @staticmethod
    @override
    def short_code():
        return "OJM_Integral"

    @staticmethod
    @override
    def code():
        short_code = ObradovicParetoGofStatistic.short_code()
        return f"{short_code}_{AbstractParetoGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Obradovic et al. integral test statistic (Tn) for Pareto distribution.
        
        :param rvs: array of observations.
        :return: Tn test statistic value.
        """
        x = np.asarray(rvs, dtype=float)
        n = x.size
        if n < 2:
            raise ValueError("At least 2 observations are required for this statistic.")
            
        if np.any(x <= 0):
            raise ValueError("Sample values must be strictly positive.")

        N = n * (n - 1) // 2
        
        idx_j, idx_k = np.triu_indices(n, k=1)
        pairs_max = np.maximum(x[idx_j] / x[idx_k], x[idx_k] / x[idx_j])
        
        pooled = np.concatenate([x, pairs_max])
        
        from scipy.stats import rankdata
        all_ranks = rankdata(pooled, method='average')
        
        r_j = all_ranks[:n]

        numerator = 2 * np.sum(r_j) - (n + 1) * (n + N)
        denominator = 2 * n * N
        
        return float(numerator / denominator)
    
class GreenwoodParetoGofStatistic(AbstractParetoGofStatistic):
    """
    Greenwood's goodness-of-fit statistic for the Pareto distribution.

    This statistic tests the Pareto assumption by transforming the sample
    via Y_i = ln(X_i / min(X)) and evaluating Greenwood's statistic on the 
    resulting exponentially distributed spacing candidates.

    References:
        - Greenwood, M. (1946). The statistical study of infectious diseases. 
          Journal of the Royal Statistical Society, 109(2), 85-110.
    """

    @staticmethod
    @override
    def short_code():
        return "Greenwood"

    @staticmethod
    @override
    def code():
        short_code = GreenwoodParetoGofStatistic.short_code()
        return f"{short_code}_{AbstractParetoGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute Greenwood's test statistic for Pareto distribution.

        :param rvs: array of observations assumed to follow Pareto.
        :return: Greenwood's G statistic value.
        :raises ValueError: if sample size is less than 2 or values are non-positive.
        """
        sample = np.asarray(rvs, dtype=float)
        n = sample.size
        if n < 2:
            raise ValueError("At least 2 observations are required for Greenwood statistic.")

        scale_hat = np.min(sample)
        if scale_hat <= 0:
            raise ValueError("Sample values must be strictly positive for Pareto distribution.")

        y = np.log(sample / scale_hat)

        sum_y = np.sum(y)
        if sum_y == 0:
            return 1.0

        g = np.sum(y**2) / (sum_y**2)

        return float(g)
    
class LequesneKlParetoGofStatistic(AbstractParetoGofStatistic):
    """
    Lequesne's entropy-based goodness-of-fit test for Pareto distribution
    using Kullback-Leibler divergence and Vasicek's entropy estimator.
    
    References:
        - Lequesne, J. (2013). Entropy-based goodness-of-fit test: Application 
          to the Pareto distribution. AIP Conference Proceedings, 1553, 155-162.
    """

    @staticmethod
    @override
    def short_code():
        return "Lequesne_KL"

    @staticmethod
    @override
    def code():
        return f"{LequesneKlParetoGofStatistic.short_code()}_{AbstractParetoGofStatistic.code()}"

    @override
    @override
    def execute_statistic(self, rvs, **kwargs):
        x = np.asarray(rvs, dtype=float)
        n = x.size
        if n < 4:
            raise ValueError("Lequesne's KL statistic requires n >= 4 for reliable window spacing.")

        sigma_hat = np.min(x)
        if sigma_hat <= 0:
            raise ValueError("Sample values must be strictly positive.")
        beta_hat = n / np.sum(np.log(x / sigma_hat))


        m = kwargs.get('m', int(np.floor(np.sqrt(n))))
        if m < 1 or m >= n // 2:
            m = max(1, n // 4)

        x_sorted = np.sort(x)

        def get_order_stat(idx):
            if idx < 1:
                return x_sorted[0]
            elif idx > n:
                return x_sorted[-1]
            return x_sorted[idx - 1]

        vasicek_sum = 0.0
        for i in range(1, n + 1):
            diff = get_order_stat(i + m) - get_order_stat(i - m)
            if diff <= 0:
                diff = 1e-10 
            vasicek_sum += np.log((n / (2 * m)) * diff)
        v_mn = vasicek_sum / n

        mean_log_scaled = np.mean(np.log(x / sigma_hat))

        kl_div = - v_mn - np.log(beta_hat) + np.log(sigma_hat) + (beta_hat + 1) * mean_log_scaled

        return float(np.abs(kl_div))