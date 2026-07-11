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
)
from pysatl_criterion.statistics.hypothesis import GoodnessOfFitHypothesis


class AbstractLaplaceGofStatistic(AbstractGoodnessOfFitStatistic, ABC):
    def __init__(self, t: float = 0.0, s: float = 1.0):
        if s <= 0:
            raise ValueError("Scale must be positive.")
        self.t = t
        self.s = s

    @override
    def hypothesis(self) -> GoodnessOfFitHypothesis:
        return GoodnessOfFitHypothesis({"t": self.t, "s": self.s})

    @staticmethod
    @override
    def distribution() -> DistributionType:
        return DistributionType.LAPLACE

    @staticmethod
    @override
    def code():
        return f"LAPLACE_{AbstractGoodnessOfFitStatistic.code()}"


class KolmogorovSmirnovLaplaceGofStatistic(AbstractLaplaceGofStatistic, KSStatistic):
    def __init__(
        self,
        alternative_type: AlternativeType = AlternativeType.TWO_TAILED,
        mode="auto",
        t: float = 0.0,
        s: float = 1.0,
    ):
        AbstractLaplaceGofStatistic.__init__(self, t=t, s=s)
        KSStatistic.__init__(self, alternative_type=alternative_type, mode=mode)

    @staticmethod
    @override
    def short_code():
        return "KS"

    @staticmethod
    @override
    def code():
        short_code = KolmogorovSmirnovLaplaceGofStatistic.short_code()
        return f"{short_code}_{AbstractLaplaceGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        sorted_rvs = np.sort(np.asarray(rvs))
        cdf_vals = scipy_stats.laplace.cdf(sorted_rvs, loc=self.t, scale=self.s)
        return KSStatistic.do_execute_statistic(self, sorted_rvs, cdf_vals)


class CramerVonMisesLaplaceGofStatistic(AbstractLaplaceGofStatistic, CrammerVonMisesStatistic):
    @staticmethod
    @override
    def short_code():
        return "CVM"

    @staticmethod
    @override
    def code():
        short_code = CramerVonMisesLaplaceGofStatistic.short_code()
        return f"{short_code}_{AbstractLaplaceGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        sorted_rvs = np.sort(np.asarray(rvs))
        cdf_vals = scipy_stats.laplace.cdf(sorted_rvs, loc=self.t, scale=self.s)
        return CrammerVonMisesStatistic.do_execute_statistic(self, sorted_rvs, cdf_vals)


class AndersonDarlingLaplaceGofStatistic(AbstractLaplaceGofStatistic, ADStatistic):
    @staticmethod
    @override
    def short_code():
        return "AD"

    @staticmethod
    @override
    def code():
        short_code = AndersonDarlingLaplaceGofStatistic.short_code()
        return f"{short_code}_{AbstractLaplaceGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        sorted_rvs = np.sort(np.asarray(rvs))
        log_cdf = scipy_stats.laplace.logcdf(sorted_rvs, loc=self.t, scale=self.s)
        log_sf = scipy_stats.laplace.logsf(sorted_rvs, loc=self.t, scale=self.s)
        return ADStatistic.do_execute_statistic(self, sorted_rvs, log_cdf=log_cdf, log_sf=log_sf)
