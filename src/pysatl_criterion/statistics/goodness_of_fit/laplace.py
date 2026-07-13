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
    """Abstract base class for Laplace distribution goodness-of-fit statistics."""

    def __init__(self, t: float = 0.0, s: float = 1.0):
        """Initialize a Laplace distribution goodness-of-fit statistic.

        :param t: location parameter.
        :param s: scale parameter greater than zero.
        :raises ValueError: if the scale parameter is not positive.
        """
        if s <= 0:
            raise ValueError("Scale must be positive.")
        self.t = t
        self.s = s

    @override
    def hypothesis(self) -> GoodnessOfFitHypothesis:
        """Get the goodness-of-fit hypothesis for the Laplace distribution.

        :return: hypothesis containing the location and scale parameters.
        """
        return GoodnessOfFitHypothesis({"t": self.t, "s": self.s})

    @staticmethod
    @override
    def distribution() -> DistributionType:
        """Get the distribution type.

        :return: Laplace distribution type.
        """
        return DistributionType.LAPLACE

    @staticmethod
    @override
    def code():
        """Get the code identifier for Laplace distribution statistics.

        :return: string code in format ``LAPLACE_{parent_code}``.
        """
        return f"LAPLACE_{AbstractGoodnessOfFitStatistic.code()}"


class KolmogorovSmirnovLaplaceGofStatistic(AbstractLaplaceGofStatistic, KSStatistic):
    """Kolmogorov--Smirnov EDF test computed with the Laplace reference CDF.

    Compares the empirical distribution function with the theoretical Laplace
    distribution function and measures their maximum deviation.
    """

    def __init__(
        self,
        alternative_type: AlternativeType = AlternativeType.TWO_TAILED,
        mode="auto",
        t: float = 0.0,
        s: float = 1.0,
    ):
        """Initialize a Kolmogorov--Smirnov test for a Laplace distribution.

        :param alternative_type: alternative hypothesis used by the KS test.
        :param mode: calculation mode used by the Kolmogorov--Smirnov statistic.
        :param t: location parameter of the reference Laplace distribution.
        :param s: positive scale parameter of the reference Laplace distribution.
        :raises ValueError: if the scale parameter is not positive.
        """
        AbstractLaplaceGofStatistic.__init__(self, t=t, s=s)
        KSStatistic.__init__(self, alternative_type=alternative_type, mode=mode)

    @staticmethod
    @override
    def short_code():
        """Get the short code identifier for this test.

        :return: short code string ``KS``.
        """
        return "KS"

    @staticmethod
    @override
    def code():
        """Get the unique code identifier for this test.

        :return: string code in format ``KS_LAPLACE_{parent_code}``.
        """
        short_code = KolmogorovSmirnovLaplaceGofStatistic.short_code()
        return f"{short_code}_{AbstractLaplaceGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Execute the Kolmogorov--Smirnov statistic for a Laplace distribution.

        :param rvs: observations assumed to follow a Laplace distribution.
        :return: Kolmogorov--Smirnov D statistic computed with the Laplace CDF.
        """
        sorted_rvs = np.sort(np.asarray(rvs))
        cdf_vals = scipy_stats.laplace.cdf(sorted_rvs, loc=self.t, scale=self.s)
        return KSStatistic.do_execute_statistic(self, sorted_rvs, cdf_vals)


class CramerVonMisesLaplaceGofStatistic(AbstractLaplaceGofStatistic, CrammerVonMisesStatistic):
    """Cramer--von Mises quadratic EDF test for Laplace samples.

    Measures the integrated squared difference between the empirical and
    theoretical Laplace cumulative distribution functions.
    """

    @staticmethod
    @override
    def short_code():
        """Get the short code identifier for this test.

        :return: short code string ``CVM``.
        """
        return "CVM"

    @staticmethod
    @override
    def code():
        """Get the unique code identifier for this test.

        :return: string code in format ``CVM_LAPLACE_{parent_code}``.
        """
        short_code = CramerVonMisesLaplaceGofStatistic.short_code()
        return f"{short_code}_{AbstractLaplaceGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Execute the Cramer--von Mises statistic for a Laplace distribution.

        :param rvs: observations assumed to follow a Laplace distribution.
        :return: Cramer--von Mises W^2 statistic computed with the Laplace CDF.
        """
        sorted_rvs = np.sort(np.asarray(rvs))
        cdf_vals = scipy_stats.laplace.cdf(sorted_rvs, loc=self.t, scale=self.s)
        return CrammerVonMisesStatistic.do_execute_statistic(self, sorted_rvs, cdf_vals)


class AndersonDarlingLaplaceGofStatistic(AbstractLaplaceGofStatistic, ADStatistic):
    """Anderson--Darling EDF statistic for the Laplace distribution.

    Weights deviations in the distribution tails more heavily than the
    Kolmogorov--Smirnov and Cramer--von Mises statistics.
    """

    @staticmethod
    @override
    def short_code():
        """Get the short code identifier for this test.

        :return: short code string ``AD``.
        """
        return "AD"

    @staticmethod
    @override
    def code():
        """Get the unique code identifier for this test.

        :return: string code in format ``AD_LAPLACE_{parent_code}``.
        """
        short_code = AndersonDarlingLaplaceGofStatistic.short_code()
        return f"{short_code}_{AbstractLaplaceGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Execute the Anderson--Darling statistic for a Laplace distribution.

        :param rvs: observations assumed to follow a Laplace distribution.
        :return: Anderson--Darling A^2 statistic computed from log-CDF values.
        """
        sorted_rvs = np.sort(np.asarray(rvs))
        log_cdf = scipy_stats.laplace.logcdf(sorted_rvs, loc=self.t, scale=self.s)
        log_sf = scipy_stats.laplace.logsf(sorted_rvs, loc=self.t, scale=self.s)
        return ADStatistic.do_execute_statistic(self, sorted_rvs, log_cdf=log_cdf, log_sf=log_sf)
