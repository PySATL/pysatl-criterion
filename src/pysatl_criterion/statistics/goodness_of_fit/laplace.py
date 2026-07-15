from __future__ import annotations

from abc import ABC

import numpy as np
import scipy.stats as scipy_stats
from typing_extensions import override

from pysatl_criterion import DistributionType
from pysatl_criterion.statistics import AbstractGoodnessOfFitStatistic
from pysatl_criterion.statistics.alternative import Alternative, AlternativeType, RightAlternative
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


class KuiperLaplaceGofStatistic(AbstractLaplaceGofStatistic):
    """
    Kuiper EDF statistic for the Laplace distribution.

    Sums the maximum positive and negative deviations between the empirical
    distribution function and the theoretical Laplace distribution function.
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """
        Get the short code identifier for this test.

        :return: short code string ``KUI``.
        """
        return "KUI"

    @staticmethod
    @override
    def code():
        """
        Get the unique code identifier for this test.

        :return: string code in format ``KUI_LAPLACE_{parent_code}``.
        """
        short_code = KuiperLaplaceGofStatistic.short_code()
        return f"{short_code}_{AbstractLaplaceGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs):
        """
        Execute the Kuiper statistic for a Laplace distribution.
        :param rvs: observations assumed to follow a Laplace distribution.
        :return: Kuiper statistic ``V = D+ + D-`` computed with the Laplace CDF.
        :raises ValueError: if the sample is empty.
        """

        sorted_rvs = np.sort(np.asarray(rvs))
        n = len(sorted_rvs)

        if n == 0:
            raise ValueError(
                "At least one observation is required to compute the Kuiper statistic."
            )

        cdf_vals = scipy_stats.laplace.cdf(sorted_rvs, loc=self.t, scale=self.s)

        i = np.arange(1, n + 1)
        d_plus = np.max(i / n - cdf_vals)
        d_minus = np.max(cdf_vals - (i - 1) / n)

        return float(d_plus + d_minus)


class WatsonLaplaceGofStatistic(AbstractLaplaceGofStatistic):
    """Watson EDF statistic for the Laplace distribution.

    Computes a centered modification of the Cramer--von Mises statistic
    by removing the squared mean deviation of the transformed observations.
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "WAT".
        """
        return "WAT"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "WAT_LAPLACE_{parent_code}".
        """
        short_code = WatsonLaplaceGofStatistic.short_code()
        return f"{short_code}_{AbstractLaplaceGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs):
        """
        Execute the Watson test statistic for Laplace distribution.

        :param rvs: array of observations assumed to follow Laplace(location, scale).
        :return: Watson U^2 statistic derived from the Laplace CDF values.
        :raises ValueError: if sample is empty.
        """

        sorted_rvs = np.sort(np.asarray(rvs))
        n = len(sorted_rvs)
        if n == 0:
            raise ValueError(
                "At least one observation is required to compute the Watson statistic."
            )

        cdf_vals = scipy_stats.laplace.cdf(sorted_rvs, loc=self.t, scale=self.s)
        u = (2 * np.arange(1, n + 1) - 1) / (2 * n)
        diff = cdf_vals - u
        w_squared = 1.0 / (12 * n) + np.sum(diff**2)
        mean_adj = np.sum(cdf_vals) - n / 2
        return float(w_squared - (mean_adj**2) / n)


class GreenwoodLaplaceGofStatistic(AbstractLaplaceGofStatistic):
    """Greenwood spacing statistic for the Laplace distribution.

    Computes the sum of squared spacings between consecutive values of the
    Laplace cumulative distribution function. Large values indicate clustering
    or uneven spacing in the probability-transformed sample.
    """

    @override
    def alternative(self) -> Alternative:
        """Return the right-tailed alternative for the Greenwood statistic.

        :return: right-tailed alternative.
        """
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """Get the short code identifier for this test.

        :return: short code string ``GRW``.
        """
        return "GRW"

    @staticmethod
    @override
    def code():
        """Get the unique code identifier for this test.

        :return: string code in format ``GRW_LAPLACE_{parent_code}``.
        """
        short_code = GreenwoodLaplaceGofStatistic.short_code()
        return f"{short_code}_{AbstractLaplaceGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Execute the Greenwood spacing statistic for a Laplace distribution.

        :param rvs: observations assumed to follow a Laplace distribution.
        :return: Greenwood statistic computed from the spacings of the Laplace CDF.
        :raises ValueError: if the sample is empty.
        """
        sorted_rvs = np.sort(np.asarray(rvs))
        n = len(sorted_rvs)
        if n == 0:
            raise ValueError(
                "At least one observation is required to compute the Greenwood statistic."
            )

        cdf_vals = scipy_stats.laplace.cdf(sorted_rvs, loc=self.t, scale=self.s)
        spacings = np.diff(np.concatenate(([0.0], cdf_vals, [1.0])))

        if np.any(spacings < 0):
            raise ValueError("Spacings must be non-negative; check input data ordering.")

        return float(np.sum(spacings**2))
