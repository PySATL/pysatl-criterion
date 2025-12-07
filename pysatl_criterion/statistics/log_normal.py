import inspect
import sys
import numpy as np
import scipy.stats as scipy_stats
from numpy import histogram
from typing_extensions import override

from pysatl_criterion.statistics import normal
from pysatl_criterion.statistics.common import (
    CrammerVonMisesStatistic,
    KSStatistic,
)
from pysatl_criterion.statistics.goodness_of_fit import AbstractGoodnessOfFitStatistic
from pysatl_criterion.statistics.normal import AbstractNormalityGofStatistic


class AbstractLogNormalGofStatistic(AbstractGoodnessOfFitStatistic):
    """
    Base class for Log-Normal distribution Goodness-of-Fit statistics.
    """

    def __init__(self, s=1, scale=1):
        """
        :param s: shape parameter (sigma)
        :param scale: scale parameter (exp(mu))
        """
        if s <= 0:
            raise ValueError("Shape parameter s must be positive")
        if scale <= 0:
            raise ValueError("Scale parameter must be positive")

        self.s = s
        self.scale = scale

    @staticmethod
    @override
    def code():
        return f"LOGNORMAL_{AbstractGoodnessOfFitStatistic.code()}"


# =================================================================================================
# EXPLICIT IMPLEMENTATIONS
# =================================================================================================


class KolmogorovSmirnovLogNormalGofStatistic(AbstractLogNormalGofStatistic, KSStatistic):
    @override
    def __init__(self, alternative="two-sided", mode="auto", s=1, scale=1):
        AbstractLogNormalGofStatistic.__init__(self, s=s, scale=scale)
        KSStatistic.__init__(self, alternative, mode)

    @staticmethod
    @override
    def code():
        return f"KS_{AbstractLogNormalGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        rvs = np.sort(rvs)
        cdf_vals = scipy_stats.lognorm.cdf(rvs, s=self.s, scale=self.scale)
        return KSStatistic.execute_statistic(self, rvs, cdf_vals)


class CramerVonMiseLogNormalGofStatistic(AbstractLogNormalGofStatistic, CrammerVonMisesStatistic):
    @staticmethod
    @override
    def code():
        return f"CVM_{AbstractLogNormalGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        rvs_sorted = np.sort(rvs)
        cdf_vals = scipy_stats.lognorm.cdf(rvs_sorted, s=self.s, scale=self.scale)
        return super().execute_statistic(rvs, cdf_vals)


class QuesenberryMillerLogNormalGofStatistic(AbstractLogNormalGofStatistic):
    """
    Quesenberry and Miller's Q-test for Log-Normal distribution.

    References
    ----------
    .. [1] Quesenberry, C. P., & Miller Jr, F. L. (1977).
           Power studies of some tests for uniformity.
           Journal of Statistical Computation and Simulation, 5, 169-191.
    """

    @staticmethod
    @override
    def code():
        return f"QUESENBERRY_MILLER_{AbstractLogNormalGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
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


# =================================================================================================
# DYNAMIC GENERATION
# =================================================================================================


def _create_lognormal_class(normal_cls):
    new_class_name = normal_cls.__name__.replace("Normality", "LogNormal")
    class_code = normal_cls.code().replace("NORMALITY", "LOGNORMAL")

    class LogNormalClass(AbstractLogNormalGofStatistic):
        @override
        def execute_statistic(self, rvs, **kwargs):
            rvs_arr = np.array(rvs)

            if np.any(rvs_arr <= 0):
                return float("inf")

            log_rvs = np.log(rvs_arr)
            standardized_log_rvs = (log_rvs - np.log(self.scale)) / self.s

            return normal_cls().execute_statistic(standardized_log_rvs, **kwargs)

        @staticmethod
        @override
        def code():
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
    "BHSNormalityGofStatistic",
]

current_module = sys.modules[__name__]
__all__ = [
    "AbstractLogNormalGofStatistic",
    "KolmogorovSmirnovLogNormalGofStatistic",
    "CramerVonMiseNormalityGofStatistic",
    "QuesenberryMillerLogNormalGofStatistic",
]

for name, obj in inspect.getmembers(normal):
    if (
        inspect.isclass(obj)
        and issubclass(obj, AbstractNormalityGofStatistic)
        and obj is not AbstractNormalityGofStatistic
        and name not in EXPLICITLY_IMPLEMENTED_NORMAL_STATS
        and not name.startswith("Abstract")
        and not name.startswith("Graph")  # maybe shoud fix
    ):
        ln_class = _create_lognormal_class(obj)
        setattr(current_module, ln_class.__name__, ln_class)
        __all__.append(ln_class.__name__)
