import numpy as np
import scipy.stats as scipy_stats
from typing_extensions import override

from pysatl_criterion.critical_value.critical_area.critical_areas import (
    LeftCriticalArea,
    RightCriticalArea,
    TwoSidedCriticalArea,
)
from pysatl_criterion.critical_value.critical_area.model import CriticalArea
from pysatl_criterion.critical_value.resolver.model import CriticalValueResolver
from pysatl_criterion.persistence.model.limit_distribution.limit_distribution import (
    CriticalValueQuery,
    ILimitDistributionStorage,
)
from pysatl_criterion.statistics.models import HypothesisType


class StorageCriticalValueResolver(CriticalValueResolver):
    """
    Critical value resolver.

    :param limit_distribution_storage: limit distribution storage
    """

    def __init__(self, limit_distribution_storage: ILimitDistributionStorage):
        self.limit_distribution_storage = limit_distribution_storage

    @override
    def resolve(
        self,
        criterion_code: str,
        sample_size: int,
        sl: float,
        alternative: HypothesisType = HypothesisType.RIGHT,
    ) -> CriticalArea | None:
        """
        Resolver critical value for given criterion from storage.

        :param criterion_code: criterion code
        :param sample_size: sample size
        :param sl: significance level
        :param alternative: test alternative

        :return: critical value if critical value exists, None otherwise
        """

        query = CriticalValueQuery(criterion_code=criterion_code, sample_size=sample_size)
        limit_distribution = self.limit_distribution_storage.get_data_for_cv(query)

        if limit_distribution is None:
            return None

        ecdf = scipy_stats.ecdf(limit_distribution.results_statistics)

        if alternative == HypothesisType.RIGHT:
            return RightCriticalArea(float(np.quantile(ecdf.cdf.quantiles, q=1 - sl)))
        elif alternative == HypothesisType.LEFT:
            return LeftCriticalArea(float(np.quantile(ecdf.cdf.quantiles, q=sl)))
        elif alternative == HypothesisType.TWO_TAILED:
            left = float(np.quantile(ecdf.cdf.quantiles, q=sl / 2))
            right = float(np.quantile(ecdf.cdf.quantiles, q=1 - sl / 2))
            return TwoSidedCriticalArea(left, right)
        else:
            raise ValueError(f"Unknown alternative: {alternative}.")
