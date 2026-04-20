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
    def resolve_bulk(
        self,
        criterion_codes: list[str],
        sample_size: int,
        sl: float,
        alternative: HypothesisType = HypothesisType.RIGHT,
    ) -> dict[str, CriticalArea]:
        """
        Bulk resolver critical values for given criteria from storage.

        :param criterion_codes: list of criterion codes
        :param sample_size: sample size
        :param sl: significance level
        :param alternative: test alternative

        :return: dict of criterion code to critical value for criteria with existing critical values
        """

        limit_distributions = self.limit_distribution_storage.get_bulk_data(
            criterion_codes, sample_size
        )

        results = {}
        for d in limit_distributions:
            ecdf = scipy_stats.ecdf(d.results_statistics)
            area = self._calculate_area(ecdf, sl, alternative)
            if area:
                results[d.criterion_code] = area
        return results

    def _calculate_area(self, ecdf, sl, alternative) -> CriticalArea:
        if alternative == HypothesisType.RIGHT:
            return RightCriticalArea(float(np.quantile(ecdf.cdf.quantiles, q=1 - sl)))
        elif alternative == HypothesisType.LEFT:
            return LeftCriticalArea(float(np.quantile(ecdf.cdf.quantiles, q=sl)))
        elif alternative == HypothesisType.TWO_TAILED:
            left = float(np.quantile(ecdf.cdf.quantiles, q=sl / 2))
            right = float(np.quantile(ecdf.cdf.quantiles, q=1 - sl / 2))
            return TwoSidedCriticalArea(left, right)
        raise ValueError(f"Unknown alternative: {alternative}.")
