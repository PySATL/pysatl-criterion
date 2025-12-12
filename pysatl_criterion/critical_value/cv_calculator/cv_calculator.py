import numpy as np
import scipy.stats as scipy_stats

from pysatl_criterion.persistence.model.limit_distribution.limit_distribution import (
    CriticalValueQuery,
    ILimitDistributionStorage,
)
from pysatl_criterion.statistics.models import HypothesisType


class CVCalculator:
    """
    Critical value calculator.

    :param limit_distribution_storage: limit distribution storage
    """

    def __init__(self, limit_distribution_storage: ILimitDistributionStorage):
        self.limit_distribution_storage = limit_distribution_storage

    def calculate_critical_value(
        self,
        criterion_code: str,
        sample_size: int,
        sl: float,
        alternative: HypothesisType = HypothesisType.RIGHT,
    ) -> float | tuple[float, float]:
        """
        Calculate critical value for given criterion.

        :param criterion_code: criterion code.
        :param sample_size: sample size.
        :param sl: significance level.
        :param alternative: test alternative

        :return: critical value.
        """

        query = CriticalValueQuery(criterion_code=criterion_code, sample_size=sample_size)

        limit_distribution_from_db = self.limit_distribution_storage.get_data_for_cv(query)
        if limit_distribution_from_db is None:
            raise ValueError(
                "Limit distribution for given criterion and sample size does not exist."
            )

        statistics_values = limit_distribution_from_db.results_statistics

        ecdf = scipy_stats.ecdf(statistics_values)

        if alternative == HypothesisType.RIGHT:
            return float(np.quantile(ecdf.cdf.quantiles, q=1 - sl))
        elif alternative == HypothesisType.LEFT:
            return float(np.quantile(ecdf.cdf.quantiles, q=sl))
        elif alternative == HypothesisType.TWO_TAILED:
            left = float(np.quantile(ecdf.cdf.quantiles, q=sl / 2))
            right = float(np.quantile(ecdf.cdf.quantiles, q=1 - sl / 2))
            return left, right
        else:
            raise ValueError("Unknown alternative.")
