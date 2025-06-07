import numpy as np
import scipy.stats as scipy_stats

from pysatl_criterion.persistence.model.limit_distribution.limit_distribution import (
    CriticalValueQuery,
    ILimitDistributionStorage,
)


class CVCalculator:
    """
    Critical value calculator.

    :param limit_distribution_storage: limit distribution storage
    """

    def __init__(self, limit_distribution_storage: ILimitDistributionStorage):
        self.limit_distribution_storage = limit_distribution_storage

    def calculate_critical_value(self, criterion_code: str, sample_size: int, sl: float) -> float:
        """
        Calculate critical value for given criterion.

        :param criterion_code: criterion code.
        :param sample_size: sample size.
        :param sl: significance level.

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

        critical_value = float(np.quantile(ecdf.cdf.quantiles, q=1 - sl))

        return critical_value

    def calculate_two_tailed_critical_values(
        self, criterion_code: str, sample_size: int, sl: float
    ) -> tuple[float, float]:
        """
        Calculate critical values for two-tailed criterion.

        :param criterion_code: criterion code.
        :param sample_size: sample size.
        :param sl: significance level.

        :return: critical values.
        """

        query = CriticalValueQuery(criterion_code=criterion_code, sample_size=sample_size)

        limit_distribution_from_db = self.limit_distribution_storage.get_data_for_cv(query)
        if limit_distribution_from_db is None:
            raise ValueError(
                "Limit distribution for given criterion and sample size does not exist."
            )

        statistics_values = limit_distribution_from_db.results_statistics

        ecdf = scipy_stats.ecdf(statistics_values)

        critical_value_left = float(np.quantile(ecdf.cdf.quantiles, q=sl / 2))
        critical_value_right = float(np.quantile(ecdf.cdf.quantiles, q=1 - sl / 2))

        return critical_value_left, critical_value_right
