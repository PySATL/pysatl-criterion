import scipy.stats as scipy_stats
from typing_extensions import override

from pysatl_criterion.p_value.resolver.model import PValueResolver
from pysatl_criterion.persistence.model.limit_distribution.limit_distribution import (
    CriticalValueQuery,
    ILimitDistributionStorage,
)
from pysatl_criterion.statistics.models import HypothesisType


class CalculationPValueResolver(PValueResolver):
    """
    P-value calculator.

    :param limit_distribution_storage: limit distribution storage
    """

    def __init__(self, limit_distribution_storage: ILimitDistributionStorage):
        self.limit_distribution_storage = limit_distribution_storage

    @override
    def resolve(
        self,
        criterion_code: str,
        sample_size: int,
        statistics_value: float,
        alternative: HypothesisType = HypothesisType.RIGHT,
    ) -> float:
        """
        Calculate p-value.

        :param criterion_code: criterion code
        :param sample_size: sample size
        :param statistics_value: statistics value
        :param alternative: test alternative

        :return: p-value
        """

        query = CriticalValueQuery(criterion_code=criterion_code, sample_size=sample_size)
        limit_distribution_from_db = self.limit_distribution_storage.get_data_for_cv(query)

        if limit_distribution_from_db is None:
            raise ValueError(
                f"Limit distribution for criterion {criterion_code} "
                f"and sample size {sample_size} does not exist."
            )

        simulation_results = limit_distribution_from_db.results_statistics

        ecdf = scipy_stats.ecdf(simulation_results)

        cdf_value = float(ecdf.cdf.evaluate(statistics_value))

        if alternative == HypothesisType.RIGHT:
            return 1.0 - cdf_value
        elif alternative == HypothesisType.TWO_TAILED:
            return 2.0 * min(cdf_value, 1.0 - cdf_value)
        elif alternative == HypothesisType.LEFT:
            return cdf_value
        else:
            raise ValueError(f"Unknown alternative {alternative}")
