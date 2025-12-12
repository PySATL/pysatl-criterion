from typing_extensions import Protocol

from pysatl_criterion.statistics.models import HypothesisType


class PValueResolver(Protocol):
    """
    P-value resolver.
    """

    def resolve(
        self,
        criterion_code: str,
        sample_size: int,
        statistics_value: float,
        alternative: HypothesisType = HypothesisType.RIGHT,
    ) -> float | None:
        """
        Resolve p-value.

        :param criterion_code: criterion code
        :param sample_size: sample size
        :param statistics_value: statistic value
        :param alternative: alternative
        """
        pass
