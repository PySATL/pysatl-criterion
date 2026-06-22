from abc import ABC, abstractmethod

from pysatl_criterion.statistics.alternative import AlternativeType


class PValueResolver(ABC):
    """
    P-value resolver.
    """

    @abstractmethod
    def resolve(
        self,
        criterion_code: str,
        sample_size: int,
        statistics_value: float,
        alternative: AlternativeType = AlternativeType.RIGHT,
    ) -> float | None:
        """
        Resolve p-value.

        :param criterion_code: criterion code
        :param sample_size: sample size
        :param statistics_value: statistic value
        :param alternative: alternative
        """
        pass
