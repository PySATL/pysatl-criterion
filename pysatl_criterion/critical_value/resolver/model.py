from abc import ABC, abstractmethod

from pysatl_criterion.critical_value.critical_area.model import CriticalArea
from pysatl_criterion.statistics.models import HypothesisType


class CriticalValueResolver(ABC):
    """
    Critical value calculator interface. Calculate critical area.
    """

    @abstractmethod
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
        pass
