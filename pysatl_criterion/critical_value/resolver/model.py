from abc import ABC, abstractmethod

from pysatl_criterion.critical_value.critical_area.model import CriticalArea
from pysatl_criterion.statistics.models import HypothesisType


class CriticalValueResolver(ABC):
    """
    Critical value calculator interface. Calculate critical area.
    """

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

        return self.resolve_bulk([criterion_code], sample_size, sl, alternative).get(criterion_code)

    @abstractmethod
    def resolve_bulk(
        self,
        criterion_codes: list[str],
        sample_size: int,
        sl: float,
        alternative: HypothesisType = HypothesisType.RIGHT,
    ) -> dict[str, CriticalArea]:
        """
        Resolve multiple critical areas for a fixed sample size and alpha.

        :param criterion_codes: list of criterion identifiers.
        :param sample_size: target sample size.
        :param sl: significance level.
        :param alternative: test alternative.
        :return: dictionary mapping codes to critical areas.
        """
        pass
