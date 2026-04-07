from abc import ABC, abstractmethod
from dataclasses import dataclass

from pysatl_criterion.persistence.model.common.data_storage.data_storage import (
    DataModel,
    DataQuery,
    IDataStorage,
)


@dataclass
class LimitDistributionModel(DataModel):
    experiment_id: int
    criterion_code: str
    criterion_parameters: list[float]
    sample_size: int
    monte_carlo_count: int
    results_statistics: list[float]


@dataclass
class LimitDistributionQuery(DataQuery):
    criterion_code: str
    criterion_parameters: list[float]
    sample_size: int
    monte_carlo_count: int


@dataclass
class CriticalValueQuery(DataQuery):
    criterion_code: str
    sample_size: int
    sample_size_error: int = 0


class ILimitDistributionStorage(IDataStorage[LimitDistributionModel, LimitDistributionQuery], ABC):
    """
    Limit distribution storage interface.
    """

    @abstractmethod
    def get_data_for_cv(self, query: CriticalValueQuery) -> LimitDistributionModel | None:
        """
        Get limit distribution data for critical value calculation.

        :param query: calculation parameters.

        :return: limit distribution data.
        """

        pass
