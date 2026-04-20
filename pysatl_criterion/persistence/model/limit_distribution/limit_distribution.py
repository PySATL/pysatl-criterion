from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from pysatl_criterion.persistence.model.common.data_storage.data_storage import (
    DataModel,
    DataQuery,
    IDataStorage,
)


@dataclass
class BulkLoadResult:
    """
    Detailed statistics for the bulk load operation.
    """

    requested_count: int
    already_cached_count: int
    newly_cached_count: int
    not_found_codes: list[str] = field(default_factory=list)


@dataclass
class LimitDistributionModel(DataModel):
    """
    Model for storing limit distribution data from Monte Carlo simulations.
    """

    experiment_id: int
    criterion_code: str
    criterion_parameters: list[float]
    sample_size: int
    monte_carlo_count: int
    results_statistics: list[float]


@dataclass
class LimitDistributionQuery(DataQuery):
    """
    Query object for retrieving specific limit distribution data.
    """

    criterion_code: str
    criterion_parameters: list[float]
    sample_size: int
    monte_carlo_count: int


@dataclass
class CriticalValueQuery(DataQuery):
    """
    Query object for retrieving limit distribution data for critical value calculation.
    """

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

    @abstractmethod
    def get_bulk_data(
        self, criterion_codes: list[str], sample_size: int, sample_size_error: int = 0
    ) -> list[LimitDistributionModel]:
        """
        Fetch multiple limit distributions using a batch query.

        :param criterion_codes: list of criterion codes.
        :param sample_size: number of samples.
        :param sample_size_error: number of samples error.
        :return: list of limit distribution data.
        """
        pass

    @abstractmethod
    def insert_bulk_data(self, models: list[LimitDistributionModel]) -> None:
        """
        Batch insert or update multiple distribution records.

        :param models: list of limit distribution data.
        """
        pass
