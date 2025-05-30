from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LimitDistributionModel:
    """
    Limit distribution model for storage.
    """

    experiment_id: int
    criterion_code: str
    criterion_parameters: list[float]
    sample_size: int
    monte_carlo_count: int
    results_statistics: list[float]


@dataclass
class LimitDistributionQuery:
    """
    Query for limit distribution storage.
    """

    criterion_code: str
    criterion_parameters: list[float]
    sample_size: int
    monte_carlo_count: int


class ILimitDistributionStorage(ABC):
    @abstractmethod
    def get_data(self, query: LimitDistributionQuery) -> LimitDistributionModel:
        """
        Get limit distribution from storage.

        :param query: query for storage

        :return: limit distribution
        """
        pass

    @abstractmethod
    def insert_data(self, data: LimitDistributionModel) -> None:
        """
        Insert limit distribution to storage.

        :param data: data to insert

        :return: None
        """
        pass
