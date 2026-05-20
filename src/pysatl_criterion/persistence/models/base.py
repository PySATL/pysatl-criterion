from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar


class IStorage(ABC):
    """
    Storage interface.
    """

    @abstractmethod
    def init(self) -> None:
        """
        Initialize storage.
        """
        pass


@dataclass
class DataModel:
    """
    Data model for data storage.
    """

    pass


@dataclass
class DataQuery:
    """
    Query for data storage.
    """

    pass


M = TypeVar("M", bound=DataModel)
Q = TypeVar("Q", contravariant=True, bound=DataQuery)


class IDataStorage(IStorage, Generic[M, Q], ABC):
    """
    Data storage interface.
    """

    @abstractmethod
    def get_data(self, query: Q) -> M | None:
        """
        Get data from data storage.

        :param query: query for storage

        :return: data in storage
        """
        pass

    @abstractmethod
    def insert_data(self, data: M) -> None:
        """
        Insert data to data storage.

        :param data: data to insert

        :return: None
        """
        pass

    @abstractmethod
    def delete_data(self, query: Q) -> None:
        """
        Delete data from data storage.

        :param query: data to delete

        :return: None
        """
