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


@dataclass
class DataModel:
    """
    Data model for data storage.
    """


@dataclass
class DataQuery:
    """
    Query for data storage.
    """


M = TypeVar("M", bound=DataModel)
Q_contra = TypeVar("Q_contra", contravariant=True, bound=DataQuery)


class IDataStorage(IStorage, ABC, Generic[M, Q_contra]):
    """
    Data storage interface.
    """

    @abstractmethod
    def get_data(self, query: Q_contra) -> M | None:
        """
        Get data from data storage.

        :param query: query for storage

        :return: data in storage
        """

    @abstractmethod
    def insert_data(self, data: M) -> None:
        """
        Insert data to data storage.

        :param data: data to insert

        :return: None
        """

    @abstractmethod
    def delete_data(self, query: Q_contra) -> None:
        """
        Delete data from data storage.

        :param query: data to delete

        :return: None
        """
