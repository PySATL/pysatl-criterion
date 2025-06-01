from dataclasses import dataclass
from typing import Protocol, TypeVar


class IStorage(Protocol):
    """
    Storage interface.
    """

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


class IDataStorage(IStorage, Protocol[M, Q]):
    """
    Data storage interface.
    """

    def get_data(self, query: Q) -> M:
        """
        Get data from data storage.

        :param query: query for storage

        :return: data in storage
        """
        pass

    def insert_data(self, data: M) -> None:
        """
        Insert data to data storage.

        :param data: data to insert

        :return: None
        """
        pass
