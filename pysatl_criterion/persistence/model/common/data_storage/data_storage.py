from dataclasses import dataclass
from typing import Protocol

from pysatl_criterion.persistence.model.common.storage import IStorage


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


class IDataStorage(IStorage, Protocol):
    """
    Data storage interface.
    """

    def get_data(self, query: DataQuery) -> DataModel:
        """
        Get data from data storage.

        :param query: query for storage

        :return: data in storage
        """
        pass

    def insert_data(self, data: DataModel) -> None:
        """
        Insert data to data storage.

        :param data: data to insert

        :return: None
        """
        pass
