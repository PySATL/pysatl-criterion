from pysatl_criterion.persistence.model.limit_distribution.limit_distribution import (
    CriticalValueQuery,
    ILimitDistributionStorage,
    LimitDistributionModel,
    LimitDistributionQuery,
)


class SQLiteLimitDistributionStorage(ILimitDistributionStorage):
    """
    SQLite limit distribution storage.
    """

    def __init__(self, connection_string: str):
        self.connection_string = connection_string

    def init(self) -> None:
        """
        Initialize SQLite limit distribution storage.

        :return: None
        """

    def get_data(self, query: LimitDistributionQuery) -> LimitDistributionModel:
        """
        Get limit distribution from SQLite storage.

        :param query: limit distribution query.

        :return: limit distribution data.
        """
        raise NotImplementedError("Method is not yet implemented")

    def insert_data(self, data: LimitDistributionModel) -> None:
        """
        Insert limit distribution to SQLite storage.

        :param data: limit distribution data.

        :return: None
        """
        raise NotImplementedError("Method is not yet implemented")

    def delete_data(self, query: LimitDistributionQuery) -> None:
        """
        Delete limit distribution from SQLite storage.

        :param query: limit distribution to delete.

        :return: None
        """
        raise NotImplementedError("Method is not yet implemented")

    def get_data_for_cv(self, query: CriticalValueQuery) -> LimitDistributionModel:
        """
        Get limit distribution data for critical value calculation from SQLite storage.

        :param query: calculation parameters.

        :return: limit distribution data.
        """

        raise NotImplementedError("Method is not yet implemented")
