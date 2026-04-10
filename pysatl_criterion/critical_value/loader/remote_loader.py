import logging

from pysatl_criterion.persistence.model.limit_distribution.limit_distribution import (
    CriticalValueQuery,
    ILimitDistributionStorage,
)


class CriticalValueLoader:
    def __init__(
        self,
        local_storage: ILimitDistributionStorage,
        remote_storage: ILimitDistributionStorage | None,
    ):
        self.__local_storage = local_storage
        self.__remote_storage = remote_storage

    def load(self, criterion_code: str, sample_size: int, sample_size_error: int = 0) -> bool:
        """
        Load data from remote distribution storage to local distribution storage.

        Get sample_size - sample_size_error <= sample_size <= sample_size + sample_size_error

        :param criterion_code: criterion code
        :param sample_size: sample size
        :param sample_size_error: sample size error.
        :return: True if data exists, False otherwise.
        """

        logging.info(f"Load criterion {criterion_code} with size {sample_size} from remote")
        query = CriticalValueQuery(criterion_code, sample_size, sample_size_error)

        if self.__remote_storage is None:
            logging.error("Cannot load data: remote storage is not initialized.")
            return False

        remote_data = self.__remote_storage.get_data_for_cv(query)

        if remote_data is not None:
            self.__local_storage.insert_data(remote_data)
            return True
        else:
            logging.warning(
                f"Remote data for criterion {criterion_code} with size {sample_size} not found"
            )
            return False
