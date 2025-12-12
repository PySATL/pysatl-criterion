import logging

from pysatl_criterion.persistence.model.limit_distribution.limit_distribution import (
    CriticalValueQuery,
    ILimitDistributionStorage,
)


class CriticalValueLoader:
    def __init__(
        self, local_storage: ILimitDistributionStorage, remote_storage: ILimitDistributionStorage
    ):
        self.__local_storage = local_storage
        self.__remote_storage = remote_storage

    def load(self, criterion_code: str, sample_size: int, sample_size_error: int = 0):
        """
        Load data from remote distribution storage to local distribution storage.

        :param criterion_code: criterion code
        :param sample_size: sample size
        :param sample_size_error: sample size error.
        Get sample_size - sample_size_error <= sample_size <= sample_size + sample_size_error
        """

        logging.info(f"Load criterion {criterion_code} with size {sample_size} from remote")
        query = CriticalValueQuery(criterion_code, sample_size, sample_size_error)
        remote_data = self.__remote_storage.get_data_for_cv(query)

        if remote_data is not None:
            self.__local_storage.insert_data(remote_data)
        else:
            logging.warning(
                f"Remote data for criterion {criterion_code} " f"with size {sample_size} not found"
            )
