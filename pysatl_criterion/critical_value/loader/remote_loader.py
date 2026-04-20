import logging

from pysatl_criterion.persistence.model.limit_distribution.limit_distribution import (
    BulkLoadResult,
    CriticalValueQuery,
    ILimitDistributionStorage,
)

logger = logging.getLogger(__name__)

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

        if self.__remote_storage.get_data_for_cv(query):
            return True

        remote_data = self.__remote_storage.get_data_for_cv(query)
        if remote_data is not None:
            self.__local_storage.insert_data(remote_data)
            logger.debug(f"Loaded {criterion_code} (n={sample_size}) from remote.")
            return True

        logging.warning(
            f"Remote data for criterion {criterion_code} with size {sample_size} not found"
        )
        return False

    def load_bulk(
        self, criterion_codes: list[str], sample_size: int, sample_size_error: int = 0
    ) -> BulkLoadResult:
        """
        Synchronize multiple criteria with cache-miss optimization and bulk insert.

        :param criterion_codes: list of criterion codes to load.
        :param sample_size: sample size for the query.
        :param sample_size_error: acceptable error in sample size for matching.
        :return: BulkLoadResult containing counts of requested, cached, newly loaded, and not
        found criteria.
        """
        if not criterion_codes:
            return BulkLoadResult(0, 0, 0, [])

        if self.__remote_storage is None:
            logging.error("Remote storage not initialized.")
            return BulkLoadResult(len(criterion_codes), 0, 0, criterion_codes)

        # 1. Identify what's already in the local cache
        already_cached = []
        missing_codes = []

        for code in criterion_codes:
            query = CriticalValueQuery(code, sample_size, sample_size_error)
            if self.__local_storage.get_data_for_cv(query):
                already_cached.append(code)
            else:
                missing_codes.append(code)

        if not missing_codes:
            logging.info("All requested criteria are already in the local cache.")
            return BulkLoadResult(len(criterion_codes), len(already_cached), 0)

        # 2. Fetch only missing data from remote
        logger.info(f"Fetching {len(missing_codes)} missing criteria from remote...")
        remote_models = self.__remote_storage.get_bulk_data(
            missing_codes, sample_size, sample_size_error
        )

        # 3. Bulk insert into local
        try:
            self.__local_storage.insert_bulk_data(remote_models)
            logger.info(f"Successfully cached {len(remote_models)} new distributions.")
        except Exception as e:
            logger.error(f"Failed to save bulk data to local storage: {e}")
            pass

        # 4. Calculate what's still missing (not found even on remote)
        found_codes = {m.criterion_code for m in remote_models}
        not_found = [c for c in missing_codes if c not in found_codes]

        logging.info(
            f"Bulk load finished: {len(remote_models)} new, "
            f"{len(already_cached)} skipped, {len(not_found)} failed."
        )

        return BulkLoadResult(
            requested_count=len(criterion_codes),
            already_cached_count=len(already_cached),
            newly_cached_count=len(remote_models),
            not_found_codes=not_found,
        )
