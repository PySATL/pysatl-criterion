import numpy as np
import pytest

from pysatl_criterion.critical_value.loader.remote_loader import CriticalValueLoader
from pysatl_criterion.persistence.limit_distribution.datastorage.datastorage import (
    AlchemyLimitDistributionStorage,
)
from pysatl_criterion.persistence.model.limit_distribution.limit_distribution import (
    CriticalValueQuery,
    LimitDistributionModel,
)


@pytest.fixture
def local_storage():
    """Create local storage using SQLAlchemy in-memory database."""
    storage = AlchemyLimitDistributionStorage.create_safe("sqlite:///:memory:", label="Test Local")
    if not storage:
        pytest.fail("Could not initialize in-memory local storage")
    return storage


@pytest.fixture
def remote_storage():
    """Create remote storage using SQLAlchemy in-memory database."""
    storage = AlchemyLimitDistributionStorage.create_safe("sqlite:///:memory:", label="Test Remote")
    if not storage:
        pytest.fail("Could not initialize in-memory remote storage")
    return storage


@pytest.fixture
def loader(local_storage, remote_storage):
    """Create CriticalValueLoader instance with real storages."""
    return CriticalValueLoader(local_storage, remote_storage)


def test_load_bulk_all_new_data(loader, local_storage, remote_storage):
    """
    Test that load_bulk insert all new data to local storage.
    """
    criterion_codes = ["ks", "ad"]
    sample_size = 100

    # Prepare remote data
    models = [
        LimitDistributionModel(1, "ks", [], sample_size, 1000, [0.1, 0.2]),
        LimitDistributionModel(2, "ad", [], sample_size, 1000, [0.3, 0.4]),
    ]
    for model in models:
        remote_storage.insert_data(model)

    # Act
    result = loader.load_bulk(criterion_codes, sample_size)

    # Assert
    assert result.requested_count == 2
    assert result.already_cached_count == 0
    assert result.newly_cached_count == 2
    assert result.not_found_codes == []

    for model in models:
        query = CriticalValueQuery(
            criterion_code=model.criterion_code, sample_size=model.sample_size
        )
        local_data = local_storage.get_data_for_cv(query)
        assert local_data is not None
        assert np.allclose(local_data.results_statistics, model.results_statistics, rtol=1e-7)


def test_load_bulk_partial_cache(loader, local_storage, remote_storage):
    """
    Test the load_bulk method when some criteria are already in local storage and some are missing.
    """
    criterion_codes = ["ks", "ad", "cvm"]
    sample_size = 100

    local_storage.insert_data(LimitDistributionModel(1, "ks", [], sample_size, 1000, [0.1]))
    remote_storage.insert_data(LimitDistributionModel(2, "ad", [], sample_size, 1000, [0.2]))

    # Act
    result = loader.load_bulk(criterion_codes, sample_size)

    # Assert
    assert result.requested_count == 3
    assert result.already_cached_count == 1  # 'ks'
    assert result.newly_cached_count == 1  # 'ad'
    assert result.not_found_codes == ["cvm"]

    assert local_storage.get_data_for_cv(CriticalValueQuery("ks", sample_size)) is not None
    assert local_storage.get_data_for_cv(CriticalValueQuery("ad", sample_size)) is not None
    assert local_storage.get_data_for_cv(CriticalValueQuery("cvm", sample_size)) is None


def test_load_bulk_empty_remote(loader, local_storage, remote_storage):
    """
    Check that load_bulk do not add data to local storage when remote is empty.
    """
    criterion_codes = ["missing_1", "missing_2"]
    sample_size = 100

    # Act
    result = loader.load_bulk(criterion_codes, sample_size)

    # Assert Counters
    assert result.requested_count == 2
    assert result.already_cached_count == 0
    assert result.newly_cached_count == 0
    assert sorted(result.not_found_codes) == sorted(criterion_codes)

    for code in criterion_codes:
        assert local_storage.get_data_for_cv(CriticalValueQuery(code, sample_size)) is None


def test_load_bulk_all_cached(loader, local_storage, remote_storage):
    """
    Test load_bulk when all requested criteria are already in local cache.
    """
    criterion_codes = ["ks", "ad"]
    sample_size = 100

    for code in criterion_codes:
        local_storage.insert_data(LimitDistributionModel(1, code, [], sample_size, 1000, [0.5]))

    # Act
    result = loader.load_bulk(criterion_codes, sample_size)

    # Assert Counters
    assert result.requested_count == 2
    assert result.already_cached_count == 2
    assert result.newly_cached_count == 0
    assert result.not_found_codes == []


def test_load_bulk_with_unavailable_remote(local_storage, sample_size=100):
    """
    Test load_bulk when remote storage is None (unavailable).
    """
    criterion_codes = ["ks", "ad"]
    loader = CriticalValueLoader(local_storage, remote_storage=None)

    # Act
    result = loader.load_bulk(criterion_codes, sample_size)

    # Assert
    assert result.requested_count == 2
    assert result.already_cached_count == 0
    assert result.newly_cached_count == 0
    assert sorted(result.not_found_codes) == sorted(criterion_codes)


def test_load_bulk_preserves_best_monte_carlo(loader, local_storage, remote_storage):
    """
    In this test get_bulk_data should return the one with the highest MC count, and it gets cached.
    """
    criterion_codes = ["ks"]
    sample_size = 100

    # Insert data
    remote_storage.insert_data(LimitDistributionModel(1, "ks", [], sample_size, 500, [0.1]))
    remote_storage.insert_data(LimitDistributionModel(2, "ks", [], sample_size, 2000, [0.9]))

    # Act
    loader.load_bulk(criterion_codes, sample_size)

    # Assert
    local_data = local_storage.get_data_for_cv(CriticalValueQuery("ks", sample_size))
    assert local_data is not None
    assert local_data.monte_carlo_count == 2000
    assert np.allclose(local_data.results_statistics, [0.9])


def test_alchemy_storage_create_safe_on_failure():
    """
    Verify that create_safe returns None instead of raising on invalid URL.
    """
    bad_url = "postgresql://non_existent_user:pass@localhost:9999/nothing"
    storage = AlchemyLimitDistributionStorage.create_safe(bad_url, label="Broken DB")
    assert storage is None
