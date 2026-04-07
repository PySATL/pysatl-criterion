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
    storage = AlchemyLimitDistributionStorage("sqlite:///:memory:")
    storage.init()
    return storage


@pytest.fixture
def remote_storage():
    """Create remote storage using SQLAlchemy in-memory database."""
    storage = AlchemyLimitDistributionStorage("sqlite:///:memory:")
    storage.init()
    return storage


@pytest.fixture
def loader(local_storage, remote_storage):
    """Create CriticalValueLoader instance with real storages."""
    return CriticalValueLoader(local_storage, remote_storage)


@pytest.fixture
def sample_query():
    """Create a sample CriticalValueQuery for testing."""
    return CriticalValueQuery(criterion_code="test_criterion", sample_size=100)


@pytest.fixture
def sample_model():
    """Create a sample LimitDistributionModel for testing."""
    return LimitDistributionModel(
        experiment_id=1,
        criterion_code="test_criterion",
        criterion_parameters=[1.0, 2.0],
        sample_size=100,
        monte_carlo_count=1000,
        results_statistics=[0.1, 0.2, 0.3, 0.4, 0.5],
    )


def test_load_success_with_remote_data(
    loader, sample_query, sample_model, remote_storage, local_storage
):
    """Test load function when remote data is found and successfully inserted to local storage."""
    # Arrange - insert data into remote storage
    remote_storage.insert_data(sample_model)

    # Act
    loader.load(sample_query.criterion_code, sample_query.sample_size)

    # Assert - verify data was copied to local storage
    local_data = local_storage.get_data_for_cv(sample_query)
    assert local_data is not None
    assert local_data.criterion_code == sample_model.criterion_code
    assert local_data.sample_size == sample_model.sample_size
    assert local_data.results_statistics == sample_model.results_statistics


def test_load_no_remote_data(loader, sample_query, local_storage):
    """Test load function when remote data is not found (returns None)."""
    # Act
    loader.load(sample_query.criterion_code, sample_query.sample_size)

    # Assert - verify no data was inserted into local storage
    local_data = local_storage.get_data_for_cv(sample_query)
    assert local_data is None


def test_load_storage_interactions_correct_order(
    loader, sample_query, sample_model, remote_storage, local_storage
):
    """Test that storage methods are called in the correct order and data flows correctly."""
    # Arrange
    remote_storage.insert_data(sample_model)

    # Act
    loader.load(sample_query.criterion_code, sample_query.sample_size)

    # Assert - verify data exists in both storages
    remote_data = remote_storage.get_data_for_cv(sample_query)
    local_data = local_storage.get_data_for_cv(sample_query)

    assert remote_data is not None
    assert local_data is not None
    assert local_data.results_statistics == remote_data.results_statistics


def test_load_with_different_query_parameters(loader, remote_storage, local_storage):
    """Test load function with different query parameters."""
    # Test with different criterion codes and sample sizes
    test_cases = [
        (
            CriticalValueQuery(criterion_code="ks_test", sample_size=50),
            LimitDistributionModel(
                experiment_id=1,
                criterion_code="ks_test",
                criterion_parameters=[],
                sample_size=50,
                monte_carlo_count=1000,
                results_statistics=[0.1, 0.2, 0.3],
            ),
        ),
        (
            CriticalValueQuery(criterion_code="ad_test", sample_size=200),
            LimitDistributionModel(
                experiment_id=1,
                criterion_code="ad_test",
                criterion_parameters=[],
                sample_size=200,
                monte_carlo_count=1000,
                results_statistics=[0.4, 0.5, 0.6],
            ),
        ),
        (
            CriticalValueQuery(criterion_code="cvm_test", sample_size=1000),
            LimitDistributionModel(
                experiment_id=1,
                criterion_code="cvm_test",
                criterion_parameters=[],
                sample_size=1000,
                monte_carlo_count=1000,
                results_statistics=[0.7, 0.8, 0.9],
            ),
        ),
    ]

    for query, model in test_cases:
        # Arrange
        remote_storage.insert_data(model)

        # Act
        loader.load(query.criterion_code, query.sample_size)

        # Assert
        local_data = local_storage.get_data_for_cv(query)
        assert local_data is not None
        assert local_data.criterion_code == model.criterion_code
        assert local_data.sample_size == model.sample_size
        assert local_data.results_statistics == model.results_statistics


def test_load_with_empty_model_data(loader, sample_query, remote_storage, local_storage):
    """Test load function with empty model data."""
    # Arrange
    empty_model = LimitDistributionModel(
        experiment_id=1,
        criterion_code="test_criterion",
        criterion_parameters=[],
        sample_size=100,
        monte_carlo_count=1000,
        results_statistics=[],
    )
    remote_storage.insert_data(empty_model)

    # Act
    loader.load(sample_query.criterion_code, sample_query.sample_size)

    # Assert
    local_data = local_storage.get_data_for_cv(sample_query)
    assert local_data is not None
    assert local_data.results_statistics == []


def test_load_data_already_exists_locally(
    loader, sample_query, sample_model, remote_storage, local_storage
):
    """Test load function when data already exists in local storage."""
    # Arrange - insert same data into both storages
    remote_storage.insert_data(sample_model)

    sample_model_local = LimitDistributionModel(
        experiment_id=1,
        criterion_code="ks_test",
        criterion_parameters=[],
        sample_size=100,
        monte_carlo_count=1000,
        results_statistics=[0.1, 0.2],
    )
    local_storage.insert_data(sample_model_local)

    # Act
    loader.load(sample_query.criterion_code, sample_query.sample_size)

    # Assert - verify data still exists and is correct
    local_data = local_storage.get_data_for_cv(sample_query)
    assert local_data is not None
    assert local_data.results_statistics == sample_model.results_statistics


def test_load_multiple_criteria_same_sample_size(loader, remote_storage, local_storage):
    """Test load function with multiple criteria but same sample size."""
    # Arrange
    models = [
        LimitDistributionModel(
            experiment_id=1,
            criterion_code="ks_test",
            criterion_parameters=[],
            sample_size=100,
            monte_carlo_count=1000,
            results_statistics=[0.1, 0.2],
        ),
        LimitDistributionModel(
            experiment_id=2,
            criterion_code="ad_test",
            criterion_parameters=[],
            sample_size=100,
            monte_carlo_count=1000,
            results_statistics=[0.3, 0.4],
        ),
    ]

    for model in models:
        remote_storage.insert_data(model)

    # Act - load each criterion
    for model in models:
        query = CriticalValueQuery(
            criterion_code=model.criterion_code, sample_size=model.sample_size
        )
        loader.load(query.criterion_code, query.sample_size)

    # Assert - verify all data was loaded correctly
    for model in models:
        query = CriticalValueQuery(
            criterion_code=model.criterion_code, sample_size=model.sample_size
        )
        local_data = local_storage.get_data_for_cv(query)
        assert local_data is not None
        assert local_data.criterion_code == model.criterion_code
        assert local_data.results_statistics == model.results_statistics
