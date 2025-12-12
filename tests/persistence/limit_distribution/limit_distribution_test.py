import pytest

from pysatl_criterion.persistence.limit_distribution.datastorage.datastorage import (
    AlchemyLimitDistributionStorage,
)
from pysatl_criterion.persistence.limit_distribution.sqlite.sqlite import (
    SQLiteLimitDistributionStorage,
)
from pysatl_criterion.persistence.model.limit_distribution.limit_distribution import (
    CriticalValueQuery,
    LimitDistributionModel,
    LimitDistributionQuery,
)


@pytest.fixture(
    params=[
        pytest.param(lambda: SQLiteLimitDistributionStorage(":memory:"), id="sqlite"),
        pytest.param(
            lambda: AlchemyLimitDistributionStorage("sqlite:///:memory:"), id="sqlalchemy"
        ),
    ]
)
def storage_factory(request):
    return request.param


@pytest.fixture
def storage(storage_factory):
    store = storage_factory()
    store.init()
    yield store
    if hasattr(store, "conn") and store.conn:
        store.conn.close()


@pytest.fixture
def model_factory():
    def _create(**overrides):
        base = dict(
            experiment_id=1,
            criterion_code="test_code",
            criterion_parameters=[-1],
            sample_size=100,
            monte_carlo_count=1000,
            results_statistics=[-4.0],
        )
        base.update(overrides)
        return LimitDistributionModel(**base)

    return _create


@pytest.fixture
def query_factory():
    def _create(**overrides):
        base = dict(
            criterion_code="test_code",
            criterion_parameters=[-1],
            sample_size=100,
            monte_carlo_count=1000,
        )
        base.update(overrides)
        return LimitDistributionQuery(**base)

    return _create


@pytest.mark.parametrize(
    "sample_size, expected",
    [
        (100, True),
        (200, False),
    ],
)
def test_get_data_parametrized(storage, model_factory, query_factory, sample_size, expected):
    model = model_factory(sample_size=100)
    storage.insert_data(model)

    query = query_factory(sample_size=sample_size)
    retrieved_model = storage.get_data(query)

    if expected:
        assert retrieved_model is not None
    else:
        assert retrieved_model is None


def test_get_data(storage, model_factory, query_factory):
    model = model_factory()
    storage.insert_data(model)

    query = query_factory()
    retrieved_model = storage.get_data(query)

    assert retrieved_model is not None, "No data find"
    assert retrieved_model.experiment_id == 1
    assert retrieved_model.criterion_code == "test_code"
    assert retrieved_model.criterion_parameters == [-1]
    assert retrieved_model.sample_size == 100
    assert retrieved_model.monte_carlo_count == 1000
    assert retrieved_model.results_statistics == [-4]


def test_delete_data(storage, model_factory, query_factory):
    model = model_factory()
    storage.insert_data(model)

    query = query_factory()
    storage.delete_data(query)

    retrieved_model = storage.get_data(query)
    assert retrieved_model is None, "Data isn't deleted"


def test_unique_constraint(storage, model_factory, query_factory):
    model1 = model_factory(results_statistics=[1.0])
    model2 = model_factory(results_statistics=[2.0])
    storage.insert_data(model1)
    storage.insert_data(model2)  # Должна заменить первую запись

    query = query_factory()
    retrieved_model = storage.get_data(query)
    assert retrieved_model.results_statistics == [2.0], "No unique constraint found"


def test_get_data_for_cv(storage, model_factory, query_factory):
    model1 = model_factory(monte_carlo_count=500)
    model2 = model_factory(monte_carlo_count=1000)
    storage.insert_data(model1)
    storage.insert_data(model2)

    query = CriticalValueQuery(criterion_code="test_code", sample_size=100)
    retrieved_model = storage.get_data_for_cv(query)

    assert retrieved_model is not None, "Data not found"
    assert retrieved_model.monte_carlo_count == 1000, "Not max monte carlo count"


def test_get_data_for_cv_not_found(storage):
    query = CriticalValueQuery(criterion_code="missing", sample_size=999)
    assert storage.get_data_for_cv(query) is None, "Find mystic data"
