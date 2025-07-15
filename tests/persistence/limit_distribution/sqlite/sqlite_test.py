import pytest

from pysatl_criterion.persistence.limit_distribution.sqlite.sqlite import (
    SQLiteLimitDistributionStorage,
)
from pysatl_criterion.persistence.model.limit_distribution.limit_distribution import (
    LimitDistributionModel,
)


@pytest.fixture
def storage(tmp_path):
    storage = SQLiteLimitDistributionStorage(":memory:")
    storage.init()
    yield storage
    if storage.conn:
        storage.conn.close()


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


def test_init(storage):
    conn = storage._get_connection()
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='limit_distributions'"
    )
    assert cursor.fetchone() is not None, "Not find limit_distributions tables"


def test_index_created(storage):
    conn = storage._get_connection()
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_cv_queries'"
    )
    assert cursor.fetchone() is not None, "Index idx_cv_queries not created"


def test_index_usage(storage, model_factory):
    storage.insert_data(model_factory())
    conn = storage._get_connection()
    cursor = conn.execute("""
            EXPLAIN QUERY PLAN
            SELECT * FROM limit_distributions
            WHERE criterion_code = 'test_code' AND sample_size = 100
        """)
    plans = cursor.fetchall()
    assert any("using index" in p[3].lower() for p in plans), f"Index not used: {plans}"
