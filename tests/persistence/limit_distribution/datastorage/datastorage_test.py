import pytest
from sqlalchemy import inspect

from pysatl_criterion.persistence.limit_distribution.datastorage.datastorage import (
    AlchemyLimitDistributionStorage,
)


@pytest.fixture
def storage(tmp_path):
    storage = AlchemyLimitDistributionStorage.create_safe(
        "sqlite:///:memory:", label="Test Storage"
    )
    if storage is None:
        pytest.fail("Failed to initialize in-memory database for testing")
    return storage


def test_tables_exist(storage):
    inspector = inspect(storage.engine)
    tables = inspector.get_table_names()
    assert "limit_distributions" in tables
