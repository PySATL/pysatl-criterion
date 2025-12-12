import pytest
from sqlalchemy import inspect

from pysatl_criterion.persistence.limit_distribution.datastorage.datastorage import (
    AlchemyLimitDistributionStorage,
)


@pytest.fixture
def storage(tmp_path):
    storage = AlchemyLimitDistributionStorage("sqlite:///:memory:")
    storage.init()
    return storage


def test_tables_exist(storage):
    inspector = inspect(storage.engine)
    tables = inspector.get_table_names()
    assert "limit_distributions" in tables
