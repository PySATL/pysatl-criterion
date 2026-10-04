import pytest
from sqlalchemy import inspect

from pysatl_criterion.persistence.sqlalchemy.database import AlchemyDatabase


@pytest.fixture
def database():
    database = AlchemyDatabase("sqlite:///:memory:")
    database.init()
    yield database
    database.dispose()


def test_tables_exist(database):
    inspector = inspect(database.engine)
    tables = inspector.get_table_names()
    assert "limit_distributions" in tables


def test_limit_distribution_keys(database):
    inspector = inspect(database.engine)
    primary_key = inspector.get_pk_constraint("limit_distributions")
    assert primary_key["constrained_columns"] == [
        "criterion_code_hash",
        "criterion_parameters_hash",
        "sample_size",
    ]
    indexes = inspector.get_indexes("limit_distributions")
    index = next(i for i in indexes if i["name"] == "ix_ld_monte_carlo_count")
    assert index["column_names"] == ["monte_carlo_count"]
    assert not index["unique"]
