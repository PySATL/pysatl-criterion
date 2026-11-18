import pytest
from sqlalchemy import CheckConstraint, Column, Integer, MetaData, String, Table, select
from sqlalchemy.dialects import mssql, mysql, oracle, postgresql, sqlite
from sqlalchemy.exc import IntegrityError
from sqlalchemy.schema import CreateIndex, CreateTable

from pysatl_criterion.persistence.sqlalchemy.database import AlchemyDatabase
from pysatl_criterion.persistence.sqlalchemy.models.critical_value import CriticalValueORM
from pysatl_criterion.persistence.sqlalchemy.models.limit_distribution import LimitDistributionORM
from pysatl_criterion.persistence.sqlalchemy.upsert import save_if_newer


@pytest.mark.parametrize(
    "dialect",
    [
        sqlite.dialect(),
        postgresql.dialect(),
        mysql.dialect(),
        oracle.dialect(),
        oracle.dialect(max_identifier_length=30),
        mssql.dialect(),
    ],
)
def test_schema_has_portable_primary_keys(dialect):
    for model in (LimitDistributionORM, CriticalValueORM):
        table = model.__table__
        assert str(CreateTable(table).compile(dialect=dialect))
        for index in table.indexes:
            assert str(CreateIndex(index).compile(dialect=dialect))
        for column in table.primary_key:
            if isinstance(column.type, String):
                assert column.type.length is not None
                assert column.type.length <= 64


def test_integrity_error_does_not_discard_prior_work():
    database = AlchemyDatabase("sqlite:///:memory:")
    table = Table(
        "versioned",
        MetaData(),
        Column("id", Integer, primary_key=True),
        Column("monte_carlo_count", Integer, nullable=False),
        CheckConstraint("monte_carlo_count > 0"),
    )
    table.create(database.engine)
    try:
        with database.session_factory() as session:
            assert save_if_newer(session, table, {"id": 1, "monte_carlo_count": 10})
            with pytest.raises(IntegrityError):
                save_if_newer(session, table, {"id": 2, "monte_carlo_count": -1})
            session.commit()
        with database.session_factory() as session:
            assert session.execute(select(table)).all() == [(1, 10)]
    finally:
        database.dispose()
