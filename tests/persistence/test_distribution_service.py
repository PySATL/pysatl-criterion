import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from decimal import Decimal
from threading import Barrier, Event
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, delete, event, func, select
from sqlalchemy.engine import make_url
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.schema import CreateSchema, DropSchema

from pysatl_criterion.hypothesis_testing.distribution_service import (
    DistributionChangedError,
    DistributionNotFoundError,
    DistributionService,
)
from pysatl_criterion.loader.limit_distribution_loader import LimitDistributionLoader
from pysatl_criterion.loader.store_adapters import LimitDistributionReader, LimitDistributionWriter
from pysatl_criterion.persistence.models.critical_value import CriticalValueModel
from pysatl_criterion.persistence.models.distribution_key import DistributionKey
from pysatl_criterion.persistence.models.limit_distribution import LimitDistributionModel
from pysatl_criterion.persistence.queries.critical_value import CriticalValueLookupQuery
from pysatl_criterion.persistence.queries.limit_distribution import LimitDistributionFilter
from pysatl_criterion.persistence.sqlalchemy.database import AlchemyDatabase
from pysatl_criterion.persistence.sqlalchemy.models.critical_value import CriticalValueORM
from pysatl_criterion.persistence.sqlalchemy.models.limit_distribution import (
    LimitDistributionORM,
    criterion_code_hash,
    criterion_parameters_hash,
)
from pysatl_criterion.persistence.sqlalchemy.unit_of_work import AlchemyUnitOfWork
from pysatl_criterion.statistics.alternative import AlternativeType


@pytest.fixture(
    params=[
        "sqlite",
        pytest.param(
            "mysql",
            marks=pytest.mark.skipif(
                not os.environ.get("CRITERION_TEST_MYSQL_URL"),
                reason="Set CRITERION_TEST_MYSQL_URL for MySQL integration tests",
            ),
        ),
        pytest.param(
            "postgresql",
            marks=pytest.mark.skipif(
                not os.environ.get("CRITERION_TEST_POSTGRES_URL"),
                reason="Set CRITERION_TEST_POSTGRES_URL for PostgreSQL integration tests",
            ),
        ),
    ]
)
def database(tmp_path, request):
    if request.param == "sqlite":
        db = AlchemyDatabase(f"sqlite:///{tmp_path / 'distributions.db'}")
        db.init()
        yield db
        db.dispose()
        return

    # Each test owns an isolated schema; existing application tables are never touched.
    backend = request.param
    url = make_url(
        os.environ[f"CRITERION_TEST_{'MYSQL' if backend == 'mysql' else 'POSTGRES'}_URL"]
    )
    admin = create_engine(url)
    schema = "criterion_test_" + uuid4().hex
    with admin.begin() as connection:
        connection.execute(CreateSchema(schema))
    test_url = (
        url.set(database=schema)
        if backend == "mysql"
        else url.update_query_dict({"options": f"-csearch_path={schema}"})
    )
    db = AlchemyDatabase(test_url.render_as_string(hide_password=False))

    try:
        db.init()
        yield db
    finally:
        db.dispose()
        with admin.begin() as connection:
            connection.execute(DropSchema(schema, cascade=backend == "postgresql"))
        admin.dispose()


@pytest.fixture
def uow_factory(database):
    return lambda: AlchemyUnitOfWork(database.session_factory)


@pytest.fixture
def service(uow_factory):
    return DistributionService(uow_factory)


@pytest.fixture
def distribution():
    return LimitDistributionModel("criterion", {"shape": 1.0}, 20, 100, [0.0, 1.0, 2.0, 3.0])


def query_for(value):
    return CriticalValueLookupQuery(
        value.distribution_key,
        value.monte_carlo_count,
        value.significance_level,
        value.alternative_type,
    )


def cache_count(database):
    with database.session_factory() as session:
        return session.scalar(select(func.count()).select_from(CriticalValueORM))


@pytest.mark.parametrize("count, expected", [(50, False), (100, False), (200, True)])
def test_save_only_better(service, uow_factory, distribution, count, expected):
    assert service.save_distribution(distribution)
    incoming = replace(distribution, monte_carlo_count=count, results_statistics=[9.0])
    assert service.save_distribution(incoming) is expected
    with uow_factory() as uow:
        stored = uow.limit_distributions.get(distribution.key)
    assert stored == (incoming if expected else distribution)


@pytest.mark.parametrize(
    "alternative, expected",
    [
        (AlternativeType.LEFT, (0.3, None)),
        (AlternativeType.RIGHT, (None, 2.7)),
        (AlternativeType.TWO_TAILED, (0.15, 2.85)),
    ],
)
def test_calculate_and_cache(service, distribution, alternative, expected, monkeypatch):
    service.save_distribution(distribution)
    result = service.get_critical_value(distribution.key, Decimal("0.1"), alternative)
    assert result.lower_value == (pytest.approx(expected[0]) if expected[0] is not None else None)
    assert result.upper_value == (pytest.approx(expected[1]) if expected[1] is not None else None)
    assert result.monte_carlo_count == 100

    def unexpected_calculation(*args):
        pytest.fail("Cached critical values should not be recalculated")

    monkeypatch.setattr(service, "_calculate", unexpected_calculation)
    assert service.get_critical_value(distribution.key, Decimal("0.10"), alternative) == result


def test_improvement_invalidates_all_levels_and_alternatives(
    service, database, uow_factory, distribution
):
    service.save_distribution(distribution)
    other = replace(distribution, criterion_parameters={"shape": 2.0})
    service.save_distribution(other)
    other_value = service.get_critical_value(other.key, Decimal("0.05"), AlternativeType.RIGHT)
    for level in (Decimal("0.05"), Decimal("0.1")):
        for alternative in AlternativeType:
            service.get_critical_value(distribution.key, level, alternative)
    assert cache_count(database) == 7
    assert not service.save_distribution(replace(distribution, monte_carlo_count=50))
    assert not service.save_distribution(distribution)
    assert cache_count(database) == 7
    assert service.save_distribution(replace(distribution, monte_carlo_count=200))
    assert cache_count(database) == 1
    with uow_factory() as uow:
        assert uow.critical_values.get(query_for(other_value)) == other_value
    result = service.get_critical_value(distribution.key, Decimal("0.05"), AlternativeType.RIGHT)
    assert result.monte_carlo_count == 200


@pytest.mark.parametrize("fail", [False, True])
def test_uncommitted_work_rolls_back_both_stores(
    service, database, uow_factory, distribution, fail
):
    service.save_distribution(distribution)
    cached = service.get_critical_value(distribution.key, Decimal("0.05"), AlternativeType.RIGHT)
    try:
        with uow_factory() as uow:
            assert uow.limit_distributions.insert(replace(distribution, monte_carlo_count=200))
            uow.critical_values.delete_for_distribution(distribution.key)
            if fail:
                raise RuntimeError("abort")
    except RuntimeError:
        pass
    with uow_factory() as uow:
        assert uow.limit_distributions.get(distribution.key) == distribution
        assert uow.critical_values.get(query_for(cached)) == cached
    assert cache_count(database) == 1


def test_stale_calculation_cannot_be_saved(service, uow_factory, database, distribution):
    service.save_distribution(distribution)
    old = service.get_critical_value(distribution.key, Decimal("0.05"), AlternativeType.RIGHT)
    service.save_distribution(replace(distribution, monte_carlo_count=200))
    with uow_factory() as uow:
        assert not uow.critical_values.save_if_current(old)
        uow.commit()
    assert cache_count(database) == 0


def test_change_during_calculation_retries(service, distribution, monkeypatch):
    service.save_distribution(distribution)
    calculate = service._calculate
    versions = []

    def calculate_with_update(source, level, alternative):
        versions.append(source.monte_carlo_count)
        if len(versions) == 1:
            service.save_distribution(replace(distribution, monte_carlo_count=200))
        return calculate(source, level, alternative)

    monkeypatch.setattr(service, "_calculate", calculate_with_update)
    result = service.get_critical_value(distribution.key, Decimal("0.05"), AlternativeType.RIGHT)
    assert versions == [100, 200]
    assert result.monte_carlo_count == 200


def test_retries_are_bounded(service, distribution, monkeypatch, database):
    service.save_distribution(distribution)
    calculate = service._calculate

    def calculate_with_update(source, level, alternative):
        service.save_distribution(replace(source, monte_carlo_count=source.monte_carlo_count + 1))
        return calculate(source, level, alternative)

    monkeypatch.setattr(service, "_calculate", calculate_with_update)
    with pytest.raises(DistributionChangedError):
        service.get_critical_value(distribution.key, Decimal("0.05"), AlternativeType.RIGHT)
    assert cache_count(database) == 0


def test_missing_distribution(service, distribution, uow_factory):
    with pytest.raises(DistributionNotFoundError):
        service.get_critical_value(distribution.key, Decimal("0.05"), AlternativeType.RIGHT)
    with uow_factory() as uow:
        assert not uow.critical_values.save_if_current(
            CriticalValueModel(
                distribution.key,
                100,
                Decimal("0.05"),
                AlternativeType.RIGHT,
                upper_value=1.0,
            )
        )


def test_source_deletion_cascades(service, database, distribution):
    service.save_distribution(distribution)
    service.get_critical_value(distribution.key, Decimal("0.05"), AlternativeType.RIGHT)
    with database.session_factory() as session:
        session.execute(
            delete(LimitDistributionORM).where(
                LimitDistributionORM.criterion_code_hash
                == criterion_code_hash(distribution.criterion_code),
                LimitDistributionORM.criterion_parameters_hash
                == criterion_parameters_hash(distribution.criterion_parameters),
                LimitDistributionORM.sample_size == distribution.sample_size,
            )
        )
        session.commit()
    assert cache_count(database) == 0


def test_concurrent_writers_keep_maximum(service, uow_factory, distribution):
    barrier = Barrier(2)

    def save(count):
        barrier.wait(timeout=5)
        return service.save_distribution(
            replace(distribution, monte_carlo_count=count, results_statistics=[float(count)])
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(save, [100, 200]))
    assert any(results)
    with uow_factory() as uow:
        result = uow.limit_distributions.get(distribution.key)
        assert result.monte_carlo_count == 200
        assert result.results_statistics == [200.0]


def test_critical_value_write_and_distribution_update_are_serialized(
    service, uow_factory, database, distribution
):
    service.save_distribution(distribution)
    critical_value = CriticalValueModel(
        distribution.key,
        100,
        Decimal("0.05"),
        AlternativeType.RIGHT,
        upper_value=2.0,
    )
    update_started = Event()

    def update_source():
        update_started.set()
        return service.save_distribution(replace(distribution, monte_carlo_count=200))

    with ThreadPoolExecutor(max_workers=1) as executor:
        with uow_factory() as uow:
            assert uow.critical_values.save_if_current(critical_value)
            update = executor.submit(update_source)
            assert update_started.wait(timeout=5)
            uow.commit()
        assert update.result(timeout=5)
    assert cache_count(database) == 0


@pytest.mark.parametrize("level", ["0", "1", "NaN", "Infinity", "0.00000000001"])
def test_invalid_significance_levels(distribution, level):
    with pytest.raises(ValueError):
        CriticalValueModel(
            distribution.key,
            100,
            Decimal(level),
            AlternativeType.RIGHT,
            upper_value=1.0,
        )


@pytest.mark.parametrize(
    "alternative, lower, upper",
    [
        (AlternativeType.LEFT, None, 1.0),
        (AlternativeType.RIGHT, 1.0, None),
        (AlternativeType.TWO_TAILED, None, 1.0),
        (AlternativeType.TWO_TAILED, 2.0, 1.0),
        (AlternativeType.RIGHT, None, float("inf")),
    ],
)
def test_invalid_bounds(distribution, alternative, lower, upper):
    with pytest.raises(ValueError):
        CriticalValueModel(distribution.key, 100, Decimal("0.05"), alternative, lower, upper)


def test_foreign_key_is_enforced(database, distribution):
    value = CriticalValueModel(
        distribution.key,
        100,
        Decimal("0.05"),
        AlternativeType.RIGHT,
        upper_value=1.0,
    )
    with database.session_factory() as session:
        session.add(CriticalValueORM.from_model(value))
        with pytest.raises(IntegrityError):
            session.commit()


def test_same_code_different_parameters_remains_distinct(service, distribution):
    service.save_distribution(distribution)
    second = replace(
        distribution, criterion_parameters={"shape": 2}, results_statistics=[10.0, 20.0]
    )
    service.save_distribution(second)
    first_value = service.get_critical_value(
        DistributionKey("criterion", {"shape": 1}, 20), Decimal("0.1"), AlternativeType.RIGHT
    )
    second_value = service.get_critical_value(second.key, Decimal("0.1"), AlternativeType.RIGHT)
    assert first_value.upper_value == pytest.approx(2.7)
    assert second_value.upper_value == pytest.approx(19.0)


def test_service_rolls_back_if_invalidation_fails(service, distribution, uow_factory, monkeypatch):
    from pysatl_criterion.persistence.sqlalchemy.stores.critical_value import (
        AlchemyCriticalValueStorage,
    )

    service.save_distribution(distribution)
    cached = service.get_critical_value(distribution.key, Decimal("0.05"), AlternativeType.RIGHT)

    def failed_invalidation(self, key):
        raise RuntimeError("invalidation failed")

    monkeypatch.setattr(AlchemyCriticalValueStorage, "delete_for_distribution", failed_invalidation)
    with pytest.raises(RuntimeError, match="invalidation failed"):
        service.save_distribution(replace(distribution, monte_carlo_count=200))
    with uow_factory() as uow:
        assert uow.limit_distributions.get(distribution.key) == distribution
        assert uow.critical_values.get(query_for(cached)) == cached


def test_read_rejects_cache_from_old_source_version(service, distribution, uow_factory):
    service.save_distribution(distribution)
    cached = service.get_critical_value(distribution.key, Decimal("0.05"), AlternativeType.RIGHT)
    with uow_factory() as uow:
        # Deliberately bypass service invalidation: the read must still check source version.
        assert uow.limit_distributions.insert(replace(distribution, monte_carlo_count=200))
        uow.commit()
    with uow_factory() as uow:
        assert uow.critical_values.get(query_for(cached)) is None


@pytest.mark.parametrize(
    "changes",
    [
        {"significance_level": Decimal("0")},
        {"monte_carlo_count": 0},
        {"lower_value": 1.0},
        {"upper_value": None},
    ],
)
def test_database_constraints_enforce_bounds_and_level(service, distribution, database, changes):
    service.save_distribution(distribution)
    value = CriticalValueModel(
        distribution.key,
        100,
        Decimal("0.05"),
        AlternativeType.RIGHT,
        upper_value=1.0,
    )
    orm = CriticalValueORM.from_model(value)
    for name, value in changes.items():
        setattr(orm, name, value)
    with database.session_factory() as session:
        session.add(orm)
        with pytest.raises(DBAPIError, match="ck_(critical_values|cv)_"):
            session.commit()


def test_concurrent_first_insert_preserves_other_transaction_work(
    database, uow_factory, distribution
):
    if database.engine.dialect.name != "postgresql":
        pytest.skip("Force the savepoint conflict on PostgreSQL; SQLite serializes writers")
    barrier = Barrier(2)

    def before_insert(connection, cursor, statement, parameters, context, executemany):
        if (
            statement.startswith("INSERT INTO limit_distributions")
            and parameters.get("criterion_code") == distribution.criterion_code
        ):
            barrier.wait(timeout=5)

    event.listen(database.engine, "before_cursor_execute", before_insert)

    def save(count):
        with uow_factory() as uow:
            other = replace(distribution, criterion_code=f"other_{count}")
            uow.limit_distributions.insert(other)
            changed = uow.limit_distributions.insert(replace(distribution, monte_carlo_count=count))
            if changed:
                uow.critical_values.delete_for_distribution(distribution.key)
            uow.commit()
        return other.key

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            other_keys = list(executor.map(save, [100, 200]))
    finally:
        event.remove(database.engine, "before_cursor_execute", before_insert)
    with uow_factory() as uow:
        assert uow.limit_distributions.get(distribution.key).monte_carlo_count == 200
        assert all(uow.limit_distributions.get(key) is not None for key in other_keys)


def test_long_unicode_keys_are_preserved(service, distribution):
    model = replace(
        distribution, criterion_code="Критерий" * 500, criterion_parameters={"параметр" * 100: 1.0}
    )
    assert service.save_distribution(model)
    cached = service.get_critical_value(model.key, Decimal("0.1"), AlternativeType.RIGHT)
    assert cached.distribution_key == model.key


def test_large_distribution_round_trip(service, distribution, uow_factory):
    import numpy as np

    values = np.random.default_rng(42).normal(size=40_000).tolist()
    model = replace(distribution, monte_carlo_count=40_000, results_statistics=values)
    assert service.save_distribution(model)
    with uow_factory() as uow:
        stored = uow.limit_distributions.get(model.key)
    assert stored.results_statistics == pytest.approx(values, rel=1e-6, abs=1e-7)


def test_batch_save_invalidates_only_changed_distributions(service, uow_factory, distribution):
    first = distribution
    second = replace(distribution, criterion_code="second", monte_carlo_count=200)
    service.save_distributions([first, second])
    first_cv = service.get_critical_value(first.key, Decimal("0.05"), AlternativeType.RIGHT)
    second_cv = service.get_critical_value(second.key, Decimal("0.05"), AlternativeType.RIGHT)
    improved = replace(first, monte_carlo_count=300)
    incoming = [improved, replace(second, monte_carlo_count=100)]
    assert service.save_distributions(incoming) == [improved]
    with uow_factory() as uow:
        assert uow.critical_values.get(query_for(first_cv)) is None
        assert uow.critical_values.get(query_for(second_cv)) == second_cv
        assert uow.limit_distributions.get(second.key) == second


def test_batch_and_invalidation_roll_back_together(service, uow_factory, distribution, monkeypatch):
    from pysatl_criterion.persistence.sqlalchemy.stores.critical_value import (
        AlchemyCriticalValueStorage,
    )

    first = distribution
    second = replace(distribution, criterion_code="second")
    service.save_distributions([first, second])
    cached = [
        service.get_critical_value(m.key, Decimal("0.05"), AlternativeType.RIGHT)
        for m in [first, second]
    ]
    invalidate = AlchemyCriticalValueStorage.delete_for_distribution
    calls = []

    def fail_on_second(store, key):
        calls.append(key)
        invalidate(store, key)
        if len(calls) == 2:
            raise RuntimeError("batch invalidation failed")

    monkeypatch.setattr(AlchemyCriticalValueStorage, "delete_for_distribution", fail_on_second)
    with pytest.raises(RuntimeError, match="batch invalidation failed"):
        service.save_distributions([replace(m, monte_carlo_count=200) for m in [first, second]])
    with uow_factory() as uow:
        assert uow.limit_distributions.get(first.key) == first
        assert uow.limit_distributions.get(second.key) == second
        assert all(uow.critical_values.get(query_for(value)) == value for value in cached)


def test_bulk_insert_does_not_commit_its_own_transaction(uow_factory, distribution):
    with uow_factory() as uow:
        assert uow.limit_distributions.bulk_insert([distribution]) == [distribution]
    with uow_factory() as uow:
        assert uow.limit_distributions.get(distribution.key) is None


def test_batch_transfer_can_read_and_write_the_same_database(
    service, uow_factory, database, distribution, monkeypatch
):
    models = [
        replace(distribution, criterion_code=code, criterion_parameters=params, sample_size=size)
        for code in ["ks", "ad"]
        for params in [{}, {"shape": 2}]
        for size in [20, 21, 22]
    ]
    service.save_distributions(models)
    reader = LimitDistributionReader(uow_factory)
    writer = LimitDistributionWriter(service)
    write = writer.bulk_insert
    transferred = []

    def accept_batch(batch):
        # The source must release its connection even when the consumer keeps iterating.
        assert database.engine.pool.checkedout() == 0
        assert 0 < len(batch) <= 2
        transferred.extend(batch)
        return write(batch)

    monkeypatch.setattr(writer, "bulk_insert", accept_batch)
    result = LimitDistributionLoader(reader, writer).load(LimitDistributionFilter(), batch_size=2)

    assert (result.fetched_count, result.saved_count, result.skipped_count) == (12, 0, 12)
    assert transferred == sorted(
        models,
        key=lambda m: (
            criterion_code_hash(m.criterion_code),
            criterion_parameters_hash(m.criterion_parameters),
            m.sample_size,
        ),
    )
    assert database.engine.pool.checkedout() == 0
