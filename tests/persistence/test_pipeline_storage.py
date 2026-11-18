import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from decimal import Decimal
from threading import Barrier, Event, get_ident

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError
from sqlalchemy.pool import NullPool

from pysatl_criterion.persistence.models.critical_value import CriticalValueModel
from pysatl_criterion.persistence.models.limit_distribution import LimitDistributionModel
from pysatl_criterion.persistence.queries.critical_value import (
    CriticalValueFilter,
    CriticalValueLookupQuery,
)
from pysatl_criterion.persistence.queries.limit_distribution import LimitDistributionFilter
from pysatl_criterion.persistence.sqlalchemy.config import StorageConfig
from pysatl_criterion.persistence.sqlalchemy.database import AlchemyDatabase
from pysatl_criterion.persistence.sqlalchemy.storage import (
    CriticalValueStorage,
    DistributionStorage,
)
from pysatl_criterion.persistence.sqlalchemy.stores.critical_value import (
    AlchemyCriticalValueStorage,
)
from pysatl_criterion.statistics.alternative import AlternativeType


@pytest.fixture
def database(tmp_path):
    database = AlchemyDatabase(f"sqlite:///{tmp_path / 'pipeline.db'}")
    database.init()
    try:
        yield database
    finally:
        database.dispose()


def model(code="example"):
    return LimitDistributionModel(code, {}, 10, 2, [1.0, 2.0])


def critical(distribution):
    return CriticalValueModel(
        distribution.key,
        distribution.monte_carlo_count,
        Decimal(".05"),
        AlternativeType.RIGHT,
        upper_value=2.0,
    )


def query(value):
    return CriticalValueLookupQuery(
        value.distribution_key,
        value.monte_carlo_count,
        value.significance_level,
        value.alternative_type,
    )


def test_thirty_workers_share_storages_but_only_one_sqlite_writer(database):
    distributions = DistributionStorage(database)
    criticals = CriticalValueStorage(database)
    barrier = Barrier(database.config.max_workers)
    writer_threads = set()

    @event.listens_for(database.engine, "before_cursor_execute")
    def record_writer(_connection, _cursor, statement, _parameters, _context, _many):
        if statement.lstrip().split()[0] in ("UPDATE", "INSERT", "DELETE"):
            writer_threads.add(get_ident())

    def pipeline(index):
        data = model(str(index))
        barrier.wait(timeout=10)
        assert distributions.get(data.key) is None
        assert distributions.insert(data)
        assert distributions.get(data.key) == data
        bounds = critical(data)
        assert criticals.insert(bounds)
        assert criticals.get(query(bounds)) == bounds
        assert distributions.insert(replace(data, monte_carlo_count=3))
        assert criticals.get(query(bounds)) is None
        assert distributions.get(data.key).monte_carlo_count == 3

    assert database.config.max_workers == 30
    with ThreadPoolExecutor(max_workers=database.config.max_workers) as executor:
        list(executor.map(pipeline, range(30)))
    assert len(writer_threads) == 1
    assert get_ident() not in writer_threads
    assert database.engine.pool.checkedout() == 0


def test_batches_release_connection_before_yield_and_support_loader(database):
    from pysatl_criterion.loader.limit_distribution_loader import LimitDistributionLoader

    storage = DistributionStorage(database)
    items = [model(str(index)) for index in range(5)]
    assert len(storage.bulk_insert(items)) == 5
    seen = []
    for batch in storage.iter_batches(LimitDistributionFilter(), batch_size=2):
        assert database.engine.pool.checkedout() == 0
        seen.extend(batch)
    assert {item.criterion_code for item in seen} == {item.criterion_code for item in items}
    result = LimitDistributionLoader(storage, storage, print_summary=False).load(
        LimitDistributionFilter(), batch_size=2
    )
    assert (result.fetched_count, result.saved_count, result.skipped_count) == (5, 0, 5)


def test_critical_batches_and_stale_writes(database):
    distributions = DistributionStorage(database)
    storage = CriticalValueStorage(database)
    data = model()
    distributions.insert(data)
    bounds = critical(data)
    assert storage.bulk_insert([bounds, bounds]) == [bounds]
    assert storage.save_if_current(bounds)
    iterator = storage.iter_batches(CriticalValueFilter(), batch_size=1)
    assert next(iterator) == [bounds]
    assert database.engine.pool.checkedout() == 0
    assert list(iterator) == []
    distributions.insert(replace(data, monte_carlo_count=3))
    assert not storage.save_if_current(bounds)
    assert not storage.insert(bounds)
    assert storage.get_batch(CriticalValueFilter()) == []
    assert storage.get(query(bounds)) is None


def test_failed_invalidation_rolls_back_and_writer_recovers(database, monkeypatch):
    distributions = DistributionStorage(database)
    criticals = CriticalValueStorage(database)
    data = model()
    distributions.insert(data)
    bounds = critical(data)
    criticals.insert(bounds)
    delete = AlchemyCriticalValueStorage.delete_for_distribution

    def fail(_self, _key):
        raise ValueError("invalidation failed")

    monkeypatch.setattr(AlchemyCriticalValueStorage, "delete_for_distribution", fail)
    with pytest.raises(ValueError, match="invalidation failed"):
        distributions.insert(replace(data, monte_carlo_count=3))
    assert distributions.get(data.key) == data
    assert criticals.get(query(bounds)) == bounds
    monkeypatch.setattr(AlchemyCriticalValueStorage, "delete_for_distribution", delete)
    assert distributions.insert(replace(data, monte_carlo_count=3))
    assert criticals.get(query(bounds)) is None
    assert database.engine.pool.checkedout() == 0


def test_transient_write_failure_retries_whole_transaction(database, monkeypatch):
    storage = DistributionStorage(database)
    data = model()
    storage.insert(data)
    delete = AlchemyCriticalValueStorage.delete_for_distribution
    sessions = []

    def fail_once(repository, key):
        sessions.append(repository._session)
        if len(sessions) == 1:
            raise OperationalError("DELETE", {}, sqlite3.OperationalError("database is locked"))
        delete(repository, key)

    monkeypatch.setattr(AlchemyCriticalValueStorage, "delete_for_distribution", fail_once)
    assert storage.insert(replace(data, monte_carlo_count=3))
    assert len(sessions) == 2
    assert sessions[0] is not sessions[1]
    assert storage.get(data.key).monte_carlo_count == 3


def test_retry_limit_preserves_original_error_and_rolls_back(database, monkeypatch):
    storage = DistributionStorage(database)
    data = model()
    calls = []

    def fail(repository, _key):
        calls.append(repository._session)
        raise OperationalError("DELETE", {}, sqlite3.OperationalError("database is locked"))

    monkeypatch.setattr(AlchemyCriticalValueStorage, "delete_for_distribution", fail)
    with pytest.raises(OperationalError, match="database is locked"):
        storage.insert(data)
    assert len(calls) == database.config.retry_attempts
    assert storage.get(data.key) is None
    assert database.engine.pool.checkedout() == 0


def test_bounded_queue_applies_backpressure_and_recovers(tmp_path, monkeypatch):
    database = AlchemyDatabase(
        f"sqlite:///{tmp_path / 'queue.db'}",
        config=StorageConfig(write_queue_size=1, write_queue_timeout=0.05),
    )
    running, queued, release = Event(), Event(), Event()
    submit = database._writer._executor.submit
    calls = []

    def record_submit(*args, **kwargs):
        future = submit(*args, **kwargs)
        calls.append(future)
        if len(calls) == 2:
            queued.set()
        return future

    monkeypatch.setattr(database._writer._executor, "submit", record_submit)

    def block():
        running.set()
        assert release.wait(timeout=5)
        return 1

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            try:
                first = executor.submit(database.run, block, write=True)
                assert running.wait(timeout=5)
                second = executor.submit(database.run, lambda: 2, write=True)
                assert queued.wait(timeout=5)
                with pytest.raises(TimeoutError, match="write queue"):
                    database.run(lambda: 3, write=True)
            finally:
                release.set()
            assert first.result(timeout=5) == 1
            assert second.result(timeout=5) == 2
        assert database.run(lambda: 4, write=True) == 4
    finally:
        release.set()
        database.dispose()


def test_pool_exhaustion_does_not_leak_connections(tmp_path):
    database = AlchemyDatabase(
        f"sqlite:///{tmp_path / 'pool.db'}", config=StorageConfig(pool_size=1, pool_timeout=0.05)
    )
    try:
        database.init()
        storage = DistributionStorage(database)
        with database.session_factory() as session:
            session.scalar(select(1))
            with pytest.raises(PoolTimeoutError):
                storage.get(model().key)
        assert storage.get(model().key) is None
        assert database.engine.pool.checkedout() == 0
    finally:
        database.dispose()


def test_dispose_waits_for_accepted_operations_and_rejects_new_ones(database):
    running, release = Event(), Event()

    def block():
        running.set()
        assert release.wait(timeout=5)
        return "committed"

    with ThreadPoolExecutor(max_workers=2) as executor:
        try:
            write = executor.submit(database.run, block, write=True)
            assert running.wait(timeout=5)
            close = executor.submit(database.dispose)
            with database._condition:
                assert database._condition.wait_for(lambda: database._closing, timeout=5)
            with pytest.raises(RuntimeError, match="shutting down"):
                DistributionStorage(database).get(model().key)
        finally:
            release.set()
        assert write.result(timeout=5) == "committed"
        close.result(timeout=5)
    with pytest.raises(RuntimeError, match="closed"):
        database.init()
    database.dispose()


def test_memory_database_works_locally_but_rejects_threaded_public_storage():
    database = AlchemyDatabase("sqlite:///:memory:")
    try:
        database.init()
        storage = DistributionStorage(database)
        data = model()
        assert storage.insert(data)
        assert storage.get(data.key) == data
        with (
            ThreadPoolExecutor(max_workers=1) as executor,
            pytest.raises(RuntimeError, match="file-backed SQLite"),
        ):
            executor.submit(storage.get, data.key).result(timeout=5)
    finally:
        database.dispose()


def test_database_cannot_be_reused_in_another_process(database, monkeypatch):
    import pysatl_criterion.persistence.sqlalchemy.database as database_module

    with monkeypatch.context() as patch:
        patch.setattr(database_module.os, "getpid", lambda: database._pid + 1)
        with pytest.raises(RuntimeError, match="worker process"):
            DistributionStorage(database).get(model().key)


@pytest.mark.parametrize(
    "options",
    [
        {"max_workers": 0},
        {"max_workers": True},
        {"pool_size": 0},
        {"pool_size": -1},
        {"max_overflow": -1},
        {"pool_timeout": 0},
        {"sqlite_timeout": float("nan")},
        {"write_queue_size": 0},
        {"write_queue_timeout": -1},
        {"retry_attempts": 0},
        {"retry_backoff": float("inf")},
        {"pool_recycle": -2},
    ],
)
def test_invalid_limits_fail_before_creating_resources(options):
    with pytest.raises((TypeError, ValueError)):
        StorageConfig(**options)


def test_backend_pool_defaults_and_explicit_engine_overrides(database):
    assert database.engine.pool.size() == 5
    server = AlchemyDatabase("postgresql+psycopg2://localhost/unused")
    try:
        assert server.engine.pool.size() == 30
        assert server.engine.pool._max_overflow == 0
    finally:
        server.dispose()


def test_existing_engine_options_override_config_and_custom_pools_work(tmp_path):
    url = f"sqlite:///{tmp_path / 'options.db'}"
    database = AlchemyDatabase(
        url,
        config=StorageConfig(pool_size=5),
        pool_size=2,
        connect_args={"timeout": 0.1},
    )
    try:
        database.init()
        assert database.engine.pool.size() == 2
    finally:
        database.dispose()
    external = create_engine(url)
    try:
        for options in ({"poolclass": NullPool}, {"pool": external.pool}):
            database = AlchemyDatabase(url, **options)
            try:
                database.init()
                data = model()
                DistributionStorage(database).insert(data)
                assert DistributionStorage(database).get(data.key) == data
            finally:
                database.dispose()
    finally:
        external.dispose()


@pytest.mark.parametrize("write", [False, True])
def test_nested_public_calls_fail_instead_of_deadlocking(database, write):
    with pytest.raises(RuntimeError, match="Cannot nest"):
        database.run(lambda: DistributionStorage(database).get(model().key), write=write)
    with pytest.raises(RuntimeError, match="Cannot dispose"):
        database.run(database.dispose, write=write)
    assert DistributionStorage(database).get(model().key) is None
