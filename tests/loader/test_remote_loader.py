from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import pytest
from sqlalchemy import delete, event

from pysatl_criterion.hypothesis_testing.distribution_service import DistributionService
from pysatl_criterion.loader.remote_loader import RemoteLoader
from pysatl_criterion.loader.store_adapters import (
    CriticalValueReader,
    CriticalValueWriter,
    LimitDistributionReader,
    LimitDistributionWriter,
)
from pysatl_criterion.persistence.models.limit_distribution import LimitDistributionModel
from pysatl_criterion.persistence.queries.critical_value import (
    CriticalValueFilter,
    CriticalValueLookupQuery,
)
from pysatl_criterion.persistence.queries.limit_distribution import LimitDistributionFilter
from pysatl_criterion.persistence.sqlalchemy.database import AlchemyDatabase
from pysatl_criterion.persistence.sqlalchemy.models.limit_distribution import (
    LimitDistributionORM,
    criterion_code_hash,
    criterion_parameters_hash,
)
from pysatl_criterion.persistence.sqlalchemy.stores.critical_value import (
    AlchemyCriticalValueStorage,
)
from pysatl_criterion.persistence.sqlalchemy.unit_of_work import AlchemyUnitOfWork
from pysatl_criterion.persistence.stores.base import IBulkStoreReader, IBulkStoreWriter
from pysatl_criterion.statistics.alternative import AlternativeType


@pytest.fixture
def setup():
    local = AlchemyDatabase("sqlite:///:memory:")
    remote = AlchemyDatabase("sqlite:///:memory:")
    local.init()
    remote.init()
    local_uow = lambda: AlchemyUnitOfWork(local.session_factory)  # noqa: E731
    remote_uow = lambda: AlchemyUnitOfWork(remote.session_factory)  # noqa: E731
    local_service = DistributionService(local_uow)
    remote_service = DistributionService(remote_uow)
    local_reader = LimitDistributionReader(local_uow)
    remote_reader = LimitDistributionReader(remote_uow)
    local_writer = LimitDistributionWriter(local_service)
    yield SimpleNamespace(
        local_reader=local_reader,
        remote_reader=remote_reader,
        local_service=local_service,
        remote_service=remote_service,
        local_uow=local_uow,
        remote_uow=remote_uow,
        local_database=local,
        remote_database=remote,
        local_writer=local_writer,
        loader=RemoteLoader(remote_reader, local_writer),
    )
    local.dispose()
    remote.dispose()


@pytest.fixture
def model():
    return LimitDistributionModel("ks", {}, 100, 1000, [1.0, 2.0])


def test_bulk_load_all_variants_in_batches(setup, model, mocker):
    models = [
        model,
        replace(model, sample_size=101),
        replace(model, criterion_parameters={"shape": 2.0}),
        replace(model, criterion_code="ad"),
        replace(model, criterion_code="ad", sample_size=101),
    ]
    setup.remote_service.save_distributions(models)
    read = mocker.spy(setup.remote_reader, "iter_batches")
    write = mocker.spy(setup.local_writer, "bulk_insert")
    query = LimitDistributionFilter(
        criterion_codes=["ks", "ad", "missing"], min_sample_size=100, max_sample_size=101
    )
    result = setup.loader.load(query, batch_size=2)
    assert (result.fetched_count, result.saved_count, result.skipped_count) == (5, 5, 0)
    assert result.not_found_codes == ["missing"]
    read.assert_called_once_with(query, batch_size=2)
    assert [len(call.args[0]) for call in write.call_args_list] == [2, 2, 1]
    for expected in models:
        assert setup.local_reader.get(expected.key) == expected


def test_batch_transactions_do_not_overlap(setup, model):
    setup.remote_service.save_distributions(
        [replace(model, sample_size=size) for size in range(100, 105)]
    )
    transactions = []
    for name, database in [
        ("source", setup.remote_database),
        ("destination", setup.local_database),
    ]:
        for action in ("begin", "rollback", "commit"):
            event.listen(
                database.engine,
                action,
                lambda connection, name=name, action=action: transactions.append((name, action)),
            )

    result = setup.loader.load(LimitDistributionFilter(), batch_size=2)

    assert result.saved_count == 5
    assert (
        transactions
        == [
            ("source", "begin"),
            ("source", "rollback"),
            ("destination", "begin"),
            ("destination", "commit"),
        ]
        * 3
    )


def test_source_batches_are_lazy_and_release_sessions_before_yield(setup, model):
    setup.remote_service.save_distributions([model, replace(model, sample_size=101)])
    units_of_work = []

    def source_uow():
        uow = setup.remote_uow()
        units_of_work.append(uow)
        return uow

    batches = LimitDistributionReader(source_uow).iter_batches(
        LimitDistributionFilter(), batch_size=1
    )
    assert units_of_work == []
    assert next(batches) == [model]
    assert len(units_of_work) == 1
    assert units_of_work[0]._session is None
    assert next(batches) == [replace(model, sample_size=101)]
    assert len(units_of_work) == 2
    assert units_of_work[1]._session is None
    with pytest.raises(StopIteration):
        next(batches)
    assert all(uow._session is None for uow in units_of_work)


def test_source_iteration_continues_after_deleted_cursor(setup, model):
    setup.remote_service.save_distributions(
        [replace(model, sample_size=size) for size in (100, 102, 104, 106)]
    )
    batches = setup.remote_reader.iter_batches(LimitDistributionFilter(), batch_size=2)
    assert [m.sample_size for m in next(batches)] == [100, 102]

    # Remove the cursor and earlier rows, then insert on both sides of the cursor.
    with setup.remote_database.session_factory() as session:
        session.execute(delete(LimitDistributionORM).where(LimitDistributionORM.sample_size <= 102))
        session.commit()
    setup.remote_service.save_distributions(
        [replace(model, sample_size=size) for size in (101, 103)]
    )

    assert [[m.sample_size for m in batch] for batch in batches] == [[103, 104], [106]]


@pytest.mark.parametrize("batch_size", [0, -1])
def test_nonpositive_batch_size_fails_before_io(setup, mocker, batch_size):
    read = mocker.spy(setup.remote_reader, "iter_batches")
    write = mocker.spy(setup.local_writer, "bulk_insert")
    with pytest.raises(ValueError, match="batch_size must be positive"):
        setup.loader.load(LimitDistributionFilter(), batch_size=batch_size)
    read.assert_not_called()
    write.assert_not_called()

    factory = mocker.Mock()
    with pytest.raises(ValueError, match="batch_size must be positive"):
        next(
            LimitDistributionReader(factory).iter_batches(
                LimitDistributionFilter(), batch_size=batch_size
            )
        )
    factory.assert_not_called()


def test_load_upgrades_existing_local_distribution(setup, model):
    setup.local_service.save_distribution(model)
    improved = replace(model, monte_carlo_count=2000, results_statistics=[3.0])
    setup.remote_service.save_distribution(improved)
    result = setup.loader.load(LimitDistributionFilter(criterion_codes=["ks"]))
    assert result.saved_count == 1
    assert setup.local_reader.get(model.key) == improved


@pytest.mark.parametrize("remote_count", [500, 1000])
def test_load_keeps_equal_or_better_local_distribution(setup, model, remote_count):
    setup.local_service.save_distribution(model)
    setup.remote_service.save_distribution(replace(model, monte_carlo_count=remote_count))
    result = setup.loader.load(LimitDistributionFilter())
    assert (result.fetched_count, result.saved_count, result.skipped_count) == (1, 0, 1)
    assert setup.local_reader.get(model.key) == model


def test_bulk_filter_controls_loaded_parameters_sizes_and_quality(setup, model):
    models = [
        replace(model, criterion_parameters={"shape": 2.0}, sample_size=101),
        replace(
            model, criterion_parameters={"shape": 2.0}, sample_size=102, monte_carlo_count=2000
        ),
        replace(
            model, criterion_parameters={"shape": 3.0}, sample_size=102, monte_carlo_count=2000
        ),
        replace(
            model, criterion_parameters={"shape": 2.0}, sample_size=103, monte_carlo_count=2000
        ),
    ]
    setup.remote_service.save_distributions(models)
    result = setup.loader.load(
        LimitDistributionFilter(
            criterion_parameters={"shape": 2},
            min_sample_size=101,
            max_sample_size=102,
            min_monte_carlo_count=2000,
        )
    )
    assert result.saved_count == 1
    assert list(setup.local_reader.iter_batches(LimitDistributionFilter())) == [[models[1]]]


def test_empty_source_does_not_write(setup, mocker, capsys):
    write = mocker.spy(setup.local_writer, "bulk_insert")
    result = setup.loader.load(LimitDistributionFilter(criterion_codes=["missing", "missing"]))
    assert result.fetched_count == result.saved_count == 0
    assert result.not_found_codes == ["missing"]
    write.assert_not_called()
    assert capsys.readouterr().out == "Loaded 0 records (fetched: 0, skipped: 0).\n"


def test_destination_writer_invalidates_critical_values(setup, model):
    setup.local_service.save_distribution(model)
    cached = setup.local_service.get_critical_value(
        model.key, Decimal("0.05"), AlternativeType.RIGHT
    )
    setup.remote_service.save_distribution(replace(model, monte_carlo_count=2000))
    setup.loader.load(LimitDistributionFilter())
    with setup.local_uow() as uow:
        assert (
            uow.critical_values.get(
                CriticalValueLookupQuery(
                    model.key,
                    cached.monte_carlo_count,
                    cached.significance_level,
                    cached.alternative_type,
                )
            )
            is None
        )
    recalculated = setup.local_service.get_critical_value(
        model.key, Decimal("0.05"), AlternativeType.RIGHT
    )
    assert recalculated.monte_carlo_count == 2000


def test_loader_needs_only_read_and_write_capabilities(model, mocker):
    source = mocker.Mock(spec=IBulkStoreReader)
    destination = mocker.Mock(spec=IBulkStoreWriter)
    source.iter_batches.return_value = iter([[model]])
    destination.bulk_insert.return_value = [model]
    query = LimitDistributionFilter()
    result = RemoteLoader(source, destination).load(query)
    assert result.saved_count == 1
    source.iter_batches.assert_called_once_with(query, batch_size=100)
    source.get.assert_not_called()
    destination.bulk_insert.assert_called_once_with([model])
    assert not hasattr(source, "insert")
    assert not hasattr(destination, "get")


def test_loader_writes_before_requesting_next_batch_and_combines_results(model, mocker, capsys):
    source = mocker.Mock(spec=IBulkStoreReader)
    destination = mocker.Mock(spec=IBulkStoreWriter)
    other = replace(model, criterion_code="ad")

    def batches():
        yield [model]
        destination.bulk_insert.assert_called_once_with([model])
        yield [other]

    source.iter_batches.return_value = batches()
    destination.bulk_insert.side_effect = [[], [other]]
    query = LimitDistributionFilter(criterion_codes=["missing", "ks", "ad", "missing"])

    result = RemoteLoader(source, destination).load(query, batch_size=1)

    assert (result.fetched_count, result.saved_count, result.skipped_count) == (2, 1, 1)
    assert result.not_found_codes == ["missing"]
    source.iter_batches.assert_called_once_with(query, batch_size=1)
    assert destination.bulk_insert.call_args_list == [mocker.call([model]), mocker.call([other])]
    assert capsys.readouterr().out == "Loaded 1 record (fetched: 2, skipped: 1).\n"


def test_failed_batch_rolls_back_with_cache_and_keeps_prior_commits(setup, model, mocker):
    models = [replace(model, sample_size=size) for size in range(100, 105)]
    improved = [replace(m, monte_carlo_count=2000) for m in models]
    setup.local_service.save_distributions(models)
    setup.remote_service.save_distributions(improved)
    cached = [
        setup.local_service.get_critical_value(m.key, Decimal("0.05"), AlternativeType.RIGHT)
        for m in models
    ]
    invalidate = AlchemyCriticalValueStorage.delete_for_distribution

    def fail_after_invalidation(storage, key):
        invalidate(storage, key)
        if key == models[3].key:
            raise RuntimeError("batch failed")

    mocker.patch.object(
        AlchemyCriticalValueStorage,
        "delete_for_distribution",
        autospec=True,
        side_effect=fail_after_invalidation,
    )
    write = mocker.spy(setup.local_writer, "bulk_insert")

    with pytest.raises(RuntimeError, match="batch failed"):
        setup.loader.load(LimitDistributionFilter(), batch_size=2)

    assert write.call_count == 2
    with setup.local_uow() as uow:
        for i, old in enumerate(cached):
            assert uow.limit_distributions.get(models[i].key) == (
                improved[i] if i < 2 else models[i]
            )
            value = uow.critical_values.get(
                CriticalValueLookupQuery(
                    old.distribution_key,
                    old.monte_carlo_count,
                    old.significance_level,
                    old.alternative_type,
                )
            )
            assert value == (None if i < 2 else old)


def test_later_source_error_keeps_saved_batches(setup, model, mocker):
    def batches():
        yield [model]
        raise ConnectionError("source unavailable")

    source = mocker.Mock(spec=IBulkStoreReader)
    source.iter_batches.return_value = batches()
    write = mocker.spy(setup.local_writer, "bulk_insert")
    loader = RemoteLoader(source, setup.local_writer)

    with pytest.raises(ConnectionError, match="source unavailable"):
        loader.load(LimitDistributionFilter(), batch_size=1)

    write.assert_called_once_with([model])
    assert setup.local_reader.get(model.key) == model


def test_write_error_is_not_reported_as_success(model, mocker, capsys):
    source = mocker.Mock(spec=IBulkStoreReader)
    source.iter_batches.return_value = iter([[model]])
    destination = mocker.Mock(spec=IBulkStoreWriter)
    destination.bulk_insert.side_effect = RuntimeError("write failed")
    with pytest.raises(RuntimeError, match="write failed"):
        RemoteLoader(source, destination).load(LimitDistributionFilter())
    assert capsys.readouterr().out == ""


def test_source_error_propagates_without_writing(mocker):
    source = mocker.Mock(spec=IBulkStoreReader)
    source.iter_batches.side_effect = ConnectionError("source unavailable")
    destination = mocker.Mock(spec=IBulkStoreWriter)
    with pytest.raises(ConnectionError, match="source unavailable"):
        RemoteLoader(source, destination).load(LimitDistributionFilter())
    destination.bulk_insert.assert_not_called()


def test_load_critical_values_across_the_complete_primary_key(setup, model, capsys):
    distributions = [
        replace(model, criterion_code=code, criterion_parameters=params, sample_size=size)
        for code in ("ks", "ad")
        for params in ({}, {"shape": 2})
        for size in (100, 101)
    ]
    setup.remote_service.save_distributions(distributions)
    values = [
        setup.remote_service.get_critical_value(distribution.key, level, alternative)
        for distribution in distributions
        for level in (Decimal("0.05"), Decimal("0.1"))
        for alternative in AlternativeType
    ]
    # The same loader first supplies the local parent distributions.
    setup.loader.load(LimitDistributionFilter(), batch_size=3)
    source = CriticalValueReader(setup.remote_uow)
    destination = CriticalValueWriter(setup.local_uow)
    loader = RemoteLoader(source, destination)
    query = CriticalValueFilter(criterion_codes=["missing", "ad", "ks", "missing"])
    capsys.readouterr()

    result = loader.load(query, batch_size=5)

    assert (result.fetched_count, result.saved_count, result.skipped_count) == (48, 48, 0)
    assert result.not_found_codes == ["missing"]
    assert capsys.readouterr().out == "Loaded 48 records (fetched: 48, skipped: 0).\n"
    reader = CriticalValueReader(setup.local_uow)
    copied = [value for batch in reader.iter_batches(query, batch_size=5) for value in batch]
    expected = sorted(
        values,
        key=lambda v: (
            criterion_code_hash(v.distribution_key.criterion_code),
            criterion_parameters_hash(v.distribution_key.criterion_parameters),
            v.distribution_key.sample_size,
            v.significance_level,
            v.alternative_type.value,
        ),
    )
    assert copied == expected
    result = loader.load(query, batch_size=5)
    assert (result.fetched_count, result.saved_count, result.skipped_count) == (48, 0, 48)
    assert capsys.readouterr().out == "Loaded 0 records (fetched: 48, skipped: 48).\n"


@pytest.mark.parametrize("local_count", [None, 500, 1000, 2000])
def test_critical_value_load_requires_matching_local_distribution(setup, model, local_count):
    setup.remote_service.save_distribution(model)
    value = setup.remote_service.get_critical_value(
        model.key, Decimal("0.05"), AlternativeType.RIGHT
    )
    if local_count is not None:
        setup.local_service.save_distribution(replace(model, monte_carlo_count=local_count))
    loader = RemoteLoader(
        CriticalValueReader(setup.remote_uow), CriticalValueWriter(setup.local_uow)
    )

    result = loader.load(CriticalValueFilter())

    assert result.fetched_count == 1
    assert result.saved_count == (1 if local_count == 1000 else 0)
    assert result.not_found_codes == []
    query = CriticalValueLookupQuery(model.key, 1000, Decimal("0.05"), AlternativeType.RIGHT)
    assert CriticalValueReader(setup.local_uow).get(query) == (
        value if local_count == 1000 else None
    )


def test_critical_value_filter_combines_all_fields_and_excludes_stale_values(setup, model):
    models = [
        replace(model, criterion_parameters={"shape": 2}, sample_size=size, monte_carlo_count=count)
        for size, count in ((99, 2000), (100, 1000), (101, 2000), (102, 2000), (103, 2000))
    ] + [
        replace(model, criterion_parameters={"shape": 3}, monte_carlo_count=2000),
        replace(
            model, criterion_code="other", criterion_parameters={"shape": 2}, monte_carlo_count=2000
        ),
    ]
    setup.remote_service.save_distributions(models)
    for distribution in models:
        for level in (Decimal("0.05"), Decimal("0.1")):
            for alternative in AlternativeType:
                setup.remote_service.get_critical_value(distribution.key, level, alternative)
    # Simulate a stale cache without the service's automatic invalidation.
    with setup.remote_uow() as uow:
        uow.limit_distributions.insert(replace(models[3], monte_carlo_count=3000))
        uow.commit()
    reader = CriticalValueReader(setup.remote_uow)
    query = CriticalValueFilter(
        criterion_codes=["ks"],
        criterion_parameters={"shape": 2},
        min_sample_size=100,
        max_sample_size=102,
        min_monte_carlo_count=2000,
        significance_levels=[Decimal("0.050")],
        alternative_types=[AlternativeType.RIGHT],
    )

    values = [value for batch in reader.iter_batches(query, batch_size=1) for value in batch]

    assert len(values) == 1
    assert values[0].distribution_key == models[2].key
    assert values[0].significance_level == Decimal("0.05")
    assert values[0].alternative_type == AlternativeType.RIGHT
    for name in ("criterion_codes", "significance_levels", "alternative_types"):
        assert list(reader.iter_batches(replace(query, **{name: []}))) == []
    assert list(reader.iter_batches(replace(query, criterion_parameters={}))) == []


def test_critical_value_transfer_closes_source_sessions_and_commits_batches(setup, model):
    setup.remote_service.save_distribution(model)
    setup.local_service.save_distribution(model)
    for alternative in AlternativeType:
        setup.remote_service.get_critical_value(model.key, Decimal("0.05"), alternative)
    transactions = []
    for name, database in [
        ("source", setup.remote_database),
        ("destination", setup.local_database),
    ]:
        for action in ("begin", "rollback", "commit"):
            event.listen(
                database.engine,
                action,
                lambda connection, name=name, action=action: transactions.append((name, action)),
            )
    units_of_work = []

    def source_uow():
        uow = setup.remote_uow()
        units_of_work.append(uow)
        return uow

    reader = CriticalValueReader(source_uow)
    batches = reader.iter_batches(CriticalValueFilter(), batch_size=2)
    assert units_of_work == []
    assert len(next(batches)) == 2
    assert units_of_work[0]._session is None
    batches.close()
    transactions.clear()
    result = RemoteLoader(reader, CriticalValueWriter(setup.local_uow)).load(
        CriticalValueFilter(), batch_size=2
    )

    assert result.saved_count == 3
    assert (
        transactions
        == [
            ("source", "begin"),
            ("source", "rollback"),
            ("destination", "begin"),
            ("destination", "commit"),
        ]
        * 2
    )
    assert all(uow._session is None for uow in units_of_work)


def test_failed_critical_value_batch_keeps_prior_commits(setup, model, mocker, capsys):
    setup.remote_service.save_distribution(model)
    setup.local_service.save_distribution(model)
    for level in (Decimal("0.05"), Decimal("0.1"), Decimal("0.2")):
        setup.remote_service.get_critical_value(model.key, level, AlternativeType.RIGHT)
    insert = AlchemyCriticalValueStorage.insert

    def fail_after_insert(store, value):
        changed = insert(store, value)
        if value.significance_level == Decimal("0.1"):
            raise RuntimeError("critical value batch failed")
        return changed

    mocker.patch.object(
        AlchemyCriticalValueStorage, "insert", autospec=True, side_effect=fail_after_insert
    )
    loader = RemoteLoader(
        CriticalValueReader(setup.remote_uow), CriticalValueWriter(setup.local_uow)
    )
    with pytest.raises(RuntimeError, match="critical value batch failed"):
        loader.load(CriticalValueFilter(), batch_size=1)

    assert capsys.readouterr().out == ""
    values = [
        value
        for batch in CriticalValueReader(setup.local_uow).iter_batches(CriticalValueFilter())
        for value in batch
    ]
    assert [value.significance_level for value in values] == [Decimal("0.05")]


@pytest.mark.parametrize("batch_size", [0, -1])
def test_critical_value_readers_reject_nonpositive_batch_size(setup, mocker, batch_size):
    query = CriticalValueFilter()
    factory = mocker.Mock()
    with pytest.raises(ValueError, match="batch_size must be positive"):
        next(CriticalValueReader(factory).iter_batches(query, batch_size=batch_size))
    factory.assert_not_called()
    with setup.remote_uow() as uow:
        with pytest.raises(ValueError, match="batch_size must be positive"):
            uow.critical_values.get_batch(query, batch_size=batch_size)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"significance_levels": [Decimal("0")]},
        {"significance_levels": [Decimal("1")]},
        {"significance_levels": [Decimal("0.12345678901")]},
        {"alternative_types": ["right"]},
    ],
)
def test_critical_value_filter_rejects_invalid_options(kwargs):
    with pytest.raises(ValueError):
        CriticalValueFilter(**kwargs)
