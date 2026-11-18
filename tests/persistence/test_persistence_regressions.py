import struct
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

import pytest
from sqlalchemy import Column, ForeignKeyConstraint, MetaData, PrimaryKeyConstraint, Table, select
from sqlalchemy.exc import InvalidRequestError

from pysatl_criterion.hypothesis_testing.distribution_service import DistributionService
from pysatl_criterion.persistence.models.limit_distribution import LimitDistributionModel
from pysatl_criterion.persistence.queries.critical_value import CriticalValueFilter
from pysatl_criterion.persistence.queries.limit_distribution import LimitDistributionFilter
from pysatl_criterion.persistence.sqlalchemy.database import AlchemyDatabase
from pysatl_criterion.persistence.sqlalchemy.models.critical_value import CriticalValueORM
from pysatl_criterion.persistence.sqlalchemy.models.limit_distribution import LimitDistributionORM
from pysatl_criterion.persistence.sqlalchemy.types import CompressedFloatArray
from pysatl_criterion.persistence.sqlalchemy.unit_of_work import AlchemyUnitOfWork
from pysatl_criterion.statistics.alternative import AlternativeType


@pytest.fixture
def database(tmp_path):
    database = AlchemyDatabase(
        f"sqlite:///{tmp_path / 'regressions.db'}", connect_args={"timeout": 0.1}
    )
    database.init()
    try:
        yield database
    finally:
        database.dispose()


def test_large_and_close_statistics_preserve_float64_precision(database):
    service = DistributionService(lambda: AlchemyUnitOfWork(database.session_factory))
    values = [1.000000000000001, 1.000000000000002, 1e40, 2e40]
    model = LimitDistributionModel("precise", {}, 10, len(values), values)
    assert service.save_distribution(model)
    with AlchemyUnitOfWork(database.session_factory) as uow:
        assert uow.limit_distributions.get(model.key).results_statistics == values
    result = service.get_critical_value(model.key, Decimal(".05"), AlternativeType.RIGHT)
    assert result.upper_value is not None
    assert 1e40 <= result.upper_value <= 2e40


def test_old_float32_payload_remains_readable():
    legacy = CompressedFloatArray(use_float32=True)
    payload = legacy.process_bind_param([1.25, -2.5], None)
    assert CompressedFloatArray().process_result_value(payload, None) == [1.25, -2.5]


@pytest.mark.parametrize("values", [[float("nan")], [float("inf")], [[1.0, 2.0]]])
def test_invalid_arrays_are_rejected(values):
    with pytest.raises(ValueError):
        CompressedFloatArray().process_bind_param(values, None)


def test_explicit_float32_overflow_is_rejected():
    with pytest.raises(ValueError):
        CompressedFloatArray(use_float32=True).process_bind_param([1e40], None)


@pytest.mark.parametrize("dtype_code,length", [(99, 2), (2, 1), (2, 3)])
def test_corrupt_payload_cannot_silently_change_array(dtype_code, length):
    codec = CompressedFloatArray()
    packed = codec.process_bind_param([1.0, 2.0], None)
    corrupt = struct.pack("<BBI", codec.VERSION, dtype_code, length) + packed[6:]
    with pytest.raises(ValueError):
        codec.process_result_value(corrupt, None)


def test_concurrent_codec_operations_have_independent_contexts(monkeypatch):
    # A shared context is deterministic to detect; a real compression race can crash Python.
    import pysatl_criterion.persistence.sqlalchemy.types as types_module

    compressor = types_module.zstd.ZstdCompressor
    decompressor = types_module.zstd.ZstdDecompressor
    contexts = []

    def make_compressor():
        context = compressor()
        contexts.append(context)
        return context

    def make_decompressor():
        context = decompressor()
        contexts.append(context)
        return context

    monkeypatch.setattr(types_module.zstd, "ZstdCompressor", make_compressor)
    monkeypatch.setattr(types_module.zstd, "ZstdDecompressor", make_decompressor)
    codec = CompressedFloatArray()
    assert not contexts, "Codecs must not retain shared zstd contexts"

    def round_trip(index):
        values = [float(index)] * 1000
        local_codec = codec.copy()
        packed = local_codec.process_bind_param(values, None)
        assert local_codec.process_result_value(packed, None) == values

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(round_trip, range(20)))
    assert len(contexts) == len({id(context) for context in contexts}) == 40


def test_sqlite_read_transactions_do_not_reserve_a_writer(database):
    with database.session_factory() as first, database.session_factory() as second:
        assert first.scalar(select(1)) == 1
        assert second.scalar(select(2)) == 2


@pytest.mark.parametrize("store_name", ["limit_distributions", "critical_values"])
def test_repositories_cannot_start_transactions_after_uow_exit(database, store_name):
    uow = AlchemyUnitOfWork(database.session_factory)
    query = (
        LimitDistributionFilter() if store_name == "limit_distributions" else CriticalValueFilter()
    )
    with uow:
        repository = getattr(uow, store_name)
    with pytest.raises(InvalidRequestError, match="permanently closed"):
        repository.get_batch(query)
    # Re-entering the UoW creates a fresh session without reviving the old repository.
    with uow:
        assert getattr(uow, store_name).get_batch(query) == []
        with pytest.raises(InvalidRequestError, match="permanently closed"):
            repository.get_batch(query)


@pytest.mark.parametrize("model", [LimitDistributionORM, CriticalValueORM])
@pytest.mark.parametrize("change", ["missing", "reordered"])
def test_existing_schema_requires_complete_primary_key(tmp_path, model, change):
    database = AlchemyDatabase(f"sqlite:///{tmp_path / 'bad_schema.db'}")
    expected = model.__table__
    keys = [column.name for column in expected.primary_key]
    if change == "missing":
        keys.remove("sample_size")
    else:
        keys.reverse()
    malformed = Table(
        expected.name,
        MetaData(),
        *(Column(c.name, c.type, nullable=c.nullable) for c in expected.columns),
        PrimaryKeyConstraint(*keys),
    )
    malformed.create(database.engine)
    try:
        with pytest.raises(ValueError, match="primary key"):
            database.init()
    finally:
        database.dispose()


@pytest.mark.parametrize("change", ["no_cascade", "reordered", "wrong_parent"])
def test_existing_schema_requires_matching_cascading_foreign_key(tmp_path, change):
    database = AlchemyDatabase(f"sqlite:///{tmp_path / 'bad_fk.db'}")
    metadata = MetaData()
    parent = LimitDistributionORM.__table__.to_metadata(metadata)
    wrong_parent = parent.to_metadata(metadata, name="other_distributions")
    for index in wrong_parent.indexes:
        index.name = f"other_{index.name}"
    expected = CriticalValueORM.__table__
    keys = [column.name for column in parent.primary_key]
    if change == "reordered":
        keys.reverse()
    target = wrong_parent if change == "wrong_parent" else parent
    Table(
        expected.name,
        metadata,
        *(Column(c.name, c.type, nullable=c.nullable) for c in expected.columns),
        PrimaryKeyConstraint(*(c.name for c in expected.primary_key)),
        ForeignKeyConstraint(
            keys,
            [f"{target.name}.{name}" for name in keys],
            ondelete=None if change == "no_cascade" else "CASCADE",
        ),
    )
    metadata.create_all(database.engine)
    try:
        with pytest.raises(ValueError, match="foreign key"):
            database.init()
    finally:
        database.dispose()
