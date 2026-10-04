from dataclasses import replace
from decimal import Decimal

import pytest
from sqlalchemy.exc import ArgumentError, OperationalError

from pysatl_criterion.hypothesis_testing.distribution_service import DistributionService
from pysatl_criterion.loader import (
    AbstractLoader,
    CriticalValueLoader,
    CriticalValueReader,
    DistributionLoader,
    LimitDistributionReader,
)
from pysatl_criterion.persistence.queries.critical_value import (
    CriticalValueFilter,
    CriticalValueLookupQuery,
)
from pysatl_criterion.persistence.queries.limit_distribution import (
    CriterionFilter,
    LimitDistributionFilter,
)
from pysatl_criterion.persistence.sqlalchemy.database import AlchemyDatabase
from pysatl_criterion.persistence.sqlalchemy.stores.critical_value import (
    AlchemyCriticalValueStorage,
)
from pysatl_criterion.persistence.sqlalchemy.unit_of_work import AlchemyUnitOfWork
from pysatl_criterion.statistics.alternative import AlternativeType
from pysatl_criterion.utils import constants


@pytest.fixture
def loader(setup):
    return DistributionLoader(
        remote=setup.remote_database,
        local=setup.local_database,
    )


def test_base_loader_is_abstract(setup):
    with pytest.raises(TypeError, match="abstract"):
        AbstractLoader(setup.remote_reader, setup.local_writer)


def test_facade_finishes_distributions_before_reading_critical_values(
    setup, loader, model, monkeypatch, capsys
):
    models = [replace(model, sample_size=size) for size in range(100, 105)]
    setup.remote_service.save_distributions(models)
    values = [
        setup.remote_service.get_critical_value(m.key, Decimal("0.05"), AlternativeType.RIGHT)
        for m in models
    ]
    setup.local_service.save_distribution(replace(model, monte_carlo_count=500))
    setup.local_service.get_critical_value(model.key, Decimal("0.1"), AlternativeType.LEFT)
    read_values = CriticalValueReader.iter_batches
    observed_batch_sizes = []

    def read_after_distributions(reader, query, *, batch_size):
        # All distribution batches must be committed before the next stage begins.
        assert [setup.local_reader.get(m.key) for m in models] == models
        observed_batch_sizes.append(batch_size)
        return read_values(reader, query, batch_size=batch_size)

    monkeypatch.setattr(CriticalValueReader, "iter_batches", read_after_distributions)
    result = loader.load(LimitDistributionFilter(), batch_size=2)

    assert observed_batch_sizes == [2]
    assert result.limit_distributions.saved_count == 5
    assert result.critical_values.saved_count == 5
    assert (result.fetched_count, result.saved_count, result.skipped_count) == (10, 10, 0)
    assert capsys.readouterr().out == (
        "Loaded 10 records (limit distributions: 5, critical values: 5; fetched: 10, skipped: 0).\n"
    )
    with setup.local_uow() as uow:
        stored = uow.critical_values.get_batch(CriticalValueFilter())
    assert stored == values


def test_unchanged_distributions_still_load_missing_critical_values(setup, loader, model, capsys):
    setup.remote_service.save_distribution(model)
    setup.local_service.save_distribution(model)
    value = setup.remote_service.get_critical_value(
        model.key, Decimal("0.05"), AlternativeType.RIGHT
    )

    result = loader.load(LimitDistributionFilter())

    assert result.limit_distributions.saved_count == 0
    assert result.critical_values.saved_count == 1
    assert (result.fetched_count, result.saved_count, result.skipped_count) == (2, 1, 1)
    query = CriticalValueLookupQuery(model.key, 1000, Decimal("0.05"), AlternativeType.RIGHT)
    assert CriticalValueReader(setup.local_uow).get(query) == value
    assert capsys.readouterr().out == (
        "Loaded 1 record (limit distributions: 0, critical values: 1; fetched: 2, skipped: 1).\n"
    )
    repeated = loader.load(LimitDistributionFilter())
    assert (repeated.fetched_count, repeated.saved_count, repeated.skipped_count) == (2, 0, 2)
    assert capsys.readouterr().out == (
        "Loaded 0 records (limit distributions: 0, critical values: 0; fetched: 2, skipped: 2).\n"
    )


@pytest.mark.parametrize("filter_type", [LimitDistributionFilter, CriticalValueFilter])
def test_facade_applies_distribution_filters_to_both_stages(setup, loader, model, filter_type):
    selected = replace(model, monte_carlo_count=2000)
    models = [
        selected,
        replace(selected, sample_size=99),
        replace(selected, sample_size=101, monte_carlo_count=1000),
        replace(selected, sample_size=102),
        replace(selected, criterion_code="other"),
        replace(selected, criterion_parameters={"shape": 2}),
    ]
    setup.remote_service.save_distributions(models)
    for distribution in models:
        for level in (Decimal("0.05"), Decimal("0.1")):
            for alternative in (AlternativeType.LEFT, AlternativeType.RIGHT):
                setup.remote_service.get_critical_value(distribution.key, level, alternative)
    options = {}
    expected_values = 4
    if filter_type is CriticalValueFilter:
        options = {
            "significance_levels": [Decimal("0.05")],
            "alternative_types": [AlternativeType.RIGHT],
        }
        expected_values = 1
    query = filter_type(
        criteria=[CriterionFilter("ks", {}), CriterionFilter("missing", {})],
        min_sample_size=100,
        max_sample_size=101,
        min_monte_carlo_count=2000,
        **options,
    )

    result = loader.load(query, batch_size=1)

    assert result.limit_distributions.fetched_count == result.limit_distributions.saved_count == 1
    assert result.critical_values.fetched_count == expected_values
    assert result.critical_values.saved_count == expected_values
    assert result.limit_distributions.not_found_codes == ["missing"]
    assert result.critical_values.not_found_codes == ["missing"]
    assert list(setup.local_reader.iter_batches(LimitDistributionFilter())) == [[selected]]
    with setup.local_uow() as uow:
        values = uow.critical_values.get_batch(CriticalValueFilter())
    assert len(values) == expected_values
    assert all(value.distribution_key == selected.key for value in values)
    if filter_type is CriticalValueFilter:
        assert values[0].significance_level == Decimal("0.05")
        assert values[0].alternative_type == AlternativeType.RIGHT


@pytest.mark.parametrize("filter_type", [LimitDistributionFilter, CriticalValueFilter])
def test_facade_preserves_code_and_parameter_pairs_across_batches(
    setup, loader, model, filter_type
):
    selected = [
        replace(model, criterion_code="ks", criterion_parameters={"shape": 2}),
        replace(model, criterion_code="ad", criterion_parameters={"shape": 3}),
        replace(model, criterion_code="ks", criterion_parameters={"shape": 4}),
    ]
    excluded = [
        replace(model, criterion_code="ks", criterion_parameters={"shape": 3}),
        replace(model, criterion_code="ad", criterion_parameters={"shape": 2}),
        replace(model, criterion_code="ad", criterion_parameters={"shape": 4}),
        model,
        replace(model, criterion_code="ad"),
    ]
    setup.remote_service.save_distributions(selected + excluded)
    expected_values = []
    for distribution in selected + excluded:
        value = setup.remote_service.get_critical_value(
            distribution.key, Decimal("0.05"), AlternativeType.RIGHT
        )
        if distribution in selected:
            expected_values.append(value)
    query = filter_type(
        criteria=[
            CriterionFilter("ks", {"shape": 2}),
            CriterionFilter("ad", {"shape": 3}),
            CriterionFilter("ks", {"shape": 4}),
        ]
    )

    result = loader.load(query, batch_size=1)

    assert result.limit_distributions.fetched_count == result.limit_distributions.saved_count == 3
    assert result.critical_values.fetched_count == result.critical_values.saved_count == 3
    assert result.limit_distributions.not_found_codes == []
    assert result.critical_values.not_found_codes == []
    assert [setup.local_reader.get(distribution.key) for distribution in selected] == selected
    assert all(setup.local_reader.get(distribution.key) is None for distribution in excluded)
    with setup.local_uow() as uow:
        values = uow.critical_values.get_batch(CriticalValueFilter())
    assert len(values) == len(expected_values)
    assert all(value in values for value in expected_values)


def test_distribution_error_prevents_critical_value_stage(setup, loader, model, mocker, capsys):
    def source_batches(reader, query, *, batch_size):
        yield [model]
        raise ConnectionError("distribution source failed")

    mocker.patch.object(LimitDistributionReader, "iter_batches", source_batches)
    critical_load = mocker.spy(CriticalValueLoader, "load")

    with pytest.raises(ConnectionError, match="distribution source failed"):
        loader.load(LimitDistributionFilter(), batch_size=1)

    critical_load.assert_not_called()
    assert setup.local_reader.get(model.key) == model
    assert capsys.readouterr().out == ""


def test_critical_value_error_keeps_distributions_and_prior_batches(
    setup, loader, model, monkeypatch, capsys
):
    models = [model, replace(model, sample_size=101)]
    setup.remote_service.save_distributions(models)
    values = [
        setup.remote_service.get_critical_value(m.key, Decimal("0.05"), AlternativeType.RIGHT)
        for m in models
    ]
    insert = AlchemyCriticalValueStorage.insert

    def fail_after_insert(store, value):
        changed = insert(store, value)
        if value.distribution_key == models[1].key:
            raise RuntimeError("critical value write failed")
        return changed

    monkeypatch.setattr(AlchemyCriticalValueStorage, "insert", fail_after_insert)

    with pytest.raises(RuntimeError, match="critical value write failed"):
        loader.load(LimitDistributionFilter(), batch_size=1)

    assert [setup.local_reader.get(m.key) for m in models] == models
    with setup.local_uow() as uow:
        assert uow.critical_values.get_batch(CriticalValueFilter()) == values[:1]
    assert capsys.readouterr().out == ""


def test_empty_transfer_reports_zero_once(loader, capsys):
    result = loader.load(
        LimitDistributionFilter(criteria=[CriterionFilter("missing"), CriterionFilter("missing")])
    )

    assert result.fetched_count == result.saved_count == result.skipped_count == 0
    assert result.limit_distributions.not_found_codes == ["missing"]
    assert result.critical_values.not_found_codes == ["missing"]
    assert capsys.readouterr().out == (
        "Loaded 0 records (limit distributions: 0, critical values: 0; fetched: 0, skipped: 0).\n"
    )


@pytest.mark.parametrize("batch_size", [0, -1])
def test_invalid_batch_size_opens_no_transactions(mocker, capsys, batch_size):
    database = mocker.patch("pysatl_criterion.loader.distribution_loader.AlchemyDatabase")
    loader = DistributionLoader(remote="sqlite:///:memory:", local="sqlite:///:memory:")

    with pytest.raises(ValueError, match="batch_size must be positive"):
        loader.load(LimitDistributionFilter(), batch_size=batch_size)

    database.assert_not_called()
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("use_default_remote", [False, True])
def test_url_connections_create_local_tables_and_release_resources(
    tmp_path, model, mocker, monkeypatch, use_default_remote
):
    remote_url = f"sqlite:///{tmp_path / 'remote.db'}"
    local_url = f"sqlite:///{tmp_path / 'local.db'}"
    remote = AlchemyDatabase(remote_url)
    try:
        remote.init()
        service = DistributionService(lambda: AlchemyUnitOfWork(remote.session_factory))
        service.save_distribution(model)
        value = service.get_critical_value(model.key, Decimal("0.05"), AlternativeType.RIGHT)
    finally:
        remote.dispose()
    initialize = mocker.spy(AlchemyDatabase, "init")
    dispose = mocker.spy(AlchemyDatabase, "dispose")
    monkeypatch.setattr(
        constants, "REMOTE_PYSATL_URL", remote_url if use_default_remote else "invalid-url"
    )
    loader = (
        DistributionLoader(local=local_url)
        if use_default_remote
        else DistributionLoader(remote=remote_url, local=local_url)
    )
    initialize.assert_not_called()

    result = loader.load()
    repeated = loader.load()

    assert result.limit_distributions.saved_count == result.critical_values.saved_count == 1
    assert (repeated.fetched_count, repeated.saved_count, repeated.skipped_count) == (2, 0, 2)
    assert initialize.call_count == 2
    assert all(str(call.args[0].engine.url) == local_url for call in initialize.call_args_list)
    assert dispose.call_count == 4
    local = AlchemyDatabase(local_url)
    try:
        with AlchemyUnitOfWork(local.session_factory) as uow:
            assert uow.limit_distributions.get(model.key) == model
            assert uow.critical_values.get_batch(CriticalValueFilter()) == [value]
    finally:
        local.dispose()


@pytest.mark.parametrize(
    "local_url, error, disposed_count",
    [
        ("sqlite:///:memory:", OperationalError, 2),
        ("invalid-url", ArgumentError, 1),
    ],
)
def test_url_connections_are_released_on_failure(mocker, capsys, local_url, error, disposed_count):
    # The first case reaches an empty remote database; the second fails while opening local.
    dispose = mocker.spy(AlchemyDatabase, "dispose")
    loader = DistributionLoader(remote="sqlite:///:memory:", local=local_url)

    with pytest.raises(error):
        loader.load(LimitDistributionFilter())

    assert dispose.call_count == disposed_count
    assert capsys.readouterr().out == ""
