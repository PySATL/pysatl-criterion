from dataclasses import replace

import pytest

from pysatl_criterion.persistence.models.distribution_key import DistributionKey
from pysatl_criterion.persistence.models.limit_distribution import LimitDistributionModel
from pysatl_criterion.persistence.queries.limit_distribution import (
    CriterionFilter,
    LimitDistributionFilter,
)
from pysatl_criterion.persistence.sqlalchemy.database import AlchemyDatabase
from pysatl_criterion.persistence.sqlalchemy.models.limit_distribution import (
    criterion_code_hash,
    criterion_parameters_hash,
)
from pysatl_criterion.persistence.sqlalchemy.stores.limit_distribution import (
    AlchemyLimitDistributionStorage,
)


@pytest.fixture
def storage():
    database = AlchemyDatabase("sqlite:///:memory:")
    database.init()
    try:
        with database.session_factory() as session:
            yield AlchemyLimitDistributionStorage(session)
    finally:
        database.dispose()


@pytest.fixture
def model():
    return LimitDistributionModel("test_code", {"x": -1.0}, 100, 1000, [-4.0])


def test_insert_and_get(storage, model):
    assert storage.insert(model)
    assert storage.get(model.key) == model
    assert storage.get(replace(model.key, sample_size=200)) is None
    assert storage.get(replace(model.key, criterion_parameters={"x": 2.0})) is None


@pytest.mark.parametrize("count, changed", [(500, False), (1000, False), (2000, True)])
def test_insert_keeps_best_count(storage, model, count, changed):
    assert storage.insert(model)
    candidate = replace(model, monte_carlo_count=count, results_statistics=[2.0])
    assert storage.insert(candidate) is changed
    assert storage.get(model.key) == (candidate if changed else model)


def test_parameter_order_does_not_change_identity(storage, model):
    original = replace(model, criterion_parameters={"b": 2.0, "a": 1.0})
    replacement = replace(
        original,
        criterion_parameters={"a": 1.0, "b": 2.0},
        monte_carlo_count=2000,
        results_statistics=[2.0],
    )
    storage.insert(original)
    assert storage.insert(replacement)
    retrieved = storage.get(original.key)
    assert retrieved.results_statistics == [2.0]
    assert list(retrieved.criterion_parameters) == ["a", "b"]
    assert len(storage.get_batch(LimitDistributionFilter())) == 1


@pytest.mark.parametrize("stored_value, query_value", [(1, 1.0), (1.0, 1)])
def test_numeric_parameter_representations_share_record(storage, model, stored_value, query_value):
    original = replace(model, criterion_parameters={"x": stored_value})
    storage.insert(original)
    key = DistributionKey(model.criterion_code, {"x": query_value}, model.sample_size)
    retrieved = storage.get(key)
    assert retrieved is not None
    assert isinstance(retrieved.criterion_parameters["x"], float)
    replacement = replace(original, criterion_parameters={"x": query_value}, monte_carlo_count=2000)
    assert storage.insert(replacement)
    assert storage.get(original.key).monte_carlo_count == 2000
    assert len(storage.get_batch(LimitDistributionFilter())) == 1


@pytest.mark.parametrize("counts", [(300, 100, 200), (100, 300, 200)])
def test_bulk_insert_returns_only_final_changes(storage, model, counts):
    models = [
        replace(model, monte_carlo_count=count, results_statistics=[float(count)])
        for count in counts
    ]
    saved = storage.bulk_insert(models)
    assert saved == [next(m for m in models if m.monte_carlo_count == 300)]
    assert storage.get(model.key).results_statistics == [300.0]
    assert storage.bulk_insert(models) == []


def test_bulk_insert_deduplicates_normalized_keys_and_preserves_ties(storage, model):
    models = [
        replace(model, criterion_parameters={"b": 2, "a": 1}, results_statistics=[1.0]),
        replace(model, criterion_parameters={"a": 1.0, "b": 2.0}, results_statistics=[2.0]),
    ]
    assert storage.bulk_insert(models) == [models[0]]
    assert storage.get(models[1].key).results_statistics == [1.0]


def test_bulk_insert_omits_unchanged_identities(storage, model):
    storage.insert(model)
    new = replace(model, criterion_code="new")
    assert storage.bulk_insert([model, new]) == [new]
    assert storage.bulk_insert([]) == []


def test_get_batch_preserves_parameter_and_size_variants(storage, model):
    models = [
        model,
        replace(model, criterion_parameters={"x": 2.0}),
        replace(model, sample_size=101),
    ]
    storage.bulk_insert(models)
    result = storage.get_batch(LimitDistributionFilter(criteria=[CriterionFilter("test_code")]))
    assert len(result) == 3
    assert {(m.sample_size, m.criterion_parameters["x"]) for m in result} == {
        (100, -1.0),
        (100, 2.0),
        (101, -1.0),
    }


@pytest.mark.parametrize("batch_size", [1, 3, 8, 20])
@pytest.mark.parametrize("filtered", [False, True])
def test_get_batch_traverses_the_complete_primary_key(storage, model, batch_size, filtered):
    models = [
        replace(model, criterion_code=code, criterion_parameters=params, sample_size=size)
        for code in ["ks", "ad"]
        for params in [{}, {"x": 2}]
        for size in [101, 100]
    ]
    storage.bulk_insert(models)
    criteria = [CriterionFilter("ks", {}), CriterionFilter("ad", {"x": 2})] if filtered else None
    matching = (
        [
            candidate
            for candidate in models
            if (candidate.criterion_code, candidate.criterion_parameters)
            in [("ks", {}), ("ad", {"x": 2})]
        ]
        if filtered
        else models
    )
    expected = sorted(
        matching,
        key=lambda m: (
            criterion_code_hash(m.criterion_code),
            criterion_parameters_hash(m.criterion_parameters),
            m.sample_size,
        ),
    )
    query = LimitDistributionFilter(criteria=criteria)
    for start in range(0, len(expected) + batch_size, batch_size):
        after = expected[min(start, len(expected)) - 1].key if start else None
        batch = storage.get_batch(query, batch_size=batch_size, after=after)
        assert batch == expected[start : start + batch_size]


@pytest.mark.parametrize("batch_size", [0, -1])
def test_get_batch_rejects_nonpositive_batch_size(storage, batch_size):
    with pytest.raises(ValueError, match="batch_size must be positive"):
        storage.get_batch(LimitDistributionFilter(), batch_size=batch_size)


def test_get_batch_combines_inclusive_filters(storage, model):
    models = [
        replace(model, sample_size=size, monte_carlo_count=count)
        for size, count in [(99, 2000), (100, 1000), (101, 2000), (102, 3000)]
    ] + [
        replace(model, criterion_code="other", monte_carlo_count=2000),
        replace(model, criterion_parameters={"x": 2.0}, monte_carlo_count=2000),
    ]
    storage.bulk_insert(models)
    query = LimitDistributionFilter(
        criteria=[CriterionFilter("test_code", {"x": -1})],
        min_sample_size=100,
        max_sample_size=101,
        min_monte_carlo_count=1000,
    )
    assert [m.sample_size for m in storage.get_batch(query)] == [100, 101]
    assert [
        m.sample_size for m in storage.get_batch(replace(query, min_monte_carlo_count=2000))
    ] == [101]
    assert [m.sample_size for m in storage.get_batch(query, after=model.key, batch_size=1)] == [101]


@pytest.mark.parametrize(
    "min_size, max_size, expected_sizes",
    [
        (None, None, [99, 100, 101]),
        (100, None, [100, 101]),
        (None, 100, [99, 100]),
        (100, 100, [100]),
        (100, 101, [100, 101]),
    ],
)
def test_get_batch_allows_independent_inclusive_size_bounds(
    storage, model, min_size, max_size, expected_sizes
):
    storage.bulk_insert([replace(model, sample_size=size) for size in [99, 100, 101]])
    query = LimitDistributionFilter(min_sample_size=min_size, max_sample_size=max_size)
    assert [candidate.sample_size for candidate in storage.get_batch(query)] == expected_sizes


@pytest.mark.parametrize(
    "criteria, expected",
    [
        (None, {("test_code", -1), ("test_code", None), ("other", -1), ("other", None)}),
        ([], set()),
        ([CriterionFilter("test_code")], {("test_code", -1), ("test_code", None)}),
        ([CriterionFilter("test_code", {})], {("test_code", None)}),
        (
            [CriterionFilter("test_code"), CriterionFilter("other", {})],
            {("test_code", -1), ("test_code", None), ("other", None)},
        ),
    ],
)
def test_get_batch_distinguishes_empty_filters_from_unrestricted(
    storage, model, criteria, expected
):
    storage.bulk_insert(
        [
            replace(model, criterion_code=code, criterion_parameters=parameters)
            for code in ["test_code", "other"]
            for parameters in [{"x": -1}, {}]
        ]
    )
    result = storage.get_batch(LimitDistributionFilter(criteria=criteria))
    assert {
        (candidate.criterion_code, candidate.criterion_parameters.get("x")) for candidate in result
    } == expected


@pytest.mark.parametrize(
    "criteria, expected",
    [
        (
            [CriterionFilter("test_code", {"x": -1}), CriterionFilter("other", {"x": 2})],
            {("test_code", -1), ("other", 2)},
        ),
        (
            [CriterionFilter("test_code", {"x": -1}), CriterionFilter("test_code", {"x": 2})],
            {("test_code", -1), ("test_code", 2)},
        ),
    ],
)
def test_get_batch_binds_each_criterion_code_to_its_parameters(storage, model, criteria, expected):
    storage.bulk_insert(
        [
            replace(model, criterion_code=code, criterion_parameters={"x": value})
            for code in ["test_code", "other"]
            for value in [-1, 2, 3]
        ]
    )
    result = storage.get_batch(LimitDistributionFilter(criteria=criteria))
    assert {
        (candidate.criterion_code, candidate.criterion_parameters["x"]) for candidate in result
    } == expected


def test_get_batch_matches_normalized_parameters(storage, model):
    model = replace(model, criterion_parameters={"b": 2, "a": 1.0})
    storage.insert(model)
    assert storage.get_batch(
        LimitDistributionFilter(criteria=[CriterionFilter("test_code", {"a": 1, "b": 2.0})])
    )
    assert not storage.get_batch(
        LimitDistributionFilter(criteria=[CriterionFilter("test_code", {"a": 1})])
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_sample_size": 0},
        {"max_sample_size": -1},
        {"min_monte_carlo_count": 0},
        {"min_sample_size": 200, "max_sample_size": 100},
    ],
)
def test_filter_validates_bounds(kwargs):
    with pytest.raises(ValueError):
        LimitDistributionFilter(**kwargs)


def test_bulk_insert_validates_before_writing(storage, model):
    with pytest.raises(ValueError, match="monte_carlo_count"):
        storage.bulk_insert([model, replace(model, criterion_code="invalid", monte_carlo_count=0)])
    assert storage.get_batch(LimitDistributionFilter()) == []
