"""Schema inference uses keys, preserving partial hypotheses and explicit schemas."""

from dataclasses import replace

import pytest

from pysatl_criterion.distribution import LogNormalDistributionDescriptor as LogNormal
from pysatl_criterion.distribution import NormalDistributionDescriptor as Normal


@pytest.mark.parametrize("fixed", [{}, {"mean": 10}, {"mean": 10, "var": 16}])
def test_single_schema_preserves_exact_fixed_parameters(fixed):
    values = Normal.parse_parameters(fixed)
    assert values.parameterization is Normal.DEFAULT
    assert values.as_dict() == fixed


@pytest.mark.parametrize(
    "fixed,schema",
    [
        ({"mu": 0, "s": 1}, LogNormal.LOG_LOCATION_SCALE),
        ({"scale": 2, "s": 1}, LogNormal.SHAPE_SCALE),
        ({"mu": 0}, LogNormal.LOG_LOCATION_SCALE),
        ({"scale": 2}, LogNormal.SHAPE_SCALE),
        ({LogNormal.MEDIAN: 2, "s": 1}, LogNormal.SHAPE_SCALE),
    ],
)
def test_unique_keys_select_schema_without_conversion(fixed, schema):
    values = LogNormal.parse_parameters(fixed)
    assert values.parameterization is schema
    assert values.as_dict() == schema.parse(fixed).as_dict()


@pytest.mark.parametrize("fixed", [{}, {"s": 1}, {LogNormal.LOG_SCALE: 1}, {"s": -1}])
def test_shared_keys_are_ambiguous_even_with_invalid_values(fixed):
    with pytest.raises(ValueError, match="Ambiguous") as error:
        LogNormal.parse_parameters(fixed)
    assert LogNormal.LOG_LOCATION_SCALE.id in str(error.value)
    assert LogNormal.SHAPE_SCALE.id in str(error.value)


@pytest.mark.parametrize(
    "fixed",
    [{"unknown": 1}, {"mu": 0, "scale": 1}, {Normal.MEAN: 0}],
)
def test_unknown_mixed_and_foreign_keys_are_rejected(fixed):
    with pytest.raises(ValueError, match="No matching") as error:
        LogNormal.parse_parameters(fixed)
    assert LogNormal.LOG_LOCATION_SCALE.id in str(error.value)
    assert LogNormal.SHAPE_SCALE.id in str(error.value)


def test_invalid_values_propagate_parser_error():
    with pytest.raises(ValueError, match="Invalid value for s"):
        LogNormal.parse_parameters({"mu": 0, "s": -1})


def test_aliases_and_duplicate_keys_use_schema_parser(monkeypatch):
    mean = replace(Normal.MEAN, aliases=("location",))
    monkeypatch.setattr(
        Normal, "DEFAULT", replace(Normal.DEFAULT, parameters=(mean, Normal.VARIANCE))
    )
    assert Normal.parse_parameters({"location": 10}).as_dict() == {"mean": 10}
    with pytest.raises(ValueError, match="Duplicate"):
        Normal.parse_parameters({"mean": 10, "location": 10})


@pytest.mark.parametrize("schema", LogNormal.parameterizations())
def test_explicit_values_keep_schema_even_when_keys_are_ambiguous(schema):
    values = LogNormal.parse_parameters(schema.parse({"s": 1}))
    assert values.parameterization is schema
    assert values.as_dict() == {"s": 1}


def test_foreign_explicit_schema_is_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        LogNormal.parse_parameters(Normal.DEFAULT.parse({}))
    other = replace(LogNormal.DEFAULT, id="log_normal.other")
    with pytest.raises(ValueError, match="Unsupported"):
        LogNormal.parse_parameters(other.parse({"mu": 0}))
