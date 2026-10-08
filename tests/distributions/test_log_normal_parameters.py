"""Explicit conversion between logarithmic and shape/median coordinates."""

import math
from dataclasses import replace

import numpy as np
import pytest
from scipy import stats

from pysatl_criterion.distribution.distributions import LogNormalDistributionDescriptor as LogNormal
from pysatl_criterion.distribution.distributions import NormalDistributionDescriptor as Normal
from pysatl_criterion.statistics.goodness_of_fit.log_normal import (
    KolmogorovSmirnovLogNormalGofStatistic as KS,
)
from pysatl_criterion.utils.generator import get_hypothesis_generator


def test_coordinate_meanings_and_shared_identity():
    assert LogNormal.DEFAULT is LogNormal.LOG_LOCATION_SCALE
    assert LogNormal.LOG_LOCATION != LogNormal.MEDIAN
    assert LogNormal.LOG_SCALE in LogNormal.LOG_LOCATION_SCALE.parameters
    assert LogNormal.LOG_SCALE in LogNormal.SHAPE_SCALE.parameters
    assert LogNormal.parameterizations() == (LogNormal.LOG_LOCATION_SCALE, LogNormal.SHAPE_SCALE)


@pytest.mark.parametrize("fixed", [{}, {"mu": 1.25}, {"s": 0.7}, {"mu": 1.25, "s": 0.7}])
def test_conversion_preserves_exact_set_of_fixed_parameters(fixed):
    original = LogNormal.LOG_LOCATION_SCALE.parse(fixed)
    converted = LogNormal.convert_parameters(original, LogNormal.SHAPE_SCALE)
    expected = {
        "scale" if key == "mu" else key: math.exp(value) if key == "mu" else value
        for key, value in fixed.items()
    }
    assert converted.as_dict() == expected
    assert LogNormal.convert_parameters(
        converted, LogNormal.LOG_LOCATION_SCALE
    ).as_dict() == pytest.approx(fixed)
    assert original.as_dict() == fixed


@pytest.mark.parametrize("median", [np.nextafter(0.0, 1.0), 0.25, 1, 4.5, np.finfo(float).max])
def test_finite_positive_median_can_be_converted_to_log_location(median):
    values = LogNormal.SHAPE_SCALE.parse({LogNormal.MEDIAN: median})
    converted = LogNormal.convert_parameters(values, LogNormal.LOG_LOCATION_SCALE)
    assert converted[LogNormal.LOG_LOCATION] == pytest.approx(math.log(median))
    assert LogNormal.LOG_SCALE not in converted


@pytest.mark.parametrize("mu", [-1000, 1000])
def test_unrepresentable_median_is_rejected(mu):
    values = LogNormal.LOG_LOCATION_SCALE.parse({"mu": mu})
    with pytest.raises(ValueError, match="representable"):
        LogNormal.convert_parameters(values, LogNormal.SHAPE_SCALE)


def test_foreign_parameterizations_and_implicit_parse_conversion_are_rejected():
    original = LogNormal.DEFAULT.parse({"mu": 1, "s": 0.5})
    with pytest.raises(ValueError, match="explicit conversion"):
        LogNormal.SHAPE_SCALE.parse(original)
    for foreign in (Normal.DEFAULT, replace(LogNormal.DEFAULT, id="log_normal.other")):
        with pytest.raises(ValueError, match="Unsupported"):
            KS(foreign.parse({}))
        with pytest.raises(ValueError, match="Unsupported"):
            LogNormal.convert_parameters(foreign.parse({}), LogNormal.SHAPE_SCALE)
        with pytest.raises(ValueError, match="Unsupported"):
            LogNormal.convert_parameters(original, foreign)


def test_converted_values_match_cdf_and_generator():
    logarithmic = LogNormal.DEFAULT.parse({"mu": 0.8, "s": 0.6})
    values = LogNormal.convert_parameters(logarithmic, LogNormal.SHAPE_SCALE)
    statistic = KS(values)
    sample = np.array([0.4, 0.8, 1.5, 2.0, 3.5, 5.0])
    reference = stats.lognorm(0.6, scale=math.exp(0.8))
    assert statistic.hypothesis().parameter_values is values
    assert statistic.execute_statistic(sample) == pytest.approx(
        stats.kstest(sample, reference.cdf).statistic
    )
    generator = get_hypothesis_generator(statistic)
    assert generator.parameters() == pytest.approx(logarithmic.as_dict())
    np.testing.assert_allclose(
        generator.generate(32, random_state=np.random.default_rng(42)),
        reference.rvs(size=32, random_state=np.random.default_rng(42)),
    )


@pytest.mark.parametrize("fixed", [{}, {"s": 0.7}, {"scale": 2}])
def test_criterion_requires_both_parameters(fixed):
    with pytest.raises(ValueError, match="Unsupported"):
        KS(LogNormal.SHAPE_SCALE.parse(fixed))


def test_conversion_uses_parameter_identities_not_names(monkeypatch):
    location = replace(LogNormal.LOG_LOCATION, name="log_location")
    median = replace(LogNormal.MEDIAN, name="median")
    logarithmic = replace(LogNormal.LOG_LOCATION_SCALE, parameters=(location, LogNormal.LOG_SCALE))
    shape_scale = replace(LogNormal.SHAPE_SCALE, parameters=(LogNormal.LOG_SCALE, median))
    monkeypatch.setattr(LogNormal, "LOG_LOCATION", location)
    monkeypatch.setattr(LogNormal, "MEDIAN", median)
    monkeypatch.setattr(LogNormal, "LOG_LOCATION_SCALE", logarithmic)
    monkeypatch.setattr(LogNormal, "DEFAULT", logarithmic)
    monkeypatch.setattr(LogNormal, "SHAPE_SCALE", shape_scale)
    values = logarithmic.parse({"log_location": 0.5, "s": 0.8})
    converted = LogNormal.convert_parameters(values, shape_scale)
    assert converted.as_dict() == {"median": math.exp(0.5), "s": 0.8}
    assert KS(converted).scale == math.exp(0.5)


@pytest.mark.parametrize("fixed", [{}, {"mu": 0.8}, {"s": 0.6}])
def test_logarithmic_hypothesis_also_requires_both_parameters(fixed):
    with pytest.raises(ValueError, match="Unsupported"):
        KS(LogNormal.LOG_LOCATION_SCALE.parse(fixed))


@pytest.mark.parametrize("mu", [-1000, 1000])
def test_constructor_propagates_unrepresentable_conversion(mu):
    with pytest.raises(ValueError, match="representable"):
        KS(LogNormal.LOG_LOCATION_SCALE.parse({"mu": mu, "s": 1}))


def test_constructor_normalizes_calculation_but_preserves_input_hypothesis():
    logarithmic = LogNormal.LOG_LOCATION_SCALE.parse({"mu": 0.8, "s": 0.6})
    native = LogNormal.convert_parameters(logarithmic, LogNormal.SHAPE_SCALE)
    converted_statistic = KS(logarithmic)
    native_statistic = KS(native)
    assert converted_statistic.hypothesis().parameter_values is logarithmic
    assert native_statistic.hypothesis().parameter_values is native
    assert converted_statistic.hypothesis().parameters() == {"mu": 0.8, "s": 0.6}
    assert converted_statistic.scale == native_statistic.scale == math.exp(0.8)
    assert converted_statistic.s == native_statistic.s == 0.6
    assert KS.calculation_parameterization() is LogNormal.SHAPE_SCALE
    assert LogNormal.default_parameterization() is LogNormal.LOG_LOCATION_SCALE
    sample = [0.4, 0.8, 1.5, 2.0, 3.5, 5.0]
    assert converted_statistic.execute_statistic(sample) == native_statistic.execute_statistic(
        sample
    )
    assert converted_statistic.hypothesis().parameter_values is logarithmic
    for statistic in (converted_statistic, native_statistic):
        assert get_hypothesis_generator(statistic).parameters() == pytest.approx(
            logarithmic.as_dict()
        )


def test_every_lognormal_criterion_accepts_both_coordinate_systems():
    from pysatl_criterion.utils.statistic import get_available_criteria

    logarithmic = LogNormal.LOG_LOCATION_SCALE.parse({"mu": 0.8, "s": 0.6})
    native = LogNormal.convert_parameters(logarithmic, LogNormal.SHAPE_SCALE)
    criteria = get_available_criteria(LogNormal.type())
    assert criteria
    for criterion in criteria:
        for values in (logarithmic, native):
            statistic = criterion(values)
            assert statistic.hypothesis().parameter_values is values
            assert statistic.s == 0.6
            assert statistic.scale == math.exp(0.8)


def test_constructor_rejects_invalid_converter_result(monkeypatch):
    values = LogNormal.LOG_LOCATION_SCALE.parse({"mu": 0.8, "s": 0.6})
    monkeypatch.setattr(
        LogNormal, "convert_parameters", classmethod(lambda cls, values, target: target.parse({}))
    )
    with pytest.raises(ValueError, match="calculation hypothesis"):
        KS(values)


def test_default_calculation_schema_and_identity_conversion():
    from pysatl_criterion.distribution.distributions import BetaDistributionDescriptor as Beta
    from pysatl_criterion.statistics.goodness_of_fit.beta import KolmogorovSmirnovBetaGofStatistic

    values = Beta.DEFAULT.parse({"a": 2, "b": 5})
    assert Beta.default_parameterization() is Beta.DEFAULT
    assert Beta.convert_parameters(values, Beta.DEFAULT) is values
    assert KolmogorovSmirnovBetaGofStatistic.calculation_parameterization() is Beta.DEFAULT
    statistic = KolmogorovSmirnovBetaGofStatistic(values)
    assert statistic.hypothesis().parameter_values is values
    assert (statistic.alpha, statistic.beta) == (2, 5)
    with pytest.raises(ValueError, match="Unsupported"):
        Beta.convert_parameters(values, Normal.DEFAULT)
    with pytest.raises(ValueError, match="Unsupported"):
        Beta.convert_parameters(Normal.DEFAULT.parse({}), Beta.DEFAULT)


def test_base_converter_does_not_guess_transformations(monkeypatch):
    from pysatl_criterion.distribution.distributions import BetaDistributionDescriptor as Beta

    other = replace(Beta.DEFAULT, id="beta.other")
    monkeypatch.setattr(Beta, "parameterizations", classmethod(lambda cls: (cls.DEFAULT, other)))
    with pytest.raises(ValueError, match="not implemented"):
        Beta.convert_parameters(Beta.DEFAULT.parse({"a": 2, "b": 5}), other)
