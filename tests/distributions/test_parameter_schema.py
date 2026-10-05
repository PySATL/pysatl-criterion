"""Schema identity, hypothesis semantics and Student integration."""

from dataclasses import replace

import numpy as np
import pytest
from scipy import stats

from pysatl_criterion import DistributionType
from pysatl_criterion.distribution.distributions import StudentDistributionDescriptor as Student
from pysatl_criterion.distribution.parameters import (
    HypothesisSupport,
    ParameterizationDescriptor,
    ParameterSpec,
)
from pysatl_criterion.generator.generators import TRVSGenerator
from pysatl_criterion.statistics.goodness_of_fit.student import (
    KolmogorovSmirnovStudentGofStatistic,
    LillieforsStudentGofStatistic,
)
from pysatl_criterion.statistics.hypothesis import GoodnessOfFitHypothesis
from pysatl_criterion.utils.generator import get_available_generator, get_hypothesis_generator


def test_partial_hypotheses_do_not_acquire_defaults():
    values = Student.DEFAULT.parse({Student.DF: 5})
    hypothesis = GoodnessOfFitHypothesis(values)
    assert hypothesis.parameters() == {"df": 5}
    assert not KolmogorovSmirnovStudentGofStatistic.supports_hypothesis(hypothesis)
    with pytest.raises(ValueError, match="Unsupported"):
        KolmogorovSmirnovStudentGofStatistic.from_parameters(values)
    with pytest.raises(ValueError, match="complete"):
        TRVSGenerator.from_parameters(values)
    assert Student.DEFAULT.parse({}, fill_defaults=True).as_dict() == {
        "df": 1,
        "loc": 0,
        "scale": 1,
    }


def test_student_descriptor_statistic_and_sampler_share_parameters():
    values = Student.DEFAULT.parse({"df": 5, "loc": 3, "scale": 2})
    statistic = KolmogorovSmirnovStudentGofStatistic.from_parameters(values)
    generator = get_hypothesis_generator(statistic)
    assert generator.parameters() == statistic.hypothesis().parameters() == values.as_dict()
    actual = generator.generate(32, random_state=np.random.default_rng(42))
    expected = stats.t.rvs(5, loc=3, scale=2, size=32, random_state=np.random.default_rng(42))
    np.testing.assert_allclose(actual, expected)
    assert statistic.execute_statistic(actual) == pytest.approx(
        stats.kstest(actual, stats.t(5, loc=3, scale=2).cdf).statistic
    )


def test_rename_in_schema_updates_all_consumers(monkeypatch):
    original = Student.DEFAULT.parse({"df": 5, "loc": 3, "scale": 2})
    location = replace(Student.LOCATION, name="location", aliases=("loc",))
    schema = replace(Student.DEFAULT, parameters=(Student.DF, location, Student.SCALE))
    monkeypatch.setattr(Student, "LOCATION", location)
    monkeypatch.setattr(Student, "DEFAULT", schema)
    values = schema.parse({"df": 5, "location": 3, "scale": 2})
    assert values.storage_identity() == original.storage_identity()
    assert location == next(p for p in original if p.id == location.id)
    statistic = KolmogorovSmirnovStudentGofStatistic.from_parameters(values)
    generator = get_hypothesis_generator(statistic)
    factory_generator = get_available_generator(DistributionType.STUDENT, values.as_dict())
    assert statistic.hypothesis().parameters() == generator.parameters() == values.as_dict()
    assert factory_generator.parameters() == values.as_dict()
    assert "location" in {p.name for p in Student.parameters()}
    assert schema.parse({"loc": 3}).as_dict() == {"location": 3}
    with pytest.raises(ValueError, match="Duplicate"):
        schema.parse({"loc": 3, "location": 4})


def test_second_parameterization_is_not_implicitly_converted():
    other = replace(Student.DEFAULT, id="student.other")
    values = other.parse({}, fill_defaults=True)
    with pytest.raises(ValueError, match="explicit conversion"):
        Student.DEFAULT.parse(values)
    with pytest.raises(ValueError, match="Unsupported"):
        KolmogorovSmirnovStudentGofStatistic.from_parameters(values)
    with pytest.raises(ValueError, match="parameterization"):
        TRVSGenerator.from_parameters(values)
    assert (
        values.storage_identity()
        != Student.DEFAULT.parse({}, fill_defaults=True).storage_identity()
    )


def test_uniform_width_is_not_a_fixed_endpoint():
    lower = ParameterSpec("uniform.lower", "lower", "a")
    upper = ParameterSpec("uniform.upper", "upper", "b")
    scale = ParameterSpec("uniform.scale", "scale", "s")
    bounds = ParameterizationDescriptor("uniform.bounds", DistributionType.UNIFORM, (lower, upper))
    location_scale = ParameterizationDescriptor(
        "uniform.loc_scale", DistributionType.UNIFORM, (lower, scale)
    )
    width = location_scale.parse({scale: 2})
    assert not HypothesisSupport(bounds, frozenset()).supports(width)
    assert HypothesisSupport(location_scale, frozenset({scale})).supports(width)
    with pytest.raises(ValueError, match="belong"):
        HypothesisSupport(bounds, frozenset({scale}))


@pytest.mark.parametrize("name,value", [("df", 0), ("df", np.inf), ("scale", -1), ("loc", np.nan)])
def test_invalid_values_are_rejected_at_the_schema_boundary(name, value):
    with pytest.raises(ValueError, match="Invalid"):
        Student.DEFAULT.parse({name: value})


def test_unknown_names_and_foreign_tokens_are_rejected():
    for key in ("variance", ParameterSpec("normal.df", "df", "ν")):
        with pytest.raises(ValueError, match="Unknown"):
            Student.DEFAULT.parse({key: 1})


def test_current_lilliefors_cannot_claim_arbitrary_location_scale():
    standard = Student.DEFAULT.parse({"df": 5}, fill_defaults=True)
    statistic = LillieforsStudentGofStatistic.from_parameters(standard)
    assert statistic.hypothesis().parameters() == standard.as_dict()
    shifted = Student.DEFAULT.parse({"df": 5, "loc": 1}, fill_defaults=True)
    with pytest.raises(ValueError, match="Unsupported"):
        LillieforsStudentGofStatistic.from_parameters(shifted)


def test_parsing_copies_input_and_exports_independent_dictionaries():
    source = {"df": 5}
    values = Student.DEFAULT.parse(source)
    source["df"] = 10
    exported = values.as_dict()
    exported["df"] = 20
    assert values[Student.DF] == 5
    with pytest.raises(TypeError):
        values[Student.DF] = 30


def test_ambiguous_schema_is_rejected():
    with pytest.raises(ValueError, match="unique"):
        replace(Student.DEFAULT, parameters=(Student.DF, Student.DF))
    aliased_scale = replace(Student.SCALE, aliases=("df",))
    with pytest.raises(ValueError, match="unique"):
        replace(Student.DEFAULT, parameters=(Student.DF, aliased_scale))
