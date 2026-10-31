"""Contract and numerical checks for the dynamically exported EDF classes."""

import ast
import inspect
import pickle
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from pysatl_criterion.distribution import UniformDistributionDescriptor as Uniform
from pysatl_criterion.statistics.alternative import AlternativeType
from pysatl_criterion.statistics.goodness_of_fit import edf
from pysatl_criterion.statistics.goodness_of_fit.continuous import uniform as module
from pysatl_criterion.statistics.goodness_of_fit.continuous._edf_factory import (
    export_edf_statistics,
)
from pysatl_criterion.statistics.goodness_of_fit.continuous.uniform.uniform_statistics import (
    _EDF_STATISTICS,
)


@pytest.mark.parametrize("name", _EDF_STATISTICS)
def test_export_and_serialization(name):
    cls = getattr(module, name)
    assert name in module.__all__
    assert cls.__module__ == module.__name__
    assert cls.__qualname__ == name
    assert not inspect.isabstract(cls)
    assert "short_code" not in cls.__dict__
    assert cls.short_code() == _EDF_STATISTICS[name].algorithm.short_code()
    criterion = cls(Uniform.DEFAULT.parse({"a": 2, "b": 5}))
    assert criterion.alternative().type() == AlternativeType.RIGHT
    assert criterion.code() == f"{cls.short_code()}_UNIFORM_GOODNESS_OF_FIT"
    sample = [2.3, 3.2, 4.7]
    restored_cls = pickle.loads(pickle.dumps(cls))  # noqa: S301 - local class reference
    assert restored_cls is cls
    restored = restored_cls(Uniform.DEFAULT.parse({"a": 2, "b": 5}))
    assert restored.execute_statistic(sample) == criterion.execute_statistic(sample)


def test_independent_numerical_references():
    sample = np.array([0.8, 0.1, 0.3, 0.3, 0.6])
    ordered = np.sort(sample)
    n = len(sample)
    plus = stats.kstest(sample, "uniform", alternative="greater").statistic
    minus = stats.kstest(sample, "uniform", alternative="less").statistic
    cvm = stats.cramervonmises(sample, "uniform").statistic
    ad = -n - np.mean((2 * np.arange(1, n + 1) - 1) * (np.log(ordered) + np.log1p(-ordered[::-1])))
    expected = {
        "KS_PLUS": plus,
        "KS_MINUS": minus,
        "KS": max(plus, minus),
        "AD": ad,
        "CVM": cvm,
        "WATSON": cvm - n * (np.mean(sample) - 0.5) ** 2,
        "KUIPER": plus + minus,
        "KUIPER_SCALED": np.sqrt(n) * (plus + minus),
        "KUIPER_CORRECTED": (np.sqrt(n) + 0.155 + 0.24 / np.sqrt(n)) * (plus + minus),
        "LIAO_SHIMOKAWA": sum(
            max(i / n - u, u - (i - 1) / n) / np.sqrt(u * (1 - u))
            for i, u in enumerate(ordered, start=1)
        )
        / np.sqrt(n),
        "ZHANG_ZK": max(
            (i - 0.5) * np.log((i - 0.5) / (n * u))
            + (n - i + 0.5) * np.log((n - i + 0.5) / (n * (1 - u)))
            for i, u in enumerate(ordered, start=1)
        ),
        "ZHANG_ZA": -sum(
            np.log(u) / (n - i + 0.5) + np.log(1 - u) / (i - 0.5)
            for i, u in enumerate(ordered, start=1)
        ),
        "ZHANG_ZC": sum(
            np.log((1 / u - 1) / ((n - 0.5) / (i - 0.75) - 1)) ** 2
            for i, u in enumerate(ordered, start=1)
        ),
    }
    for name, spec in _EDF_STATISTICS.items():
        criterion = getattr(module, name)(Uniform.DEFAULT.parse({"a": 2, "b": 5}))
        assert criterion.execute_statistic(2 + 3 * sample) == pytest.approx(
            expected[spec.algorithm.short_code()]
        )
    np.testing.assert_array_equal(sample, [0.8, 0.1, 0.3, 0.3, 0.6])


def test_catalog_covers_all_edf_algorithms():
    algorithms = {
        cls
        for _, cls in inspect.getmembers(edf, inspect.isclass)
        if issubclass(cls, edf.AbstractEDFStatistic) and cls is not edf.AbstractEDFStatistic
    }
    assert {spec.algorithm for spec in _EDF_STATISTICS.values()} == algorithms


@pytest.mark.parametrize(
    "sample", [[], [[0.5]], [np.nan], [np.inf], [1j], np.ma.array([0.5], mask=[True])]
)
def test_invalid_sample(sample):
    criterion = module.WatsonUniformGofStatistic(Uniform.DEFAULT.parse({"a": 0, "b": 1}))
    with pytest.raises(ValueError):
        criterion.execute_statistic(sample)


def test_boundaries_and_unknown_parameters():
    parameters = Uniform.DEFAULT.parse({"a": 0, "b": 1})
    assert module.AndersonDarlingUniformGofStatistic(parameters).execute_statistic([0, 1]) == np.inf
    assert module.KolmogorovSmirnovUniformGofStatistic(parameters).execute_statistic([-1, 2]) == 0.5
    with pytest.raises(ValueError):
        module.WatsonUniformGofStatistic(Uniform.DEFAULT.parse({}))


def test_public_stub_matches_runtime():
    tree = ast.parse(Path(module.__file__).with_name("uniform_statistics.pyi").read_text())
    classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert classes == set(module.__all__)


def test_export_collision_does_not_mutate_namespace():
    namespace = {"__name__": "example", "WatsonUniformGofStatistic": object()}
    before = namespace.copy()
    with pytest.raises(ValueError, match="already exported"):
        export_edf_statistics(namespace, module.AbstractUniformGofStatistic, _EDF_STATISTICS)
    assert namespace == before


@pytest.mark.parametrize("name", module.__all__)
def test_package_exports_preserve_class_identity_and_pickle(name):
    from pysatl_criterion.statistics.goodness_of_fit.continuous.uniform import uniform_statistics

    cls = getattr(module, name)
    assert cls is getattr(uniform_statistics, name)
    assert cls.__module__ == module.__name__
    # Protocol-0 GLOBAL represents class references saved with the former module path.
    old_reference = f"c{module.__name__}\n{name}\n.".encode()
    assert pickle.loads(old_reference) is cls  # noqa: S301 - locally constructed class reference
    assert pickle.loads(pickle.dumps(cls)) is cls  # noqa: S301 - local class reference
    if not inspect.isabstract(cls):
        from pysatl_criterion.utils.statistic import get_available_criteria, get_criterion_by_code

        assert get_criterion_by_code(cls.code()) is cls
        assert get_available_criteria(Uniform.type()).count(cls) == 1


def test_hypothesis_support_belongs_to_concrete_criteria():
    from pysatl_criterion.statistics.hypothesis import GoodnessOfFitHypothesis

    assert "supported_hypotheses" not in module.AbstractUniformGofStatistic.__dict__
    for name in module.__all__:
        cls = getattr(module, name)
        if inspect.isabstract(cls):
            continue
        assert "supported_hypotheses" in cls.__dict__
        for values in ({}, {"a": 0}, {"b": 1}, {"a": 0, "b": 1}):
            hypothesis = GoodnessOfFitHypothesis(Uniform.DEFAULT.parse(values))
            assert cls.supports_hypothesis(hypothesis) is (len(values) == 2)


def test_uniform_base_allows_a_criterion_with_unknown_bounds():
    from pysatl_criterion.distribution.parameters import HypothesisSupport
    from pysatl_criterion.statistics.alternative import RightAlternative

    class UnknownBoundsCriterion(module.AbstractUniformGofStatistic):
        @classmethod
        def supported_hypotheses(cls):
            return (HypothesisSupport(Uniform.DEFAULT, frozenset()),)

        @staticmethod
        def short_code():
            return "TEST_UNKNOWN_BOUNDS"

        def alternative(self):
            return RightAlternative()

        def execute_statistic(self, rvs, **kwargs):
            return float(len(rvs))

    parameters = Uniform.DEFAULT.parse({})
    criterion = UnknownBoundsCriterion(parameters)
    assert criterion.hypothesis().parameter_values is parameters
    assert criterion.execute_statistic([1, 2]) == 2
    with pytest.raises(ValueError, match="Unsupported"):
        UnknownBoundsCriterion(Uniform.DEFAULT.parse({"a": 0, "b": 1}))


@pytest.mark.parametrize("kind", ["edf", "frequency"])
def test_factories_validate_each_criterion_hypothesis_before_export(kind):
    from pysatl_criterion.statistics.goodness_of_fit import frequency
    from pysatl_criterion.statistics.goodness_of_fit.continuous._edf_factory import EDFSpec
    from pysatl_criterion.statistics.goodness_of_fit.continuous._frequency_factory import (
        FrequencySpec,
        export_frequency_statistics,
    )

    fixed = module.WatsonUniformGofStatistic.supported_hypotheses()
    if kind == "edf":
        export = export_edf_statistics
        valid = EDFSpec(edf.AbstractKSStatistic, supported_hypotheses=fixed)
        invalid = EDFSpec(edf.AbstractWatsonStatistic, supported_hypotheses=())
    else:
        export = export_frequency_statistics
        valid = FrequencySpec(frequency.AbstractPearsonChi2Statistic, supported_hypotheses=fixed)
        invalid = FrequencySpec(frequency.AbstractCressieReadStatistic, supported_hypotheses=())
    namespace = {"__name__": "example"}
    with pytest.raises(ValueError):
        export(namespace, module.AbstractUniformGofStatistic, {"Valid": valid, "Invalid": invalid})
    assert namespace == {"__name__": "example"}
