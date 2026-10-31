"""Laplace package, generated classes, numerical kernels and documentation contracts."""

import ast
import doctest
import inspect
import pickle
import re
from pathlib import Path

import numpy as np
import pytest

from pysatl_criterion.distribution import LaplaceDistributionDescriptor as Distribution
from pysatl_criterion.statistics.alternative import AlternativeType
from pysatl_criterion.statistics.goodness_of_fit import edf
from pysatl_criterion.statistics.goodness_of_fit.continuous import laplace
from pysatl_criterion.statistics.goodness_of_fit.continuous.laplace import laplace_statistics
from pysatl_criterion.utils.statistic import get_available_criteria, get_criterion_by_code


ROOT = Path(__file__).resolve().parents[2]
ALL_CLASSES = [
    getattr(laplace, name) for name in laplace.__all__ if not name.startswith("Abstract")
]
CLASSES = [
    cls for cls in ALL_CLASSES if not issubclass(cls, laplace.AbstractFittedLaplaceGofStatistic)
]
PARAMETERS = Distribution.DEFAULT.parse({"t": 0, "s": 1})


def test_catalog_and_registry():
    assert "supported_hypotheses" not in laplace.AbstractLaplaceGofStatistic.__dict__
    assert inspect.isabstract(laplace.AbstractLaplaceGofStatistic)
    criteria = get_available_criteria(Distribution.type())
    assert set(criteria) == set(ALL_CLASSES)
    assert len({cls.code() for cls in criteria}) == 38
    for cls in criteria:
        assert "supported_hypotheses" in cls.__dict__
        assert get_criterion_by_code(cls.code()) is cls
        assert cls.__module__ == laplace.__name__
        assert pickle.loads(pickle.dumps(cls)) is cls  # noqa: S301
        parameters = (
            Distribution.DEFAULT.parse({})
            if issubclass(cls, laplace.AbstractFittedLaplaceGofStatistic)
            else PARAMETERS
        )
        sample = [-3, -1, 0, 2, 4]
        statistic = pickle.loads(pickle.dumps(cls(parameters)))  # noqa: S301
        assert statistic.execute_statistic(sample) == cls(parameters).execute_statistic(sample)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("values", [{}, {"t": 0}, {"s": 1}])
def test_composite_hypotheses_are_rejected(cls, values):
    with pytest.raises(ValueError, match="Unsupported"):
        cls(Distribution.DEFAULT.parse(values))


@pytest.mark.parametrize("cls", CLASSES)
def test_masked_observations_are_not_silently_used(cls):
    with pytest.raises(ValueError, match="masked"):
        cls(PARAMETERS).execute_statistic(np.ma.array([0.0, 999.0], mask=[False, True]))


def test_public_execution_compiles_nopython_kernels():
    for cls in CLASSES:
        cls(PARAMETERS).execute_statistic([-1, 0, 2])
    for spec in laplace_statistics._EDF_STATISTICS.values():
        kernel = getattr(
            spec.algorithm, "do_execute_log_statistic", spec.algorithm.do_execute_statistic
        )
        assert kernel.nopython_signatures
    for direction in AlternativeType:
        laplace.KolmogorovSmirnovLaplaceGofStatistic(
            PARAMETERS, alternative_type=direction
        ).execute_statistic([-1, 0, 2])
    for cls in (
        edf.AbstractKSStatistic,
        edf.AbstractKSDPlusStatistic,
        edf.AbstractKSDMinusStatistic,
    ):
        assert cls.do_execute_statistic.nopython_signatures
    assert laplace.AbstractLaplaceGofStatistic._cdf_values.nopython_signatures
    assert laplace.AbstractLaplaceGofStatistic._log_probabilities.nopython_signatures
    assert laplace.GreenwoodLaplaceGofStatistic._spacing_statistic.nopython_signatures


def test_ad_extreme_shared_kernel_does_not_overflow_before_weighting():
    # Independent analytic limit for (-a, 0, a): 2a/3 + 8log(2)/3 - 3.
    a = 1e308
    logs = np.array([-a, -np.log(2), 0.0])
    assert edf.AbstractAndersonDarlingStatistic.do_execute_log_statistic(
        logs, logs[::-1].copy()
    ) == pytest.approx(2 * (a / 3))


def test_docs_examples_links_and_catalog():
    directory = ROOT / "docs/criteria/goodness-of-fit/laplace"
    index = (directory / "index.md").read_text()
    for cls in ALL_CLASSES:
        assert cls.__name__ in index
        assert cls.code() in index
    for page in directory.glob("*.md"):
        content = page.read_text()
        for code in re.findall(r"```python\n(.*?)```", content, re.DOTALL):
            exec(compile(code, str(page), "exec"), {})  # noqa: S102
        for target in re.findall(r"\]\(([^)]+)\)", content):
            if "://" not in target and not target.startswith("#"):
                assert (page.parent / target.split("#")[0]).exists(), (page, target)
    assert doctest.testmod(laplace).failed == 0


def test_stub_and_no_module_level_numerical_helpers():
    source = Path(laplace_statistics.__file__).read_text()
    assert not any(isinstance(n, ast.FunctionDef) for n in ast.parse(source).body)
    stub = ast.parse(Path(laplace.__file__).with_name("laplace_statistics.pyi").read_text())
    assert {n.name for n in stub.body if isinstance(n, ast.ClassDef)} == set(laplace.__all__)


@pytest.mark.parametrize("direction", list(AlternativeType))
def test_pickle_preserves_ks_settings_and_fixed_parameters(direction):
    parameters = Distribution.DEFAULT.parse({"t": 3.0, "s": 2.0})
    criterion = laplace.KolmogorovSmirnovLaplaceGofStatistic(
        parameters, alternative_type=direction, mode="asymp"
    )
    restored = pickle.loads(pickle.dumps(criterion))  # noqa: S301
    assert restored.mode == "asymp"
    assert restored.alternative_type == direction
    assert restored.hypothesis().parameters() == {"t": 3.0, "s": 2.0}
    assert restored.execute_statistic([-1, 0, 2]) == criterion.execute_statistic([-1, 0, 2])


def test_all_shared_algorithms_are_generated():
    from pysatl_criterion.statistics.goodness_of_fit import frequency

    for module, base, catalog in (
        (edf, edf.AbstractEDFStatistic, laplace_statistics._EDF_STATISTICS),
        (frequency, frequency.AbstractFrequencyStatistic, laplace_statistics._FREQUENCY_STATISTICS),
    ):
        algorithms = {
            cls
            for cls in vars(module).values()
            if inspect.isclass(cls) and cls is not base and issubclass(cls, base)
        }
        assert {spec.algorithm for spec in catalog.values()} == algorithms
        for name, spec in catalog.items():
            cls = getattr(laplace, name)
            assert cls.do_execute_statistic is spec.algorithm.do_execute_statistic
    definitions = {
        node.name
        for node in ast.parse(Path(laplace_statistics.__file__).read_text()).body
        if isinstance(node, ast.ClassDef)
    }
    assert {"AbstractLaplaceGofStatistic", "GreenwoodLaplaceGofStatistic"} <= definitions
    assert all(name.startswith("Abstract") or name in laplace.__all__ for name in definitions)


@pytest.mark.parametrize(
    "prefix,power",
    [
        ("Chi2Pearson", 1),
        ("FreemanTukey", -0.5),
        ("ModifiedLogLikelihood", -1),
        ("NeymanChi2", -2),
        ("CressieRead", 2 / 3),
    ],
)
def test_generated_frequency_against_scipy(prefix, power):
    from scipy.stats import laplace as reference
    from scipy.stats import power_divergence

    from pysatl_criterion.statistics.goodness_of_fit.binning import EqualProbabilityBins

    counts = np.array([2, 3, 4, 5])
    sample = reference.ppf(np.repeat([0.125, 0.375, 0.625, 0.875], counts))
    cls = getattr(laplace, prefix + "LaplaceGofStatistic")
    criterion = cls(PARAMETERS, binning=EqualProbabilityBins(count=4))
    expected = power_divergence(counts, np.full(4, counts.sum() / 4), lambda_=power).statistic
    assert criterion.execute_statistic(sample) == pytest.approx(expected)
    assert criterion.configuration()["binning"]
    restored = pickle.loads(pickle.dumps(criterion))  # noqa: S301
    assert restored.configuration() == criterion.configuration()
    assert restored.execute_statistic(sample) == pytest.approx(expected)


def test_added_edf_against_independent_formulas():
    from scipy.stats import laplace as reference

    sample = np.array([-2.0, -0.3, 0.0, 0.2, 1.5])
    u = reference.cdf(sample)
    n = len(u)
    i = np.arange(1, n + 1)
    ranks = i - 0.5
    plus, minus = max(i / n - u), max(u - (i - 1) / n)
    values = {
        "KolmogorovSmirnovDPlus": plus,
        "KolmogorovSmirnovDMinus": minus,
        "KuiperScaled": np.sqrt(n) * (plus + minus),
        "KuiperCorrected": (np.sqrt(n) + 0.155 + 0.24 / np.sqrt(n)) * (plus + minus),
        "LiaoShimokawa": sum(np.maximum(i / n - u, u - (i - 1) / n) / np.sqrt(u * (1 - u)))
        / np.sqrt(n),
        "ZhangK": max(
            ranks * np.log(ranks / (n * u)) + (n - ranks) * np.log((n - ranks) / (n * (1 - u)))
        ),
        "ZhangA": -sum(np.log(u) / (n - ranks) + np.log1p(-u) / ranks),
        "ZhangC": sum(np.log(((1 - u) / u) / ((n - i + 0.25) / (i - 0.75))) ** 2),
    }
    for prefix, expected in values.items():
        criterion = getattr(laplace, prefix + "LaplaceGofStatistic")(PARAMETERS)
        assert criterion.execute_statistic(sample) == pytest.approx(expected)
