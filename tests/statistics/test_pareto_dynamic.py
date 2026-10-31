"""Generated Pareto catalog and fixed-hypothesis contracts."""

import ast
import inspect
import pickle
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from pysatl_criterion.distribution import ParetoDistributionDescriptor as Distribution
from pysatl_criterion.statistics.goodness_of_fit import edf, frequency
from pysatl_criterion.statistics.goodness_of_fit.binning import ExplicitBins
from pysatl_criterion.statistics.goodness_of_fit.continuous import pareto
from pysatl_criterion.statistics.goodness_of_fit.continuous.pareto import pareto_statistics
from pysatl_criterion.utils.statistic import get_available_criteria, get_criterion_by_code


CATALOG = {**pareto_statistics._EDF_STATISTICS, **pareto_statistics._FREQUENCY_STATISTICS}
SAMPLE = np.array([1.1, 1.3, 1.5, 2.1, 2.7, 3.4, 5.6, 8.0])


def test_complete_catalog_and_per_criterion_hypotheses():
    assert "supported_hypotheses" not in pareto.AbstractParetoGofStatistic.__dict__
    for module, catalog, base in (
        (edf, pareto_statistics._EDF_STATISTICS, edf.AbstractEDFStatistic),
        (frequency, pareto_statistics._FREQUENCY_STATISTICS, frequency.AbstractFrequencyStatistic),
    ):
        algorithms = {
            cls
            for cls in vars(module).values()
            if inspect.isclass(cls) and issubclass(cls, base) and cls is not base
        }
        assert {spec.algorithm for spec in catalog.values()} == algorithms
    criteria = get_available_criteria(Distribution.type())
    assert len(criteria) == len(CATALOG) + 3
    assert len({cls.code() for cls in criteria}) == len(criteria)
    for cls in criteria:
        assert "supported_hypotheses" in cls.__dict__


@pytest.mark.parametrize("name", CATALOG)
def test_generated_identity_and_hypothesis(name):
    cls = getattr(pareto, name)
    assert name in pareto.__all__
    assert cls.__module__ == pareto.__name__
    assert pickle.loads(pickle.dumps(cls)) is cls  # noqa: S301
    assert get_criterion_by_code(cls.code()) is cls
    assert cls.supported_hypotheses() == CATALOG[name].supported_hypotheses
    for values in ({}, {"shape": 2}, {"scale": 1}):
        with pytest.raises(ValueError, match="Unsupported"):
            cls(Distribution.DEFAULT.parse(values))
    criterion = cls(Distribution.DEFAULT.parse({"shape": 2, "scale": 1}))
    with pytest.raises(ValueError, match="at least scale"):
        criterion.execute_statistic([0.5, 2, 3])


@pytest.mark.parametrize("name", pareto_statistics._EDF_STATISTICS)
def test_edf_matches_shared_kernel(name):
    sample = SAMPLE.copy()
    cls = getattr(pareto, name)
    criterion = cls(Distribution.DEFAULT.parse({"shape": 2, "scale": 1}))
    cdf = stats.pareto(2, scale=1).cdf(sample)
    expected = CATALOG[name].algorithm.do_execute_statistic(cdf)
    assert criterion.execute_statistic(sample[::-1]) == pytest.approx(expected)
    np.testing.assert_array_equal(sample, SAMPLE)


@pytest.mark.parametrize("name", pareto_statistics._FREQUENCY_STATISTICS)
def test_frequency_explicit_bins(name):
    edges = (1, 1.5, 3, np.inf)
    observed = np.histogram(SAMPLE, edges)[0]
    expected = len(SAMPLE) * np.diff(stats.pareto(2, scale=1).cdf(edges))
    cls = getattr(pareto, name)
    criterion = cls(
        Distribution.DEFAULT.parse({"shape": 2, "scale": 1}),
        binning=ExplicitBins(edges),
    )
    reference = CATALOG[name].algorithm.do_execute_statistic(observed, expected)
    assert criterion.execute_statistic(SAMPLE) == pytest.approx(reference)


def test_public_stub_matches_exports():
    tree = ast.parse(Path(pareto.__file__).with_name("pareto_statistics.pyi").read_text())
    classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert classes == set(pareto.__all__)


@pytest.mark.parametrize("name", pareto.__all__)
def test_package_exports_preserve_identity_and_existing_pickles(name):
    cls = getattr(pareto, name)
    assert cls is getattr(pareto_statistics, name)
    assert cls.__module__ == pareto.__name__
    old_reference = f"c{pareto.__name__}\n{name}\n.".encode()
    assert pickle.loads(old_reference) is cls  # noqa: S301 - local class reference
    assert pickle.loads(pickle.dumps(cls)) is cls  # noqa: S301 - local class reference
