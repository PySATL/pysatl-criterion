"""Generated Gamma criteria preserve fixed hypotheses and shared formulas."""

import inspect
import pickle
from typing import get_origin

import numpy as np
import pytest
from scipy import stats

from pysatl_criterion.distribution import GammaDistributionDescriptor as Distribution
from pysatl_criterion.statistics.goodness_of_fit import edf, frequency
from pysatl_criterion.statistics.goodness_of_fit.binning import ExplicitBins
from pysatl_criterion.statistics.goodness_of_fit.continuous import gamma
from pysatl_criterion.statistics.goodness_of_fit.continuous.gamma import gamma_statistics


CATALOG = {**gamma_statistics._EDF_STATISTICS, **gamma_statistics._FREQUENCY_STATISTICS}
SAMPLE = np.array([0.0, 0.2, 0.5, 0.8, 1.2, 1.8, 2.5, 4.0])
PARAMETERS = Distribution.DEFAULT.parse({"alfa": 2, "beta": 1.5})


def test_complete_catalog_and_explicit_hypotheses():
    assert "supported_hypotheses" not in gamma.AbstractGammaGofStatistic.__dict__
    for module, catalog, base in (
        (edf, gamma_statistics._EDF_STATISTICS, edf.AbstractEDFStatistic),
        (frequency, gamma_statistics._FREQUENCY_STATISTICS, frequency.AbstractFrequencyStatistic),
    ):
        algorithms = {
            cls
            for cls in vars(module).values()
            if inspect.isclass(cls)
            and get_origin(cls) is None
            and issubclass(cls, base)
            and cls is not base
        }
        assert {spec.algorithm for spec in catalog.values()} == algorithms
    for name in gamma.__all__:
        cls = getattr(gamma, name)
        if not inspect.isabstract(cls):
            assert "supported_hypotheses" in cls.__dict__
            cls(PARAMETERS)


@pytest.mark.parametrize("name", CATALOG)
def test_generated_identity_and_validation(name):
    cls = getattr(gamma, name)
    assert pickle.loads(pickle.dumps(cls)) is cls  # noqa: S301
    assert cls.supported_hypotheses() == CATALOG[name].supported_hypotheses
    for values in ({}, {"alfa": 2}, {"beta": 1.5}):
        with pytest.raises(ValueError, match="Unsupported"):
            cls(Distribution.DEFAULT.parse(values))
    for sample in ([-1, 2], [], [np.nan], [np.inf], [[1, 2]]):
        with pytest.raises(ValueError):
            cls(PARAMETERS).execute_statistic(sample)


@pytest.mark.parametrize("name", gamma_statistics._EDF_STATISTICS)
def test_edf_matches_shared_kernel(name):
    sample = SAMPLE[1:].copy()
    criterion = getattr(gamma, name)(PARAMETERS)
    expected = CATALOG[name].algorithm.do_execute_statistic(
        stats.gamma.cdf(sample, a=2, scale=1 / 1.5)
    )
    assert criterion.execute_statistic(sample[::-1]) == pytest.approx(expected)
    np.testing.assert_array_equal(sample, SAMPLE[1:])


@pytest.mark.parametrize("name", gamma_statistics._FREQUENCY_STATISTICS)
def test_frequency_matches_shared_kernel(name):
    edges = (0, 0.7, 2, np.inf)
    observed = np.histogram(SAMPLE, edges)[0]
    expected = len(SAMPLE) * np.diff(stats.gamma.cdf(edges, a=2, scale=1 / 1.5))
    criterion = getattr(gamma, name)(PARAMETERS, binning=ExplicitBins(edges))
    reference = CATALOG[name].algorithm.do_execute_statistic(observed, expected)
    assert criterion.execute_statistic(SAMPLE) == pytest.approx(reference)


def test_likelihood_ratio_and_liao_shimokawa_remain_available():
    criterion = gamma.LikelihoodRatioGammaGofStatistic(PARAMETERS, bins=3)
    observed, expected = criterion._counts_and_expected(SAMPLE)
    assert criterion.execute_statistic(SAMPLE) == pytest.approx(
        stats.power_divergence(observed, expected, lambda_=0).statistic
    )
    assert np.isfinite(
        gamma.LiaoShimokawaGammaGofStatistic(PARAMETERS).execute_statistic(SAMPLE[1:])
    )


@pytest.mark.parametrize("name", gamma.__all__)
def test_package_exports_preserve_identity_and_pickle(name):
    cls = getattr(gamma, name)
    assert cls is getattr(gamma_statistics, name)
    assert cls.__module__ == gamma.__name__
    assert pickle.loads(pickle.dumps(cls)) is cls  # noqa: S301
    # Protocol-0 global reference uses the module path from before the move.
    old_reference = f"c{gamma.__name__}\n{name}\n.".encode()
    assert pickle.loads(old_reference) is cls  # noqa: S301


def test_package_stub_matches_exports():
    import ast
    from pathlib import Path

    tree = ast.parse(Path(gamma.__file__).with_name("gamma_statistics.pyi").read_text())
    assert {node.name for node in tree.body if isinstance(node, ast.ClassDef)} == set(gamma.__all__)
    assert set(gamma.__all__) == set(gamma_statistics.__all__)
