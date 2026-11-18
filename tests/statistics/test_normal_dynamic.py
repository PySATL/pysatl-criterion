"""Normal EDF/frequency generation and explicit hypothesis contracts."""

import inspect
import pickle
from typing import get_origin

import numpy as np
import pytest
from scipy import stats

from pysatl_criterion.statistics.goodness_of_fit import edf, frequency
from pysatl_criterion.statistics.goodness_of_fit.binning import ExplicitBins
from pysatl_criterion.statistics.goodness_of_fit.continuous import normal
from pysatl_criterion.statistics.goodness_of_fit.continuous.normal import normal_statistics
from tests.parameter_cases import parameters_for


CATALOG = {**normal_statistics._EDF_STATISTICS, **normal_statistics._FREQUENCY_STATISTICS}
SAMPLE = np.array([-2.1, -1.0, -0.5, -0.2, 0.1, 0.4, 0.9, 1.3, 2.5])


def test_catalog_covers_all_shared_algorithms():
    for module, catalog, base in (
        (edf, normal_statistics._EDF_STATISTICS, edf.AbstractEDFStatistic),
        (frequency, normal_statistics._FREQUENCY_STATISTICS, frequency.AbstractFrequencyStatistic),
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


def test_hypothesis_support_belongs_to_concrete_statistics():
    assert "supported_hypotheses" not in vars(normal.AbstractNormalityGofStatistic)
    for cls in vars(normal).values():
        if (
            inspect.isclass(cls)
            and cls.__module__ == normal.__name__
            and not inspect.isabstract(cls)
        ):
            assert "supported_hypotheses" in vars(cls)


@pytest.mark.parametrize("name", CATALOG)
def test_generated_classes_are_exported_and_pickleable(name):
    cls = getattr(normal, name)
    assert name in normal.__all__
    assert cls.__module__ == normal.__name__
    assert pickle.loads(pickle.dumps(cls)) is cls  # noqa: S301


@pytest.mark.parametrize("name,spec", normal_statistics._EDF_STATISTICS.items())
def test_generated_edf_matches_shared_kernel(name, spec):
    cls = getattr(normal, name)
    criterion = cls(parameters_for(cls))
    sample = np.sort(SAMPLE)
    if not criterion.hypothesis().parameters():
        sample = (sample - sample.mean()) / sample.std(ddof=1)
    expected = spec.algorithm.do_execute_statistic(stats.norm.cdf(sample))
    if name == "ZhangWuANormalityGofStatistic":
        expected = 10 * expected - 32
    assert criterion.execute_statistic(SAMPLE) == pytest.approx(expected)


@pytest.mark.parametrize("name,spec", normal_statistics._FREQUENCY_STATISTICS.items())
def test_generated_frequency_uses_fitted_probabilities_and_custom_bins(name, spec):
    cls = getattr(normal, name)
    edges = np.array([-np.inf, -0.4, 0.7, np.inf])
    criterion = cls(parameters_for(cls), binning=ExplicitBins(edges))
    for sample in (SAMPLE, SAMPLE * 1.5 + 0.2):
        observed = np.histogram(sample, edges)[0]
        probabilities = np.diff(stats.norm.cdf(edges, loc=sample.mean(), scale=sample.std(ddof=1)))
        expected = spec.algorithm.do_execute_statistic(observed, len(sample) * probabilities)
        assert criterion.execute_statistic(sample) == pytest.approx(expected)
    assert criterion.hypothesis().parameters() == {}
