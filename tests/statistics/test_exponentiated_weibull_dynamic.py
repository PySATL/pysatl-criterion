"""Generated exponentiated Weibull catalogs and fixed-hypothesis contracts."""

import inspect
import pickle
from typing import get_origin

import numpy as np
import pytest
from scipy import stats

from pysatl_criterion.distribution import ExponentiatedWeibullDistributionDescriptor as Distribution
from pysatl_criterion.statistics.goodness_of_fit import edf, frequency
from pysatl_criterion.statistics.goodness_of_fit.binning import ExplicitBins
from pysatl_criterion.statistics.goodness_of_fit.continuous import exponentiated_weibull as module
from pysatl_criterion.statistics.goodness_of_fit.continuous.exponentiated_weibull import (
    exponentiated_weibull_statistics as implementation,
)
from pysatl_criterion.utils.statistic import get_criterion_by_code


CATALOG = {**implementation._EDF_STATISTICS, **implementation._FREQUENCY_STATISTICS}
PARAMETERS = Distribution.DEFAULT.parse({"exponent": 1.4, "shape": 2.1, "scale": 1.3})
SAMPLE = np.array([0.12, 0.23, 0.4, 0.61, 0.88, 1.05, 1.37, 1.7, 2.3])


def test_catalog_covers_shared_algorithms():
    assert "supported_hypotheses" not in module.AbstractExponentiatedWeibullGofStatistic.__dict__
    for source, catalog, base in (
        (edf, implementation._EDF_STATISTICS, edf.AbstractEDFStatistic),
        (frequency, implementation._FREQUENCY_STATISTICS, frequency.AbstractFrequencyStatistic),
    ):
        algorithms = {
            cls
            for cls in vars(source).values()
            if inspect.isclass(cls)
            and get_origin(cls) is None
            and issubclass(cls, base)
            and cls is not base
        }
        assert {spec.algorithm for spec in catalog.values()} == algorithms


@pytest.mark.parametrize("name", CATALOG)
def test_generated_contract_and_registration(name):
    cls = getattr(module, name)
    assert name in module.__all__
    assert "supported_hypotheses" in cls.__dict__
    assert cls.supported_hypotheses() == CATALOG[name].supported_hypotheses
    assert get_criterion_by_code(cls.code()) is cls
    assert pickle.loads(pickle.dumps(cls)) is cls  # noqa: S301
    with pytest.raises(ValueError, match="Unsupported"):
        cls(Distribution.DEFAULT.parse({"exponent": 1, "shape": 2}))
    with pytest.raises(ValueError, match="regenerate"):
        cls(PARAMETERS)._validate_storage_calibration()


@pytest.mark.parametrize("name", CATALOG)
@pytest.mark.parametrize("sample", [[-1, 2], [np.inf], [], np.ma.array([1, 2], mask=[True, False])])
def test_generated_input_validation(name, sample):
    with pytest.raises(ValueError):
        getattr(module, name)(PARAMETERS).execute_statistic(sample)


@pytest.mark.parametrize("name", implementation._EDF_STATISTICS)
def test_edf_matches_independent_cdf(name):
    cdf = stats.exponweib(1.4, 2.1, scale=1.3).cdf(SAMPLE)
    expected = implementation._EDF_STATISTICS[name].algorithm.do_execute_statistic(cdf)
    original = SAMPLE.copy()
    actual = getattr(module, name)(PARAMETERS).execute_statistic(SAMPLE)
    assert actual == pytest.approx(expected)
    np.testing.assert_array_equal(SAMPLE, original)


@pytest.mark.parametrize("name", implementation._FREQUENCY_STATISTICS)
def test_frequency_supports_explicit_bins(name):
    edges = (0, 0.5, 1.5, np.inf)
    observed = np.histogram(SAMPLE, edges)[0]
    expected = len(SAMPLE) * np.diff(stats.exponweib(1.4, 2.1, scale=1.3).cdf(edges))
    reference = implementation._FREQUENCY_STATISTICS[name].algorithm.do_execute_statistic(
        observed, expected
    )
    criterion = getattr(module, name)(PARAMETERS, binning=ExplicitBins(edges))
    assert criterion.execute_statistic(SAMPLE) == pytest.approx(reference)


class TestPublicPackage:
    @pytest.mark.parametrize("name", module.__all__)
    def test_exports_and_old_pickle_paths(self, name):
        cls = getattr(module, name)
        assert cls is getattr(implementation, name)
        assert cls.__module__ == module.__name__
        reference = f"c{module.__name__}\n{name}\n.".encode()
        assert pickle.loads(reference) is cls  # noqa: S301
        assert pickle.loads(pickle.dumps(cls)) is cls  # noqa: S301

    def test_stub_matches_catalog_and_constructor_options(self):
        import ast
        from pathlib import Path

        stub = Path(module.__file__).with_name("exponentiated_weibull_statistics.pyi")
        tree = ast.parse(stub.read_text())
        classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
        assert set(classes) == set(module.__all__)
        for name in CATALOG:
            methods = {m.name: m for m in classes[name].body if isinstance(m, ast.FunctionDef)}
            assert "supported_hypotheses" in methods
            assert "execute_statistic" in methods
            if name in implementation._FREQUENCY_STATISTICS:
                assert [arg.arg for arg in methods["__init__"].args.kwonlyargs] == ["binning"]
        ks = classes["KolmogorovSmirnovExponentiatedWeibullGofStatistic"]
        constructor = next(
            m for m in ks.body if isinstance(m, ast.FunctionDef) and m.name == "__init__"
        )
        assert [arg.arg for arg in constructor.args.kwonlyargs] == ["alternative_type", "mode"]
