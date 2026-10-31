"""Generated Beta catalogs and independent numerical references."""

import ast
import inspect
import pickle
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from pysatl_criterion.distribution import BetaDistributionDescriptor as Beta
from pysatl_criterion.statistics.goodness_of_fit import edf, frequency
from pysatl_criterion.statistics.goodness_of_fit.binning import EqualProbabilityBins, FixedCount
from pysatl_criterion.statistics.goodness_of_fit.continuous import beta
from pysatl_criterion.statistics.goodness_of_fit.continuous.beta import beta_statistics


CATALOG = {**beta_statistics._EDF_STATISTICS, **beta_statistics._FREQUENCY_STATISTICS}
SAMPLE = np.array([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])


def test_complete_catalog_without_duplicates_or_aliases():
    for source, catalog, base in (
        (edf, beta_statistics._EDF_STATISTICS, edf.AbstractEDFStatistic),
        (frequency, beta_statistics._FREQUENCY_STATISTICS, frequency.AbstractFrequencyStatistic),
    ):
        algorithms = {
            cls
            for cls in vars(source).values()
            if inspect.isclass(cls) and issubclass(cls, base) and cls is not base
        }
        assert {spec.algorithm for spec in catalog.values()} == algorithms
    tree = ast.parse(Path(beta_statistics.__file__).read_text())
    assert not set(CATALOG).intersection(
        node.name for node in tree.body if isinstance(node, ast.ClassDef)
    )
    classes = [getattr(beta, name) for name in beta.__all__]
    assert len(classes) == len(set(classes))
    concrete = [cls for cls in classes if not inspect.isabstract(cls)]
    assert len({cls.code() for cls in concrete}) == len(concrete)


@pytest.mark.parametrize("name", CATALOG)
def test_generated_contract(name):
    cls = getattr(beta, name)
    assert cls.__module__ == beta.__name__
    assert name in beta.__all__
    assert pickle.loads(pickle.dumps(cls)) is cls  # noqa: S301
    for fixed in ({}, {"a": 2}, {"b": 5}):
        with pytest.raises(ValueError):
            cls(Beta.DEFAULT.parse(fixed))
    criterion = cls(Beta.DEFAULT.parse({"a": 2, "b": 5}))
    sample = SAMPLE.copy()
    # Empty bins may give an infinite divergence for negative powers.
    assert criterion.execute_statistic(sample) >= 0
    np.testing.assert_array_equal(sample, SAMPLE)
    for invalid in ([], [-0.1], [1.1], [np.nan], [[0.2]], [0.2 + 1j]):
        with pytest.raises(ValueError):
            criterion.execute_statistic(invalid)


@pytest.mark.parametrize(
    "name,power",
    [
        ("Chi2PearsonBetaGofStatistic", 1),
        ("FreemanTukeyBetaGofStatistic", -0.5),
        ("ModifiedLogLikelihoodBetaGofStatistic", -1),
        ("NeymanChi2BetaGofStatistic", -2),
        ("CressieReadBetaGofStatistic", 2 / 3),
    ],
)
def test_frequency_against_scipy(name, power):
    criterion = getattr(beta, name)(
        Beta.DEFAULT.parse({"a": 2, "b": 5}),
        binning=EqualProbabilityBins(count=FixedCount(3)),
    )
    edges = stats.beta.ppf(np.linspace(0, 1, 4), 2, 5)
    observed = np.histogram(SAMPLE, edges)[0]
    expected = np.full(3, len(SAMPLE) / 3)
    reference = stats.power_divergence(observed, expected, lambda_=power).statistic
    assert criterion.execute_statistic(SAMPLE) == pytest.approx(reference)
