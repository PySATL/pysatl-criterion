"""Log-normal factories use the specified null, not a fitted normal distribution."""

import ast
import inspect
import pickle
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from pysatl_criterion.distribution import LogNormalDistributionDescriptor as Distribution
from pysatl_criterion.statistics.alternative import AlternativeType
from pysatl_criterion.statistics.goodness_of_fit.binning import EqualProbabilityBins
from pysatl_criterion.statistics.goodness_of_fit.continuous import log_normal
from pysatl_criterion.statistics.goodness_of_fit.continuous.log_normal import (
    log_normal_statistics,
)


PARAMETERS = Distribution.DEFAULT.parse({"mu": 0.8, "s": 0.6})
SAMPLE = np.exp(np.array([-1.7, -0.4, 0.1, 0.3, 0.9, 1.8, 2.5]))


@pytest.mark.parametrize("name", log_normal.__all__)
def test_each_concrete_class_declares_its_own_hypotheses(name):
    cls = getattr(log_normal, name)
    if inspect.isabstract(cls):
        assert "supported_hypotheses" not in cls.__dict__
        return
    assert "supported_hypotheses" in cls.__dict__
    assert cls.supported_hypotheses()
    for schema in Distribution.parameterizations():
        for fixed in ({}, {Distribution.LOG_SCALE: 0.6}):
            if not fixed and hasattr(cls, "_standard_calibration_parameters"):
                assert not cls(schema.parse(fixed)).hypothesis().parameters()
            else:
                with pytest.raises(ValueError, match="Unsupported"):
                    cls(schema.parse(fixed))
    assert cls is getattr(log_normal_statistics, name)
    assert cls.__module__ == log_normal.__name__
    assert pickle.loads(pickle.dumps(cls)) is cls  # noqa: S301


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("name,spec", log_normal_statistics._EDF_STATISTICS.items())
def test_edf_matches_kernel_on_specified_probabilities(name, spec, native):
    parameters = (
        Distribution.convert_parameters(PARAMETERS, Distribution.SHAPE_SCALE)
        if native
        else PARAMETERS
    )
    criterion = getattr(log_normal, name)(parameters)
    reference = stats.lognorm(s=0.6, scale=np.exp(0.8))
    probabilities = reference.cdf(np.sort(SAMPLE))
    expected = spec.algorithm.do_execute_statistic(probabilities)
    if name == "ZhangWuALogNormalGofStatistic":
        expected = 10 * expected - 32
    assert criterion.execute_statistic(SAMPLE[::-1]) == pytest.approx(expected)
    assert criterion.hypothesis().parameter_values is parameters
    assert spec.algorithm in type(criterion).__bases__


@pytest.mark.parametrize("name,spec", log_normal_statistics._FREQUENCY_STATISTICS.items())
def test_frequency_uses_original_sample_and_requested_bins(name, spec):
    binning = EqualProbabilityBins(3)
    criterion = getattr(log_normal, name)(PARAMETERS, binning=binning)
    edges = stats.lognorm(s=0.6, scale=np.exp(0.8)).ppf([0, 1 / 3, 2 / 3, 1])
    observed = np.histogram(SAMPLE, bins=edges)[0].astype(float)
    expected = np.full(3, len(SAMPLE) / 3)
    assert criterion.execute_statistic(SAMPLE) == pytest.approx(
        spec.algorithm.do_execute_statistic(observed, expected)
    )
    assert criterion.binning is binning
    assert "binning" in criterion.configuration()
    assert spec.algorithm in type(criterion).__bases__


@pytest.mark.parametrize(
    "direction,alternative",
    [
        (AlternativeType.TWO_TAILED, "two-sided"),
        (AlternativeType.RIGHT, "greater"),
        (AlternativeType.LEFT, "less"),
    ],
)
def test_ks_preserves_direction_options(direction, alternative):
    criterion = log_normal.KolmogorovSmirnovLogNormalGofStatistic(
        PARAMETERS, alternative_type=direction, mode="asymp"
    )
    reference = stats.kstest(
        SAMPLE, stats.lognorm(s=0.6, scale=np.exp(0.8)).cdf, alternative=alternative
    )
    assert criterion.execute_statistic(SAMPLE) == pytest.approx(reference.statistic)
    assert criterion.mode == "asymp"


def test_logarithmic_tail_kernel_avoids_rounded_cdf():
    criterion = log_normal.AndersonDarlingLogNormalGofStatistic(
        Distribution.DEFAULT.parse({"mu": 0, "s": 1})
    )
    sample = np.exp([8, 9, 10, 11])
    ranks = np.arange(1, 5)
    reference = stats.lognorm(s=1)
    expected = (
        -4
        - np.sum((2 * ranks - 1) * (reference.logcdf(sample) + reference.logsf(sample[::-1]))) / 4
    )
    assert np.isfinite(expected)
    assert criterion.execute_statistic(sample) == pytest.approx(expected)


def test_stub_matches_runtime():
    tree = ast.parse(Path(log_normal.__file__).with_name("log_normal_statistics.pyi").read_text())
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    assert set(classes) == set(log_normal.__all__)
    for name, node in classes.items():
        cls = getattr(log_normal, name)
        assert [ast.unparse(b).split(".")[-1] for b in node.bases] == [
            b.__name__ for b in cls.__bases__
        ]
