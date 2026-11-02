"""Fitted Weibull catalogs, numerical references and calibration contracts."""

import ast
import inspect
import pickle
from pathlib import Path
from typing import get_origin

import numpy as np
import pytest
from scipy import stats

from pysatl_criterion.distribution import WeibullDistributionDescriptor as Weibull
from pysatl_criterion.statistics.goodness_of_fit import edf, frequency
from pysatl_criterion.statistics.goodness_of_fit.binning import (
    EqualProbabilityBins,
    ExplicitBins,
)
from pysatl_criterion.statistics.goodness_of_fit.continuous import weibull as module
from pysatl_criterion.statistics.goodness_of_fit.continuous._edf_factory import (
    export_edf_statistics,
)
from pysatl_criterion.statistics.goodness_of_fit.continuous._frequency_factory import (
    export_frequency_statistics,
)
from pysatl_criterion.statistics.goodness_of_fit.continuous.weibull import weibull_statistics
from pysatl_criterion.utils.generator import get_hypothesis_generator
from pysatl_criterion.utils.statistic import get_available_criteria, get_criterion_by_code


SAMPLE = np.array([0.14, 0.27, 0.5, 0.66, 0.84, 1.1, 1.39, 1.72, 2.03, 2.65, 3.7])
CATALOG = {**weibull_statistics._EDF_STATISTICS, **weibull_statistics._FREQUENCY_STATISTICS}


def criterion(name, **kwargs):
    return getattr(module, name)(Weibull.DEFAULT.parse({}), **kwargs)


@pytest.mark.parametrize("name", CATALOG)
def test_identity_hypothesis_and_registry(name):
    cls = getattr(module, name)
    assert not inspect.isabstract(cls)
    assert cls.__module__ == module.__name__
    assert name in module.__all__
    assert pickle.loads(pickle.dumps(cls)) is cls  # noqa: S301 - local class reference
    assert get_criterion_by_code(cls.code()) is cls
    assert get_available_criteria(Weibull.type()).count(cls) == 1
    value = criterion(name)
    assert not value.hypothesis().parameter_values
    with pytest.raises(ValueError, match="Unsupported hypothesis"):
        cls(Weibull.DEFAULT.parse({"shape": 2, "scale": 3}))
    with pytest.raises(ValueError, match="regenerate"):
        value._validate_storage_calibration()
    assert get_hypothesis_generator(value).parameters() == {"shape": 1, "scale": 1}


def test_catalog_is_complete_without_explicit_duplicates():
    for source, catalog, base in (
        (edf, weibull_statistics._EDF_STATISTICS, edf.AbstractEDFStatistic),
        (frequency, weibull_statistics._FREQUENCY_STATISTICS, frequency.AbstractFrequencyStatistic),
    ):
        algorithms = {
            c
            for c in vars(source).values()
            if inspect.isclass(c)
            and get_origin(c) is None
            and issubclass(c, base)
            and c is not base
        }
        assert {s.algorithm for s in catalog.values()} == algorithms
    tree = ast.parse(Path(weibull_statistics.__file__).read_text())
    assert not set(CATALOG).intersection(n.name for n in tree.body if isinstance(n, ast.ClassDef))
    criteria = get_available_criteria(Weibull.type())
    assert len(criteria) == 33
    assert len({cls.code() for cls in criteria}) == len(criteria)
    assert (
        get_criterion_by_code("SB_WEIBULL_GOODNESS_OF_FIT")
        is module.ShapiroBrainWeibullGofStatistic
    )
    assert not hasattr(module, "SbWeibullGofStatistic")
    assert module.AndersonDarlingWeibullGofStatistic.short_code() == "AD"
    assert module.LiaoShimokawaWeibullGofStatistic.short_code() == "LS"


@pytest.mark.parametrize("name", CATALOG)
def test_refits_without_mutation_and_is_invariant(name):
    test = criterion(name)
    sample = SAMPLE.copy()
    baseline = test.execute_statistic(sample, common_keyword=True)
    assert np.isfinite(baseline)
    assert test.execute_statistic(sample[::-1]) == pytest.approx(baseline)
    for changed in (sample * 1e150, sample * 1e-150, sample**0.05, sample**5):
        assert test.execute_statistic(changed) == pytest.approx(baseline, rel=1e-8, abs=1e-8)
    other = np.array([0.2, 0.4, 0.41, 0.42, 0.43, 1.1, 1.15])
    assert test.execute_statistic(other) == criterion(name).execute_statistic(other)
    np.testing.assert_array_equal(sample, SAMPLE)
    assert not test._parameters


@pytest.mark.parametrize("name", CATALOG)
@pytest.mark.parametrize(
    "sample",
    [
        [],
        [1],
        [1, 1],
        [0, 1],
        [-1, 2],
        [np.nan, 1],
        [np.inf, 1],
        [[1, 2]],
        [1j, 2],
        np.ma.array([1.0, 2.0], mask=[True, False]),
    ],
)
def test_invalid_sample(name, sample):
    with pytest.raises(ValueError):
        criterion(name).execute_statistic(sample)


@pytest.mark.parametrize("sample", [SAMPLE, np.array([1e-300, 1, 1e300]), SAMPLE**0.005])
def test_independent_scipy_edf_references(sample):
    logs = np.log(np.sort(sample))
    location, scale = stats.gumbel_l.fit(logs)
    z = (logs - location) / scale
    u = stats.gumbel_l.cdf(z)
    n = len(sample)
    indices = np.arange(1, n + 1)
    expected = {
        "KolmogorovSmirnovDPlus": np.max(indices / n - u),
        "KolmogorovSmirnovDMinus": np.max(u - (indices - 1) / n),
        "KolmogorovSmirnov": stats.kstest(z, "gumbel_l").statistic,
        "AndersonDarling": stats.anderson(logs, dist="gumbel_l").statistic,
        "CrammerVonMises": stats.cramervonmises(z, "gumbel_l").statistic,
        "LiaoShimokawa": np.sum(
            np.maximum(indices / n - u, u - (indices - 1) / n) / np.sqrt(u * (1 - u))
        )
        / np.sqrt(n),
    }
    for prefix, reference in expected.items():
        assert criterion(prefix + "WeibullGofStatistic").execute_statistic(sample) == pytest.approx(
            reference, rel=1e-7, abs=1e-8
        )


@pytest.mark.parametrize("name", weibull_statistics._EDF_STATISTICS)
def test_shared_kernel_agrees_with_independent_fit(name):
    loc, scale = stats.gumbel_l.fit(np.log(SAMPLE))
    u = stats.gumbel_l.cdf((np.log(np.sort(SAMPLE)) - loc) / scale)
    expected = weibull_statistics._EDF_STATISTICS[name].algorithm.do_execute_statistic(u)
    assert criterion(name).execute_statistic(SAMPLE) == pytest.approx(expected, rel=1e-7)


@pytest.mark.parametrize("name", weibull_statistics._FREQUENCY_STATISTICS)
def test_frequency_independent_fit_and_explicit_binning(name):
    loc, scale = stats.gumbel_l.fit(np.log(SAMPLE))
    fitted = stats.weibull_min(1 / scale, scale=np.exp(loc))
    edges = [0, 0.4, 1, 2, np.inf]
    observed = np.histogram(SAMPLE, edges)[0]
    expected = len(SAMPLE) * np.diff(fitted.cdf(edges))
    value = criterion(name, binning=ExplicitBins(tuple(edges)))
    reference = weibull_statistics._FREQUENCY_STATISTICS[name].algorithm.do_execute_statistic(
        observed, expected
    )
    assert value.execute_statistic(SAMPLE) == pytest.approx(reference, rel=1e-7)
    if name.startswith("Chi2Pearson"):
        assert value.execute_statistic(SAMPLE) == pytest.approx(
            stats.chisquare(observed, expected).statistic
        )
    with pytest.raises(ValueError, match="external calibration"):
        get_hypothesis_generator(value)
    assert value.configuration() != criterion(name).configuration()
    alternate = criterion(name, binning=EqualProbabilityBins(count=3))
    observed = np.histogram(fitted.cdf(SAMPLE), np.linspace(0, 1, 4))[0]
    reference = weibull_statistics._FREQUENCY_STATISTICS[name].algorithm.do_execute_statistic(
        observed, np.full(3, len(SAMPLE) / 3)
    )
    assert alternate.execute_statistic(SAMPLE) == pytest.approx(reference)


@pytest.mark.parametrize(
    "algorithm",
    [
        edf.AbstractAndersonDarlingStatistic,
        edf.AbstractLiaoShimokawaStatistic,
        edf.AbstractZhangKStatistic,
        edf.AbstractZhangAStatistic,
        edf.AbstractZhangCStatistic,
    ],
)
def test_log_kernels_retain_tail_precision(algorithm):
    from pysatl_criterion.statistics.goodness_of_fit.continuous._weibull_common import (
        _log_probabilities,
    )

    logcdf, logsf = _log_probabilities(np.array([-700.0, -1.0, 0.0, 5.0]))
    assert np.exp(logcdf[-1]) == 1
    assert np.isfinite(algorithm.do_execute_log_statistic(logcdf, logsf))


def test_composite_factories_require_explicit_fitting_callbacks():
    for export, catalog in (
        (export_edf_statistics, weibull_statistics._EDF_STATISTICS),
        (export_frequency_statistics, weibull_statistics._FREQUENCY_STATISTICS),
    ):
        namespace = {"__name__": "example"}
        with pytest.raises(ValueError, match="fully specified"):
            export(namespace, module.AbstractWeibullGofStatistic, catalog)
        assert namespace == {"__name__": "example"}


def test_monte_carlo_refits_each_replicate(mocker):
    from pysatl_criterion.hypothesis_testing.limit_distribution.base import (
        MonteCarloLimitDistributionResolver,
    )

    samples = [SAMPLE, np.linspace(0.1, 5, len(SAMPLE))]
    sampler = mocker.patch(
        "pysatl_criterion.generator.generators.generate_weibull", side_effect=samples
    )
    result = MonteCarloLimitDistributionResolver(2).resolve(
        criterion("AndersonDarlingWeibullGofStatistic"), sample_size=len(SAMPLE)
    )
    expected = [stats.anderson(np.log(sample), dist="gumbel_l").statistic for sample in samples]
    np.testing.assert_allclose(result, expected)
    assert sampler.call_count == 2
    assert all(call.kwargs["shape"] == call.kwargs["scale"] == 1 for call in sampler.call_args_list)


def test_public_stub_matches_exports():
    tree = ast.parse(Path(module.__file__).with_name("weibull_statistics.pyi").read_text())
    classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    aliases = {
        node.targets[0].id
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name)
    }
    assert classes | aliases == set(module.__all__)


def test_hypothesis_support_is_declared_per_criterion():
    assert "supported_hypotheses" not in module.AbstractWeibullGofStatistic.__dict__
    for name in module.__all__:
        cls = getattr(module, name)
        if inspect.isabstract(cls):
            continue
        assert "supported_hypotheses" in cls.__dict__
        if name in CATALOG:
            assert cls.supported_hypotheses() == CATALOG[name].supported_hypotheses
        assert cls(Weibull.DEFAULT.parse({})).hypothesis().parameters() == {}
        for values in ({"shape": 2}, {"scale": 3}, {"shape": 2, "scale": 3}):
            with pytest.raises(ValueError, match="Unsupported"):
                cls(Weibull.DEFAULT.parse(values))


@pytest.mark.parametrize("name", module.__all__)
def test_package_exports_preserve_identity_and_old_pickles(name):
    cls = getattr(module, name)
    assert cls is getattr(weibull_statistics, name)
    assert cls.__module__ == module.__name__
    old_reference = f"c{module.__name__}\n{name}\n.".encode()
    assert pickle.loads(old_reference) is cls  # noqa: S301 - local class reference
    assert pickle.loads(pickle.dumps(cls)) is cls  # noqa: S301 - local class reference
