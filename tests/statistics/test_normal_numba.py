"""Exercise compiled kernels through all public normality statistics."""

import inspect
from itertools import combinations

import numpy as np
import pytest
from numba.core.registry import CPUDispatcher

from pysatl_criterion.statistics.goodness_of_fit.continuous import normal
from pysatl_criterion.statistics.goodness_of_fit.continuous.normal import normal_statistics
from tests.parameter_cases import parameters_for


CLASSES = [
    cls
    for _, cls in inspect.getmembers(normal, inspect.isclass)
    if cls.__module__ == normal.__name__ and not inspect.isabstract(cls)
]
KERNEL_NAMES = {
    "BHS": "do_execute_statistic",
    "BM2": "stat15",
    "BS": "stat17",
    "CC1": "_hermite_means",
    "CC2": "_hermite_means",
    "DLDMZEPD": "stat35",
    "FILLI": "_statistic",
    "GMG": "stat33",
    "HOSKING1": "stat10",
    "KURTOSIS": "kurtosis_test",
    "LG": "_statistic",
    "MI": "stat32",
    "RJ": "_statistic",
    "SH": "stat41",
    "SKEW": "skew_test",
}


@pytest.mark.parametrize("cls", CLASSES, ids=lambda cls: cls.short_code())
@pytest.mark.parametrize("tied", [False, True])
def test_public_execution_compiles_kernel_and_matches_python(cls, tied, monkeypatch):
    sample = np.random.default_rng(132).normal(size=32)
    if tied:
        sample = np.round(sample, 1)
    names = [KERNEL_NAMES.get(cls.short_code(), "do_execute_statistic")]
    if cls.short_code() == "DAP":
        names = ["skew_test", "kurtosis_test"]
    if cls.__name__ in normal_statistics._EDF_STATISTICS and hasattr(
        cls, "do_execute_log_statistic"
    ):
        names = ["do_execute_log_statistic"]
    calls = []

    def observe(kernel):
        def execute(*args, **kwargs):
            actual = kernel(*args, **kwargs)
            expected = kernel.py_func(*args, **kwargs)
            np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=2e-10)
            assert kernel.nopython_signatures
            assert kernel.targetoptions["nopython"]
            assert kernel._cache.__class__.__name__ == "FunctionCache"
            calls.append(kernel)
            return actual

        return execute

    for name in names:
        owner = next(base for base in cls.__mro__ if name in vars(base))
        kernel = getattr(owner, name)
        assert isinstance(vars(owner)[name], staticmethod)
        assert isinstance(kernel, CPUDispatcher)
        monkeypatch.setattr(owner, name, staticmethod(observe(kernel)))
    original = sample.copy()
    assert np.isfinite(cls(parameters_for(cls)).execute_statistic(sample))
    np.testing.assert_array_equal(sample, original)
    assert len(calls) == len(names)


@pytest.mark.parametrize("n", [10, 100, 10000, 100000])
def test_hosking_combination_weights_do_not_overflow_int64(n):
    import math

    for k in range(11):
        assert normal.AbstractNormalityGofStatistic._comb(n, k) == pytest.approx(
            float(math.comb(n, k)), rel=2e-15
        )


@pytest.mark.parametrize("radius", [0.5, 1.0, np.nextafter(1.0, 2.0), 4.0])
def test_graph_scan_preserves_strict_edges_with_ties(radius):
    x = np.array([0.0, 0.0, 0.5, 1.0, 2.0, 2.0, 4.0])
    for code, cls in enumerate(
        [
            normal.GraphEdgesNumberNormalityGofStatistic,
            normal.GraphMaxDegreeNormalityGofStatistic,
            normal.GraphAverageDegreeNormalityGofStatistic,
            normal.GraphConnectedComponentsNormalityGofStatistic,
            normal.GraphCliqueNumberNormalityGofStatistic,
            normal.GraphIndependenceNumberNormalityGofStatistic,
        ]
    ):
        criterion = cls(parameters_for(cls))
        graph = criterion._make_adjacency_list(x, radius)
        if code >= 4:
            expected = max(
                len(subset)
                for k in range(1, len(x) + 1)
                for subset in combinations(range(len(x)), k)
                if all((j in graph[i]) == (code == 4) for i, j in combinations(subset, 2))
            )
        else:
            expected = criterion.get_graph_stat(graph)
        assert criterion.do_execute_statistic(x, radius, code) == pytest.approx(expected)
