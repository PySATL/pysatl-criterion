"""Compiled Gamma kernels preserve mathematical results and input arrays."""

import ast
from pathlib import Path

import numpy as np
import pytest
from numba.core.registry import CPUDispatcher

from pysatl_criterion.statistics.goodness_of_fit.continuous.gamma import gamma_statistics as gamma
from tests.parameter_cases import parameters_for


LOCAL_CLASSES = [
    getattr(gamma, name)
    for name in gamma.__all__
    if not name.startswith("Abstract")
    and name not in gamma._EDF_STATISTICS
    and name not in gamma._FREQUENCY_STATISTICS
]


class TestGammaNumba:
    @pytest.mark.parametrize("cls", LOCAL_CLASSES)
    @pytest.mark.parametrize(
        "sample",
        [
            np.array([0.2, 0.7, 1.3, 2.1, 3.4]),
            np.array([0.0, 0.2, 0.2, 5.0, 40.0]),
            np.array([40.0, 41.0, 42.0]),
        ],
    )
    def test_compiled_matches_python_and_is_used(self, cls, sample, monkeypatch):
        kernel = cls.do_execute_statistic
        assert isinstance(kernel, CPUDispatcher)
        statistic = cls(parameters_for(cls))
        before = sample.copy()
        actual = statistic.execute_statistic(sample)
        assert kernel.nopython_signatures
        calls = []

        def python_kernel(*args):
            calls.append(args)
            copies = [array.copy() for array in args]
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                result = kernel.py_func(*args)
            for array, copy in zip(args, copies, strict=True):
                np.testing.assert_array_equal(array, copy)
            return result

        monkeypatch.setattr(cls, "do_execute_statistic", staticmethod(python_kernel))
        assert statistic.execute_statistic(sample) == pytest.approx(actual)
        assert len(calls) == 1
        np.testing.assert_array_equal(sample, before)

    def test_all_criteria_use_compiled_kernels(self):
        for name in gamma.__all__:
            if name.startswith("Abstract"):
                continue
            cls = getattr(gamma, name)
            assert isinstance(cls.do_execute_statistic, CPUDispatcher)
            cls(parameters_for(cls)).execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
            kernel = getattr(cls, "do_execute_log_statistic", cls.do_execute_statistic)
            assert kernel.nopython_signatures

    def test_no_module_level_functions(self):
        tree = ast.parse(Path(gamma.__file__).read_text())
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in tree.body
        )

    @pytest.mark.parametrize(
        "values",
        [
            np.array([0.0]),
            np.zeros(5),
            np.array([0.0, 0.1, 0.2, 1.0]),
            np.array([0.0, np.nextafter(0.1, 0), 0.1, np.nextafter(0.1, 1), 1.0]),
        ],
    )
    def test_graph_strict_boundary_against_adjacency(self, values):
        import networkx as nx

        distance = np.ptp(values) / 10
        adjacency = np.abs(values[:, None] - values[None, :]) < distance
        np.fill_diagonal(adjacency, False)
        graph = nx.from_numpy_array(adjacency)
        references = {
            gamma.GraphEdgesNumberGammaGofStatistic: graph.number_of_edges(),
            gamma.GraphAverageDegreeGammaGofStatistic: 2 * graph.number_of_edges() / len(values),
            gamma.GraphMaxDegreeGammaGofStatistic: max(dict(graph.degree()).values()),
            gamma.GraphConnectedComponentsGammaGofStatistic: nx.number_connected_components(graph),
            gamma.GraphCliqueNumberGammaGofStatistic: max(map(len, nx.find_cliques(graph))),
            gamma.GraphIndependenceNumberGammaGofStatistic: max(
                map(len, nx.find_cliques(nx.complement(graph)))
            ),
        }
        for cls, expected in references.items():
            assert cls.do_execute_statistic(values) == expected
