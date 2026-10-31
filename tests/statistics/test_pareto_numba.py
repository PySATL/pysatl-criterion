"""Compiled Pareto execution paths and exact fallback regressions."""

import ast
from fractions import Fraction
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
from numba.core.registry import CPUDispatcher

from pysatl_criterion.distribution import ParetoDistributionDescriptor as Pareto
from pysatl_criterion.statistics.goodness_of_fit.continuous import pareto
from pysatl_criterion.statistics.goodness_of_fit.continuous.pareto import pareto_statistics
from tests.parameter_cases import parameters_for


NAMES = [name for name in pareto.__all__ if not name.startswith("Abstract")]
SAMPLE = np.array([1.1, 1.4, 1.9, 2.7, 4.2, 8.0])


@pytest.mark.parametrize("name", NAMES)
def test_public_execution_compiles_nopython_kernel(name):
    cls = getattr(pareto, name)
    criterion = cls(parameters_for(cls))
    value = criterion.execute_statistic(SAMPLE)
    assert np.isfinite(value) or np.isposinf(value)
    # Logarithmic EDF kernels supersede their CDF-only kernels on the public path.
    kernel = getattr(cls, "do_execute_log_statistic", cls.do_execute_statistic)
    assert isinstance(kernel, CPUDispatcher)
    assert kernel.nopython_signatures
    assert kernel.targetoptions["nopython"]
    assert kernel._cache is not None


def test_helpers_are_class_methods_and_compile():
    cls = pareto.AbstractParetoGofStatistic
    tree = ast.parse(Path(pareto_statistics.__file__).read_text())
    assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in tree.body)
    criterion = pareto.AndersonDarlingParetoGofStatistic(
        Pareto.DEFAULT.parse({"shape": 2, "scale": 1})
    )
    criterion.execute_statistic(SAMPLE)
    for name in ("_log_ratio", "_log_probabilities"):
        assert isinstance(cls.__dict__[name], staticmethod)
        assert getattr(cls, name).nopython_signatures


def test_obradovic_common_path_avoids_fraction_fallback(mocker):
    criterion = pareto.ObradovicParetoGofStatistic(Pareto.DEFAULT.parse({"scale": 1}))
    fallback = mocker.patch.object(
        criterion, "_execute_exact_statistic", side_effect=AssertionError
    )
    # Compare with a rational brute-force reference, independently of the compiled kernel.
    x = [Fraction(float(value)) for value in SAMPLE]
    ratios = [max(a / b, b / a) for a, b in combinations(x, 2)]
    expected = sum(
        Fraction(sum(r <= t for r in ratios), len(ratios))
        - Fraction(sum(v <= t for v in x), len(x))
        for t in x
    ) / len(x)
    assert criterion.execute_statistic(SAMPLE) == pytest.approx(float(expected))
    fallback.assert_not_called()


@pytest.mark.parametrize(
    "sample,scale",
    [
        ([1, 2, 2, 4], 1),
        ([1, 5, 25, 125], 1),
        ([2] * 4, 1),
        ([1e-300, 1e-100, 1e100, 1e300], 1e-300),
    ],
)
def test_obradovic_boundary_fallback_remains_exact(sample, scale, mocker):
    criterion = pareto.ObradovicParetoGofStatistic(Pareto.DEFAULT.parse({"scale": scale}))
    fallback = mocker.spy(criterion, "_execute_exact_statistic")
    z = [Fraction(float(x)) / Fraction(scale) for x in sample]
    ratios = [max(a / b, b / a) for a, b in combinations(z, 2)]
    expected = sum(
        Fraction(sum(r <= t for r in ratios), len(ratios))
        - Fraction(sum(v <= t for v in z), len(z))
        for t in z
    ) / len(z)
    assert criterion.execute_statistic(sample) == pytest.approx(float(expected))
    # Constant data above scale have no ambiguous comparison; the other cases do.
    if len(set(sample)) > 1:
        fallback.assert_called_once()


@pytest.mark.parametrize("name", ["Greenwood", "LequesneKl"])
def test_compiled_and_python_kernels_agree(name):
    cls = getattr(pareto, name + "ParetoGofStatistic")
    criterion = cls(Pareto.DEFAULT.parse({}))
    for x in (SAMPLE, SAMPLE * 1e250, 1e300 + np.arange(8) * np.spacing(1e300)):
        y = criterion._fitted_logs(x)
        args = (y,) if name == "Greenwood" else (x, y, 2)
        assert cls.do_execute_statistic(*args) == pytest.approx(
            cls.do_execute_statistic.py_func(*args)
        )
