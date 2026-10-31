"""Compiled probability transform and dispatch for all exponentiated Weibull tests."""

import ast
from pathlib import Path

import numpy as np
import pytest

from pysatl_criterion.statistics.goodness_of_fit.continuous import exponentiated_weibull as ew
from pysatl_criterion.statistics.goodness_of_fit.continuous.exponentiated_weibull import (
    exponentiated_weibull_statistics as implementation,
)


class TestExponentiatedWeibullNumba:
    @pytest.mark.parametrize(
        "name", [*implementation._EDF_STATISTICS, *implementation._FREQUENCY_STATISTICS]
    )
    def test_every_criterion_uses_compiled_transform_and_statistic(self, name, monkeypatch):
        parameters = implementation.Distribution.DEFAULT.parse(
            {"exponent": 1.4, "shape": 2.1, "scale": 1.3}
        )
        criterion = getattr(ew, name)(parameters)
        sample = np.random.default_rng(42).weibull(2.1, 100)
        original = sample.copy()
        compiled = criterion.execute_statistic(sample)
        transform = ew.AbstractExponentiatedWeibullGofStatistic.do_log_probabilities
        assert transform.nopython_signatures
        spec = {**implementation._EDF_STATISTICS, **implementation._FREQUENCY_STATISTICS}[name]
        kernel = getattr(
            spec.algorithm, "do_execute_log_statistic", spec.algorithm.do_execute_statistic
        )
        assert kernel.nopython_signatures
        monkeypatch.setattr(
            ew.AbstractExponentiatedWeibullGofStatistic,
            "do_log_probabilities",
            staticmethod(transform.py_func),
        )
        python = criterion.execute_statistic(sample)
        assert compiled == pytest.approx(python, rel=1e-12, abs=1e-12)
        np.testing.assert_array_equal(sample, original)

    @pytest.mark.parametrize(
        "exponent,shape,scale,sample",
        [
            (1.0, 1.0, 1.0, [0.0, 1e-300, 0.1, 1.0, 36.0, 40.0, 1000.0, np.inf, np.nan]),
            (1e-320, 1.0, 1.0, [0.0, 0.1, 1.0, 2.0, 10.0, 1000.0]),
            (1e-308, 1e308, 1.0, [0.01, 0.1, 0.4, 0.8, 1.0, 2.0]),
            (1e200, 1.0, 1.0, [0.0, 1.0, 460.0, 461.0, 462.0, 1000.0]),
            (2.3, 0.1, 1e100, [0.0, 1e-200, 0.1, 1e100, 1e300]),
        ],
    )
    def test_compiled_probability_boundaries_match_python(self, exponent, shape, scale, sample):
        kernel = ew.AbstractExponentiatedWeibullGofStatistic.do_log_probabilities
        sample = np.asarray(sample, dtype=float)
        compiled = kernel(sample, exponent, shape, scale)
        with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
            python = kernel.py_func(sample, exponent, shape, scale)
        for actual, expected in zip(compiled, python, strict=True):
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-13)
        assert kernel.nopython_signatures

    def test_helpers_are_class_methods(self):
        tree = ast.parse(Path(implementation.__file__).read_text())
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in tree.body
        )
