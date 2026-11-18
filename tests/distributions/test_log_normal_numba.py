"""Compiled log-normal kernels preserve their Python numerical results."""

import ast
from functools import partial
from pathlib import Path

import numpy as np
import pytest

from pysatl_criterion.distribution import LogNormalDistributionDescriptor as Distribution
from pysatl_criterion.statistics.goodness_of_fit.continuous import log_normal as ln
from pysatl_criterion.statistics.goodness_of_fit.continuous import normal
from pysatl_criterion.statistics.goodness_of_fit.continuous.log_normal import (
    log_normal_statistics,
)
from tests.parameter_cases import parameters_for
from tests.statistics.test_normal_numba import KERNEL_NAMES


class TestLogNormalNumba:
    @pytest.mark.parametrize(
        "cls",
        [
            ln.QuesenberryMillerLogNormalGofStatistic,
            ln.KLSupremumLogNormalGoFStatistic,
            ln.KLIntegralLogNormalGoFStatistic,
        ],
    )
    @pytest.mark.parametrize("tied", [False, True])
    def test_public_execution_uses_cached_nopython_kernels(self, cls, tied, monkeypatch):
        logs = np.random.default_rng(613).normal(size=40)
        if tied:
            logs = np.round(logs)
        sample = np.exp(logs)
        criterion = cls(Distribution.SHAPE_SCALE.parse({"s": 0.7, "scale": 1.2}))
        actual = criterion.execute_statistic(sample)
        owners_and_names = []
        if cls is not ln.QuesenberryMillerLogNormalGofStatistic:
            owners_and_names.extend(
                (ln.AbstractLogNormalGofStatistic, name)
                for name in ("_kl_coefficients", "_kl_ratio")
            )
        owners_and_names.append((cls, "do_execute_statistic"))
        for owner, name in owners_and_names:
            kernel = getattr(owner, name)
            assert isinstance(vars(owner)[name], staticmethod)
            assert kernel.nopython_signatures
            assert kernel.targetoptions["nopython"]
            assert kernel._cache.__class__.__name__ == "FunctionCache"
            monkeypatch.setattr(owner, name, staticmethod(kernel.py_func))
        assert actual == pytest.approx(criterion.execute_statistic(sample), rel=1e-9, abs=1e-10)

    @staticmethod
    def _observe(kernel, calls, *args, **kwargs):
        actual = kernel(*args, **kwargs)
        expected = kernel.py_func(*args, **kwargs)
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=2e-10)
        assert kernel.nopython_signatures
        assert kernel.targetoptions["nopython"]
        assert not kernel.targetoptions.get("fastmath", False)
        assert kernel._cache.__class__.__name__ == "FunctionCache"
        calls.append(kernel)
        return actual

    @pytest.mark.parametrize("name", [n for n in ln.__all__ if not n.startswith("Abstract")])
    @pytest.mark.parametrize("tied", [False, True])
    def test_every_public_statistic_reaches_a_cached_kernel(self, name, tied, monkeypatch):
        cls = getattr(ln, name)
        owner_class = cls
        if (
            name not in log_normal_statistics._EDF_STATISTICS
            and name not in log_normal_statistics._FREQUENCY_STATISTICS
        ):
            normal_name = name.replace("LogNormal", "Normality")
            if normal_name in normal.__all__:
                owner_class = getattr(normal, normal_name)
        names = [KERNEL_NAMES.get(cls.short_code(), "do_execute_statistic")]
        if cls.short_code() == "DAP":
            names = ["skew_test", "kurtosis_test"]
        if name in log_normal_statistics._EDF_STATISTICS and hasattr(
            cls, "do_execute_log_statistic"
        ):
            names = ["do_execute_log_statistic"]
        calls = []
        for method in names:
            owner = next(base for base in owner_class.__mro__ if method in vars(base))
            kernel = getattr(owner, method)
            assert isinstance(vars(owner)[method], staticmethod)
            monkeypatch.setattr(owner, method, staticmethod(partial(self._observe, kernel, calls)))
        logs = np.random.default_rng(132).normal(size=32)
        if tied:
            logs = np.round(logs, 1)
        sample = np.exp(logs)
        original = sample.copy()
        assert np.isfinite(cls(parameters_for(cls)).execute_statistic(sample))
        np.testing.assert_array_equal(sample, original)
        assert len(calls) >= len(names)

    def test_helpers_are_defined_inside_classes(self):
        tree = ast.parse(Path(log_normal_statistics.__file__).read_text())
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in tree.body
        )
