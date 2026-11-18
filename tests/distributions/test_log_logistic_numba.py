"""Compilation contracts and independent checks for optimized fitted criteria."""

import ast
from pathlib import Path

import numpy as np
import pytest
from numba.core.registry import CPUDispatcher
from scipy.optimize._numdiff import approx_derivative
from scipy.stats import logistic

from pysatl_criterion.distribution import LogLogisticDistributionDescriptor as Distribution
from pysatl_criterion.statistics.goodness_of_fit.continuous.log_logistic import (
    log_logistic_statistics as module,
)


class TestLogLogisticNumba:
    @pytest.mark.parametrize("mu,log_scale", [(0, 0), (1.2, -0.7), (-0.5, 1.4)])
    def test_censored_likelihood_and_gradient(self, mu, log_scale):
        y = np.array([-2.3, -0.2, 0.7, 1.1, 3.5])
        events = np.array([1.0, 0, 1, 1, 0])
        params = np.array([mu, log_scale])

        def reference(parameters):
            fitted = logistic(loc=parameters[0], scale=np.exp(parameters[1]))
            return -np.sum(np.where(events == 1, fitted.logpdf(y), fitted.logsf(y)))

        loss, gradient = module.NikulinLogLogisticGofStatistic._objective(params, y, events)
        assert loss == pytest.approx(reference(params), rel=1e-13)
        np.testing.assert_allclose(
            gradient,
            approx_derivative(reference, params, method="3-point").ravel(),
            rtol=1e-7,
            atol=1e-8,
        )

    def test_local_kernels_compile_in_nopython_mode(self):
        base = module.AbstractLogLogisticGofStatistic
        nikulin = module.NikulinLogLogisticGofStatistic
        mirvaliev = module.MirvalievLogLogisticGofStatistic
        sample = np.exp(np.linspace(-2, 2, 100))
        base._cdf_values(sample, 1.0, 2.0)
        base._log_probability_values(sample, 1.0, 2.0)
        nikulin(Distribution.DEFAULT.parse({})).execute_statistic(sample)
        mirvaliev(Distribution.DEFAULT.parse({})).execute_statistic(sample)
        for kernel in (
            base._cdf_values,
            base._log_probability_values,
            nikulin._likelihood_terms,
            nikulin._components,
            mirvaliev._residuals,
            mirvaliev._moment_integrand,
            mirvaliev.do_execute_statistic,
        ):
            assert isinstance(kernel, CPUDispatcher)
            assert kernel.nopython_signatures
            assert kernel.targetoptions["nopython"]

    def test_generated_statistics_keep_compiled_shared_kernels(self):
        parameters = Distribution.DEFAULT.parse({"alpha": 1, "beta": 1})
        for name in (*module._EDF_STATISTICS, *module._FREQUENCY_STATISTICS):
            statistic = getattr(module, name)(parameters)
            statistic.execute_statistic([0.1, 0.5, 1.5, 4.0])
            kernel = getattr(statistic, "do_execute_log_statistic", statistic.do_execute_statistic)
            assert isinstance(kernel, CPUDispatcher)
            assert kernel.nopython_signatures

    def test_moment_cache_is_bounded_readonly_and_independent_of_samples(self):
        cls = module.MirvalievLogLogisticGofStatistic
        cls._cell_moments.cache_clear()
        first = cls(Distribution.DEFAULT.parse({}), n_intervals=5)
        second = cls(Distribution.DEFAULT.parse({}), n_intervals=5)
        sample = np.exp(np.linspace(-2, 2, 100))
        value = first.execute_statistic(sample)
        assert cls._cell_moments.cache_info().misses == 1
        assert second.execute_statistic(7 * sample**2) == pytest.approx(value, rel=1e-10)
        assert cls._cell_moments.cache_info().hits == 1
        assert cls._cell_moments.cache_info().maxsize == 32
        for matrix in cls._cell_moments(5):
            assert not matrix.flags.writeable
        assert first.hypothesis().parameters() == {}

    def test_helpers_are_class_methods(self):
        tree = ast.parse(Path(module.__file__).read_text())
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in tree.body
        )
