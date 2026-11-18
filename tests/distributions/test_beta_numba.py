"""Regression checks for compiled Beta reductions and their execution paths."""

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest
from numba.core.registry import CPUDispatcher
from scipy.special import betaln
from scipy.stats import beta as scipy_beta
from scipy.stats import differential_entropy

from pysatl_criterion.distribution import BetaDistributionDescriptor as Beta
from pysatl_criterion.statistics.goodness_of_fit.continuous import beta
from pysatl_criterion.statistics.goodness_of_fit.continuous.beta import beta_statistics


class TestBetaNumba:
    @pytest.mark.parametrize(
        "name",
        [name for name in beta.__all__ if not inspect.isabstract(getattr(beta, name))],
    )
    def test_every_criterion_executes_nopython_kernel(self, name):
        cls = getattr(beta, name)
        fitted = cls in (
            beta.EbnerLiebenbergBetaGofStatistic,
            beta.RiadAbdElMaboodBetaGofStatistic,
            beta.KullbackLeiblerBetaGofStatistic,
            beta.RaschkeBetaGofStatistic,
        )
        criterion = cls(Beta.DEFAULT.parse({} if fitted else {"a": 2, "b": 5}))
        sample = np.random.default_rng(15).beta(2, 5, 31)
        criterion.execute_statistic(sample)
        if name in beta_statistics._EDF_STATISTICS:
            kernel = getattr(cls, "do_execute_log_statistic", cls.do_execute_statistic)
        elif name in beta_statistics._FREQUENCY_STATISTICS:
            kernel = cls.do_execute_statistic
        elif cls is beta.RiadAbdElMaboodBetaGofStatistic:
            kernel = cls._laplace_kernel
        elif cls is beta.RaschkeBetaGofStatistic:
            kernel = cls._ad_kernel
        else:
            kernel = cls._statistic_kernel
        assert isinstance(kernel, CPUDispatcher)
        assert kernel.nopython_signatures
        assert kernel.targetoptions["nopython"]
        assert not kernel.targetoptions.get("fastmath", False)

    @pytest.mark.parametrize("shapes", [(0.3, 0.7), (2, 5), (1000, 2000)])
    def test_fused_kl_matches_scipy_for_large_samples(self, shapes):
        x = np.sort(np.random.default_rng(71).beta(*shapes, size=1001))
        a, b, _, _ = scipy_beta.fit(x, floc=0, fscale=1)
        expected = -differential_entropy(x, window_length=17, method="vasicek")
        expected -= scipy_beta.logpdf(x, a, b).mean()
        actual = beta.KullbackLeiblerBetaGofStatistic._statistic_kernel(x, 17, a, b, betaln(a, b))
        assert actual == pytest.approx(expected, abs=1e-10)

    def test_compiled_quadrature_integrand(self):
        x = np.array([0.01, 0.05, 0.4, 0.8])
        a, b = beta.RiadAbdElMaboodBetaGofStatistic._coefficients_kernel(x, 2.0, 5.0)
        np.testing.assert_allclose(a, x * (x - 1) / 2)
        np.testing.assert_allclose(b, 1 - 3.5 * x)
        rate = 2 + 2 * x.min()
        shifted = x - x.min()
        kernel = beta.RiadAbdElMaboodBetaGofStatistic._integrand_kernel
        for u in (0, 0.1, 2, 50):
            t = u / rate
            expected = len(x) / rate * np.mean((a * t + b) * np.exp(-shifted * t)) ** 2
            expected *= np.exp(-u)
            assert kernel(u, shifted, a, b, rate) == pytest.approx(expected, abs=1e-15)
        assert kernel.nopython_signatures

    @pytest.mark.parametrize("values", [[1, 2, 3], [1e8, 1e8 + 1, 1e8 + 2]])
    def test_standardization_matches_normal_mle_without_mutation(self, values):
        y = np.asarray(values, dtype=float)
        original = y.copy()
        result = beta.RaschkeBetaGofStatistic._standardize_kernel(y)
        np.testing.assert_allclose(result, (y - y.mean()) / y.std(ddof=0))
        np.testing.assert_array_equal(y, original)
        assert beta.RaschkeBetaGofStatistic._standardize_kernel.nopython_signatures

    @pytest.mark.parametrize("values", [[1, 1, 1], [0, np.inf, 2], [0, np.nan, 2]])
    def test_standardization_rejects_degenerate_or_nonfinite_values(self, values):
        with pytest.raises(ValueError):
            beta.RaschkeBetaGofStatistic._standardize_kernel(np.asarray(values, dtype=float))

    def test_helpers_are_class_methods(self):
        tree = ast.parse(Path(beta_statistics.__file__).read_text())
        assert not any(isinstance(node, ast.FunctionDef) for node in tree.body)
