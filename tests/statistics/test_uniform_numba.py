"""Verify that the public execution path compiles each explicit numerical kernel."""

import numpy as np
import pytest
from numba.core.registry import CPUDispatcher
from scipy.special import eval_legendre

from pysatl_criterion.distribution import UniformDistributionDescriptor as Uniform
from pysatl_criterion.statistics.goodness_of_fit.continuous import uniform


@pytest.mark.parametrize(
    "name",
    [
        "GreenwoodTestUniformGofStatistic",
        "BickelRosenblattUniformGofStatistic",
        "SteinUniformGofStatistic",
        "CensoredSteinUniformGofStatistic",
        "NeymanSmoothTestUniformGofStatistic",
        "ShermanUniformGofStatistic",
        "QuesenberryMillerUniformGofStatistic",
    ],
)
def test_execution_uses_nopython_kernel(name):
    cls = getattr(uniform, name)
    statistic = cls(Uniform.DEFAULT.parse({"a": 0, "b": 1}))
    sample = np.array([0.0, 0.2, 0.2, 0.5, 0.8, 1.0])
    kwargs = {"censoring_indices": [0, 1, 0, 0, 1, 0]} if name.startswith("Censored") else {}
    assert np.isfinite(statistic.execute_statistic(sample, **kwargs))
    kernel = cls.do_execute_statistic
    assert isinstance(kernel, CPUDispatcher)
    assert kernel.nopython_signatures
    assert isinstance(cls.__dict__["do_execute_statistic"], staticmethod)


@pytest.mark.parametrize("k", [1, 4, 20, 100])
def test_neyman_recurrence_matches_scipy_at_high_orders(k):
    sample = np.r_[0.0, 1e-12, np.random.default_rng(42).uniform(size=50), 1 - 1e-12, 1.0]
    expected = sum(
        (2 * j + 1) * np.sum(eval_legendre(j, 2 * sample - 1)) ** 2 / len(sample)
        for j in range(1, k + 1)
    )
    statistic = uniform.NeymanSmoothTestUniformGofStatistic(
        Uniform.DEFAULT.parse({"a": 0, "b": 1}), k=k
    )
    assert statistic.execute_statistic(sample) == pytest.approx(expected, rel=1e-11, abs=1e-12)
