"""Ensure every local exponential numerical kernel compiles in nopython mode."""

import numpy as np
import pytest
from numba.core.registry import CPUDispatcher

from pysatl_criterion.statistics.goodness_of_fit.continuous import exponent
from tests.parameter_cases import parameters_for


COMPILED_STATISTICS = [
    cls
    for cls in vars(exponent).values()
    if isinstance(cls, type)
    and cls.__module__ == exponent.__name__
    and any(name in cls.__dict__ for name in ("_statistic", "_kernel"))
]


@pytest.mark.parametrize("cls", COMPILED_STATISTICS, ids=lambda cls: cls.__name__)
def test_exponential_kernel_compiles_and_preserves_input(cls):
    statistic = cls(parameters_for(cls))
    sample = np.array([0.125, 0.25, 0.5, 1.0, 2.0, 4.0])
    original = sample.copy()
    sample.flags.writeable = False

    result = statistic.execute_statistic(sample)

    assert np.isfinite(result)
    np.testing.assert_array_equal(sample, original)
    assert statistic.execute_statistic(sample * 8) == pytest.approx(result, abs=1e-13)
    for name in ("_statistic", "_kernel"):
        if name in cls.__dict__:
            kernel = getattr(cls, name)
            assert isinstance(kernel, CPUDispatcher)
            assert kernel.nopython_signatures
            assert kernel.targetoptions["nopython"]


def test_epstein_returns_infinity_for_zero_order_spacing():
    cls = exponent.EpsteinExponentialityGofStatistic
    assert np.isposinf(cls(parameters_for(cls)).execute_statistic([0.125, 0.5, 0.5, 1, 2, 4]))
