"""Regression checks using exact arithmetic and analytic limiting values."""

from fractions import Fraction

import numpy as np
import pytest

from pysatl_criterion.distribution.distributions import ExponentialDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit import exponent as exp
from tests.parameter_cases import parameters_for


@pytest.mark.parametrize("offsets", [[0, 0, 1], [0, 1, 1], [0, 1, 2, 4]])
@pytest.mark.parametrize("scale", [2.0**-1000, 1.0, 2.0**1000])
def test_shapiro_wilk_nearly_constant_against_exact_arithmetic(offsets, scale):
    x = (1 + np.asarray(offsets) * np.spacing(1.0)) * scale
    exact = [Fraction(float(value)) for value in x]
    n = len(exact)
    mean = sum(exact) / n
    expected = n * (mean - min(exact)) ** 2 / ((n - 1) * sum((v - mean) ** 2 for v in exact))
    with np.errstate(all="raise"):
        actual = exp.ShapiroWilkExponentialityGofStatistic(
            ExponentialDistributionDescriptor.DEFAULT.parse({})
        ).execute_statistic(x)
    assert actual == pytest.approx(float(expected), rel=1e-14)


@pytest.mark.parametrize("p", [1e-320, -1e-320, 1e-310, -1e-310])
def test_atkinson_subnormal_power_matches_geometric_mean_limit(p):
    # The product of these observations is one, so their geometric mean is one.
    x = [0.25, 0.5, 2.0, 4.0]
    expected = 2 * abs(1 / (6.75 / 4) - np.exp(-np.euler_gamma))
    with np.errstate(all="raise"):
        actual = exp.AtkinsonExponentialityGofStatistic(
            ExponentialDistributionDescriptor.DEFAULT.parse({}), p=p
        ).execute_statistic(x)
    assert actual == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize(
    "cls,parameter",
    [
        (exp.KolmogorovSmirnovExponentialityGofStatistic, "lam"),
        (exp.CramerVonMisesExponentialityGofStatistic, "lam"),
        (exp.AtkinsonExponentialityGofStatistic, "p"),
        (exp.DeshpandeExponentialityGofStatistic, "b"),
        (exp.LorenzExponentialityGofStatistic, "p"),
    ],
)
def test_unrepresentable_integer_setting_raises_value_error(cls, parameter):
    if parameter == "lam":
        with pytest.raises(ValueError, match="Invalid value for lam"):
            cls(parameters_for(cls, lam=10**400))
    else:
        with pytest.raises(ValueError, match="finite real scalar"):
            cls(parameters_for(cls), **{parameter: 10**400})
