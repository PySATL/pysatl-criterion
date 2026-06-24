import pytest

from pysatl_criterion.hypothesis_testing.critical_values.calculator.critical_value_calculator import (
    LeftCriticalValueCalculator,
    RightCriticalValueCalculator,
    TwoSidedCriticalValueCalculator,
)
from pysatl_criterion.hypothesis_testing.critical_values.calculator.model import (
    CriticalValueCalculator,
)


@pytest.mark.parametrize(
    ("calculator", "significance_level", "expected_critical_value"),
    [
        (LeftCriticalValueCalculator(), 0.25, 1.75),
        (RightCriticalValueCalculator(), 0.25, 3.25),
        (TwoSidedCriticalValueCalculator(), 0.5, (1.75, 3.25)),
    ],
)
def test_critical_value_calculators_return_expected_values(
    calculator,
    significance_level,
    expected_critical_value,
):
    limit_distribution = [1.0, 2.0, 3.0, 4.0]

    critical_value = calculator.calculate(limit_distribution, significance_level)

    assert critical_value == pytest.approx(expected_critical_value)


def test_critical_value_calculator_is_abstract():
    with pytest.raises(TypeError):
        CriticalValueCalculator()
