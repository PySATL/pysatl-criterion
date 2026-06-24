import pytest

from pysatl_criterion.hypothesis_testing.p_value.calculator.model import PValueCalculator
from pysatl_criterion.hypothesis_testing.p_value.calculator.p_value_calculator import (
    LeftPValueCalculator,
    RightPValueCalculator,
    TwoSidedPValueCalculator,
)


@pytest.mark.parametrize(
    ("calculator", "statistic_value", "expected_p_value"),
    [
        (LeftPValueCalculator(), 2.0, 0.5),
        (RightPValueCalculator(), 2.0, 0.5),
        (TwoSidedPValueCalculator(), 1.0, 0.5),
    ],
)
def test_p_value_calculators_return_expected_values(
    calculator,
    statistic_value,
    expected_p_value,
):
    limit_distribution = [1.0, 2.0, 3.0, 4.0]

    p_value = calculator.calculate(limit_distribution, statistic_value)

    assert p_value == pytest.approx(expected_p_value)


def test_p_value_calculator_is_abstract():
    with pytest.raises(TypeError):
        PValueCalculator()
