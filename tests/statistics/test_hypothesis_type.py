from enum import Enum, auto

import pytest


class HypothesisType(Enum):
    RIGHT = auto()
    LEFT = auto()
    TWO_SIDED = auto()

    def check_hypothesis(
        self,
        statistic_value,
        cv: float | tuple[float, float],
    ) -> bool:
        if self == HypothesisType.RIGHT:
            return statistic_value <= cv
        if self == HypothesisType.LEFT:
            return statistic_value >= cv
        if self == HypothesisType.TWO_SIDED:
            if not isinstance(cv, tuple):
                raise TypeError("For a TWO_SIDED hypothesis, 'cv' must be a tuple of two floats.")
            left_cv, right_cv = cv
            return left_cv <= statistic_value <= right_cv


@pytest.mark.parametrize(
    "hypothesis_type, statistic_value, critical_value, expected_result",
    [
        (HypothesisType.RIGHT, 9.9, 10, True),
        (HypothesisType.RIGHT, 10.0, 10, True),
        (HypothesisType.RIGHT, 10.1, 10, False),
        (HypothesisType.LEFT, 9.9, 10, False),
        (HypothesisType.LEFT, 10.0, 10, True),
        (HypothesisType.LEFT, 10.1, 10, True),
        (HypothesisType.TWO_SIDED, 9.9, (10, 20), False),
        (HypothesisType.TWO_SIDED, 10.0, (10, 20), True),
        (HypothesisType.TWO_SIDED, 15.0, (10, 20), True),
        (HypothesisType.TWO_SIDED, 20.0, (10, 20), True),
        (HypothesisType.TWO_SIDED, 20.1, (10, 20), False),
    ],
)
def test_check_hypothesis(hypothesis_type, statistic_value, critical_value, expected_result):
    result = hypothesis_type.check_hypothesis(statistic_value, critical_value)

    assert result == expected_result
