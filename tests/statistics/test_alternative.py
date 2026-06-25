import pytest

from pysatl_criterion.statistics.alternative import (
    Alternative,
    AlternativeType,
    LeftAlternative,
    RightAlternative,
    TwoSidedAlternative,
)


@pytest.mark.parametrize(
    ("alternative_type", "expected_class"),
    [
        (AlternativeType.LEFT, LeftAlternative),
        (AlternativeType.RIGHT, RightAlternative),
        (AlternativeType.TWO_TAILED, TwoSidedAlternative),
    ],
)
def test_get_alternative_returns_matching_alternative(alternative_type, expected_class):
    alternative = Alternative.get_alternative(alternative_type)

    assert isinstance(alternative, expected_class)
    assert alternative.type() == alternative_type


def test_get_alternative_raises_for_unknown_alternative_type():
    with pytest.raises(ValueError) as exc_info:
        Alternative.get_alternative(object())

    assert str(exc_info.value) == "alternative must be 'LEFT',  'RIGHT' or 'TWO_TAILED'"
