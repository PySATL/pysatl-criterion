import pytest

from pysatl_criterion.critical_value.critical_area.critical_areas import (
    LeftCriticalArea,
    RightCriticalArea,
    TwoSidedCriticalArea,
)


def test_left_critical_area_contains_expected_values():
    critical_value = 3.0
    area = LeftCriticalArea(critical_value)

    assert area.critical_value == critical_value
    assert area.contains(3.0)
    assert area.contains(5.5)
    assert not area.contains(2.999)


def test_right_critical_area_contains_expected_values():
    critical_value = 4.0
    area = RightCriticalArea(critical_value)

    assert area.critical_value == critical_value
    assert area.contains(4.0)
    assert area.contains(-10.0)
    assert not area.contains(4.001)


@pytest.mark.parametrize(
    ("left_cv", "right_cv", "value", "is_inside"),
    [
        (1.0, 5.0, 1.0, True),
        (1.0, 5.0, 5.0, True),
        (1.0, 5.0, 3.0, True),
        (1.0, 5.0, 0.999, False),
        (1.0, 5.0, 5.001, False),
    ],
)
def test_two_sided_critical_area_contains_range(left_cv, right_cv, value, is_inside):
    area = TwoSidedCriticalArea(left_cv, right_cv)

    assert area.left_cv == left_cv
    assert area.right_cv == right_cv
    assert area.contains(value) is is_inside
