import pytest as pytest

from pysatl_criterion.multiple_testing import (
    BonferroniMultipleTesting,
    SidakMultipleTesting,
)


METHODS = [BonferroniMultipleTesting, SidakMultipleTesting]


@pytest.fixture(params=METHODS)
def method(request):
    return request.param


@pytest.mark.parametrize(
    ("p_values", "func"),
    [
        ([-0.1, 0.5, 0.9], "adjust"),
        ([0.1, -0.5, 0.9], "test"),
        ([0.1, 100, 0.9], "adjust"),
        ([0.1, 0.5, 1.9], "test"),
    ],
)
def test_invalid_input(method, p_values, func):
    method_func = getattr(method, func)
    with pytest.raises(ValueError) as exc_info:
        method_func(p_values)
    assert "All p-values must be in range [0,1]." in str(exc_info.value)


def test_empty_input(method):
    assert method.adjust([]) == []
    rejected, adjusted = method.test([])
    assert rejected == []
    assert adjusted == []


def test_no_rejected_when_threshold_is_zero(method):
    p_values = [0.01, 0.02, 0.3, 0.4, 0.5]

    rejected, _ = method.test(p_values, 0.0)
    assert all(not r for r in rejected)


@pytest.mark.parametrize(
    ("p_values", "expected_adjusted"),
    [
        ([0.04], [0.04]),
        ([0.01, 0.02, 0.03], [0.03, 0.06, 0.09]),
        ([0.025, 0.05, 0.1, 0.3, 0.0004], [0.125, 0.25, 0.5, 1.0, 0.002]),
    ],
)
def test_bonferroni_adjust(p_values, expected_adjusted):
    adjusted = BonferroniMultipleTesting.adjust(p_values)
    assert adjusted == pytest.approx(expected_adjusted, 0.00001)


@pytest.mark.parametrize(
    ("p_values", "expected_rejected"),
    [
        ([0.04], [True]),
        ([0.05], [False]),
        ([0.01, 0.02, 0.03], [True, False, False]),
        ([0.025, 0.05, 0.1, 0.3, 0.0004], [False, False, False, False, True]),
    ],
)
def test_bonferroni_rejection_decisions(p_values, expected_rejected):
    rejected, _ = BonferroniMultipleTesting.test(p_values)
    assert rejected == expected_rejected


@pytest.mark.parametrize(
    ("p_values", "expected_adjusted"),
    [
        ([0.02], [0.02]),
        ([0.01, 0.02, 0.03], [0.029701, 0.058808, 0.087327]),
        ([0.3, 0.7, 0.9, 0.2, 0.3], [0.831930, 0.99757, 1, 0.672319, 0.831930]),
    ],
)
def test_sidak_adjust(p_values, expected_adjusted):
    adjusted = SidakMultipleTesting.adjust(p_values)
    assert adjusted == pytest.approx(expected_adjusted, 0.00001)


@pytest.mark.parametrize(
    ("p_values", "expected_rejected"),
    [
        ([0.04], [True]),
        ([0.05], [False]),
        ([0.01, 0.02, 0.03], [True, False, False]),
        (
            [0.001, 0.02, 0.003, 0.04, 0.05, 0.006, 0.07],
            [True, False, True, False, False, True, False],
        ),
    ],
)
def test_sidak_rejection_decisions(p_values, expected_rejected):
    rejected, _ = SidakMultipleTesting.test(p_values)
    assert rejected == expected_rejected
