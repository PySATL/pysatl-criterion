import pytest
from pysatl_criterion.multiple_testing.fwer import Holm, SidakHolm

def test_holm_correction():
    p_values = [0.04, 0.001, 0.7, 0.02]
    alpha = 0.05

    rejected, adjusted = Holm.test(p_values, alpha)

    expected_adjusted = [0.08, 0.004, 0.7, 0.06]

    assert rejected == [False, True, False, False]
    assert adjusted == pytest.approx(expected_adjusted, abs=1e-3)


def test_holm_adjust():
    p_values = [0.01, 0.04, 0.03]

    expected = [0.03, 0.04, 0.04]

    adjusted = Holm.adjust(p_values)
    assert adjusted == pytest.approx(expected, abs=1e-3)


def test_holm_with_early_stop():
    p_values = [0.1, 0.2, 0.3]
    alpha = 0.05

    rejected, adjusted = Holm.test(p_values, alpha)

    expected_adjusted = [0.3, 0.3, 0.3]

    assert rejected == [False, False, False]
    assert adjusted == pytest.approx(expected_adjusted, abs=1e-3)


def test_holm_all_rejected():
    p_values = [0.001, 0.002, 0.003]
    alpha = 0.05

    rejected, adjusted = Holm.test(p_values, alpha)

    expected_adjusted = [0.003, 0.003, 0.003]

    assert rejected == [True, True, True]
    assert adjusted == pytest.approx(expected_adjusted, abs=1e-3)


def test_sidak_correction():
    p_values = [0.01, 0.02, 0.1]
    alpha = 0.05

    rejected, adjusted = SidakHolm.test(p_values, alpha)

    assert rejected == [True, False, False]

    expected = [
        1 - (1 - 0.01) ** 3,
        1 - (1 - 0.02) ** 3,
        1 - (1 - 0.1) ** 3
    ]
    assert adjusted == pytest.approx(expected, abs=1e-10)


def test_sidak_edge_cases():
    p_values = [0.0, 0.0, 0.0]
    rejected, adjusted = SidakHolm.test(p_values, 0.05001)
    assert adjusted == [0.0, 0.0, 0.0]
    assert all(rejected)

    p_values = [1.0, 1.0, 1.0]
    rejected, adjusted = SidakHolm.test(p_values, 0.05001)
    assert adjusted == [1.0, 1.0, 1.0]
    assert not any(rejected)

    p_values = [0.05]
    rejected, adjusted = SidakHolm.test(p_values, 0.05001)
    assert adjusted == pytest.approx([0.05], abs=1e-10)
    assert rejected == [True]

    p_values = [0.05, 0.1, 0.15]
    n = len(p_values)
    expected = [1 - (1 - p) ** n for p in p_values]
    rejected, adjusted = SidakHolm.test(p_values, 0.05001)
    assert adjusted == pytest.approx(expected, abs=1e-10)