import pytest
from pysatl_criterion.multiple_testing.fdr import BenjaminiYekutieli

def test_benjamini_yekutieli_complex():
    p_values = [0.001, 0.002, 0.01, 0.03, 0.04, 0.1, 0.15, 0.2, 0.25, 0.5]
    alpha = 0.05

    rejected, _ = BenjaminiYekutieli.test(p_values, alpha)

    assert rejected == [True, True, False, False, False, False, False, False, False, False]


def test_benjamini_yekutieli_adjust_complex():
    p_values = [0.001, 0.005, 0.05, 0.1, 0.15]
    adjusted = BenjaminiYekutieli.adjust(p_values)

    n = 5
    c = sum(1.0 / i for i in range(1, n + 1))
    sorted_p = sorted(p_values)
    expected = [
        min(1.0, sorted_p[0] * n * c / 1),
        min(1.0, sorted_p[1] * n * c / 2),
        min(1.0, sorted_p[2] * n * c / 3),
        min(1.0, sorted_p[3] * n * c / 4),
        min(1.0, sorted_p[4] * n * c / 5)
    ]

    for i in range(n - 2, -1, -1):
        if expected[i] > expected[i + 1]:
            expected[i] = expected[i + 1]

    assert adjusted == pytest.approx(expected, abs=1e-4)


def test_benjamini_yekutieli_edge_cases():
    p_values = [0.00001, 0.99999]
    rejected, _ = BenjaminiYekutieli.test(p_values, 0.05)
    assert rejected == [True, False]


def test_benjamini_yekutieli_large_input():
    import numpy as np
    np.random.seed(42)
    p_values = np.random.rand(1000).tolist()

    rejected, adjusted = BenjaminiYekutieli.test(p_values, 0.05)

    assert len(rejected) == 1000
    assert len(adjusted) == 1000
    assert all(0 <= p <= 1 for p in adjusted)