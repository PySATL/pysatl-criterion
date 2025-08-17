import pytest

from pysatl_criterion.multiple_testing.fdr import BenjaminiHochberg


# TESTS FOR ADJUST METHOD
def test_bh_adjust_on_empty_list():
    p_values = []
    assert BenjaminiHochberg.adjust(p_values) == []


def test_bh_adjust_with_single_p_value():
    p_values = [0.05]
    assert BenjaminiHochberg.adjust(p_values) == [0.05]


def test_bh_adjust_all_zeros():
    p_values = [0.0, 0.0, 0.0]
    assert BenjaminiHochberg.adjust(p_values) == [0.0, 0.0, 0.0]


def test_bh_adjust_with_known_values():
    """Тест на наборе данных с известным результатом."""
    p_values = [0.005, 0.05, 0.03, 0.01, 0.1]

    # Правильный ожидаемый результат, который соответствует ручному расчету
    expected_adjusted = [0.025, 0.0625, 0.05, 0.025, 0.1]

    actual_adjusted = BenjaminiHochberg.adjust(p_values)

    # Сравнение вашего результата с правильным ожидаемым
    assert actual_adjusted == pytest.approx(expected_adjusted, abs=1e-4)


def test_bh_adjust_with_complex_values():
    """Тест на сложном наборе данных, проверяющий монотонность."""
    p_values = [0.001, 0.002, 0.05, 0.005, 0.06, 0.004]
    expected_adjusted = [0.006, 0.006, 0.06, 0.0075, 0.06, 0.0075]
    actual_adjusted = BenjaminiHochberg.adjust(p_values)
    assert actual_adjusted == pytest.approx(expected_adjusted, abs=1e-4)


def test_bh_adjust_with_values_greater_than_one():
    """Проверка, что метод вызывает ошибку на p-значениях > 1."""
    p_values = [0.1, 1.2, 0.9]
    with pytest.raises(ValueError):
        BenjaminiHochberg.adjust(p_values)


def test_bh_adjust_with_non_numeric_values():
    """Проверка, что метод вызывает ошибку на нечисловых значениях."""
    p_values = [0.1, "hello", 0.9]
    with pytest.raises(TypeError):
        BenjaminiHochberg.adjust(p_values)


# TESTS FOR TEST METHOD
def test_bh_test_with_no_rejections():
    """Проверка, что гипотезы не отклоняются при очень низком q."""
    p_values = [0.01, 0.02, 0.03, 0.04, 0.05]
    q = 0.001
    rejected, _ = BenjaminiHochberg.test(p_values, q)
    assert rejected == [False, False, False, False, False]


def test_bh_test_with_all_rejections():
    """Проверка, что все гипотезы отклоняются при очень высоком q."""
    p_values = [0.01, 0.02, 0.03]
    q = 0.5
    rejected, _ = BenjaminiHochberg.test(p_values, q)
    assert rejected == [True, True, True]


def test_bh_test_with_partial_rejections():
    """Проверка на смешанном наборе p-значений."""
    p_values = [0.005, 0.05, 0.03, 0.01, 0.1]
    q = 0.05
    # Ожидаемый результат: сравнение скорректированных значений с q
    # Скорректированные значения: [0.025, 0.0625, 0.05, 0.025, 0.1]
    # Те, что <= 0.05, это 0.025, 0.05, 0.025
    expected_rejected = [True, False, True, True, False]

    rejected, _ = BenjaminiHochberg.test(p_values, q)
    assert rejected == expected_rejected
