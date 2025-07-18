from unittest.mock import MagicMock

import numpy as np
import pytest

from pysatl_criterion.p_value_calculator.p_value_calculator.p_value_calculator import (
    PValueCalculator,
)


def test_calc_p_value_one_tailed():
    mock_storage = MagicMock()
    mock_distribution = MagicMock()

    mock_distribution.results_statistics = np.array(range(100))

    mock_storage.get_data_for_cv.return_value = mock_distribution
    calculator = PValueCalculator(limit_distribution_storage=mock_storage)

    p_value = calculator.calculate_p_value(
        criterion_code="any_code", sample_size=100, statistics_value=89.5, two_tailed=False
    )

    assert p_value == pytest.approx(0.1)
    mock_storage.get_data_for_cv.assert_called_once()


def test_calc_p_value_two_tailed():
    mock_storage = MagicMock()
    mock_distribution = MagicMock()

    mock_distribution.results_statistics = np.array(range(100))

    mock_storage.get_data_for_cv.return_value = mock_distribution
    calculator = PValueCalculator(limit_distribution_storage=mock_storage)

    p_value = calculator.calculate_p_value(
        criterion_code="any_code", sample_size=100, statistics_value=89.5, two_tailed=True
    )

    assert p_value == pytest.approx(0.2)
    mock_storage.get_data_for_cv.assert_called_once()


def test_calculate_p_value_statistic_outside_simulation_range():
    mock_storage = MagicMock()
    mock_distribution = MagicMock()

    mock_distribution.results_statistics = np.array([10, 20, 30, 40, 50])
    mock_storage.get_data_for_cv.return_value = mock_distribution
    calculator = PValueCalculator(limit_distribution_storage=mock_storage)

    p_value_high = calculator.calculate_p_value("any", 10, statistics_value=100.0, two_tailed=False)
    assert p_value_high == pytest.approx(0.0)

    p_value_low = calculator.calculate_p_value("any", 10, statistics_value=5.0, two_tailed=False)
    assert p_value_low == pytest.approx(1.0)
