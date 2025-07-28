from unittest.mock import MagicMock

import numpy as np
import pytest

from pysatl_criterion.p_value_calculator.p_value_calculator.p_value_calculator import (
    PValueCalculator,
)
from pysatl_criterion.statistics.models import HypothesisType


def test_calc_p_value_right_tailed():
    mock_storage = MagicMock()
    mock_distribution = MagicMock()

    mock_distribution.results_statistics = np.array(range(100))

    mock_storage.get_data_for_cv.return_value = mock_distribution
    calculator = PValueCalculator(limit_distribution_storage=mock_storage)

    p_value = calculator.calculate_p_value(
        criterion_code="any_code",
        sample_size=100,
        statistics_value=89.5,
        alternative=HypothesisType.RIGHT,
    )

    assert p_value == pytest.approx(0.1)


def test_calc_p_value_left_tailed():
    mock_storage = MagicMock()
    mock_distribution = MagicMock()

    mock_distribution.results_statistics = np.array(range(100))

    mock_storage.get_data_for_cv.return_value = mock_distribution
    calculator = PValueCalculator(limit_distribution_storage=mock_storage)

    p_value = calculator.calculate_p_value(
        criterion_code="any_code",
        sample_size=100,
        statistics_value=89.5,
        alternative=HypothesisType.LEFT,
    )
    assert p_value == pytest.approx(0.9)


def test_calc_p_value_two_tailed():
    mock_storage = MagicMock()
    mock_distribution = MagicMock()

    mock_distribution.results_statistics = np.array(range(100))

    mock_storage.get_data_for_cv.return_value = mock_distribution
    calculator = PValueCalculator(limit_distribution_storage=mock_storage)

    p_value = calculator.calculate_p_value(
        criterion_code="any_code",
        sample_size=100,
        statistics_value=89.5,
        alternative=HypothesisType.TWO_TAILED,
    )

    assert p_value == pytest.approx(0.2)
    mock_storage.get_data_for_cv.assert_called_once()


def test_calculate_p_value_statistic_outside_simulation_range():
    mock_storage = MagicMock()
    mock_distribution = MagicMock()

    mock_distribution.results_statistics = np.array([10, 20, 30, 40, 50])
    mock_storage.get_data_for_cv.return_value = mock_distribution
    calculator = PValueCalculator(limit_distribution_storage=mock_storage)

    p_value_high = calculator.calculate_p_value("any", 10, statistics_value=100.0)
    assert p_value_high == pytest.approx(0.0)

    p_value_low = calculator.calculate_p_value("any", 10, statistics_value=5.0)
    assert p_value_low == pytest.approx(1.0)


def test_calculate_p_value_raises_limit_distribution_error():
    mock_storage = MagicMock()
    mock_storage.get_data_for_cv.return_value = None

    calculator = PValueCalculator(limit_distribution_storage=mock_storage)

    with pytest.raises(
        ValueError, match="Limit distribution for given criterion and sample size does not exist."
    ):
        calculator.calculate_p_value(
            criterion_code="any_code",
            sample_size=100,
            statistics_value=89.5,
        )
    mock_storage.get_data_for_cv.assert_called_once()


def test_calculate_p_value_raises_unknown_alternative():
    mock_storage = MagicMock()
    mock_distribution = MagicMock()

    mock_distribution.results_statistics = np.array(range(100))
    mock_storage.get_data_for_cv.return_value = mock_distribution
    calculator = PValueCalculator(limit_distribution_storage=mock_storage)
    with pytest.raises(ValueError, match="Unknown alternative"):
        calculator.calculate_p_value(
            criterion_code="any_code",
            sample_size=100,
            statistics_value=89.5,
            alternative="not_valid",
        )
    mock_storage.get_data_for_cv.assert_called_once()
