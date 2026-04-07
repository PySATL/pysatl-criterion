from unittest.mock import MagicMock

import numpy as np
import pytest

from pysatl_criterion.critical_value.cv_calculator.cv_calculator import CVCalculator
from pysatl_criterion.statistics.models import HypothesisType


def test_calc_critical_value_right_tailed():
    mock_storage = MagicMock()
    mock_distribution = MagicMock()

    mock_distribution.results_statistics = np.array(range(100))

    mock_storage.get_data_for_cv.return_value = mock_distribution
    calculator = CVCalculator(limit_distribution_storage=mock_storage)

    critical_value = calculator.calculate_critical_value(
        criterion_code="any_code", sample_size=100, sl=0.05, alternative=HypothesisType.RIGHT
    )
    assert critical_value == pytest.approx(94.05)


def test_calc_critical_value_left_tailed():
    mock_storage = MagicMock()
    mock_distribution = MagicMock()

    mock_distribution.results_statistics = np.array(range(100))

    mock_storage.get_data_for_cv.return_value = mock_distribution
    calculator = CVCalculator(limit_distribution_storage=mock_storage)

    critical_value = calculator.calculate_critical_value(
        criterion_code="any_code", sample_size=100, sl=0.05, alternative=HypothesisType.LEFT
    )
    assert critical_value == pytest.approx(4.95)


def test_calc_critical_value_two_tailed():
    mock_storage = MagicMock()
    mock_distribution = MagicMock()

    mock_distribution.results_statistics = np.array(range(100))

    mock_storage.get_data_for_cv.return_value = mock_distribution
    calculator = CVCalculator(limit_distribution_storage=mock_storage)
    critical_value = calculator.calculate_critical_value(
        criterion_code="any_code", sample_size=100, sl=0.05, alternative=HypothesisType.TWO_TAILED
    )
    values = (2.475, 96.525)
    assert critical_value == pytest.approx(values)


def test_calc_critical_value_raises_limit_distribution_error():
    mock_storage = MagicMock()
    mock_storage.get_data_for_cv.return_value = None

    calculator = CVCalculator(limit_distribution_storage=mock_storage)

    with pytest.raises(
        ValueError, match="Limit distribution for given criterion and sample size does not exist."
    ):
        calculator.calculate_critical_value(criterion_code="any_code", sample_size=100, sl=0.05)
    mock_storage.get_data_for_cv.assert_called_once()


def test_calc_critical_value_raises_unknown_alternative():
    mock_storage = MagicMock()
    mock_distribution = MagicMock()

    mock_distribution.results_statistics = np.array(range(100))
    mock_storage.get_data_for_cv.return_value = mock_distribution
    calculator = CVCalculator(limit_distribution_storage=mock_storage)
    with pytest.raises(ValueError, match="Unknown alternative"):
        calculator.calculate_critical_value(
            criterion_code="any_code", sample_size=100, sl=0.05, alternative="not_valid"
        )
    mock_storage.get_data_for_cv.assert_called_once()
