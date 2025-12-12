"""
Tests for PValueCalculator functionality.

This module contains comprehensive tests for the PValueCalculator class,
covering different hypothesis types, edge cases, and error conditions.
"""
from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pytest

from pysatl_criterion.p_value.resolver.calculation_resolver import CalculationPValueResolver
from pysatl_criterion.persistence.model.limit_distribution.limit_distribution import (
    ILimitDistributionStorage,
    LimitDistributionModel,
)
from pysatl_criterion.statistics.models import HypothesisType


@pytest.fixture
def mock_storage():
    """Create a mock storage object for testing."""
    return cast(ILimitDistributionStorage, MagicMock())

@pytest.fixture
def mock_distribution():
    """Create a mock distribution object with test statistics."""
    mock_dist = MagicMock()
    mock_dist.results_statistics = np.array(range(100))
    return mock_dist

@pytest.fixture
def calculator_with_mock_data(mock_storage, mock_distribution):
    """Create a calculator with mocked storage and distribution data."""
    mock_storage.get_data_for_cv.return_value = mock_distribution
    return CalculationPValueResolver(mock_storage)

@pytest.fixture
def calculator_with_empty_storage(mock_storage):
    """Create a calculator with empty storage (returns None)."""
    mock_storage.get_data_for_cv.return_value = None
    return CalculationPValueResolver(mock_storage)

@pytest.mark.parametrize(
    "alternative,expected_p_value,test_name",
    [
        (HypothesisType.RIGHT, 0.1, "right_tailed"),
        (HypothesisType.LEFT, 0.9, "left_tailed"),
        (HypothesisType.TWO_TAILED, 0.2, "two_tailed"),
    ],
)
def test_calculate_p_value_for_different_alternatives(
    calculator_with_mock_data, alternative, expected_p_value, test_name
):
    """
    Test p-value calculation for different hypothesis types.

    Args:
        alternative: The hypothesis type to test
        expected_p_value: Expected p-value result
        test_name: Name of the test case for identification
    """
    # Given
    criterion_code = "test_criterion"
    sample_size = 100
    statistics_value = 89.5

    # When
    p_value = calculator_with_mock_data.resolve(
        criterion_code=criterion_code,
        sample_size=sample_size,
        statistics_value=statistics_value,
        alternative=alternative,
    )

    # Then
    assert p_value == pytest.approx(
        expected_p_value, abs=1e-10
    ), f"Expected p-value {expected_p_value} for {test_name}, got {p_value}"

def test_calculate_p_value_with_statistic_outside_simulation_range(mock_storage):
    """
    Test p-value calculation when statistic is outside the simulation range.

    This tests edge cases where the observed statistic is either much higher
    or much lower than the simulated distribution range.
    """

    # Given
    mock_distribution = cast(LimitDistributionModel, MagicMock())
    mock_distribution.results_statistics = np.array([10, 20, 30, 40, 50])
    mock_storage.get_data_for_cv.return_value = mock_distribution
    calculator = CalculationPValueResolver(limit_distribution_storage=mock_storage)

    # When & Then - Statistic much higher than simulation range
    p_value_high = calculator.resolve(
            criterion_code="test_criterion", sample_size=10, statistics_value=100.0
    )
    assert p_value_high == pytest.approx(
        0.0
    ), "P-value should be 0 for statistic above simulation range"

    # When & Then - Statistic much lower than simulation range
    p_value_low = calculator.resolve(
            criterion_code="test_criterion", sample_size=10, statistics_value=5.0
    )
    assert p_value_low == pytest.approx(
        1.0
    ), "P-value should be 1 for statistic below simulation range"

def test_calculate_p_value_raises_error_when_limit_distribution_not_found(
        calculator_with_empty_storage
):
    """
    Test that appropriate error is raised when limit distribution is not found.

    This tests the error handling when the storage cannot provide
    the required limit distribution data.
    """
    # Given
    criterion_code = "nonexistent_criterion"
    sample_size = 100
    statistics_value = 89.5

    # When & Then
    with pytest.raises(
        ValueError,
        match="Limit distribution for criterion nonexistent_criterion "
              "and sample size 100 does not exist.",
    ):
        calculator_with_empty_storage.resolve(
            criterion_code=criterion_code,
            sample_size=sample_size,
            statistics_value=statistics_value,
        )

def test_calculate_p_value_raises_error_for_unknown_alternative(calculator_with_mock_data):
    """
    Test that appropriate error is raised for unknown hypothesis alternatives.

    This tests the validation of the alternative parameter to ensure
    only valid hypothesis types are accepted.
    """
    # Given
    criterion_code = "test_criterion"
    sample_size = 100
    statistics_value = 89.5
    invalid_alternative = "invalid_hypothesis_type"

    # When & Then
    with pytest.raises(ValueError, match="Unknown alternative"):
        calculator_with_mock_data.resolve(
            criterion_code=criterion_code,
            sample_size=sample_size,
            statistics_value=statistics_value,
            alternative=invalid_alternative,
        )

@pytest.mark.parametrize(
    "statistics_array,statistics_value,expected_p_value",
    [
        ([1, 2, 3, 4, 5], 3.0, 0.4),
        ([10, 20, 30, 40, 50], 25.0, 0.6),
        ([1, 1, 1, 1, 1], 1.0, 0.0),
    ],
)
def test_p_value_calculation_with_different_distributions(
    statistics_array, statistics_value, expected_p_value
):
    """
    Test p-value calculation with various distribution patterns.

    Args:
        statistics_array: Array of simulated statistics
        statistics_value: Observed statistic value
        expected_p_value: Expected p-value result
    """
    # Given
    mock_storage = cast(ILimitDistributionStorage, MagicMock())
    mock_distribution = cast(LimitDistributionModel, MagicMock())
    mock_distribution.results_statistics = statistics_array
    mock_storage.get_data_for_cv.return_value = mock_distribution
    calculator = CalculationPValueResolver(mock_storage)

    # When
    p_value = calculator.resolve(
        criterion_code="test_criterion",
        sample_size=len(statistics_array),
        statistics_value=statistics_value,
        alternative=HypothesisType.RIGHT,
    )

    # Then
    assert p_value == pytest.approx(
        expected_p_value
    ), (f"Failed for statistics_array: {statistics_array}, "
        f"statistics_value: {statistics_value} expected {expected_p_value}, got {p_value}")
