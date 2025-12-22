"""
Tests for PValueCalculator functionality.

This module contains comprehensive tests for the PValueCalculator class,
covering different hypothesis types, edge cases, and error conditions.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pysatl_criterion.p_value.resolver.calculation_resolver import CalculationPValueResolver
from pysatl_criterion.statistics.models import HypothesisType


@patch("pysatl_criterion.p_value.resolver.calculation_resolver.ILimitDistributionStorage")
def test_calculate_p_value_for_different_alternatives(mock_storage_cls):
    """
    Test p-value calculation for different hypothesis types using mocked storage.
    """
    # Setup mock distribution
    mock_distribution = MagicMock()
    mock_distribution.results_statistics = np.array(range(100))

    # Setup mock storage
    mock_storage = mock_storage_cls.return_value
    mock_storage.get_data_for_cv.return_value = mock_distribution

    calculator = CalculationPValueResolver(mock_storage)

    test_cases = [
        (HypothesisType.RIGHT, 0.1, "right_tailed"),
        (HypothesisType.LEFT, 0.9, "left_tailed"),
        (HypothesisType.TWO_TAILED, 0.2, "two_tailed"),
    ]

    for alternative, expected_p_value, test_name in test_cases:
        p_value = calculator.resolve(
            criterion_code="test_criterion",
            sample_size=100,
            statistics_value=89.5,
            alternative=alternative,
        )
        assert p_value == pytest.approx(
            expected_p_value, abs=1e-10
        ), f"Expected p-value {expected_p_value} for {test_name}, got {p_value}"


@patch("pysatl_criterion.p_value.resolver.calculation_resolver.ILimitDistributionStorage")
def test_calculate_p_value_with_statistic_outside_simulation_range(mock_storage_cls):
    """Test p-value calculation when statistic is outside the simulation range."""
    mock_distribution = MagicMock()
    mock_distribution.results_statistics = np.array([10, 20, 30, 40, 50])

    mock_storage = mock_storage_cls.return_value
    mock_storage.get_data_for_cv.return_value = mock_distribution

    calculator = CalculationPValueResolver(mock_storage)

    # Statistic above simulation range
    p_value_high = calculator.resolve(
        criterion_code="test_criterion", sample_size=10, statistics_value=100.0
    )
    assert p_value_high == pytest.approx(0.0)

    # Statistic below simulation range
    p_value_low = calculator.resolve(
        criterion_code="test_criterion", sample_size=10, statistics_value=5.0
    )
    assert p_value_low == pytest.approx(1.0)


@patch("pysatl_criterion.p_value.resolver.calculation_resolver.ILimitDistributionStorage")
def test_calculate_p_value_raises_error_when_limit_distribution_not_found(mock_storage_cls):
    """Test error when limit distribution is not found."""
    mock_storage = mock_storage_cls.return_value
    mock_storage.get_data_for_cv.return_value = None

    calculator = CalculationPValueResolver(mock_storage)

    with pytest.raises(
        ValueError,
        match="Limit distribution for criterion nonexistent_criterion and "
        "sample size 100 does not exist.",
    ):
        calculator.resolve(
            criterion_code="nonexistent_criterion",
            sample_size=100,
            statistics_value=89.5,
        )


@patch("pysatl_criterion.p_value.resolver.calculation_resolver.ILimitDistributionStorage")
def test_calculate_p_value_raises_error_for_unknown_alternative(mock_storage_cls):
    """Test error for unknown hypothesis alternative."""
    mock_distribution = MagicMock()
    mock_distribution.results_statistics = np.array(range(100))

    mock_storage = mock_storage_cls.return_value
    mock_storage.get_data_for_cv.return_value = mock_distribution

    calculator = CalculationPValueResolver(mock_storage)

    with pytest.raises(ValueError, match="Unknown alternative"):
        calculator.resolve(
            criterion_code="test_criterion",
            sample_size=100,
            statistics_value=89.5,
            alternative="invalid_hypothesis_type",
        )


@pytest.mark.parametrize(
    "statistics_array,statistics_value,expected_p_value",
    [
        ([1, 2, 3, 4, 5], 3.0, 0.4),
        ([10, 20, 30, 40, 50], 25.0, 0.6),
        ([1, 1, 1, 1, 1], 1.0, 0.0),
    ],
)
@patch("pysatl_criterion.p_value.resolver.calculation_resolver.ILimitDistributionStorage")
def test_p_value_calculation_with_different_distributions(
    mock_storage_cls, statistics_array, statistics_value, expected_p_value
):
    """Test p-value calculation with various distributions."""
    mock_distribution = MagicMock()
    mock_distribution.results_statistics = statistics_array

    mock_storage = mock_storage_cls.return_value
    mock_storage.get_data_for_cv.return_value = mock_distribution

    calculator = CalculationPValueResolver(mock_storage)

    p_value = calculator.resolve(
        criterion_code="test_criterion",
        sample_size=len(statistics_array),
        statistics_value=statistics_value,
        alternative=HypothesisType.RIGHT,
    )

    assert p_value == pytest.approx(expected_p_value), (
        f"Failed for statistics_array: {statistics_array}, "
        f"statistics_value: {statistics_value}, expected {expected_p_value}, got {p_value}"
    )
