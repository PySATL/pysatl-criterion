from unittest.mock import MagicMock

from pysatl_criterion.critical_value.critical_area.critical_areas import RightCriticalArea
from pysatl_criterion.critical_value.resolver.composite_resolver import (
    CompositeCriticalValueResolver,
)
from pysatl_criterion.statistics.models import HypothesisType


def test_composite_resolver_cache_miss_logic():
    """
    Scenario:
    1. Local resolver returns None (cache miss).
    2. Loader successfully fetches data (returns True).
    3. Local resolver is called again and returns the area.
    """
    # Setup mocks
    mock_local_resolver = MagicMock()
    mock_loader = MagicMock()

    expected_area = RightCriticalArea(1.96)

    mock_local_resolver.resolve.side_effect = [None, expected_area]

    mock_loader.load.return_value = True

    resolver = CompositeCriticalValueResolver(
        local_resolver=mock_local_resolver, cv_loader=mock_loader
    )

    # Act
    result = resolver.resolve(
        criterion_code="test_crit", sample_size=100, sl=0.05, alternative=HypothesisType.RIGHT
    )

    # Assert
    assert mock_local_resolver.resolve.call_count == 2
    mock_loader.load.assert_called_once_with("test_crit", 100)
    assert result == expected_area


def test_composite_resolver_loader_fails():
    """
    If loader return False, there should be no second attempt to search the local database
    """
    # Setup mocks
    mock_local_resolver = MagicMock()
    mock_loader = MagicMock()

    mock_local_resolver.resolve.return_value = None
    mock_loader.load.return_value = False  # Данных нет на сервере

    resolver = CompositeCriticalValueResolver(mock_local_resolver, mock_loader)

    # Act
    result = resolver.resolve("test_crit", 100, 0.05)

    # Assert
    assert mock_local_resolver.resolve.call_count == 1
    mock_loader.load.assert_called_once()
    assert result is None
