from unittest.mock import MagicMock

from pysatl_criterion.critical_value.critical_area.critical_areas import RightCriticalArea
from pysatl_criterion.critical_value.resolver.composite_resolver import (
    CompositeCriticalValueResolver,
)
from pysatl_criterion.persistence.model.limit_distribution.limit_distribution import BulkLoadResult


def test_composite_resolver_cache_hit():
    """
    All data is in local cache. Loader should NOT be called.
    """
    mock_local_resolver = MagicMock()
    mock_loader = MagicMock()

    code1 = "crit1"
    area1 = RightCriticalArea(1.96)

    mock_local_resolver.resolve_bulk.return_value = {code1: area1}

    resolver = CompositeCriticalValueResolver(mock_local_resolver, mock_loader)

    results = resolver.resolve_bulk([code1], 100, 0.05)

    assert results == {code1: area1}
    mock_loader.load_bulk.assert_not_called()
    assert mock_local_resolver.resolve_bulk.call_count == 1


def test_composite_resolver_cache_miss_and_load_success():
    """
    1. Local cache miss.
    2. Loader successfully fetches data.
    3. Second local call returns the data.
    """
    mock_local_resolver = MagicMock()
    mock_loader = MagicMock()

    code1 = "crit1"
    area1 = RightCriticalArea(1.96)

    mock_local_resolver.resolve_bulk.side_effect = [{}, {code1: area1}]

    mock_loader.load_bulk.return_value = BulkLoadResult(
        requested_count=1, already_cached_count=0, newly_cached_count=1, not_found_codes=[]
    )

    resolver = CompositeCriticalValueResolver(mock_local_resolver, mock_loader)

    results = resolver.resolve_bulk([code1], 100, 0.05)

    assert results == {code1: area1}
    assert mock_local_resolver.resolve_bulk.call_count == 2
    mock_loader.load_bulk.assert_called_once_with([code1], 100)


def test_composite_resolver_loader_fails():
    """
    Loader returns failure (not_found). Result should be empty for that code.
    """
    mock_local_resolver = MagicMock()
    mock_loader = MagicMock()

    code1 = "crit1"

    mock_local_resolver.resolve_bulk.side_effect = [{}, {}]

    mock_loader.load_bulk.return_value = BulkLoadResult(
        requested_count=1, already_cached_count=0, newly_cached_count=0, not_found_codes=[code1]
    )

    resolver = CompositeCriticalValueResolver(mock_local_resolver, mock_loader)

    results = resolver.resolve_bulk([code1], 100, 0.05)

    assert code1 not in results
    assert mock_local_resolver.resolve_bulk.call_count == 1
    mock_loader.load_bulk.assert_called_once()


def test_composite_resolver_partial_cache_hit():
    """
    Some codes in cache, some missing. Only missing are loaded.
    """

    mock_local_resolver = MagicMock()
    mock_loader = MagicMock()

    code1 = "crit1"
    code2 = "crit2"
    area1 = RightCriticalArea(1.96)
    area2 = RightCriticalArea(2.58)

    mock_local_resolver.resolve_bulk.side_effect = [{code1: area1}, {code2: area2}]

    mock_loader.load_bulk.return_value = BulkLoadResult(
        requested_count=1, already_cached_count=0, newly_cached_count=1, not_found_codes=[]
    )

    resolver = CompositeCriticalValueResolver(mock_local_resolver, mock_loader)

    results = resolver.resolve_bulk([code1, code2], 100, 0.05)

    assert results == {code1: area1, code2: area2}
    assert mock_local_resolver.resolve_bulk.call_count == 2

    mock_loader.load_bulk.assert_called_once_with([code2], 100)
