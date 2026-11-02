"""Classify tests so unit and integration checks can run independently."""

from pathlib import Path

import pytest


def pytest_collection_modifyitems(items):
    root = Path(__file__).parent
    for item in items:
        if not item.path.is_relative_to(root):
            continue
        area = item.path.relative_to(root).parts[0]
        if area in {"persistence", "loader"}:
            item.add_marker(pytest.mark.integration)
