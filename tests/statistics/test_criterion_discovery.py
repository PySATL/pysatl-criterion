"""Public discovery works through subclass inheritance without a separate registry."""

import inspect
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from pysatl_criterion import DistributionType
from pysatl_criterion.statistics.goodness_of_fit.continuous.normal import (
    KolmogorovSmirnovNormalityGofStatistic,
)
from pysatl_criterion.utils.statistic import (
    get_available_criteria,
    get_available_criteria_codes,
    get_criterion_by_code,
)


@pytest.fixture
def extension():
    class ExtensionNormalKS(KolmogorovSmirnovNormalityGofStatistic):
        @staticmethod
        def short_code():
            return "TEST_EXTENSION_KS"

    yield ExtensionNormalKS
    # Discovery traverses live subclasses; exclude this test-only class afterwards.
    ExtensionNormalKS.__abstractmethods__ = frozenset({"test_scope_closed"})


def test_extension_is_discovered_by_class_and_both_codes(extension):
    assert get_available_criteria(DistributionType.NORMAL).count(extension) == 1
    assert "TEST_EXTENSION_KS" in get_available_criteria_codes(DistributionType.NORMAL)
    assert get_criterion_by_code(extension.code()) is extension
    assert extension not in get_available_criteria(DistributionType.EXPONENTIAL)


def test_abstract_extension_is_excluded(extension, monkeypatch):
    monkeypatch.setattr(extension, "__abstractmethods__", frozenset({"execute_statistic"}))
    assert inspect.isabstract(extension)
    assert extension not in get_available_criteria(DistributionType.NORMAL)
    with pytest.raises(ValueError, match="not available"):
        get_criterion_by_code(extension.code())


def test_discovery_is_stable_under_concurrent_calls(extension):
    expected = set(get_available_criteria(DistributionType.NORMAL))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(get_available_criteria, [DistributionType.NORMAL] * 12))
    assert extension in expected
    assert all(set(result) == expected and len(result) == len(expected) for result in results)


def test_lookup_loads_builtins_in_fresh_interpreter():
    program = """
import json
from pysatl_criterion.utils.statistic import get_criterion_by_code
cls = get_criterion_by_code('KS_NORMALITY_GOODNESS_OF_FIT')
print(json.dumps([cls.__name__, cls.short_code()]))
"""
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=True
    )
    assert json.loads(result.stdout) == ["KolmogorovSmirnovNormalityGofStatistic", "KS"]
