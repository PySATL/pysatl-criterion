"""Full identifiers follow overridden short codes throughout the GoF hierarchy."""

import pytest

from pysatl_criterion import DistributionType
from pysatl_criterion.utils.statistic import get_available_criteria


CRITERIA = sorted(
    {
        criterion
        for distribution in DistributionType
        for criterion in get_available_criteria(distribution)
    },
    key=lambda criterion: criterion.__name__,
)


@pytest.mark.parametrize("criterion", CRITERIA, ids=lambda criterion: criterion.__name__)
def test_full_code_uses_the_concrete_short_code(criterion, monkeypatch):
    original_short_code = criterion.short_code()
    original_code = criterion.code()
    assert original_code.startswith(f"{original_short_code}_")
    suffix = original_code[len(original_short_code) :]
    monkeypatch.setattr(criterion, "short_code", staticmethod(lambda: "CUSTOM"))
    assert criterion.code() == f"CUSTOM{suffix}"
