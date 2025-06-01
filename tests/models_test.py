import pytest

from pysatl_criterion.statistics.common import AbstractStatistic


def test_abstract_test_criterion_code():
    with pytest.raises(NotImplementedError) as exc_info:
        AbstractStatistic.code()
    assert str(exc_info.value) == "Method is not implemented"
