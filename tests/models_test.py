import pytest

from criterion import AbstractTestStatistic


def test_abstract_test_criterion_code():
    with pytest.raises(NotImplementedError) as exc_info:
        AbstractTestStatistic.code()
    assert str(exc_info.value) == "Method is not implemented"
