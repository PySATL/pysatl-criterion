import pytest

from pysatl_criterion import IStatistic


def test_abstract_test_criterion_code():
    with pytest.raises(NotImplementedError) as exc_info:
        IStatistic.code()
    assert str(exc_info.value) == "Method is not implemented"
