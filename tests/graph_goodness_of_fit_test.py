import pytest

from pysatl_criterion import AbstractGraphTestStatistic


def test_abstract_graph_criterion():
    with pytest.raises(NotImplementedError) as exc_info:
        AbstractGraphTestStatistic.get_graph_stat([])
    assert str(exc_info.value) == "Method is not implemented"
