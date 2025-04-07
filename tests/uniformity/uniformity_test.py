import pytest

from pysatl_criterion.uniformity.uniformity import KolmogorovSmirnovUniformityStatistic


@pytest.mark.parametrize(
    ("x", "y", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], [3, 5, 7, 9, 9, 0, 7], 0.42857142),
    ],
)
def test_ks_uniformity_criterion(x, y, result):
    statistic = KolmogorovSmirnovUniformityStatistic().execute_statistic(x, y)
    assert pytest.approx(statistic, 0.001) == result


def test_ks_uniformity_criterion_code():
    assert "KS_UNIFORMITY" == KolmogorovSmirnovUniformityStatistic().code()
