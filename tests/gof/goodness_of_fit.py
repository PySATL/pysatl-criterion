from pysatl_criterion.gof.goodness_of_fit import AbstractGoodnessOfFitStatistic


def test_gof_criterion_code():
    assert "GOODNESS_OF_FIT" == AbstractGoodnessOfFitStatistic.code()
