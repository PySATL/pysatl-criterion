from unittest.mock import MagicMock

from pysatl_criterion.statistics.goodness_of_fit import AbstractGoodnessOfFitStatistic
from pysatl_criterion.statistics.models import HypothesisType
from pysatl_criterion.test.goodness_of_fit_test.goodness_of_fit_test import GoodnessOfFitTest


MODULE_PATH = "pysatl_criterion.test.goodness_of_fit_test.goodness_of_fit_test"


def test_goodness_of_fit_cv_path_accepts_hypothesis(mocker):
    mock_statistic = MagicMock(spec=AbstractGoodnessOfFitStatistic)
    mock_statistic.code.return_value = "test_criterion"
    mock_statistic.execute_statistic.return_value = 10.0

    MockCVCalculator = mocker.patch(f"{MODULE_PATH}.CVCalculator")
    MockPValueCalculator = mocker.patch(f"{MODULE_PATH}.PValueCalculator")
    mocker.patch(f"{MODULE_PATH}.SQLiteLimitDistributionStorage")

    mock_cv_instance = MockCVCalculator.return_value
    mock_cv_instance.calculate_critical_value.return_value = 15.0

    gof_test = GoodnessOfFitTest(
        statistics=mock_statistic,
        significance_level=0.05,
        test_method="critical_value",
        alternative=HypothesisType.RIGHT,
    )

    assert gof_test.test(data=[1, 2, 3]) is True
    mock_cv_instance.calculate_critical_value.assert_called_once()
    MockPValueCalculator.assert_not_called()


def test_goodness_of_fit_cv_path_rejects_hypothesis(mocker):
    mock_statistic = MagicMock(spec=AbstractGoodnessOfFitStatistic)
    mock_statistic.code.return_value = "test_criterion"
    mock_statistic.execute_statistic.return_value = 10.0

    MockCVCalculator = mocker.patch(f"{MODULE_PATH}.CVCalculator")
    MockPValueCalculator = mocker.patch(f"{MODULE_PATH}.PValueCalculator")
    mocker.patch(f"{MODULE_PATH}.SQLiteLimitDistributionStorage")

    mock_cv_instance = MockCVCalculator.return_value
    mock_cv_instance.calculate_critical_value.return_value = 9.0

    gof_test = GoodnessOfFitTest(
        statistics=mock_statistic,
        significance_level=0.05,
        test_method="critical_value",
        alternative=HypothesisType.RIGHT,
    )

    assert gof_test.test(data=[1, 2, 3]) is False
    mock_cv_instance.calculate_critical_value.assert_called_once()
    MockPValueCalculator.assert_not_called()


def test_goodness_of_fit_p_value_path_accepts_hypothesis(mocker):
    mock_statistic = MagicMock(spec=AbstractGoodnessOfFitStatistic)
    mock_statistic.code.return_value = "test_criterion"
    mock_statistic.execute_statistic.return_value = 10.0

    MockCVCalculator = mocker.patch(f"{MODULE_PATH}.CVCalculator")
    MockPValueCalculator = mocker.patch(f"{MODULE_PATH}.PValueCalculator")
    mocker.patch(f"{MODULE_PATH}.SQLiteLimitDistributionStorage")

    mock_p_value_instance = MockPValueCalculator.return_value
    mock_p_value_instance.calculate_p_value.return_value = 0.1

    gof_test = GoodnessOfFitTest(
        statistics=mock_statistic, significance_level=0.05, test_method="p_value"
    )

    assert gof_test.test(data=[1, 2, 3]) is True
    mock_p_value_instance.calculate_p_value.assert_called_once()
    MockCVCalculator.assert_not_called()


def test_goodness_of_fit_p_value_path_rejects_hypothesis(mocker):
    mock_statistic = MagicMock(spec=AbstractGoodnessOfFitStatistic)
    mock_statistic.code.return_value = "test_criterion"
    mock_statistic.execute_statistic.return_value = 10.0

    MockCVCalculator = mocker.patch(f"{MODULE_PATH}.CVCalculator")
    MockPValueCalculator = mocker.patch(f"{MODULE_PATH}.PValueCalculator")
    mocker.patch(f"{MODULE_PATH}.SQLiteLimitDistributionStorage")

    mock_p_value_instance = MockPValueCalculator.return_value
    mock_p_value_instance.calculate_p_value.return_value = 0.01

    gof_test = GoodnessOfFitTest(
        statistics=mock_statistic, significance_level=0.05, test_method="p_value"
    )

    assert gof_test.test(data=[1, 2, 3]) is False
    mock_p_value_instance.calculate_p_value.assert_called_once()
    MockCVCalculator.assert_not_called()


# TODO: cannot check raise, because creating "sqlite:/" directory, CI failing
"""
def test_goodness_of_fit_raises_for_invalid_method():
    mock_statistic = MagicMock(spec=AbstractGoodnessOfFitStatistic)
    mock_statistic.code.return_value = "test_criterion"
    mock_statistic.execute_statistic.return_value = 10.0

    gof_test = GoodnessOfFitTest(
        statistics=mock_statistic, significance_level=0.05, test_method="this_is_wrong"
    )
    with pytest.raises(ValueError, match="Invalid test method."):
        gof_test.test(data=[1, 2, 3])

"""
