from typing import cast
from unittest.mock import MagicMock

from pysatl_criterion.critical_value.critical_area.critical_areas import RightCriticalArea
from pysatl_criterion.critical_value.resolver.model import CriticalValueResolver
from pysatl_criterion.p_value.resolver.model import PValueResolver
from pysatl_criterion.statistics.goodness_of_fit import AbstractGoodnessOfFitStatistic
from pysatl_criterion.statistics.models import HypothesisType
from pysatl_criterion.test.goodness_of_fit_test.goodness_of_fit_test import GoodnessOfFitTest
from pysatl_criterion.test.model import TestMethod


def test_goodness_of_fit_cv_path_accepts_hypothesis():
    mock_statistic = MagicMock(spec=AbstractGoodnessOfFitStatistic)
    mock_statistic.code.return_value = "test_criterion"
    mock_statistic.execute_statistic.return_value = 10.0

    # Mock the critical value calculator
    mock_cv_calculator = cast(CriticalValueResolver, MagicMock())
    mock_cv_calculator.resolve.return_value = MagicMock(
        contains=lambda x: x < 15.0
    )  # Accepts hypothesis

    gof_test = GoodnessOfFitTest(
        statistics=mock_statistic,
        significance_level=0.05,
        test_method=TestMethod.CRITICAL_VALUE,
        alternative=HypothesisType.RIGHT,
        cv_resolver=mock_cv_calculator,
    )

    assert gof_test.test(data=[1, 2, 3]) is True


def test_goodness_of_fit_cv_path_rejects_hypothesis():
    mock_statistic = cast(AbstractGoodnessOfFitStatistic, MagicMock())
    mock_statistic.code.return_value = "test_criterion"
    mock_statistic.execute_statistic.return_value = 10.0

    # Mock the critical value calculator
    mock_cv_calculator = cast(CriticalValueResolver, MagicMock())
    mock_cv_calculator.resolve.return_value = RightCriticalArea(9)

    gof_test = GoodnessOfFitTest(
        statistics=mock_statistic,
        significance_level=0.05,
        test_method=TestMethod.CRITICAL_VALUE,
        alternative=HypothesisType.RIGHT,
        cv_resolver=mock_cv_calculator,
    )

    assert gof_test.test(data=[1, 2, 3]) is False


def test_goodness_of_fit_p_value_path_accepts_hypothesis(mocker):
    mock_statistic = MagicMock(spec=AbstractGoodnessOfFitStatistic)
    mock_statistic.code.return_value = "test_criterion"
    mock_statistic.execute_statistic.return_value = 10.0

    # Mock PValueCalculator using the imported class
    mock_p_value_calculator = mocker.patch.object(PValueResolver, "__new__")
    mock_p_value_instance = cast(PValueResolver, MagicMock())
    mock_p_value_calculator.return_value = mock_p_value_instance
    mock_p_value_instance.resolve.return_value = 0.1

    gof_test = GoodnessOfFitTest(
        statistics=mock_statistic, significance_level=0.05, test_method=TestMethod.P_VALUE
    )

    assert gof_test.test(data=[1, 2, 3]) is True


def test_goodness_of_fit_p_value_path_rejects_hypothesis():
    mock_statistic = cast(AbstractGoodnessOfFitStatistic, MagicMock())
    mock_statistic.code.return_value = "test_criterion"
    mock_statistic.execute_statistic.return_value = 10.0

    # Mock PValueCalculator using the imported class
    mock_p_value_instance = cast(PValueResolver, MagicMock())
    mock_p_value_instance.resolve.return_value = 0.01

    gof_test = GoodnessOfFitTest(
        p_value_resolver=mock_p_value_instance,
        statistics=mock_statistic,
        significance_level=0.05,
        test_method=TestMethod.P_VALUE,
    )

    assert gof_test.test(data=[1, 2, 3]) is False
