from unittest.mock import MagicMock, patch

from pysatl_criterion.critical_value.critical_area.critical_areas import RightCriticalArea
from pysatl_criterion.critical_value.resolver.composite_resolver import (
    CompositeCriticalValueResolver,
)
from pysatl_criterion.statistics.models import HypothesisType
from pysatl_criterion.test.goodness_of_fit_test.goodness_of_fit_test import GoodnessOfFitTest
from pysatl_criterion.test.model import TestMethod


@patch("pysatl_criterion.critical_value.resolver.model.CriticalValueResolver")
@patch("pysatl_criterion.statistics.goodness_of_fit.AbstractGoodnessOfFitStatistic")
def test_goodness_of_fit_cv_path_accepts_hypothesis(mock_stat_cls, mock_cv_cls):
    # Setup mocks
    mock_stat = mock_stat_cls.return_value
    mock_stat.code.return_value = "test_criterion"
    mock_stat.execute_statistic.return_value = 10.0

    mock_cv = mock_cv_cls.return_value
    mock_area = RightCriticalArea(15.0)
    mock_cv.resolve_bulk.return_value = {"test_criterion": mock_area}

    gof_test = GoodnessOfFitTest(
        statistics=[mock_stat],
        significance_level=0.05,
        test_method=TestMethod.CRITICAL_VALUE,
        alternative=HypothesisType.RIGHT,
        cv_resolver=mock_cv,
    )

    # Act
    result_dict = gof_test.test(data=[1, 2, 3])

    # Assert
    assert isinstance(result_dict, dict)
    assert result_dict["test_criterion"] is True


@patch("pysatl_criterion.critical_value.resolver.model.CriticalValueResolver")
@patch("pysatl_criterion.statistics.goodness_of_fit.AbstractGoodnessOfFitStatistic")
def test_goodness_of_fit_cv_path_rejects_hypothesis(mock_stat_cls, mock_cv_cls):
    mock_stat = mock_stat_cls.return_value
    mock_stat.code.return_value = "test_criterion"
    mock_stat.execute_statistic.return_value = 10.0

    mock_cv = mock_cv_cls.return_value
    mock_area = RightCriticalArea(9.0)
    mock_cv.resolve_bulk.return_value = {"test_criterion": mock_area}

    gof_test = GoodnessOfFitTest(
        statistics=[mock_stat],
        significance_level=0.05,
        test_method=TestMethod.CRITICAL_VALUE,
        alternative=HypothesisType.RIGHT,
        cv_resolver=mock_cv,
    )

    # Act
    result_dict = gof_test.test(data=[1, 2, 3])

    # Assert
    assert isinstance(result_dict, dict)
    assert result_dict["test_criterion"] is False


@patch("pysatl_criterion.p_value.resolver.model.PValueResolver")
@patch("pysatl_criterion.statistics.goodness_of_fit.AbstractGoodnessOfFitStatistic")
def test_goodness_of_fit_p_value_path_accepts_hypothesis(mock_stat_cls, mock_p_value_cls):
    mock_stat = mock_stat_cls.return_value
    mock_stat.code.return_value = "test_criterion"
    mock_stat.execute_statistic.return_value = 10.0

    mock_p_value = mock_p_value_cls.return_value
    mock_p_value.resolve.return_value = 0.1

    gof_test = GoodnessOfFitTest(
        statistics=[mock_stat],
        significance_level=0.05,
        test_method=TestMethod.P_VALUE,
        p_value_resolver=mock_p_value,
    )

    # Act
    result_dict = gof_test.test(data=[1, 2, 3])

    # Assert
    assert isinstance(result_dict, dict)
    assert result_dict["test_criterion"] is True


@patch("pysatl_criterion.p_value.resolver.model.PValueResolver")
@patch("pysatl_criterion.statistics.goodness_of_fit.AbstractGoodnessOfFitStatistic")
def test_goodness_of_fit_p_value_path_rejects_hypothesis(mock_stat_cls, mock_p_value_cls):
    mock_stat = mock_stat_cls.return_value
    mock_stat.code.return_value = "test_criterion"
    mock_stat.execute_statistic.return_value = 10.0

    mock_p_value = mock_p_value_cls.return_value
    mock_p_value.resolve.return_value = 0.01

    gof_test = GoodnessOfFitTest(
        statistics=[mock_stat],
        significance_level=0.05,
        test_method=TestMethod.P_VALUE,
        p_value_resolver=mock_p_value,
    )

    # Act
    result_dict = gof_test.test(data=[1, 2, 3])

    # Assert
    assert isinstance(result_dict, dict)
    assert result_dict["test_criterion"] is False


@patch(
    "pysatl_criterion.test.goodness_of_fit_test.goodness_of_fit_test."
    "AlchemyLimitDistributionStorage"
)
def test_goodness_of_fit_default_resolver_initialization(mock_storage_cls):
    mock_instance = MagicMock()
    mock_storage_cls.create_safe.return_value = mock_instance

    mock_stat = MagicMock()
    mock_stat.code.return_value = "dummy_code"

    gof_test = GoodnessOfFitTest(
        statistics=[mock_stat], significance_level=0.05, test_method=TestMethod.CRITICAL_VALUE
    )

    assert isinstance(gof_test.cv_calculator, CompositeCriticalValueResolver)
    assert mock_storage_cls.create_safe.called
