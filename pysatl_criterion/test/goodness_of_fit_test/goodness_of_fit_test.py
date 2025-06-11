from pysatl_criterion.cv_calculator.cv_calculator.cv_calculator import CVCalculator
from pysatl_criterion.persistence.limit_distribution.sqlite.sqlite import (
    SQLiteLimitDistributionStorage,
)
from pysatl_criterion.statistics.goodness_of_fit import AbstractGoodnessOfFitStatistic


class GoodnessOfFitTest:
    """
    Goodness of fit test.

    :param statistics: statistics.
    :param significance_level: significance level.
    """

    def __init__(
        self,
        statistics: AbstractGoodnessOfFitStatistic,
        significance_level: float,
        db_connection_string: str = "sqlite:///limit_distributions.sqlite",
    ):
        self.statistics = statistics
        self.significance_level = significance_level
        self.db_connection_string = db_connection_string

    def test(self, data: list[float]) -> bool:
        """
        Perform goodness of fit.

        :param data: data.

        :return: True if data is good, False otherwise.
        """

        limit_distribution_storage = SQLiteLimitDistributionStorage(self.db_connection_string)
        limit_distribution_storage.init()

        cv_calculator = CVCalculator(limit_distribution_storage)

        data_size = len(data)
        criterion_code = self.statistics.code()
        statistics_value = self.statistics.execute_statistic(data)
        if self.statistics.two_tailed:
            critical_value_left, critical_value_right = (
                cv_calculator.calculate_two_tailed_critical_values(
                    criterion_code, data_size, self.significance_level
                )
            )
            if critical_value_left <= statistics_value <= critical_value_right:
                return True
            else:
                return False
        else:
            critical_value = cv_calculator.calculate_critical_value(
                criterion_code, data_size, self.significance_level
            )
            if statistics_value <= critical_value:
                return True
            else:
                return False
