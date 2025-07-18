from pysatl_criterion.cv_calculator.cv_calculator.cv_calculator import CVCalculator
from pysatl_criterion.p_value_calculator.p_value_calculator.p_value_calculator import (
    PValueCalculator,
)
from pysatl_criterion.persistence.limit_distribution.sqlite.sqlite import (
    SQLiteLimitDistributionStorage,
)
from pysatl_criterion.statistics.goodness_of_fit import AbstractGoodnessOfFitStatistic


class GoodnessOfFitTest:
    """
    Goodness of fit test.

    :param statistics: statistics.
    :param significance_level: significance level.
    :param test_method: test method either 'critical_value' or 'p_value'.

    """

    def __init__(
        self,
        statistics: AbstractGoodnessOfFitStatistic,
        significance_level: float,
        db_connection_string: str = "sqlite:///limit_distributions.sqlite",
        test_method: str = "critical_value",
    ):
        self.statistics = statistics
        self.significance_level = significance_level
        self.db_connection_string = db_connection_string
        self.test_method = test_method

    def test(self, data: list[float]) -> bool:
        """
        Perform goodness of fit.

        :param data: data.

        :return: True if data is good, False otherwise.
        """

        limit_distribution_storage = SQLiteLimitDistributionStorage(self.db_connection_string)
        limit_distribution_storage.init()

        data_size = len(data)
        criterion_code = self.statistics.code()
        statistics_value = self.statistics.execute_statistic(data)

        if self.test_method == "critical_value":
            cv_calculator = CVCalculator(limit_distribution_storage)

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

        else:
            p_value_calculator = PValueCalculator(limit_distribution_storage)
            p_value = p_value_calculator.calculate_p_value(
                criterion_code, data_size, statistics_value, two_tailed=self.statistics.two_tailed
            )
            return p_value >= self.significance_level
