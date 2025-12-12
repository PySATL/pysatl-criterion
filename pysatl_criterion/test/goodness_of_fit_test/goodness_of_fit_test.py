from pysatl_criterion.constants import LOCAL_LIMIT_DISTRIBUTION_URL
from pysatl_criterion.critical_value.resolver.composite_resolver import (
    CompositeCriticalValueResolver,
)
from pysatl_criterion.critical_value.resolver.model import CriticalValueResolver
from pysatl_criterion.critical_value.resolver.storage_resolver import StorageCriticalValueResolver
from pysatl_criterion.p_value.resolver.calculation_resolver import CalculationPValueResolver
from pysatl_criterion.p_value.resolver.model import PValueResolver
from pysatl_criterion.persistence.limit_distribution.datastorage.datastorage import (
    AlchemyLimitDistributionStorage,
)
from pysatl_criterion.statistics.goodness_of_fit import AbstractGoodnessOfFitStatistic
from pysatl_criterion.statistics.models import HypothesisType
from pysatl_criterion.test.model import TestMethod


class GoodnessOfFitTest:
    """
    Goodness of fit test.

    :param statistics: statistics.
    :param significance_level: significance level.
    :param test_method: test method either 'critical_value' or 'p_value'.
    :param alternative: test alternative.

    """

    def __init__(
        self,
        statistics: AbstractGoodnessOfFitStatistic,
        significance_level: float,
        cv_resolver: CriticalValueResolver | None = None,
        p_value_resolver: PValueResolver | None = None,
        test_method: TestMethod = TestMethod.CRITICAL_VALUE,
        alternative: HypothesisType = HypothesisType.RIGHT,
    ):
        self.statistics = statistics
        self.significance_level = significance_level
        self.test_method = test_method
        self.alternative = alternative

        if cv_resolver is None and test_method == TestMethod.CRITICAL_VALUE:
            cv_local_storage = AlchemyLimitDistributionStorage(LOCAL_LIMIT_DISTRIBUTION_URL)
            cv_local_storage.init()

            cv_remote_storage = AlchemyLimitDistributionStorage(LOCAL_LIMIT_DISTRIBUTION_URL)
            cv_remote_storage.init()

            cv_resolver = CompositeCriticalValueResolver(
                StorageCriticalValueResolver(cv_local_storage),
                StorageCriticalValueResolver(cv_remote_storage),
            )

        if p_value_resolver is None and test_method == TestMethod.P_VALUE:
            p_storage = AlchemyLimitDistributionStorage(LOCAL_LIMIT_DISTRIBUTION_URL)
            p_storage.init()

            p_value_resolver = CalculationPValueResolver(p_storage)

        self.cv_calculator = cv_resolver
        self.p_value_resolver = p_value_resolver

    def test(self, data: list[float]) -> bool:
        """
        Perform goodness of fit.

        :param data: data to test.

        :return: True if data is fitted distribution, False otherwise.
        """

        data_size = len(data)
        criterion_code = self.statistics.code()
        statistics_value = self.statistics.execute_statistic(data)

        if self.test_method == TestMethod.CRITICAL_VALUE:
            critical_area = self.cv_calculator.resolve(
                criterion_code,
                data_size,
                self.significance_level,
                self.alternative,
            )

            if critical_area is None:
                raise ValueError(
                    f"Limit distribution for criterion {criterion_code} and "
                    f"sample size {data_size} does not exist."
                )

            return critical_area.contains(statistics_value)

        elif self.test_method == TestMethod.P_VALUE:
            p_value = self.p_value_resolver.resolve(
                criterion_code,
                data_size,
                statistics_value,
                self.alternative,
            )

            return p_value is not None and p_value >= self.significance_level
        else:
            raise ValueError(f"Invalid test method {self.test_method}.")
