from pysatl_criterion.constants import LOCAL_PYSATL_URL, REMOTE_PYSATL_URL
from pysatl_criterion.critical_value.loader.remote_loader import CriticalValueLoader
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
        statistics: list[AbstractGoodnessOfFitStatistic],
        significance_level: float,
        cv_resolver: CriticalValueResolver | None = None,
        p_value_resolver: PValueResolver | None = None,
        test_method: TestMethod = TestMethod.CRITICAL_VALUE,
        alternative: HypothesisType = HypothesisType.RIGHT,
    ):
        self.statistics_list = statistics
        self.significance_level = significance_level
        self.test_method = test_method
        self.alternative = alternative

        cv_local_storage = AlchemyLimitDistributionStorage.create_safe(
            LOCAL_PYSATL_URL, label="Local CV Storage"
        )
        if cv_local_storage is None:
            raise RuntimeError(
                "Local storage is required for caching, but could not be initialized."
            )

        cv_remote_storage = AlchemyLimitDistributionStorage.create_safe(
            REMOTE_PYSATL_URL, label="Remote CV Storage"
        )

        if cv_resolver is None and test_method == TestMethod.CRITICAL_VALUE:
            cv_loader = CriticalValueLoader(cv_local_storage, cv_remote_storage)
            cv_local_resolver = StorageCriticalValueResolver(cv_local_storage)
            cv_resolver = CompositeCriticalValueResolver(cv_local_resolver, cv_loader)

        if p_value_resolver is None and test_method == TestMethod.P_VALUE:
            p_storage = AlchemyLimitDistributionStorage.create_safe(
                REMOTE_PYSATL_URL, "Remote P-Storage"
            ) or AlchemyLimitDistributionStorage.create_safe(LOCAL_PYSATL_URL, "Local P-Storage")

            if p_storage is None:
                raise RuntimeError("No available storage for P-value calculation.")

            p_value_resolver = CalculationPValueResolver(p_storage)

        self.cv_calculator = cv_resolver
        self.p_value_resolver = p_value_resolver

    def test(self, data: list[float]) -> dict[str, bool]:
        """
        Perform goodness of fit.

        :param data: data to test.

        :return: a dictionary mapping criterion code to test result (True/False).
        """

        data_size = len(data)
        stats_map = {s.code(): s.execute_statistic(data) for s in self.statistics_list}
        codes = list(stats_map.keys())

        if self.cv_calculator and self.test_method == TestMethod.CRITICAL_VALUE:
            critical_areas = self.cv_calculator.resolve_bulk(
                codes, data_size, self.significance_level, self.alternative
            )
            return {
                code: (
                    critical_areas[code].contains(stats_map[code])
                    if code in critical_areas
                    else False
                )
                for code in codes
            }

        elif self.p_value_resolver and self.test_method == TestMethod.P_VALUE:
            results = {}
            for code, stat in stats_map.items():
                p_value = self.p_value_resolver.resolve(code, data_size, stat, self.alternative)
                results[code] = p_value is not None and p_value >= self.significance_level
            return results

        else:
            raise ValueError(f"Invalid test method {self.test_method}.")
