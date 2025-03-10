from abc import ABC

from typing_extensions import override

from criterion.models import AbstractTestStatistic


class AbstractGoodnessOfFitTestStatistic(AbstractTestStatistic, ABC):
    @staticmethod
    @override
    def code():
        return "GOODNESS_OF_FIT"

    # TODO: separate abstract class for testing?
    def test(self, rvs, alpha):
        statistic = self.execute_statistic(rvs)
        if self.two_tailed:
            l_cr, r_cr = self.calculate_two_tailed_critical_values(len(rvs), alpha)
            return not (l_cr > statistic or statistic > r_cr)

        x_cr = self.calculate_critical_value(len(rvs), alpha)
        return not (statistic > x_cr)
