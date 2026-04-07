import math
from abc import ABC

import numpy as np
import scipy.special as scipy_special
import scipy.stats as scipy_stats
from typing_extensions import override

from pysatl_criterion.statistics.common import KSStatistic
from pysatl_criterion.statistics.goodness_of_fit import AbstractGoodnessOfFitStatistic
from pysatl_criterion.statistics.graph_goodness_of_fit import (
    AbstractGraphTestStatistic,
    GraphAverageDegreeTestStatistic,
    GraphCliqueNumberTestStatistic,
    GraphConnectedComponentsTestStatistic,
    GraphEdgesNumberTestStatistic,
    GraphIndependenceNumberTestStatistic,
    GraphMaxDegreeTestStatistic,
)


class AbstractExponentialityGofStatistic(AbstractGoodnessOfFitStatistic, ABC):
    """
    Abstract base class for exponentiality goodness-of-fit statistics.

    Provides common interface and utility methods for tests that check
    whether data follows an exponential distribution with given rate parameter.
    """

    def __init__(self, lam=1):
        self.lam = lam

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for exponentiality statistics.

        :return: string code in format "EXPONENTIALITY_{parent_code}".
        """
        return f"EXPONENTIALITY_{AbstractGoodnessOfFitStatistic.code()}"


class EppsPulleyExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """
    Epps-Pulley test statistic for exponentiality.

    Test based on characteristic function approach for testing exponentiality.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "EP".
        """
        return "EP"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "EP_EXPONENTIALITY_{parent_code}".
        """
        short_code = EppsPulleyExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Epps-Pulley test statistic for exponentiality.

        :param rvs: array of sample data.
        :return: Epps-Pulley test statistic value.
        """

        n = len(rvs)
        y = rvs / np.mean(rvs)
        ep = np.sqrt(48 * n) * np.sum(np.exp(-y) - 1 / 2) / n

        return ep


class KolmogorovSmirnovExponentialityGofStatistic(AbstractExponentialityGofStatistic, KSStatistic):
    """
    Kolmogorov-Smirnov test statistic for exponentiality.

    Applies the KS test to check if data follows exponential distribution.
    """

    def __init__(self, alternative="two-sided", lam=1):
        super().__init__()
        self.alternative = alternative
        self.lam = lam

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "KS".
        """
        return "KS"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "KS_EXPONENTIALITY_{parent_code}".
        """
        short_code = KolmogorovSmirnovExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Kolmogorov-Smirnov test statistic for exponentiality.

        :param rvs: array of sample data.
        :return: Kolmogorov-Smirnov test statistic value.
        """

        rvs = np.sort(rvs)
        cdf_vals = scipy_stats.expon.cdf(rvs)
        return KSStatistic.execute_statistic(self, rvs, cdf_vals)


class AhsanullahExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """
    Ahsanullah characterization test statistic for exponentiality.

    Test based on Ahsanullah's characterization of exponential distribution
    using order statistics properties.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "AHS".
        """
        return "AHS"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "AHS_EXPONENTIALITY_{parent_code}".
        """
        short_code = AhsanullahExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Ahsanullah test statistic for exponentiality.

        :param rvs: array of sample data.
        :return: Ahsanullah test statistic value.
        """

        n = len(rvs)
        h = 0
        g = 0
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if abs(rvs[i] - rvs[j]) < rvs[k]:
                        h += 1
                    if 2 * min(rvs[i], rvs[j]) < rvs[k]:
                        g += 1
        a = (h - g) / (n**3)

        return a


class AtkinsonExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """
    Atkinson test statistic for exponentiality.

    Test based on comparison of sample moments with theoretical moments
    of exponential distribution using power transformation.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "ATK".
        """
        return "ATK"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "ATK_EXPONENTIALITY_{parent_code}".
        """
        short_code = AtkinsonExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, p=0.99):
        """
        Execute the Atkinson test statistic for exponentiality.

        :param rvs: array of sample data.
        :param p: power parameter for moment comparison (default is 0.99).
        :return: Atkinson test statistic value.
        """

        n = len(rvs)
        y = np.mean(rvs)
        m = np.mean(np.power(rvs, p))
        r = (m ** (1 / p)) / y
        atk = np.sqrt(n) * np.abs(r - scipy_special.gamma(1 + p) ** (1 / p))

        return atk


class CoxOakesExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """
    Cox-Oakes test statistic for exponentiality.

    Test based on score function approach for testing exponentiality
    against increasing failure rate alternatives.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "CO".
        """
        return "CO"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "CO_EXPONENTIALITY_{parent_code}".
        """
        short_code = CoxOakesExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Cox-Oakes test statistic for exponentiality.

        :param rvs: array of sample data.
        :return: Cox-Oakes test statistic value.
        """

        n = len(rvs)
        y = rvs / np.mean(rvs)
        y = np.log(y) * (1 - y)
        co = np.sum(y) + n

        return co


class CramerVonMisesExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """
    Cramér-von Mises test statistic for exponentiality.

    Applies the CVM test to check if data follows exponential distribution,
    measuring integrated squared difference between empirical and theoretical CDF.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "CVM".
        """
        return "CVM"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "CVM_EXPONENTIALITY_{parent_code}".
        """
        short_code = CramerVonMisesExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Cramér-von Mises test statistic for exponentiality.

        :param rvs: array of sample data.
        :return: Cramér-von Mises test statistic value.
        """

        n = len(rvs)
        y = rvs / np.mean(rvs)
        z = np.sort(1 - np.exp(-y))
        c = (2 * np.arange(1, n + 1) - 1) / (2 * n)
        z = (z - c) ** 2
        cvm = 1 / (12 * n) + np.sum(z)

        return cvm


class DeshpandeExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """
    Deshpande test statistic for exponentiality.

    Test based on spacings and order statistics for testing exponentiality.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "DSP".
        """
        return "DSP"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "DSP_EXPONENTIALITY_{parent_code}".
        """
        short_code = DeshpandeExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, b=0.44):
        """
        Execute the Deshpande test statistic for exponentiality.

        :param rvs: array of sample data.
        :param b: threshold parameter for spacing comparison (default is 0.44).
        :return: Deshpande test statistic value.
        """

        n = len(rvs)
        des = 0
        for i in range(n):
            for k in range(n):
                if (i != k) and (rvs[i] > b * rvs[k]):
                    des += 1
        des /= n * (n - 1)

        return des


class EpsteinExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """
    Epstein test statistic for exponentiality.

    Test based on normalized spacings between order statistics.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "EPS".
        """
        return "EPS"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "EPS_EXPONENTIALITY_{parent_code}".
        """
        short_code = EpsteinExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Epstein test statistic for exponentiality.

        :param rvs: array of sample data.
        :return: Epstein test statistic value.
        """

        n = len(rvs)
        rvs.sort()
        x = np.concatenate(([0], rvs))
        d = (np.arange(n, 0, -1)) * (x[1 : n + 1] - x[0:n])
        eps = 2 * n * (np.log(np.sum(d) / n) - (np.sum(np.log(d))) / n) / (1 + (n + 1) / (6 * n))

        return eps


class FroziniExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """
    Frozini test statistic for exponentiality.

    Test based on empirical distribution function comparison with exponential CDF.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "FZ".
        """
        return "FZ"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "FZ_EXPONENTIALITY_{parent_code}".
        """
        short_code = FroziniExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Frozini test statistic for exponentiality.

        :param rvs: array of sample data.
        :return: Frozini test statistic value.
        """

        n = len(rvs)
        rvs.sort()
        rvs = np.array(rvs)
        y = np.mean(rvs)
        froz = (1 / np.sqrt(n)) * np.sum(
            np.abs(1 - np.exp(-rvs / y) - (np.arange(1, n + 1) - 0.5) / n)
        )

        return froz


class GiniExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """
    Gini test statistic for exponentiality.

    Test based on Gini's mean difference applied to exponential distribution testing.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "GINI".
        """
        return "GINI"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "GINI_EXPONENTIALITY_{parent_code}".
        """
        short_code = GiniExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Gini test statistic for exponentiality.

        :param rvs: array of sample data.
        :return: Gini test statistic value.
        """

        n = len(rvs)
        a = np.arange(1, n)
        b = np.arange(n - 1, 0, -1)
        a = a * b
        x = np.sort(rvs)
        k = x[1:] - x[:-1]
        gini = np.sum(k * a) / ((n - 1) * np.sum(x))

        return gini


class GnedenkoExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """
    Gnedenko F-test statistic for exponentiality.

    Test based on ratio of mean spacings for testing exponentiality.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "GD".
        """
        return "GD"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "GD_EXPONENTIALITY_{parent_code}".
        """
        short_code = GnedenkoExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, r=None):
        """
        Execute the Gnedenko F-test statistic for exponentiality.

        :param rvs: array of sample data.
        :param r: split point for spacing ratio (default is n//2).
        :return: Gnedenko F-test statistic value.
        """

        if r is None:
            r = round(len(rvs) / 2)
        n = len(rvs)
        x = np.sort(np.concatenate(([0], rvs)))
        d = (np.arange(n, 0, -1)) * (x[1 : n + 1] - x[0:n])
        gd = (sum(d[:r]) / r) / (sum(d[r:]) / (n - r))

        return gd


class HarrisExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """
    Harris modification of Gnedenko F-test for exponentiality.

    Improved version of Gnedenko test using symmetric spacing comparison.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "HM".
        """
        return "HM"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "HM_EXPONENTIALITY_{parent_code}".
        """
        short_code = HarrisExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, r=None):
        """
        Execute the Harris modification of Gnedenko F-test for exponentiality.

        :param rvs: array of sample data.
        :param r: split point for symmetric spacing comparison (default is n//4).
        :return: Harris test statistic value.
        """

        if r is None:
            r = round(len(rvs) / 4)
        n = len(rvs)
        x = np.sort(np.concatenate(([0], rvs)))
        d = (np.arange(n, 0, -1)) * (x[1 : n + 1] - x[:n])
        hm = ((np.sum(d[:r]) + np.sum(d[-r:])) / (2 * r)) / ((np.sum(d[r:-r])) / (n - 2 * r))

        return hm


class HegazyGreen1ExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """
    Hegazy-Green 1 test statistic for exponentiality.

    Test based on L1 distance between ordered sample and theoretical quantiles.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "HG1".
        """
        return "HG1"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "HG1_EXPONENTIALITY_{parent_code}".
        """
        short_code = HegazyGreen1ExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Hegazy-Green 1 test statistic for exponentiality.

        :param rvs: array of sample data.

        :return: Hegazy-Green 1 test statistic value.
        """

        n = len(rvs)
        x = np.sort(rvs)
        b = -np.log(1 - np.arange(1, n + 1) / (n + 1))
        hg = (n ** (-1)) * np.sum(np.abs(x - b))

        return hg


class HollanderProshanExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """
    Hollander-Proshan test statistic for exponentiality.

    Test based on total time on test transform for testing exponentiality.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "HP".
        """
        return "HP"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "HP_EXPONENTIALITY_{parent_code}".
        """
        short_code = HollanderProshanExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Hollander-Proshan test statistic for exponentiality.

        :param rvs: array of sample data.
        :return: Hollander-Proshan test statistic value.
        """

        n = len(rvs)
        t = 0
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if (i != j) and (i != k) and (j < k) and (rvs[i] > rvs[j] + rvs[k]):
                        t += 1
        hp = (2 / (n * (n - 1) * (n - 2))) * t

        return hp


class KimberMichaelExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """
    Kimber-Michael test statistic for exponentiality.

    Test based on arcsine transformation of empirical and theoretical CDFs.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "KM".
        """
        return "KM"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "KM_EXPONENTIALITY_{parent_code}".
        """
        short_code = KimberMichaelExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Kimber-Michael test statistic for exponentiality.

        :param rvs: array of sample data.
        :return: Kimber-Michael test statistic value.
        """

        n = len(rvs)
        rvs.sort()
        y = np.mean(rvs)
        s = (2 / np.pi) * np.arcsin(np.sqrt(1 - np.exp(-(rvs / y))))
        r = (2 / np.pi) * np.arcsin(np.sqrt((np.arange(1, n + 1) - 0.5) / n))
        km = max(abs(r - s))

        return km


class KocharExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """
    Kochar test statistic for exponentiality.

    Test based on weighted linear combination of order statistics.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "KC".
        """
        return "KC"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "KC_EXPONENTIALITY_{parent_code}".
        """
        short_code = KocharExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Kochar test statistic for exponentiality.

        :param rvs: array of sample data.
        :return: Kochar test statistic value.
        """

        n = len(rvs)
        rvs.sort()
        u = np.array([(i + 1) / (n + 1) for i in range(n)])
        j = 2 * (1 - u) * (1 - np.log(1 - u)) - 1
        kc = np.sqrt(108 * n / 17) * (np.sum(j * rvs)) / np.sum(rvs)

        return kc


class LorenzExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """
    Lorenz test statistic for exponentiality.

    Test based on Lorenz curve comparison for testing exponentiality.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "LZ".
        """
        return "LZ"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "LZ_EXPONENTIALITY_{parent_code}".
        """
        short_code = LorenzExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, p=0.5):
        """
        Execute the Lorenz test statistic for exponentiality.

        :param rvs: array of sample data.
        :param p: quantile parameter for Lorenz curve (default is 0.5).
        :return: Lorenz test statistic value.
        """

        n = len(rvs)
        rvs.sort()
        lz = sum(rvs[: int(n * p)]) / sum(rvs)

        return lz


class MoranExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """
    Moran test statistic for exponentiality.

    Test based on digamma function and log-transformed data for testing exponentiality.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "MN".
        """
        return "MN"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "MN_EXPONENTIALITY_{parent_code}".
        """
        short_code = MoranExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Moran test statistic for exponentiality.

        :param rvs: array of sample data.
        :return: Moran test statistic value.
        """

        # n = len(rvs)
        y = np.mean(rvs)
        mn = -scipy_special.digamma(1) + np.mean(np.log(rvs / y))

        return mn


class PietraExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """
    Pietra test statistic for exponentiality.

    Test based on mean absolute deviation for testing exponentiality.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "PT".
        """
        return "PT"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "PT_EXPONENTIALITY_{parent_code}".
        """
        short_code = PietraExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Pietra test statistic for exponentiality.

        :param rvs: array of sample data.
        :return: Pietra test statistic value.
        """

        n = len(rvs)
        xm = np.mean(rvs)
        pt = np.sum(np.abs(rvs - xm)) / (2 * n * xm)

        return pt


class ShapiroWilkExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """
    Shapiro-Wilk test statistic for exponentiality.

    Adaptation of Shapiro-Wilk test for testing exponential distribution.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "SW".
        """
        return "SW"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "SW_EXPONENTIALITY_{parent_code}".
        """
        short_code = ShapiroWilkExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Shapiro-Wilk test statistic for exponentiality.

        :param rvs: array of sample data.
        :return: Shapiro-Wilk test statistic value.
        """

        n = len(rvs)
        rvs.sort()
        y = np.mean(rvs)
        sw = n * (y - rvs[0]) ** 2 / ((n - 1) * np.sum((rvs - y) ** 2))

        return sw


class RossbergExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """
    Rossberg characterization test statistic for exponentiality.

    Test based on Rossberg's characterization using triplets of order statistics.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "RS".
        """
        return "RS"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "RS_EXPONENTIALITY_{parent_code}".
        """
        short_code = RossbergExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Rossberg test statistic for exponentiality.

        :param rvs: array of sample data.
        :return: Rossberg test statistic value.
        """

        n = len(rvs)
        sh = 0
        sg = 0
        for m in range(n):
            h = 0
            for i in range(n - 2):
                for j in range(i + 1, n - 1):
                    for k in range(j + 1, n):
                        if (
                            rvs[i]
                            + rvs[j]
                            + rvs[k]
                            - 2 * min(rvs[i], rvs[j], rvs[k])
                            - max(rvs[i], rvs[j], rvs[k])
                            < rvs[m]
                        ):
                            h += 1
            h = ((6 * math.factorial(n - 3)) / math.factorial(n)) * h
            sh += h
        for m in range(n):
            g = 0
            for i in range(n - 1):
                for j in range(i + 1, n):
                    if min(rvs[i], rvs[j]) < rvs[m]:
                        g += 1
            g = ((2 * math.factorial(n - 2)) / math.factorial(n)) * g
            sg += g
        rs = sh - sg
        rs /= n

        return rs


class WeExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """
    WE test statistic for exponentiality.

    Test based on coefficient of variation for testing exponentiality.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "WE".
        """
        return "WE"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "WE_EXPONENTIALITY_{parent_code}".
        """
        short_code = WeExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the WE test statistic for exponentiality.

        :param rvs: array of sample data.
        :return: WE test statistic value.
        """

        n = len(rvs)
        m = np.mean(rvs)
        v = np.var(rvs)
        we = (n - 1) * v / (n**2 * m**2)

        return we


class WongWongExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """
    Wong-Wong test statistic for exponentiality.

    Test based on ratio of maximum to minimum observation for testing exponentiality.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "WW".
        """
        return "WW"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "WW_EXPONENTIALITY_{parent_code}".
        """
        short_code = WongWongExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Wong-Wong test statistic for exponentiality.

        :param rvs: array of sample data.
        :return: Wong-Wong test statistic value.
        """

        # n = len(rvs)
        ww = max(rvs) / min(rvs)

        return ww


class HegazyGreen2ExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    """
    Hegazy-Green 2 test statistic for exponentiality.

    Test based on L2 distance between ordered sample and theoretical quantiles.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "HG2".
        """
        return "HG2"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "HG2_EXPONENTIALITY_{parent_code}".
        """
        short_code = HegazyGreen2ExponentialityGofStatistic.short_code()
        return f"{short_code}_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Hegazy-Green 2 test statistic for exponentiality.

        :param rvs: array of sample data.
        :return: Hegazy-Green 2 test statistic value.
        """

        n = len(rvs)
        rvs.sort()
        b = -np.log(1 - np.arange(1, n + 1) / (n + 1))
        hg = (n ** (-1)) * np.sum((rvs - b) ** 2)

        return hg


class AbstractGraphExponentialityGofStatistic(
    AbstractExponentialityGofStatistic, AbstractGraphTestStatistic
):
    """
    Abstract base class for graph-based exponentiality tests.

    Combines exponentiality testing with graph-theoretic statistics
    for analyzing data structure properties.
    """

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for graph-based exponentiality statistics.

        :return: string code in format "GRAPH_EXPONENTIALITY_{parent_code}".
        """
        parent_code = AbstractExponentialityGofStatistic.code()
        return f"GRAPH_{parent_code}"

    @staticmethod
    @override
    def _compute_dist(rvs):
        """
        Compute normalized distance for graph-based exponentiality test.

        :param rvs: array of sample data.
        :return: normalized distance value.
        """
        base_dist = AbstractGraphTestStatistic._compute_dist(rvs)
        mean = np.mean(rvs)
        return base_dist / mean if mean != 0 else base_dist


class GraphEdgesNumberExponentialityGofStatistic(
    AbstractGraphExponentialityGofStatistic, GraphEdgesNumberTestStatistic
):
    """
    Graph edges number test statistic for exponentiality.

    Applies exponentiality test using graph edges count as test statistic.
    """

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "{short_code}_GRAPH_EXPONENTIALITY_{parent_code}".
        """
        parent_code = AbstractGraphExponentialityGofStatistic.code()
        short_code = GraphEdgesNumberExponentialityGofStatistic.short_code()
        return f"{short_code}_{parent_code}"


class GraphMaxDegreeExponentialityGofStatistic(
    AbstractGraphExponentialityGofStatistic, GraphMaxDegreeTestStatistic
):
    """
    Graph maximum degree test statistic for exponentiality.

    Applies exponentiality test using graph maximum vertex degree as test statistic.
    """

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "{short_code}_GRAPH_EXPONENTIALITY_{parent_code}".
        """
        parent_code = AbstractGraphExponentialityGofStatistic.code()
        short_code = GraphMaxDegreeExponentialityGofStatistic.short_code()
        return f"{short_code}_{parent_code}"


class GraphAverageDegreeExponentialityGofStatistic(
    AbstractGraphExponentialityGofStatistic, GraphAverageDegreeTestStatistic
):
    """
    Graph average degree test statistic for exponentiality.

    Applies exponentiality test using graph average vertex degree as test statistic.
    """

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "{short_code}_GRAPH_EXPONENTIALITY_{parent_code}".
        """
        parent_code = AbstractGraphExponentialityGofStatistic.code()
        short_code = GraphAverageDegreeExponentialityGofStatistic.short_code()
        return f"{short_code}_{parent_code}"


class GraphConnectedComponentsExponentialityGofStatistic(
    AbstractGraphExponentialityGofStatistic, GraphConnectedComponentsTestStatistic
):
    """
    Graph connected components test statistic for exponentiality.

    Applies exponentiality test using number of connected components as test statistic.
    """

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "{short_code}_GRAPH_EXPONENTIALITY_{parent_code}".
        """
        parent_code = AbstractGraphExponentialityGofStatistic.code()
        short_code = GraphConnectedComponentsExponentialityGofStatistic.short_code()
        return f"{short_code}_{parent_code}"


class GraphCliqueNumberExponentialityGofStatistic(
    AbstractGraphExponentialityGofStatistic, GraphCliqueNumberTestStatistic
):
    """
    Graph clique number test statistic for exponentiality.

    Applies exponentiality test using graph clique number as test statistic.
    """

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "{short_code}_GRAPH_EXPONENTIALITY_{parent_code}".
        """
        parent_code = AbstractGraphExponentialityGofStatistic.code()
        short_code = GraphCliqueNumberExponentialityGofStatistic.short_code()
        return f"{short_code}_{parent_code}"


class GraphIndependenceNumberExponentialityGofStatistic(
    AbstractGraphExponentialityGofStatistic, GraphIndependenceNumberTestStatistic
):
    """
    Graph independence number test statistic for exponentiality.

    Applies exponentiality test using graph independence number as test statistic.
    """

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "{short_code}_GRAPH_EXPONENTIALITY_{parent_code}".
        """
        parent_code = AbstractGraphExponentialityGofStatistic.code()
        short_code = GraphIndependenceNumberExponentialityGofStatistic.short_code()
        return f"{short_code}_{parent_code}"
