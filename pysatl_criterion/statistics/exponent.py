import math
from abc import ABC

import numpy as np
import scipy.special as scipy_special
import scipy.stats as scipy_stats
from typing_extensions import override

from pysatl_criterion.statistics.common import KSStatistic
from pysatl_criterion.statistics.goodness_of_fit import AbstractGoodnessOfFitStatistic
from pysatl_criterion.statistics.graph_goodness_of_fit import (
    GraphAverageDegreeTestStatistic,
    GraphConnectedComponentsTestStatistic,
    GraphEdgesNumberTestStatistic,
    GraphMaxDegreeTestStatistic,
)


class AbstractExponentialityGofStatistic(AbstractGoodnessOfFitStatistic, ABC):
    def __init__(self, lam=1):
        self.lam = lam

    @staticmethod
    @override
    def code():
        return f"EXPONENTIALITY_{AbstractGoodnessOfFitStatistic.code()}"


class EppsPulleyExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    @staticmethod
    @override
    def code():
        return f"EP_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Epps and Pulley test statistic for exponentiality.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.

        Returns
        -------
        ep : float
            The test statistic.
        """

        n = len(rvs)
        y = rvs / np.mean(rvs)
        ep = np.sqrt(48 * n) * np.sum(np.exp(-y) - 1 / 2) / n

        return ep


class KolmogorovSmirnovExponentialityGofStatistic(AbstractExponentialityGofStatistic, KSStatistic):
    def __init__(self, alternative="two-sided", lam=1):
        super().__init__()
        self.alternative = alternative
        self.lam = lam

    @staticmethod
    @override
    def code():
        return f"KS_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Kolmogorov and Smirnov test statistic for exponentiality.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.

        Returns
        -------
        ks : float
            The test statistic.
        """

        rvs = np.sort(rvs)
        cdf_vals = scipy_stats.expon.cdf(rvs)
        return KSStatistic.execute_statistic(self, rvs, cdf_vals)


class AhsanullahExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    @staticmethod
    @override
    def code():
        return f"AHS_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Statistic of the exponentiality test based on Ahsanullah characterization.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.

        Returns
        -------
        a : float
            The test statistic.
            :param rvs:
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
    @staticmethod
    @override
    def code():
        return f"ATK_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, p=0.99):
        """
        Atkinson test statistic for exponentiality.

        Parameters
        ----------
        p : float
            Statistic parameter.
        rvs : array_like
            Array of sample data.

        Returns
        -------
        atk : float
            The test statistic.
        """

        n = len(rvs)
        y = np.mean(rvs)
        m = np.mean(np.power(rvs, p))
        r = (m ** (1 / p)) / y
        atk = np.sqrt(n) * np.abs(r - scipy_special.gamma(1 + p) ** (1 / p))

        return atk


class CoxOakesExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    @staticmethod
    @override
    def code():
        return f"CO_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Cox and Oakes test statistic for exponentiality.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.

        Returns
        -------
        co : float
            The test statistic.
            :param rvs:
        """

        n = len(rvs)
        y = rvs / np.mean(rvs)
        y = np.log(y) * (1 - y)
        co = np.sum(y) + n

        return co


class CramerVonMisesExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    @staticmethod
    @override
    def code():
        return f"CVM_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Cramer-von Mises test statistic for exponentiality.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.

        Returns
        -------
        cvm : float
            The test statistic.
        """

        n = len(rvs)
        y = rvs / np.mean(rvs)
        z = np.sort(1 - np.exp(-y))
        c = (2 * np.arange(1, n + 1) - 1) / (2 * n)
        z = (z - c) ** 2
        cvm = 1 / (12 * n) + np.sum(z)

        return cvm


class DeshpandeExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    @staticmethod
    @override
    def code():
        return f"DSP_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, b=0.44):
        """
        Deshpande test statistic for exponentiality.

        Parameters
        ----------
        b : float
            Statistic parameter.
        rvs : array_like
            Array of sample data.

        Returns
        -------
        des : float
            The test statistic.
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
    @staticmethod
    @override
    def code():
        return f"EPS_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Epstein test statistic for exponentiality.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.

        Returns
        -------
        eps : float
            The test statistic.
        """

        n = len(rvs)
        rvs.sort()
        x = np.concatenate(([0], rvs))
        d = (np.arange(n, 0, -1)) * (x[1 : n + 1] - x[0:n])
        eps = 2 * n * (np.log(np.sum(d) / n) - (np.sum(np.log(d))) / n) / (1 + (n + 1) / (6 * n))

        return eps


class FroziniExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    @staticmethod
    @override
    def code():
        return f"FZ_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Frozini test statistic for exponentiality.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.

        Returns
        -------
        froz : float
            The test statistic.
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
    @staticmethod
    @override
    def code():
        return f"GINI_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Gini test statistic for exponentiality.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.

        Returns
        -------
        gini : float
            The test statistic.
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
    @staticmethod
    @override
    def code():
        return f"GD_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, r=None):
        """
        Gnedenko F-test statistic for exponentiality.

        Parameters
        ----------
        r : float
            Statistic parameter.
        rvs : array_like
            Array of sample data.

        Returns
        -------
        gd : float
            The test statistic.
        """

        if r is None:
            r = round(len(rvs) / 2)
        n = len(rvs)
        x = np.sort(np.concatenate(([0], rvs)))
        d = (np.arange(n, 0, -1)) * (x[1 : n + 1] - x[0:n])
        gd = (sum(d[:r]) / r) / (sum(d[r:]) / (n - r))

        return gd


class HarrisExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    @staticmethod
    @override
    def code():
        return f"HM_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, r=None):
        """
        Harris' modification of Gnedenko F-test.

        Parameters
        ----------
        r : float
            Statistic parameter.
        rvs : array_like
            Array of sample data.

        Returns
        -------
        hm : float
            The test statistic.
        """

        if r is None:
            r = round(len(rvs) / 4)
        n = len(rvs)
        x = np.sort(np.concatenate(([0], rvs)))
        d = (np.arange(n, 0, -1)) * (x[1 : n + 1] - x[:n])
        hm = ((np.sum(d[:r]) + np.sum(d[-r:])) / (2 * r)) / ((np.sum(d[r:-r])) / (n - 2 * r))

        return hm


class HegazyGreen1ExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    @staticmethod
    @override
    def code():
        return f"HG1_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Hegazy-Green 1 test statistic for exponentiality.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.

        Returns
        -------
        hg : float
            The test statistic.
        """

        n = len(rvs)
        x = np.sort(rvs)
        b = -np.log(1 - np.arange(1, n + 1) / (n + 1))
        hg = (n ** (-1)) * np.sum(np.abs(x - b))

        return hg


class HollanderProshanExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    @staticmethod
    @override
    def code():
        return f"HP_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Hollander-Proshan test statistic for exponentiality.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.

        Returns
        -------
        hp : float
            The test statistic.
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
    @staticmethod
    @override
    def code():
        return f"KM_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Kimber-Michael test statistic for exponentiality.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.

        Returns
        -------
        km : float
            The test statistic.
        """

        n = len(rvs)
        rvs.sort()
        y = np.mean(rvs)
        s = (2 / np.pi) * np.arcsin(np.sqrt(1 - np.exp(-(rvs / y))))
        r = (2 / np.pi) * np.arcsin(np.sqrt((np.arange(1, n + 1) - 0.5) / n))
        km = max(abs(r - s))

        return km


class KocharExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    @staticmethod
    @override
    def code():
        return f"KC_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Kochar test statistic for exponentiality.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.

        Returns
        -------
        kc : float
            The test statistic.
        """

        n = len(rvs)
        rvs.sort()
        u = np.array([(i + 1) / (n + 1) for i in range(n)])
        j = 2 * (1 - u) * (1 - np.log(1 - u)) - 1
        kc = np.sqrt(108 * n / 17) * (np.sum(j * rvs)) / np.sum(rvs)

        return kc


class LorenzExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    @staticmethod
    @override
    def code():
        return f"LZ_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, p=0.5):
        """
        Lorenz test statistic for exponentiality.

        Parameters
        ----------
        p : float
            Statistic parameter.
        rvs : array_like
            Array of sample data.

        Returns
        -------
        lz : float
            The test statistic.
        """

        n = len(rvs)
        rvs.sort()
        lz = sum(rvs[: int(n * p)]) / sum(rvs)

        return lz


class MoranExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    @staticmethod
    @override
    def code():
        return f"MN_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Moran test statistic for exponentiality.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.

        Returns
        -------
        mn : float
            The test statistic.
        """

        # n = len(rvs)
        y = np.mean(rvs)
        mn = -scipy_special.digamma(1) + np.mean(np.log(rvs / y))

        return mn


class PietraExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    @staticmethod
    @override
    def code():
        return f"PT_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Pietra test statistic for exponentiality.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.

        Returns
        -------
        pt : float
            The test statistic.
        """

        n = len(rvs)
        xm = np.mean(rvs)
        pt = np.sum(np.abs(rvs - xm)) / (2 * n * xm)

        return pt


class ShapiroWilkExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    @staticmethod
    @override
    def code():
        return f"SW_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Shapiro-Wilk test statistic for exponentiality.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.

        Returns
        -------
        sw : float
            The test statistic.
        """

        n = len(rvs)
        rvs.sort()
        y = np.mean(rvs)
        sw = n * (y - rvs[0]) ** 2 / ((n - 1) * np.sum((rvs - y) ** 2))

        return sw


class RossbergExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    @staticmethod
    @override
    def code():
        return f"RS_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Statistic of the exponentiality test based on Rossberg characterization.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.

        Returns
        -------
        rs : float
            The test statistic.
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
    @staticmethod
    @override
    def code():
        return f"WE_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        WE test statistic for exponentiality.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.

        Returns
        -------
        we : float
            The test statistic.
        """

        n = len(rvs)
        m = np.mean(rvs)
        v = np.var(rvs)
        we = (n - 1) * v / (n**2 * m**2)

        return we


class WongWongExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    @staticmethod
    @override
    def code():
        return f"WW_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Wong and Wong test statistic for exponentiality.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.

        Returns
        -------
        ww : float
            The test statistic.
        """

        # n = len(rvs)
        ww = max(rvs) / min(rvs)

        return ww


class HegazyGreen2ExponentialityGofStatistic(AbstractExponentialityGofStatistic):
    @staticmethod
    @override
    def code():
        return f"HG2_{AbstractExponentialityGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Hegazy-Green 2 test statistic for exponentiality.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.

        Returns
        -------
        hg : float
            The test statistic.
        """

        n = len(rvs)
        rvs.sort()
        b = -np.log(1 - np.arange(1, n + 1) / (n + 1))
        hg = (n ** (-1)) * np.sum((rvs - b) ** 2)

        return hg


class GraphEdgesNumberExponentialityGofStatistic(
    AbstractExponentialityGofStatistic, GraphEdgesNumberTestStatistic
):
    @staticmethod
    @override
    def code():
        super_class = AbstractExponentialityGofStatistic
        parent_code = super(super_class, super_class).code()
        return f"EdgesNumber_{parent_code}"

    @staticmethod
    @override
    def _compute_dist(rvs):
        super_class = GraphEdgesNumberTestStatistic
        parent_code = super(super_class, super_class)._compute_dist(rvs)
        return parent_code / np.mean(rvs)


class GraphMaxDegreeExponentialityGofStatistic(
    AbstractExponentialityGofStatistic, GraphMaxDegreeTestStatistic
):
    @staticmethod
    @override
    def code():
        super_class = AbstractExponentialityGofStatistic
        parent_code = super(super_class, super_class).code()
        return f"MaxDegree_{parent_code}"

    @staticmethod
    @override
    def _compute_dist(rvs):
        super_class = GraphMaxDegreeTestStatistic
        parent_code = super(super_class, super_class)._compute_dist(rvs)
        return parent_code / np.mean(rvs)


class GraphAverageDegreeExponentialityGofStatistic(
    AbstractExponentialityGofStatistic, GraphAverageDegreeTestStatistic
):
    @staticmethod
    @override
    def code():
        super_class = AbstractExponentialityGofStatistic
        parent_code = super(super_class, super_class).code()
        return f"AverageDegree_{parent_code}"

    @staticmethod
    @override
    def _compute_dist(rvs):
        super_class = GraphAverageDegreeTestStatistic
        parent_dist = super(super_class, super_class)._compute_dist(rvs)
        return parent_dist / np.mean(rvs)


class GraphConnectedComponentsExponentialityGofStatistic(
    AbstractExponentialityGofStatistic, GraphConnectedComponentsTestStatistic
):
    @staticmethod
    @override
    def code():
        super_class = AbstractExponentialityGofStatistic
        parent_code = super(super_class, super_class).code()
        return f"ConnectedComponents_{parent_code}"

    @staticmethod
    @override
    def _compute_dist(rvs):
        super_class = GraphConnectedComponentsTestStatistic
        parent_dist = super(super_class, super_class)._compute_dist(rvs)
        return parent_dist / np.mean(rvs)
