from abc import ABC

import numpy as np
import scipy.stats as scipy_stats
from typing_extensions import override

from pysatl_criterion.statistics.common import (
    ADStatistic,
    Chi2Statistic,
    CrammerVonMisesStatistic,
    KSStatistic,
    LillieforsTest,
)
from pysatl_criterion.statistics.goodness_of_fit import AbstractGoodnessOfFitStatistic


class AbstractUniformGofStatistic(AbstractGoodnessOfFitStatistic, ABC):
    """
    Abstract base class for Uniform distribution goodness-of-fit statistics.
    """

    def __init__(self, a=0, b=1):
        if b <= a:
            raise ValueError("b must be greater than a")
        self.a = a
        self.b = b

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for Uniform statistics.

        :return: string code in format "UNIFORM_{parent_code}".
        """
        return f"UNIFORM_{AbstractGoodnessOfFitStatistic.code()}"

    def _validate_input(self, rvs):
        """
        Validate input data for Uniform distribution tests.

        :param rvs: array of sample data to validate.
        :return: validated numpy array.
        :raises ValueError: if any value is outside the interval [a, b].
        """
        rvs_array = np.asarray(rvs)
        if np.any((rvs_array < self.a) | (rvs_array > self.b)):
            raise ValueError(
                f"Uniform distribution values must be in the interval [{self.a}, {self.b}]"
            )
        return rvs_array


class KolmogorovSmirnovUniformGofStatistic(AbstractUniformGofStatistic, KSStatistic):
    """
    Kolmogorov-Smirnov test statistic for Uniform distribution.
    """

    def __init__(self, a=0, b=1, alternative="two-sided", mode="auto"):
        AbstractUniformGofStatistic.__init__(self, a, b)
        KSStatistic.__init__(self, alternative, mode)

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

        :return: string code in format "KS_UNIFORM_{parent_code}".
        """
        short_code = KolmogorovSmirnovUniformGofStatistic.short_code()
        return f"{short_code}_{AbstractUniformGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Kolmogorov-Smirnov test statistic.

        :param rvs: array of sample data from Uniform distribution (values in [a, b]).
        :return: Kolmogorov-Smirnov test statistic value.
        """
        rvs = self._validate_input(rvs)

        rvs_sorted = np.sort(rvs)
        cdf_vals = scipy_stats.uniform.cdf(rvs_sorted, loc=self.a, scale=self.b - self.a)
        return KSStatistic.execute_statistic(self, rvs_sorted, cdf_vals)


class AndersonDarlingUniformGofStatistic(AbstractUniformGofStatistic, ADStatistic):
    """
    Anderson-Darling test statistic for Uniform distribution.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "AD".
        """
        return "AD"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "AD_UNIFORM_{parent_code}".
        """
        short_code = AndersonDarlingUniformGofStatistic.short_code()
        return f"{short_code}_{AbstractUniformGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Anderson-Darling test statistic.

        :param rvs: array of sample data from Uniform distribution (values in [a, b]).
        :return: Anderson-Darling test statistic value.
        """
        rvs = self._validate_input(rvs)

        rvs_sorted = np.sort(rvs)
        logcdf = scipy_stats.uniform.logcdf(rvs_sorted, loc=self.a, scale=self.b - self.a)
        logsf = scipy_stats.uniform.logsf(rvs_sorted, loc=self.a, scale=self.b - self.a)
        return ADStatistic.execute_statistic(self, rvs=rvs, log_cdf=logcdf, log_sf=logsf)


class CrammerVonMisesUniformGofStatistic(AbstractUniformGofStatistic, CrammerVonMisesStatistic):
    """
    Cramér-von Mises test statistic for Uniform distribution.
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

        :return: string code in format "CVM_UNIFORM_{parent_code}".
        """
        short_code = CrammerVonMisesUniformGofStatistic.short_code()
        return f"{short_code}_{AbstractUniformGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Cramér-von Mises test statistic.

        :param rvs: array of sample data from Uniform distribution (values in [a, b]).
        :return: Cramér-von Mises test statistic value.
        """
        rvs = self._validate_input(rvs)

        rvs_sorted = np.sort(rvs)
        cdf_vals = scipy_stats.uniform.cdf(rvs_sorted, loc=self.a, scale=self.b - self.a)
        return CrammerVonMisesStatistic.execute_statistic(self, rvs_sorted, cdf_vals)


class LillieforsTestUniformGofStatistic(AbstractUniformGofStatistic, LillieforsTest):
    """
    Lilliefors test statistic for Uniform distribution.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "LILLIE".
        """
        return "LILLIE"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "LILLIE_UNIFORM_{parent_code}".
        """
        short_code = LillieforsTestUniformGofStatistic.short_code()
        return f"{short_code}_{AbstractUniformGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Lilliefors test statistic.

        :param rvs: array of sample data from Uniform distribution (values in [a, b]).
        :return: Lilliefors test statistic value.
        """
        rvs = self._validate_input(rvs)

        rvs_sorted = np.sort(rvs)
        cdf_vals = scipy_stats.uniform.cdf(rvs_sorted, loc=self.a, scale=self.b - self.a)
        return LillieforsTest.execute_statistic(self, rvs_sorted, cdf_vals)


class Chi2PearsonUniformGofStatistic(AbstractUniformGofStatistic, Chi2Statistic):
    """
    Pearson's Chi-squared test statistic for Uniform distribution.
    """

    def __init__(self, a=0, b=1, lambda_=1, bins="sturges"):
        AbstractUniformGofStatistic.__init__(self, a, b)
        Chi2Statistic.__init__(self)
        self.lambda_ = lambda_
        self.bins = bins

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "CHI2_PEARSON".
        """
        return "CHI2_PEARSON"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "CHI2_PEARSON_UNIFORM_{parent_code}".
        """
        short_code = Chi2PearsonUniformGofStatistic.short_code()
        return f"{short_code}_{AbstractUniformGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute Pearson's Chi-squared test statistic.

        :param rvs: array of sample data from Uniform distribution (values in [a, b]).
        :return: Chi-squared test statistic value.
        """
        rvs = self._validate_input(rvs)

        n = len(rvs)
        if isinstance(self.bins, str):
            if self.bins == "sturges":
                num_bins = int(np.ceil(np.log2(n) + 1))
            elif self.bins == "sqrt":
                num_bins = int(np.ceil(np.sqrt(n)))
            elif self.bins == "auto":
                h = 3.5 * np.std(rvs) / (n ** (1 / 3))
                num_bins = int(np.ceil((self.b - self.a) / h))
            else:
                num_bins = 10
        else:
            num_bins = int(self.bins)
        num_bins = max(2, num_bins)

        observed, bin_edges = np.histogram(rvs, bins=num_bins, range=(self.a, self.b))
        expected = np.full(num_bins, n / num_bins)

        return Chi2Statistic.execute_statistic(self, observed, expected, self.lambda_)


class WatsonUniformGofStatistic(AbstractUniformGofStatistic):
    """
    Watson's U² test statistic for Uniform distribution.
    """

    def __init__(self, a=0, b=1):
        AbstractUniformGofStatistic.__init__(self, a, b)

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "WATSON".
        """
        return "WATSON"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "WATSON_UNIFORM_{parent_code}".
        """
        short_code = WatsonUniformGofStatistic.short_code()
        return f"{short_code}_{AbstractUniformGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute Watson's U² test statistic.

        :param rvs: array of sample data from Uniform distribution (values in [a, b]).
        :return: Watson's U² test statistic value.
        """
        rvs = self._validate_input(rvs)

        n = len(rvs)
        rvs_sorted = np.sort(rvs)
        rvs_standardized = (rvs_sorted - self.a) / (self.b - self.a)

        i = np.arange(1, n + 1)
        fn = i / n
        mean_f = np.mean(rvs_standardized)
        u2 = np.sum((rvs_standardized - fn + 0.5 / n - mean_f) ** 2) / n + 1 / (12 * n**2)

        return u2


class KuiperUniformGofStatistic(AbstractUniformGofStatistic):
    """
    Kuiper test statistic for Uniform distribution.
    """

    def __init__(self, a=0, b=1):
        AbstractUniformGofStatistic.__init__(self, a, b)

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "KUIPER".
        """
        return "KUIPER"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "KUIPER_UNIFORM_{parent_code}".
        """
        short_code = KuiperUniformGofStatistic.short_code()
        return f"{short_code}_{AbstractUniformGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Kuiper test statistic.

        :param rvs: array of sample data from Uniform distribution (values in [a, b]).
        :return: Kuiper test statistic value (D+ + D-).
        """
        rvs = self._validate_input(rvs)

        n = len(rvs)
        rvs_sorted = np.sort(rvs)
        if self.a != 0 or self.b != 1:
            rvs_standardized = (rvs_sorted - self.a) / (self.b - self.a)
        else:
            rvs_standardized = rvs_sorted.copy()

        i = np.arange(1, n + 1)
        fn = i / n

        d_plus = np.max(fn - rvs_standardized)
        d_minus = np.max(rvs_standardized - (i - 1) / n)
        v = float(d_plus) + float(d_minus)

        return v


class GreenwoodTestUniformGofStatistic(AbstractUniformGofStatistic):
    """
    Greenwood's test for Uniform distribution.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "GREENWOOD".
        """
        return "GREENWOOD"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "GREENWOOD_UNIFORM_{parent_code}".
        """
        short_code = GreenwoodTestUniformGofStatistic.short_code()
        return f"{short_code}_{AbstractUniformGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute Greenwood's test statistic.

        :param rvs: array of sample data from Uniform distribution (values in [a, b]).
        :return: Greenwood test statistic value.
        """
        rvs = self._validate_input(rvs)

        rvs_sorted = np.sort(rvs)
        rvs_std = (rvs_sorted - self.a) / (self.b - self.a)
        rvs_with_boundaries = np.concatenate([[0], rvs_std, [1]])
        spacings = np.diff(rvs_with_boundaries)

        g = np.sum(spacings**2)

        return g


class BickelRosenblattUniformGofStatistic(AbstractUniformGofStatistic):
    """
    Bickel-Rosenblatt test for Uniform distribution.
    """

    def __init__(self, a=0, b=1, bandwidth="auto"):
        AbstractUniformGofStatistic.__init__(self, a, b)
        self.bandwidth = bandwidth

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "BICKEL_ROSENBLATT".
        """
        return "BICKEL_ROSENBLATT"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "BICKEL_ROSENBLATT_UNIFORM_{parent_code}".
        """
        short_code = BickelRosenblattUniformGofStatistic.short_code()
        return f"{short_code}_{AbstractUniformGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute Bickel-Rosenblatt test statistic.

        :param rvs: array of sample data from Uniform distribution (values in [a, b]).
        :return: integrated squared difference statistic value.
        """
        rvs = self._validate_input(rvs)

        n = len(rvs)
        rvs_std = (rvs - self.a) / (self.b - self.a)

        if self.bandwidth == "auto":
            h = 1.06 * np.std(rvs_std) * (n ** (-1 / 5))
        else:
            h = self.bandwidth

        x_grid = np.linspace(0, 1, 1000)
        kde_vals = np.zeros_like(x_grid)

        for i, x in enumerate(x_grid):
            kde_vals[i] = np.mean(scipy_stats.norm.pdf((x - rvs_std) / h)) / h

        uniform_density = np.ones_like(x_grid)

        dx = x_grid[1] - x_grid[0]
        statistic = float(np.sum((kde_vals - uniform_density) ** 2)) * dx

        return statistic


class ZhangTestsUniformGofStatistic(AbstractUniformGofStatistic):
    """
    Zhang's tests (Z_A, Z_C, Z_K) for Uniform distribution.
    """

    def __init__(self, a=0, b=1, test_type="A"):
        AbstractUniformGofStatistic.__init__(self, a, b)
        self.test_type = test_type.upper()
        if self.test_type not in ["A", "C", "K"]:
            raise ValueError("test_type must be 'A', 'C', or 'K'")

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "ZHANG".
        """
        return "ZHANG"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "ZHANG_UNIFORM_{parent_code}".
        """
        short_code = ZhangTestsUniformGofStatistic.short_code()
        return f"{short_code}_{AbstractUniformGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute Zhang's test statistic.

        :param rvs: array of sample data from Uniform distribution (values in [a, b]).
        :return: Zhang test statistic value (depends on test_type).
        """
        rvs = self._validate_input(rvs)

        n = len(rvs)
        rvs_sorted = np.sort(rvs)

        rvs_std = (rvs_sorted - self.a) / (self.b - self.a)

        i = np.arange(1, n + 1)

        if self.test_type == "A":
            term1 = np.sum(np.log(rvs_std) / (n - i + 0.5))
            term2 = np.sum(np.log(1 - rvs_std) / (i - 0.5))
            statistic = -term1 - term2

        elif self.test_type == "C":
            term1 = np.sum((np.log(rvs_std) / (n - i + 0.5)) ** 2)
            term2 = np.sum((np.log(1 - rvs_std) / (i - 0.5)) ** 2)
            statistic = term1 + term2

        else:
            term1 = np.sum(np.log(rvs_std / (1 - rvs_std)) / (n - i + 0.5))
            term2 = np.sum(np.log((1 - rvs_std) / rvs_std) / (i - 0.5))
            statistic = max(np.abs(term1), np.abs(term2))

        return statistic


class SteinUniformGofStatistic(AbstractUniformGofStatistic):
    """
    Stein-type test statistic for Uniform distribution based on U-statistics.
    """

    def __init__(self, a=0, b=1):
        AbstractUniformGofStatistic.__init__(self, a, b)

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "STEIN_U".
        """
        return "STEIN_U"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "STEIN_U_UNIFORM_{parent_code}".
        """
        short_code = SteinUniformGofStatistic.short_code()
        return f"{short_code}_{AbstractUniformGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Stein-type test statistic.

        :param rvs: array of sample data from Uniform distribution (values in [a, b]).
        :return: Stein-type U-statistic value.
        """
        rvs = self._validate_input(rvs)
        if self.a != 0 or self.b != 1:
            rvs_std = (rvs - self.a) / (self.b - self.a)
        else:
            rvs_std = rvs.copy()

        statistic = self._compute_u_statistic(rvs_std)

        return statistic

    @staticmethod
    def _compute_u_statistic(rvs_std):
        """
        Compute U-statistic directly using double sum.

        :param rvs_std: array of standardized data in [0, 1].
        :return: U-statistic value.
        """
        n = len(rvs_std)

        def h1(x, y):
            return 0.5 * (2 * max(x, y) - 2 * x - 2 * y + x**2 + y**2)

        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += h1(rvs_std[i], rvs_std[j])

        statistic = 2 * total / (n * (n - 1)) if n > 1 else 0

        return statistic


class CensoredSteinUniformGofStatistic(AbstractUniformGofStatistic):
    """
    Stein-type test statistic for Uniform distribution with right censoring.
    """

    def __init__(self, a=0, b=1):
        AbstractUniformGofStatistic.__init__(self, a, b)

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "CENSORED_STEIN_U".
        """
        return "CENSORED_STEIN_U"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "CENSORED_STEIN_U_UNIFORM_{parent_code}".
        """
        short_code = CensoredSteinUniformGofStatistic.short_code()
        return f"{short_code}_{AbstractUniformGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, censoring_indices=None, **kwargs):
        """
        Execute the censored Stein-type test statistic.

        :param rvs: array of sample data (observed times).
        :param censoring_indices: binary array where 1 indicates censored observation,
            0 indicates uncensored (default is None, meaning no censoring).
        :return: censored Stein-type test statistic value.
        """
        rvs = self._validate_input(rvs)

        if self.a != 0 or self.b != 1:
            rvs_std = (rvs - self.a) / (self.b - self.a)
        else:
            rvs_std = rvs.copy()

        if censoring_indices is None or np.all(censoring_indices == 0):
            return SteinUniformGofStatistic(self.a, self.b).execute_statistic(rvs)

        censoring_indices = np.asarray(censoring_indices)

        km_estimator = self._kaplan_meier(rvs_std, censoring_indices)

        statistic = self._compute_weighted_u_statistic(rvs_std, censoring_indices, km_estimator)

        return statistic

    @staticmethod
    def _kaplan_meier(times, delta):
        sort_idx = np.argsort(times)
        times_sorted = times[sort_idx]
        delta_sorted = delta[sort_idx]

        n = len(times)
        at_risk = np.arange(n, 0, -1)
        km_survival = np.ones(n + 1)

        for i in range(n):
            if delta_sorted[i] == 1:
                km_survival[i + 1] = km_survival[i] * (1 - 1 / at_risk[i])
            else:
                km_survival[i + 1] = km_survival[i]

        def survival_func(t):
            idx = np.searchsorted(times_sorted, t, side="right")
            return km_survival[idx]

        return survival_func

    @staticmethod
    def _compute_weighted_u_statistic(rvs, delta, km_func):
        n = len(rvs)

        def h1(x, y):
            return 0.5 * (2 * max(x, y) - 2 * x - 2 * y + x**2 + y**2)

        weights = np.zeros(n)
        for i in range(n):
            if delta[i] == 0:
                weights[i] = 1.0 / max(km_func(rvs[i]), 1e-10)
            else:
                weights[i] = 0

        total = 0
        count = 0

        for i in range(n):
            if delta[i] == 0:
                for j in range(i + 1, n):
                    if delta[j] == 0:
                        weight_ij = weights[i] * weights[j]
                        total += weight_ij * h1(rvs[i], rvs[j])
                        count += 1

        if count > 0:
            statistic = 2 * total / count
        else:
            statistic = 0

        return statistic


class NeymanSmoothTestUniformGofStatistic(AbstractUniformGofStatistic):
    """
    Neyman's smooth test for Uniform distribution.
    """

    def __init__(self, a=0, b=1, k=4):
        AbstractUniformGofStatistic.__init__(self, a, b)
        self.k = k

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "NEYMAN".
        """
        return "NEYMAN"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "NEYMAN_UNIFORM_{parent_code}".
        """
        short_code = NeymanSmoothTestUniformGofStatistic.short_code()
        return f"{short_code}_{AbstractUniformGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute Neyman's smooth test statistic.

        :param rvs: array of sample data from Uniform distribution (values in [a, b]).
        :return: chi-square-like statistic value with k degrees of freedom.
        """
        rvs = self._validate_input(rvs)

        n = len(rvs)

        rvs_std = (rvs - self.a) / (self.b - self.a)

        from scipy.special import legendre

        statistic = 0

        def phi(j, x):
            if j == 1:
                return np.sqrt(12) * (x - 0.5)
            elif j == 2:
                return np.sqrt(5) * (6 * (x - 0.5) ** 2 - 0.5)
            elif j == 3:
                return np.sqrt(7) * (20 * (x - 0.5) ** 3 - 3 * (x - 0.5))
            else:
                pj = legendre(j)
                return pj(2 * x - 1) * np.sqrt(2 * j + 1)

        for j in range(1, self.k + 1):
            vj = float(np.sum(phi(j, rvs_std))) / np.sqrt(n)
            statistic += vj**2

        return statistic


class ShermanUniformGofStatistic(AbstractUniformGofStatistic):
    """
    Sherman's test for Uniform distribution.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "SHERMAN".
        """
        return "SHERMAN"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "SHERMAN_UNIFORM_{parent_code}".
        """
        short_code = ShermanUniformGofStatistic.short_code()
        return f"{short_code}_{AbstractUniformGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute Sherman's test statistic.

        :param rvs: array of sample data from Uniform distribution (values in [a, b]).
        :return: Sherman test statistic value.
        """
        rvs = self._validate_input(rvs)

        n = len(rvs)
        x_sorted = np.sort(rvs)

        x_with_boundaries = np.concatenate([[self.a], x_sorted, [self.b]])

        spacings = np.diff(x_with_boundaries)

        expected_spacing = (self.b - self.a) / (n + 1)
        s = 0.5 * np.sum(np.abs(spacings - expected_spacing))

        return s


class QuesenberryMillerUniformGofStatistic(AbstractUniformGofStatistic):
    """
    Quesenberry and Miller's Q-test for Uniform distribution.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "QUESENBERRY_MILLER".
        """
        return "QUESENBERRY_MILLER"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "QUESENBERRY_MILLER_UNIFORM_{parent_code}".
        """
        short_code = QuesenberryMillerUniformGofStatistic.short_code()
        return f"{short_code}_{AbstractUniformGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute Quesenberry-Miller Q-test statistic.

        :param rvs: array of sample data from Uniform distribution (values in [a, b]).
        :return: Q-test statistic value.
        """
        rvs = self._validate_input(rvs)

        x_sorted = np.sort(rvs)

        x_with_boundaries = np.concatenate([[self.a], x_sorted, [self.b]])

        spacings = np.diff(x_with_boundaries)

        sum_squares = np.sum(spacings**2)

        sum_consecutive_products = np.sum(spacings[:-1] * spacings[1:])

        q = float(sum_squares) + float(sum_consecutive_products)

        return q
