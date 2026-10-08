"""Uniform goodness-of-fit statistics.

These tests use a specified U(a, b) null and expose both bounds in the
hypothesis. Standardizing by the specified bounds does not make them unknown.
"""

from abc import ABC
from numbers import Integral

import numpy as np
import scipy.stats as scipy_stats
from numba import njit
from scipy.special import eval_legendre, ndtr
from typing_extensions import override

from pysatl_criterion.distribution.distributions import (
    UniformDistributionDescriptor as Distribution,
)
from pysatl_criterion.distribution.distributions import UniformDistributionDescriptor as Uniform
from pysatl_criterion.distribution.parameters import HypothesisSupport, ParameterValues
from pysatl_criterion.statistics import AbstractGoodnessOfFitStatistic
from pysatl_criterion.statistics.alternative import (
    Alternative,
    AlternativeType,
    RightAlternative,
    TwoSidedAlternative,
)
from pysatl_criterion.statistics.goodness_of_fit.common import (
    ADStatistic,
    Chi2Statistic,
    CrammerVonMisesStatistic,
    KSStatistic,
)


def _validate_sample(rvs, *, min_size=1):
    sample = np.asarray(rvs, dtype=float)
    if sample.ndim != 1 or sample.size < min_size:
        raise ValueError(f"Sample must be one-dimensional with at least {min_size} observations")
    if not np.all(np.isfinite(sample)):
        raise ValueError("Sample must contain only finite values")
    return sample


class AbstractUniformGofStatistic(AbstractGoodnessOfFitStatistic, ABC):
    """
    Abstract base class for Uniform distribution goodness-of-fit statistics.
    """

    @property
    def a(self) -> float:
        """Read a by its stable parameter identity."""
        return self._parameters[Distribution.LOWER]

    @property
    def b(self) -> float:
        """Read b by its stable parameter identity."""
        return self._parameters[Distribution.UPPER]

    def __init__(self, parameters: ParameterValues):
        AbstractGoodnessOfFitStatistic.__init__(self, parameters)
        a, b = self.a, self.b
        if b <= a:
            raise ValueError("b must be greater than a")
        if not np.isfinite(b - a):
            raise ValueError("b - a must be finite")

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Uniform.DEFAULT, frozenset(Uniform.DEFAULT.parameters)),)

    @staticmethod
    @override
    def distribution() -> type[Uniform]:
        """Return the distribution descriptor class."""
        return Uniform

    @classmethod
    @override
    def code(cls) -> str:
        """Return the family identifier or the concrete statistic's full identifier."""
        family_code = f"UNIFORM_{AbstractGoodnessOfFitStatistic.code()}"
        if "short_code" in cls.__abstractmethods__:
            return family_code
        return f"{cls.short_code()}_{family_code}"

    def _validate_input(self, rvs, *, require_bounds=True, min_size=1):
        """Validate a finite sample; spacing and moment tests require support membership."""
        rvs_array = _validate_sample(rvs, min_size=min_size)
        if require_bounds and np.any((rvs_array < self.a) | (rvs_array > self.b)):
            raise ValueError(
                f"Uniform distribution values must be in the interval [{self.a}, {self.b}]"
            )
        return rvs_array


class KolmogorovSmirnovUniformGofStatistic(AbstractUniformGofStatistic, KSStatistic):
    """Parameters
    ----------
    parameters : ParameterValues
        Values with a, b fixed; omitted parameters are unknown.

    Kolmogorov-Smirnov test statistic for Uniform distribution.
    """

    def __init__(
        self,
        parameters: ParameterValues,
        *,
        alternative_type: AlternativeType = AlternativeType.TWO_TAILED,
        mode="auto",
    ):
        AbstractUniformGofStatistic.__init__(self, parameters)
        KSStatistic.__init__(self, alternative_type, mode)

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "KS".
        """
        return "KS"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Kolmogorov-Smirnov test statistic.

        :param rvs: nonempty one-dimensional sample of finite observations.
        :return: Kolmogorov-Smirnov test statistic value.
        """
        rvs = self._validate_input(rvs, require_bounds=False)

        rvs_sorted = np.sort(rvs)
        cdf_vals = scipy_stats.uniform.cdf(rvs_sorted, loc=self.a, scale=self.b - self.a)
        return KSStatistic.do_execute_statistic(self, rvs_sorted, cdf_vals)


class AndersonDarlingUniformGofStatistic(AbstractUniformGofStatistic, ADStatistic):
    """Parameters
    ----------
    parameters : ParameterValues
        Values with a, b fixed; omitted parameters are unknown.

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

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Anderson-Darling test statistic.

        :param rvs: nonempty one-dimensional sample of finite observations.
        :return: Anderson-Darling test statistic value.
        """
        rvs = self._validate_input(rvs, require_bounds=False)

        rvs_sorted = np.sort(rvs)
        logcdf = scipy_stats.uniform.logcdf(rvs_sorted, loc=self.a, scale=self.b - self.a)
        logsf = scipy_stats.uniform.logsf(rvs_sorted, loc=self.a, scale=self.b - self.a)
        return ADStatistic.do_execute_statistic(self, rvs=rvs, log_cdf=logcdf, log_sf=logsf)


class CrammerVonMisesUniformGofStatistic(AbstractUniformGofStatistic, CrammerVonMisesStatistic):
    """Parameters
    ----------
    parameters : ParameterValues
        Values with a, b fixed; omitted parameters are unknown.

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

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Cramér-von Mises test statistic.

        :param rvs: nonempty one-dimensional sample of finite observations.
        :return: Cramér-von Mises test statistic value.
        """
        rvs = self._validate_input(rvs, require_bounds=False)

        rvs_sorted = np.sort(rvs)
        cdf_vals = scipy_stats.uniform.cdf(rvs_sorted, loc=self.a, scale=self.b - self.a)
        return CrammerVonMisesStatistic.do_execute_statistic(self, rvs_sorted, cdf_vals)


class Chi2PearsonUniformGofStatistic(AbstractUniformGofStatistic, Chi2Statistic):
    """Parameters
    ----------
    parameters : ParameterValues
        Values with a, b fixed; omitted parameters are unknown.

    Pearson's Chi-squared test statistic for Uniform distribution.
    """

    def __init__(self, parameters: ParameterValues, *, lambda_=1, bins="sturges"):
        AbstractUniformGofStatistic.__init__(self, parameters)
        Chi2Statistic.__init__(self)
        if not np.isfinite(lambda_):
            raise ValueError("lambda_ must be finite")
        if isinstance(bins, str):
            if bins not in {"sturges", "sqrt", "auto"}:
                raise ValueError("bins must be 'sturges', 'sqrt', 'auto', or an integer >= 2")
        elif isinstance(bins, bool) or not isinstance(bins, Integral) or bins < 2:
            raise ValueError("bins must be 'sturges', 'sqrt', 'auto', or an integer >= 2")
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
                # Work in unit coordinates, with Sturges as the constant-sample fallback.
                h = 3.5 * np.std((rvs - self.a) / (self.b - self.a)) / (n ** (1 / 3))
                # No more than n bins: nearly constant samples otherwise allocate huge arrays.
                num_bins = (
                    int(np.ceil(1 / max(h, 1 / n))) if h > 0 else int(np.ceil(np.log2(n) + 1))
                )
        else:
            num_bins = int(self.bins)
        num_bins = max(2, num_bins)

        observed, _bin_edges = np.histogram(rvs, bins=num_bins, range=(self.a, self.b))
        expected = np.full(num_bins, n / num_bins)

        return Chi2Statistic.do_execute_statistic(self, observed, expected, self.lambda_)


class WatsonUniformGofStatistic(AbstractUniformGofStatistic):
    """Parameters
    ----------
    parameters : ParameterValues
        Values with a, b fixed; omitted parameters are unknown.

    Watson's U² test statistic for Uniform distribution.
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "WATSON".
        """
        return "WATSON"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute Watson's U² test statistic.

        :param rvs: nonempty one-dimensional sample of finite observations.
        :return: Watson's U² test statistic value.
        """
        rvs = self._validate_input(rvs, require_bounds=False)

        n = len(rvs)
        rvs_sorted = np.sort(rvs)
        rvs_standardized = scipy_stats.uniform.cdf(rvs_sorted, loc=self.a, scale=self.b - self.a)

        i = np.arange(1, n + 1)
        centered = rvs_standardized - (i - 0.5) / n - np.mean(rvs_standardized) + 0.5
        u2 = np.sum(centered**2) + 1 / (12 * n)

        return u2


class KuiperUniformGofStatistic(AbstractUniformGofStatistic):
    """Parameters
    ----------
    parameters : ParameterValues
        Values with a, b fixed; omitted parameters are unknown.

    Kuiper test statistic for Uniform distribution.
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "KUIPER".
        """
        return "KUIPER"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Kuiper test statistic.

        :param rvs: nonempty one-dimensional sample of finite observations.
        :return: Kuiper test statistic value (D+ + D-).
        """
        rvs = self._validate_input(rvs, require_bounds=False)

        n = len(rvs)
        rvs_sorted = np.sort(rvs)
        rvs_standardized = scipy_stats.uniform.cdf(rvs_sorted, loc=self.a, scale=self.b - self.a)

        i = np.arange(1, n + 1)
        fn = i / n

        d_plus = np.max(fn - rvs_standardized)
        d_minus = np.max(rvs_standardized - (i - 1) / n)
        v = float(d_plus) + float(d_minus)

        return v


class GreenwoodTestUniformGofStatistic(AbstractUniformGofStatistic):
    """Parameters
    ----------
    parameters : ParameterValues
        Values with a, b fixed; omitted parameters are unknown.

    Greenwood's test for Uniform distribution.
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "GREENWOOD".
        """
        return "GREENWOOD"

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
    """Parameters
    ----------
    parameters : ParameterValues
        Values with a, b fixed; omitted parameters are unknown.

    Bickel-Rosenblatt-type integrated squared density error on the unit interval.

    Returns integral_0^1 (f_hat_h(u) - 1)^2 du for a Gaussian KDE of
    (X-a)/(b-a), without asymptotic centering or n*sqrt(h) scaling. The bandwidth
    is in unit-interval coordinates. Gaussian product integrals are evaluated
    analytically, avoiding a grid that can miss narrow kernels. Calibration must
    use this same definition and bandwidth rule, including boundary bias.
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    def __init__(self, parameters: ParameterValues, *, bandwidth="auto"):
        AbstractUniformGofStatistic.__init__(self, parameters)
        if isinstance(bandwidth, str):
            if bandwidth != "auto":
                raise ValueError("bandwidth must be 'auto' or a positive finite number")
        elif not np.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError("bandwidth must be 'auto' or a positive finite number")
        self.bandwidth = bandwidth

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "BICKEL_ROSENBLATT".
        """
        return "BICKEL_ROSENBLATT"

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

        if not np.isfinite(h) or h <= 0:
            raise ValueError("Automatic bandwidth is zero; provide a positive bandwidth")

        # Product of two N(x_i,h^2) densities: a constant times N(midpoint,h^2/2).
        square_integral = 0.0
        for x in rvs_std:
            midpoint = (x + rvs_std) / 2
            mass = ndtr(np.sqrt(2) * (1 - midpoint) / h) - ndtr(-np.sqrt(2) * midpoint / h)
            square_integral += np.sum(np.exp(-0.25 * ((x - rvs_std) / h) ** 2) * mass)
        square_integral /= 2 * np.sqrt(np.pi) * h * n**2
        density_integral = np.mean(ndtr((1 - rvs_std) / h) - ndtr(-rvs_std / h))
        return float(max(0.0, square_integral - 2 * density_integral + 1))


class ZhangTestsUniformGofStatistic(AbstractUniformGofStatistic):
    """Parameters
    ----------
    parameters : ParameterValues
        Values with a, b fixed; omitted parameters are unknown.

    Zhang's tests (Z_A, Z_C, Z_K) for Uniform distribution.
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    def __init__(self, parameters: ParameterValues, *, test_type="A"):
        AbstractUniformGofStatistic.__init__(self, parameters)
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

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute Zhang's test statistic.

        :param rvs: nonempty one-dimensional sample of finite observations.
        :return: Zhang test statistic value (depends on test_type).
        """
        rvs = self._validate_input(rvs, require_bounds=False)

        n = len(rvs)
        rvs_sorted = np.sort(rvs)

        u = scipy_stats.uniform.cdf(rvs_sorted, loc=self.a, scale=self.b - self.a)
        if np.any((u == 0) | (u == 1)):
            return np.inf
        i = np.arange(1, n + 1)
        log_u = np.log(u)
        log_sf = np.log1p(-u)

        # Zhang (2002), DOI: 10.1111/1467-9868.00337.
        if self.test_type == "A":
            statistic = -np.sum(log_u / (n - i + 0.5) + log_sf / (i - 0.5))
        elif self.test_type == "C":
            log_odds = np.log(n - i + 0.25) - np.log(i - 0.75)
            statistic = np.sum((log_sf - log_u - log_odds) ** 2)
        else:
            p = (i - 0.5) / n
            statistic = np.max(
                (i - 0.5) * (np.log(p) - log_u) + (n - i + 0.5) * (np.log1p(-p) - log_sf)
            )
        return float(statistic)


@njit
def _stein_uniform_statistic(rvs_std):  # pragma: no cover
    n = len(rvs_std)

    if n <= 1:
        return 0.0

    total = 0.0

    for i in range(n):
        x = rvs_std[i]

        for j in range(i + 1, n):
            y = rvs_std[j]
            maximum = max(y, x)

            total += 0.5 * (2.0 * maximum - 2.0 * x - 2.0 * y + x * x + y * y)
    return 2.0 * total / (n * (n - 1))


class SteinUniformGofStatistic(AbstractUniformGofStatistic):
    """Parameters
    ----------
    parameters : ParameterValues
        Values with a, b fixed; omitted parameters are unknown.

    Signed Stein U-statistic for a specified uniform distribution, n >= 2.

    Both tails are significant. The symmetric kernel is
    (x*x + y*y)/2 - min(x,y) on standardized observations.
    See Sreedevi and Kattumannil (2023), DOI: 10.1007/s42952-023-00205-8.
    """

    @override
    def alternative(self) -> Alternative:
        return TwoSidedAlternative()

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "STEIN_U".
        """
        return "STEIN_U"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Stein-type test statistic.

        :param rvs: array of sample data from Uniform distribution (values in [a, b]).
        :return: Stein-type U-statistic value.
        """
        rvs = self._validate_input(rvs, min_size=2)
        if self.a != 0 or self.b != 1:
            rvs_std = (rvs - self.a) / (self.b - self.a)
        else:
            rvs_std = rvs

        statistic = self.do_execute_statistic(rvs_std)

        return statistic

    def do_execute_statistic(self, rvs_std):
        """
        Compute the U-statistic for an already standardized sample.

        :param rvs_std: array of standardized data in [0, 1].
        :return: U-statistic value.
        """
        rvs_array = _validate_sample(rvs_std, min_size=2)
        return float(_stein_uniform_statistic(rvs_array))


class CensoredSteinUniformGofStatistic(AbstractUniformGofStatistic):
    """Parameters
    ----------
    parameters : ParameterValues
        Values with a, b fixed; omitted parameters are unknown.

    Signed IPCW Stein U-statistic under independent right censoring, n >= 2.

    The indicator convention is 1=censored, 0=observed. Weights use the left
    limit of the reverse Kaplan-Meier survival estimator. The denominator uses
    all n observations, including censored ones. With fewer than two observed
    events the sum is zero, which alone is not evidence of fit. Calibration for
    censored data must reproduce the censoring mechanism; the generic complete-
    data Monte Carlo resolver does not do that.

    See Sreedevi and Kattumannil (2023), DOI: 10.1007/s42952-023-00205-8.
    """

    @override
    def alternative(self) -> Alternative:
        return TwoSidedAlternative()

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "CENSORED_STEIN_U".
        """
        return "CENSORED_STEIN_U"

    @override
    def execute_statistic(self, rvs, censoring_indices=None, **kwargs):
        """
        Execute the censored Stein-type test statistic.

        :param rvs: array of sample data (observed times).
        :param censoring_indices: binary array where 1 indicates censored observation,
            0 indicates uncensored (default is None, meaning no censoring).
        :return: censored Stein-type test statistic value.
        """
        rvs = self._validate_input(rvs, min_size=2)

        if self.a != 0 or self.b != 1:
            rvs_std = (rvs - self.a) / (self.b - self.a)
        else:
            rvs_std = rvs.copy()

        if censoring_indices is None:
            return SteinUniformGofStatistic(self._parameters).execute_statistic(rvs)
        censoring_indices = np.asarray(censoring_indices)
        if censoring_indices.shape != rvs.shape or not np.all(np.isin(censoring_indices, [0, 1])):
            raise ValueError("censoring_indices must match the sample and contain only 0 or 1")
        if not np.any(censoring_indices):
            return SteinUniformGofStatistic(self._parameters).execute_statistic(rvs)

        km_estimator = self._kaplan_meier(rvs_std, censoring_indices)

        statistic = self._compute_weighted_u_statistic(rvs_std, censoring_indices, km_estimator)

        return statistic

    @staticmethod
    def _kaplan_meier(times, delta):
        """Return K_c(t-); observed events precede censorings at tied times."""
        times = np.asarray(times)
        delta = np.asarray(delta)
        unique_times, inverse, counts = np.unique(times, return_inverse=True, return_counts=True)
        censored_counts = np.bincount(inverse, weights=delta)
        at_risk = len(times) - np.concatenate(([0], np.cumsum(counts[:-1])))
        censoring_risk = at_risk - (counts - censored_counts)
        hazard = np.divide(
            censored_counts,
            censoring_risk,
            out=np.zeros_like(censored_counts),
            where=censoring_risk > 0,
        )
        survival = np.concatenate(([1.0], np.cumprod(1 - hazard)))

        def survival_func(t):
            return survival[np.searchsorted(unique_times, t, side="left")]

        return survival_func

    @staticmethod
    def _compute_weighted_u_statistic(rvs, delta, km_func):
        n = len(rvs)
        weights = np.zeros(n)
        for i in range(n):
            if delta[i] == 0:
                survival = km_func(rvs[i])
                if survival <= 0:
                    raise ValueError("Censoring survival must be positive at observed events")
                weights[i] = 1 / survival

        total = 0.0
        for i in range(n):
            if weights[i] == 0:
                continue
            for j in range(i + 1, n):
                if weights[j] != 0:
                    kernel = (rvs[i] ** 2 + rvs[j] ** 2) / 2 - min(rvs[i], rvs[j])
                    total += weights[i] * weights[j] * kernel
        return 2 * total / (n * (n - 1))


class NeymanSmoothTestUniformGofStatistic(AbstractUniformGofStatistic):
    """Parameters
    ----------
    parameters : ParameterValues
        Values with a, b fixed; omitted parameters are unknown.

    Neyman's smooth test for Uniform distribution.
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    def __init__(self, parameters: ParameterValues, *, k=4):
        AbstractUniformGofStatistic.__init__(self, parameters)
        if isinstance(k, bool) or not isinstance(k, Integral) or k < 1:
            raise ValueError("k must be a positive integer")
        self.k = k

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "NEYMAN".
        """
        return "NEYMAN"

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

        statistic = 0.0
        for j in range(1, self.k + 1):
            # Orthonormal shifted Legendre polynomials on [0,1].
            phi = np.sqrt(2 * j + 1) * eval_legendre(j, 2 * rvs_std - 1)
            statistic += np.sum(phi) ** 2 / n
        return float(statistic)


class ShermanUniformGofStatistic(AbstractUniformGofStatistic):
    """Parameters
    ----------
    parameters : ParameterValues
        Values with a, b fixed; omitted parameters are unknown.

    Sherman's test for Uniform distribution.
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "SHERMAN".
        """
        return "SHERMAN"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute Sherman's test statistic.

        :param rvs: array of sample data from Uniform distribution (values in [a, b]).
        :return: Sherman test statistic value.
        """
        rvs = self._validate_input(rvs)

        n = len(rvs)
        x_sorted = np.sort((rvs - self.a) / (self.b - self.a))

        x_with_boundaries = np.concatenate([[0.0], x_sorted, [1.0]])

        spacings = np.diff(x_with_boundaries)

        expected_spacing = 1 / (n + 1)
        s = 0.5 * np.sum(np.abs(spacings - expected_spacing))

        return s


class QuesenberryMillerUniformGofStatistic(AbstractUniformGofStatistic):
    """Parameters
    ----------
    parameters : ParameterValues
        Values with a, b fixed; omitted parameters are unknown.

    Quesenberry and Miller's Q-test for Uniform distribution.
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "QUESENBERRY_MILLER".
        """
        return "QUESENBERRY_MILLER"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute Quesenberry-Miller Q-test statistic.

        :param rvs: array of sample data from Uniform distribution (values in [a, b]).
        :return: Q-test statistic value.
        """
        rvs = self._validate_input(rvs)

        x_sorted = np.sort((rvs - self.a) / (self.b - self.a))

        x_with_boundaries = np.concatenate([[0.0], x_sorted, [1.0]])

        spacings = np.diff(x_with_boundaries)

        sum_squares = np.sum(spacings**2)

        sum_consecutive_products = np.sum(spacings[:-1] * spacings[1:])

        q = float(sum_squares) + float(sum_consecutive_products)

        return q
