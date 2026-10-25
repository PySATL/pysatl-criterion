"""Uniform goodness-of-fit statistics.

These tests use a specified U(a, b) null and expose both bounds in the
hypothesis. Standardizing by the specified bounds does not make them unknown.
"""

from abc import ABC
from math import erf
from numbers import Integral

import numpy as np
from numba import jit
from typing_extensions import override

from pysatl_criterion.distribution import (
    UniformDistributionDescriptor as Distribution,
)
from pysatl_criterion.distribution import UniformDistributionDescriptor as Uniform
from pysatl_criterion.distribution.parameters import HypothesisSupport, ParameterValues
from pysatl_criterion.statistics.alternative import (
    Alternative,
    RightAlternative,
    TwoSidedAlternative,
)
from pysatl_criterion.statistics.goodness_of_fit.continuous import edf, frequency
from pysatl_criterion.statistics.goodness_of_fit.continuous._edf_factory import (
    EDFSpec,
    export_edf_statistics,
)
from pysatl_criterion.statistics.goodness_of_fit.continuous._frequency_factory import (
    FrequencySpec,
    export_frequency_statistics,
)
from pysatl_criterion.statistics.goodness_of_fit.goodness_of_fit_statistic import (
    AbstractGoodnessOfFitStatistic,
)


class AbstractUniformGofStatistic(AbstractGoodnessOfFitStatistic, ABC):
    """
    Abstract base class for Uniform distribution goodness-of-fit statistics.
    """

    @staticmethod
    def _validate_sample(rvs, *, min_size=1):
        sample = np.asarray(rvs, dtype=float)
        if sample.ndim != 1 or sample.size < min_size:
            raise ValueError(
                f"Sample must be one-dimensional with at least {min_size} observations"
            )
        if not np.all(np.isfinite(sample)):
            raise ValueError("Sample must contain only finite values")
        return sample

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
        if not {Distribution.LOWER, Distribution.UPPER}.issubset(self._parameters):
            return
        a, b = self.a, self.b
        if b <= a:
            raise ValueError("b must be greater than a")
        if not np.isfinite(b - a):
            raise ValueError("b - a must be finite")

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
        rvs_array = self._validate_sample(rvs, min_size=min_size)
        if require_bounds and np.any((rvs_array < self.a) | (rvs_array > self.b)):
            raise ValueError(
                f"Uniform distribution values must be in the interval [{self.a}, {self.b}]"
            )
        return rvs_array


class GreenwoodTestUniformGofStatistic(AbstractUniformGofStatistic):
    """Greenwood sum of squared uniform spacings.

    For u_i=(x_(i)-a)/(b-a), include u_0=0 and u_(n+1)=1 and return
    G=sum((u_i-u_(i-1))**2), without multiplying by n. Large values reject.
    Under the specified continuous uniform null, E[G]=2/(n+2).

    Parameters
    ----------
    parameters : ParameterValues
        Both uniform bounds a and b must be fixed.

    References
    ----------
    Does, R. J. M. M., Helmers, R., and Klaassen, C. A. J. (1988).
    Approximating the distribution of Greenwood's statistic.
    Statistica Neerlandica, 42(3), 153-161, Section 1, equations (1)-(2).
    https://ir.cwi.nl/pub/1694/1694D.pdf
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Uniform.DEFAULT, frozenset(Uniform.DEFAULT.parameters)),)

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
        rvs_std = (rvs - self.a) / (self.b - self.a)
        return self.do_execute_statistic(rvs_std)

    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(rvs_std) -> float:
        """Compute the statistic from validated unit-interval observations."""
        ordered = np.sort(rvs_std)
        total = 0.0
        previous = 0.0
        for x in ordered:
            total += (x - previous) ** 2
            previous = x
        return total + (1.0 - previous) ** 2


class BickelRosenblattUniformGofStatistic(AbstractUniformGofStatistic):
    """Bickel-Rosenblatt-type squared density error on the unit interval.

    Returns integral_0^1 (f_hat_h(u)-1)**2 du, where f_hat_h is the Gaussian
    KDE of u_i=(x_i-a)/(b-a). Gaussian product integrals are evaluated
    analytically. No n*sqrt(h) scaling, centering, or boundary correction is
    applied. This is an adaptation of the quadratic-density discrepancy,
    not the normalized limiting statistic in the reference.

    Parameters
    ----------
    parameters : ParameterValues
        Both uniform bounds a and b must be fixed.
    bandwidth : positive float or "auto", keyword-only
        Bandwidth in unit-interval coordinates. "auto" uses
        1.06*std(u, ddof=0)*n**(-1/5), a library bandwidth policy.
        Constant samples require an explicit positive bandwidth.

    Notes
    -----
    Calibrate this exact integral and bandwidth rule, including boundary bias.
    Large values reject; the function returns neither a p-value nor an
    asymptotically standardized statistic.

    References
    ----------
    Bickel, P. J., and Rosenblatt, M. (1973). On Some Global Measures of the
    Deviations of Density Function Estimates. Annals of Statistics,
    1(6), 1071-1095. https://doi.org/10.1214/aos/1176342558
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Uniform.DEFAULT, frozenset(Uniform.DEFAULT.parameters)),)

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

        return self.do_execute_statistic(rvs_std, float(h))

    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(rvs_std, h) -> float:
        """Integrate Gaussian density products using the normal CDF's erf form."""
        n = len(rvs_std)
        square_integral = 0.0
        density_integral = 0.0
        for i in range(n):
            x = rvs_std[i]
            # Off-diagonal products occur twice in the symmetric double sum.
            square_integral += 0.5 * (erf((1.0 - x) / h) + erf(x / h))
            for j in range(i + 1, n):
                y = rvs_std[j]
                midpoint = (x + y) / 2.0
                mass = 0.5 * (erf((1.0 - midpoint) / h) + erf(midpoint / h))
                square_integral += 2.0 * np.exp(-0.25 * ((x - y) / h) ** 2) * mass
            density_integral += 0.5 * (
                erf((1.0 - x) / (np.sqrt(2.0) * h)) + erf(x / (np.sqrt(2.0) * h))
            )
        square_integral /= 2 * np.sqrt(np.pi) * h * n**2
        return max(0.0, square_integral - 2 * density_integral / n + 1)


class SteinUniformGofStatistic(AbstractUniformGofStatistic):
    """Signed Stein U-statistic for a specified uniform distribution, n >= 2.

    On standardized observations u_i=(x_i-a)/(b-a), return the average over
    unordered pairs of h(x,y)=(x*x+y*y)/2-min(x,y). This equals the raw
    Delta-hat in Section 2 of the preprint. Both tails are significant;
    sqrt(45*n)*abs(Delta-hat), used for asymptotic testing there, is not
    applied here. The null mean is zero.

    Parameters
    ----------
    parameters : ParameterValues
        Both uniform bounds a and b must be fixed.

    References
    ----------
    Sreedevi, E. P., and Kattumannil, S. K. (2023). Goodness of fit test for
    uniform distribution with censored observation. Journal of the Korean
    Statistical Society, 52, 382-394.
    https://doi.org/10.1007/s42952-023-00205-8
    Author preprint (2021), Section 2:
    https://arxiv.org/html/2106.06368v1#S2
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Uniform.DEFAULT, frozenset(Uniform.DEFAULT.parameters)),)

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

    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(rvs_std) -> float:
        """Compute the pairwise U-statistic from a validated standardized sample."""
        n = len(rvs_std)
        if n < 2:
            raise ValueError("At least two observations are required")
        total = 0.0
        for i in range(n):
            x = rvs_std[i]
            for j in range(i + 1, n):
                y = rvs_std[j]
                total += 0.5 * (2.0 * max(y, x) - 2.0 * x - 2.0 * y + x * x + y * y)
        return 2.0 * total / (n * (n - 1))


class CensoredSteinUniformGofStatistic(AbstractUniformGofStatistic):
    """Signed IPCW Stein U-statistic under independent right censoring.

    Return 2/(n*(n-1))*sum_{i<j} w_i*w_j*h(u_i,u_j), where
    h(x,y)=(x*x+y*y)/2-min(x,y), u_i=(y_i-a)/(b-a), and
    w_i=(1-censoring_indices[i])/K_c(u_i-). The denominator uses all n
    observations, including censored ones; n must be at least two.

    Parameters
    ----------
    parameters : ParameterValues
        Both uniform bounds a and b must be fixed.

    Notes
    -----
    execute_statistic accepts observed times and censoring_indices with
    1=censored and 0=observed, the reverse of the paper's event indicator.
    K_c is estimated by reverse Kaplan-Meier; observed events precede
    censorings at ties. This tie convention extends the continuous model.
    With fewer than two observed events the sum is zero, which alone is not
    evidence of fit. Both tails are significant. The variance estimator and
    studentization in the paper are not included. Calibration must reproduce
    the censoring mechanism; complete-data Monte Carlo is insufficient.

    References
    ----------
    Sreedevi, E. P., and Kattumannil, S. K. (2023). Goodness of fit test for
    uniform distribution with censored observation. Journal of the Korean
    Statistical Society, 52, 382-394.
    https://doi.org/10.1007/s42952-023-00205-8
    Author preprint (2021), Section 3:
    https://arxiv.org/html/2106.06368v1#S3
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Uniform.DEFAULT, frozenset(Uniform.DEFAULT.parameters)),)

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

        return CensoredSteinUniformGofStatistic.do_execute_statistic(rvs, weights)

    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(rvs, weights) -> float:
        """Compute the censored U-statistic from validated IPCW weights."""
        n = len(rvs)
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
    """Neyman smooth statistic using orthonormal shifted Legendre polynomials.

    Return sum_{j=1}^k (sum_i phi_j(u_i))**2/n, with
    phi_j(u)=sqrt(2*j+1)*P_j(2*u-1) and u_i=(x_i-a)/(b-a).
    The constant polynomial is excluded. For a fixed k under the specified
    uniform null the limiting distribution is chi-square(k); this is not an
    exact finite-sample law. Large values reject.

    Parameters
    ----------
    parameters : ParameterValues
        Both uniform bounds a and b must be fixed.
    k : positive int, keyword-only, default 4
        Number of polynomial components, chosen independently of the sample.

    References
    ----------
    Neyman, J. (1937). "Smooth test" for goodness of fit.
    Skandinavisk Aktuarietidskrift, 1937(3-4), 149-199.
    https://doi.org/10.1080/03461238.1937.10404821
    Rayner, G. D., and Rayner, J. C. W. (2001). Power of the Neyman Smooth
    Tests for the Uniform Distribution. Journal of Applied Mathematics and
    Decision Sciences, 5(3), 181-191.
    https://doi.org/10.1155/S117391260100013X
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Uniform.DEFAULT, frozenset(Uniform.DEFAULT.parameters)),)

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
        rvs_std = (rvs - self.a) / (self.b - self.a)
        return self.do_execute_statistic(rvs_std, self.k)

    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(rvs_std, k) -> float:
        """Compute the statistic from validated unit-interval observations."""
        n = len(rvs_std)
        z = 2.0 * rvs_std - 1.0
        previous = np.ones(n)
        current = z.copy()
        statistic = 0.0
        for j in range(1, k + 1):
            statistic += (2 * j + 1) * np.sum(current) ** 2 / n
            following = ((2 * j + 1) * z * current - j * previous) / (j + 1)
            previous = current
            current = following
        return statistic


class ShermanUniformGofStatistic(AbstractUniformGofStatistic):
    """Sherman absolute-deviation statistic for uniform spacings.

    Include both boundary spacings after standardizing to [0,1]. Return
    S=0.5*sum_{i=1}^{n+1} abs(d_i-1/(n+1)), with no further scaling.
    Large values reject the fully specified uniform null.

    Parameters
    ----------
    parameters : ParameterValues
        Both uniform bounds a and b must be fixed.

    References
    ----------
    Sherman, B. (1950). A Random Variable Related to the Spacing of Sample
    Values. Annals of Mathematical Statistics, 21(3), 339-361.
    https://doi.org/10.1214/aoms/1177729794
    Formula also reproduced in Kattumannil and Sreedevi (2021), Section 4.1:
    https://arxiv.org/html/2106.06368v1#S4.SS1
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Uniform.DEFAULT, frozenset(Uniform.DEFAULT.parameters)),)

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
        rvs_std = (rvs - self.a) / (self.b - self.a)
        return self.do_execute_statistic(rvs_std)

    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(rvs_std) -> float:
        """Compute the statistic from validated unit-interval observations."""
        ordered = np.sort(rvs_std)
        expected = 1.0 / (len(ordered) + 1)
        total = 0.0
        previous = 0.0
        for x in ordered:
            total += abs(x - previous - expected)
            previous = x
        return 0.5 * (total + abs(1.0 - previous - expected))


class QuesenberryMillerUniformGofStatistic(AbstractUniformGofStatistic):
    """Quesenberry-Miller Q statistic for uniform spacings.

    With u_0=0, u_(n+1)=1 and d_i=u_i-u_(i-1), return
    Q=sum_{i=1}^{n+1} d_i**2 + sum_{i=1}^n d_i*d_(i+1).
    The two boundary spacings are included, without a cyclic end-to-end
    product or further scaling. Large values reject.

    Parameters
    ----------
    parameters : ParameterValues
        Both uniform bounds a and b must be fixed.

    References
    ----------
    Quesenberry, C. P., and Miller, F. L., Jr. (1977). Power studies of some
    tests for uniformity. Journal of Statistical Computation and Simulation,
    5(3), 169-191. https://doi.org/10.1080/00949657708810150
    Formula also reproduced in Kattumannil and Sreedevi (2021), Section 4.1:
    https://arxiv.org/html/2106.06368v1#S4.SS1
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Uniform.DEFAULT, frozenset(Uniform.DEFAULT.parameters)),)

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
        rvs_std = (rvs - self.a) / (self.b - self.a)
        return self.do_execute_statistic(rvs_std)

    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(rvs_std) -> float:
        """Compute the statistic from validated unit-interval observations."""
        ordered = np.sort(rvs_std)
        total = 0.0
        previous = 0.0
        previous_spacing = 0.0
        for x in ordered:
            spacing = x - previous
            total += spacing * spacing + previous_spacing * spacing
            previous = x
            previous_spacing = spacing
        spacing = 1.0 - previous
        return total + spacing * spacing + previous_spacing * spacing


_EDF_STATISTICS = {
    "KolmogorovSmirnovDPlusUniformGofStatistic": EDFSpec(
        edf.AbstractKSDPlusStatistic,
        supported_hypotheses=(
            HypothesisSupport(Uniform.DEFAULT, frozenset(Uniform.DEFAULT.parameters)),
        ),
    ),
    "KolmogorovSmirnovDMinusUniformGofStatistic": EDFSpec(
        edf.AbstractKSDMinusStatistic,
        supported_hypotheses=(
            HypothesisSupport(Uniform.DEFAULT, frozenset(Uniform.DEFAULT.parameters)),
        ),
    ),
    "KolmogorovSmirnovUniformGofStatistic": EDFSpec(
        edf.AbstractKSStatistic,
        supported_hypotheses=(
            HypothesisSupport(Uniform.DEFAULT, frozenset(Uniform.DEFAULT.parameters)),
        ),
    ),
    "AndersonDarlingUniformGofStatistic": EDFSpec(
        edf.AbstractAndersonDarlingStatistic,
        supported_hypotheses=(
            HypothesisSupport(Uniform.DEFAULT, frozenset(Uniform.DEFAULT.parameters)),
        ),
    ),
    "CrammerVonMisesUniformGofStatistic": EDFSpec(
        edf.AbstractCramerVonMisesStatistic,
        supported_hypotheses=(
            HypothesisSupport(Uniform.DEFAULT, frozenset(Uniform.DEFAULT.parameters)),
        ),
    ),
    "LiaoShimokawaUniformGofStatistic": EDFSpec(
        edf.AbstractLiaoShimokawaStatistic,
        supported_hypotheses=(
            HypothesisSupport(Uniform.DEFAULT, frozenset(Uniform.DEFAULT.parameters)),
        ),
    ),
    "WatsonUniformGofStatistic": EDFSpec(
        edf.AbstractWatsonStatistic,
        supported_hypotheses=(
            HypothesisSupport(Uniform.DEFAULT, frozenset(Uniform.DEFAULT.parameters)),
        ),
    ),
    "KuiperUniformGofStatistic": EDFSpec(
        edf.AbstractKuiperStatistic,
        supported_hypotheses=(
            HypothesisSupport(Uniform.DEFAULT, frozenset(Uniform.DEFAULT.parameters)),
        ),
    ),
    "KuiperScaledUniformGofStatistic": EDFSpec(
        edf.AbstractKuiperScaledStatistic,
        supported_hypotheses=(
            HypothesisSupport(Uniform.DEFAULT, frozenset(Uniform.DEFAULT.parameters)),
        ),
    ),
    "KuiperCorrectedUniformGofStatistic": EDFSpec(
        edf.AbstractKuiperCorrectedStatistic,
        supported_hypotheses=(
            HypothesisSupport(Uniform.DEFAULT, frozenset(Uniform.DEFAULT.parameters)),
        ),
    ),
    "ZhangKUniformGofStatistic": EDFSpec(
        edf.AbstractZhangKStatistic,
        supported_hypotheses=(
            HypothesisSupport(Uniform.DEFAULT, frozenset(Uniform.DEFAULT.parameters)),
        ),
    ),
    "ZhangAUniformGofStatistic": EDFSpec(
        edf.AbstractZhangAStatistic,
        supported_hypotheses=(
            HypothesisSupport(Uniform.DEFAULT, frozenset(Uniform.DEFAULT.parameters)),
        ),
    ),
    "ZhangCUniformGofStatistic": EDFSpec(
        edf.AbstractZhangCStatistic,
        supported_hypotheses=(
            HypothesisSupport(Uniform.DEFAULT, frozenset(Uniform.DEFAULT.parameters)),
        ),
    ),
}

_FREQUENCY_STATISTICS = {
    "Chi2PearsonUniformGofStatistic": FrequencySpec(
        frequency.AbstractPearsonChi2Statistic,
        supported_hypotheses=(
            HypothesisSupport(Uniform.DEFAULT, frozenset(Uniform.DEFAULT.parameters)),
        ),
    ),
    "FreemanTukeyUniformGofStatistic": FrequencySpec(
        frequency.AbstractFreemanTukeyStatistic,
        supported_hypotheses=(
            HypothesisSupport(Uniform.DEFAULT, frozenset(Uniform.DEFAULT.parameters)),
        ),
    ),
    "ModifiedLogLikelihoodUniformGofStatistic": FrequencySpec(
        frequency.AbstractModifiedLogLikelihoodStatistic,
        supported_hypotheses=(
            HypothesisSupport(Uniform.DEFAULT, frozenset(Uniform.DEFAULT.parameters)),
        ),
    ),
    "NeymanChi2UniformGofStatistic": FrequencySpec(
        frequency.AbstractNeymanChi2Statistic,
        supported_hypotheses=(
            HypothesisSupport(Uniform.DEFAULT, frozenset(Uniform.DEFAULT.parameters)),
        ),
    ),
    "CressieReadUniformGofStatistic": FrequencySpec(
        frequency.AbstractCressieReadStatistic,
        supported_hypotheses=(
            HypothesisSupport(Uniform.DEFAULT, frozenset(Uniform.DEFAULT.parameters)),
        ),
    ),
}

__all__ = [
    "AbstractUniformGofStatistic",
    "BickelRosenblattUniformGofStatistic",
    "CensoredSteinUniformGofStatistic",
    "GreenwoodTestUniformGofStatistic",
    "NeymanSmoothTestUniformGofStatistic",
    "QuesenberryMillerUniformGofStatistic",
    "ShermanUniformGofStatistic",
    "SteinUniformGofStatistic",
]
__all__.extend(export_edf_statistics(globals(), AbstractUniformGofStatistic, _EDF_STATISTICS))

__all__.extend(
    export_frequency_statistics(globals(), AbstractUniformGofStatistic, _FREQUENCY_STATISTICS)
)
