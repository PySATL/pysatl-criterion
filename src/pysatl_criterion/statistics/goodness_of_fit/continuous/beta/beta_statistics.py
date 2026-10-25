"""Goodness-of-fit statistics on the fixed Beta support [0, 1].

Samples must be real-valued; masked observations are rejected, not discarded.

Numerical reductions use cached Numba kernels without fast-math. SciPy handles
special functions and fitting. Generated EDF and frequency criteria and Neyman
use specified shape parameters. The four fitted-Beta criteria fit both shapes
inside execute_statistic.
"""

from abc import ABC
from numbers import Integral, Real

import numpy as np
import scipy.stats as scipy_stats
from numba import jit
from scipy.integrate import quad
from scipy.special import betainc, betaln, hyp2f1
from typing_extensions import override

from pysatl_criterion.distribution import BetaDistributionDescriptor as Beta
from pysatl_criterion.distribution import UniformDistributionDescriptor as Uniform
from pysatl_criterion.distribution.parameters import HypothesisSupport, ParameterValues
from pysatl_criterion.statistics.alternative import Alternative, RightAlternative
from pysatl_criterion.statistics.goodness_of_fit.continuous import edf, frequency
from pysatl_criterion.statistics.goodness_of_fit.continuous._edf_factory import (
    EDFSpec,
    export_edf_statistics,
)
from pysatl_criterion.statistics.goodness_of_fit.continuous._frequency_factory import (
    FrequencySpec,
    export_frequency_statistics,
)
from pysatl_criterion.statistics.goodness_of_fit.continuous.uniform import (
    NeymanSmoothTestUniformGofStatistic,
)
from pysatl_criterion.statistics.goodness_of_fit.goodness_of_fit_statistic import (
    AbstractGoodnessOfFitStatistic,
)


class AbstractBetaGofStatistic(AbstractGoodnessOfFitStatistic, ABC):
    """
    Abstract base class for Beta distribution goodness-of-fit statistics.

    The Beta distribution is a continuous probability distribution defined on the interval [0, 1]
    parameterized by two positive shape parameters, denoted by α (alpha) and β (beta).
    Subclasses declare which parameters are fixed by their hypothesis.
    """

    @staticmethod
    def _beta_logtails(x, a, b):
        """Beta log tails, retaining interior probabilities below float64 range.

        On underflow use DLMF 8.17.8 in log form. Its hypergeometric series
        has positive terms, avoiding the cancellation in DLMF 8.17.7.
        Exact endpoints retain their mathematically infinite log tails.
        """
        logcdf = scipy_stats.beta.logcdf(x, a, b)
        logsf = scipy_stats.beta.logsf(x, a, b)
        for logs, points, first, second in ((logcdf, x, a, b), (logsf, 1 - x, b, a)):
            missing = np.isneginf(logs) & (points > 0) & (points < 1)
            t = points[missing]
            if t.size:
                factor = hyp2f1(first + second, 1, first + 1, t)
                if np.any(~np.isfinite(factor)) or np.any(factor <= 0):
                    raise ValueError("Beta log tail could not be evaluated with finite precision")
                logs[missing] = (
                    first * np.log(t)
                    + second * np.log1p(-t)
                    - np.log(first)
                    - betaln(first, second)
                    + np.log(factor)
                )
                if np.any(~np.isfinite(logs[missing])) or np.any(logs[missing] > 0):
                    raise ValueError("Beta log tail could not be evaluated with finite precision")
        if np.any(np.isnan(logcdf)) or np.any(np.isnan(logsf)):
            raise ValueError("Beta log tails could not be evaluated with finite precision")
        return logcdf, logsf

    @staticmethod
    @jit(nopython=True, cache=True)
    def _ad_kernel(logcdf, logsf):
        n = len(logcdf)
        total = 0.0
        for i in range(n):
            total += (2 * i + 1.0) / n * (logcdf[i] + logsf[n - 1 - i])
        return -n - total

    def _validate_storage_calibration(self):
        """Reject shape-dependent composite laws and settings missing from storage keys."""
        if not self.hypothesis().parameters():
            raise ValueError(
                "Composite Beta calibration depends on fitted shapes; "
                "use parametric_bootstrap_beta with the observed sample"
            )
        if getattr(self, "k", 4) != 4:
            raise ValueError(
                "Stored Beta calibration does not identify these settings; "
                "use MonteCarloLimitDistributionResolver"
            )

    @staticmethod
    def _validate_rvs(rvs, min_size=1):
        if np.ma.isMaskedArray(rvs) and np.any(np.ma.getmaskarray(rvs)):
            raise ValueError("Sample must not contain masked observations")
        rvs = np.asarray(rvs)
        if np.iscomplexobj(rvs):
            raise ValueError("Sample values must be real")
        rvs = np.asarray(rvs, dtype=float)
        if rvs.ndim != 1 or rvs.size < min_size:
            raise ValueError(f"Sample must be one-dimensional with at least {min_size} values")
        if not np.all(np.isfinite(rvs)):
            raise ValueError("Sample values must be finite")
        # All values in [0, 1]
        if np.any((rvs < 0) | (rvs > 1)):
            raise ValueError("Beta distribution values must be in the interval [0, 1]")
        return rvs

    @classmethod
    def _fit(cls, rvs):
        sample = cls._validate_rvs(rvs, min_size=2)
        if np.any((sample <= 0) | (sample >= 1)):
            raise ValueError("Beta maximum likelihood requires observations strictly in (0, 1)")
        if np.ptp(sample) == 0:
            raise ValueError("Beta maximum likelihood requires a nonconstant sample")
        alpha, beta, _, _ = scipy_stats.beta.fit(sample, floc=0, fscale=1)
        if not np.isfinite(alpha) or not np.isfinite(beta) or alpha <= 0 or beta <= 0:
            raise ValueError("Beta maximum likelihood did not produce finite positive shapes")
        return sample, float(alpha), float(beta)

    @staticmethod
    @override
    def distribution() -> type[Beta]:
        """Return the distribution descriptor class."""
        return Beta

    @classmethod
    @override
    def code(cls) -> str:
        """Return the identifier using the concrete statistic's short code."""
        family_code = f"BETA_{AbstractGoodnessOfFitStatistic.code()}"
        if "short_code" in cls.__abstractmethods__:
            return family_code
        return f"{cls.short_code()}_{family_code}"


class AbstractSpecifiedBetaGofStatistic(AbstractBetaGofStatistic, ABC):
    """Base for Beta statistics with both shape parameters fixed by the hypothesis."""

    @property
    def alpha(self) -> float:
        """Read the first shape by its stable parameter identity."""
        return self._parameters[Beta.ALPHA]

    @property
    def beta(self) -> float:
        """Read the second shape by its stable parameter identity."""
        return self._parameters[Beta.BETA]

    def _validate_input(self, rvs):
        return self._validate_rvs(rvs)

    def _beta_log_probabilities(self, sample):
        """Supply validated log tails to the shared compiled EDF kernels."""
        sample = self._validate_rvs(sample)
        return self._beta_logtails(sample, self.alpha, self.beta)

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Beta.DEFAULT, frozenset(Beta.DEFAULT.parameters)),)


class EbnerLiebenbergBetaGofStatistic(AbstractBetaGofStatistic):
    """Conditional-moment Beta goodness-of-fit statistic of Ebner and Liebenberg.

    Parameters
    ----------
    parameters : ParameterValues
        Beta.DEFAULT.parse({}); both shapes unknown, support fixed at [0, 1].

    Notes
    -----
    Fit a and b by maximum likelihood on each call. For c_j=(a+b)*x_j-a,
    compute T_n=n*integral_0^1 (mean_j(c_j*1{x_j>=t})
    - t**a*(1-t)**b/B(a,b))**2 dt, using equation (3) of [1]_.
    Sorting and cumulative sums evaluate the pairwise minimum term in
    O(n log n) time and O(n) space. Beta ratios use log beta functions.

    Reject for large values. The null law depends on the unknown shapes.
    Calibrate by parametric bootstrap from the fitted Beta distribution,
    refitting both shapes for every replicate (Section 2 of [1]_).
    This class returns only the statistic, without p-values or fitted state.

    References
    ----------
    .. [1] B. Ebner and S. C. Liebenberg, "On a new test of fit to the beta
       distribution" (2020), equation (3). https://arxiv.org/abs/2009.13995
       Published in Stat (2021): https://doi.org/10.1002/sta4.341

    Examples
    --------
    >>> statistic = EbnerLiebenbergBetaGofStatistic(Beta.DEFAULT.parse({}))
    >>> value = statistic.execute_statistic([0.1, 0.2, 0.3, 0.5])
    >>> value >= 0
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Beta.DEFAULT, frozenset()),)

    @staticmethod
    @override
    def short_code():
        return "EL"

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @override
    def execute_statistic(self, rvs, **kwargs) -> float:
        """Return T_n for a finite, nonconstant sample in (0, 1), n >= 2.

        Invalid samples raise ValueError; MLE solver failures propagate.
        Additional keyword arguments are ignored for interface compatibility.

        Returns
        -------
        statistic : float
            Nonnegative discrepancy; reject for large values.
        """
        sample, a, b = self._fit(rvs)
        x = np.sort(sample)
        integrals = betainc(a + 1, b + 1, x)
        third = len(x) * np.exp(betaln(2 * a + 1, 2 * b + 1) - 2 * betaln(a, b))
        return self._statistic_kernel(x, a, b, integrals, third)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _statistic_kernel(x, a, b, integrals, third):
        n = len(x)
        tail, first, cross = 0.0, 0.0, 0.0
        for i in range(n - 1, -1, -1):
            c = (a + b) * x[i] - a
            tail += c
            width = x[i] - x[i - 1] if i else x[i]
            first += width * tail**2
            cross += c * integrals[i]
        first /= n
        ratio = (a / (a + b)) * (b / (a + b + 1))
        second = 2 * ratio * cross
        value = first - second + third
        if not np.isfinite(value):
            raise ValueError("Beta statistic could not be evaluated with finite precision")
        tolerance = 64 * np.finfo(np.float64).eps * (abs(first) + abs(second) + abs(third))
        if value < -tolerance:
            raise ValueError("Beta statistic lost numerical precision")
        return max(0.0, value)


class RiadAbdElMaboodBetaGofStatistic(AbstractBetaGofStatistic):
    """Empirical-Laplace Beta test using the normalized Kummer residual.

    Parameters
    ----------
    parameters : ParameterValues
        Beta.DEFAULT.parse({}); both shapes are estimated by maximum likelihood.
    weight : float, default 2.0
        Positive finite exponential weight rate, fixed before inspecting data.

    Notes
    -----
    With A_i=x_i*(x_i-1)/alpha and B_i=1-(1+beta/alpha)*x_i,
    return n*integral_0^infinity mean((A_i*t+B_i)*exp(-t*x_i))**2
    *exp(-weight*t) dt. Reject large values; calibrate with
    parametric_bootstrap_beta, retaining weight and refitting every replicate.
    The support is fixed at [0, 1]. Samples require n >= 2, interior finite
    observations and nonzero variation. Invalid inputs raise ValueError.

    The analytic pairwise integral costs O(n**2) time and O(n) space.
    Coefficients, compensated summation and the positive quadrature integrand
    use cached nopython Numba methods. Cancellation triggers SciPy quadrature
    with the compiled integrand. Failed numerical evaluation raises ValueError.
    This implements the defining differential equation in [1, 2]_, with MLE
    shapes and fitted mean alpha/(alpha+beta), not the arithmetic sample mean.
    The cross term has denominator alpha, not alpha**2 as printed in the
    expanded RF expression in [2]_; see the mathematical verification report.

    References
    ----------
    .. [1] M. Riad and O. F. Abd El Mabood (2018), A new goodness of fit test
       for the beta distribution based on the empirical Laplace transform.
       https://doi.org/10.17654/AS053020165
    .. [2] B. Ebner and S. C. Liebenberg (2021), Section 4.
       https://doi.org/10.1002/sta4.341

    Examples
    --------
    >>> test = RiadAbdElMaboodBetaGofStatistic(Beta.DEFAULT.parse({}), weight=2)
    >>> test.execute_statistic([0.1, 0.2, 0.3, 0.5]) >= 0
    True
    """

    def __init__(self, parameters: ParameterValues, *, weight=2.0):
        super().__init__(parameters)
        if (
            isinstance(weight, bool)
            or not isinstance(weight, Real)
            or not np.isfinite(weight)
            or weight <= 0
        ):
            raise ValueError("weight must be a finite positive real number")
        self.weight = float(weight)

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Beta.DEFAULT, frozenset()),)

    @staticmethod
    def short_code():
        return "RF"

    def alternative(self) -> Alternative:
        return RightAlternative()

    def configuration(self) -> dict:
        return {**super().configuration(), "weight": self.weight}

    @staticmethod
    @jit(nopython=True, cache=True)
    def _laplace_kernel(x, a, b, weight):
        total = 0.0
        correction = 0.0
        magnitude = 0.0
        for i in range(len(x)):
            for j in range(i + 1):
                inv = 1.0 / (x[i] + x[j] + weight)
                first = 2.0 * (a[i] * inv) * (a[j] * inv)
                cross = (a[i] * b[j] + a[j] * b[i]) * inv
                last = b[i] * b[j]
                factor = inv * (1.0 if i == j else 2.0)
                term = (first + cross + last) * factor
                magnitude += (abs(first) + abs(cross) + abs(last)) * factor
                adjusted = term - correction
                updated = total + adjusted
                correction = (updated - total) - adjusted
                total = updated
        return total / len(x), magnitude / len(x)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _coefficients_kernel(x, alpha, beta):
        """Prepare the two residual coefficients in one allocation pass."""
        a = np.empty(len(x))
        b = np.empty(len(x))
        ratio = 1 + beta / alpha
        for i in range(len(x)):
            a[i] = x[i] * (x[i] - 1) / alpha
            b[i] = 1 - ratio * x[i]
        return a, b

    @staticmethod
    @jit(nopython=True, cache=True)
    def _integrand_kernel(u, shifted, a, b, rate):
        """Evaluate the scaled quadrature integrand without temporary arrays."""
        t = u / rate
        total = 0.0
        for i in range(len(shifted)):
            total += (a[i] * t + b[i]) * np.exp(-shifted[i] * t)
        residual = total / len(shifted)
        return len(shifted) / rate * residual**2 * np.exp(-u)

    def execute_statistic(self, rvs, **kwargs) -> float:
        """Return the weighted squared Kummer residual after fitting both shapes."""
        x, alpha, beta = self._fit(rvs)
        a, b = self._coefficients_kernel(x, alpha, beta)
        value, magnitude = self._laplace_kernel(x, a, b, self.weight)
        if not np.isfinite(value) or not np.isfinite(magnitude):
            raise ValueError("Laplace statistic exceeds floating-point range")
        if value <= 256 * np.finfo(float).eps * magnitude:
            # Scale out the slowest exponential decay before adaptive integration.
            minimum = np.min(x)
            rate = self.weight + 2 * minimum
            value, error = quad(
                self._integrand_kernel,
                0,
                np.inf,
                args=(x - minimum, a, b, rate),
                epsabs=1e-14,
                epsrel=1e-9,
                limit=200,
            )
            if not np.isfinite(value) or error > max(1e-14, 1e-7 * value):
                raise ValueError("Laplace statistic quadrature failed to converge")
        return float(value)


class KullbackLeiblerBetaGofStatistic(AbstractBetaGofStatistic):
    """Vasicek-entropy Beta test of Alizadeh Noughabi and Arghami (2013).

    Parameters
    ----------
    parameters : ParameterValues
        Beta.DEFAULT.parse({}); both shapes unknown, support fixed at [0, 1].
    window : int or None, default None
        Vasicek half-window m, with 1 <= m < n/2. None selects
        min(max(1, floor(sqrt(n))), (n-1)//2), a library default.

    Notes
    -----
    Return -H_mn - mean(log f_alpha,beta(x)), using MLE shapes and
    H_mn=mean(log(n/(2*m)*(x_(i+m)-x_(i-m)))). Indices outside 1..n
    repeat the nearest sample endpoint. No entropy bias correction or
    nonnegative truncation is applied: finite-sample estimates can be negative.
    Reject large values. Use parametric_bootstrap_beta with the same window
    and refitting; published tables are not supplied by this implementation.
    Samples require n >= 3, finite interior observations, nonzero variation
    and positive m-spacings. Zero spacings raise ValueError (continuous-data
    estimator undefined); ties with positive m-spacings are allowed.
    Cost is O(n log n) time and O(n) space, excluding MLE iterations.
    A cached nopython Numba method accumulates entropy and log density in one
    pass without spacing or log-density arrays. SciPy evaluates the MLE and
    scalar log beta normalization; fast-math is disabled.

    References
    ----------
    .. [1] H. Alizadeh Noughabi and N. R. Arghami (2013; online 2012),
       General treatment of goodness-of-fit tests based on Kullback-Leibler
       information. https://doi.org/10.1080/00949655.2012.667100
    .. [2] O. Vasicek (1976), A test for normality based on sample entropy.
       https://doi.org/10.1111/j.2517-6161.1976.tb01566.x
    .. [3] H. Alizadeh Noughabi and J. Jarrahiferiz (2020), Section 2 reproduces
       the general statistic in [1]. https://doi.org/10.1080/02664763.2019.1698522

    Examples
    --------
    >>> test = KullbackLeiblerBetaGofStatistic(Beta.DEFAULT.parse({}), window=1)
    >>> bool(np.isfinite(test.execute_statistic([0.1, 0.2, 0.3, 0.5])))
    True
    """

    def __init__(self, parameters: ParameterValues, *, window=None):
        super().__init__(parameters)
        if window is not None and (
            isinstance(window, bool) or not isinstance(window, Integral) or window < 1
        ):
            raise ValueError("window must be a positive integer or None")
        self.window = None if window is None else int(window)

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Beta.DEFAULT, frozenset()),)

    @staticmethod
    def short_code():
        return "KL"

    def alternative(self) -> Alternative:
        return RightAlternative()

    def configuration(self) -> dict:
        return {**super().configuration(), "window": self.window}

    @staticmethod
    @jit(nopython=True, cache=True)
    def _statistic_kernel(x, m, alpha, beta, log_beta):
        """Reduce Vasicek entropy and Beta log density in one compiled pass.

        Input is sorted and strictly interior; log_beta is evaluated by SciPy.
        Repeated endpoints define the window gaps. Negative KL is retained.
        """
        n = len(x)
        log_spacings = 0.0
        log_density = 0.0
        for i in range(n):
            spacing = x[min(i + m, n - 1)] - x[max(i - m, 0)]
            if spacing <= 0:
                raise ValueError("Vasicek entropy requires positive window spacings")
            log_spacings += np.log(spacing)
            log_density += (alpha - 1) * np.log(x[i]) + (beta - 1) * np.log1p(-x[i]) - log_beta
        entropy = log_spacings / n + np.log(n / (2 * m))
        value = -entropy - log_density / n
        if not np.isfinite(value):
            raise ValueError("KL statistic could not be evaluated with finite precision")
        return value

    def execute_statistic(self, rvs, **kwargs) -> float:
        """Return estimated KL discrepancy; reject invalid windows or zero spacings."""
        sample = self._validate_rvs(rvs, min_size=3)
        n = len(sample)
        m = self.window if self.window is not None else min(max(1, int(np.sqrt(n))), (n - 1) // 2)
        if 2 * m >= n:
            raise ValueError("window must satisfy 1 <= window < n/2")
        _, alpha, beta = self._fit(sample)
        return float(self._statistic_kernel(np.sort(sample), m, alpha, beta, betaln(alpha, beta)))


class NeymanSmoothBetaGofStatistic(AbstractSpecifiedBetaGofStatistic):
    """Neyman's smooth test applied to the specified Beta probability transform.

    Parameters
    ----------
    parameters : ParameterValues
        Both positive Beta shapes must be fixed independently of the sample.
    k : int, optional
        Number of shifted Legendre components, positive and fixed; default 4.

    Notes
    -----
    With u_i=F_{a,b}(x_i), return sum_{j=1}^k (sum_i phi_j(u_i))**2/n,
    phi_j(u)=sqrt(2*j+1)*P_j(2*u-1), evaluated by the Legendre recurrence.
    Reject for large values. For fixed k the null limit is chi-square(k)
    and the finite-sample null is shape-free. This is not omnibus for fixed k.
    Fitted shapes require a different calibration and are not supported here.

    References
    ----------
    J. Neyman (1937), Smooth test for goodness of fit, 149-199.
    https://doi.org/10.1080/03461238.1937.10404821
    """

    def __init__(self, parameters: ParameterValues, *, k=4):
        super().__init__(parameters)
        self._uniform = NeymanSmoothTestUniformGofStatistic(
            Uniform.DEFAULT.parse({"a": 0, "b": 1}), k=k
        )
        self.k = self._uniform.k

    @staticmethod
    @override
    def short_code():
        return "NEYMAN"

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @override
    def execute_statistic(self, rvs, **kwargs) -> float:
        """Return the smooth statistic for a nonempty finite sample in [0, 1]."""
        sample = self._validate_rvs(rvs)
        return self._statistic_kernel(scipy_stats.beta.cdf(sample, self.alpha, self.beta), self.k)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _statistic_kernel(cdf, k):
        sums = np.zeros(k)
        for u in cdf:
            x = 2 * u - 1
            previous, current = 1.0, x
            for j in range(1, k + 1):
                sums[j - 1] += np.sqrt(2 * j + 1) * current
                previous, current = current, ((2 * j + 1) * x * current - j * previous) / (j + 1)
        return np.sum(sums**2) / len(cdf)


class RaschkeBetaGofStatistic(AbstractBetaGofStatistic, edf.AbstractAndersonDarlingStatistic):
    """Biased-transformation Beta test, Raschke (2009), Sections 4-5.

    Parameters
    ----------
    parameters : ParameterValues
        Beta.DEFAULT.parse({}); both shapes are estimated by MLE.

    Notes
    -----
    Transform y_i=Phi^{-1}(F_{a_hat,b_hat}(x_i)), fit normal mean and
    standard deviation by MLE (ddof=0), then compute fitted-normal AD.
    Return A**2*(1+0.75/n+2.25/n**2), the paper's corrected statistic.
    Uses the shared AD kernel and both Beta tails to avoid 1-CDF cancellation.
    Normal standardization and AD reduction use cached nopython Numba methods;
    distribution transforms and Beta fitting remain in SciPy.
    Requires at least three nonconstant observations strictly in (0,1).
    Use parametric_bootstrap_beta in the hypothesis_testing layer to refit
    the whole procedure. The paper's normal critical values are approximate;
    no universal null law or omnibus consistency is asserted here.

    References
    ----------
    M. Raschke (2009), The Biased Transformation and Its Application in
    Goodness-of-Fit Tests for the Beta and Gamma Distribution, 1870-1890.
    https://doi.org/10.1080/03610910903152631
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Beta.DEFAULT, frozenset()),)

    @staticmethod
    @override
    def short_code():
        return "RASCHKE"

    @staticmethod
    @jit(nopython=True, cache=True)
    def _standardize_kernel(y):
        """Fit normal location/scale in two passes and return standardized data."""
        n = len(y)
        total = 0.0
        for value in y:
            if not np.isfinite(value):
                raise ValueError("Beta normal transformation lost numerical precision")
            total += value
        mean = total / n
        z = np.empty(n)
        squared = 0.0
        for i in range(n):
            z[i] = y[i] - mean
            squared += z[i] ** 2
        scale = np.sqrt(squared / n)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Normal maximum likelihood requires positive finite scale")
        for i in range(n):
            z[i] /= scale
        return z

    @override
    def execute_statistic(self, rvs, **kwargs) -> float:
        """Return corrected AD after Beta and normal MLE; reject large values.

        Invalid samples or unrepresentable transformations raise ValueError.
        MLE solver errors propagate; observations are never clipped.
        """
        sample = self._validate_rvs(rvs, min_size=3)
        sample, a, b = self._fit(sample)
        x = np.sort(sample)
        lower = scipy_stats.beta.cdf(x, a, b)
        upper = scipy_stats.beta.sf(x, a, b)
        y = np.empty_like(x)
        left = lower <= 0.5
        y[left] = scipy_stats.norm.ppf(lower[left])
        y[~left] = scipy_stats.norm.isf(upper[~left])
        z = self._standardize_kernel(y)
        value = self._ad_kernel(scipy_stats.norm.logcdf(z), scipy_stats.norm.logsf(z))
        n = len(x)
        return float(value * (1 + 0.75 / n + 2.25 / n**2))


_EDF_STATISTICS = {
    "KolmogorovSmirnovDPlusBetaGofStatistic": EDFSpec(
        edf.AbstractKSDPlusStatistic,
    ),
    "KolmogorovSmirnovDMinusBetaGofStatistic": EDFSpec(
        edf.AbstractKSDMinusStatistic,
    ),
    "KolmogorovSmirnovBetaGofStatistic": EDFSpec(
        edf.AbstractKSStatistic,
    ),
    "AndersonDarlingBetaGofStatistic": EDFSpec(
        edf.AbstractAndersonDarlingStatistic,
    ),
    "CrammerVonMisesBetaGofStatistic": EDFSpec(
        edf.AbstractCramerVonMisesStatistic,
    ),
    "LiaoShimokawaBetaGofStatistic": EDFSpec(
        edf.AbstractLiaoShimokawaStatistic,
        short_code="LS",
    ),
    "WatsonBetaGofStatistic": EDFSpec(
        edf.AbstractWatsonStatistic,
    ),
    "KuiperBetaGofStatistic": EDFSpec(
        edf.AbstractKuiperStatistic,
    ),
    "KuiperScaledBetaGofStatistic": EDFSpec(
        edf.AbstractKuiperScaledStatistic,
    ),
    "KuiperCorrectedBetaGofStatistic": EDFSpec(
        edf.AbstractKuiperCorrectedStatistic,
    ),
    "ZhangKBetaGofStatistic": EDFSpec(
        edf.AbstractZhangKStatistic,
    ),
    "ZhangABetaGofStatistic": EDFSpec(
        edf.AbstractZhangAStatistic,
    ),
    "ZhangCBetaGofStatistic": EDFSpec(
        edf.AbstractZhangCStatistic,
    ),
}

_FREQUENCY_STATISTICS = {
    "Chi2PearsonBetaGofStatistic": FrequencySpec(
        frequency.AbstractPearsonChi2Statistic,
    ),
    "FreemanTukeyBetaGofStatistic": FrequencySpec(
        frequency.AbstractFreemanTukeyStatistic,
    ),
    "ModifiedLogLikelihoodBetaGofStatistic": FrequencySpec(
        frequency.AbstractModifiedLogLikelihoodStatistic,
    ),
    "NeymanChi2BetaGofStatistic": FrequencySpec(
        frequency.AbstractNeymanChi2Statistic,
    ),
    "CressieReadBetaGofStatistic": FrequencySpec(
        frequency.AbstractCressieReadStatistic,
    ),
}

__all__ = [
    "AbstractBetaGofStatistic",
    "AbstractSpecifiedBetaGofStatistic",
    "EbnerLiebenbergBetaGofStatistic",
    "KullbackLeiblerBetaGofStatistic",
    "NeymanSmoothBetaGofStatistic",
    "RaschkeBetaGofStatistic",
    "RiadAbdElMaboodBetaGofStatistic",
]
__all__.extend(
    export_edf_statistics(
        globals(),
        AbstractSpecifiedBetaGofStatistic,
        _EDF_STATISTICS,
        log_probabilities=AbstractSpecifiedBetaGofStatistic._beta_log_probabilities,
    )
)
__all__.extend(
    export_frequency_statistics(
        globals(),
        AbstractSpecifiedBetaGofStatistic,
        _FREQUENCY_STATISTICS,
    )
)

for _name in (*_EDF_STATISTICS, *_FREQUENCY_STATISTICS):
    globals()[_name].__doc__ += """

    Beta hypothesis
    ---------------
    Both positive shape parameters must be specified with
    Beta.DEFAULT.parse({"a": 2, "b": 5}). Samples must be nonempty,
    one-dimensional, finite and real, with every value in [0, 1].
    Parameters are never fitted. Reject for large statistic values.
    """
    if _name in _FREQUENCY_STATISTICS:
        globals()[_name].__doc__ += """
    The binning keyword accepts a BinningStrategy; the default is
    EqualProbabilityBins(). Calibration must use the same binning strategy.
    """
del _name
