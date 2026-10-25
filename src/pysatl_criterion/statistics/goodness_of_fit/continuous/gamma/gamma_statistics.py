from __future__ import annotations

from abc import ABC

import numpy as np
import scipy.stats as scipy_stats
from numba import jit
from typing_extensions import override

from pysatl_criterion.distribution import GammaDistributionDescriptor
from pysatl_criterion.distribution import GammaDistributionDescriptor as Distribution
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
from pysatl_criterion.statistics.goodness_of_fit.continuous.graph_goodness_of_fit import (
    AbstractGraphTestStatistic,
    GraphAverageDegreeTestStatistic,
    GraphCliqueNumberTestStatistic,
    GraphConnectedComponentsTestStatistic,
    GraphEdgesNumberTestStatistic,
    GraphIndependenceNumberTestStatistic,
    GraphMaxDegreeTestStatistic,
)
from pysatl_criterion.statistics.goodness_of_fit.goodness_of_fit_statistic import (
    AbstractGoodnessOfFitStatistic,
)


class AbstractGammaGofStatistic(AbstractGoodnessOfFitStatistic, ABC):
    """
    Abstract base class for Gamma distribution goodness-of-fit statistics.
    """

    @staticmethod
    def _validate_input(rvs, minimum=1):
        """Validate the closed Gamma support, retaining zero as a boundary value."""
        if np.iscomplexobj(rvs) or (np.ma.isMaskedArray(rvs) and np.any(rvs.mask)):
            raise ValueError("Sample must be real and contain no masked observations")
        try:
            sample = np.array(rvs, dtype=float, copy=True)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Sample must be numeric") from exc
        if sample.ndim != 1:
            raise ValueError("Sample must be one-dimensional")
        if sample.size == 0:
            raise ValueError("At least one observation is required")
        if sample.size < minimum:
            raise ValueError("At least two observations are required")
        if not np.all(np.isfinite(sample)) or np.any(sample < 0):
            raise ValueError("Sample must contain finite nonnegative observations")
        return sample

    @staticmethod
    def _sorted_sample(rvs):
        return np.sort(AbstractGammaGofStatistic._validate_input(rvs))

    @property
    def alpha(self) -> float:
        """Read alpha by its stable parameter identity."""
        return self._parameters[Distribution.SHAPE]

    @property
    def beta(self) -> float:
        """Read beta by its stable parameter identity."""
        return self._parameters[Distribution.RATE]

    def _scaled(self, sample):
        with np.errstate(over="ignore", under="ignore"):
            return sample * self.beta

    def _cdf(self, sample):
        return scipy_stats.gamma.cdf(self._scaled(sample), a=self.alpha)

    def _logs(self, sample):
        scaled = self._scaled(sample)
        return (
            scipy_stats.gamma.logcdf(scaled, a=self.alpha),
            scipy_stats.gamma.logsf(scaled, a=self.alpha),
        )

    @staticmethod
    @override
    def distribution() -> type[GammaDistributionDescriptor]:
        """Return the distribution descriptor class."""
        return GammaDistributionDescriptor

    @classmethod
    @override
    def code(cls) -> str:
        """Return the family identifier or the concrete statistic's full identifier."""
        family_code = f"GAMMA_{AbstractGoodnessOfFitStatistic.code()}"
        if "short_code" in cls.__abstractmethods__:
            return family_code
        return f"{cls.short_code()}_{family_code}"


class GreenwoodGammaGofStatistic(AbstractGammaGofStatistic):
    """Sum of squared Gamma probability spacings.

    Parameters
    ----------
    parameters : ParameterValues
        Values with alfa, beta fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute one scalar statistic without changing the sample.
    hypothesis()
        Return fixed null parameters; omitted keys mean unknown parameters.

    Notes
    -----
    For ordered observations x_(i), let u_i=F(x_(i); alpha, beta),
    where Gamma has shape alpha, rate beta and known origin zero.
    Parameters are fixed, not fitted. Samples must be finite, real,
    one-dimensional and nonempty, in the closed support [0, infinity).
    Zero is accepted as a boundary value. Ties and constant samples are
    allowed except where stated. Inputs and instance state are unchanged.
    Continuous-null calibration assumes iid observations; rounded data
    require calibration of the observation process.

    G = sum(D_j**2), j=1,...,n+1, where D_j=u_j-u_(j-1),
    u_0=0 and u_(n+1)=1. Both endpoint spacings are included.
    Repeated observations give zero spacings and are permitted.
    The selected right tail detects unusually large squared spacings;
    it does not test against exceptionally regular spacing.

    The fixed-CDF probability transform gives a parameter-free null law
    in exact arithmetic. Reject for large values. The referenced general
    statistic is applied through the Gamma CDF, not a special Gamma table.

    References
    ----------
    .. [1] M. Greenwood (1946), "The Statistical Study of Infectious Diseases",
       J. R. Statist. Soc. 109, 85-103.
       https://doi.org/10.1111/j.2397-2335.1946.tb04649.x

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'alfa': 1, 'beta': 1})
    >>> statistic = GreenwoodGammaGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
    >>> bool(np.isfinite(value))
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.RATE})
            ),
        )

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "GRW".
        """
        return "GRW"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the scalar statistic defined in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real nonnegative observations. Zero is a support boundary.
            At least one observation; ties and constant samples are allowed.
        **kwargs : dict, optional
            Reserved for interface compatibility; ignored.

        Returns
        -------
        statistic : float or numpy.float64
            The statistic in the class Notes. No p-value is calculated.

        Raises
        ------
        ValueError
            If the sample violates the stated shape, support or size constraints,
            or the required estimates/quantiles are numerically degenerate.

        Notes
        -----
        See the class Notes for the exact formula, critical tail and
        calibration requirements.
        """

        sorted_rvs = np.sort(self._validate_input(rvs))
        cdf_vals = self._cdf(sorted_rvs)
        return float(self.do_execute_statistic(cdf_vals))

    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(cdf_vals) -> float:
        """Sum all squared spacings without allocating an extended sample."""
        total = 0.0
        previous = 0.0
        for value in cdf_vals:
            spacing = value - previous
            if spacing < 0:
                raise ValueError("Spacings must be non-negative; check input data ordering.")
            total += spacing * spacing
            previous = value
        if previous > 1:
            raise ValueError("Spacings must be non-negative; check input data ordering.")
        return total + (1.0 - previous) ** 2


class MoranGammaGofStatistic(AbstractGammaGofStatistic):
    """Negative log product of Gamma probability spacings.

    Parameters
    ----------
    parameters : ParameterValues
        Values with alfa, beta fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute one scalar statistic without changing the sample.
    hypothesis()
        Return fixed null parameters; omitted keys mean unknown parameters.

    Notes
    -----
    For ordered observations x_(i), let u_i=F(x_(i); alpha, beta),
    where Gamma has shape alpha, rate beta and known origin zero.
    Parameters are fixed, not fitted. Samples must be finite, real,
    one-dimensional and nonempty, in the closed support [0, infinity).
    Zero is accepted as a boundary value. Ties and constant samples are
    allowed except where stated. Inputs and instance state are unchanged.
    Continuous-null calibration assumes iid observations; rounded data
    require calibration of the observation process.

    M = -sum(log(n*D_j)), j=1,...,n+1, with D_j=u_j-u_(j-1),
    u_0=0 and u_(n+1)=1. The historical n scaling is retained: this
    is shifted upward by (n+1)*log((n+1)/n) relative to (n+1) scaling.
    Zero observations or repeated values give positive infinity.
    Log-CDF differences in the lower tail and log-survival differences
    in the upper tail avoid subtracting CDF values rounded to one.
    Unresolved spacings at float64 precision also produce infinity.
    This historical normalization differs by a deterministic shift from
    the usual n+1 convention; do not reuse tables without adjusting it.

    Reject for large values. No p-value is computed by this class.

    References
    ----------
    Moran, P. A. P. (1951). The Random Division of an Interval--Part II.
    https://doi.org/10.1111/j.2517-6161.1951.tb00079.x
    Background for log spacings; see Notes for the retained n normalization.

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'alfa': 1, 'beta': 1})
    >>> statistic = MoranGammaGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
    >>> bool(np.isfinite(value))
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.RATE})
            ),
        )

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "MOR".
        """
        return "MOR"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the scalar statistic defined in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real nonnegative observations. Zero is a support boundary.
            At least one observation; ties and constant samples are allowed.
        **kwargs : dict, optional
            Reserved for interface compatibility; ignored.

        Returns
        -------
        statistic : float or numpy.float64
            The statistic in the class Notes. No p-value is calculated.

        Raises
        ------
        ValueError
            If the sample violates the stated shape, support or size constraints,
            or the required estimates/quantiles are numerically degenerate.

        Notes
        -----
        Zero spacings (ties or the zero boundary) return positive infinity.
        Unresolved spacings at float64 precision also return infinity.
        """

        sorted_rvs = np.sort(self._validate_input(rvs))
        log_cdf, log_sf = self._logs(sorted_rvs)
        return float(self.do_execute_statistic(log_cdf, log_sf))

    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(log_cdf, log_sf) -> float:
        """Evaluate log spacings using the better-resolved tail."""
        n = len(log_cdf)
        total = log_cdf[0] + log_sf[-1]
        for i in range(1, n):
            if log_cdf[i] <= -np.log(2.0):
                high, low = log_cdf[i], log_cdf[i - 1]
            else:
                high, low = log_sf[i - 1], log_sf[i]
            if high == low:
                return np.inf
            total += high + np.log(-np.expm1(low - high))
        return -total - (n + 1) * np.log(n)


class MinToshiyukiGammaGofStatistic(AbstractGammaGofStatistic):
    """Locally defined tail-weighted Gamma EDF discrepancy.

    Parameters
    ----------
    parameters : ParameterValues
        Values with alfa, beta fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute one scalar statistic without changing the sample.
    hypothesis()
        Return fixed null parameters; omitted keys mean unknown parameters.

    Notes
    -----
    For ordered observations x_(i), let u_i=F(x_(i); alpha, beta),
    where Gamma has shape alpha, rate beta and known origin zero.
    Parameters are fixed, not fitted. Samples must be finite, real,
    one-dimensional and nonempty, in the closed support [0, infinity).
    Zero is accepted as a boundary value. Ties and constant samples are
    allowed except where stated. Inputs and instance state are unchanged.
    Continuous-null calibration assumes iid observations; rounded data
    require calibration of the observation process.

    T = sum(max(i/n-u_i, u_i-(i-1)/n)/sqrt(u_i*(1-u_i)))/sqrt(n).
    Weights use log-CDF and log-survival values for tail accuracy.
    Zero observations yield positive infinity; extreme weights may overflow.
    This is the same formula as LiaoShimokawaGammaGofStatistic. The public
    name is retained for compatibility. Liao and Shimokawa (1999) studied
    fitted Weibull/extreme-value models; their tables do not calibrate the
    fully specified Gamma null used here.
    Use simulation of this formula under the specified Gamma null.

    Reject for large values. No p-value is computed by this class.

    References
    ----------
    Liao, M. and Shimokawa, T. (1999). A new goodness-of-fit test for
    type-I extreme-value and 2-parameter Weibull distributions with
    estimated parameters.
    https://doi.org/10.1080/00949659908811965
    Formula source, not a Gamma-specific calibration.

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'alfa': 1, 'beta': 1})
    >>> statistic = MinToshiyukiGammaGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
    >>> bool(np.isfinite(value))
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.RATE})
            ),
        )

    def alternative(self):
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "MT".
        """
        return "MT"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the scalar statistic defined in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real nonnegative observations. Zero is a support boundary.
            At least one observation; ties and constant samples are allowed.
        **kwargs : dict, optional
            Reserved for interface compatibility; ignored.

        Returns
        -------
        statistic : float or numpy.float64
            The statistic in the class Notes. No p-value is calculated.

        Raises
        ------
        ValueError
            If the sample violates the stated shape, support or size constraints,
            or the required estimates/quantiles are numerically degenerate.

        Notes
        -----
        Zero observations return positive infinity. Extreme tail weights
        can overflow; no epsilon clipping is applied.
        """

        sorted_rvs = np.sort(self._validate_input(rvs))
        cdf_vals = self._cdf(sorted_rvs)
        log_cdf, log_sf = self._logs(sorted_rvs)
        return float(self.do_execute_statistic(cdf_vals, log_cdf, log_sf))

    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(cdf_vals, log_cdf, log_sf) -> float:
        """Accumulate weighted deviations without temporary rank arrays."""
        n = len(cdf_vals)
        total = 0.0
        for i in range(n):
            deviation = max((i + 1) / n - cdf_vals[i], cdf_vals[i] - i / n)
            total += deviation * np.exp(-0.5 * (log_cdf[i] + log_sf[i]))
        return total / np.sqrt(n)


class AbstractBinnedGammaGofStatistic(AbstractGammaGofStatistic, ABC):
    """
    Base class for Gamma GOF tests built on equiprobable histogram bins.

    Provides common infrastructure for chi-squared type tests that bin data
    using Gamma quantile function to ensure equal theoretical probability per bin.
    """

    def alternative(self):
        return RightAlternative()

    lambda_value: float = 1.0

    def __init__(self, parameters: ParameterValues, *, bins: int = 8):
        if (
            isinstance(bins, (bool, np.bool_))
            or not isinstance(bins, (int, np.integer))
            or bins < 2
        ):
            raise ValueError("At least two bins are required for binned Gamma statistics.")
        self.bins = bins
        AbstractGammaGofStatistic.__init__(self, parameters)
        self.lambda_value = getattr(self, "lambda_value", 1.0)

    def _validate_storage_calibration(self):
        raise ValueError(
            "Gamma histogram calibration must include bins and power; "
            "the storage key does not include these settings"
        )

    def _counts_and_expected(self, rvs):
        sample = self._validate_input(rvs)
        n = sample.size

        quantiles = np.linspace(0.0, 1.0, self.bins + 1)
        edges = scipy_stats.gamma.ppf(quantiles, a=self.alpha)
        if not np.all(np.isfinite(edges[1:-1])) or np.any(np.diff(edges) <= 0):
            raise ValueError("Gamma quantile bins collapse at float64 precision")
        edges[0] = -np.inf
        edges[-1] = np.inf
        counts, _ = np.histogram(self._scaled(sample), bins=edges)
        expected = np.full(self.bins, n / self.bins)
        return counts, expected

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the scalar statistic defined in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real nonnegative observations. Zero is a support boundary.
            At least one observation; ties and constant samples are allowed.
        **kwargs : dict, optional
            Reserved for interface compatibility; ignored.

        Returns
        -------
        statistic : float or numpy.float64
            The statistic in the class Notes. No p-value is calculated.

        Raises
        ------
        ValueError
            If the sample violates the stated shape, support or size constraints,
            or the required estimates/quantiles are numerically degenerate.

        Notes
        -----
        Empty bins return positive infinity for power <= -1.
        For power > -1 their contribution is evaluated by continuity.
        """
        counts, expected = self._counts_and_expected(rvs)
        return float(
            scipy_stats.power_divergence(counts, expected, lambda_=self.lambda_value).statistic
        )


class LikelihoodRatioGammaGofStatistic(AbstractBinnedGammaGofStatistic):
    """Multinomial likelihood-ratio statistic on Gamma quantile bins.

    Parameters
    ----------
    parameters : ParameterValues
        Values with alfa, beta fixed; omitted parameters are unknown.
    bins : int, optional
        Number of equal-probability bins, at least 2. Default is 8.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute one scalar statistic without changing the sample.
    hypothesis()
        Return fixed null parameters; omitted keys mean unknown parameters.

    Notes
    -----
    For ordered observations x_(i), let u_i=F(x_(i); alpha, beta),
    where Gamma has shape alpha, rate beta and known origin zero.
    Parameters are fixed, not fitted. Samples must be finite, real,
    one-dimensional and nonempty, in the closed support [0, infinity).
    Zero is accepted as a boundary value. Ties and constant samples are
    allowed except where stated. Inputs and instance state are unchanged.
    Continuous-null calibration assumes iid observations; rounded data
    require calibration of the observation process.

    G2 = 2*sum(O_j*log(O_j/E_j)), with E_j=n/bins.
    Zero counts contribute zero by continuity. Bins use Gamma quantiles
    at j/bins, include the left boundary, and cover the entire support.
    This is the power-divergence limit at power=0.
    Reject for large values. Under the fixed null, counts are multinomial
    with probabilities 1/bins. Null calibration depends on n, bins and power.
    The chi-square(bins-1) approximation requires sufficiently large expected
    counts. Storage lookup is blocked because its key omits bins and power;
    use MonteCarloLimitDistributionResolver, which uses this instance.
    Unrepresentable quantile boundaries raise ValueError.

    References
    ----------
    .. [1] N. Cressie and T. R. C. Read (1984), "Multinomial Goodness-Of-Fit
       Tests", J. R. Statist. Soc. B 46, 440-464.
       https://doi.org/10.1111/j.2517-6161.1984.tb01318.x

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'alfa': 1, 'beta': 1})
    >>> statistic = LikelihoodRatioGammaGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
    >>> bool(np.isfinite(value))
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.RATE})
            ),
        )

    lambda_value = 0.0

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "G_TEST".
        """
        return "G_TEST"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the scalar statistic defined in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real nonnegative observations. Zero is a support boundary.
            At least one observation; ties and constant samples are allowed.
        **kwargs : dict, optional
            Reserved for interface compatibility; ignored.

        Returns
        -------
        statistic : float or numpy.float64
            The statistic in the class Notes. No p-value is calculated.

        Raises
        ------
        ValueError
            If the sample violates the stated shape, support or size constraints,
            or the required estimates/quantiles are numerically degenerate.

        Notes
        -----
        See the class Notes for the exact formula, critical tail and
        calibration requirements.
        """

        counts, expected = self._counts_and_expected(rvs)
        return float(self.do_execute_statistic(counts, expected))

    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(observed, expected) -> float:
        """Multinomial log likelihood ratio with the zero-count limit."""
        total = 0.0
        for i in range(len(observed)):
            if observed[i] > 0:
                total += observed[i] * np.log(observed[i] / expected[i])
        return 2.0 * total


class ProbabilityPlotCorrelationGammaGofStatistic(AbstractGammaGofStatistic):
    """Gamma quantile correlation discrepancy with Blom positions.

    Parameters
    ----------
    parameters : ParameterValues
        Values with alfa, beta fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute one scalar statistic without changing the sample.
    hypothesis()
        Return fixed null parameters; omitted keys mean unknown parameters.

    Notes
    -----
    For ordered observations x_(i), let u_i=F(x_(i); alpha, beta),
    where Gamma has shape alpha, rate beta and known origin zero.
    Parameters are fixed, not fitted. Samples must be finite, real,
    one-dimensional and nonempty, in the closed support [0, infinity).
    Zero is accepted as a boundary value. Ties and constant samples are
    allowed except where stated. Inputs and instance state are unchanged.
    Continuous-null calibration assumes iid observations; rounded data
    require calibration of the observation process.

    Return 1-r, where r is the Pearson correlation of ordered x with
    Gamma.ppf((i-0.375)/(n+0.25), alpha, scale=1). The rate cancels
    from correlation; beta remains fixed in the declared null hypothesis.
    The statistic cannot detect positive affine changes of the sample.
    The null law depends on alpha. A nonconstant sample with n>=2 is
    required; for n=2 the statistic is degenerate and has no useful power.
    Independent rescaling of data and quantiles avoids moment overflow.
    No primary source validating this exact Gamma/Blom test was found;
    Filliben normality tables do not calibrate it.

    Reject for large values. No p-value is computed by this class.

    References
    ----------
    Filliben, J. J. (1975). The Probability Plot Correlation Coefficient
    Test for Normality.
    https://doi.org/10.1080/00401706.1975.10489279
    Methodological background only; the Gamma/Blom variant is a local adaptation.

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'alfa': 1, 'beta': 1})
    >>> statistic = ProbabilityPlotCorrelationGammaGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
    >>> bool(np.isfinite(value))
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.RATE})
            ),
        )

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "PPCC".
        """
        return "PPCC"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the scalar statistic defined in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real nonnegative observations. Zero is a support boundary.
            At least two observations; the sample must be nonconstant.
        **kwargs : dict, optional
            Reserved for interface compatibility; ignored.

        Returns
        -------
        statistic : float or numpy.float64
            The statistic in the class Notes. No p-value is calculated.

        Raises
        ------
        ValueError
            If the sample violates the stated shape, support or size constraints,
            or the required estimates/quantiles are numerically degenerate.

        Notes
        -----
        See the class Notes for the exact formula, critical tail and
        calibration requirements.
        """

        sample = np.sort(self._validate_input(rvs, minimum=2))
        n = sample.size

        plotting_positions = (np.arange(1, n + 1) - 0.375) / (n + 0.25)
        expected = scipy_stats.gamma.ppf(plotting_positions, a=self.alpha)

        return float(self.do_execute_statistic(sample, expected))

    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(sample, expected) -> float:
        """Scale and center in two passes without changing either input."""
        max_sample = np.max(sample)
        max_expected = np.max(expected)
        if max_sample == 0 or max_expected == 0 or not np.all(np.isfinite(expected)):
            raise ValueError("Degenerate data encountered while computing PPCC statistic.")
        mean_sample = 0.0
        mean_expected = 0.0
        n = len(sample)
        for i in range(n):
            mean_sample += sample[i] / max_sample
            mean_expected += expected[i] / max_expected
        mean_sample /= n
        mean_expected /= n
        numerator = 0.0
        sample_variance = 0.0
        expected_variance = 0.0
        for i in range(n):
            x = sample[i] / max_sample - mean_sample
            y = expected[i] / max_expected - mean_expected
            numerator += x * y
            sample_variance += x * x
            expected_variance += y * y
        denominator = np.sqrt(sample_variance * expected_variance)
        if denominator == 0:
            raise ValueError("Degenerate data encountered while computing PPCC statistic.")
        return 1.0 - min(1.0, max(-1.0, numerator / denominator))


class AbstractGraphGammaGofStatistic(AbstractGammaGofStatistic, AbstractGraphTestStatistic):
    """
    Base class for Gamma graph-based GOF statistics using EDF transforms.

    Combines Gamma distribution testing with graph-theoretic statistics by
    transforming data through Gamma CDF and analyzing resulting structure.
    """

    @override
    def alternative(self) -> Alternative:
        return TwoSidedAlternative()

    @classmethod
    @override
    def code(cls) -> str:
        """Return the family identifier or the concrete statistic's full identifier."""
        family_code = f"GRAPH_GAMMA_{AbstractGoodnessOfFitStatistic.code()}"
        if "short_code" in cls.__abstractmethods__:
            return family_code
        return f"{cls.short_code()}_{family_code}"

    def _transform_sample(self, rvs):
        sample = self._validate_input(rvs)

        sorted_sample = np.sort(sample)
        uniformized = self._cdf(sorted_sample)
        return uniformized

    def _evaluate_graph_statistic(self, transformed_sample, **kwargs):
        """Evaluate the sorted one-dimensional graph without adjacency allocation."""
        return self.do_execute_statistic(transformed_sample)

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the scalar statistic defined in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real nonnegative observations. Zero is a support boundary.
            At least one observation; ties and constant samples are allowed.
        **kwargs : dict, optional
            Reserved for interface compatibility; ignored.

        Returns
        -------
        statistic : float or numpy.float64
            The statistic in the class Notes. No p-value is calculated.

        Raises
        ------
        ValueError
            If the sample violates the stated shape, support or size constraints,
            or the required estimates/quantiles are numerically degenerate.

        Notes
        -----
        Uses the strict distance rule in the concrete class Notes.
        A nonempty constant sample yields an edgeless graph.
        """
        transformed_sample = self._transform_sample(rvs)
        return float(self._evaluate_graph_statistic(transformed_sample, **kwargs))


class GraphEdgesNumberGammaGofStatistic(
    AbstractGraphGammaGofStatistic, GraphEdgesNumberTestStatistic
):
    """Number of undirected edges, sum(degrees)/2.

    Parameters
    ----------
    parameters : ParameterValues
        Values with alfa, beta fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute one scalar statistic without changing the sample.
    hypothesis()
        Return fixed null parameters; omitted keys mean unknown parameters.

    Notes
    -----
    For ordered observations x_(i), let u_i=F(x_(i); alpha, beta),
    where Gamma has shape alpha, rate beta and known origin zero.
    Parameters are fixed, not fitted. Samples must be finite, real,
    one-dimensional and nonempty, in the closed support [0, infinity).
    Zero is accepted as a boundary value. Ties and constant samples are
    allowed except where stated. Inputs and instance state are unchanged.
    Continuous-null calibration assumes iid observations; rounded data
    require calibration of the observation process.

    Number of undirected edges, sum(degrees)/2.
    Vertices are observations transformed by the fixed Gamma CDF.
    An edge joins i and j exactly when abs(u_i-u_j) < h, with
    h=(max(u)-min(u))/10. This uses CDF positions, not their spacings.
    Ties are separate vertices; h=0 yields an edgeless graph.
    Both tails are selected as a local convention; calibrate this exact
    graph construction. No primary source establishing this Gamma-specific
    procedure or its power properties was found. Floating-point saturation
    of the CDF can merge distinct tail values.

    References
    ----------
    Ebner, B., Nestmann, F. and Schulte, M. (2020 preprint). Testing
    multivariate uniformity based on random geometric graphs.
    https://arxiv.org/abs/1812.08638
    Related methodology only; not the range/10 rule implemented here.

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'alfa': 1, 'beta': 1})
    >>> statistic = GraphEdgesNumberGammaGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
    >>> bool(np.isfinite(value))
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.RATE})
            ),
        )

    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(values) -> float:
        """Evaluate the strict-distance graph in O(n) time on sorted CDF values."""
        n = len(values)
        distance = (values[-1] - values[0]) / 10
        left = 0
        edges = 0
        for right in range(n):
            while left < right and values[right] - values[left] >= distance:
                left += 1
            edges += right - left
        return float(edges)


class GraphMaxDegreeGammaGofStatistic(AbstractGraphGammaGofStatistic, GraphMaxDegreeTestStatistic):
    """Maximum vertex degree.

    Parameters
    ----------
    parameters : ParameterValues
        Values with alfa, beta fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute one scalar statistic without changing the sample.
    hypothesis()
        Return fixed null parameters; omitted keys mean unknown parameters.

    Notes
    -----
    For ordered observations x_(i), let u_i=F(x_(i); alpha, beta),
    where Gamma has shape alpha, rate beta and known origin zero.
    Parameters are fixed, not fitted. Samples must be finite, real,
    one-dimensional and nonempty, in the closed support [0, infinity).
    Zero is accepted as a boundary value. Ties and constant samples are
    allowed except where stated. Inputs and instance state are unchanged.
    Continuous-null calibration assumes iid observations; rounded data
    require calibration of the observation process.

    Maximum vertex degree.
    Vertices are observations transformed by the fixed Gamma CDF.
    An edge joins i and j exactly when abs(u_i-u_j) < h, with
    h=(max(u)-min(u))/10. This uses CDF positions, not their spacings.
    Ties are separate vertices; h=0 yields an edgeless graph.
    Both tails are selected as a local convention; calibrate this exact
    graph construction. No primary source establishing this Gamma-specific
    procedure or its power properties was found. Floating-point saturation
    of the CDF can merge distinct tail values.

    References
    ----------
    Ebner, B., Nestmann, F. and Schulte, M. (2020 preprint). Testing
    multivariate uniformity based on random geometric graphs.
    https://arxiv.org/abs/1812.08638
    Related methodology only; not the range/10 rule implemented here.

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'alfa': 1, 'beta': 1})
    >>> statistic = GraphMaxDegreeGammaGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
    >>> bool(np.isfinite(value))
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.RATE})
            ),
        )

    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(values) -> float:
        """Evaluate the strict-distance graph in O(n) time on sorted CDF values."""
        n = len(values)
        distance = (values[-1] - values[0]) / 10
        if distance == 0:
            return 0.0
        left = 0
        right = 0
        largest = 0
        for i in range(n):
            while left < i and values[i] - values[left] >= distance:
                left += 1
            while right < n and values[right] - values[i] < distance:
                right += 1
            largest = max(largest, right - left - 1)
        return float(largest)


class GraphAverageDegreeGammaGofStatistic(
    AbstractGraphGammaGofStatistic, GraphAverageDegreeTestStatistic
):
    """Mean vertex degree, 2*edges/n.

    Parameters
    ----------
    parameters : ParameterValues
        Values with alfa, beta fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute one scalar statistic without changing the sample.
    hypothesis()
        Return fixed null parameters; omitted keys mean unknown parameters.

    Notes
    -----
    For ordered observations x_(i), let u_i=F(x_(i); alpha, beta),
    where Gamma has shape alpha, rate beta and known origin zero.
    Parameters are fixed, not fitted. Samples must be finite, real,
    one-dimensional and nonempty, in the closed support [0, infinity).
    Zero is accepted as a boundary value. Ties and constant samples are
    allowed except where stated. Inputs and instance state are unchanged.
    Continuous-null calibration assumes iid observations; rounded data
    require calibration of the observation process.

    Mean vertex degree, 2*edges/n.
    Vertices are observations transformed by the fixed Gamma CDF.
    An edge joins i and j exactly when abs(u_i-u_j) < h, with
    h=(max(u)-min(u))/10. This uses CDF positions, not their spacings.
    Ties are separate vertices; h=0 yields an edgeless graph.
    Both tails are selected as a local convention; calibrate this exact
    graph construction. No primary source establishing this Gamma-specific
    procedure or its power properties was found. Floating-point saturation
    of the CDF can merge distinct tail values.

    References
    ----------
    Ebner, B., Nestmann, F. and Schulte, M. (2020 preprint). Testing
    multivariate uniformity based on random geometric graphs.
    https://arxiv.org/abs/1812.08638
    Related methodology only; not the range/10 rule implemented here.

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'alfa': 1, 'beta': 1})
    >>> statistic = GraphAverageDegreeGammaGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
    >>> bool(np.isfinite(value))
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.RATE})
            ),
        )

    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(values) -> float:
        """Evaluate the strict-distance graph in O(n) time on sorted CDF values."""
        n = len(values)
        distance = (values[-1] - values[0]) / 10
        left = 0
        edges = 0
        for right in range(n):
            while left < right and values[right] - values[left] >= distance:
                left += 1
            edges += right - left
        return 2.0 * edges / n


class GraphConnectedComponentsGammaGofStatistic(
    AbstractGraphGammaGofStatistic, GraphConnectedComponentsTestStatistic
):
    """Number of connected components, including isolated vertices.

    Parameters
    ----------
    parameters : ParameterValues
        Values with alfa, beta fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute one scalar statistic without changing the sample.
    hypothesis()
        Return fixed null parameters; omitted keys mean unknown parameters.

    Notes
    -----
    For ordered observations x_(i), let u_i=F(x_(i); alpha, beta),
    where Gamma has shape alpha, rate beta and known origin zero.
    Parameters are fixed, not fitted. Samples must be finite, real,
    one-dimensional and nonempty, in the closed support [0, infinity).
    Zero is accepted as a boundary value. Ties and constant samples are
    allowed except where stated. Inputs and instance state are unchanged.
    Continuous-null calibration assumes iid observations; rounded data
    require calibration of the observation process.

    Number of connected components, including isolated vertices.
    Vertices are observations transformed by the fixed Gamma CDF.
    An edge joins i and j exactly when abs(u_i-u_j) < h, with
    h=(max(u)-min(u))/10. This uses CDF positions, not their spacings.
    Ties are separate vertices; h=0 yields an edgeless graph.
    Both tails are selected as a local convention; calibrate this exact
    graph construction. No primary source establishing this Gamma-specific
    procedure or its power properties was found. Floating-point saturation
    of the CDF can merge distinct tail values.

    References
    ----------
    Ebner, B., Nestmann, F. and Schulte, M. (2020 preprint). Testing
    multivariate uniformity based on random geometric graphs.
    https://arxiv.org/abs/1812.08638
    Related methodology only; not the range/10 rule implemented here.

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'alfa': 1, 'beta': 1})
    >>> statistic = GraphConnectedComponentsGammaGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
    >>> bool(np.isfinite(value))
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.RATE})
            ),
        )

    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(values) -> float:
        """Evaluate the strict-distance graph in O(n) time on sorted CDF values."""
        n = len(values)
        distance = (values[-1] - values[0]) / 10
        components = 1
        for i in range(1, n):
            if values[i] - values[i - 1] >= distance:
                components += 1
        return float(components)


class GraphCliqueNumberGammaGofStatistic(
    AbstractGraphGammaGofStatistic, GraphCliqueNumberTestStatistic
):
    """Size of the largest clique (1 for a nonempty edgeless graph).

    Parameters
    ----------
    parameters : ParameterValues
        Values with alfa, beta fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute one scalar statistic without changing the sample.
    hypothesis()
        Return fixed null parameters; omitted keys mean unknown parameters.

    Notes
    -----
    For ordered observations x_(i), let u_i=F(x_(i); alpha, beta),
    where Gamma has shape alpha, rate beta and known origin zero.
    Parameters are fixed, not fitted. Samples must be finite, real,
    one-dimensional and nonempty, in the closed support [0, infinity).
    Zero is accepted as a boundary value. Ties and constant samples are
    allowed except where stated. Inputs and instance state are unchanged.
    Continuous-null calibration assumes iid observations; rounded data
    require calibration of the observation process.

    Size of the largest clique (1 for a nonempty edgeless graph).
    Vertices are observations transformed by the fixed Gamma CDF.
    An edge joins i and j exactly when abs(u_i-u_j) < h, with
    h=(max(u)-min(u))/10. This uses CDF positions, not their spacings.
    Ties are separate vertices; h=0 yields an edgeless graph.
    Both tails are selected as a local convention; calibrate this exact
    graph construction. No primary source establishing this Gamma-specific
    procedure or its power properties was found. Floating-point saturation
    of the CDF can merge distinct tail values.

    References
    ----------
    Ebner, B., Nestmann, F. and Schulte, M. (2020 preprint). Testing
    multivariate uniformity based on random geometric graphs.
    https://arxiv.org/abs/1812.08638
    Related methodology only; not the range/10 rule implemented here.

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'alfa': 1, 'beta': 1})
    >>> statistic = GraphCliqueNumberGammaGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
    >>> bool(np.isfinite(value))
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.RATE})
            ),
        )

    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(values) -> float:
        """Evaluate the strict-distance graph in O(n) time on sorted CDF values."""
        n = len(values)
        distance = (values[-1] - values[0]) / 10
        if distance == 0:
            return 1.0
        left = 0
        largest = 1
        for right in range(n):
            while values[right] - values[left] >= distance:
                left += 1
            largest = max(largest, right - left + 1)
        return float(largest)


class GraphIndependenceNumberGammaGofStatistic(
    AbstractGraphGammaGofStatistic, GraphIndependenceNumberTestStatistic
):
    """Size of the largest independent set (n for an edgeless graph).

    Parameters
    ----------
    parameters : ParameterValues
        Values with alfa, beta fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute one scalar statistic without changing the sample.
    hypothesis()
        Return fixed null parameters; omitted keys mean unknown parameters.

    Notes
    -----
    For ordered observations x_(i), let u_i=F(x_(i); alpha, beta),
    where Gamma has shape alpha, rate beta and known origin zero.
    Parameters are fixed, not fitted. Samples must be finite, real,
    one-dimensional and nonempty, in the closed support [0, infinity).
    Zero is accepted as a boundary value. Ties and constant samples are
    allowed except where stated. Inputs and instance state are unchanged.
    Continuous-null calibration assumes iid observations; rounded data
    require calibration of the observation process.

    Size of the largest independent set (n for an edgeless graph).
    Vertices are observations transformed by the fixed Gamma CDF.
    An edge joins i and j exactly when abs(u_i-u_j) < h, with
    h=(max(u)-min(u))/10. This uses CDF positions, not their spacings.
    Ties are separate vertices; h=0 yields an edgeless graph.
    Both tails are selected as a local convention; calibrate this exact
    graph construction. No primary source establishing this Gamma-specific
    procedure or its power properties was found. Floating-point saturation
    of the CDF can merge distinct tail values.

    References
    ----------
    Ebner, B., Nestmann, F. and Schulte, M. (2020 preprint). Testing
    multivariate uniformity based on random geometric graphs.
    https://arxiv.org/abs/1812.08638
    Related methodology only; not the range/10 rule implemented here.

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'alfa': 1, 'beta': 1})
    >>> statistic = GraphIndependenceNumberGammaGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
    >>> bool(np.isfinite(value))
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.RATE})
            ),
        )

    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(values) -> float:
        """Evaluate the strict-distance graph in O(n) time on sorted CDF values."""
        n = len(values)
        distance = (values[-1] - values[0]) / 10
        last = values[0]
        count = 1
        for i in range(1, n):
            if values[i] - last >= distance:
                count += 1
                last = values[i]
        return float(count)


_EDF_STATISTICS = {
    "KolmogorovSmirnovDPlusGammaGofStatistic": EDFSpec(
        edf.AbstractKSDPlusStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.RATE})
            ),
        ),
    ),
    "KolmogorovSmirnovDMinusGammaGofStatistic": EDFSpec(
        edf.AbstractKSDMinusStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.RATE})
            ),
        ),
    ),
    "KolmogorovSmirnovGammaGofStatistic": EDFSpec(
        edf.AbstractKSStatistic,
        legacy_ks_options=True,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.RATE})
            ),
        ),
    ),
    "AndersonDarlingGammaGofStatistic": EDFSpec(
        edf.AbstractAndersonDarlingStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.RATE})
            ),
        ),
    ),
    "CramerVonMisesGammaGofStatistic": EDFSpec(
        edf.AbstractCramerVonMisesStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.RATE})
            ),
        ),
    ),
    "LiaoShimokawaGammaGofStatistic": EDFSpec(
        edf.AbstractLiaoShimokawaStatistic,
        short_code="LS",
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.RATE})
            ),
        ),
    ),
    "WatsonGammaGofStatistic": EDFSpec(
        edf.AbstractWatsonStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.RATE})
            ),
        ),
    ),
    "KuiperGammaGofStatistic": EDFSpec(
        edf.AbstractKuiperStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.RATE})
            ),
        ),
    ),
    "KuiperScaledGammaGofStatistic": EDFSpec(
        edf.AbstractKuiperScaledStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.RATE})
            ),
        ),
    ),
    "KuiperCorrectedGammaGofStatistic": EDFSpec(
        edf.AbstractKuiperCorrectedStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.RATE})
            ),
        ),
    ),
    "ZhangKGammaGofStatistic": EDFSpec(
        edf.AbstractZhangKStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.RATE})
            ),
        ),
    ),
    "ZhangAGammaGofStatistic": EDFSpec(
        edf.AbstractZhangAStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.RATE})
            ),
        ),
    ),
    "ZhangCGammaGofStatistic": EDFSpec(
        edf.AbstractZhangCStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.RATE})
            ),
        ),
    ),
}

_FREQUENCY_STATISTICS = {
    "Chi2PearsonGammaGofStatistic": FrequencySpec(
        frequency.AbstractPearsonChi2Statistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.RATE})
            ),
        ),
    ),
    "FreemanTukeyGammaGofStatistic": FrequencySpec(
        frequency.AbstractFreemanTukeyStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.RATE})
            ),
        ),
    ),
    "ModifiedLogLikelihoodGammaGofStatistic": FrequencySpec(
        frequency.AbstractModifiedLogLikelihoodStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.RATE})
            ),
        ),
    ),
    "NeymanChi2GammaGofStatistic": FrequencySpec(
        frequency.AbstractNeymanChi2Statistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.RATE})
            ),
        ),
    ),
    "CressieReadGammaGofStatistic": FrequencySpec(
        frequency.AbstractCressieReadStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.RATE})
            ),
        ),
    ),
}

__all__ = [
    "AbstractBinnedGammaGofStatistic",
    "AbstractGammaGofStatistic",
    "AbstractGraphGammaGofStatistic",
    "GraphAverageDegreeGammaGofStatistic",
    "GraphCliqueNumberGammaGofStatistic",
    "GraphConnectedComponentsGammaGofStatistic",
    "GraphEdgesNumberGammaGofStatistic",
    "GraphIndependenceNumberGammaGofStatistic",
    "GraphMaxDegreeGammaGofStatistic",
    "GreenwoodGammaGofStatistic",
    "LikelihoodRatioGammaGofStatistic",
    "MinToshiyukiGammaGofStatistic",
    "MoranGammaGofStatistic",
    "ProbabilityPlotCorrelationGammaGofStatistic",
]
__all__.extend(
    export_edf_statistics(
        globals(),
        AbstractGammaGofStatistic,
        _EDF_STATISTICS,
        cdf=AbstractGammaGofStatistic._cdf,
        log_probabilities=AbstractGammaGofStatistic._logs,
        sample_validator=AbstractGammaGofStatistic._sorted_sample,
    )
)
__all__.extend(
    export_frequency_statistics(globals(), AbstractGammaGofStatistic, _FREQUENCY_STATISTICS)
)


for _name in (*_EDF_STATISTICS, *_FREQUENCY_STATISTICS):
    globals()[_name].__doc__ += """

    Gamma hypothesis
    ----------------
    Both alfa (shape) and beta (rate, inverse scale) are fixed and positive;
    origin is zero. No parameters are fitted. Input is a finite, real,
    nonempty one-dimensional sample in [0, infinity). Tail probabilities
    can underflow at extreme values; no epsilon clipping is applied.
    """
    globals()[_name].__doc__ += f"""

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({{'alfa': 2.5, 'beta': 1.5}})
    >>> statistic = {_name}(parameters)
    >>> bool(np.isfinite(statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])))
    True
    """
    if _name in _FREQUENCY_STATISTICS:
        globals()[_name].__doc__ += """
    Binning is configured with binning=EqualProbabilityBins(count=k).
    The default count follows PowerCount(); Cressie--Read power is fixed
    at 2/3. Calibration must use the same binning configuration.
    """
del _name
