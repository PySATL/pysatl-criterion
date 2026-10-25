"""Ordinary Weibull criteria with zero location and fitted shape and scale."""

from abc import ABC
from numbers import Integral, Real

import numpy as np
from numba import jit
from scipy.optimize import brentq
from scipy.special import gamma, gammaln, logsumexp
from scipy.stats import gumbel_l
from typing_extensions import override

from pysatl_criterion.distribution import (
    WeibullDistributionDescriptor as Distribution,
)
from pysatl_criterion.distribution.parameters import HypothesisSupport
from pysatl_criterion.statistics.alternative import (
    LeftAlternative,
    RightAlternative,
    TwoSidedAlternative,
)
from pysatl_criterion.statistics.goodness_of_fit.binning import EqualProbabilityBins
from pysatl_criterion.statistics.goodness_of_fit.continuous import edf, frequency
from pysatl_criterion.statistics.goodness_of_fit.continuous._edf_factory import (
    EDFSpec,
    export_edf_statistics,
)
from pysatl_criterion.statistics.goodness_of_fit.continuous._frequency_factory import (
    FrequencySpec,
    export_frequency_statistics,
)
from pysatl_criterion.statistics.goodness_of_fit.continuous._weibull_common import (
    _log_probabilities,
    _sample,
)
from pysatl_criterion.statistics.goodness_of_fit.goodness_of_fit_statistic import (
    AbstractGoodnessOfFitStatistic,
)


def _scalar(value, name, *, positive=False):
    message = f"{name}: expected finite {'positive ' if positive else ''}scalars"
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(message)  # noqa: TRY004 - shared validation contract uses ValueError
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError(message) from None
    if not np.isfinite(result) or (positive and result <= 0):
        raise ValueError(message)
    return result


def _logs(rvs, min_size=2):
    x = _sample(rvs, min_size=min_size, positive=True, nonconstant=True)
    y = np.sort(np.log(x))
    if np.ptp(y) == 0:
        raise ValueError("Logarithms are indistinguishable at float64 precision")
    return y


def _fit_logs(y, counts=None):
    """Profile likelihood root in a rescaled log domain; optional record counts."""
    spread = np.ptp(y)
    v = (y - y[-1]) / spread
    weights = np.ones(y.size) if counts is None else np.asarray(counts, dtype=float)
    log_weights = np.log(weights)

    mean_v = np.mean(v)
    score = AbstractWeibullGofStatistic._likelihood_score
    upper = 1.0
    while score(upper, v, log_weights, mean_v) <= 0:
        upper *= 2
        if not np.isfinite(upper):
            raise ValueError("Weibull likelihood root cannot be represented")
    t = brentq(score, 1e-12, upper, args=(v, log_weights, mean_v), xtol=1e-14)
    log_mean_power = logsumexp(log_weights + t * v) - np.log(y.size)
    standardized = t * v - log_mean_power
    shape = t / spread
    log_scale = y[-1] + spread * log_mean_power / t
    if not np.all(np.isfinite(standardized)) or not np.isfinite(shape):
        raise ValueError("Weibull fit cannot be represented")
    return standardized, shape, log_scale


def _fitted(rvs):
    return _fit_logs(_logs(rvs))[0]


def _correlation_squared(y, scores):
    return AbstractWeibullGofStatistic._correlation_squared(y, scores)


def _moment_components(rvs):
    return AbstractWeibullGofStatistic._moment_components(-_logs(rvs))


class AbstractWeibullGofStatistic(AbstractGoodnessOfFitStatistic, ABC):
    """Ordinary Weibull with zero location and unknown shape and scale.

    Parameters
    ----------
    parameters : ParameterValues
        Distribution.DEFAULT.parse({}); shape and scale must both be unknown.
    """

    @staticmethod
    @jit(nopython=True, cache=True)
    def _correlation_squared(y, scores):
        y = y - np.mean(y)
        scores = scores - np.mean(scores)
        y = y / np.linalg.norm(y)
        scores = scores / np.linalg.norm(scores)
        return float(min(max(np.dot(y, scores) ** 2, 0.0), 1.0))

    @staticmethod
    @jit(nopython=True, cache=True)
    def _moment_components(y):
        y = (y - np.mean(y)) / np.std(y)
        return len(y), np.mean(y**3), np.mean(y**4)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _sb_components(y):
        n = len(y)
        y = y - np.mean(y)
        w = np.log((n + 1) / (n - np.arange(1, n) + 1))
        wi = np.concatenate((w, np.array([n - np.sum(w)])))
        wn = w * (1 + np.log(w)) - 1
        wn = np.concatenate((wn, np.array([0.4228 * n - np.sum(wn)])))
        b = np.dot(0.6079 * wn - 0.2570 * wi, y) / n
        return y, b

    @staticmethod
    @jit(nopython=True, cache=True)
    def _likelihood_score(t, v, log_weights, mean_v):
        powers = log_weights + t * v
        maximum = np.max(powers)
        weights = np.exp(powers - maximum)
        return np.dot(weights, v) / np.sum(weights) - mean_v - 1 / t

    @staticmethod
    def distribution() -> type[Distribution]:
        return Distribution

    @classmethod
    def code(cls) -> str:
        family_code = f"WEIBULL_{AbstractGoodnessOfFitStatistic.code()}"
        if "short_code" in cls.__abstractmethods__:
            return family_code
        return f"{cls.short_code()}_{family_code}"

    @classmethod
    def _standard_calibration_parameters(cls):
        """Standardize this fitted family using log-location-scale invariance.

        Every replicate must refit shape and scale. Record-based criteria still
        require external sampling schemes via _validate_monte_carlo_calibration.
        """
        return Distribution.DEFAULT.parse({Distribution.SHAPE: 1, Distribution.SCALE: 1})

    def _validate_storage_calibration(self):
        raise ValueError(
            "Weibull statistics were revised: regenerate calibration; "
            "unversioned stored critical values are not supported"
        )


def _weibull_log_probabilities(criterion, sample):
    return _log_probabilities(_fitted(sample))


def _weibull_bound_distribution(criterion, sample):
    _, shape, log_scale = _fit_logs(_logs(sample))
    with np.errstate(over="ignore", under="ignore"):
        scale = np.exp(log_scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Fitted Weibull scale cannot be represented")
    return Distribution.bind(Distribution.DEFAULT.parse({"shape": shape, "scale": scale}))


class AbstractWeibullFrequencyGofStatistic(AbstractWeibullGofStatistic, ABC):
    """Frequency discrepancies from a Weibull fitted by maximum likelihood.

    Shape and scale are re-estimated on every sample. Built-in Monte Carlo
    calibration is available for EqualProbabilityBins, whose cell probabilities
    are invariant under affine transformations of log observations. Other binning
    strategies require external calibration with explicit simulation parameters.
    """

    def _validate_monte_carlo_calibration(self):
        if type(self.binning) is not EqualProbabilityBins:
            raise ValueError(
                "Fitted Weibull with this binning requires external calibration; "
                "only EqualProbabilityBins supports standard Weibull simulation"
            )


def _sb_components(y):
    return AbstractWeibullGofStatistic._sb_components(y)


class WPPWeibullGofStatistic(AbstractWeibullGofStatistic, ABC):
    """Abstract probability-plot family; select a concrete statistic to execute."""

    def alternative(self):
        return RightAlternative()

    @staticmethod
    def MLEst(x):
        """Return ordinary-Weibull MLEs and ascending standardized log data.

        Parameters
        ----------
        x : array_like
            At least two finite positive, nonconstant observations.

        Returns
        -------
        estimates : dict
            ``eta`` (scale), ``beta`` (shape), and ``y`` (standardized logs).
            Internal statistics use log-scale fits to avoid scale overflow.

        Raises
        ------
        ValueError
            Invalid input or a scale not representable as a positive float.
        """
        z, shape, log_scale = _fit_logs(_logs(x))
        with np.errstate(over="ignore", under="ignore"):
            scale = np.exp(log_scale)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Fitted scale cannot be represented")
        return {"eta": float(scale), "beta": float(shape), "y": z}


class ShapiroBrainWeibullGofStatistic(WPPWeibullGofStatistic):
    """Shapiro-Brain Weibull W statistic with centered log observations.

    Parameters
    ----------
    parameters : ParameterValues
        Distribution.DEFAULT.parse({}); shape and scale are both unknown,
        with Weibull location fixed at zero.

    Notes
    -----
    For ascending y_i=log(x_(i))-mean(log(x)), define
    w_i=log((n+1)/(n-i+1)), i<n, and w_n=n-sum(w_i).
    Set v_i=w_i*(1+log(w_i))-1, i<n, and v_n=0.4228*n-sum(v_i).
    Then b=sum((0.6079*v_i-0.2570*w_i)*y_i)/n and
    W=n*b**2/sum(y_i**2). These weights and normalization are reproduced
    in Shell [2], equation (9), attributed there to Shapiro and Brain [1].

    Centering is an explicit library adjustment to the rounded formula:
    0.6079*0.4228-0.2570=0.00002012, so the uncentered b changes with
    log-location. Here b_centered=b_raw-0.00002012*mean(log(x)).
    Consequently W is invariant under x -> c*x**d for c,d>0, up to
    floating-point error. No maximum-likelihood fit is performed.

    Both tails reject, following the two-quantile comparison in [2],
    Section VI. Calibrate this centered implementation by simulation at
    shape=scale=1 with the same sample size. Published rounded-formula
    quantiles are not used as exact critical values.
    Unversioned stored Weibull calibrations remain unsupported.

    References
    ----------
    .. [1] Shapiro, S. S. and Brain, C. W. (1987). W-test for the Weibull
       distribution. Communications in Statistics - Simulation and
       Computation, 16(1), 209-219. https://doi.org/10.1080/03610918708812586
    .. [2] Shell, J. (2003). Parametric Evaluation of Lifetime Data.
       IPN Progress Report 42-155, Section III.B, equation (9), and Section VI.
       https://ipnpr.jpl.nasa.gov/progress_report/42-155/155B.pdf

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({})
    >>> test = ShapiroBrainWeibullGofStatistic(parameters)
    >>> value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
    >>> bool(np.isfinite(value))
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "SB".
        """
        return "SB"

    def alternative(self):
        return TwoSidedAlternative()

    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like
            One-dimensional finite strictly positive observations, n >= 2.
            Constant samples are invalid; ties are allowed unless stated below.
        **kwargs : dict
            Unused common-interface keywords.

        Returns
        -------
        statistic : float
            Scalar discrepancy. Infinite boundary penalties are preserved.

        Raises
        ------
        ValueError
            Invalid sample, unsupported settings, or unrepresentable fit.

        Notes
        -----
        The input is never changed and estimated parameters are not retained.
        """
        y, b = _sb_components(_logs(rvs))
        return self.do_execute_statistic(y, b)

    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(y, b):
        return float(len(y) * b**2 / np.dot(y, y))


class OzturkKorukogluWeibullGofStatistic(WPPWeibullGofStatistic):
    """Ozturk-Korukoglu standardized scale-ratio test for Weibull samples.

    Parameters
    ----------
    parameters : ParameterValues
        Empty Weibull schema; shape and scale are unknown, location is zero.

    Notes
    -----
    This is the standardized OK* statistic [1]_, as reproduced in [2]_,
    Section 3.2.2, equations (3.16)-(3.20), with centered logarithms.
    No maximum-likelihood fit is performed.

    For y_i=log(x_(i))-mean(log(x)), let w_i=log((n+1)/(n-i+1)),
    i<n, w_n=n-sum(w_i). Let v_i=w_i*(1+log(w_i))-1 for i<n and
    v_n=0.4228*n-sum(v_i). Set b=sum((0.6079*v_i-0.2570*w_i)*y_i)/n.
    Let S=sum((2*i-1-n)*y_i)/(log(2)*(n-1)*n).
    Return (b/S-1-0.13/sqrt(n)+1.18/n)/(0.49/sqrt(n)-0.36/n).

    The published rounded coefficients have residual sum per observation
    0.6079*0.4228-0.2570=0.00002012. Centering subtracts this constant
    times mean(log(x)) from the uncentered b, restoring invariance under
    x -> c*x**d for positive c,d. S is unchanged by centering and is
    positive for every nonconstant sample. The normalization denominator
    is positive for n >= 2. At n=2 the statistic is sample-independent.

    Reject in both tails. Calibrate this centered implementation at the
    actual sample size using Weibull samples with shape=scale=1. Empirical
    standardization does not supply an exact standard-normal null law.

    References
    ----------
    .. [1] Ozturk, A. and Korukoglu, S. (1988). A new test for the extreme
       value distribution. Communications in Statistics - Simulation and
       Computation, 17(4), 1375-1393.
       https://doi.org/10.1080/03610918808812730
    .. [2] Krit, M. (2014). Goodness-of-fit tests in reliability: Weibull
       distribution and imperfect maintenance models. PhD thesis,
       Universite de Grenoble, Section 3.2.2, pp. 48-49.
       https://theses.hal.science/tel-01126901/document

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({})
    >>> test = OzturkKorukogluWeibullGofStatistic(parameters)
    >>> value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
    >>> bool(np.isfinite(value))
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "OK".
        """
        return "OK"

    def alternative(self):
        return TwoSidedAlternative()

    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like
            One-dimensional finite strictly positive observations, n >= 2.
            Constant samples are invalid; other ties are allowed.
        **kwargs : dict
            Unused common-interface keywords.

        Returns
        -------
        statistic : float
            Signed standardized scale ratio.

        Raises
        ------
        ValueError
            Invalid sample or zero spread after taking logarithms.

        Notes
        -----
        The input is never changed and estimated parameters are not retained.
        """
        y, b = _sb_components(_logs(rvs))
        return self.do_execute_statistic(y, b)

    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(y, b):
        n = len(y)
        s = np.sum((2 * np.arange(1, n + 1) - 1 - n) * y) / (np.log(2) * (n - 1) * n)
        return float((b / s - 1 - 0.13 / np.sqrt(n) + 1.18 / n) / (0.49 / np.sqrt(n) - 0.36 / n))


class SmithBainWeibullGofStatistic(AbstractWeibullGofStatistic):
    """Smith--Bain mean-rank log-probability-plot correlation discrepancy.

    Parameters
    ----------
    parameters : ParameterValues
        An empty distribution schema; all distribution parameters are unknown.


    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar statistic, fitting parameters internally when needed.
    hypothesis()
        Return the fixed parameters; omitted parameters are unknown.
    alternative()
        Return the critical-region tail.

    Notes
    -----
    F(x)=1-exp(-(x/eta)**k), x > 0, with location fixed at zero
    and unknown positive scale eta and shape k.
    Pass Distribution.DEFAULT.parse({}) to declare both parameters unknown.
    Logarithmic location and scale are eliminated within every call.

    T=n*(1-r**2), where r=corr(log(x_(i)), log(-log(1-i/(n+1)))).

    Reject for right tail values.
    Calibrate with repeated execute_statistic calls on ordinary Weibull
    samples. Affine invariance in log(x) permits eta=k=1 for simulation.
    Ordinary fixed-parameter KS tables do not apply to fitted CDFs.
    Previously stored unversioned Weibull calibrations must be regenerated.
    The formula and mean ranks agree with Krit [2], Section 3.2.1,
    equation (3.12), which attributes the statistic to Smith and Bain [1].
    The original article's full text was not verified directly. Krit's
    accompanying phrase "too close to 1" is inconsistent with Z2=n*(1-r**2):
    poor linear fit means large Z2, so the implementation uses the right tail.
    No chi-square or standard-normal null approximation is asserted.

    Under X -> c*X**d, c,d > 0, log(X) -> log(c)+d*log(X), so r**2 and
    T are unchanged. No parameter estimation is needed. For n=2 every
    nonconstant sample gives T=0 in exact arithmetic; use n>=3 for a
    nondegenerate test. Ties are accepted, but calibration for rounded data
    must reproduce the rounding. Only complete, uncensored samples are supported.

    References
    ----------
    .. [1] Smith, R. M. and Bain, L. J. (1976). Correlation type goodness-of-fit
       statistics with censored sampling. Communications in Statistics --
       Theory and Methods, 5(2), 119--132.
       https://doi.org/10.1080/03610927608827337
    .. [2] Krit, M. (2014). Goodness-of-fit tests in reliability: Weibull
       distribution and imperfect maintenance models. PhD thesis,
       Universite de Grenoble, Section 3.2.1, equation (3.12), p. 47.
       https://theses.hal.science/tel-01126901/document
    .. [3] Krit, M. (2014). Goodness-of-fit tests for the Weibull distribution
       based on the Laplace transform. Journal de la Societe Francaise de
       Statistique, 155(3), 135--151, Section 2, equations (3), (8).
       https://numdam.org/item/JSFS_2014__155_3_135_0.pdf

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({})
    >>> test = SmithBainWeibullGofStatistic(parameters)
    >>> value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
    >>> bool(np.isfinite(value))
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "RSB".
        """
        return "RSB"

    def alternative(self):
        return RightAlternative()

    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like
            One-dimensional finite strictly positive observations, n >= 2.
            Constant samples are invalid; ties are allowed unless stated below.
        **kwargs : dict
            Unused common-interface keywords.

        Returns
        -------
        statistic : float
            Finite discrepancy in [0, n], up to floating-point rounding.

        Raises
        ------
        ValueError
            Invalid sample, unsupported settings, or unrepresentable fit.

        Notes
        -----
        The input is never changed and estimated parameters are not retained.
        """
        y = _logs(rvs)
        n = len(y)
        scores = np.log(-np.log1p(-np.arange(1, n + 1) / (n + 1)))
        return float(n * (1 - _correlation_squared(y, scores)))


class EvansJohnsonGreenWeibullGofStatistic(WPPWeibullGofStatistic):
    """Evans--Johnson--Green squared log-probability-plot correlation.

    Parameters
    ----------
    parameters : ParameterValues
        An empty distribution schema; all distribution parameters are unknown.


    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar statistic, fitting parameters internally when needed.
    hypothesis()
        Return the fixed parameters; omitted parameters are unknown.
    alternative()
        Return the critical-region tail.

    Notes
    -----
    F(x)=1-exp(-(x/eta)**k), x > 0, with location fixed at zero
    and unknown positive scale eta and shape k.
    Pass Distribution.DEFAULT.parse({}) to declare both parameters unknown.
    Logarithmic location and scale are eliminated within every call.

    R2=corr(log(x_(i)), log(-log(1-p_i)))**2,
    p_i=(i-0.3175)/(n+0.365). No fitted shape is needed: it cancels.

    Reject for left tail values.
    Calibrate with repeated execute_statistic calls on ordinary Weibull
    samples. Affine invariance in log(x) permits eta=k=1 for simulation.
    This agrees with Krit [2], Section 3.2.1, equation (3.13), attributed
    there to Evans, Johnson and Green [1]. In Krit's notation the scores
    are M_i=log(-log(1-p_i))/shape_hat. Their common positive scale cancels
    from the correlation, and centering M_i in the numerator is equivalent
    because sum(log(x_(i))-mean(log(x)))=0. No MLE is performed.
    The log-scale statistic must be distinguished from the original-scale
    correlation statistic also discussed in [1]. The exact rank formula
    was checked through [2]; the equations in [1] were not directly readable.
    Filliben [3] provides the historical probability-plot context, not
    critical values for this Weibull implementation.

    The result is R2, not R4 or n*(1-R2). Reject for small R2; calibrate
    the actual sample size and rank convention. No chi-square null law
    is assumed. The former implementation returned R4, so its stored
    critical values cannot be reused unchanged.
    Under X -> a*X**b, a,b > 0, log(X) -> log(a)+b*log(X), proving
    invariance. At n=2 every nonconstant sample gives R2=1 in exact
    arithmetic: use n>=3 for a nondegenerate test. Ties are accepted,
    but rounded-data calibration must reproduce the rounding mechanism.
    Only complete, uncensored samples with location fixed at zero are supported.

    References
    ----------
    .. [1] Evans, J. W., Johnson, R. A. and Green, D. W. (1989).
       Two- and three-parameter Weibull goodness-of-fit tests.
       USDA Forest Service, Research Paper FPL-RP-493, p. 3, equation (3).
       https://www.govinfo.gov/content/pkg/GOVPUB-A13-PURL-gpo123139/pdf/GOVPUB-A13-PURL-gpo123139.pdf
    .. [2] Krit, M. (2014). Goodness-of-fit tests in reliability: Weibull
       distribution and imperfect maintenance models. PhD thesis,
       Universite de Grenoble, Section 3.2.1, equation (3.13), p. 47.
       https://theses.hal.science/tel-01126901/document
    .. [3] Filliben, J. J. (1975). The probability plot correlation
       coefficient test for normality. Technometrics, 17(1), 111--117.
       https://doi.org/10.1080/00401706.1975.10489279

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({})
    >>> test = EvansJohnsonGreenWeibullGofStatistic(parameters)
    >>> value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
    >>> bool(np.isfinite(value))
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "REJG".
        """
        return "REJG"

    def alternative(self):
        return LeftAlternative()

    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like
            One-dimensional finite strictly positive observations, n >= 2.
            Constant samples are invalid; ties are allowed unless stated below.
        **kwargs : dict
            Unused common-interface keywords.

        Returns
        -------
        statistic : float
            Squared correlation in [0, 1], up to floating-point rounding.

        Raises
        ------
        ValueError
            Invalid sample, unsupported settings, or unrepresentable fit.

        Notes
        -----
        The input is never changed and estimated parameters are not retained.
        """
        y = _logs(rvs)
        n = len(y)
        p = (np.arange(1, n + 1) - 0.3175) / (n + 0.365)
        return _correlation_squared(y, np.log(-np.log1p(-p)))


class ColesWeibullGofStatistic(WPPWeibullGofStatistic):
    """Coles stabilized probability-plot distance with Blom estimates.

    Parameters
    ----------
    parameters : ParameterValues
        An empty distribution schema; all distribution parameters are unknown.


    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar statistic, fitting parameters internally when needed.
    hypothesis()
        Return the fixed parameters; omitted parameters are unknown.
    alternative()
        Return the critical-region tail.

    Notes
    -----
    F(x)=1-exp(-(x/eta)**k), x > 0, with location fixed at zero
    and unknown positive scale eta and shape k.
    Pass Distribution.DEFAULT.parse({}) to declare both parameters unknown.
    Blom log-location and log-scale estimates are recomputed on every call.

    T=max(abs(2/pi*asin(sqrt((i-0.5)/n))-2/pi*asin(sqrt(G(z_i))))),
    where z_i=(log(x_(i))-mu_B)/sigma_B and G(z)=1-exp(-exp(z)).

    Reject for right tail values.
    Calibrate with repeated execute_statistic calls on ordinary Weibull
    samples. Affine invariance in log(x) permits eta=k=1 for simulation.
    Re-estimate with Blom on every simulation replicate. Critical values
    from the former MLE implementation must be regenerated.

    The SPP distance follows Krit [2], (3.14)-(3.15), and Coles [1].
    Blom weights follow EWGoF::BLOMEst [3], using the library's fourth-order
    Taylor approximation to expected minimum-Gumbel order statistics m_i.
    Let p_i=i/(n+1), theta_i=-(1-p_i)*log(1-p_i), with zero endpoints.
    Set A_j=(theta_j-theta_(j+1), theta_j*m_j-theta_(j+1)*m_(j+1))
    and b_j=theta_j*y_j-theta_(j+1)*y_(j+1), j=0,...,n, y_i=log(x_(i)).
    Solve min ||A*(mu_B,sigma_B)-b||^2, then standardize the logs.
    This is algebraically equivalent to the linear weights in [3].
    Log observations are shifted and rescaled first for numerical stability.
    The GoFNS expansion uses floating-point 1/3 and 1/8 coefficients;
    the EWGoF C++ source uses integer division at these two coefficients.
    Thus this is an explicit approximation convention, not bitwise
    reproduction of EWGoF or a verified reproduction of published tables.
    EWGoF's WPP.test SPP branch itself uses MLE; only its BLOMEst estimator
    is the reference here. Coles' abstract confirms the use of Blom;
    the full original article was not available for direct verification.

    Under X -> a*X**b, a,b>0, the linear fit transforms equivariantly,
    leaving standardized logs and T unchanged. At n=2, T is constant
    over nonconstant samples: use n>=3 for a nondegenerate test.
    Only complete zero-location Weibull samples are supported. Ties are
    accepted, but calibration of rounded data must reproduce rounding.
    Reject nonpositive fitted log-scale. No chi-square or normal null law
    is assumed; this is the maximum distance, not Coles' correlation R_sp.

    References
    ----------
    .. [1] Coles, S. G. (1989). On goodness-of-fit tests for the two-parameter
       Weibull distribution derived from the stabilized probability plot.
       Biometrika, 76(3), 593--598. https://doi.org/10.1093/biomet/76.3.593
    .. [2] Krit, M. (2014). Goodness-of-fit tests in reliability: Weibull
       distribution and imperfect maintenance models. PhD thesis,
       Section 3.2.1, (3.14)-(3.15), p. 48.
       https://theses.hal.science/tel-01126901/document
    .. [3] Krit, M. EWGoF, BLOMEst and GoFNS source code.
       https://github.com/cran/EWGoF/blob/master/R/BLOMEst.R
       https://github.com/cran/EWGoF/blob/master/src/GoFNS.cpp
    .. [4] Michael, J. R. (1983). The stabilized probability plot.
       Biometrika, 70(1), 11--17. https://doi.org/10.1093/biomet/70.1.11

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({})
    >>> test = ColesWeibullGofStatistic(parameters)
    >>> value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
    >>> bool(np.isfinite(value))
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "SPP".
        """
        return "SPP"

    def alternative(self):
        return RightAlternative()

    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like
            One-dimensional finite strictly positive observations, n >= 2.
            Constant samples are invalid; ties are allowed unless stated below.
        **kwargs : dict
            Unused common-interface keywords.

        Returns
        -------
        statistic : float
            Finite maximum stabilized distance in [0, 1].

        Raises
        ------
        ValueError
            Invalid sample, unsupported settings, or unrepresentable fit.

        Notes
        -----
        The input is never changed and estimated parameters are not retained.
        """
        y = _logs(rvs)
        n = len(y)
        # Affine normalization avoids cancellation for large log-location.
        v = (y - y[0]) / np.ptp(y)
        means = NormalizeSpaceWeibullGofStatistic.GoFNS(1, n, n)
        probabilities = np.arange(1, n + 1) / (n + 1)
        theta = -(1 - probabilities) * np.log1p(-probabilities)
        # Common signs of the differences do not affect the least-squares fit.
        design = np.column_stack((np.diff(np.pad(theta, 1)), np.diff(np.pad(theta * means, 1))))
        response = np.diff(np.pad(theta * v, 1))
        coefficients, _, rank, _ = np.linalg.lstsq(design, response, rcond=None)
        location, scale = coefficients
        if rank != 2 or not np.all(np.isfinite(coefficients)) or scale <= 0:
            raise ValueError("Blom log-scale must be finite and positive")
        z = (v - location) / scale
        if not np.all(np.isfinite(z)):
            raise ValueError("Blom standardized logs cannot be represented")
        return self.do_execute_statistic(z)

    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(z):
        r = 2 / np.pi * np.arcsin(np.sqrt((np.arange(1, len(z) + 1) - 0.5) / len(z)))
        s = 2 / np.pi * np.arcsin(np.sqrt(-np.expm1(-np.exp(z))))
        return float(np.max(np.abs(r - s)))


class BestRaynerThasSkewnessWeibullGofStatistic(AbstractWeibullGofStatistic):
    """Best--Rayner--Thas skewness component V3 squared for Weibull fit.

    Parameters
    ----------
    parameters : ParameterValues
        Distribution.DEFAULT.parse({}); shape and scale must be unknown.

    Notes
    -----
    For zero-location Weibull X with shape k and scale eta, Y=-log(X)
    is maximum Gumbel with location -log(eta) and scale 1/k.
    Let m_j=mean((Y-mean(Y))**j), g_n=m_3/m_2**1.5 and b_n=m_4/m_2**2.
    All moments use divisor n; b_n is Pearson kurtosis, not excess.
    The returned squared smooth component is n*(g_n-1.139547)**2/20.

    This implements the rounded coefficients in EWGoF 2.2.2 WPP.test,
    also reproduced by Krit (2014), section 3.2.5, equations (3.33)--(3.36).
    Moment standardization eliminates the unknown parameters; no MLE is used.
    The ST1 code follows EWGoF's executable source. Its help page reverses
    the ST1/ST2 labels: the source uses ST1 for V3 squared (skewness) and
    ST2 for V4 squared (kurtosis). ST1WeibullGofStatistic remains an alias.

    Large values reject. Calibrate by Monte Carlo at the observed sample
    size, applying this same formula to Weibull shape=scale=1 replicates.
    Invariance under X -> a*X**b for a,b>0 removes the nuisance parameters.
    At n=2 the statistic is constant and has no discriminatory power.
    Do not use an unqualified chi-square(1) approximation: these denominators
    are smooth-polynomial normalizations, not delta-method variances after
    moment estimation. Rounded coefficients are part of this implementation.
    Earlier local delta-method formulas have been replaced; regenerate all
    critical values. Unversioned stored calibration remains blocked.

    References
    ----------
    Best, D. J., Rayner, J. C. W., and Thas, O. (2007).
    Comparison of Five Tests of Fit for the Extreme Value Distribution.
    Journal of Statistical Theory and Practice, 1(1), 89--99.
    https://doi.org/10.1080/15598608.2007.10411826
    For Table 1 (Port Pirie), X=exp(-Y) gives 0.5966736536522815,
    whereas section 4 reports 0.71. Sample skewness is 0.7110707539472332;
    this coincidence does not prove a typo. The discrepancy is unresolved.
    The accessible author PDF omits pages 92--95 and the appendices.
    Exact coefficient verification therefore uses these accessible sources:
    https://github.com/cran/EWGoF/blob/51ef071b72a340b31807dacfa0e8f1d4a5a5eeb1/R/WPP.test.R
    https://theses.hal.science/tel-01126901/document

    Examples
    --------
    >>> test = BestRaynerThasSkewnessWeibullGofStatistic(Distribution.DEFAULT.parse({}))
    >>> bool(np.isfinite(test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])))
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @staticmethod
    @override
    def short_code():
        """Return the historical EWGoF source identifier ST1."""
        return "ST1"

    def alternative(self):
        return RightAlternative()

    def execute_statistic(self, rvs, **kwargs):
        """Compute V3 squared from standardized negative logarithms.

        Parameters
        ----------
        rvs : array_like
            Finite, strictly positive, nonconstant one-dimensional sample,
            with n >= 2. Ties are allowed. The input is not modified.
        **kwargs : dict
            Unused common-interface keywords.

        Returns
        -------
        float
            Nonnegative squared skewness component.

        Raises
        ------
        ValueError
            Invalid sample or logarithms indistinguishable at float64 precision.
        """
        n, skew, _kurt = _moment_components(rvs)
        return float(n * (skew - 1.139547) ** 2 / 20.0)


class BestRaynerThasKurtosisWeibullGofStatistic(AbstractWeibullGofStatistic):
    """Best--Rayner--Thas kurtosis component V4 squared for Weibull fit.

    Parameters
    ----------
    parameters : ParameterValues
        Distribution.DEFAULT.parse({}); shape and scale must be unknown.

    Notes
    -----
    For zero-location Weibull X with shape k and scale eta, Y=-log(X)
    is maximum Gumbel with location -log(eta) and scale 1/k.
    Let m_j=mean((Y-mean(Y))**j), g_n=m_3/m_2**1.5 and b_n=m_4/m_2**2.
    All moments use divisor n; b_n is Pearson kurtosis, not excess.
    The returned squared smooth component is n*(b_n-7.55*g_n+3.21)**2/219.72.
    In particular, the fourth component includes a skewness correction.

    This implements the rounded coefficients in EWGoF 2.2.2 WPP.test,
    also reproduced by Krit (2014), section 3.2.5, equations (3.33)--(3.36).
    Moment standardization eliminates the unknown parameters; no MLE is used.
    The ST2 code follows EWGoF's executable source. Its help page reverses
    the ST1/ST2 labels: the source uses ST1 for V3 squared (skewness) and
    ST2 for V4 squared (kurtosis). ST2WeibullGofStatistic remains an alias.

    Large values reject. Calibrate by Monte Carlo at the observed sample
    size, applying this same formula to Weibull shape=scale=1 replicates.
    Invariance under X -> a*X**b for a,b>0 removes the nuisance parameters.
    At n=2 the statistic is constant and has no discriminatory power.
    Do not use an unqualified chi-square(1) approximation: these denominators
    are smooth-polynomial normalizations, not delta-method variances after
    moment estimation. Rounded coefficients are part of this implementation.
    Earlier local delta-method formulas have been replaced; regenerate all
    critical values. Unversioned stored calibration remains blocked.

    References
    ----------
    Best, D. J., Rayner, J. C. W., and Thas, O. (2007).
    Comparison of Five Tests of Fit for the Extreme Value Distribution.
    Journal of Statistical Theory and Practice, 1(1), 89--99.
    https://doi.org/10.1080/15598608.2007.10411826
    For Table 1 (Port Pirie), X=exp(-Y) gives 0.37580214526699485,
    whereas section 4 reports 0.36. This discrepancy is unresolved.
    The accessible author PDF omits pages 92--95 and the appendices.
    Exact coefficient verification therefore uses these accessible sources:
    https://github.com/cran/EWGoF/blob/51ef071b72a340b31807dacfa0e8f1d4a5a5eeb1/R/WPP.test.R
    https://theses.hal.science/tel-01126901/document

    Examples
    --------
    >>> test = BestRaynerThasKurtosisWeibullGofStatistic(Distribution.DEFAULT.parse({}))
    >>> bool(np.isfinite(test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])))
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @staticmethod
    @override
    def short_code():
        """Return the historical EWGoF source identifier ST2."""
        return "ST2"

    def alternative(self):
        return RightAlternative()

    def execute_statistic(self, rvs, **kwargs):
        """Compute V4 squared from standardized negative logarithms.

        Parameters
        ----------
        rvs : array_like
            Finite, strictly positive, nonconstant one-dimensional sample,
            with n >= 2. Ties are allowed. The input is not modified.
        **kwargs : dict
            Unused common-interface keywords.

        Returns
        -------
        float
            Nonnegative squared kurtosis component.

        Raises
        ------
        ValueError
            Invalid sample or logarithms indistinguishable at float64 precision.
        """
        n, skew, kurt = _moment_components(rvs)
        return float(n * (kurt - 7.55 * skew + 3.21) ** 2 / 219.72)


# Historical EWGoF source names, retained for imports and serialized class references.
ST1WeibullGofStatistic = BestRaynerThasSkewnessWeibullGofStatistic
ST2WeibullGofStatistic = BestRaynerThasKurtosisWeibullGofStatistic


class NormalizeSpaceWeibullGofStatistic(AbstractWeibullGofStatistic, ABC):
    """Abstract complete-sample normalized-spacing family (no censoring API)."""

    @staticmethod
    @jit(nopython=True, cache=True)
    def GoFNS(t, n, m):
        """
        Compute normalized spacing coefficients.

        :param t: starting index offset.
        :param n: total sample size including censoring.
        :param m: number of observed failures.
        :return: array of normalized spacing coefficients.
        """
        res = np.zeros(m)
        for i in range(m):
            p_r = (t + i) / (n + 1)
            q_r = 1 - p_r
            Q_r = np.log(-np.log(1 - p_r))
            d_Q_r = -1 / ((1 - p_r) * np.log(1 - p_r))
            d2_Q_r = d_Q_r / q_r - d_Q_r * d_Q_r
            d3_Q_r = d2_Q_r / q_r + d_Q_r / (q_r * q_r) - 2 * (d_Q_r * d2_Q_r)
            d4_Q_r = d3_Q_r / q_r + 2 * d2_Q_r / (q_r * q_r)
            d4_Q_r += 2 * d_Q_r / (q_r * q_r * q_r) - 2 * (d2_Q_r * d2_Q_r + d_Q_r * d3_Q_r)
            res[i] = Q_r + (p_r * q_r / (2 * (n + 2))) * d2_Q_r
            res[i] += (
                q_r
                * p_r
                / ((n + 2) * (n + 2))
                * (1 / 3 * (q_r - p_r) * d3_Q_r + 1 / 8 * p_r * q_r * d4_Q_r)
            )
        return res

    def alternative(self):
        return RightAlternative()

    def _spacings(self, rvs):
        y = _logs(rvs, min_size=3)
        means = self.GoFNS(1, len(y), len(y))
        return self._normalize_spacings(y, means)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _normalize_spacings(y, means):
        intervals = np.diff(means)
        if not np.all(np.isfinite(intervals)) or np.any(intervals <= 0):
            raise ValueError("Approximate expected log spacings must be positive")
        gaps = np.diff(y) / intervals
        return gaps / np.sum(gaps)


class TikuSinghWeibullGofStatistic(NormalizeSpaceWeibullGofStatistic):
    """Tiku--Singh normalized log-spacing statistic for complete samples.

    Parameters
    ----------
    parameters : ParameterValues
        Use Distribution.DEFAULT.parse({}): zero-location Weibull with
        unknown positive shape k and scale eta.

    Notes
    -----
    For sorted observations y_i=log(x_(i)), define
    h_i=(y_(i+1)-y_i)/(m_(i+1)-m_i), i=1,...,n-1, and g_i=h_i/sum(h).
    Return T=2*sum((n-1-i)*g_i, i=1,...,n-2)/(n-2).
    This is the generalized-spacing Z* functional in Surucu (2008),
    equations (3.4)--(3.5), applied to the minimum-Gumbel log-Weibull family.
    GoFNS approximates m_i=E[log(E_(i))], E_j iid Exp(1), using Q(p)=
    log(-log(1-p)), p=i/(n+1), q=1-p, and the expansion
    Q(p)+p*q*Q''(p)/(2*(n+2))
    +p*q*((q-p)*Q3(p)/3+p*q*Q4(p)/8)/(n+2)**2.
    Here Q3 and Q4 denote the third and fourth derivatives. The expansion
    truncates at the fourth derivative and uses approximate higher
    beta central moments; it is not exact integration of order statistics.

    No parameters are fitted. Under X -> a*X**b (a,b>0), every h_i is
    multiplied by b and g_i is unchanged. Thus the null distribution is
    independent of Weibull shape and scale. Both small and large T reject;
    calibrate both tails at the observed n using this implementation on
    Weibull shape=scale=1 replicates. Symmetry about 1 is not assumed.
    In exact arithmetic 0 <= T <= 2; distinct observations give 0 < T < 2.
    Ties are accepted computationally, but continuous-null calibration
    does not model rounding. Require n >= 3 and a nonconstant positive
    finite one-dimensional sample. Censoring is not supported.

    The original Tiku--Singh full text was not checked directly. Krit
    (2014), equation (30), uses interval indices 2,...,n, inconsistent
    with its preceding definition at i,i+1. With E_i=h_(i-1), its TS
    expression agrees with the functional above. Historical tables need
    not match the explicit expectation approximation used here; simulate
    fresh critical values. No chi-square limit is asserted.

    References
    ----------
    Tiku, M. L. (1980). Goodness of fit statistics based on the spacings
    of complete or censored samples. Australian Journal of Statistics,
    22(3), 260--275. https://doi.org/10.1111/j.1467-842X.1980.tb01175.x
    Tiku, M. L., and Singh, M. (1981). Testing the two parameter Weibull
    distribution. Communications in Statistics - Theory and Methods,
    10(9), 907--918. https://doi.org/10.1080/03610928108828082
    Surucu, B. (2008). A power comparison and simulation study of
    goodness-of-fit tests. Computers & Mathematics with Applications,
    56(6), 1617--1625. https://doi.org/10.1016/j.camwa.2008.03.010
    Krit, M. (2014). Goodness-of-fit tests for the Weibull distribution
    based on the Laplace transform. Journal de la Societe Francaise de
    Statistique, 155(3), 135--151, equation (30).
    https://numdam.org/item/JSFS_2014__155_3_135_0.pdf

    Examples
    --------
    >>> test = TikuSinghWeibullGofStatistic(Distribution.DEFAULT.parse({}))
    >>> bool(np.isfinite(test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])))
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "TS".
        """
        return "TS"

    def alternative(self):
        return TwoSidedAlternative()

    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like
            One-dimensional finite strictly positive observations, n >= 3.
            Constant samples are invalid; ties are allowed.
        **kwargs : dict
            Unused common-interface keywords.

        Returns
        -------
        statistic : float
            Normalized spacing statistic in [0, 2], up to rounding error.

        Raises
        ------
        ValueError
            Invalid sample, indistinguishable logarithms, or nonpositive
            approximate expected spacing.

        Notes
        -----
        The input is never changed; no parameters are fitted.
        """
        g = self._spacings(rvs)
        return self.do_execute_statistic(g)

    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(g):
        n = len(g) + 1
        return float(2 * np.sum(np.arange(n - 2, 0, -1) * g[:-1]) / (n - 2))


class LockhartOReillyStephensWeibullGofStatistic(NormalizeSpaceWeibullGofStatistic):
    """Lockhart--O'Reilly--Stephens spacing Anderson--Darling statistic.

    Parameters
    ----------
    parameters : ParameterValues
        Use Distribution.DEFAULT.parse({}): zero-location Weibull with
        unknown positive shape k and scale eta, F(x)=1-exp(-(x/eta)**k).

    Notes
    -----
    For sorted log observations y_i=log(x_(i)), set
    h_i=(y_(i+1)-y_i)/(m_(i+1)-m_i), i=1,...,n-1, and g_i=h_i/sum(h).
    Here m_i approximates E[log(E_(i))] for iid unit exponential E_j.
    For q=n-2, z_j=sum(g_i, i=1,...,j), j=1,...,q, return
    A2=-q-sum((2*j-1)*(log(z_j)+log(1-z_(q+1-j))), j=1,...,q)/q.
    This is equations (1)--(3) of Lockhart, O'Reilly and Stephens
    (1986, JRSS B), pp. 344--345, with their k=1 and r=n-2 for a
    complete sample. The final cumulative value 1 is excluded.

    GoFNS uses p=i/(n+1), q0=1-p, Q(p)=log(-log(1-p)) and
    m_i approximately Q(p)+p*q0*Q2(p)/(2*(n+2))
    +p*q0*((q0-p)*Q3(p)/3+p*q0*Q4(p)/8)/(n+2)**2,
    where Qj denotes the jth derivative. This fourth-derivative expansion
    also approximates higher beta central moments. It replaces the exact
    expected order statistics in the paper; no exact-table agreement is claimed.

    The statistic is nonnegative in exact arithmetic and has no finite
    upper bound. Ties at either end give positive infinity; interior ties
    are allowed and can give a finite statistic. Require finite, positive,
    nonconstant one-dimensional observations with n >= 3. No censoring API
    is provided, although the papers also treat censored samples.

    Large values reject (RightAlternative). No parameters are fitted:
    X -> a*X**b, a,b>0, multiplies all h_i by b and leaves z_j unchanged.
    Calibrate at the observed n using this implementation on complete
    Weibull shape=scale=1 replicates. The transformed z_j are not uniform
    order statistics under the Weibull null (JRSS B, p. 345). Ordinary
    uniform Anderson--Darling tables therefore do not apply, nor is a
    chi-square limit asserted. Continuous-null calibration does not model
    rounding-induced ties. Reverse cumulative sums evaluate 1-z_j without
    subtracting a rounded z_j from 1.

    References
    ----------
    Lockhart, R. A., O'Reilly, F. J., and Stephens, M. A. (1986).
    Tests of Fit Based on Normalized Spacings. Journal of the Royal
    Statistical Society, Series B, 48(3), 344--352, equations (1)--(3).
    https://doi.org/10.1111/j.2517-6161.1986.tb01418.x
    Author copy checked directly:
    https://www.sfu.ca/~lockhart/Research/Papers/LockhartOReillyStephensJRSSB86.pdf
    Lockhart, R. A., O'Reilly, F. J., and Stephens, M. A. (1986).
    Tests for the extreme value and Weibull distributions based on
    normalized spacings. Naval Research Logistics Quarterly, 33(3), 413--421.
    https://doi.org/10.1002/nav.3800330307
    The companion article covers Weibull critical values and power;
    its full text was not used for direct formula verification.

    Examples
    --------
    >>> test = LockhartOReillyStephensWeibullGofStatistic(Distribution.DEFAULT.parse({}))
    >>> bool(np.isfinite(test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])))
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "LOS".
        """
        return "LOS"

    def alternative(self):
        return RightAlternative()

    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like
            One-dimensional finite strictly positive observations, n >= 3.
            Constant samples are invalid; ties are allowed.
        **kwargs : dict
            Unused common-interface keywords.

        Returns
        -------
        statistic : float
            Nonnegative spacing AD statistic, or infinity for endpoint ties.

        Raises
        ------
        ValueError
            Invalid sample, indistinguishable logarithms, or nonpositive
            approximate expected spacing.

        Notes
        -----
        The input is never changed; no parameters are fitted.
        """
        g = self._spacings(rvs)
        return self.do_execute_statistic(g)

    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(g):
        z = np.cumsum(g)[:-1]
        sf = np.cumsum(g[::-1])[:-1][::-1]
        if np.any(z == 0) or np.any(sf == 0):
            return float("inf")
        q = len(z)
        return float(
            -q - np.sum((2 * np.arange(1, q + 1) - 1) * (np.log(z) + np.log(sf[::-1]))) / q
        )


class MannScheuerFertigWeibullGofStatistic(NormalizeSpaceWeibullGofStatistic):
    """Mann--Scheuer--Fertig upper log-spacing fraction (complete samples).

    Parameters
    ----------
    parameters : ParameterValues
        Use Distribution.DEFAULT.parse({}): zero-location Weibull with
        unknown positive shape and scale.

    Notes
    -----
    For y_i=log(x_(i)), i=1,...,n, define
    h_i=(y_(i+1)-y_i)/(m_(i+1)-m_i), i=1,...,n-1,
    where m_i approximates E[log(E_(i))] for iid unit exponential E_j.
    With r=floor(n/2), return S=sum(h_i, i=r+1,...,n-1)/sum(h_i).
    This matches Krit (2014), equation (29), and Surucu (2008), (4.13),
    attributed to Mann, Scheuer and Fertig (1973). The original 1973
    full text was not checked directly.

    GoFNS uses p=i/(n+1), q=1-p, Q(p)=log(-log(1-p)) and
    m_i approximately Q(p)+p*q*Q2(p)/(2*(n+2))
    +p*q*((q-p)*Q3(p)/3+p*q*Q4(p)/8)/(n+2)**2,
    where Qj denotes the jth derivative. This fourth-derivative expansion
    also approximates the higher beta central moments; it is not an exact
    order-statistic expectation. Its convention is part of the statistic.

    Require n >= 3 and finite, positive, nonconstant one-dimensional data.
    The first r intervals belong only to the denominator; the numerator
    has n-1-r intervals, including the last one. In exact arithmetic,
    0 <= S <= 1, with strict bounds for distinct observations. Ties are
    accepted computationally, but continuous-null calibration does not
    model rounding. Censoring is not supported.

    No parameters are fitted. Under X -> a*X**b for a,b>0, h_i -> b*h_i,
    so S is invariant. Calibrate at the observed n by Monte Carlo using
    this implementation on Weibull shape=scale=1 samples. Large S rejects
    (RightAlternative); this is a one-sided version, not an omnibus
    two-tail rule, and may have poor power for alternatives giving small S.
    Do not substitute historical tables or an exact beta/F/chi-square law
    for calibration of these approximate expected spacings.

    References
    ----------
    Mann, N. R., Scheuer, E. M., and Fertig, K. W. (1973).
    A new goodness-of-fit test for the two-parameter Weibull or extreme-value
    distribution with unknown parameters. Communications in Statistics,
    2(5), 383--400. https://doi.org/10.1080/03610927308827085
    Surucu, B. (2008). A power comparison and simulation study of
    goodness-of-fit tests. Computers & Mathematics with Applications,
    56(6), 1617--1625. https://doi.org/10.1016/j.camwa.2008.03.010
    Krit, M. (2014). Goodness-of-fit tests for the Weibull distribution
    based on the Laplace transform. Journal de la Societe Francaise de
    Statistique, 155(3), 135--151, equation (29).
    https://numdam.org/item/JSFS_2014__155_3_135_0.pdf

    Examples
    --------
    >>> test = MannScheuerFertigWeibullGofStatistic(Distribution.DEFAULT.parse({}))
    >>> bool(np.isfinite(test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])))
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "MSF".
        """
        return "MSF"

    def alternative(self):
        return RightAlternative()

    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like
            One-dimensional finite strictly positive observations, n >= 3.
            Constant samples are invalid; ties are allowed.
        **kwargs : dict
            Unused common-interface keywords.

        Returns
        -------
        statistic : float
            Upper spacing fraction in [0, 1], up to rounding error.

        Raises
        ------
        ValueError
            Invalid sample, indistinguishable logarithms, or nonpositive
            approximate expected spacing.

        Notes
        -----
        The input is never changed; no parameters are fitted.
        """
        g = self._spacings(rvs)
        return self.do_execute_statistic(g)

    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(g):
        return float(np.sum(g[(len(g) + 1) // 2 :]))


class KullbackLeiblerWeibullGofStatistic(AbstractWeibullGofStatistic):
    """Vasicek-entropy KL test for log-Weibull, using maximum likelihood.

    Parameters
    ----------
    parameters : ParameterValues
        Use Distribution.DEFAULT.parse({}): zero-location Weibull with
        unknown positive shape k and scale eta, F(x)=1-exp(-(x/eta)**k).

    Notes
    -----
    Y=log(X) is minimum Gumbel with location log(eta) and scale 1/k.
    Fit both parameters by MLE on every call and set
    z_i=k_hat*(log(x_(i))-log(eta_hat)). Its null density is
    g(z)=exp(z-exp(z)), not the maximum-Gumbel density.
    Define the Vasicek entropy estimate with repeated endpoint values:
    H_m=mean(log(n*(z_(min(n,i+m))-z_(max(1,i-m)))/(2*m))).
    Return I_m=-H_m-mean(z)+mean(exp(z)). This estimates
    D(f||g)=-H(f)-E_f[log(g)], without multiplying by n or exponentiating.
    At the MLE, mean(exp(z))=1 up to numerical error.
    The finite-sample estimate need not be nonnegative; it is not clipped.

    The functional matches the KL branch of EWGoF 2.2.2 WEDF.test with
    funEstimate='MLE' and the same effective window m. Its MLEst returns
    the same minimum-Gumbel standardized orientation despite initially
    taking -log(X). Our root solver differs from EWGoF's bounded optimizer.
    The Gumbel test is attributed to Perez-Rodriguez, Vaquera-Huerta and
    Villasenor-Alva (2009); their full text was not checked directly.

    Require an integer 1 <= m < n/2 (booleans are invalid). The local
    default is min(floor(sqrt(n)),floor((n-1)/2)), whereas EWGoF uses a
    piecewise sample-size rule and overwrites paramKL in its wrapper.
    For example, n=20 gives m=4 here and m=3 there. Do not assume equality
    with EWGoF's default output or historical critical values.

    Large values reject. Positive affine transformations of log samples
    leave fitted z and I_m unchanged. Calibrate at the observed n using
    Weibull shape=scale=1 simulations, refitting both parameters each time
    and preserving m. Custom m requires matching external calibration;
    the generic resolver uses default execution settings. No chi-square
    or unqualified normal limit is asserted.
    Only complete finite positive nonconstant one-dimensional samples,
    n >= 3, are supported. A zero entropy-window gap returns infinity;
    ties with positive window gaps are allowed. The input is not modified.

    References
    ----------
    Perez-Rodriguez, P., Vaquera-Huerta, H., and Villasenor-Alva, J. A.
    (2009). A Goodness-of-Fit Test for the Gumbel Distribution Based on
    Kullback--Leibler Information. Communications in Statistics - Theory
    and Methods, 38(6), 842--855. https://doi.org/10.1080/03610920802316658
    Vasicek, O. (1976). A Test for Normality Based on Sample Entropy.
    Journal of the Royal Statistical Society, Series B, 38(1), 54--59.
    https://doi.org/10.1111/j.2517-6161.1976.tb01566.x
    Song, K. S. (2002). Goodness-of-fit tests based on Kullback-Leibler
    discrimination information. IEEE Transactions on Information Theory,
    48(5), 1103--1117. https://doi.org/10.1109/18.995548
    Executable formula and standardization checked in EWGoF 2.2.2:
    https://github.com/cran/EWGoF/blob/51ef071b72a340b31807dacfa0e8f1d4a5a5eeb1/R/WEDF.test.R
    https://github.com/cran/EWGoF/blob/51ef071b72a340b31807dacfa0e8f1d4a5a5eeb1/R/MLEst.R

    Examples
    --------
    >>> test = KullbackLeiblerWeibullGofStatistic(Distribution.DEFAULT.parse({}))
    >>> bool(np.isfinite(test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])))
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "KL".
        """
        return "KL"

    def alternative(self):
        return RightAlternative()

    def execute_statistic(self, rvs, m=None, **kwargs):
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like
            One-dimensional finite strictly positive observations, n >= 3.
            Constant samples are invalid; ties require positive entropy-window gaps
            for a finite result.
        m : int, optional
            Integer entropy window, 1 <= m < n/2; booleans are invalid.
            Default: min(floor(sqrt(n)), floor((n-1)/2)).
        **kwargs : dict
            Unused common-interface keywords.

        Returns
        -------
        statistic : float
            Unclipped entropy-based KL estimate, possibly negative; infinity
            if an entropy-window gap is zero.

        Raises
        ------
        ValueError
            Invalid sample, unsupported settings, or unrepresentable fit.

        Notes
        -----
        The input is never changed and estimated parameters are not retained.
        """
        z = _fitted(_sample(rvs, min_size=3, positive=True, nonconstant=True))
        n = len(z)
        if m is None:
            m = min(int(np.sqrt(n)), (n - 1) // 2)
        if isinstance(m, (bool, np.bool_)) or not isinstance(m, Integral) or not 1 <= m < n / 2:
            raise ValueError("m must be an integer with 1 <= m < n/2")
        return self.do_execute_statistic(z, m)

    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(z, m):
        n = len(z)
        i = np.arange(n)
        gaps = z[np.minimum(n - 1, i + m)] - z[np.maximum(0, i - m)]
        if np.any(gaps == 0):
            return float("inf")
        h = np.mean(np.log(gaps) + np.log(n / (2 * m)))
        return float(-h - np.mean(z) + np.mean(np.exp(z)))


class LaplaceTransformWeibullGofStatistic(AbstractWeibullGofStatistic, ABC):
    """Abstract fitted-Weibull Laplace family; choose LT2 or LT3."""

    def alternative(self):
        return RightAlternative()

    def _laplace(self, rvs, m, a, kind):
        if isinstance(m, (bool, np.bool_)) or not isinstance(m, Integral) or m < 1:
            raise ValueError("m must be a positive integer")
        a = _scalar(a, "a")
        z = _fitted(rvs)
        if kind == 2:
            t = np.arange(-m, 0) / m
        else:
            # Equation (23): step 1/m, NOT m points on the interval.
            t = np.arange(int(np.ceil(-2.5 * m)), int(np.floor(0.49 * m)) + 1) / m
        return self.do_execute_statistic(z, t, gammaln(1 - t), a)

    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(z, t, log_theoretical, a):
        total = 0.0
        for j in range(len(t)):
            powers = -z * t[j]
            maximum = np.max(powers)
            log_empirical = maximum + np.log(np.sum(np.exp(powers - maximum))) - np.log(len(z))
            difference = abs(log_empirical - log_theoretical[j])
            log_difference = max(log_empirical, log_theoretical[j]) + np.log(-np.expm1(-difference))
            at = a * t[j]
            log_weight = at - np.exp(at) if np.isfinite(at) else -np.inf
            total += np.exp(log_weight + 2 * log_difference)
        return float(len(z) * total)


class LaplaceTransform2WeibullGofStatistic(LaplaceTransformWeibullGofStatistic):
    """Krit LT2 discrete Laplace statistic with maximum-likelihood fits.

    Parameters
    ----------
    parameters : ParameterValues
        An empty distribution schema; all distribution parameters are unknown.


    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar statistic, fitting parameters internally when needed.
    hypothesis()
        Return the fixed parameters; omitted parameters are unknown.
    alternative()
        Return the critical-region tail.

    Notes
    -----
    F(x)=1-exp(-(x/eta)**k), x > 0, with location fixed at zero
    and unknown positive scale eta and shape k.
    Pass Distribution.DEFAULT.parse({}) to declare both parameters unknown.
    Logarithmic location and scale are eliminated within every call.

    T=n*sum(exp(a*t-exp(a*t))*(mean(exp(-t*z))-Gamma(1-t))**2).
    Here z are MLE-standardized logs and t=-1,-1+1/m,...,-1/m.
    This is a discrete sum, with no integration-step multiplier.

    Reject for right tail values.
    Calibrate with repeated execute_statistic calls on ordinary Weibull
    samples. Affine invariance in log(x) permits eta=k=1 for simulation.
    Ordinary fixed-parameter KS tables do not apply to fitted CDFs.
    Previously stored unversioned Weibull calibrations must be regenerated.
    Every simulation must preserve m and a. The generic resolver uses
    the default execution settings only. LT3 at m=100 has 300 grid points.
    Transform differences are evaluated in log space; statistics beyond the
    float64 range return positive infinity.

    References
    ----------
    .. [1] M. Krit (2014), Goodness-of-fit tests for the Weibull distribution based
       on the Laplace transform, sections 2, 4, 5.
       https://numdam.org/item/JSFS_2014__155_3_135_0.pdf

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({})
    >>> test = LaplaceTransform2WeibullGofStatistic(parameters)
    >>> value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
    >>> bool(np.isfinite(value))
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "LT2".
        """
        return "LT2"

    def alternative(self):
        return RightAlternative()

    def execute_statistic(self, rvs, m=100, a=-5, **kwargs):
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like
            One-dimensional finite strictly positive observations, n >= 2.
            Constant samples are invalid; ties are allowed unless stated below.
        m : int, optional
            Positive grid resolution, default 100.
        a : float, optional
            Finite weight parameter, default -5; not a distribution parameter.
        **kwargs : dict
            Unused common-interface keywords.

        Returns
        -------
        statistic : float
            Scalar discrepancy. Infinite boundary penalties are preserved.

        Raises
        ------
        ValueError
            Invalid sample, unsupported settings, or unrepresentable fit.

        Notes
        -----
        The input is never changed and estimated parameters are not retained.
        """
        return self._laplace(rvs, m, a, 2)


class LaplaceTransform3WeibullGofStatistic(LaplaceTransformWeibullGofStatistic):
    """Krit LT3 discrete Laplace statistic with maximum-likelihood fits.

    Parameters
    ----------
    parameters : ParameterValues
        An empty distribution schema; all distribution parameters are unknown.


    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar statistic, fitting parameters internally when needed.
    hypothesis()
        Return the fixed parameters; omitted parameters are unknown.
    alternative()
        Return the critical-region tail.

    Notes
    -----
    F(x)=1-exp(-(x/eta)**k), x > 0, with location fixed at zero
    and unknown positive scale eta and shape k.
    Pass Distribution.DEFAULT.parse({}) to declare both parameters unknown.
    Logarithmic location and scale are eliminated within every call.

    T=n*sum(exp(a*t-exp(a*t))*(mean(exp(-t*z))-Gamma(1-t))**2).
    Here z are MLE-standardized logs and t=j/m, ceil(-2.5*m)<=j<=floor(0.49*m).
    This is a discrete sum, with no integration-step multiplier.

    Reject for right tail values.
    Calibrate with repeated execute_statistic calls on ordinary Weibull
    samples. Affine invariance in log(x) permits eta=k=1 for simulation.
    Ordinary fixed-parameter KS tables do not apply to fitted CDFs.
    Previously stored unversioned Weibull calibrations must be regenerated.
    Every simulation must preserve m and a. The generic resolver uses
    the default execution settings only. LT3 at m=100 has 300 grid points.
    Transform differences are evaluated in log space; statistics beyond the
    float64 range return positive infinity.

    References
    ----------
    .. [1] M. Krit (2014), Goodness-of-fit tests for the Weibull distribution based
       on the Laplace transform, sections 2, 4, 5.
       https://numdam.org/item/JSFS_2014__155_3_135_0.pdf

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({})
    >>> test = LaplaceTransform3WeibullGofStatistic(parameters)
    >>> value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
    >>> bool(np.isfinite(value))
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "LT3".
        """
        return "LT3"

    def alternative(self):
        return RightAlternative()

    def execute_statistic(self, rvs, m=100, a=-5, **kwargs):
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like
            One-dimensional finite strictly positive observations, n >= 2.
            Constant samples are invalid; ties are allowed unless stated below.
        m : int, optional
            Positive grid resolution, default 100.
        a : float, optional
            Finite weight parameter, default -5; not a distribution parameter.
        **kwargs : dict
            Unused common-interface keywords.

        Returns
        -------
        statistic : float
            Scalar discrepancy. Infinite boundary penalties are preserved.

        Raises
        ------
        ValueError
            Invalid sample, unsupported settings, or unrepresentable fit.

        Notes
        -----
        The input is never changed and estimated parameters are not retained.
        """
        return self._laplace(rvs, m, a, 3)


class CabanaQuirozWeibullGofStatistic(AbstractWeibullGofStatistic):
    """Krit maximum-likelihood Cabana-Quiroz CQ* quadratic statistic.

    Parameters
    ----------
    parameters : ParameterValues
        An empty distribution schema; all distribution parameters are unknown.


    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar statistic, fitting parameters internally when needed.
    hypothesis()
        Return the fixed parameters; omitted parameters are unknown.
    alternative()
        Return the critical-region tail.

    Notes
    -----
    F(x)=1-exp(-(x/eta)**k), x > 0, with location fixed at zero
    and unknown positive scale eta and shape k.
    Pass Distribution.DEFAULT.parse({}) to declare both parameters unknown.
    Logarithmic location and scale are eliminated within every call.

    v=sqrt(n)*(mean(exp(-s*z))-Gamma(1-s)), s=(-0.1,0.02).
    Return v.T @ inverse(A) @ v, A=[[1.59,0.91],[0.91,0.53]],
    where z are MLE-standardized log observations.

    Reject for right tail values.
    Calibrate with repeated execute_statistic calls on ordinary Weibull
    samples. Affine invariance in log(x) permits eta=k=1 for simulation.
    Ordinary fixed-parameter KS tables do not apply to fitted CDFs.
    Previously stored unversioned Weibull calibrations must be regenerated.
    This uses the specified nonsingular weighting matrix, not an estimated
    covariance matrix. A chi-square reference law is not asserted.
    Transform values beyond the float64 range give numerical infinity.

    References
    ----------
    .. [1] M. Krit (2014), Goodness-of-fit tests for the Weibull distribution based
       on the Laplace transform, sections 2, 4, 5.
       https://numdam.org/item/JSFS_2014__155_3_135_0.pdf

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({})
    >>> test = CabanaQuirozWeibullGofStatistic(parameters)
    >>> value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
    >>> bool(np.isfinite(value))
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "CQ*".
        """
        return "CQ*"

    def alternative(self):
        return RightAlternative()

    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like
            One-dimensional finite strictly positive observations, n >= 2.
            Constant samples are invalid; ties are allowed unless stated below.
        **kwargs : dict
            Unused common-interface keywords.

        Returns
        -------
        statistic : float
            Scalar discrepancy. Infinite boundary penalties are preserved.

        Raises
        ------
        ValueError
            Invalid sample, unsupported settings, or unrepresentable fit.

        Notes
        -----
        The input is never changed and estimated parameters are not retained.
        """
        z = _fitted(rvs)
        return self.do_execute_statistic(z, gamma(np.array([1.1, 0.98])))

    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(z, theoretical):
        v0 = np.sqrt(len(z)) * (np.mean(np.exp(0.1 * z)) - theoretical[0])
        v1 = np.sqrt(len(z)) * (np.mean(np.exp(-0.02 * z)) - theoretical[1])
        if not np.isfinite(v0) or not np.isfinite(v1):
            return float("inf")
        # Cholesky norm of the fixed weighting matrix.
        l00 = np.sqrt(1.59)
        l10 = 0.91 / l00
        l11 = np.sqrt(0.53 - l10 * l10)
        x0 = v0 / l00
        x1 = (v1 - l10 * x0) / l11
        return float(x0 * x0 + x1 * x1)


class MahdiDoostparastWeibullGofStatistic(AbstractWeibullGofStatistic):
    """Doostparast weighted record EDF distance with record-likelihood fitting.

    Parameters
    ----------
    parameters : ParameterValues
        An empty distribution schema; all distribution parameters are unknown.


    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar statistic, fitting parameters internally when needed.
    hypothesis()
        Return the fixed parameters; omitted parameters are unknown.
    alternative()
        Return the critical-region tail.

    Notes
    -----
    F(x)=1-exp(-(x/eta)**k), x > 0, with location fixed at zero
    and unknown positive scale eta and shape k.
    Pass Distribution.DEFAULT.parse({}) to declare both parameters unknown.
    Logarithmic location and scale are eliminated within every call.

    The input is a complete sequence in acquisition order. Extract lower
    records R_j and the counts K_j of observations until the next record,
    including the record itself. Order records increasingly, carrying counts.
    Fit eta,k by the record likelihood product f(R_j)*S(R_j)**(K_j-1).
    At ordered records, the survival estimate is the product of
    1-1/sum(K_l,l>=j). Return n*integral((S_hat-S_fit)**2/F_fit dF_fit)
    over the entire support, including the first and last intervals.

    Reject for right tail values.
    Calibration must reproduce record extraction and condition on at least
    two records. The generic unconditional resolver is blocked.
    Previously stored unversioned Weibull calibrations must be regenerated.
    Order matters for full acquisition sequences. At least two records
    are required. Precompressed records require counts; do not sort an
    ordinary sample before calling. Inverse-sampling calibration is external.

    References
    ----------
    .. [1] M. Doostparast (2011), Goodness-of-fit tests for Weibull populations
       on the basis of records, equations (9), (10), (13), (14), (17).
       https://arxiv.org/abs/1110.5509

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({})
    >>> test = MahdiDoostparastWeibullGofStatistic(parameters)
    >>> value = test.execute_statistic([3.0, 1.2, 2.1, 0.4, 0.8, 0.2])
    >>> bool(np.isfinite(value))
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "MD".
        """
        return "MD"

    def alternative(self):
        return RightAlternative()

    def execute_statistic(self, rvs, record_counts=None, **kwargs):
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like
            One-dimensional finite strictly positive observations, n >= 2.
            Constant samples are invalid; ties are allowed unless stated below.
        record_counts : array_like of int, optional
            If given, rvs must be strictly decreasing lower records and these
            positive counts include each record. Otherwise extract from rvs.
        **kwargs : dict
            Unused common-interface keywords.

        Returns
        -------
        statistic : float
            Scalar discrepancy. Infinite boundary penalties are preserved.

        Raises
        ------
        ValueError
            Invalid sample, unsupported settings, or unrepresentable fit.

        Notes
        -----
        The input is never changed and estimated parameters are not retained.
        """
        x = _sample(rvs, min_size=2, positive=True)
        if record_counts is None:
            indices = np.flatnonzero(np.r_[True, x[1:] < np.minimum.accumulate(x)[:-1]])
            records = x[indices][::-1]
            counts = np.diff(np.r_[indices, len(x)])[::-1].astype(float)
        else:
            raw_counts = np.asarray(record_counts)
            if (
                raw_counts.shape != x.shape
                or raw_counts.dtype.kind not in "iu"
                or np.any(raw_counts <= 0)
                or np.any(np.diff(x) >= 0)
            ):
                raise ValueError(
                    "Records must decrease strictly and counts must be positive integers"
                )
            records, counts = x[::-1], raw_counts[::-1].astype(float)
        if len(records) < 2:
            raise ValueError("At least two lower records are required for fitting")
        y = _logs(records)
        z = _fit_logs(y, counts)[0]
        u = gumbel_l.cdf(z)
        log_u = _log_probabilities(z)[0]
        return self.do_execute_statistic(u, log_u, counts)

    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(u, log_u, counts):
        risk = np.cumsum(counts[::-1])[::-1]
        survival = np.concatenate((np.ones(1), np.cumprod(1 - 1 / risk)))
        c = survival - 1
        # The first c is zero, so its logarithmic term is exactly zero.
        log_widths = np.diff(np.concatenate((log_u, np.zeros(1))))
        if np.any(~np.isfinite(log_widths)):
            raise ValueError("Record CDF intervals cannot be represented")
        widths = np.diff(np.concatenate((np.zeros(1), u, np.ones(1))))
        value = np.dot(c[1:] ** 2, log_widths) + 2 * np.dot(c, widths) + 0.5
        return float(max(0.0, np.sum(counts) * value))

    def _validate_monte_carlo_calibration(self):
        raise ValueError(
            "Record calibration must reproduce the sampling scheme and "
            "condition on at least two records; use external calibration"
        )


_EDF_STATISTICS = {
    "KolmogorovSmirnovDPlusWeibullGofStatistic": EDFSpec(
        edf.AbstractKSDPlusStatistic,
        supported_hypotheses=(HypothesisSupport(Distribution.DEFAULT, frozenset()),),
    ),
    "KolmogorovSmirnovDMinusWeibullGofStatistic": EDFSpec(
        edf.AbstractKSDMinusStatistic,
        supported_hypotheses=(HypothesisSupport(Distribution.DEFAULT, frozenset()),),
    ),
    "KolmogorovSmirnovWeibullGofStatistic": EDFSpec(
        edf.AbstractKSStatistic,
        supported_hypotheses=(HypothesisSupport(Distribution.DEFAULT, frozenset()),),
    ),
    "AndersonDarlingWeibullGofStatistic": EDFSpec(
        edf.AbstractAndersonDarlingStatistic,
        supported_hypotheses=(HypothesisSupport(Distribution.DEFAULT, frozenset()),),
    ),
    "CrammerVonMisesWeibullGofStatistic": EDFSpec(
        edf.AbstractCramerVonMisesStatistic,
        supported_hypotheses=(HypothesisSupport(Distribution.DEFAULT, frozenset()),),
    ),
    "LiaoShimokawaWeibullGofStatistic": EDFSpec(
        edf.AbstractLiaoShimokawaStatistic,
        short_code="LS",
        supported_hypotheses=(HypothesisSupport(Distribution.DEFAULT, frozenset()),),
    ),
    "WatsonWeibullGofStatistic": EDFSpec(
        edf.AbstractWatsonStatistic,
        supported_hypotheses=(HypothesisSupport(Distribution.DEFAULT, frozenset()),),
    ),
    "KuiperWeibullGofStatistic": EDFSpec(
        edf.AbstractKuiperStatistic,
        supported_hypotheses=(HypothesisSupport(Distribution.DEFAULT, frozenset()),),
    ),
    "KuiperScaledWeibullGofStatistic": EDFSpec(
        edf.AbstractKuiperScaledStatistic,
        supported_hypotheses=(HypothesisSupport(Distribution.DEFAULT, frozenset()),),
    ),
    "KuiperCorrectedWeibullGofStatistic": EDFSpec(
        edf.AbstractKuiperCorrectedStatistic,
        supported_hypotheses=(HypothesisSupport(Distribution.DEFAULT, frozenset()),),
    ),
    "ZhangKWeibullGofStatistic": EDFSpec(
        edf.AbstractZhangKStatistic,
        supported_hypotheses=(HypothesisSupport(Distribution.DEFAULT, frozenset()),),
    ),
    "ZhangAWeibullGofStatistic": EDFSpec(
        edf.AbstractZhangAStatistic,
        supported_hypotheses=(HypothesisSupport(Distribution.DEFAULT, frozenset()),),
    ),
    "ZhangCWeibullGofStatistic": EDFSpec(
        edf.AbstractZhangCStatistic,
        supported_hypotheses=(HypothesisSupport(Distribution.DEFAULT, frozenset()),),
    ),
}

_FREQUENCY_STATISTICS = {
    "Chi2PearsonWeibullGofStatistic": FrequencySpec(
        frequency.AbstractPearsonChi2Statistic,
        supported_hypotheses=(HypothesisSupport(Distribution.DEFAULT, frozenset()),),
    ),
    "FreemanTukeyWeibullGofStatistic": FrequencySpec(
        frequency.AbstractFreemanTukeyStatistic,
        supported_hypotheses=(HypothesisSupport(Distribution.DEFAULT, frozenset()),),
    ),
    "ModifiedLogLikelihoodWeibullGofStatistic": FrequencySpec(
        frequency.AbstractModifiedLogLikelihoodStatistic,
        supported_hypotheses=(HypothesisSupport(Distribution.DEFAULT, frozenset()),),
    ),
    "NeymanChi2WeibullGofStatistic": FrequencySpec(
        frequency.AbstractNeymanChi2Statistic,
        supported_hypotheses=(HypothesisSupport(Distribution.DEFAULT, frozenset()),),
    ),
    "CressieReadWeibullGofStatistic": FrequencySpec(
        frequency.AbstractCressieReadStatistic,
        supported_hypotheses=(HypothesisSupport(Distribution.DEFAULT, frozenset()),),
    ),
}

__all__ = [
    "AbstractWeibullFrequencyGofStatistic",
    "AbstractWeibullGofStatistic",
    "BestRaynerThasKurtosisWeibullGofStatistic",
    "BestRaynerThasSkewnessWeibullGofStatistic",
    "CabanaQuirozWeibullGofStatistic",
    "ColesWeibullGofStatistic",
    "EvansJohnsonGreenWeibullGofStatistic",
    "KullbackLeiblerWeibullGofStatistic",
    "LaplaceTransform2WeibullGofStatistic",
    "LaplaceTransform3WeibullGofStatistic",
    "LaplaceTransformWeibullGofStatistic",
    "LockhartOReillyStephensWeibullGofStatistic",
    "MahdiDoostparastWeibullGofStatistic",
    "MannScheuerFertigWeibullGofStatistic",
    "NormalizeSpaceWeibullGofStatistic",
    "OzturkKorukogluWeibullGofStatistic",
    "ST1WeibullGofStatistic",
    "ST2WeibullGofStatistic",
    "ShapiroBrainWeibullGofStatistic",
    "SmithBainWeibullGofStatistic",
    "TikuSinghWeibullGofStatistic",
    "WPPWeibullGofStatistic",
]
__all__.extend(
    export_edf_statistics(
        globals(),
        AbstractWeibullGofStatistic,
        _EDF_STATISTICS,
        log_probabilities=_weibull_log_probabilities,
    )
)
__all__.extend(
    export_frequency_statistics(
        globals(),
        AbstractWeibullFrequencyGofStatistic,
        _FREQUENCY_STATISTICS,
        bind_distribution=_weibull_bound_distribution,
    )
)

for _name in (*_EDF_STATISTICS, *_FREQUENCY_STATISTICS):
    globals()[_name].__doc__ += """

    Weibull hypothesis
    ------------------
    Use Distribution.DEFAULT.parse({}): shape and scale are both unknown,
    location is zero. Maximum likelihood estimates are recomputed for every
    sample (finite, strictly positive, nonconstant, with at least two values).
    Calibrate with the same fitting procedure on every simulated sample;
    fully specified distribution tables do not apply. No fitted parameters
    are retained on the criterion instance.

    For fitted Weibull EDF calibration, see Liao and Shimokawa (1999):
    https://doi.org/10.1080/00949659908811965
    """
    if _name in _FREQUENCY_STATISTICS:
        globals()[_name].__doc__ += """
    The binning keyword accepts a BinningStrategy; the default is
    EqualProbabilityBins(). Built-in Monte Carlo calibration requires this
    strategy; other rules need external simulation with explicit parameters.
    """
del _name
