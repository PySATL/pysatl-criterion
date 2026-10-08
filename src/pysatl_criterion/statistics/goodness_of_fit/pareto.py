"""Pareto I goodness-of-fit statistics.

F(x)=1-(scale/x)**shape on x>=scale>0, with zero location.
Every call returns a scalar and leaves observations and fitted state unchanged.
Built-in Pareto simulation is unavailable; calibration is external.
"""

from __future__ import annotations

from abc import ABC
from fractions import Fraction

import numpy as np
from typing_extensions import override

from pysatl_criterion.distribution.distributions import ParetoDistributionDescriptor
from pysatl_criterion.distribution.distributions import ParetoDistributionDescriptor as Distribution
from pysatl_criterion.distribution.parameters import HypothesisSupport, ParameterValues
from pysatl_criterion.statistics import AbstractGoodnessOfFitStatistic
from pysatl_criterion.statistics.alternative import (
    AlternativeType,
    RightAlternative,
    TwoSidedAlternative,
)
from pysatl_criterion.statistics.goodness_of_fit.common import (
    ADStatistic,
    CrammerVonMisesStatistic,
    KSStatistic,
    MinToshiyukiStatistic,
)


def _sample(rvs, minimum=1, scale=None):
    if np.iscomplexobj(rvs):
        raise ValueError("Sample must be real")
    try:
        x = np.array(rvs, dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Sample must contain finite real observations") from error
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError("Sample must be finite and one-dimensional")
    if x.size < minimum:
        raise ValueError(f"At least {minimum} observations are required")
    if np.any(x <= 0):
        raise ValueError("Sample values must be strictly positive")
    if scale is not None and np.any(x < scale):
        raise ValueError("Sample values must be at least scale")
    return np.sort(x)


def _log_ratio(x, scale):
    # log1p retains adjacent float64 values even at extreme common scales.
    with np.errstate(over="ignore"):
        relative = (x - scale) / scale
    close = relative <= 1
    result = np.log(x) - np.log(scale)
    result[close] = np.log1p(relative[close])
    return result


def _fitted_logs(x):
    y = _log_ratio(x, x[0])
    if y[-1] == 0:
        raise ValueError("A nonconstant sample is required for a finite shape MLE")
    return y


def _log_probabilities(x, shape, scale):
    y = _log_ratio(x, scale)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        log_sf = -shape * y
        log_cdf = np.log(-np.expm1(log_sf))
        # A positive shape*y can underflow although its logarithm is finite.
        tiny = (y > 0) & (-log_sf < np.finfo(float).tiny)
        log_cdf[tiny] = np.log(shape) + np.log(y[tiny])
    return log_cdf, log_sf


class AbstractParetoGofStatistic(AbstractGoodnessOfFitStatistic, ABC):
    @property
    def shape(self) -> float:
        """Read shape by its stable parameter identity."""
        return self._parameters[Distribution.SHAPE]

    @property
    def scale(self) -> float:
        """Read scale by its stable parameter identity."""
        return self._parameters[Distribution.SCALE]

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.SCALE})
            ),
        )

    @staticmethod
    @override
    def distribution() -> type[ParetoDistributionDescriptor]:
        """Return the distribution descriptor class."""
        return ParetoDistributionDescriptor

    @classmethod
    @override
    def code(cls) -> str:
        """Return the family identifier or the concrete statistic's full identifier."""
        family_code = f"PARETO_{AbstractGoodnessOfFitStatistic.code()}"
        if "short_code" in cls.__abstractmethods__:
            return family_code
        return f"{cls.short_code()}_{family_code}"


class KolmogorovSmirnovParetoGofStatistic(AbstractParetoGofStatistic, KSStatistic):
    """Kolmogorov-Smirnov distance to a specified Pareto CDF.

    Parameters
    ----------
    parameters : ParameterValues
        Values with shape, scale fixed; omitted parameters are unknown.
    alternative_type : AlternativeType, optional
        CDF deviation direction, default TWO_TAILED.
    mode : {'auto', 'exact', 'approx', 'asymp'}, optional
        Compatibility setting, default 'auto'; does not change the statistic.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic without modifying data or retaining fits.
    hypothesis()
        Report fixed parameters; omitted keys denote unknown parameters.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 fixes both shape and scale in F(x)=1-(scale/x)**shape, x>=scale.
    D+ = max(i/n-u_i), D- = max(u_i-(i-1)/n), u_i=F(x_(i)).
    Return max(D+,D-) for TWO_TAILED, D+ for RIGHT, D- for LEFT.
    All directions reject for large statistics. No parameters are fitted.
    The probability transform makes the continuous null law parameter-free.
    Stored calibration supports TWO_TAILED only because its key omits direction.
    Accept a finite real one-dimensional sample of length n>=1.
    Built-in Pareto simulation is unavailable; use external calibration.

    References
    ----------
    .. [1] Smirnov, N. (1948). Table for Estimating the Goodness of Fit of
       Empirical Distributions. https://doi.org/10.1214/aoms/1177730256
       This is the general specified-CDF test, applied to Pareto by its CDF.

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'shape': 1, 'scale': 1})
    >>> statistic = KolmogorovSmirnovParetoGofStatistic(parameters)
    >>> value = statistic.execute_statistic([1.1, 1.4, 1.9, 2.7, 4.2, 8.0])
    >>> bool(np.isfinite(value))
    True
    """

    def __init__(
        self,
        parameters: ParameterValues,
        *,
        alternative_type: AlternativeType = AlternativeType.TWO_TAILED,
        mode="auto",
    ):
        AbstractParetoGofStatistic.__init__(self, parameters)
        if not isinstance(alternative_type, AlternativeType):
            raise TypeError("alternative_type must be an AlternativeType")
        if mode not in ("auto", "exact", "approx", "asymp"):
            raise ValueError("Unsupported KS mode")
        KSStatistic.__init__(self, alternative_type=alternative_type, mode=mode)

    def _validate_storage_calibration(self):
        if self.alternative_type != AlternativeType.TWO_TAILED:
            raise ValueError("Stored KS calibration does not encode the CDF direction")

    @staticmethod
    @override
    def short_code():
        return "KS"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute KolmogorovSmirnov for this sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real observations, n >= 1; x >= scale.
        **kwargs : dict, optional
            Unused compatibility keywords.

        Returns
        -------
        statistic : float
            Scalar in the normalization in Notes; no p-value is computed.

        Raises
        ------
        ValueError
            If input is nonreal, nonfinite, not one-dimensional, too short,
            outside the support, or a required fit or window is undefined.
        """
        sorted_rvs = _sample(rvs, scale=self.scale)
        _, log_sf = _log_probabilities(sorted_rvs, self.shape, self.scale)
        cdf_vals = -np.expm1(log_sf)
        return KSStatistic.do_execute_statistic(self, sorted_rvs, cdf_vals)


class AndersonDarlingParetoGofStatistic(AbstractParetoGofStatistic, ADStatistic):
    """Anderson-Darling A-squared for a specified Pareto CDF.

    Parameters
    ----------
    parameters : ParameterValues
        Values with shape, scale fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic without modifying data or retaining fits.
    hypothesis()
        Report fixed parameters; omitted keys denote unknown parameters.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 fixes both shape and scale in F(x)=1-(scale/x)**shape, x>=scale.
    A2 = -n - sum((2*i-1)/n * (log(u_i)+log(1-u_(n+1-i)))).
    Here u_i=F(x_(i)); no finite-sample correction or parameter fit is used.
    The probability transform makes the continuous null law parameter-free.
    Reject for large values. An observation at scale gives positive infinity.
    Log survival probabilities are evaluated without subtracting the CDF from 1.
    Accept a finite real one-dimensional sample of length n>=1.
    Built-in Pareto simulation is unavailable; use external calibration.

    References
    ----------
    .. [1] Anderson, T. W. and Darling, D. A. (1952). Asymptotic Theory of
       Certain Goodness of Fit Criteria Based on Stochastic Processes.
       https://doi.org/10.1214/aoms/1177729437
       The general weighted EDF criterion is applied through the Pareto CDF.

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'shape': 1, 'scale': 1})
    >>> statistic = AndersonDarlingParetoGofStatistic(parameters)
    >>> value = statistic.execute_statistic([1.1, 1.4, 1.9, 2.7, 4.2, 8.0])
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def short_code():
        return "AD"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute AndersonDarling for this sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real observations, n >= 1; x >= scale.
        **kwargs : dict, optional
            Unused compatibility keywords.

        Returns
        -------
        statistic : float
            Scalar in the normalization in Notes; no p-value is computed.
            An observation at scale gives positive infinity.

        Raises
        ------
        ValueError
            If input is nonreal, nonfinite, not one-dimensional, too short,
            outside the support, or a required fit or window is undefined.
        """
        sorted_rvs = _sample(rvs, scale=self.scale)
        log_cdf, log_sf = _log_probabilities(sorted_rvs, self.shape, self.scale)
        return super().do_execute_statistic(sorted_rvs, log_cdf=log_cdf, log_sf=log_sf)


class CramerVonMisesParetoGofStatistic(AbstractParetoGofStatistic, CrammerVonMisesStatistic):
    """Cramer-von Mises W-squared for a specified Pareto CDF.

    Parameters
    ----------
    parameters : ParameterValues
        Values with shape, scale fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic without modifying data or retaining fits.
    hypothesis()
        Report fixed parameters; omitted keys denote unknown parameters.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 fixes both shape and scale in F(x)=1-(scale/x)**shape, x>=scale.
    W2 = 1/(12*n) + sum((u_i-(2*i-1)/(2*n))**2), u_i=F(x_(i)).
    No parameters are fitted and no finite-sample correction is used.
    The probability transform makes the continuous null law parameter-free.
    Reject for large values; observations at scale are permitted.
    Accept a finite real one-dimensional sample of length n>=1.
    Built-in Pareto simulation is unavailable; use external calibration.

    References
    ----------
    .. [1] Anderson, T. W. and Darling, D. A. (1952). Asymptotic Theory of
       Certain Goodness of Fit Criteria Based on Stochastic Processes.
       https://doi.org/10.1214/aoms/1177729437
       The general unweighted EDF criterion is applied through the Pareto CDF.

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'shape': 1, 'scale': 1})
    >>> statistic = CramerVonMisesParetoGofStatistic(parameters)
    >>> value = statistic.execute_statistic([1.1, 1.4, 1.9, 2.7, 4.2, 8.0])
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def short_code():
        return "CVM"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute CramerVonMises for this sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real observations, n >= 1; x >= scale.
        **kwargs : dict, optional
            Unused compatibility keywords.

        Returns
        -------
        statistic : float
            Scalar in the normalization in Notes; no p-value is computed.

        Raises
        ------
        ValueError
            If input is nonreal, nonfinite, not one-dimensional, too short,
            outside the support, or a required fit or window is undefined.
        """
        sorted_rvs = _sample(rvs, scale=self.scale)
        _, log_sf = _log_probabilities(sorted_rvs, self.shape, self.scale)
        cdf_vals = -np.expm1(log_sf)
        return CrammerVonMisesStatistic.do_execute_statistic(self, sorted_rvs, cdf_vals)


class MinToshiyukiParetoGofStatistic(AbstractParetoGofStatistic, MinToshiyukiStatistic):
    """Locally defined weighted EDF discrepancy for specified Pareto.

    Parameters
    ----------
    parameters : ParameterValues
        Values with shape, scale fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic without modifying data or retaining fits.
    hypothesis()
        Report fixed parameters; omitted keys denote unknown parameters.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 fixes both shape and scale in F(x)=1-(scale/x)**shape, x>=scale.
    Let u_i=F(x_(i)) and d_i=max(i/n-u_i, u_i-(i-1)/n).
    Return sum(d_i/sqrt(u_i*(1-u_i)))/sqrt(n); reject for large values.
    No parameters are fitted. The probability transform makes the continuous
    null law parameter-free. At scale the result is positive infinity.
    Log probabilities avoid artificial clipping; float64 overflow may also
    produce infinity for unrepresentably large penalties.
    A primary source confirming this exact Pareto statistic was not located.
    The Liao-Shimokawa work concerns fitted Weibull/extreme-value models and
    is not used to justify this fixed-Pareto implementation.
    Accept a finite real one-dimensional sample of length n>=1.
    Built-in Pareto simulation is unavailable; use external calibration.

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'shape': 1, 'scale': 1})
    >>> statistic = MinToshiyukiParetoGofStatistic(parameters)
    >>> value = statistic.execute_statistic([1.1, 1.4, 1.9, 2.7, 4.2, 8.0])
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def short_code():
        return "MT"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute MinToshiyuki for this sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real observations, n >= 1; x >= scale.
        **kwargs : dict, optional
            Unused compatibility keywords.

        Returns
        -------
        statistic : float
            Scalar in the normalization in Notes; no p-value is computed.
            An observation at scale gives positive infinity.

        Raises
        ------
        ValueError
            If input is nonreal, nonfinite, not one-dimensional, too short,
            outside the support, or a required fit or window is undefined.
        """
        x = _sample(rvs, scale=self.scale)
        log_cdf, log_sf = _log_probabilities(x, self.shape, self.scale)
        u = -np.expm1(log_sf)
        n = len(x)
        d = np.maximum(np.arange(1, n + 1) / n - u, u - np.arange(n) / n)
        with np.errstate(over="ignore", divide="ignore"):
            terms = np.exp(np.log(d) - (log_cdf + log_sf) / 2)
        return float(np.sum(terms) / np.sqrt(n))


class ObradovicParetoGofStatistic(AbstractParetoGofStatistic):
    """Obradovic-Jovanovic-Milosevic signed integral statistic.

    Parameters
    ----------
    parameters : ParameterValues
        Values with scale fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic without modifying data or retaining fits.
    hypothesis()
        Report fixed parameters; omitted keys denote unknown parameters.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is Pareto I with known scale and unknown shape. Let z_i=x_i/scale,
    F_n be its empirical CDF and M_n(t)=mean_{j<k} I(max(z_j/z_k,z_k/z_j)<=t).
    Return T_n=mean_i(M_n(z_i)-F_n(z_i)), without absolute value or sqrt(n).
    Use the upper critical tail as in [1]. The original unit-scale model is
    obtained by dividing observations by the specified scale. No fit is made.
    Positive powers of z preserve the comparisons, eliminating shape from
    the continuous null law. Ties use inclusive indicators, not midranks.
    The implementation uses log ratios and exact rational comparisons near
    rounding boundaries, with O(n**2 log(n)) comparisons and O(n) working
    memory. Exact comparisons refer to the supplied binary floating values.
    Constant samples are allowed. A one-sided signed integral is not claimed
    to detect every alternative.
    Accept a finite real one-dimensional sample of length n>=2.
    Built-in Pareto simulation is unavailable; use external calibration.

    References
    ----------
    .. [1] Obradovic, M., Jovanovic, M., Milosevic, B. (2014).
       Goodness of Fit Tests for Pareto Distribution Based on a Characterization
       and their Asymptotics. Section 2, definition of T_n.
       https://arxiv.org/abs/1310.5510

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'scale': 1})
    >>> statistic = ObradovicParetoGofStatistic(parameters)
    >>> value = statistic.execute_statistic([1.1, 1.4, 1.9, 2.7, 4.2, 8.0])
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def short_code():
        return "OJM_Integral"

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset({Distribution.SCALE})),)

    @override
    def alternative(self):
        return RightAlternative()

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute Obradovic for this sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real observations, n >= 2; x >= scale.
        **kwargs : dict, optional
            Unused compatibility keywords.

        Returns
        -------
        statistic : float
            Scalar in the normalization in Notes; no p-value is computed.

        Raises
        ------
        ValueError
            If input is nonreal, nonfinite, not one-dimensional, too short,
            outside the support, or a required fit or window is undefined.
        """
        x = _sample(rvs, 2, self.scale)
        n = len(x)
        y = _log_ratio(x, self.scale)
        exact_x = [Fraction(float(value)) for value in x]
        exact_scale = Fraction(self.scale)
        # Bound cancellation error in the logs. Resolve ambiguous comparisons
        # exactly for the supplied binary floats, including true equalities.
        tolerance = (
            32
            * np.finfo(float).eps
            * max(1.0, float(np.max(np.abs(np.log(x)))), abs(np.log(self.scale)))
        )
        count = 0
        for j in range(n - 1):
            ratios = _log_ratio(x[j + 1 :], x[j])
            left = np.searchsorted(y, ratios - tolerance, side="left")
            right = np.searchsorted(y, ratios + tolerance, side="right")
            for offset in np.flatnonzero(left != right):
                lo, hi = int(left[offset]), int(right[offset])
                numerator = exact_x[j + 1 + offset] * exact_scale
                while lo < hi:
                    mid = (lo + hi) // 2
                    if exact_x[mid] * exact_x[j] < numerator:
                        lo = mid + 1
                    else:
                        hi = mid
                left[offset] = lo
            count += np.sum(n - left)
        empirical = np.searchsorted(x, x, side="right") / n
        return float(count / (n * (n * (n - 1) // 2)) - np.mean(empirical))


class GreenwoodParetoGofStatistic(AbstractParetoGofStatistic):
    """Greenwood ratio of logarithmic excesses above the sample minimum.

    Parameters
    ----------
    parameters : ParameterValues
        An empty distribution schema; all distribution parameters are unknown.


    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic without modifying data or retaining fits.
    hypothesis()
        Report fixed parameters; omitted keys denote unknown parameters.
    alternative()
        Report the critical tail.

    Notes
    -----
    H0 is the full Pareto I family with both parameters unknown.
    Let y_i=log(x_i/min(x)); return G=sum(y_i**2)/sum(y_i)**2.
    A nonconstant sample is required. At n=2 G is identically 1, so n>=3
    is required for a nontrivial test. Repeats otherwise are accepted.
    Both tails are used to detect excess or deficient relative dispersion.
    Under X=s*exp(E/a), E~Exp(1), scale cancels and the factor 1/a cancels
    from the ratio. Conditional on the minimum, n-1 positive exponential
    residuals have normalized Dirichlet(1,...,1) proportions; E[G]=2/n.
    This derivation applies to continuous iid data, not rounded samples.
    A primary source for this exact fitted-Pareto version was not located;
    it is not the sum of squared n+1 spacings of the known Pareto CDF.
    Accept a finite real one-dimensional sample of length n>=3.
    Built-in Pareto simulation is unavailable; use external calibration.

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({})
    >>> statistic = GreenwoodParetoGofStatistic(parameters)
    >>> value = statistic.execute_statistic([1.1, 1.4, 1.9, 2.7, 4.2, 8.0])
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def short_code():
        return "Greenwood"

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @override
    def alternative(self):
        return TwoSidedAlternative()

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute Greenwood for this sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real observations, n >= 3; x > 0, nonconstant.
        **kwargs : dict, optional
            Unused compatibility keywords.

        Returns
        -------
        statistic : float
            Scalar in the normalization in Notes; no p-value is computed.

        Raises
        ------
        ValueError
            If input is nonreal, nonfinite, not one-dimensional, too short,
            outside the support, or a required fit or window is undefined.
        """
        x = _sample(rvs, 3)
        y = _fitted_logs(x)
        weights = y / np.sum(y)
        return float(np.sum(weights**2))


class LequesneKlParetoGofStatistic(AbstractParetoGofStatistic):
    """Vasicek-Song KL estimate for the fitted Pareto I family.

    Parameters
    ----------
    parameters : ParameterValues
        An empty distribution schema; all distribution parameters are unknown.


    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic without modifying data or retaining fits.
    hypothesis()
        Report fixed parameters; omitted keys denote unknown parameters.
    alternative()
        Report the critical tail.

    Notes
    -----
    Both parameters are unknown and fitted on every call:
    s_hat=min(x), a_hat=1/mean(log(x/s_hat)). For ascending x_(i), extend
    indices below 1 or above n by the respective endpoint. Define
    V_mn=mean(log(n*(x_(i+m)-x_(i-m))/(2*m))). Return
    I_mn=-V_mn-log(a_hat)+log(s_hat)+(a_hat+1)*mean(log(x/s_hat)).
    This is the raw fixed-window statistic in [1], equations (5) and (8),
    not its adaptive-window or bias-corrected variant. Negative estimates
    are retained; only large values reject. The class name is retained.
    A zero spacing gives positive infinity; a constant sample has no finite
    shape MLE and raises ValueError. Scale invariance holds, but finite-sample
    shape invariance does not: external shape-specific calibration must
    refit every replicate with the same m. Generic storage is blocked because
    its key omits the simulation shape and window. A plug-in shape simulation
    is approximate, not an exact test of the entire composite hypothesis.
    Accept a finite real one-dimensional sample of length n>=4.
    Built-in Pareto simulation is unavailable; use external calibration.

    References
    ----------
    .. [1] Lequesne, J. and Regnault, P. (2020). vsgoftest: An R Package for
       Goodness-of-Fit Testing Based on Kullback-Leibler Divergence.
       https://doi.org/10.18637/jss.v096.c01
       Author preprint: https://arxiv.org/abs/1806.07244, Section 2.

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({})
    >>> statistic = LequesneKlParetoGofStatistic(parameters)
    >>> value = statistic.execute_statistic([1.1, 1.4, 1.9, 2.7, 4.2, 8.0])
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def short_code():
        return "Lequesne_KL"

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @override
    def alternative(self):
        return RightAlternative()

    def _validate_storage_calibration(self):
        raise ValueError("KL requires external shape-specific calibration with the same window m")

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute LequesneKl for this sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real observations, n >= 4; x > 0, nonconstant.
        **kwargs : dict, optional
            Only m is used: integer 1 <= m < n/2; default min(floor(sqrt(n)),
            floor((n-1)/2)). Other keywords are ignored.

        Returns
        -------
        statistic : float
            Scalar in the normalization in Notes; no p-value is computed.
            Zero spacing gives positive infinity; finite negative values are valid.

        Raises
        ------
        ValueError
            If input is nonreal, nonfinite, not one-dimensional, too short,
            outside the support, or a required fit or window is undefined.
        """
        x = _sample(rvs, 4)
        n = len(x)
        y = _fitted_logs(x)
        m = kwargs.get("m", min(int(np.sqrt(n)), (n - 1) // 2))
        if isinstance(m, (bool, np.bool_)) or not isinstance(m, (int, np.integer)):
            raise ValueError("m must be an integer with 1 <= m < n/2")  # noqa: TRY004
        if not 1 <= m < n / 2:
            raise ValueError("m must be an integer with 1 <= m < n/2")
        indices = np.arange(n)
        lower = x[np.maximum(indices - m, 0)]
        upper = x[np.minimum(indices + m, n - 1)]
        # Subtract only positive values, then take logs before rescaling.
        with np.errstate(divide="ignore"):
            log_spacings = np.log(upper - lower) - np.log(x[0])
        entropy_scaled = np.mean(log_spacings) + np.log(n / (2 * m))
        mean_log = np.mean(y)
        return float(-entropy_scaled + np.log(mean_log) + 1 + mean_log)
