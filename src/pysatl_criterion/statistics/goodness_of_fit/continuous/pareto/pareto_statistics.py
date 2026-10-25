"""Pareto I goodness-of-fit statistics.

F(x)=1-(scale/x)**shape on x>=scale>0, with zero location.
Every call returns a scalar and leaves observations and fitted state unchanged.
Built-in Pareto simulation is unavailable; calibration is external.
"""

from __future__ import annotations

from abc import ABC
from fractions import Fraction

import numpy as np
from numba import jit
from typing_extensions import override

from pysatl_criterion.distribution import ParetoDistributionDescriptor
from pysatl_criterion.distribution import ParetoDistributionDescriptor as Distribution
from pysatl_criterion.distribution.parameters import HypothesisSupport
from pysatl_criterion.statistics.alternative import (
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


class AbstractParetoGofStatistic(AbstractGoodnessOfFitStatistic, ABC):
    @staticmethod
    def _sample(rvs, minimum=1, scale=None):
        if np.iscomplexobj(rvs) or (np.ma.isMaskedArray(rvs) and np.any(rvs.mask)):
            raise ValueError("Sample must be real and contain no masked observations")
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

    @staticmethod
    @jit(nopython=True, cache=True)
    def _log_ratio(x, scale):
        """Stable logs for positive values at least scale, including adjacent floats."""
        result = np.empty(len(x), dtype=np.float64)
        log_scale = np.log(scale)
        for i in range(len(x)):
            relative = (x[i] - scale) / scale
            if relative <= 1.0:
                result[i] = np.log1p(relative)
            else:
                result[i] = np.log(x[i]) - log_scale
        return result

    def _fitted_logs(self, x):
        y = self._log_ratio(x, x[0])
        if y[-1] == 0:
            raise ValueError("A nonconstant sample is required for a finite shape MLE")
        return y

    @staticmethod
    @jit(nopython=True, cache=True)
    def _log_probabilities(y, shape):
        """Evaluate Pareto log probabilities from log(x / scale)."""
        log_cdf = np.empty(len(y), dtype=np.float64)
        log_sf = np.empty(len(y), dtype=np.float64)
        for i in range(len(y)):
            exponent = shape * y[i]
            log_sf[i] = -exponent
            if y[i] > 0 and exponent < np.finfo(np.float64).tiny:
                log_cdf[i] = np.log(shape) + np.log(y[i])
            else:
                log_cdf[i] = np.log(-np.expm1(-exponent))
        return log_cdf, log_sf

    def _pareto_log_probabilities(self, sample):
        sample = self._validate_input(sample)
        y = self._log_ratio(sample, self.scale)
        return self._log_probabilities(y, self.shape)

    @property
    def shape(self) -> float:
        """Read shape by its stable parameter identity."""
        return self._parameters[Distribution.SHAPE]

    @property
    def scale(self) -> float:
        """Read scale by its stable parameter identity."""
        return self._parameters[Distribution.SCALE]

    def _validate_input(self, rvs):
        return self._sample(rvs, scale=self.scale)

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
    A cached nopython kernel handles ordinary log-ratio comparisons. Near
    rounding boundaries it requests an exact rational fallback, preserving
    O(n**2 log(n)) comparisons and O(n) working
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
        x = self._sample(rvs, 2, self.scale)
        y = self._log_ratio(x, self.scale)
        # Bound cancellation error in the logs. Resolve ambiguous comparisons
        # exactly for the supplied binary floats, including true equalities.
        tolerance = (
            32
            * np.finfo(float).eps
            * max(1.0, float(np.max(np.abs(np.log(x)))), abs(np.log(self.scale)))
        )
        result = self.do_execute_statistic(x, y, tolerance)
        if np.isnan(result):
            return self._execute_exact_statistic(x, y, tolerance)
        return float(result)

    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(x, y, tolerance):
        """Count pairs in nopython mode; NaN requests exact boundary handling."""
        n = len(x)
        count = 0.0
        empirical_count = 0.0
        for j in range(n):
            empirical_count += np.searchsorted(x, x[j], side="right")
            for k in range(j + 1, n):
                relative = (x[k] - x[j]) / x[j]
                if relative <= 1.0:
                    ratio = np.log1p(relative)
                else:
                    ratio = np.log(x[k]) - np.log(x[j])
                left = np.searchsorted(y, ratio - tolerance, side="left")
                right = np.searchsorted(y, ratio + tolerance, side="right")
                if left != right:
                    return np.nan
                count += n - left
        return count / (n * (n * (n - 1) / 2.0)) - empirical_count / (n * n)

    def _execute_exact_statistic(self, x, y, tolerance):
        """Resolve close boundaries using the original binary float values exactly."""
        n = len(x)
        exact_x = [Fraction(float(value)) for value in x]
        exact_scale = Fraction(self.scale)
        count = 0
        for j in range(n - 1):
            ratios = self._log_ratio(x[j + 1 :], x[j])
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

    References
    ----------
    .. [1] Does, R. J. M. M., Helmers, R., and Klaassen, C. A. J. (1988).
       Approximating the distribution of Greenwood's statistic.
       Statistica Neerlandica, 42(3). Equations (1)-(2).
       https://ir.cwi.nl/pub/1694/1694D.pdf
       This is a reference for the Greenwood concentration functional;
       the Pareto log-residual transformation and n-1 components are a
       library adaptation, not the paper's n+1 CDF spacings.

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
        x = self._sample(rvs, 3)
        y = self._fitted_logs(x)
        return float(self.do_execute_statistic(y))

    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(y):
        """Concentration of log residuals, normalized before squaring."""
        total = np.sum(y)
        result = 0.0
        for value in y:
            weight = value / total
            result += weight * weight
        return result


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
        x = self._sample(rvs, 4)
        n = len(x)
        y = self._fitted_logs(x)
        m = kwargs.get("m", min(int(np.sqrt(n)), (n - 1) // 2))
        if isinstance(m, (bool, np.bool_)) or not isinstance(m, (int, np.integer)):
            raise ValueError("m must be an integer with 1 <= m < n/2")  # noqa: TRY004
        if not 1 <= m < n / 2:
            raise ValueError("m must be an integer with 1 <= m < n/2")
        return float(self.do_execute_statistic(x, y, int(m)))

    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(x, y, m):
        """Raw fixed-window KL estimate with clamped endpoint spacings."""
        n = len(x)
        log_scale = np.log(x[0])
        total = 0.0
        for i in range(n):
            spacing = x[min(i + m, n - 1)] - x[max(i - m, 0)]
            total += np.log(spacing) - log_scale
        entropy_scaled = total / n + np.log(n / (2.0 * m))
        mean_log = np.mean(y)
        return -entropy_scaled + np.log(mean_log) + 1 + mean_log


_EDF_STATISTICS = {
    "KolmogorovSmirnovDPlusParetoGofStatistic": EDFSpec(
        edf.AbstractKSDPlusStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.SCALE})
            ),
        ),
    ),
    "KolmogorovSmirnovDMinusParetoGofStatistic": EDFSpec(
        edf.AbstractKSDMinusStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.SCALE})
            ),
        ),
    ),
    "KolmogorovSmirnovParetoGofStatistic": EDFSpec(
        edf.AbstractKSStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.SCALE})
            ),
        ),
    ),
    "AndersonDarlingParetoGofStatistic": EDFSpec(
        edf.AbstractAndersonDarlingStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.SCALE})
            ),
        ),
    ),
    "CramerVonMisesParetoGofStatistic": EDFSpec(
        edf.AbstractCramerVonMisesStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.SCALE})
            ),
        ),
    ),
    "MinToshiyukiParetoGofStatistic": EDFSpec(
        edf.AbstractLiaoShimokawaStatistic,
        short_code="MT",
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.SCALE})
            ),
        ),
    ),
    "WatsonParetoGofStatistic": EDFSpec(
        edf.AbstractWatsonStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.SCALE})
            ),
        ),
    ),
    "KuiperParetoGofStatistic": EDFSpec(
        edf.AbstractKuiperStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.SCALE})
            ),
        ),
    ),
    "KuiperScaledParetoGofStatistic": EDFSpec(
        edf.AbstractKuiperScaledStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.SCALE})
            ),
        ),
    ),
    "KuiperCorrectedParetoGofStatistic": EDFSpec(
        edf.AbstractKuiperCorrectedStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.SCALE})
            ),
        ),
    ),
    "ZhangKParetoGofStatistic": EDFSpec(
        edf.AbstractZhangKStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.SCALE})
            ),
        ),
    ),
    "ZhangAParetoGofStatistic": EDFSpec(
        edf.AbstractZhangAStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.SCALE})
            ),
        ),
    ),
    "ZhangCParetoGofStatistic": EDFSpec(
        edf.AbstractZhangCStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.SCALE})
            ),
        ),
    ),
}

_FREQUENCY_STATISTICS = {
    "Chi2PearsonParetoGofStatistic": FrequencySpec(
        frequency.AbstractPearsonChi2Statistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.SCALE})
            ),
        ),
    ),
    "FreemanTukeyParetoGofStatistic": FrequencySpec(
        frequency.AbstractFreemanTukeyStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.SCALE})
            ),
        ),
    ),
    "ModifiedLogLikelihoodParetoGofStatistic": FrequencySpec(
        frequency.AbstractModifiedLogLikelihoodStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.SCALE})
            ),
        ),
    ),
    "NeymanChi2ParetoGofStatistic": FrequencySpec(
        frequency.AbstractNeymanChi2Statistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.SCALE})
            ),
        ),
    ),
    "CressieReadParetoGofStatistic": FrequencySpec(
        frequency.AbstractCressieReadStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.SCALE})
            ),
        ),
    ),
}

__all__ = [
    "AbstractParetoGofStatistic",
    "GreenwoodParetoGofStatistic",
    "LequesneKlParetoGofStatistic",
    "ObradovicParetoGofStatistic",
]
__all__.extend(
    export_edf_statistics(
        globals(),
        AbstractParetoGofStatistic,
        _EDF_STATISTICS,
        log_probabilities=AbstractParetoGofStatistic._pareto_log_probabilities,
    )
)
__all__.extend(
    export_frequency_statistics(globals(), AbstractParetoGofStatistic, _FREQUENCY_STATISTICS)
)

for _name in (*_EDF_STATISTICS, *_FREQUENCY_STATISTICS):
    globals()[_name].__doc__ += """

    Pareto hypothesis
    -----------------
    Shape and scale must both be specified, with zero location. No parameters
    are fitted. Samples must be finite, nonempty and at least scale.
    Built-in Pareto simulation is unavailable; use external calibration.
    """

    globals()[_name].__doc__ += f"""
    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({{'shape': 2, 'scale': 1}})
    >>> statistic = {_name}(parameters)
    >>> value = statistic.execute_statistic([1.1, 1.4, 1.9, 2.7, 4.2, 8.0])
    >>> bool(np.isfinite(value) or np.isposinf(value))
    True
    """
