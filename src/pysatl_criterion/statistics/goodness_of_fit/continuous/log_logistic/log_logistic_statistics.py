from abc import ABC
from functools import lru_cache

import numpy as np
from numba import jit
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import logit
from typing_extensions import override

from pysatl_criterion.distribution import LogLogisticDistributionDescriptor
from pysatl_criterion.distribution import (
    LogLogisticDistributionDescriptor as Distribution,
)
from pysatl_criterion.distribution.parameters import HypothesisSupport, ParameterValues
from pysatl_criterion.statistics.alternative import RightAlternative
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


class AbstractLogLogisticGofStatistic(AbstractGoodnessOfFitStatistic, ABC):
    @staticmethod
    def _interval_count(value):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < 2
        ):
            raise ValueError("At least two bins are required; the count must be an integer")
        return int(value)

    @staticmethod
    def _sample(rvs):
        if np.iscomplexobj(rvs):
            raise ValueError("Observations must be real")
        sample = np.asarray(rvs, dtype=float)
        if sample.ndim != 1 or not sample.size:
            raise ValueError("At least one observation in a one-dimensional sample is required")
        if not np.all(np.isfinite(sample)):
            raise ValueError("Observations must be finite")
        if np.any(sample <= 0):
            raise ValueError("All times must be positive.")
        return sample

    @staticmethod
    def _survival_data(rvs):
        if isinstance(rvs, tuple) and len(rvs) == 2 and np.ndim(rvs[0]) != 0:
            times = AbstractLogLogisticGofStatistic._sample(rvs[0])
            events = np.asarray(rvs[1])
            if events.shape != times.shape or not np.all(np.isin(events, [0, 1])):
                raise ValueError("Event indicators must be a matching 1D array of zeros and ones")
            events = events.astype(float)
        else:
            times = AbstractLogLogisticGofStatistic._sample(rvs)
            events = np.ones(times.size)
        if times.size < 3 or np.unique(times).size < 2:
            raise ValueError("At least three observations and two distinct times are required")
        if np.std(np.log(times)) == 0:
            raise ValueError("Log-times are numerically indistinguishable")
        return times, events

    @staticmethod
    def _solve_positive(matrix, rhs):
        try:
            factor = np.linalg.cholesky((matrix + matrix.T) / 2)
            if np.linalg.cond(matrix) > 1e12:
                raise np.linalg.LinAlgError("Ill-conditioned matrix")
            return np.linalg.solve(factor.T, np.linalg.solve(factor, rhs))
        except np.linalg.LinAlgError as exc:
            raise ValueError("Singular or ill-conditioned correction matrix") from exc

    @staticmethod
    def _validate_input(rvs):
        return AbstractLogLogisticGofStatistic._sample(rvs)

    @staticmethod
    def _sorted_sample(rvs):
        return np.sort(AbstractLogLogisticGofStatistic._sample(rvs))

    @staticmethod
    @jit(nopython=True, cache=True)
    def _cdf_values(sample, alpha, beta):
        values = np.empty(len(sample))
        log_alpha = np.log(alpha)
        for i in range(len(sample)):
            z = beta * (np.log(sample[i]) - log_alpha)
            if z >= 0:
                values[i] = 1 / (1 + np.exp(-z))
            else:
                e = np.exp(z)
                values[i] = e / (1 + e)
        return values

    @staticmethod
    @jit(nopython=True, cache=True)
    def _log_probability_values(sample, alpha, beta):
        log_cdf, log_sf = np.empty(len(sample)), np.empty(len(sample))
        log_alpha = np.log(alpha)
        for i in range(len(sample)):
            z = beta * (np.log(sample[i]) - log_alpha)
            log_cdf[i] = -np.logaddexp(0.0, -z)
            log_sf[i] = -np.logaddexp(0.0, z)
        return log_cdf, log_sf

    def _cdf(self, sample):
        return self._cdf_values(sample, self.alpha, self.beta)

    def _log_probabilities(self, sample):
        return self._log_probability_values(sample, self.alpha, self.beta)

    @property
    def alpha(self) -> float:
        """Read alpha by its stable parameter identity."""
        return self._parameters[Distribution.SCALE]

    @property
    def beta(self) -> float:
        """Read beta by its stable parameter identity."""
        return self._parameters[Distribution.SHAPE]

    @staticmethod
    @override
    def distribution() -> type[LogLogisticDistributionDescriptor]:
        """Return the distribution descriptor class."""
        return LogLogisticDistributionDescriptor

    @classmethod
    @override
    def code(cls) -> str:
        """Return the family identifier or the concrete statistic's full identifier."""
        family_code = f"LOG_LOGISTIC_{AbstractGoodnessOfFitStatistic.code()}"
        if "short_code" in cls.__abstractmethods__:
            return family_code
        return f"{cls.short_code()}_{family_code}"


class NikulinLogLogisticGofStatistic(AbstractLogLogisticGofStatistic):
    """Bagdonavicius-Nikulin statistic with fitted log-logistic survival.

    Parameters
    ----------
    parameters : ParameterValues
        An empty distribution schema; all distribution parameters are unknown.
    n_intervals : int, optional
        Number of equal-exposure intervals, at least 2; default 5.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar discrepancy.
    hypothesis()
        Return the fixed parameters; omitted parameters are unknown.

    Notes
    -----
    Both alpha and beta are unknown. Each call fits the right-censored
    likelihood in standardized log-time location and log-scale coordinates.
    Non-informative right censoring is assumed; delta=1 denotes an event.
    H(t)=logaddexp(0, beta*log(t/alpha)); hence h(t)=beta*F(t)/t.
    Section 5 of [1] chooses boundaries with sum_i min(H(t_i), H(a_j))
    equal to j/k times total exposure, so all e_j equal total/k.
    In Sections 3-4, Z=(U-e)/sqrt(n), A=diag(U/n), C contains cell sums
    of event hazard scores divided by n, and I their outer-product mean.
    Return Z'A^-1 Z + W'G^-1 W, W=C A^-1 Z, G=I-C A^-1 C'.
    Scores use equivalent location/log-scale coordinates. G is calculated
    by within-cell centering, without a ridge or pseudoinverse.
    Empty event cells have positive exposure and return infinity. Singular
    or ill-conditioned G is rejected. Small samples need fewer intervals.
    Reject large values. The paper gives an asymptotic chi-square result
    under regularity conditions; finite-sample calibration must reproduce
    censoring, interval count and refitting. Generic Monte Carlo and storage
    are blocked: they cannot express the censoring plan. Old BN_GOF
    calibration values must be discarded. No estimates are retained.

    References
    ----------
    .. [1] V. Bagdonavicius and M. Nikulin (2011), "Chi-squared tests for
       general composite hypotheses from censored samples", Sections 3-5.
       https://doi.org/10.1016/j.crma.2011.01.007
       https://www.numdam.org/item/10.1016/j.crma.2011.01.007.pdf

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({})
    >>> statistic = NikulinLogLogisticGofStatistic(parameters)
    >>> value = statistic.execute_statistic(np.exp(np.linspace(-2, 2, 100)))
    >>> bool(np.isfinite(value))
    True
    """

    def alternative(self):
        return RightAlternative()

    def __init__(self, parameters: ParameterValues, *, n_intervals: int = 5):
        AbstractGoodnessOfFitStatistic.__init__(self, parameters)
        self.n_intervals = self._interval_count(n_intervals)

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    def _validate_storage_calibration(self):
        raise ValueError("Stored calibration omits the censoring plan and interval count")

    @staticmethod
    def short_code() -> str:
        return "BN_GOF"

    @staticmethod
    def _objective(params, y, events):
        loss, location_score, scale_score = NikulinLogLogisticGofStatistic._likelihood_terms(
            params, y, events
        )
        # NumPy's pairwise reduction retains the optimizer's convergence behavior.
        return np.sum(loss), np.array(
            [np.sum(location_score) / np.exp(params[1]), np.sum(scale_score)]
        )

    @staticmethod
    @jit(nopython=True, cache=True)
    def _likelihood_terms(params, y, events):
        mu, log_scale = params
        scale = np.exp(log_scale)
        if not np.isfinite(scale) or scale == 0:
            return np.full(len(y), np.inf), np.full(len(y), np.nan), np.full(len(y), np.nan)
        loss, location_score, scale_score = np.empty(len(y)), np.empty(len(y)), np.empty(len(y))
        for i in range(len(y)):
            z = (y[i] - mu) / scale
            if z >= -700:
                cdf = 1 / (1 + np.exp(-z))
            else:
                e = np.exp(z)
                cdf = e / (1 + e)
            loss[i] = events[i] * (log_scale - z) + (1 + events[i]) * np.logaddexp(0.0, z)
            residual = events[i] - (1 + events[i]) * cdf
            location_score[i] = residual
            scale_score[i] = events[i] + z * residual
        return loss, location_score, scale_score

    @staticmethod
    def _fit_standardized_log_times(times, events):
        logs = np.log(times)
        center, spread = np.mean(logs), np.std(logs)
        y = (logs - center) / spread

        results = [
            minimize(
                NikulinLogLogisticGofStatistic._objective,
                [mu, 0.0],
                args=(y, events),
                jac=True,
                method="BFGS",
                options={"gtol": 1e-8, "maxiter": 2000},
            )
            for mu in (0.0, 1.0)
        ]
        valid = [
            r
            for r in results
            if np.isfinite(r.fun) and np.all(np.isfinite(r.x)) and np.linalg.norm(r.jac) < 1e-5
        ]
        if not valid:
            raise RuntimeError("Censored log-logistic MLE did not converge to a finite solution")
        result = min(valid, key=lambda r: r.fun)
        return (y - result.x[0]) / np.exp(result.x[1])

    def execute_statistic(self, rvs, **kwargs) -> float:
        """Compute the statistic independently for the supplied sample.

        Parameters
        ----------
        rvs : array_like
            Positive finite 1D times (at least three, not constant), or
            (times, delta) with matching binary indicators, 1 for events.
            Nikulin requires at least three events at two distinct times.
        **kwargs : dict
            Accepted for the common interface; unused.

        Returns
        -------
        statistic : float or numpy.float64
            Nonnegative discrepancy; larger values reject the null.
            Positive infinity when any event cell is empty.

        Raises
        ------
        ValueError
            Invalid sample or singular correction matrix.
        RuntimeError
            No finite MLE converges.
        """
        times, events = self._survival_data(rvs)
        if np.sum(events) < 3 or np.unique(times[events == 1]).size < 2:
            raise ValueError("At least three events at two distinct times are required")
        z = self._fit_standardized_log_times(times, events)
        pearson, g, w = self._components(z, events, self.n_intervals)
        if not np.isfinite(pearson):
            return float(pearson)
        return float(pearson + w @ self._solve_positive(g, w))

    @staticmethod
    @jit(nopython=True, cache=True)
    def _components(z, events, n_intervals):
        """Exposure grouping and centered information matrix in linear passes."""
        hazard = np.logaddexp(0.0, z)
        ordered = np.sort(hazard)
        n = len(z)
        knots, exposure = np.zeros(n + 1), np.zeros(n + 1)
        cumulative = 0.0
        for i in range(n):
            knots[i + 1] = ordered[i]
            cumulative += ordered[i]
            exposure[i + 1] = cumulative + (n - 1 - i) * ordered[i]
        total = exposure[-1]
        targets = total * np.arange(1, n_intervals) / n_intervals
        boundaries = np.interp(targets, exposure, knots)
        groups = np.searchsorted(boundaries, hazard, side="left")
        observed = np.zeros(n_intervals)
        score_sums = np.zeros((n_intervals, 2))
        scores = np.empty((n, 2))
        for i in range(n):
            sf = np.exp(-np.logaddexp(0.0, z[i]))
            scores[i, 0], scores[i, 1] = -sf, -1 - z[i] * sf
            j = groups[i]
            observed[j] += events[i]
            score_sums[j, 0] += events[i] * scores[i, 0]
            score_sums[j, 1] += events[i] * scores[i, 1]
        g, w = np.zeros((2, 2)), np.zeros(2)
        if np.any(observed == 0):
            return np.inf, g, w
        means = score_sums / observed.reshape((n_intervals, 1))
        for i in range(n):
            if events[i] == 1:
                d0 = scores[i, 0] - means[groups[i], 0]
                d1 = scores[i, 1] - means[groups[i], 1]
                g[0, 0] += d0 * d0 / n
                g[0, 1] += d0 * d1 / n
                g[1, 1] += d1 * d1 / n
        g[1, 0] = g[0, 1]
        pearson = 0.0
        for j in range(n_intervals):
            difference = observed[j] - total / n_intervals
            pearson += difference * difference / observed[j]
            w += means[j] * difference / np.sqrt(n)
        return pearson, g, w


class MirvalievLogLogisticGofStatistic(AbstractLogLogisticGofStatistic):
    """Mirvaliev moment-fitted statistic on log-transformed complete data.

    Parameters
    ----------
    parameters : ParameterValues
        An empty distribution schema; all distribution parameters are unknown.
    n_intervals : int, optional
        Number of equiprobable cells, at least 2; default 8.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar discrepancy.
    hypothesis()
        Return the fixed parameters; omitted parameters are unknown.

    Notes
    -----
    Apply equations (8)-(10) of [1] to log(X), a logistic location-scale
    family. Fit its mean and standard deviation with divisor n, per call.
    Both log-logistic parameters are unknown. In unit-variance coordinates,
    cell residuals are v_j=(N_j-n/r)/sqrt(n/r), q_j=1/sqrt(r).
    B contains fixed-boundary probability derivatives divided by sqrt(p).
    C contains cell integrals of (Y, Y²-1) divided by sqrt(p),
    V=diag(1,16/5), K=diag(1,2). Set A=I-qq'+C(V-C'C)^-1 C',
    D=C-B K^-1 V, L=V+D' A D. Return v' A v-v' A D L^-1 D' A v.
    Integrals are evaluated by adaptive quadrature. These are logistic
    moments, not normal moments. No MLE or survival-hazard correction is used.
    Reject large values. Log-location/scale invariance justifies simulation
    at alpha=beta=1 with refitting. Preserve n and r. The asymptotic
    chi-square(r-1) statement in [1] is not an exact finite-sample law.
    No built-in log-logistic generator exists; Monte Carlo is external.
    Storage omits r and is blocked; discard old MIRVALIEV calibrations.
    The cited paper treats logistic data; logarithms give the log-logistic
    application. A source for the former censored version was not found;
    censored observations now raise ValueError. No fitted state is retained.

    References
    ----------
    .. [1] N. Pya (2004), "Goodness-of-fit tests for the logistic
       distribution", Mathematical Journal 4(2), 68-75, equations (8)-(10).
       https://math.kz/media/journal/journal2017-06-0725590.pdf

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({})
    >>> statistic = MirvalievLogLogisticGofStatistic(parameters)
    >>> value = statistic.execute_statistic(np.exp(np.linspace(-2, 2, 100)))
    >>> bool(np.isfinite(value))
    True
    """

    def alternative(self):
        return RightAlternative()

    def __init__(self, parameters: ParameterValues, *, n_intervals: int = 8):
        AbstractGoodnessOfFitStatistic.__init__(self, parameters)
        self.n_intervals = self._interval_count(n_intervals)

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    def _validate_storage_calibration(self):
        raise ValueError("Stored calibration does not encode the number of intervals")

    @staticmethod
    def short_code() -> str:
        return "MIRVALIEV"

    def execute_statistic(self, rvs, **kwargs) -> float:
        """Compute the statistic independently for the supplied sample.

        Parameters
        ----------
        rvs : array_like
            Positive finite 1D times (at least three, not constant), or
            (times, delta) with matching binary indicators, 1 for events.
            All indicators must be one; censoring is unsupported.
        **kwargs : dict
            Accepted for the common interface; unused.

        Returns
        -------
        statistic : float or numpy.float64
            Nonnegative discrepancy; larger values reject the null.

        Raises
        ------
        ValueError
            Invalid sample or singular correction matrix.
        RuntimeError
            Numerical failure produces a negative quadratic form.
        """
        times, events = self._survival_data(rvs)
        if not np.all(events == 1):
            raise ValueError("Mirvaliev requires a complete sample; censoring is unsupported")
        r = self.n_intervals
        edges, b, c = self._cell_moments(r)
        residual = self._residuals(times, edges)
        v = np.diag([1.0, 16.0 / 5.0])  # Cov(Y, Y**2), not Gaussian moments.
        moment_jacobian = np.diag([1.0, 2.0])
        a = np.eye(r) - np.ones((r, r)) / r + c @ self._solve_positive(v - c.T @ c, c.T)
        d = c - b @ np.linalg.solve(moment_jacobian, v)
        ell = v + d.T @ a @ d
        ar = a @ residual
        correction = self._solve_positive(ell, d.T @ ar)
        return self.do_execute_statistic(residual, ar, d, correction)

    @staticmethod
    @lru_cache(maxsize=32)
    def _cell_moments(r):
        """Cache distribution-only quadrature, never sample-dependent fitted state."""
        scale = np.sqrt(3) / np.pi
        probabilities = np.linspace(0, 1, r + 1)
        edges = scale * logit(probabilities[1:-1])
        density = probabilities * (1 - probabilities) / scale
        edge_density = np.r_[0.0, edges * density[1:-1], 0.0]
        b = np.sqrt(r) * np.column_stack((-np.diff(density), -np.diff(edge_density)))
        c = np.empty((r, 2))
        for j in range(r):
            for k in range(2):
                c[j, k] = (
                    np.sqrt(r)
                    * quad(
                        MirvalievLogLogisticGofStatistic._moment_integrand,
                        probabilities[j],
                        probabilities[j + 1],
                        args=(k,),
                        epsabs=1e-11,
                    )[0]
                )
        for array in (edges, b, c):
            array.flags.writeable = False
        return edges, b, c

    @staticmethod
    @jit(nopython=True, cache=True)
    def _moment_integrand(u, k):
        y = np.sqrt(3) / np.pi * (np.log(u) - np.log1p(-u))
        return y ** (k + 1) - (1 if k else 0)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _residuals(times, edges):
        logs = np.log(times)
        y = (logs - np.mean(logs)) / np.std(logs)
        r = len(edges) + 1
        counts = np.zeros(r)
        for value in y:
            counts[np.searchsorted(edges, value, side="right")] += 1
        return (counts - len(times) / r) / np.sqrt(len(times) / r)

    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(residual, ar, d, correction):
        result = residual @ ar - (d.T @ ar) @ correction
        if result < -1e-10:
            raise RuntimeError("Numerically negative Mirvaliev quadratic form")
        return float(max(0.0, result))


_EDF_STATISTICS = {
    "KolmogorovSmirnovDPlusLogLogisticGofStatistic": EDFSpec(
        edf.AbstractKSDPlusStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SCALE, Distribution.SHAPE})
            ),
        ),
    ),
    "KolmogorovSmirnovDMinusLogLogisticGofStatistic": EDFSpec(
        edf.AbstractKSDMinusStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SCALE, Distribution.SHAPE})
            ),
        ),
    ),
    "KolmogorovSmirnovLogLogisticGofStatistic": EDFSpec(
        edf.AbstractKSStatistic,
        legacy_ks_options=True,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SCALE, Distribution.SHAPE})
            ),
        ),
    ),
    "AndersonDarlingLogLogisticGofStatistic": EDFSpec(
        edf.AbstractAndersonDarlingStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SCALE, Distribution.SHAPE})
            ),
        ),
    ),
    "CramerVonMisesLogLogisticGofStatistic": EDFSpec(
        edf.AbstractCramerVonMisesStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SCALE, Distribution.SHAPE})
            ),
        ),
    ),
    "LiaoShimokawaLogLogisticGofStatistic": EDFSpec(
        edf.AbstractLiaoShimokawaStatistic,
        short_code="LS",
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SCALE, Distribution.SHAPE})
            ),
        ),
    ),
    "WatsonLogLogisticGofStatistic": EDFSpec(
        edf.AbstractWatsonStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SCALE, Distribution.SHAPE})
            ),
        ),
    ),
    "KuiperLogLogisticGofStatistic": EDFSpec(
        edf.AbstractKuiperStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SCALE, Distribution.SHAPE})
            ),
        ),
    ),
    "KuiperScaledLogLogisticGofStatistic": EDFSpec(
        edf.AbstractKuiperScaledStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SCALE, Distribution.SHAPE})
            ),
        ),
    ),
    "KuiperCorrectedLogLogisticGofStatistic": EDFSpec(
        edf.AbstractKuiperCorrectedStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SCALE, Distribution.SHAPE})
            ),
        ),
    ),
    "ZhangKLogLogisticGofStatistic": EDFSpec(
        edf.AbstractZhangKStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SCALE, Distribution.SHAPE})
            ),
        ),
    ),
    "ZhangALogLogisticGofStatistic": EDFSpec(
        edf.AbstractZhangAStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SCALE, Distribution.SHAPE})
            ),
        ),
    ),
    "ZhangCLogLogisticGofStatistic": EDFSpec(
        edf.AbstractZhangCStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SCALE, Distribution.SHAPE})
            ),
        ),
    ),
}

_FREQUENCY_STATISTICS = {
    "Chi2PearsonLogLogisticGofStatistic": FrequencySpec(
        frequency.AbstractPearsonChi2Statistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SCALE, Distribution.SHAPE})
            ),
        ),
    ),
    "FreemanTukeyLogLogisticGofStatistic": FrequencySpec(
        frequency.AbstractFreemanTukeyStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SCALE, Distribution.SHAPE})
            ),
        ),
    ),
    "ModifiedLogLikelihoodLogLogisticGofStatistic": FrequencySpec(
        frequency.AbstractModifiedLogLikelihoodStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SCALE, Distribution.SHAPE})
            ),
        ),
    ),
    "NeymanChi2LogLogisticGofStatistic": FrequencySpec(
        frequency.AbstractNeymanChi2Statistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SCALE, Distribution.SHAPE})
            ),
        ),
    ),
    "CressieReadLogLogisticGofStatistic": FrequencySpec(
        frequency.AbstractCressieReadStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SCALE, Distribution.SHAPE})
            ),
        ),
    ),
}

__all__ = [
    "AbstractLogLogisticGofStatistic",
    "MirvalievLogLogisticGofStatistic",
    "NikulinLogLogisticGofStatistic",
]
__all__.extend(
    export_edf_statistics(
        globals(),
        AbstractLogLogisticGofStatistic,
        _EDF_STATISTICS,
        log_probabilities=AbstractLogLogisticGofStatistic._log_probabilities,
        cdf=AbstractLogLogisticGofStatistic._cdf,
        sample_validator=AbstractLogLogisticGofStatistic._sorted_sample,
    )
)
__all__.extend(
    export_frequency_statistics(globals(), AbstractLogLogisticGofStatistic, _FREQUENCY_STATISTICS)
)


for _name in (*_EDF_STATISTICS, *_FREQUENCY_STATISTICS):
    globals()[_name].__doc__ += f"""

    Log-logistic hypothesis
    -----------------------
    Pass Distribution.DEFAULT.parse({{'alpha': 1, 'beta': 1}}), with both
    positive parameters fixed before testing. No parameters are fitted.
    Input is a nonempty finite positive one-dimensional complete sample.
    Reject for large values. Calibration must preserve sample size and any
    binning settings; fitting or censoring requires a different null procedure.
    The built-in Monte Carlo resolver has no log-logistic generator.

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({{'alpha': 1, 'beta': 1}})
    >>> statistic = {_name}(parameters)
    >>> value = statistic.execute_statistic([0.1, 0.5, 1.5, 4.0])
    >>> bool(value >= 0)
    True
    """
    if _name in _FREQUENCY_STATISTICS:
        globals()[_name].__doc__ += """
    The keyword binning accepts a BinningStrategy; the default is
    EqualProbabilityBins(). Use EqualProbabilityBins(5) for five fixed cells.
    Zero observed counts give infinity for Neyman and modified log likelihood.
    Binning configuration is included in storage calibration identity.
    """
del _name
