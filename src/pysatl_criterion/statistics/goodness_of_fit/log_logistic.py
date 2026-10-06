from abc import ABC

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import expit, logit
from typing_extensions import override

from pysatl_criterion import DistributionType
from pysatl_criterion.statistics import AbstractGoodnessOfFitStatistic
from pysatl_criterion.statistics.alternative import AlternativeType
from pysatl_criterion.statistics.goodness_of_fit.common import (
    ADStatistic,
    Chi2Statistic,
    CrammerVonMisesStatistic,
    KSStatistic,
)
from pysatl_criterion.statistics.hypothesis import GoodnessOfFitHypothesis


def _positive_scalar(value, name):
    if np.ndim(value) != 0 or np.iscomplexobj(value) or isinstance(value, (bool, np.bool_, str)):
        raise ValueError(f"{name} must be a finite positive scalar")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive scalar") from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} parameter must be strictly greater than zero. And finite.")
    return value


def _interval_count(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 2:
        raise ValueError("At least two bins are required; the count must be an integer")
    return int(value)


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


def _log_odds(sample, alpha, beta):
    # Avoid x/alpha and (x/alpha)**beta overflowing before taking logarithms.
    with np.errstate(over="ignore"):
        return beta * (np.log(sample) - np.log(alpha))


def _survival_data(rvs):
    if isinstance(rvs, tuple) and len(rvs) == 2 and np.ndim(rvs[0]) != 0:
        times = _sample(rvs[0])
        events = np.asarray(rvs[1])
        if events.shape != times.shape or not np.all(np.isin(events, [0, 1])):
            raise ValueError("Event indicators must be a matching 1D array of zeros and ones")
        events = events.astype(float)
    else:
        times = _sample(rvs)
        events = np.ones(times.size)
    if times.size < 3 or np.unique(times).size < 2:
        raise ValueError("At least three observations and two distinct times are required")
    if np.std(np.log(times)) == 0:
        raise ValueError("Log-times are numerically indistinguishable")
    return times, events


def _solve_positive(matrix, rhs):
    try:
        factor = np.linalg.cholesky((matrix + matrix.T) / 2)
        if np.linalg.cond(matrix) > 1e12:
            raise np.linalg.LinAlgError("Ill-conditioned matrix")
        return np.linalg.solve(factor.T, np.linalg.solve(factor, rhs))
    except np.linalg.LinAlgError as exc:
        raise ValueError("Singular or ill-conditioned correction matrix") from exc


class AbstractLogLogisticGofStatistic(AbstractGoodnessOfFitStatistic, ABC):
    def __init__(self, alpha=1, beta=1):
        self.alpha = _positive_scalar(alpha, "Alpha")
        self.beta = _positive_scalar(beta, "Beta")

    @override
    def hypothesis(self) -> GoodnessOfFitHypothesis:
        return GoodnessOfFitHypothesis({"alpha": self.alpha, "beta": self.beta})

    @staticmethod
    @override
    def distribution() -> DistributionType:
        return DistributionType.LOG_LOGISTIC

    @staticmethod
    @override
    def code():
        return f"LOG_LOGISTIC_{AbstractGoodnessOfFitStatistic.code()}"


class KolmogorovSmirnovLogLogisticGofStatistic(AbstractLogLogisticGofStatistic, KSStatistic):
    """Kolmogorov-Smirnov distance to a fixed log-logistic CDF.

    Parameters
    ----------
    alternative_type : AlternativeType, optional
        TWO_TAILED (default), RIGHT for D+, or LEFT for D-. This selects
        the CDF deviation, not the rejection tail (always RIGHT).
    alpha : float, optional
        Fixed positive finite scale, default 1.
    beta : float, optional
        Fixed positive finite shape, default 1.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar discrepancy.
    hypothesis()
        Return the fixed parameters; omitted parameters are unknown.

    Notes
    -----
    The null CDF is F(x) = expit(beta * (log(x) - log(alpha))), x > 0.
    Parameters are fixed, never fitted. Reject for large values. Continuous
    iid null calibration is parameter-free by the probability transform;
    rounded observations require calibration of the rounding process.
    There is no built-in log-logistic generator; use external simulation.
    With u_i = F(x_(i)), D+ = max(i/n-u_i), D- = max(u_i-(i-1)/n).
    TWO_TAILED returns max(D+, D-). Reference [1] is the general continuous
    criterion applied through this CDF. Stored calibration cannot encode
    LEFT/RIGHT; use Monte Carlo for those directions.

    References
    ----------
    .. [1] N. Smirnov (1948), "Table for Estimating the Goodness of Fit of
       Empirical Distributions", Ann. Math. Statist. 19, 279-281.
       https://doi.org/10.1214/aoms/1177730256

    Examples
    --------
    >>> statistic = KolmogorovSmirnovLogLogisticGofStatistic()
    >>> value = statistic.execute_statistic([0.5, 1.0, 2.0])
    >>> bool(np.isfinite(value))
    True
    """

    def __init__(
        self,
        alternative_type: AlternativeType = AlternativeType.TWO_TAILED,
        alpha=1.0,
        beta=1.0,
    ):
        AbstractLogLogisticGofStatistic.__init__(self, alpha=alpha, beta=beta)
        if not isinstance(alternative_type, AlternativeType):
            raise TypeError("alternative_type must be an AlternativeType")
        KSStatistic.__init__(self, alternative_type)

    def _validate_storage_calibration(self):
        if self.alternative_type != AlternativeType.TWO_TAILED:
            raise ValueError("Stored KS calibration does not encode the CDF direction")

    @staticmethod
    @override
    def short_code() -> str:
        return "KS"

    @staticmethod
    @override
    def code() -> str:
        short_code = KolmogorovSmirnovLogLogisticGofStatistic.short_code()
        return f"{short_code}_{AbstractLogLogisticGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic independently for the supplied sample.

        Parameters
        ----------
        rvs : array_like
            Nonempty finite positive one-dimensional observations. Repeats
            and constant samples are allowed; minimum size is one.
        **kwargs : dict
            Accepted for the common interface; unused.

        Returns
        -------
        statistic : float or numpy.float64
            Nonnegative discrepancy; larger values reject the null.

        Raises
        ------
        ValueError
            Invalid sample.
        """
        rvs = np.sort(_sample(rvs))

        cdf_vals = expit(_log_odds(rvs, self.alpha, self.beta))
        return KSStatistic.do_execute_statistic(self, rvs, cdf_vals)


class AndersonDarlingLogLogisticGofStatistic(AbstractLogLogisticGofStatistic, ADStatistic):
    """Anderson-Darling statistic for a fixed log-logistic CDF.

    Parameters
    ----------
    alpha : float, optional
        Fixed positive finite scale, default 1.
    beta : float, optional
        Fixed positive finite shape, default 1.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar discrepancy.
    hypothesis()
        Return the fixed parameters; omitted parameters are unknown.

    Notes
    -----
    The null CDF is F(x) = expit(beta * (log(x) - log(alpha))), x > 0.
    Parameters are fixed, never fitted. Reject for large values. Continuous
    iid null calibration is parameter-free by the probability transform;
    rounded observations require calibration of the rounding process.
    There is no built-in log-logistic generator; use external simulation.
    A² = -n - sum((2*i-1)/n * (log(u_i)+log(1-u_(n+1-i)))),
    where u_i = F(x_(i)). Logs are computed directly from log odds without
    clipping. Floating-point overflow of beta times the log ratio may yield
    positive infinity; finite extreme log odds retain finite tail logarithms.
    Reference [1] gives the general EDF criterion applied via this CDF.

    References
    ----------
    .. [1] T. W. Anderson and D. A. Darling (1952), "Asymptotic Theory of
       Certain Goodness of Fit Criteria Based on Stochastic Processes",
       Ann. Math. Statist. 23, 193-212. https://doi.org/10.1214/aoms/1177729437

    Examples
    --------
    >>> statistic = AndersonDarlingLogLogisticGofStatistic()
    >>> value = statistic.execute_statistic([0.5, 1.0, 2.0])
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def short_code() -> str:
        return "AD"

    @staticmethod
    @override
    def code() -> str:
        short_code = AndersonDarlingLogLogisticGofStatistic.short_code()
        return f"{short_code}_{AbstractLogLogisticGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic independently for the supplied sample.

        Parameters
        ----------
        rvs : array_like
            Nonempty finite positive one-dimensional observations. Repeats
            and constant samples are allowed; minimum size is one.
        **kwargs : dict
            Accepted for the common interface; unused.

        Returns
        -------
        statistic : float or numpy.float64
            Nonnegative discrepancy; larger values reject the null.
            Positive infinity if log-odds multiplication overflows.

        Raises
        ------
        ValueError
            Invalid sample.
        """
        sorted_rvs = np.sort(_sample(rvs))

        z = _log_odds(sorted_rvs, self.alpha, self.beta)
        log_cdf = -np.logaddexp(0, -z)
        log_sf = -np.logaddexp(0, z)
        return super().do_execute_statistic(sorted_rvs, log_cdf=log_cdf, log_sf=log_sf)


class CramerVonMisesLogLogisticGofStatistic(
    AbstractLogLogisticGofStatistic, CrammerVonMisesStatistic
):
    """Cramer-von Mises statistic for a fixed log-logistic CDF.

    Parameters
    ----------
    alpha : float, optional
        Fixed positive finite scale, default 1.
    beta : float, optional
        Fixed positive finite shape, default 1.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar discrepancy.
    hypothesis()
        Return the fixed parameters; omitted parameters are unknown.

    Notes
    -----
    The null CDF is F(x) = expit(beta * (log(x) - log(alpha))), x > 0.
    Parameters are fixed, never fitted. Reject for large values. Continuous
    iid null calibration is parameter-free by the probability transform;
    rounded observations require calibration of the rounding process.
    There is no built-in log-logistic generator; use external simulation.
    W² = 1/(12*n) + sum((F(x_(i))-(2*i-1)/(2*n))**2).
    Reference [1] treats quadratic EDF criteria, here applied via the known
    log-logistic CDF. No estimated-parameter correction is used.

    References
    ----------
    .. [1] T. W. Anderson and D. A. Darling (1952), "Asymptotic Theory of
       Certain Goodness of Fit Criteria Based on Stochastic Processes",
       Ann. Math. Statist. 23, 193-212. https://doi.org/10.1214/aoms/1177729437

    Examples
    --------
    >>> statistic = CramerVonMisesLogLogisticGofStatistic()
    >>> value = statistic.execute_statistic([0.5, 1.0, 2.0])
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def short_code() -> str:
        return "CVM"

    @staticmethod
    @override
    def code() -> str:
        short_code = CramerVonMisesLogLogisticGofStatistic.short_code()
        return f"{short_code}_{AbstractLogLogisticGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic independently for the supplied sample.

        Parameters
        ----------
        rvs : array_like
            Nonempty finite positive one-dimensional observations. Repeats
            and constant samples are allowed; minimum size is one.
        **kwargs : dict
            Accepted for the common interface; unused.

        Returns
        -------
        statistic : float or numpy.float64
            Nonnegative discrepancy; larger values reject the null.

        Raises
        ------
        ValueError
            Invalid sample.
        """
        sorted_rvs = np.sort(_sample(rvs))

        cdf_vals = expit(_log_odds(sorted_rvs, self.alpha, self.beta))
        return CrammerVonMisesStatistic.do_execute_statistic(self, sorted_rvs, cdf_vals)


class AbstractBinnedLogLogisticGofStatistic(AbstractLogLogisticGofStatistic, Chi2Statistic, ABC):
    lambda_value: float = 1.0

    def __init__(self, bins: int = 8, alpha: float = 1.0, beta: float = 2.0):
        self.bins = _interval_count(bins)
        AbstractLogLogisticGofStatistic.__init__(self, alpha=alpha, beta=beta)

    def _validate_storage_calibration(self):
        raise ValueError("Stored calibration does not encode the number of bins")

    def _counts_and_expected(self, rvs):
        sample = _sample(rvs)
        z = _log_odds(sample, self.alpha, self.beta)
        edges = logit(np.arange(1, self.bins) / self.bins)
        counts = np.bincount(np.searchsorted(edges, z, side="right"), minlength=self.bins)
        return counts, np.full(self.bins, sample.size / self.bins)

    @override
    def execute_statistic(self, rvs, **kwargs):
        counts, expected = self._counts_and_expected(rvs)
        return float(
            Chi2Statistic.do_execute_statistic(self, counts, expected, lambda_=self.lambda_value)
        )


class Chi2PearsonLogLogisticGofStatistic(AbstractBinnedLogLogisticGofStatistic):
    """Pearson statistic in fixed equiprobable log-logistic cells.

    Parameters
    ----------
    bins : int, optional
        Number of cells, at least 2; default 8.
    alpha : float, optional
        Fixed positive finite scale, default 1.
    beta : float, optional
        Fixed positive finite shape, default 2.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar discrepancy.
    hypothesis()
        Return the fixed parameters; omitted parameters are unknown.

    Notes
    -----
    The null CDF is F(x) = expit(beta * (log(x) - log(alpha))), x > 0.
    Parameters are fixed, never fitted. Reject for large values. Continuous
    iid null calibration is parameter-free by the probability transform;
    rounded observations require calibration of the rounding process.
    There is no built-in log-logistic generator; use external simulation.
    X² = sum((N_j-n/bins)**2/(n/bins)). Cells are left-closed at the
    internal theoretical quantiles, compared in log-odds coordinates.
    Empty cells and repeated observations are allowed. The chi-square law
    with bins-1 degrees of freedom is only a large-expected-count limit.
    Monte Carlo must preserve bins. Storage calibration is blocked because
    its key omits bins. Reference [1] supplies the general cell-count test.

    References
    ----------
    .. [1] K. Pearson (1900), "On the criterion that a given system of
       deviations from the probable ...", Philosophical Magazine 50, 157-175.
       https://doi.org/10.1080/14786440009463897

    Examples
    --------
    >>> statistic = Chi2PearsonLogLogisticGofStatistic()
    >>> value = statistic.execute_statistic([0.5, 1.0, 2.0])
    >>> bool(np.isfinite(value))
    True
    """

    lambda_value = 1.0

    @staticmethod
    @override
    def short_code() -> str:
        return "CHI2_PEARSON"

    @staticmethod
    @override
    def code() -> str:
        short_code = Chi2PearsonLogLogisticGofStatistic.short_code()
        return f"{short_code}_{AbstractLogLogisticGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic independently for the supplied sample.

        Parameters
        ----------
        rvs : array_like
            Nonempty finite positive one-dimensional observations. Repeats
            and constant samples are allowed; minimum size is one.
        **kwargs : dict
            Accepted for the common interface; unused.

        Returns
        -------
        statistic : float or numpy.float64
            Nonnegative discrepancy; larger values reject the null.

        Raises
        ------
        ValueError
            Invalid sample.
        """
        return super().execute_statistic(rvs)


class NikulinLogLogisticGofStatistic(AbstractLogLogisticGofStatistic, Chi2Statistic):
    """Bagdonavicius-Nikulin statistic with fitted log-logistic survival.

    Parameters
    ----------
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
    >>> statistic = NikulinLogLogisticGofStatistic()
    >>> value = statistic.execute_statistic(np.exp(np.linspace(-2, 2, 100)))
    >>> bool(np.isfinite(value))
    True
    """

    def __init__(self, n_intervals: int = 5):
        self.n_intervals = _interval_count(n_intervals)

    def hypothesis(self) -> GoodnessOfFitHypothesis:
        return GoodnessOfFitHypothesis({})

    def _validate_storage_calibration(self):
        raise ValueError("Stored calibration omits the censoring plan and interval count")

    @staticmethod
    def short_code() -> str:
        return "BN_GOF"

    @staticmethod
    def code() -> str:
        return f"BN_GOF_{AbstractLogLogisticGofStatistic.code()}"

    @staticmethod
    def _fit_standardized_log_times(times, events):
        logs = np.log(times)
        center, spread = np.mean(logs), np.std(logs)
        y = (logs - center) / spread

        def objective(params):
            mu, log_scale = params
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                scale = np.exp(log_scale)
                z = (y - mu) / scale
                loss = np.sum(events * (log_scale - z) + (1 + events) * np.logaddexp(0, z))
                residual = events - (1 + events) * expit(z)
                gradient = np.array([np.sum(residual) / scale, np.sum(events + z * residual)])
            return loss, gradient

        results = [
            minimize(
                objective,
                [mu, 0.0],
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
        times, events = _survival_data(rvs)
        if np.sum(events) < 3 or np.unique(times[events == 1]).size < 2:
            raise ValueError("At least three events at two distinct times are required")
        z = self._fit_standardized_log_times(times, events)
        hazard = np.logaddexp(0, z)
        ordered = np.sort(hazard)
        n = times.size
        # g(h) = sum_i min(H_i, h) is piecewise linear in cumulative hazard.
        knots = np.r_[0.0, ordered]
        exposure = np.r_[0.0, np.cumsum(ordered) + np.arange(n - 1, -1, -1) * ordered]
        total = exposure[-1]
        targets = total * np.arange(1, self.n_intervals) / self.n_intervals
        boundaries = np.interp(targets, exposure, knots)
        groups = np.searchsorted(boundaries, hazard, side="left")
        observed = np.bincount(groups, weights=events, minlength=self.n_intervals)
        expected = total / self.n_intervals
        # The published observed-information estimator requires nonempty event cells.
        if np.any(observed == 0):
            return float("inf")
        sf = expit(-z)
        scores = np.column_stack((-sf, -1 - z * sf))
        c = (
            np.stack(
                [
                    np.sum(scores[(groups == j) & (events == 1)], axis=0)
                    for j in range(self.n_intervals)
                ],
                axis=1,
            )
            / n
        )
        a = observed / n
        residual = (observed - expected) / np.sqrt(n)
        # Within-cell centering avoids subtracting nearly equal information matrices.
        means = (c / a).T
        centered = scores - means[groups]
        g = (centered.T * events) @ centered / n
        w = c @ (residual / a)
        correction = w @ _solve_positive(g, w)
        return float(np.sum(residual**2 / a) + correction)


class MirvalievLogLogisticGofStatistic(AbstractLogLogisticGofStatistic, Chi2Statistic):
    """Mirvaliev moment-fitted statistic on log-transformed complete data.

    Parameters
    ----------
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
    >>> statistic = MirvalievLogLogisticGofStatistic()
    >>> value = statistic.execute_statistic(np.exp(np.linspace(-2, 2, 100)))
    >>> bool(np.isfinite(value))
    True
    """

    def __init__(self, n_intervals: int = 8):
        self.n_intervals = _interval_count(n_intervals)

    def hypothesis(self) -> GoodnessOfFitHypothesis:
        return GoodnessOfFitHypothesis({})

    def _validate_storage_calibration(self):
        raise ValueError("Stored calibration does not encode the number of intervals")

    @staticmethod
    def short_code() -> str:
        return "MIRVALIEV"

    @staticmethod
    def code() -> str:
        return f"MIRVALIEV_{AbstractLogLogisticGofStatistic.code()}"

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
        times, events = _survival_data(rvs)
        if not np.all(events == 1):
            raise ValueError("Mirvaliev requires a complete sample; censoring is unsupported")
        logs = np.log(times)
        y = (logs - np.mean(logs)) / np.std(logs, ddof=0)
        r = self.n_intervals
        scale = np.sqrt(3) / np.pi  # Standard logistic with unit variance.
        probabilities = np.linspace(0, 1, r + 1)
        edges = scale * logit(probabilities[1:-1])
        observed = np.bincount(np.searchsorted(edges, y, side="right"), minlength=r)
        residual = (observed - times.size / r) / np.sqrt(times.size / r)
        # B: derivatives of cell probabilities at fixed endpoints, (mu, sd)=(0, 1).
        density = probabilities * (1 - probabilities) / scale
        edge_density = np.r_[0.0, edges * density[1:-1], 0.0]
        b = np.sqrt(r) * np.column_stack((-np.diff(density), -np.diff(edge_density)))
        c = np.empty((r, 2))
        for j in range(r):
            for k in range(2):
                c[j, k] = (
                    np.sqrt(r)
                    * quad(
                        lambda u, k=k: (scale * logit(u)) ** (k + 1) - (1 if k else 0),
                        probabilities[j],
                        probabilities[j + 1],
                        epsabs=1e-11,
                    )[0]
                )
        v = np.diag([1.0, 16.0 / 5.0])  # Cov(Y, Y**2), not Gaussian moments.
        moment_jacobian = np.diag([1.0, 2.0])
        a = np.eye(r) - np.ones((r, r)) / r + c @ _solve_positive(v - c.T @ c, c.T)
        d = c - b @ np.linalg.solve(moment_jacobian, v)
        ell = v + d.T @ a @ d
        ar = a @ residual
        result = residual @ ar - (d.T @ ar) @ _solve_positive(ell, d.T @ ar)
        if result < -1e-10:
            raise RuntimeError("Numerically negative Mirvaliev quadratic form")
        return float(max(0.0, result))
