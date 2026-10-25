from abc import ABC
from numbers import Integral, Real

import numpy as np
import scipy.stats as scipy_stats
from numba import jit
from scipy.integrate import quad
from scipy.optimize import fmin
from scipy.spatial import cKDTree
from scipy.special import betaln, digamma, gammaln, kve, polygamma
from typing_extensions import override

from pysatl_criterion.distribution import (
    StudentDistributionDescriptor as Distribution,
)
from pysatl_criterion.distribution import StudentDistributionDescriptor as Student
from pysatl_criterion.distribution.parameters import HypothesisSupport
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


class AbstractStudentGofStatistic(AbstractGoodnessOfFitStatistic, ABC):
    """
    Abstract base class for Student's t-distribution goodness-of-fit statistics.
    """

    @property
    def df(self) -> float:
        """Read df by its stable parameter identity."""
        return self._parameters[Distribution.DF]

    @property
    def loc(self) -> float:
        """Read loc by its stable parameter identity."""
        return self._parameters[Distribution.LOCATION]

    @property
    def scale(self) -> float:
        """Read scale by its stable parameter identity."""
        return self._parameters[Distribution.SCALE]

    @staticmethod
    @override
    def distribution() -> type[Student]:
        """Return the distribution descriptor class."""
        return Student

    @classmethod
    @override
    def code(cls) -> str:
        """Return the family identifier or the concrete statistic's full identifier."""
        family_code = f"STUDENT_{AbstractGoodnessOfFitStatistic.code()}"
        if "short_code" in cls.__abstractmethods__:
            return family_code
        return f"{cls.short_code()}_{family_code}"

    def _standardize(self, sample):
        # Recover representable differences when subtraction alone overflows.
        with np.errstate(over="ignore", invalid="ignore"):
            difference = sample - self.loc
            z = difference / self.scale
            overflow = np.isinf(difference)
            z[overflow] = sample[overflow] / self.scale - self.loc / self.scale
        if not np.all(np.isfinite(z)):
            raise ValueError("Standardized observations exceed floating-point range")
        return z

    @staticmethod
    @jit(nopython=True, cache=True)
    def _shape_sample(sample):
        """Rescale validated observations without overflowing sample moments."""
        # Sorting makes moment calculations reproducible under permutation.
        sample = np.sort(sample)
        magnitude = np.max(np.abs(sample))
        if magnitude == 0 or sample[0] == sample[-1]:
            raise ValueError("A nonconstant sample is required to estimate scale")
        sample = sample / magnitude
        sample -= np.mean(sample)
        spread = np.max(np.abs(sample))
        if spread == 0:
            raise ValueError("Sample scale cannot be represented")
        return sample / spread

    @staticmethod
    @jit(nopython=True, cache=True)
    def _renyi_entropy(distances, delta, gamma_ratio):
        """Reduce kNN distances, retaining accuracy as Renyi approaches Shannon."""
        logs = np.log(distances)
        center = np.mean(logs)
        total = 0.0
        for value in logs:
            total += np.expm1(delta * (value - center))
        log_moment = np.log1p(total / len(logs)) / delta + center
        return np.log(2 * (len(distances) - 1)) + log_moment - gamma_ratio

    def _log_probabilities(self, standardized):
        logcdf = scipy_stats.t.logcdf(standardized, self.df)
        logsf = scipy_stats.t.logsf(standardized, self.df)
        if not np.all(np.isfinite(logcdf)) or not np.all(np.isfinite(logsf)):
            raise ValueError("Student tail probabilities exceed floating-point range")
        return logcdf, logsf


def _student_log_probabilities(criterion, sample):
    return criterion._log_probabilities(criterion._standardize(sample))


def _student_shape_sample(rvs):
    """Validate and rescale before centering, avoiding overflow in sample moments."""
    raw = np.asarray(rvs)
    if raw.dtype.kind not in "iuf" or (np.ma.isMaskedArray(rvs) and np.any(rvs.mask)):
        raise ValueError("Sample must contain finite real numbers without masks")
    sample = np.asarray(raw, dtype=float)
    if sample.ndim != 1 or sample.size < 2 or not np.all(np.isfinite(sample)):
        raise ValueError("Sample must be one-dimensional, finite and have at least two values")
    return AbstractStudentGofStatistic._shape_sample(sample)


def _positive_student_setting(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite positive number")
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def _student_integer_setting(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _student_quad(function, *, epsabs, epsrel, limit):
    """Do not silently use a quadrature result that failed its error criterion."""
    result = quad(function, 0, np.inf, epsabs=epsabs, epsrel=epsrel, limit=limit, full_output=1)
    value, error = result[:2]
    if len(result) != 3 or not np.isfinite(value) or error > max(epsabs, epsrel * abs(value)):
        raise ValueError("Student characteristic-function quadrature did not converge")
    return float(value)


def _student_cf(t, df):
    """Characteristic function of a standard (scale=1) Student variable."""
    if t == 0:
        return 1.0
    order = df / 2
    argument = np.sqrt(df) * abs(t)
    scaled_bessel = kve(order, argument)
    if np.isfinite(scaled_bessel) and scaled_bessel > 0:
        log_cf = (
            (1 - order) * np.log(2)
            - gammaln(order)
            + order * np.log(argument)
            + np.log(scaled_bessel)
            - argument
        )
        if np.isfinite(log_cf) and log_cf <= 1e-10:
            return float(np.exp(min(0.0, log_cf)))
    # The Bessel function can overflow for large order near the origin even
    # though the CF is finite. Integrate the standardized density in that case.
    value = _student_quad(
        lambda x: 2 * scipy_stats.t.pdf(x, df) * np.cos(t * x),
        epsabs=1e-10,
        epsrel=1e-10,
        limit=250,
    )
    if not -1e-10 <= value <= 1 + 1e-10:
        raise ValueError("Student characteristic function cannot be represented accurately")
    return float(np.clip(value, 0, 1))


class MeintanisStudentGofStatistic(AbstractStudentGofStatistic):
    """Univariate i.i.d. ECF test of Meintanis et al. (2024).

    Only df is fixed, with df > 2. Location and scatter are estimated by
    sample mean and (df-2)/df times the population sample variance (ddof=0),
    as in equation (3.3) and the paper's i.i.d. experiments. For
    Y_j=(X_j-mean(X))/sqrt((df-2)*var(X)), return
    n/(2*pi) * integral |mean(exp(i*t*Y))-phi_df(t)|**2 * phi_weight_df(t) dt,
    where phi_v(t)=2**(1-v/2)*|t|**(v/2)*K_(v/2)(|t|)/Gamma(v/2).

    Parameters
    ----------
    parameters : ParameterValues
        Distribution.DEFAULT.parse({'df': nu}); location and scale unknown.
    weight_df : float or None, optional
        Positive degrees of freedom for the weight. None uses the null df,
        equation (2.8); a different value gives the weight variant in (4.7).
    epsabs, epsrel : float, optional
        Positive quadrature tolerances below one, applied to the returned
        statistic. Defaults are 1e-8 and 1e-7.
    limit : int, optional
        Maximum number of quadrature subintervals, default 250.

    Notes
    -----
    This is equation (2.4) evaluated directly in one dimension with the
    paper's weight phi/(2*pi), retaining its normalization. Numerical
    integration failure raises ValueError. Constants and non-finite samples
    are invalid; ties are allowed. No fitted state is stored. Large values
    reject. Monte Carlo calibration must refit each replicate and retain df
    and the weight; unknown df, multivariate data and GARCH fitting are not
    implemented by this i.i.d. criterion.

    References
    ----------
    Meintanis, Milosevic, Obradovic and Veljovic (2024), Journal of Time
    Series Analysis 45, 298-319, equations (2.3)-(2.8), (3.3), (4.7).
    https://doi.org/10.1111/jtsa.12713
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset({Distribution.DF})),)

    def __init__(self, parameters, *, weight_df=None, epsabs=1e-8, epsrel=1e-7, limit=250):
        super().__init__(parameters)
        if self.df <= 2:
            raise ValueError("Moment-based Student ECF requires df > 2")
        self.weight_df = (
            None if weight_df is None else _positive_student_setting(weight_df, "weight_df")
        )
        self.epsabs = _positive_student_setting(epsabs, "epsabs")
        self.epsrel = _positive_student_setting(epsrel, "epsrel")
        if self.epsabs >= 1 or self.epsrel >= 1:
            raise ValueError("Quadrature tolerances must be less than one")
        self.limit = _student_integer_setting(limit, "limit")

    @staticmethod
    def short_code():
        return "MEINTANIS_ECF"

    def alternative(self):
        return RightAlternative()

    def configuration(self):
        return {
            **super().configuration(),
            "estimation": "moments_ddof0",
            "weight_df": self.weight_df,
            "epsabs": self.epsabs,
            "epsrel": self.epsrel,
            "limit": self.limit,
        }

    def execute_statistic(self, rvs, **kwargs):
        sample = _student_shape_sample(rvs)
        z = sample / np.sqrt(np.mean(sample**2))
        weight_df = self.df if self.weight_df is None else self.weight_df
        # Change t=sqrt(weight_df)*u so the quadrature scale stays near one
        # when the null and weight approach a normal distribution.
        y = z * (np.sqrt(weight_df) / np.sqrt(self.df - 2))
        null_scale = np.sqrt(weight_df) / np.sqrt(self.df)
        factor = len(sample) * np.sqrt(weight_df) / np.pi
        if not np.all(np.isfinite(y)) or not np.isfinite(factor):
            raise ValueError("Student ECF standardization exceeds floating-point range")

        def integrand(u):
            weight = _student_cf(u, weight_df)
            if weight == 0:
                return 0.0
            theoretical = weight if weight_df == self.df else _student_cf(u * null_scale, self.df)
            return self.do_execute_statistic(y, u, theoretical, weight)

        return factor * _student_quad(
            integrand, epsabs=self.epsabs / factor, epsrel=self.epsrel, limit=self.limit
        )

    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(y, u, theoretical, weight) -> float:
        """Evaluate the ECF integrand without allocating phase and trig arrays."""
        real = 0.0
        imaginary = 0.0
        for value in y:
            phase = u * value
            real += np.cos(phase)
            imaginary += np.sin(phase)
        real = real / len(y) - theoretical
        imaginary /= len(y)
        return (real**2 + imaginary**2) * weight


def _student_max_renyi_entropy(df):
    """Maximum Renyi entropy at q=1-2/(df+1), for unit variance."""
    # Beta recurrence avoids subtracting nearly equal log-beta terms as q -> 1.
    return betaln(df / 2, 0.5) + 0.5 * np.log(df - 2) + ((df + 1) / 2) * np.log1p(1 / (df - 2))


def _student_knn_entropy(sample, k, df):
    distances = cKDTree(sample[:, None]).query(sample[:, None], k=[k + 1])[0][:, 0]
    if np.any(distances <= 0) or not np.all(np.isfinite(distances)):
        raise ValueError("Nearest-neighbour distances must be positive and finite")
    delta = 2 / (df + 1)  # 1-q, retained explicitly even when q rounds to one.
    if delta < 1e-5:
        gamma_ratio = digamma(k) + delta * polygamma(1, k) / 2
        gamma_ratio += delta**2 * polygamma(2, k) / 6
    else:
        gamma_ratio = (gammaln(k + delta) - gammaln(k)) / delta
    return AbstractStudentGofStatistic._renyi_entropy(distances, delta, gamma_ratio)


class CadirciRenyiStudentGofStatistic(AbstractStudentGofStatistic):
    """Univariate kNN Renyi-entropy statistic W of Cadirci et al. (2025).

    Parameters
    ----------
    parameters : ParameterValues
        Distribution.DEFAULT.parse({'df': nu}) with nu > 2. Location and
        scale are unknown. The null fixes nu; it is not fitted to the data.
    k : int, optional
        Neighbour rank, default 3 (used in the paper's numerical comparisons).
        Must satisfy 1 <= k < sample size.

    Notes
    -----
    For q=1-2/(nu+1), return H_max(q)-H_hat(q), equation (20).
    H_max is the Student Renyi entropy with scatter
    (1-2/nu)*var(X, ddof=1). H_hat uses kth-neighbour distances rho_i:
    log(mean([2*(n-1)*C_k*rho_i]**(1-q)))/(1-q),
    C_k=[Gamma(k)/Gamma(k+1-q)]**(1/(1-q)).
    The calculation uses unit sample variance for numerical stability;
    the entropy difference is invariant to location, scale and reflection
    in one dimension. No estimates are retained on the instance.

    Large values reject. Finite-sample values can be negative and are not
    clipped. Require finite, distinct observations and n > k. Calibrate by
    Monte Carlo at the specified df, preserving k and recomputing moments
    and neighbours for every replicate. The formula is defined for nu > 2;
    mean-square convergence needs the stronger moment assumptions in
    Theorem 6, and is not asserted throughout that whole range.

    References
    ----------
    Cadirci, Evans, Leonenko, Makogin and Seleznjev (2025), Statistical tests
    based on Renyi entropy estimation, sections 2.1, 3, 4 and 5.2.
    https://arxiv.org/abs/2502.08654
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Distribution.DEFAULT, frozenset({Distribution.DF})),)

    def __init__(self, parameters, *, k=3):
        super().__init__(parameters)
        if self.df <= 2:
            raise ValueError("Student Renyi maximum-entropy test requires df > 2")
        self.k = _student_integer_setting(k, "k")

    @staticmethod
    def short_code():
        return "CADIRCI_RENYI"

    def alternative(self):
        return RightAlternative()

    def configuration(self):
        return {**super().configuration(), "k": self.k, "estimation": "variance_ddof1"}

    def execute_statistic(self, rvs, **kwargs):
        sample = _student_shape_sample(rvs)
        if len(sample) <= self.k:
            raise ValueError("k must be smaller than the sample size")
        if np.any(np.diff(sample) <= 0):
            raise ValueError("Renyi kNN requires distinct observations")
        standardized = sample / np.std(sample, ddof=1)
        value = _student_max_renyi_entropy(self.df) - _student_knn_entropy(
            standardized, self.k, self.df
        )
        if not np.isfinite(value):
            raise ValueError("Student Renyi entropy exceeds floating-point range")
        return float(value)


def _student_mle_optimizer(function, initial, args=(), disp=0):
    """SciPy fit optimizer that checks convergence instead of discarding status."""
    fitted, value, _, _, status = fmin(
        function,
        initial,
        args=args,
        disp=disp,
        full_output=True,
        xtol=1e-9,
        ftol=1e-9,
        maxiter=10000,
        maxfun=20000,
    )
    if status != 0 or not np.isfinite(value):
        raise ValueError("Student maximum-likelihood fit did not converge")
    return fitted


class LequesneRenyiStudentGofStatistic(AbstractStudentGofStatistic):
    """Lequesne (2015) Renyi-entropy statistics T1 and T2, equations (7)-(10).

    Parameters
    ----------
    parameters : ParameterValues
        Use Distribution.DEFAULT.parse({'df': nu}) for T1 (known nu > 2,
        unknown location and scale), or parse({}) for T2 (all unknown).
    k : int, optional
        Nearest-neighbour rank, default 3, with 1 <= k < sample size.

    Notes
    -----
    T1 uses sigma_hat**2=(nu-2)/nu * var(X, ddof=1). The paper permits
    a consistent variance estimator; this choice coincides with the
    univariate Cadirci statistic. T2 fits nu, location and scale by numerical
    maximum likelihood on each call, then sets q=(nu_hat-1)/(nu_hat+1).
    Both return h_q(nu, sigma_hat**2)-H_hat(n,k,q), with the corrected
    nearest-neighbour estimate in equation (10). No fitted state is retained.

    Samples must be finite, one-dimensional and distinct, with n > k;
    T2 additionally requires at least four observations. A failed fit or
    fitted nu <= 2 raises ValueError: the full-line maximum-entropy
    construction requires q > 1/3. Theorem 2 assumes nu > 4; its convergence
    conclusion is not asserted for the entire computational domain nu > 2.
    Negative finite-sample values are preserved.

    Large values reject, consistently with the entropy deficit and Theorem 2.
    The lower-tail inequality printed on page 493 contradicts this direction;
    the printed critical values are not used. With known nu, Monte Carlo
    calibration uses standard t(nu) and refits scale on each replicate. With
    unknown nu, external shape-specific parametric bootstrap is required:
    simulate at the observed fit and refit all parameters on every replicate.

    References
    ----------
    J. Lequesne (2015), A goodness-of-fit test of Student distributions based
    on Renyi entropy, AIP Conference Proceedings 1641, 487-494, (7)-(10).
    https://doi.org/10.1063/1.4906014
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (
            HypothesisSupport(Distribution.DEFAULT, frozenset({Distribution.DF})),
            HypothesisSupport(Distribution.DEFAULT, frozenset()),
        )

    def __init__(self, parameters, *, k=3):
        super().__init__(parameters)
        if Distribution.DF in self._parameters and self.df <= 2:
            raise ValueError("Lequesne maximum-entropy test requires df > 2")
        self.k = _student_integer_setting(k, "k")

    @staticmethod
    def short_code():
        return "LEQUESNE_RENYI"

    def alternative(self):
        return RightAlternative()

    def configuration(self):
        return {
            **super().configuration(),
            "k": self.k,
            "estimation": (
                "variance_ddof1" if Distribution.DF in self._parameters else "mle_scipy_t"
            ),
        }

    def _validate_storage_calibration(self):
        if Distribution.DF not in self._parameters:
            raise ValueError(
                "Unknown Student df requires external parametric bootstrap at an explicit "
                "simulation df, refitting all parameters on every replicate"
            )

    def execute_statistic(self, rvs, **kwargs):
        sample = _student_shape_sample(rvs)
        if len(sample) <= self.k:
            raise ValueError("k must be smaller than the sample size")
        if np.any(np.diff(sample) <= 0):
            raise ValueError("Lequesne Renyi kNN requires distinct observations")
        standardized = sample / np.std(sample, ddof=1)
        if Distribution.DF in self._parameters:
            df = self.df
            theoretical = _student_max_renyi_entropy(df)
        else:
            if len(sample) < 4:
                raise ValueError("Student MLE requires at least four observations")
            df, location, scale = scipy_stats.t.fit(
                standardized,
                5.0,
                loc=0.0,
                scale=np.sqrt(3 / 5),
                optimizer=_student_mle_optimizer,
            )
            if not np.all(np.isfinite([df, location, scale])) or df <= 2 or scale <= 0:
                raise ValueError("Student MLE must have finite df > 2 and positive scale")
            # h_q(nu, scale**2), rather than a moment-matched replacement:
            # the fitted scale in T2 is explicitly the MLE in equation (8).
            theoretical = (
                betaln(df / 2, 0.5)
                + 0.5 * np.log(df)
                + np.log(scale)
                + ((df + 1) / 2) * np.log1p(1 / (df - 2))
            )
        value = theoretical - _student_knn_entropy(standardized, self.k, df)
        if not np.isfinite(value):
            raise ValueError("Lequesne Renyi entropy exceeds floating-point range")
        return float(value)


_EDF_STATISTICS = {
    "KolmogorovSmirnovDPlusStudentGofStatistic": EDFSpec(
        edf.AbstractKSDPlusStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        ),
    ),
    "KolmogorovSmirnovDMinusStudentGofStatistic": EDFSpec(
        edf.AbstractKSDMinusStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        ),
    ),
    "KolmogorovSmirnovStudentGofStatistic": EDFSpec(
        edf.AbstractKSStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        ),
    ),
    "AndersonDarlingStudentGofStatistic": EDFSpec(
        edf.AbstractAndersonDarlingStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        ),
    ),
    "CramerVonMisesStudentGofStatistic": EDFSpec(
        edf.AbstractCramerVonMisesStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        ),
    ),
    "LiaoShimokawaStudentGofStatistic": EDFSpec(
        edf.AbstractLiaoShimokawaStatistic,
        short_code="LS",
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        ),
    ),
    "WatsonStudentGofStatistic": EDFSpec(
        edf.AbstractWatsonStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        ),
    ),
    "KuiperStudentGofStatistic": EDFSpec(
        edf.AbstractKuiperStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        ),
    ),
    "KuiperScaledStudentGofStatistic": EDFSpec(
        edf.AbstractKuiperScaledStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        ),
    ),
    "KuiperCorrectedStudentGofStatistic": EDFSpec(
        edf.AbstractKuiperCorrectedStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        ),
    ),
    "ZhangKStudentGofStatistic": EDFSpec(
        edf.AbstractZhangKStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        ),
    ),
    "ZhangZaStudentGofStatistic": EDFSpec(
        edf.AbstractZhangAStatistic,
        short_code="ZHANG_ZA",
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        ),
    ),
    "ZhangZcStudentGofStatistic": EDFSpec(
        edf.AbstractZhangCStatistic,
        short_code="ZHANG_ZC",
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        ),
    ),
}

_FREQUENCY_STATISTICS = {
    "ChiSquareStudentGofStatistic": FrequencySpec(
        frequency.AbstractPearsonChi2Statistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        ),
    ),
    "FreemanTukeyStudentGofStatistic": FrequencySpec(
        frequency.AbstractFreemanTukeyStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        ),
    ),
    "ModifiedLogLikelihoodStudentGofStatistic": FrequencySpec(
        frequency.AbstractModifiedLogLikelihoodStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        ),
    ),
    "NeymanChi2StudentGofStatistic": FrequencySpec(
        frequency.AbstractNeymanChi2Statistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        ),
    ),
    "CressieReadStudentGofStatistic": FrequencySpec(
        frequency.AbstractCressieReadStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        ),
    ),
}

__all__ = [
    "AbstractStudentGofStatistic",
    "CadirciRenyiStudentGofStatistic",
    "LequesneRenyiStudentGofStatistic",
    "MeintanisStudentGofStatistic",
]
__all__.extend(
    export_edf_statistics(
        globals(),
        AbstractStudentGofStatistic,
        _EDF_STATISTICS,
        log_probabilities=_student_log_probabilities,
    )
)
__all__.extend(
    export_frequency_statistics(globals(), AbstractStudentGofStatistic, _FREQUENCY_STATISTICS)
)

for _name in (*_EDF_STATISTICS, *_FREQUENCY_STATISTICS):
    globals()[_name].__doc__ += """

    Student hypothesis
    ------------------
    Specify df, loc and scale with Distribution.DEFAULT.parse(...).
    All three parameters are fixed; no parameters are fitted to the sample.
    Observations must be finite real values on the whole real line.
    """
    if _name in _FREQUENCY_STATISTICS:
        globals()[_name].__doc__ += """
    The binning keyword accepts a BinningStrategy; the default is
    EqualProbabilityBins(). Calibration must use the same binning strategy.
    """
del _name
