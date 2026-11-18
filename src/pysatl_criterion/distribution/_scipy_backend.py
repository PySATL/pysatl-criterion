"""Private SciPy adapter. All backend parameter mappings live here."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
from scipy.special import expit

from pysatl_criterion.distribution.distribution_type import DistributionType
from pysatl_criterion.distribution.evaluation import BoundDistribution, ProbabilityResult
from pysatl_criterion.distribution.parameters import ParameterValues


def _result(value: ArrayLike) -> ProbabilityResult:
    array = np.asarray(value, dtype=np.float64)
    return float(array) if array.ndim == 0 else array


@dataclass(frozen=True)
class _ScipyDistribution:
    _distribution: Any

    def cdf(self, x: ArrayLike) -> ProbabilityResult:
        return _result(self._distribution.cdf(x))

    def sf(self, x: ArrayLike) -> ProbabilityResult:
        return _result(self._distribution.sf(x))

    def logcdf(self, x: ArrayLike) -> ProbabilityResult:
        return _result(self._distribution.logcdf(x))

    def logsf(self, x: ArrayLike) -> ProbabilityResult:
        return _result(self._distribution.logsf(x))


@dataclass(frozen=True)
class _LogLogistic:
    """Evaluate Fisk probabilities without forming x/scale or its powers."""

    scale: float
    shape: float

    def _log_odds(self, x: ArrayLike):
        values = np.asarray(x, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            z = self.shape * (np.log(values) - np.log(self.scale))
        return np.where(values <= 0, -np.inf, z)

    def cdf(self, x: ArrayLike) -> ProbabilityResult:
        return _result(expit(self._log_odds(x)))

    def sf(self, x: ArrayLike) -> ProbabilityResult:
        return _result(expit(-self._log_odds(x)))

    def logcdf(self, x: ArrayLike) -> ProbabilityResult:
        with np.errstate(invalid="ignore"):
            return _result(-np.logaddexp(0, -self._log_odds(x)))

    def logsf(self, x: ArrayLike) -> ProbabilityResult:
        with np.errstate(invalid="ignore"):
            return _result(-np.logaddexp(0, self._log_odds(x)))


@dataclass(frozen=True)
class _NormalMixture:
    _probability: float
    _component: Any

    def _evaluate(self, method: str, x: ArrayLike) -> ProbabilityResult:
        p = self._probability
        # Avoid zero-weight terms, including log(0) and 0 * infinity.
        if p == 0:
            return _result(getattr(stats.norm, method)(x))
        if p == 1:
            return _result(getattr(self._component, method)(x))
        first = getattr(stats.norm, method)(x)
        second = getattr(self._component, method)(x)
        if method.startswith("log"):
            return _result(np.logaddexp(math.log1p(-p) + first, math.log(p) + second))
        return _result((1 - p) * first + p * second)

    def cdf(self, x: ArrayLike) -> ProbabilityResult:
        return self._evaluate("cdf", x)

    def sf(self, x: ArrayLike) -> ProbabilityResult:
        return self._evaluate("sf", x)

    def logcdf(self, x: ArrayLike) -> ProbabilityResult:
        return self._evaluate("logcdf", x)

    def logsf(self, x: ArrayLike) -> ProbabilityResult:
        return self._evaluate("logsf", x)


def _uniform(p: dict[str, float]) -> Any:
    if p["lower"] >= p["upper"]:
        raise ValueError("Uniform requires lower < upper")
    return stats.uniform(loc=p["lower"], scale=p["upper"] - p["lower"])


def _discrete_uniform(p: dict[str, float]) -> Any:
    if p["lower"] > p["upper"]:
        raise ValueError("Discrete uniform requires lower <= upper")
    return stats.randint(int(p["lower"]), int(p["upper"]) + 1)


def _truncated_normal(p: dict[str, float]) -> Any:
    if p["lower"] >= p["upper"]:
        raise ValueError("Truncated normal requires lower < upper")
    sigma = math.sqrt(p["variance"])
    return stats.truncnorm(
        (p["lower"] - p["mean"]) / sigma,
        (p["upper"] - p["mean"]) / sigma,
        loc=p["mean"],
        scale=sigma,
    )


def _hyperbolic(p: dict[str, float]) -> Any:
    if abs(p["skewness"]) >= p["shape"]:
        raise ValueError("Hyperbolic requires abs(skewness) < shape")
    shape = p["shape"] * p["scale"]
    skewness = p["skewness"] * p["scale"]
    if shape <= abs(skewness):
        raise ValueError("Hyperbolic coordinates cannot be represented by the backend")
    return stats.genhyperbolic(
        1,
        shape,
        skewness,
        loc=p["location"],
        scale=p["scale"],
    )


def _inverse_gaussian(p: dict[str, float]) -> Any:
    shape = p["mean"] / p["shape"]
    if shape <= 0:
        raise ValueError("Inverse Gaussian coordinates cannot be represented by the backend")
    return stats.invgauss(shape, scale=p["shape"])


def _rice(p: dict[str, float]) -> Any:
    if p["distance"] < 0 or p["scale"] <= 0:
        raise ValueError("Continuous Rice evaluation requires distance >= 0 and scale > 0")
    return stats.rice(p["distance"] / p["scale"], scale=p["scale"])


def _lognormal(p: dict[str, float]) -> Any:
    try:
        scale = math.exp(p["log_location"])
    except OverflowError as error:
        raise ValueError("LogNormal scale is not representable") from error
    if scale == 0:
        raise ValueError("LogNormal scale is not representable")
    return stats.lognorm(p["log_scale"], scale=scale)


# Keys below are stable ParameterSpec ID suffixes, never public input aliases.
_FACTORIES: dict[DistributionType, Callable[[dict[str, float]], Any]] = {
    DistributionType.FISHER: lambda p: stats.f(p["dfn"], p["dfd"]),
    DistributionType.RAYLEIGH: lambda p: stats.rayleigh(scale=p["scale"]),
    DistributionType.WIGNER: lambda p: stats.semicircular(loc=p["location"], scale=p["radius"]),
    DistributionType.DISCRETE_UNIFORM: _discrete_uniform,
    DistributionType.BERNOULLI: lambda p: stats.bernoulli(p["probability"]),
    DistributionType.BINOMIAL: lambda p: stats.binom(p["trials"], p["probability"]),
    DistributionType.POISSON: lambda p: stats.poisson(p["rate"]),
    DistributionType.GEOMETRIC: lambda p: stats.geom(p["probability"]),
    DistributionType.NORMAL: lambda p: stats.norm(p["mean"], math.sqrt(p["variance"])),
    DistributionType.EXPONENTIAL: lambda p: stats.expon(scale=1 / p["rate"]),
    DistributionType.WEIBULL: lambda p: stats.weibull_min(p["shape"], scale=p["scale"]),
    DistributionType.EXPONENTIATED_WEIBULL: lambda p: stats.exponweib(
        p["exponent"], p["shape"], scale=p["scale"]
    ),
    DistributionType.UNIFORM: _uniform,
    DistributionType.STUDENT: lambda p: stats.t(p["df"], loc=p["location"], scale=p["scale"]),
    DistributionType.GAMMA: lambda p: stats.gamma(p["shape"], scale=1 / p["rate"]),
    DistributionType.BETA: lambda p: stats.beta(p["alpha"], p["beta"]),
    DistributionType.LOG_NORMAL: _lognormal,
    DistributionType.CAUCHY: lambda p: stats.cauchy(p["location"], p["scale"]),
    DistributionType.CHI_2: lambda p: stats.chi2(p["df"]),
    DistributionType.GOMPERTZ: lambda p: stats.gompertz(p["shape"], scale=p["scale"]),
    DistributionType.GUMBEL: lambda p: stats.gumbel_r(p["location"], p["scale"]),
    DistributionType.INV_GAUSS: _inverse_gaussian,
    DistributionType.LAPLACE: lambda p: stats.laplace(p["location"], p["scale"]),
    DistributionType.HYPERBOLIC: _hyperbolic,
    DistributionType.TRUNC_NORMAL: _truncated_normal,
    DistributionType.LOGISTIC: lambda p: stats.logistic(p["location"], p["scale"]),
    DistributionType.RICE: _rice,
    DistributionType.TUKEY: lambda p: stats.tukeylambda(p["shape"]),
    DistributionType.PARETO: lambda p: stats.pareto(p["shape"], scale=p["scale"]),
    DistributionType.INVERSE_GAMMA: lambda p: stats.invgamma(p["shape"], scale=p["scale"]),
}


def bind_distribution(kind: DistributionType, values: ParameterValues) -> BoundDistribution:
    """Adapt canonical, complete descriptor parameters to the private backend."""
    p = {parameter.id.split(".", 1)[1]: value for parameter, value in values.items()}
    if kind == DistributionType.LOG_LOGISTIC:
        return _LogLogistic(p["scale"], p["shape"])
    if kind in (
        DistributionType.LO_CON_NORMAL,
        DistributionType.MIX_CON_NORMAL,
        DistributionType.SCALE_CON_NORMAL,
    ):
        location = p.get("location", p.get("mean", 0.0))
        return _NormalMixture(p["probability"], stats.norm(location, p.get("scale", 1.0)))
    factory = _FACTORIES.get(kind)
    if factory is None:
        raise NotImplementedError(f"Probability evaluation is not implemented for {kind}")
    distribution = factory(p)
    # Reject overflowing/underflowing coordinate conversions at bind time.
    if (
        not all(math.isfinite(value) for value in (*distribution.args, *distribution.kwds.values()))
        or distribution.kwds.get("scale", 1) <= 0
    ):
        raise ValueError("Parameters cannot be represented by the probability backend")
    return _ScipyDistribution(distribution)
