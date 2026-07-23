from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import beta, cauchy, chi2, f, gamma, rayleigh, t, weibull_min

from pysatl_criterion.distribution.distribution_type import DistributionType
from pysatl_criterion.estimation.base import AbstractParameterEstimator, EstimationMethod


class UniformMleEstimator(AbstractParameterEstimator):
    @staticmethod
    def distribution_type() -> DistributionType:
        return DistributionType.UNIFORM

    @staticmethod
    def method() -> EstimationMethod:
        return EstimationMethod.MLE

    def estimate(self, data) -> dict[str, float]:
        sample = np.asarray(data, dtype=float)
        a = float(np.min(sample))
        b = float(np.max(sample))
        return {"a": a, "b": b}


class NormalMleEstimator(AbstractParameterEstimator):
    @staticmethod
    def distribution_type() -> DistributionType:
        return DistributionType.NORMAL

    @staticmethod
    def method() -> EstimationMethod:
        return EstimationMethod.MLE

    def estimate(self, data) -> dict[str, float]:
        sample = np.asarray(data, dtype=float)
        mean = float(np.mean(sample))
        variance = float(np.var(sample, ddof=0))
        return {"mean": mean, "var": variance}


class LogNormalMleEstimator(AbstractParameterEstimator):
    @staticmethod
    def distribution_type() -> DistributionType:
        return DistributionType.LOG_NORMAL

    @staticmethod
    def method() -> EstimationMethod:
        return EstimationMethod.MLE

    def estimate(self, data) -> dict[str, float]:
        sample = np.asarray(data, dtype=float)
        if np.any(sample <= 0):
            raise ValueError("Lognormal distribution requires strictly positive data.")

        log_data = np.log(sample)
        mean_log = float(np.mean(log_data))
        variance_log = float(np.var(log_data, ddof=0))
        return {"mean": mean_log, "var": variance_log}


class ExponentialMleEstimator(AbstractParameterEstimator):
    @staticmethod
    def distribution_type() -> DistributionType:
        return DistributionType.EXPONENTIAL

    @staticmethod
    def method() -> EstimationMethod:
        return EstimationMethod.MLE

    def estimate(self, data) -> dict[str, float]:
        sample = np.asarray(data, dtype=float)
        if np.any(sample < 0):
            raise ValueError("Exponential distribution requires non-negative data.")

        mean = float(np.mean(sample))
        lam = float(1.0 / mean) if mean > 0 else 0.0
        return {"lam": lam}


class WeibullMleEstimator(AbstractParameterEstimator):
    @staticmethod
    def distribution_type() -> DistributionType:
        return DistributionType.WEIBULL

    @staticmethod
    def method() -> EstimationMethod:
        return EstimationMethod.MLE

    def estimate(self, data) -> dict[str, float]:
        sample = np.asarray(data, dtype=float)
        if np.any(sample <= 0):
            raise ValueError("Weibull distribution requires strictly positive data.")

        shape, _, scale = weibull_min.fit(sample, floc=0)
        return {"a": float(scale), "k": float(shape)}


class GammaMleEstimator(AbstractParameterEstimator):
    @staticmethod
    def distribution_type() -> DistributionType:
        return DistributionType.GAMMA

    @staticmethod
    def method() -> EstimationMethod:
        return EstimationMethod.MLE

    def estimate(self, data) -> dict[str, float]:
        sample = np.asarray(data, dtype=float)
        if np.any(sample <= 0):
            raise ValueError("Gamma distribution requires strictly positive data.")

        shape, _, scale = gamma.fit(sample, floc=0)
        return {"alfa": float(shape), "beta": float(1.0 / scale)}


class BetaMleEstimator(AbstractParameterEstimator):
    @staticmethod
    def distribution_type() -> DistributionType:
        return DistributionType.BETA

    @staticmethod
    def method() -> EstimationMethod:
        return EstimationMethod.MLE

    def estimate(self, data) -> dict[str, float]:
        sample = np.asarray(data, dtype=float)
        if np.any((sample <= 0) | (sample >= 1)):
            raise ValueError(
                "Beta distribution requires values strictly inside the interval (0, 1)."
            )

        alpha, beta_param, _, _ = beta.fit(sample, floc=0, fscale=1)
        return {"a": float(alpha), "b": float(beta_param)}


class CauchyMleEstimator(AbstractParameterEstimator):
    @staticmethod
    def distribution_type() -> DistributionType:
        return DistributionType.CAUCHY

    @staticmethod
    def method() -> EstimationMethod:
        return EstimationMethod.MLE

    def estimate(self, data) -> dict[str, float]:
        sample = np.asarray(data, dtype=float)
        if np.all(sample == sample[0]):
            raise ValueError("Cauchy distribution requires non-degenerate data.")

        loc, scale = cauchy.fit(sample)
        return {"t": float(loc), "s": float(scale)}


class ChiSquaredMleEstimator(AbstractParameterEstimator):
    @staticmethod
    def distribution_type() -> DistributionType:
        return DistributionType.CHI_2

    @staticmethod
    def method() -> EstimationMethod:
        return EstimationMethod.MLE

    def estimate(self, data) -> dict[str, float]:
        sample = np.asarray(data, dtype=float)
        if np.any(sample <= 0):
            raise ValueError("Chi-squared distribution requires strictly positive data.")

        df, _, _ = chi2.fit(sample, floc=0)
        return {"df": float(df)}


class StudentMleEstimator(AbstractParameterEstimator):
    @staticmethod
    def distribution_type() -> DistributionType:
        return DistributionType.STUDENT

    @staticmethod
    def method() -> EstimationMethod:
        return EstimationMethod.MLE

    def estimate(self, data) -> dict[str, float]:
        sample = np.asarray(data, dtype=float)
        if np.all(sample == sample[0]):
            raise ValueError("Student distribution requires non-degenerate data.")

        df, loc, scale = t.fit(sample)
        return {"df": float(df), "loc": float(loc), "scale": float(scale)}


class FisherMleEstimator(AbstractParameterEstimator):
    @staticmethod
    def distribution_type() -> DistributionType:
        return DistributionType.FISHER

    @staticmethod
    def method() -> EstimationMethod:
        return EstimationMethod.MLE

    def estimate(self, data) -> dict[str, float]:
        sample = np.asarray(data, dtype=float)
        if np.any(sample <= 0) or np.all(sample == sample[0]):
            raise ValueError("Fisher distribution requires strictly positive, non-degenerate data.")

        dfn, dfd, loc, scale = f.fit(sample, floc=0)
        return {"dfn": float(dfn), "dfd": float(dfd), "loc": float(loc), "scale": float(scale)}


class RayleighMleEstimator(AbstractParameterEstimator):
    @staticmethod
    def distribution_type() -> DistributionType:
        return DistributionType.RAYLEIGH

    @staticmethod
    def method() -> EstimationMethod:
        return EstimationMethod.MLE

    def estimate(self, data) -> dict[str, float]:
        sample = np.asarray(data, dtype=float)
        if np.any(sample < 0):
            raise ValueError("Rayleigh distribution requires non-negative data.")

        _, scale = rayleigh.fit(sample, floc=0)
        return {"scale": float(scale)}


class WignerMleEstimator(AbstractParameterEstimator):
    @staticmethod
    def distribution_type() -> DistributionType:
        return DistributionType.WIGNER

    @staticmethod
    def method() -> EstimationMethod:
        return EstimationMethod.MLE

    def estimate(self, data) -> dict[str, float]:
        sample = np.asarray(data, dtype=float)
        if np.all(sample == sample[0]):
            raise ValueError("Wigner distribution requires non-degenerate data.")

        max_abs_x = float(np.max(np.abs(sample)))
        n = sample.size

        def neg_log_likelihood(R: float) -> float:
            if R <= max_abs_x:
                return np.inf

            inner = R**2 - sample**2
            if np.any(inner <= 0):
                return np.inf
            return -(n * np.log(2.0 / (np.pi * R**2)) + 0.5 * np.sum(np.log(inner)))

        upper_bound = max(max_abs_x * 10.0, max_abs_x + 100.0)
        result = minimize_scalar(
            neg_log_likelihood, bounds=(max_abs_x + 1e-6, upper_bound), method="bounded"
        )
        if not result.success:
            raise ValueError("Failed to estimate Wigner parameter.")

        return {"R": float(result.x)}


class ParetoMleEstimator(AbstractParameterEstimator):
    @staticmethod
    def distribution_type() -> DistributionType:
        return DistributionType.PARETO

    @staticmethod
    def method() -> EstimationMethod:
        return EstimationMethod.MLE

    def estimate(self, data) -> dict[str, float]:
        sample = np.asarray(data, dtype=float)
        if np.any(sample <= 0):
            raise ValueError("Pareto distribution requires strictly positive data.")

        x_m = float(np.min(sample))
        n = sample.size
        log_ratios = np.log(sample / x_m)
        sum_log_ratios = float(np.sum(log_ratios))
        if sum_log_ratios == 0:
            raise ValueError("Pareto MLE is undefined for the current sample.")

        alpha = float(n / sum_log_ratios)
        return {"x_m": x_m, "alpha": alpha}


class LaplaceMleEstimator(AbstractParameterEstimator):
    @staticmethod
    def distribution_type() -> DistributionType:
        return DistributionType.LAPLACE

    @staticmethod
    def method() -> EstimationMethod:
        return EstimationMethod.MLE

    def estimate(self, data) -> dict[str, float]:
        sample = np.asarray(data, dtype=float)
        mu = float(np.median(sample))
        b = float(np.mean(np.abs(sample - mu)))
        if b == 0:
            raise ValueError("Laplace distribution requires non-degenerate data.")

        return {"t": mu, "s": b}


class DiscreteUniformMleEstimator(AbstractParameterEstimator):
    @staticmethod
    def distribution_type() -> DistributionType:
        return DistributionType.UNIFORM

    @staticmethod
    def method() -> EstimationMethod:
        return EstimationMethod.MLE

    def estimate(self, data) -> dict[str, float]:
        sample = np.asarray(data, dtype=float)
        a = float(np.min(sample))
        b = float(np.max(sample))
        return {"a": a, "b": b}


class BernoulliMleEstimator(AbstractParameterEstimator):
    @staticmethod
    def distribution_type() -> DistributionType:
        return DistributionType.UNIFORM

    @staticmethod
    def method() -> EstimationMethod:
        return EstimationMethod.MLE

    def estimate(self, data) -> dict[str, float]:
        sample = np.asarray(data, dtype=float)
        if not np.all(np.isin(sample, [0, 1])):
            raise ValueError("Bernoulli distribution requires binary data.")

        p = float(np.mean(sample))
        return {"p": p}


class BinomialMleEstimator(AbstractParameterEstimator):
    def __init__(self, n: int) -> None:
        if n <= 0:
            raise ValueError("Number of trials n must be positive.")
        self.n = n

    @staticmethod
    def distribution_type() -> DistributionType:
        return DistributionType.UNIFORM

    @staticmethod
    def method() -> EstimationMethod:
        return EstimationMethod.MLE

    def estimate(self, data) -> dict[str, float]:
        sample = np.asarray(data, dtype=float)
        if np.any((sample < 0) | (sample > self.n) | (~np.equal(np.mod(sample, 1), 0))):
            raise ValueError("Binomial distribution requires integer counts in [0, n].")

        p = float(np.sum(sample) / (sample.size * self.n)) if self.n > 0 else 0.0
        return {"p": p}


class PoissonMleEstimator(AbstractParameterEstimator):
    @staticmethod
    def distribution_type() -> DistributionType:
        return DistributionType.UNIFORM

    @staticmethod
    def method() -> EstimationMethod:
        return EstimationMethod.MLE

    def estimate(self, data) -> dict[str, float]:
        sample = np.asarray(data, dtype=float)
        if np.any(sample < 0) or not np.all(np.floor(sample) == sample):
            raise ValueError("Poisson distribution requires non-negative integer count data.")

        lam = float(np.mean(sample))
        return {"lam": lam}


class GeometricMleEstimator(AbstractParameterEstimator):
    @staticmethod
    def distribution_type() -> DistributionType:
        return DistributionType.UNIFORM

    @staticmethod
    def method() -> EstimationMethod:
        return EstimationMethod.MLE

    def estimate(self, data) -> dict[str, float]:
        sample = np.asarray(data, dtype=float)
        if np.any(sample < 1) or not np.all(np.floor(sample) == sample):
            raise ValueError("Geometric distribution requires positive integer data.")

        n = sample.size
        sum_x = float(np.sum(sample))
        p = float(n / sum_x) if sum_x > 0 else 0.0
        return {"p": p}
