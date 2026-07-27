import numpy as np
from scipy.stats import norm

from pysatl_criterion.distribution.distribution_type import DistributionType
from pysatl_criterion.estimation.base import AbstractParameterEstimator, EstimationMethod


class NormalQmeEstimator(AbstractParameterEstimator):
    @staticmethod
    def distribution_type() -> DistributionType:
        return DistributionType.NORMAL

    @staticmethod
    def method() -> EstimationMethod:
        return EstimationMethod.QME

    def estimate(self, data) -> dict[str, float]:
        sample = np.asarray(data, dtype=float)
        if sample.size < 2:
            raise ValueError("At least 2 elements are required.")

        q25, q50, q75 = np.quantile(sample, [0.25, 0.5, 0.75])

        mean = float(q50)
        z75 = norm.ppf(0.75)
        std = (q75 - q25) / (2.0 * z75)
        var = float(std**2)

        return {"mean": mean, "var": var}


class UniformQmeEstimator(AbstractParameterEstimator):
    @staticmethod
    def distribution_type() -> DistributionType:
        return DistributionType.UNIFORM

    @staticmethod
    def method() -> EstimationMethod:
        return EstimationMethod.QME

    def estimate(self, data) -> dict[str, float]:
        sample = np.asarray(data, dtype=float)
        if sample.size < 2:
            raise ValueError("At least 2 elements are required.")

        q25, q75 = np.quantile(sample, [0.25, 0.75])

        if q25 == q75:
            raise ValueError(
                "Quantiles are equal; cannot estimate continuous uniform distribution."
            )

        a_est = 1.5 * q25 - 0.5 * q75
        b_est = 1.5 * q75 - 0.5 * q25

        min_val, max_val = float(np.min(sample)), float(np.max(sample))
        a = float(min(a_est, min_val))
        b = float(max(b_est, max_val))

        return {"a": a, "b": b}


class LogNormalQmeEstimator(AbstractParameterEstimator):
    @staticmethod
    def distribution_type() -> DistributionType:
        return DistributionType.LOG_NORMAL

    @staticmethod
    def method() -> EstimationMethod:
        return EstimationMethod.QME

    def estimate(self, data) -> dict[str, float]:
        sample = np.asarray(data, dtype=float)
        if sample.size < 2:
            raise ValueError("At least 2 elements are required.")
        if np.any(sample <= 0):
            raise ValueError("All sample elements must be strictly positive.")

        log_sample = np.log(sample)
        q25, q50, q75 = np.quantile(log_sample, [0.25, 0.5, 0.75])

        mean = float(q50)
        z75 = norm.ppf(0.75)
        std = (q75 - q25) / (2.0 * z75)
        var = float(std**2)

        return {"mean": mean, "var": var}


class ExponentialQmeEstimator(AbstractParameterEstimator):
    @staticmethod
    def distribution_type() -> DistributionType:
        return DistributionType.EXPONENTIAL

    @staticmethod
    def method() -> EstimationMethod:
        return EstimationMethod.QME

    def estimate(self, data) -> dict[str, float]:
        sample = np.asarray(data, dtype=float)
        if sample.size < 1:
            raise ValueError("Sample size must be at least 1.")
        if np.any(sample < 0):
            raise ValueError("All elements must be non-negative.")

        sample_median = float(np.median(sample))
        if sample_median <= 0:
            raise ValueError("Sample median must be strictly positive.")

        rate = float(np.log(2.0) / sample_median)
        return {"rate": rate}


class WeibullQmeEstimator(AbstractParameterEstimator):
    @staticmethod
    def distribution_type() -> DistributionType:
        return DistributionType.WEIBULL

    @staticmethod
    def method() -> EstimationMethod:
        return EstimationMethod.QME

    def estimate(self, data) -> dict[str, float]:
        sample = np.asarray(data, dtype=float)
        if sample.size < 2:
            raise ValueError("At least 2 elements are required.")
        if np.any(sample < 0):
            raise ValueError("All sample elements must be non-negative.")

        p1, p2 = 0.25, 0.75
        q_p1, q_med, q_p2 = np.quantile(sample, [p1, 0.5, p2])

        if q_p1 <= 0 or q_p1 == q_p2 or q_med <= 0:
            raise ValueError(
                "Invalid quantiles: ensure data has sufficient variance and positive values."
            )

        num = np.log(-np.log(1.0 - p2)) - np.log(-np.log(1.0 - p1))
        den = np.log(q_p2) - np.log(q_p1)
        k = num / den

        scale = q_med / ((-np.log(0.5)) ** (1.0 / k))

        return {"shape": float(k), "scale": float(scale)}


class CauchyQmeEstimator(AbstractParameterEstimator):
    @staticmethod
    def distribution_type() -> DistributionType:
        return DistributionType.CAUCHY

    @staticmethod
    def method() -> EstimationMethod:
        return EstimationMethod.QME

    def estimate(self, data) -> dict[str, float]:
        sample = np.asarray(data, dtype=float)
        if sample.size < 2:
            raise ValueError("At least 2 elements are required.")

        q25, q50, q75 = np.quantile(sample, [0.25, 0.5, 0.75])

        x0 = float(q50)
        gamma = float((q75 - q25) / 2.0)

        if gamma <= 0:
            raise ValueError(
                "Scale parameter gamma must be strictly positive (insufficient sample variance)."
            )

        return {"loc": x0, "scale": gamma}


class ParetoQmeEstimator(AbstractParameterEstimator):
    @staticmethod
    def distribution_type() -> DistributionType:
        return DistributionType.PARETO

    @staticmethod
    def method() -> EstimationMethod:
        return EstimationMethod.QME

    def estimate(self, data) -> dict[str, float]:
        sample = np.asarray(data, dtype=float)
        if sample.size < 2:
            raise ValueError("At least 2 elements are required.")
        if np.any(sample <= 0):
            raise ValueError("All sample elements must be strictly positive.")

        p1, p2 = 0.25, 0.75
        q_p1, q_p2 = np.quantile(sample, [p1, p2])

        if q_p1 == q_p2:
            raise ValueError("Quantiles are equal, unable to estimate shape parameter.")

        num = np.log(1.0 - p1) - np.log(1.0 - p2)
        den = np.log(q_p2) - np.log(q_p1)
        alpha = num / den

        xm_est = q_p1 * ((1.0 - p1) ** (1.0 / alpha))

        xm = float(min(xm_est, np.min(sample)))

        return {"shape": float(alpha), "scale": xm}
