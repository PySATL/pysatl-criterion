from __future__ import annotations

import numpy as np

from scipy.optimize import root_scalar
from scipy.special import gamma

from pysatl_criterion.distribution.distribution_type import DistributionType
from pysatl_criterion.estimation.base import AbstractParameterEstimator, EstimationMethod


class NormalMmEstimator(AbstractParameterEstimator):
    @staticmethod
    def distribution_type() -> DistributionType:
        return DistributionType.NORMAL

    @staticmethod
    def method() -> EstimationMethod:
        return EstimationMethod.MM

    def estimate(self, data: list[float] | tuple[float, ...] | np.ndarray) -> dict[str, float]:
        sample = np.asarray(data, dtype=float)
        mean = float(np.mean(sample))
        variance = float(np.var(sample, ddof=0))
        return {"mean": mean, "var": variance}

class UniformMmEstimator(AbstractParameterEstimator):
    @staticmethod
    def distribution_type() -> DistributionType:
        return DistributionType.UNIFORM

    @staticmethod
    def method() -> EstimationMethod:
        return EstimationMethod.MM

    def estimate(self, data: list[float] | tuple[float, ...] | np.ndarray) -> dict[str, float]:
        sample = np.asarray(data, dtype=float)
        if sample.size < 2:
            raise ValueError("Для оценки параметров равномерного распределения нужно хотя бы 2 элемента.")

        mean = float(np.mean(sample))
        var = float(np.var(sample, ddof=0))
        
        if var < 0:
            raise ValueError("Дисперсия выборки не может быть отрицательной.")

        half_width = np.sqrt(3.0 * var)
        a = mean - half_width
        b = mean + half_width

        return {"a": float(a), "b": float(b)}

class LogNormalMmEstimator(AbstractParameterEstimator):
    @staticmethod
    def distribution_type() -> DistributionType:
        return DistributionType.LOG_NORMAL

    @staticmethod
    def method() -> EstimationMethod:
        return EstimationMethod.MM

    def estimate(self, data: list[float] | tuple[float, ...] | np.ndarray) -> dict[str, float]:
        sample = np.asarray(data, dtype=float)
        if sample.size < 2:
            raise ValueError("Для оценки параметров логнормального распределения нужно хотя бы 2 элемента.")
        if np.any(sample <= 0):
            raise ValueError("Все элементы выборки для логнормального распределения должны быть строго больше 0.")

        mean = float(np.mean(sample))
        var = float(np.var(sample, ddof=0))

        var_log = float(np.log(1.0 + var / (mean ** 2)))
        mean_log = float(np.log(mean) - 0.5 * var_log)

        return {"mean": mean_log, "var": var_log}
    
class ExponentialMmEstimator(AbstractParameterEstimator):
    @staticmethod
    def distribution_type() -> DistributionType:
        return DistributionType.EXPONENTIAL

    @staticmethod
    def method() -> EstimationMethod:
        return EstimationMethod.MM

    def estimate(self, data: list[float] | tuple[float, ...] | np.ndarray) -> dict[str, float]:
        sample = np.asarray(data, dtype=float)
        if sample.size < 1:
            raise ValueError("Выборка не может быть пустой.")
        if np.any(sample < 0):
            raise ValueError("Элементы выборки для экспоненциального распределения должны быть неотрицательными.")

        mean = float(np.mean(sample))
        if mean <= 0:
            raise ValueError("Среднее значение выборки должно быть строго больше 0.")

        rate = float(1.0 / mean)
        return {"rate": rate}

class WeibullMmEstimator(AbstractParameterEstimator):
    @staticmethod
    def distribution_type() -> DistributionType:
        return DistributionType.WEIBULL

    @staticmethod
    def method() -> EstimationMethod:
        return EstimationMethod.MM

    def estimate(self, data: list[float] | tuple[float, ...] | np.ndarray) -> dict[str, float]:
        sample = np.asarray(data, dtype=float)
        if sample.size < 2:
            raise ValueError("Для оценки параметров распределения Вейбулла нужно хотя бы 2 элемента.")
        if np.any(sample <= 0):
            raise ValueError("Все элементы выборки для Вейбулла должны быть строго больше 0.")

        mean = float(np.mean(sample))
        var = float(np.var(sample, ddof=0))

        target_ratio = 1.0 + (var / (mean ** 2))
        def equation(k: float) -> float:
            if k <= 0:
                return 1e9
            g1 = gamma(1.0 + 1.0 / k)
            g2 = gamma(1.0 + 2.0 / k)
            return (g2 / (g1 ** 2)) - target_ratio

        sol = root_scalar(equation, bracket=[1e-3, 100.0], method='brentq')
        if not sol.converged:
            raise RuntimeError("Не удалось определить параметр k для распределения Вейбулла.")

        shape = float(sol.root)
        scale = float(mean / gamma(1.0 + 1.0 / shape))

        return {"shape": shape, "scale": scale}

class GammaMmEstimator(AbstractParameterEstimator):
    @staticmethod
    def distribution_type() -> DistributionType:
        return DistributionType.GAMMA

    @staticmethod
    def method() -> EstimationMethod:
        return EstimationMethod.MM

    def estimate(self, data: list[float] | tuple[float, ...] | np.ndarray) -> dict[str, float]:
        sample = np.asarray(data, dtype=float)
        if sample.size < 2:
            raise ValueError("Для оценки параметров Гамма-распределения нужно хотя бы 2 элемента.")
        if np.any(sample <= 0):
            raise ValueError("Все элементы выборки для Гамма-распределения должны быть строго больше 0.")

        mean = float(np.mean(sample))
        var = float(np.var(sample, ddof=0))

        if var <= 0:
            raise ValueError("Дисперсия выборки должна быть больше 0.")

        # Оценка параметров формы (shape) и масштаба (scale)
        shape = float((mean ** 2) / var)
        scale = float(var / mean)

        return {"shape": shape, "scale": scale}
    
class BetaMmEstimator(AbstractParameterEstimator):
    @staticmethod
    def distribution_type() -> DistributionType:
        return DistributionType.BETA

    @staticmethod
    def method() -> EstimationMethod:
        return EstimationMethod.MM

    def estimate(self, data: list[float] | tuple[float, ...] | np.ndarray) -> dict[str, float]:
        sample = np.asarray(data, dtype=float)
        if sample.size < 2:
            raise ValueError("At least 2 elements are required to estimate Beta distribution parameters.")
        if np.any((sample <= 0) | (sample >= 1)):
            raise ValueError("All sample elements for Beta distribution must lie strictly in the interval (0, 1).")

        mean = float(np.mean(sample))
        var = float(np.var(sample, ddof=0))

        max_possible_var = mean * (1.0 - mean)
        if var >= max_possible_var or var <= 0:
            raise ValueError("Sample variance must be strictly positive and less than mean * (1 - mean).")

        common_factor = (max_possible_var / var) - 1.0
        alpha = float(mean * common_factor)
        beta = float((1.0 - mean) * common_factor)

        return {"alpha": alpha, "beta": beta}

class Chi2MmEstimator(AbstractParameterEstimator):
    @staticmethod
    def distribution_type() -> DistributionType:
        return DistributionType.CHI_2

    @staticmethod
    def method() -> EstimationMethod:
        return EstimationMethod.MM

    def estimate(self, data: list[float] | tuple[float, ...] | np.ndarray) -> dict[str, float]:
        sample = np.asarray(data, dtype=float)
        if sample.size < 1:
            raise ValueError("Sample cannot be empty.")
        if np.any(sample < 0):
            raise ValueError("All elements for Chi-Square distribution must be non-negative.")

        mean = float(np.mean(sample))
        if mean <= 0:
            raise ValueError("Sample mean must be strictly positive to estimate degrees of freedom.")

        return {"df": mean}

class StudentMmEstimator(AbstractParameterEstimator):
    @staticmethod
    def distribution_type() -> DistributionType:
        return DistributionType.STUDENT

    @staticmethod
    def method() -> EstimationMethod:
        return EstimationMethod.MM

    def estimate(self, data: list[float] | tuple[float, ...] | np.ndarray) -> dict[str, float]:
        sample = np.asarray(data, dtype=float)
        if sample.size < 2:
            raise ValueError("At least 2 elements are required to estimate Student's t-distribution parameters.")

        var = float(np.var(sample, ddof=0))
        if var <= 1.0:
            raise ValueError(
                "Sample variance must be strictly greater than 1.0 "
                "to estimate degrees of freedom for Student's t-distribution using Method of Moments."
            )

        df = float((2.0 * var) / (var - 1.0))
        return {"df": df}

class FisherMmEstimator(AbstractParameterEstimator):
    @staticmethod
    def distribution_type() -> DistributionType:
        return DistributionType.FISHER  

    @staticmethod
    def method() -> EstimationMethod:
        return EstimationMethod.MM

    def estimate(self, data: list[float] | tuple[float, ...] | np.ndarray) -> dict[str, float]:
        sample = np.asarray(data, dtype=float)
        if sample.size < 2:
            raise ValueError("At least 2 elements are required to estimate Fisher F-distribution parameters.")
        if np.any(sample <= 0):
            raise ValueError("All sample elements for Fisher distribution must be strictly positive.")

        mean = float(np.mean(sample))
        var = float(np.var(sample, ddof=0))

        if mean <= 1.0:
            raise ValueError(
                "Sample mean must be strictly greater than 1.0 "
                "to estimate valid denominator degrees of freedom (df2 > 2)."
            )

        # Estimate denominator degrees of freedom (df2)
        df2 = float((2.0 * mean) / (mean - 1.0))
        if df2 <= 4.0:
            raise ValueError(
                "Estimated denominator degrees of freedom (df2) must be greater than 4.0 "
                "for variance to be finite."
            )

        # Intermediate parameter A = S^2 * (df2 - 4) / (2 * mean^2)
        a_param = (var * (df2 - 4.0)) / (2.0 * (mean ** 2))
        if a_param <= 1.0:
            raise ValueError("Calculated variance is too small to yield a positive numerator degrees of freedom (df1).")

        # Estimate numerator degrees of freedom (df1)
        df1 = float((df2 - 2.0) / (a_param - 1.0))

        return {"df1": df1, "df2": df2}

class RayleighMmEstimator(AbstractParameterEstimator):
    @staticmethod
    def distribution_type() -> DistributionType:
        return DistributionType.RAYLEIGH 

    @staticmethod
    def method() -> EstimationMethod:
        return EstimationMethod.MM

    def estimate(self, data: list[float] | tuple[float, ...] | np.ndarray) -> dict[str, float]:
        sample = np.asarray(data, dtype=float)
        if sample.size < 1:
            raise ValueError("Sample cannot be empty.")
        if np.any(sample < 0):
            raise ValueError("All sample elements for Rayleigh distribution must be non-negative.")

        mean = float(np.mean(sample))
        if mean <= 0:
            raise ValueError("Sample mean must be strictly positive to estimate Rayleigh scale parameter.")

        scale = float(mean * np.sqrt(2.0 / np.pi))
        return {"scale": scale}

class WignerMmEstimator(AbstractParameterEstimator):
    @staticmethod
    def distribution_type() -> DistributionType:
        return DistributionType.WIGNER

    @staticmethod
    def method() -> EstimationMethod:
        return EstimationMethod.MM

    def estimate(self, data: list[float] | tuple[float, ...] | np.ndarray) -> dict[str, float]:
        sample = np.asarray(data, dtype=float)
        if sample.size < 2:
            raise ValueError("At least 2 elements are required to estimate Wigner distribution parameters.")

        var = float(np.var(sample, ddof=0))
        if var <= 0:
            raise ValueError("Sample variance must be strictly positive to estimate Wigner radius parameter.")

        radius = float(2.0 * np.sqrt(var))
        return {"radius": radius}

class ParetoMmEstimator(AbstractParameterEstimator):
    @staticmethod
    def distribution_type() -> DistributionType:
        return DistributionType.PARETO

    @staticmethod
    def method() -> EstimationMethod:
        return EstimationMethod.MM

    def estimate(self, data: list[float] | tuple[float, ...] | np.ndarray) -> dict[str, float]:
        sample = np.asarray(data, dtype=float)
        if sample.size < 2:
            raise ValueError("At least 2 elements are required to estimate Pareto distribution parameters.")
        if np.any(sample <= 0):
            raise ValueError("All sample elements for Pareto distribution must be strictly positive.")

        mean = float(np.mean(sample))
        var = float(np.var(sample, ddof=0))

        if var <= 0:
            raise ValueError("Sample variance must be strictly positive to estimate Pareto parameters.")

        # Estimate shape parameter alpha
        alpha = float(1.0 + np.sqrt(1.0 + (mean ** 2) / var))
        
        # Estimate scale/location parameter x_m
        scale = float(mean * (alpha - 1.0) / alpha)

        return {"shape": alpha, "scale": scale}

class LaplaceMmEstimator(AbstractParameterEstimator):
    @staticmethod
    def distribution_type() -> DistributionType:
        return DistributionType.LAPLACE

    @staticmethod
    def method() -> EstimationMethod:
        return EstimationMethod.MM

    def estimate(self, data: list[float] | tuple[float, ...] | np.ndarray) -> dict[str, float]:
        sample = np.asarray(data, dtype=float)
        if sample.size < 2:
            raise ValueError("At least 2 elements are required to estimate Laplace distribution parameters.")

        mean = float(np.mean(sample))
        var = float(np.var(sample, ddof=0))

        if var <= 0:
            raise ValueError("Sample variance must be strictly positive to estimate Laplace scale parameter.")

        loc = mean
        scale = float(np.sqrt(var / 2.0))

        return {"t": loc, "s": scale}
