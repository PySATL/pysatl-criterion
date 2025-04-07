from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
from scipy.stats import expon, exponweib, norm, uniform
from typing_extensions import override


HypothesisGenerationType = list[float] | tuple[list[float], list[float]]
HypothesisType = Enum("HypothesisType", [("LEFT", 1), ("RIGHT", 2), ("TWO_SIDED", 3)])


class IHypothesis(ABC):
    hypothesis_type: HypothesisType
    parameters: dict[str, any] | None

    @staticmethod
    def _validate_params(parameters: dict[str, any], keys: list[str]):
        """
        Validate params. Check that parameters contains all keys.

        :param parameters: parameters dictionary
        :param keys: keys that must be present in parameters
        :return: None
        """
        if parameters is None:
            return

        diff = set(keys).difference(set(parameters.keys()))
        if len(diff) > 0:
            raise ValueError(f"Parameters {diff} must be present")

    def __init__(self, hypothesis_type: HypothesisType, parameters: dict[str, any] | None = None):
        self.hypothesis_type = hypothesis_type
        self.parameters = parameters

    @abstractmethod
    def generate(self, size) -> HypothesisGenerationType:
        """
        Generate one or several samples as H0.

        :param size: size or list of sizes
        """
        raise NotImplementedError("Method is not implemented")


class AbstractGofHypothesis(IHypothesis, ABC):
    hypothesis_type: HypothesisType
    parameters: dict[str, any] | None

    def __init__(self, hypothesis_type: HypothesisType, parameters: dict[str, any] | None = None):
        super().__init__(hypothesis_type, parameters)

    @abstractmethod
    def generate(self, size):
        raise NotImplementedError("Method is not implemented")


class NormalityGofHypothesis(AbstractGofHypothesis):
    hypothesis_type: HypothesisType
    parameters: dict[str, any] | None

    @staticmethod
    @override
    def _validate_params(parameters: dict[str, any], keys: list[str]):
        if parameters is None:
            return

        IHypothesis._validate_params(parameters, keys)

        if parameters["var"] <= 0:
            raise ValueError("Parameter var must be positive")

    def __init__(self, hypothesis_type: HypothesisType, parameters: dict[str, any] | None = None):
        super().__init__(hypothesis_type, parameters)
        self._validate_params(parameters, ["mean", "var"])

    @override
    def generate(self, size: int):
        if self.parameters is not None:
            mean = self.parameters["mean"]
            std = np.sqrt(self.parameters["var"])
            return norm.rvs(size=size, loc=mean, scale=std)

        mean = np.random.uniform(-10000, 10000)
        std = np.random.uniform(0.001, 10000)
        return norm.rvs(size=size, loc=mean, scale=std)


class ExponentialityGofHypothesis(AbstractGofHypothesis):
    hypothesis_type: HypothesisType
    parameters: dict[str, any] | None

    @staticmethod
    @override
    def _validate_params(parameters: dict[str, any], keys: list[str]):
        if parameters is None:
            return

        IHypothesis._validate_params(parameters, keys)

        if parameters["lam"] <= 0:
            raise ValueError("Parameter lam must be positive")

    def __init__(self, hypothesis_type: HypothesisType, parameters: dict[str, any] | None = None):
        super().__init__(hypothesis_type, parameters)
        self._validate_params(parameters, ["lam"])

    @override
    def generate(self, size):
        if self.parameters is not None:
            scale = 1 / self.parameters["lam"]
            return expon.rvs(size=size, scale=scale)

        scale = 1 / np.random.uniform(0.001, 10000)
        return expon.rvs(size=size, scale=scale)


class WeibullGofHypothesis(AbstractGofHypothesis):
    hypothesis_type: HypothesisType
    parameters: dict[str, any] | None

    @staticmethod
    @override
    def _validate_params(parameters: dict[str, any], keys: list[str]):
        if parameters is None:
            return

        IHypothesis._validate_params(parameters, keys)

        if parameters["a"] <= 0:
            raise ValueError("Parameter a must be positive")
        if parameters["k"] <= 0:
            raise ValueError("Parameter lam must be positive")

    def __init__(self, hypothesis_type: HypothesisType, parameters: dict[str, any] | None = None):
        super().__init__(hypothesis_type, parameters)
        self._validate_params(parameters, ["a", "k"])

    @override
    def generate(self, size):
        if self.parameters is not None:
            return exponweib.rvs(a=self.parameters["a"], c=self.parameters["k"], size=size)

        a = np.random.uniform(0.001, 10000)
        k = np.random.uniform(0.001, 10000)
        return exponweib.rvs(a=a, c=k, size=size)


class AbstractUniformHypothesis(IHypothesis, ABC):
    hypothesis_type: HypothesisType
    parameters: dict[str, any] | None

    def __init__(self, hypothesis_type: HypothesisType, parameters: dict[str, any] | None = None):
        super().__init__(hypothesis_type, parameters)

    @abstractmethod
    def generate(self, size: list[int]):
        raise NotImplementedError("Method is not implemented")


class DistributionUniformHypothesis(AbstractUniformHypothesis):
    hypothesis_type: HypothesisType
    parameters: dict[str, any] | None

    def __init__(self, hypothesis_type: HypothesisType, parameters: dict[str, any] | None = None):
        super().__init__(hypothesis_type, parameters)

    @override
    def generate(self, size: list[int]):
        return uniform.rvs(size=size[0]), uniform.rvs(size=size[1])
