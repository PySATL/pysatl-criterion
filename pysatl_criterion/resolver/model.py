from abc import ABC, abstractmethod


CriticalValueType = float | tuple[float, float]


class IResolver(ABC):
    @abstractmethod
    def resolve(self, *params) -> any:
        """
         Resolve some value.

        :param params: parameters used to resolve value
        :return: value or None
        """
        raise NotImplementedError("Method is not implemented")


class ICriticalValueResolver(IResolver):
    @abstractmethod
    def resolve(self, size: int | list[int], significance_level: float) -> CriticalValueType | None:
        """
         Resolve critical value.

        :param size: sample size
        :param significance_level: significance level
        :return: critical value or None
        """
        raise NotImplementedError("Method is not implemented")


class IPValueResolver(IResolver):
    @abstractmethod
    def resolve(self, x: list[float]) -> CriticalValueType | None:
        """
         Resolve p-value.

        :param x: sample
        :return: p-value or None
        """
        raise NotImplementedError("Method is not implemented")
