from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from pysatl_criterion.distribution.distribution_type import DistributionType


class EstimationMethod(Enum):
    """
    Enumerate supported parameter estimation methods.
    """

    MLE = "mle"
    MM = "mm"


class AbstractParameterEstimator(ABC):
    """
    Base interface for distribution parameter estimators.

    Each concrete estimator is responsible for a single distribution type and a
    single estimation method.
    """

    @staticmethod
    @abstractmethod
    def distribution_type() -> DistributionType:
        """
        Return the distribution type handled by the estimator.
        """

    @staticmethod
    @abstractmethod
    def method() -> EstimationMethod:
        """
        Return the estimation method implemented by the estimator.
        """

    @abstractmethod
    def estimate(self, data: list[float] | tuple[float, ...] | object) -> dict[str, float]:
        """
        Estimate distribution parameters from sample data.

        The returned dictionary keys must match the ``name`` field of the
        corresponding ``DistributionParameterDescriptor`` entries.
        """
