"""Shared base class for goodness-of-fit statistics."""

from abc import ABC, abstractmethod

from numpy import float64
from typing_extensions import override

from pysatl_criterion.distribution import DistributionDescriptor
from pysatl_criterion.distribution.parameters import (
    HypothesisSupport,
    ParameterizationDescriptor,
    ParameterValues,
)
from pysatl_criterion.statistics.hypothesis import GoodnessOfFitHypothesis
from pysatl_criterion.statistics.statistic import AbstractStatistic


class AbstractGoodnessOfFitStatistic(AbstractStatistic, ABC):
    """
    Shared constructor and hypothesis storage for goodness-of-fit statistics.

    Subclasses declare exact fixed-parameter sets in supported_hypotheses().
    The constructor requires ParameterValues; omitted coordinates stay unknown.
    Calculations read _parameters in calculation_parameterization(), while
    hypothesis() retains the caller's original schema and fixed values.
    """

    def configuration(self) -> dict:
        """Versioned settings affecting the statistic, excluding the null parameters.

        Subclasses with algorithm options must override this method before their
        storage validation guard can be removed. Legacy rows have no configuration.
        """
        return {"schema_version": 1, "statistic_version": 1}

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        """Explicit capabilities; an empty tuple means no supported hypothesis."""
        return ()

    @classmethod
    def calculation_parameterization(cls) -> ParameterizationDescriptor:
        """Return the coordinates used by this criterion's implementation."""
        return cls.distribution().default_parameterization()

    @classmethod
    def supports_hypothesis(cls, hypothesis: GoodnessOfFitHypothesis) -> bool:
        values = hypothesis.parameter_values
        return values is not None and any(
            support.supports(values) for support in cls.supported_hypotheses()
        )

    def __init__(self, parameters: ParameterValues):
        """Store an explicitly supported schema and exact set of fixed parameters.

        Distribution parameters are never filled implicitly. Subclasses accept
        algorithm settings separately as keyword-only arguments and call this
        constructor before using parameter values.
        """
        if not isinstance(parameters, ParameterValues):
            raise TypeError("parameters must be ParameterValues")
        if not self.supports_hypothesis(GoodnessOfFitHypothesis(parameters)):
            raise ValueError(
                f"Unsupported hypothesis or parameterization for {type(self).__name__}"
            )
        target = self.calculation_parameterization()
        converted = self.distribution().convert_parameters(parameters, target)
        if converted.parameterization != target or not self.supports_hypothesis(
            GoodnessOfFitHypothesis(converted)
        ):
            raise ValueError("Conversion produced an unsupported calculation hypothesis")
        self._hypothesis_parameters = parameters
        self._parameters = converted

    @override
    def hypothesis(self) -> GoodnessOfFitHypothesis:
        """Return the original hypothesis in the user's parameterization."""
        return GoodnessOfFitHypothesis(self._hypothesis_parameters)

    @staticmethod
    @abstractmethod
    def distribution() -> type[DistributionDescriptor]:
        """
        Return the descriptor class for the distribution family.
        """

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for goodness-of-fit statistics.

        :return: string code "GOODNESS_OF_FIT".
        """
        return "GOODNESS_OF_FIT"

    @abstractmethod
    def execute_statistic(self, rvs) -> float | float64:
        """
        Execute test statistic and return calculated statistic value.

        :param rvs: rvs data to calculated statistic value
        """
        raise NotImplementedError("Method is not implemented")
