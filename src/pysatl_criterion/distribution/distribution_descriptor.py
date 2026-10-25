"""Shared interface for probability distribution descriptors."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar

from numpy.typing import ArrayLike

from pysatl_criterion.distribution.distribution_kind import DistributionKind
from pysatl_criterion.distribution.distribution_parameter_descriptor import (
    DistributionParameterDescriptor,
)
from pysatl_criterion.distribution.distribution_type import DistributionType
from pysatl_criterion.distribution.evaluation import BoundDistribution, ProbabilityResult
from pysatl_criterion.distribution.parameters import (
    ParameterizationDescriptor,
    ParameterSpec,
    ParameterValues,
)


class DistributionDescriptor(ABC):
    """
    Base class for descriptors that expose distribution metadata.

    Concrete descriptors identify a supported distribution and list the parameters
    needed to configure it.
    """

    DEFAULT: ClassVar[ParameterizationDescriptor]

    @classmethod
    def bind(cls, parameters: ParameterValues) -> BoundDistribution:
        """Bind a complete parameter set to a backend-independent evaluator.

        Supported coordinates are converted internally to DEFAULT. No missing
        parameters are estimated or filled. Backend-specific objects stay private.
        """
        from pysatl_criterion.distribution._scipy_backend import bind_distribution

        if not isinstance(parameters, ParameterValues):
            raise TypeError("parameters must be ParameterValues")
        if parameters.parameterization not in cls.parameterizations():
            raise ValueError("Unsupported distribution parameterization")
        missing = set(parameters.parameterization.parameters) - set(parameters)
        if missing:
            names = ", ".join(sorted(parameter.name for parameter in missing))
            raise ValueError(f"Complete parameters required; missing: {names}")
        canonical = cls.convert_parameters(parameters, cls.DEFAULT)
        # Revalidate against the descriptor's own schema and validators.
        canonical = cls.DEFAULT.parse(dict(canonical))
        return bind_distribution(cls.type(), canonical)

    @classmethod
    def cdf(cls, x: ArrayLike, parameters: ParameterValues) -> ProbabilityResult:
        """Evaluate P(X <= x) elementwise, preserving input order and shape.

        Scalars return float; array inputs return float64 arrays. NaNs propagate.
        For repeated evaluations, use bind(parameters).cdf(x).
        """
        return cls.bind(parameters).cdf(x)

    @classmethod
    def default_parameterization(cls) -> ParameterizationDescriptor:
        """Return the preferred coordinates, without filling parameter values."""
        return cls.DEFAULT

    @classmethod
    def convert_parameters(
        cls, values: ParameterValues, target: ParameterizationDescriptor
    ) -> ParameterValues:
        """Accept identity conversion; other conversions require an implementation."""
        if not isinstance(values, ParameterValues):
            raise TypeError("values must be ParameterValues")
        if (
            values.parameterization not in cls.parameterizations()
            or target not in cls.parameterizations()
        ):
            raise ValueError("Unsupported distribution parameterization")
        if values.parameterization != target:
            raise ValueError("Conversion between these parameterizations is not implemented")
        return values

    @classmethod
    def parameterizations(cls) -> tuple[ParameterizationDescriptor, ...]:
        """Return the explicit parameterizations available for the distribution."""
        return ()

    @staticmethod
    @abstractmethod
    def kind() -> DistributionKind:
        """Return the kind of probability distribution represented by the descriptor."""

    @staticmethod
    @abstractmethod
    def type() -> DistributionType:
        """
        Return the distribution type represented by the descriptor.

        :return: distribution enum member.
        """

    @staticmethod
    @abstractmethod
    def parameters() -> Sequence[DistributionParameterDescriptor | ParameterSpec]:
        """
        Return metadata for parameters accepted by the distribution.

        :return: list of distribution parameter descriptors.
        """
