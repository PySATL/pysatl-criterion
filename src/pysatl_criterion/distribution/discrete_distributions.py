"""Descriptors and parameterizations for discrete distributions."""

import math
from abc import ABC

from pysatl_criterion.distribution.distribution_descriptor import DistributionDescriptor
from pysatl_criterion.distribution.distribution_kind import DistributionKind
from pysatl_criterion.distribution.distribution_type import DistributionType
from pysatl_criterion.distribution.parameters import ParameterizationDescriptor, ParameterSpec
from pysatl_criterion.distribution.validator import NonNegativeNumberValidator, ProbabilityValidator


class DiscreteDistributionDescriptor(DistributionDescriptor, ABC):
    """Base class for discrete distribution descriptors."""

    @staticmethod
    def kind() -> DistributionKind:
        """Return the discrete distribution kind."""
        return DistributionKind.DISCRETE


class DiscreteUniformDistributionDescriptor(DiscreteDistributionDescriptor):
    """Uniform distribution on integers from lower to upper, both inclusive.

    The joint constraint lower <= upper is checked when binding.
    """

    LOWER = ParameterSpec(
        "discrete_uniform.lower",
        "lower",
        "a",
        "Inclusive integer lower endpoint",
        0,
        lambda value: value == math.floor(value),
    )
    UPPER = ParameterSpec(
        "discrete_uniform.upper",
        "upper",
        "b",
        "Inclusive integer upper endpoint",
        1,
        lambda value: value == math.floor(value),
    )
    DEFAULT = ParameterizationDescriptor(
        "discrete_uniform.lower_upper",
        DistributionType.DISCRETE_UNIFORM,
        (
            LOWER,
            UPPER,
        ),
    )

    @staticmethod
    def type() -> DistributionType:
        return DistributionType.DISCRETE_UNIFORM

    @staticmethod
    def parameters() -> list[ParameterSpec]:
        return list(DiscreteUniformDistributionDescriptor.DEFAULT.parameters)

    @classmethod
    def parameterizations(cls) -> tuple[ParameterizationDescriptor, ...]:
        return (cls.DEFAULT,)


class BernoulliDistributionDescriptor(DiscreteDistributionDescriptor):
    """Bernoulli distribution on {0, 1} with success probability p."""

    PROBABILITY = ParameterSpec(
        "bernoulli.probability",
        "p",
        "p",
        "Success probability. 0 <= p <= 1",
        0.5,
        ProbabilityValidator(),
    )
    DEFAULT = ParameterizationDescriptor(
        "bernoulli.probability", DistributionType.BERNOULLI, (PROBABILITY,)
    )

    @staticmethod
    def type() -> DistributionType:
        return DistributionType.BERNOULLI

    @staticmethod
    def parameters() -> list[ParameterSpec]:
        return list(BernoulliDistributionDescriptor.DEFAULT.parameters)

    @classmethod
    def parameterizations(cls) -> tuple[ParameterizationDescriptor, ...]:
        return (cls.DEFAULT,)


class BinomialDistributionDescriptor(DiscreteDistributionDescriptor):
    """Number of successes in n independent Bernoulli trials."""

    TRIALS = ParameterSpec(
        "binomial.trials",
        "n",
        "n",
        "Nonnegative integer number of trials",
        1,
        lambda value: value >= 0 and value == math.floor(value),
    )
    PROBABILITY = ParameterSpec(
        "binomial.probability",
        "p",
        "p",
        "Success probability. 0 <= p <= 1",
        0.5,
        ProbabilityValidator(),
    )
    DEFAULT = ParameterizationDescriptor(
        "binomial.trials_probability",
        DistributionType.BINOMIAL,
        (
            TRIALS,
            PROBABILITY,
        ),
    )

    @staticmethod
    def type() -> DistributionType:
        return DistributionType.BINOMIAL

    @staticmethod
    def parameters() -> list[ParameterSpec]:
        return list(BinomialDistributionDescriptor.DEFAULT.parameters)

    @classmethod
    def parameterizations(cls) -> tuple[ParameterizationDescriptor, ...]:
        return (cls.DEFAULT,)


class PoissonDistributionDescriptor(DiscreteDistributionDescriptor):
    """Poisson count distribution with nonnegative mean."""

    RATE = ParameterSpec(
        "poisson.rate", "lam", "λ", "Mean count. λ >= 0", 1, NonNegativeNumberValidator()
    )
    DEFAULT = ParameterizationDescriptor("poisson.rate", DistributionType.POISSON, (RATE,))

    @staticmethod
    def type() -> DistributionType:
        return DistributionType.POISSON

    @staticmethod
    def parameters() -> list[ParameterSpec]:
        return list(PoissonDistributionDescriptor.DEFAULT.parameters)

    @classmethod
    def parameterizations(cls) -> tuple[ParameterizationDescriptor, ...]:
        return (cls.DEFAULT,)


class GeometricDistributionDescriptor(DiscreteDistributionDescriptor):
    """Number of trials up to and including the first success, on {1, 2, ...}."""

    PROBABILITY = ParameterSpec(
        "geometric.probability",
        "p",
        "p",
        "Success probability. 0 < p <= 1",
        0.5,
        lambda value: 0 < value <= 1,
    )
    DEFAULT = ParameterizationDescriptor(
        "geometric.probability", DistributionType.GEOMETRIC, (PROBABILITY,)
    )

    @staticmethod
    def type() -> DistributionType:
        return DistributionType.GEOMETRIC

    @staticmethod
    def parameters() -> list[ParameterSpec]:
        return list(GeometricDistributionDescriptor.DEFAULT.parameters)

    @classmethod
    def parameterizations(cls) -> tuple[ParameterizationDescriptor, ...]:
        return (cls.DEFAULT,)
