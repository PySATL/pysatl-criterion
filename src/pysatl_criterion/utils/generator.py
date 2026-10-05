"""Utilities for resolving random value sample generators."""

import importlib
import inspect
import math
from typing import TYPE_CHECKING, Any

from pysatl_criterion import DistributionType
from pysatl_criterion.generator.model import AbstractRVSGenerator


if TYPE_CHECKING:
    from pysatl_criterion.statistics import AbstractGoodnessOfFitStatistic


def get_hypothesis_generator(
    statistic: "AbstractGoodnessOfFitStatistic",
) -> AbstractRVSGenerator:
    """Build a sampler using the statistic's hypothesis parameterization.

    Hypothesis metadata retains its public names and units. Sampling uses
    Beta's a/b, Gamma's alfa/rate, and log-normal's logarithmic location mu.
    Direct calls to get_available_generator continue to use generator parameters.
    """
    hypothesis = statistic.hypothesis()
    if hypothesis.parameter_values is not None:
        from pysatl_criterion.generator.generators import TRVSGenerator

        if statistic.distribution() == DistributionType.STUDENT:
            return TRVSGenerator.from_parameters(hypothesis.parameter_values)
        raise ValueError("No schema-aware sampler for this distribution")
    distribution = statistic.distribution()
    params = statistic.hypothesis().parameters()
    if distribution == DistributionType.BETA:
        params = {"a": params.pop("alpha"), "b": params.pop("beta"), **params}
    elif distribution == DistributionType.GAMMA:
        params = {"alfa": params.pop("alpha"), **params}
    elif distribution == DistributionType.LOG_NORMAL:
        params = {"mu": math.log(params.pop("scale")), **params}
    return get_available_generator(distribution, params)


def get_available_generator(
    distribution: DistributionType, params: dict[str, float] | None
) -> AbstractRVSGenerator:
    """
    Return a concrete random value generator for a distribution.

    The function walks all ``AbstractRVSGenerator`` subclasses, finds the
    non-abstract generator whose ``distribution_type`` matches the requested
    distribution, and initializes it with the provided parameters.

    :param distribution: distribution type to generate random values from.
    :param params: keyword parameters passed to the matching generator constructor.
    :return: initialized random value generator for the requested distribution.
    """
    if distribution == DistributionType.STUDENT:
        from pysatl_criterion.distribution.distributions import StudentDistributionDescriptor
        from pysatl_criterion.generator.generators import TRVSGenerator

        values = StudentDistributionDescriptor.DEFAULT.parse(params or {}, fill_defaults=True)
        return TRVSGenerator.from_parameters(values)
    _load_generators()
    return next(
        cls(**(params or {}))
        for cls in __get_all_subclasses(AbstractRVSGenerator)
        if not inspect.isabstract(cls) and cls.distribution_type() == distribution
    )


def _load_generators() -> None:
    """Import concrete generator classes so subclass discovery can find them."""
    importlib.import_module("pysatl_criterion.generator.generators")


def __get_all_subclasses(cls: type[Any]) -> set[type[AbstractRVSGenerator]]:
    """
    Return all direct and indirect random value generator subclasses.

    The search walks the full subclass tree recursively and returns each
    discovered ``AbstractRVSGenerator`` subclass once.

    :param cls: root class whose subclass hierarchy should be inspected.
    :return: set containing all generator subclasses below the root class.
    """
    subclasses: set[type[AbstractRVSGenerator]] = set()

    for subclass in cls.__subclasses__():
        subclasses.update(__get_all_subclasses(subclass))

        if issubclass(subclass, AbstractRVSGenerator):
            subclasses.add(subclass)

    return subclasses
