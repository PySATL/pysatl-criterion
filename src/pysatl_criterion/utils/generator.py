"""Utilities for resolving random value sample generators."""

import importlib
import inspect
from typing import Any

from pysatl_criterion import DistributionType
from pysatl_criterion.generator.model import AbstractRVSGenerator


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
