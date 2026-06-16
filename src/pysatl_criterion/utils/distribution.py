import inspect
from collections.abc import Iterator
from typing import Any, cast

from pysatl_criterion import DistributionType
from pysatl_criterion.distribution.distributions import DistributionDescriptor


def get_available_distribution_descriptor(distribution: DistributionType) -> DistributionDescriptor:
    return next(
        _instantiate_descriptor(cls)
        for cls in _iter_distribution_descriptor_classes(DistributionDescriptor)
        if cls.type() == distribution
    )


def _iter_distribution_descriptor_classes(
    cls: type[Any],
) -> Iterator[type[DistributionDescriptor]]:
    """
    Yield concrete descriptor subclasses from the descriptor hierarchy.

    Abstract classes are skipped because they cannot be instantiated.
    """
    for subclass in cls.__subclasses__():
        yield from _iter_distribution_descriptor_classes(subclass)

        if not inspect.isabstract(subclass):
            yield cast(type[DistributionDescriptor], subclass)


def _instantiate_descriptor(cls: type[DistributionDescriptor]) -> DistributionDescriptor:
    """
    Instantiate a descriptor class known to be concrete.

    Mypy cannot infer that ``inspect.isabstract`` filters out abstract classes,
    so the instantiation is cast in this small helper.
    """
    return cast(DistributionDescriptor, cast(Any, cls)())
