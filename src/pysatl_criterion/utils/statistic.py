import inspect
from typing import Generic, TypeVar

from pysatl_criterion import DistributionType
from pysatl_criterion.statistics import AbstractGoodnessOfFitStatistic


C = TypeVar("C", bound=AbstractGoodnessOfFitStatistic)


def get_available_criteria(distribution: DistributionType) -> list[str]:
    """
    Return a list of short codes for all non-abstract statistical criteria available for
    the given distribution.

    This function inspects all direct subclasses of the distribution's base_class and filters out
    abstract classes.
    For each concrete subclass, its short_code() method is invoked to obtain
    a unique short identifier.

    :param distribution: distribution descriptor whose base_class defines the root class for
    available statistical criteria.
    :return: list of short codes corresponding to all non-abstract subclasses of
    distribution.base_class.
    """
    return [
        cls.short_code()
        for cls in get_all_subclasses(AbstractGoodnessOfFitStatistic)
        if not inspect.isabstract(cls) and cls.distribution() == distribution
    ]


def get_all_subclasses(cls: Generic[C]) -> set[C]:
    """
    Return all direct and indirect subclasses of a class.

    The search walks the full subclass tree recursively and returns each discovered
    subclass once.

    :param cls: root class whose subclass hierarchy should be inspected.
    :return: set containing all subclasses below the root class.
    """
    subclasses = set()

    for subclass in cls.__subclasses__():
        subclasses.add(subclass)
        subclasses.update(get_all_subclasses(subclass))

    return subclasses
