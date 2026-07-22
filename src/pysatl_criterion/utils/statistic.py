import inspect
from importlib import import_module
from typing import Any

from pysatl_criterion import DistributionType
from pysatl_criterion.statistics import AbstractGoodnessOfFitStatistic


def get_available_criteria(
    distribution: DistributionType,
) -> list[type[AbstractGoodnessOfFitStatistic]]:
    """
    Return a list of all non-abstract statistical criteria available for
    the given distribution.

    This function inspects all subclasses of AbstractGoodnessOfFitStatistic and filters out
    abstract classes.

    :param distribution: distribution descriptor to filter available statistical criteria.
    :return: list of non-abstract subclasses of AbstractGoodnessOfFitStatistic for the
    requested distribution.
    """
    __load_goodness_of_fit_statistics()

    return [
        cls
        for cls in __get_all_subclasses(AbstractGoodnessOfFitStatistic)
        if not inspect.isabstract(cls) and cls.distribution() == distribution
    ]


def get_available_criteria_codes(distribution: DistributionType) -> list[str]:
    """
    Return a list of short codes for all non-abstract statistical criteria available for
    the given distribution.

    For each concrete criterion class, its short_code() method is invoked to obtain
    a short identifier.

    :param distribution: distribution descriptor to filter available statistical criteria.
    :return: list of short codes corresponding to all non-abstract subclasses of
    AbstractGoodnessOfFitStatistic for the requested distribution.
    """
    return [criterion.short_code() for criterion in get_available_criteria(distribution)]


def __load_goodness_of_fit_statistics() -> None:
    """
    Load goodness-of-fit statistic modules so subclasses are registered in Python runtime.
    """
    import_module("pysatl_criterion.statistics.goodness_of_fit")


def __get_all_subclasses(cls: type[Any]) -> set[type[AbstractGoodnessOfFitStatistic]]:
    """
    Return all direct and indirect subclasses of a class.

    The search walks the full subclass tree recursively and returns each discovered
    subclass once.

    :param cls: root class whose subclass hierarchy should be inspected.
    :return: set containing all subclasses below the root class.
    """
    subclasses: set[type[AbstractGoodnessOfFitStatistic]] = set()

    for subclass in cls.__subclasses__():
        subclasses.update(__get_all_subclasses(subclass))

        if issubclass(subclass, AbstractGoodnessOfFitStatistic):
            subclasses.add(subclass)

    return subclasses
