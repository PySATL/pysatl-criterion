import inspect

from pysatl_criterion.util.distribution import DistributionType


def get_available_criteria(distribution: DistributionType):
    """
    Return a list of short codes for all non-abstract statistical criteria available for the given distribution.

    This function inspects all direct subclasses of the distribution's base_class and filters out abstract classes.
    For each concrete subclass, its short_code() method is invoked to obtain a unique short identifier.

    :param distribution: distribution descriptor whose base_class defines the root class for available statistical criteria.
    :return: list of short codes corresponding to all non-abstract subclasses of distribution.base_class.
    """
    return [
        cls.short_code()
        for cls in distribution.base_class.__subclasses__()
        if not inspect.isabstract(cls)
    ]
