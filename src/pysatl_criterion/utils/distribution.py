import inspect

from pysatl_criterion import DistributionType
from pysatl_criterion.distribution.distributions import DistributionDescriptor


def get_available_distribution_descriptor(distribution: DistributionType) -> DistributionDescriptor:
    return next(
        cls()
        for cls in DistributionDescriptor.__subclasses__()
        if not inspect.isabstract(cls) and cls.type() == distribution
    )
