import inspect

from pysatl_criterion import AbstractExponentialityGofStatistic, AbstractWeibullGofStatistic
from pysatl_criterion.core import DistributionType
from pysatl_criterion.gof.normal import AbstractNormalityGofStatistic


def __all_subclasses(cls):
    subclasses = set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in __all_subclasses(c)]
    )
    return set(filter(lambda x: not inspect.isabstract(x), subclasses))


def get_weibull_statistics():
    return __all_subclasses(AbstractWeibullGofStatistic)


def get_normality_statistics():
    return __all_subclasses(AbstractNormalityGofStatistic)


def get_exponent_statistics():
    return __all_subclasses(AbstractExponentialityGofStatistic)


def get_statistics(distribution_type: DistributionType):
    if distribution_type == DistributionType.NORM:
        return get_normality_statistics()
    elif distribution_type == DistributionType.EXP:
        return get_exponent_statistics()
    elif distribution_type == DistributionType.WEIBULL:
        return get_weibull_statistics()
    else:
        raise ValueError(f"Distribution type {distribution_type} is unknown")
