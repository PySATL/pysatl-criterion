import pytest

from pysatl_criterion import DistributionKind, DistributionType
from pysatl_criterion.distribution.distributions import (
    InverseGammaDistributionDescriptor,
    LogLogisticDistributionDescriptor,
    ParetoDistributionDescriptor,
)
from pysatl_criterion.statistics.goodness_of_fit.inverse_gamma import (
    AbstractInverseGammaGofStatistic,
)
from pysatl_criterion.statistics.goodness_of_fit.log_logistic import AbstractLogLogisticGofStatistic
from pysatl_criterion.statistics.goodness_of_fit.pareto import AbstractParetoGofStatistic
from pysatl_criterion.utils.distribution import get_available_distribution_descriptor
from pysatl_criterion.utils.statistic import get_available_criteria


@pytest.mark.parametrize(
    "statistic, descriptor, distribution_type, names",
    [
        (
            AbstractParetoGofStatistic,
            ParetoDistributionDescriptor,
            DistributionType.PARETO,
            ["shape", "scale"],
        ),
        (
            AbstractInverseGammaGofStatistic,
            InverseGammaDistributionDescriptor,
            DistributionType.INVERSE_GAMMA,
            ["alpha", "beta"],
        ),
        (
            AbstractLogLogisticGofStatistic,
            LogLogisticDistributionDescriptor,
            DistributionType.LOG_LOGISTIC,
            ["alpha", "beta"],
        ),
    ],
)
def test_distribution_descriptor_contract(statistic, descriptor, distribution_type, names):
    assert statistic.distribution() is descriptor
    assert statistic.distribution().type() is distribution_type
    assert descriptor.kind() is DistributionKind.CONTINUOUS
    assert isinstance(get_available_distribution_descriptor(distribution_type), descriptor)
    assert [parameter.name for parameter in descriptor.parameters()] == names
    (schema,) = descriptor.parameterizations()
    assert schema.distribution is distribution_type
    assert schema.parse({}, fill_defaults=True).as_dict() == dict.fromkeys(names, 1)
    for name in names:
        with pytest.raises(ValueError):
            schema.parse({name: 0}, fill_defaults=True)
    criteria = get_available_criteria(distribution_type)
    assert criteria
    assert all(criterion.distribution() is descriptor for criterion in criteria)


@pytest.mark.parametrize("distribution_type", list(DistributionType))
def test_all_criteria_return_the_registered_descriptor(distribution_type):
    for criterion in get_available_criteria(distribution_type):
        descriptor = criterion.distribution()
        assert isinstance(get_available_distribution_descriptor(distribution_type), descriptor)
        assert criterion.distribution().type() is distribution_type
        assert descriptor.kind() is DistributionKind.CONTINUOUS
