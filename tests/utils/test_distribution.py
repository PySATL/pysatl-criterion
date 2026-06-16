import pytest

from pysatl_criterion import DistributionType
from pysatl_criterion.distribution.distributions import (
    BetaDistributionDescriptor,
    ExponentialDistributionDescriptor,
    GammaDistributionDescriptor,
    NormalDistributionDescriptor,
    StudentDistributionDescriptor,
    UniformDistributionDescriptor,
    WeibullDistributionDescriptor,
)
from pysatl_criterion.utils.distribution import get_available_distribution_descriptor


@pytest.mark.parametrize(
    ("distribution", "descriptor_type", "parameters"),
    [
        (
            DistributionType.NORMAL,
            NormalDistributionDescriptor,
            [
                ("μ", "mean", "Mean", 0),
                ("σ²", "var", "Variance. σ² > 0", 1),
            ],
        ),
        (
            DistributionType.EXPONENTIAL,
            ExponentialDistributionDescriptor,
            [("λ", "lam", "Rate, or inverse scale. λ > 0", 1)],
        ),
        (
            DistributionType.WEIBULL,
            WeibullDistributionDescriptor,
            [
                ("λ", "a", "Scale. λ > 0", 1),
                ("k", "k", "Shape. k > 0", 5),
            ],
        ),
        (
            DistributionType.UNIFORM,
            UniformDistributionDescriptor,
            [
                ("a", "a", "Interval start", 0),
                ("b", "b", "Interval end", 1),
            ],
        ),
        (
            DistributionType.STUDENT,
            StudentDistributionDescriptor,
            [("ν", "df", "Degrees of freedom. ν > 0", 2)],
        ),
        (
            DistributionType.GAMMA,
            GammaDistributionDescriptor,
            [
                ("α", "alfa", "Shape. α > 0", 1),
                ("β", "beta", "Rate. β > 0", 1),
            ],
        ),
        (
            DistributionType.BETA,
            BetaDistributionDescriptor,
            [
                ("α", "a", "First shape parameter. α > 0", 1),
                ("β", "b", "Second shape parameter. β > 0", 1),
            ],
        ),
    ],
)
def test_get_available_distribution_descriptor_returns_descriptor(
    distribution,
    descriptor_type,
    parameters,
):
    descriptor = get_available_distribution_descriptor(distribution)

    assert isinstance(descriptor, descriptor_type)
    assert descriptor.type() == distribution
    assert [
        (
            parameter.display_name,
            parameter.name,
            parameter.description,
            parameter.default,
        )
        for parameter in descriptor.parameters()
    ] == parameters


def test_get_available_distribution_descriptor_raises_for_missing_descriptor():
    with pytest.raises(StopIteration):
        get_available_distribution_descriptor(object())
