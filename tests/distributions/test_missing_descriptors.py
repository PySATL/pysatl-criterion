"""Coverage and probability contracts for newly supported distribution families."""

import inspect

import numpy as np
import pytest
from scipy import stats

import pysatl_criterion.distribution as d
from pysatl_criterion.distribution.distribution_kind import DistributionKind
from pysatl_criterion.distribution.distribution_type import DistributionType
from pysatl_criterion.utils.distribution import get_available_distribution_descriptor


CASES = [
    (d.FisherDistributionDescriptor, {"dfn": 3, "dfd": 7}, stats.f(3, 7)),
    (d.RayleighDistributionDescriptor, {"scale": 2}, stats.rayleigh(scale=2)),
    (d.WignerDistributionDescriptor, {"loc": 2, "radius": 3}, stats.semicircular(loc=2, scale=3)),
    (d.DiscreteUniformDistributionDescriptor, {"lower": -2, "upper": 3}, stats.randint(-2, 4)),
    (d.BernoulliDistributionDescriptor, {"p": 0.3}, stats.bernoulli(0.3)),
    (d.BinomialDistributionDescriptor, {"n": 5, "p": 0.3}, stats.binom(5, 0.3)),
    (d.PoissonDistributionDescriptor, {"lam": 2}, stats.poisson(2)),
    (d.GeometricDistributionDescriptor, {"p": 0.3}, stats.geom(0.3)),
]


def test_every_enum_has_exactly_one_descriptor():
    classes = [
        cls
        for _, cls in inspect.getmembers(d, inspect.isclass)
        if issubclass(cls, d.DistributionDescriptor) and not inspect.isabstract(cls)
    ]
    assert len(classes) == len(DistributionType)
    assert {cls.type() for cls in classes} == set(DistributionType)
    for cls in classes:
        assert isinstance(get_available_distribution_descriptor(cls.type()), cls)
        assert cls.DEFAULT.distribution is cls.type()
        assert cls.DEFAULT.parse({}, fill_defaults=True)


@pytest.mark.parametrize(("descriptor", "parameters", "reference"), CASES)
def test_probability_evaluation(descriptor, parameters, reference):
    values = descriptor.DEFAULT.parse(parameters)
    bound = descriptor.bind(values)
    x = np.array([-np.inf, -3, 0, 0.5, 1, 2, 3, 7, np.inf])
    for method in ("cdf", "sf", "logcdf", "logsf"):
        np.testing.assert_allclose(getattr(bound, method)(x), getattr(reference, method)(x))
    assert isinstance(descriptor.cdf(1, values), float)
    expected_kind = (
        DistributionKind.DISCRETE
        if reference.dist.name in ("randint", "bernoulli", "binom", "poisson", "geom")
        else DistributionKind.CONTINUOUS
    )
    assert descriptor.kind() is expected_kind


@pytest.mark.parametrize(
    ("descriptor", "parameters"),
    [
        (d.FisherDistributionDescriptor, {"dfn": 0}),
        (d.FisherDistributionDescriptor, {"dfd": -1}),
        (d.RayleighDistributionDescriptor, {"scale": 0}),
        (d.WignerDistributionDescriptor, {"radius": -1}),
        (d.DiscreteUniformDistributionDescriptor, {"lower": 0.5}),
        (d.DiscreteUniformDistributionDescriptor, {"upper": 1.5}),
        (d.BernoulliDistributionDescriptor, {"p": -0.1}),
        (d.BernoulliDistributionDescriptor, {"p": 1.1}),
        (d.BinomialDistributionDescriptor, {"n": 1.5}),
        (d.BinomialDistributionDescriptor, {"n": -1}),
        (d.PoissonDistributionDescriptor, {"lam": -1}),
        (d.GeometricDistributionDescriptor, {"p": 0}),
        (d.GeometricDistributionDescriptor, {"p": 1.1}),
    ],
)
def test_invalid_parameters(descriptor, parameters):
    with pytest.raises(ValueError):
        descriptor.DEFAULT.parse(parameters, fill_defaults=True)


@pytest.mark.parametrize(
    ("descriptor", "parameters", "point"),
    [
        (d.DiscreteUniformDistributionDescriptor, {"lower": 2, "upper": 2}, 2),
        (d.BernoulliDistributionDescriptor, {"p": 0}, 0),
        (d.BernoulliDistributionDescriptor, {"p": 1}, 1),
        (d.BinomialDistributionDescriptor, {"n": 0, "p": 0.5}, 0),
        (d.PoissonDistributionDescriptor, {"lam": 0}, 0),
        (d.GeometricDistributionDescriptor, {"p": 1}, 1),
    ],
)
def test_degenerate_distributions(descriptor, parameters, point):
    bound = descriptor.bind(descriptor.DEFAULT.parse(parameters))
    assert bound.cdf(point - 1) == 0
    assert bound.cdf(point) == 1


def test_discrete_uniform_reversed_bounds():
    descriptor = d.DiscreteUniformDistributionDescriptor
    with pytest.raises(ValueError, match="lower <= upper"):
        descriptor.bind(descriptor.DEFAULT.parse({"lower": 3, "upper": 2}))


@pytest.mark.parametrize(("descriptor", "parameters", "reference"), CASES)
def test_invalid_scalar_values_and_missing_parameters(descriptor, parameters, reference):
    for value in (True, np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError):
            descriptor.DEFAULT.parse({next(iter(parameters)): value})
    with pytest.raises(ValueError, match="Complete parameters required"):
        descriptor.bind(descriptor.DEFAULT.parse({}))
