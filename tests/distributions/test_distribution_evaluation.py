"""Contract and parameter-coordinate regressions for bound distributions."""

import numpy as np
import pytest
from scipy import stats

import pysatl_criterion.distribution as d


CASES = [
    (d.FisherDistributionDescriptor, {"dfn": 3, "dfd": 7}, stats.f(3, 7)),
    (d.RayleighDistributionDescriptor, {"scale": 2}, stats.rayleigh(scale=2)),
    (d.WignerDistributionDescriptor, {"loc": 2, "radius": 3}, stats.semicircular(loc=2, scale=3)),
    (d.NormalDistributionDescriptor, {"mean": 3, "var": 4}, stats.norm(3, 2)),
    (d.ExponentialDistributionDescriptor, {"lam": 2}, stats.expon(scale=0.5)),
    (d.WeibullDistributionDescriptor, {"scale": 3, "shape": 2}, stats.weibull_min(2, scale=3)),
    (
        d.ExponentiatedWeibullDistributionDescriptor,
        {"exponent": 3, "shape": 2, "scale": 4},
        stats.exponweib(3, 2, scale=4),
    ),
    (d.UniformDistributionDescriptor, {"a": -2, "b": 5}, stats.uniform(-2, 7)),
    (d.StudentDistributionDescriptor, {"df": 5, "loc": 2, "scale": 3}, stats.t(5, 2, 3)),
    (d.GammaDistributionDescriptor, {"alfa": 3, "beta": 2}, stats.gamma(3, scale=0.5)),
    (d.BetaDistributionDescriptor, {"a": 2, "b": 3}, stats.beta(2, 3)),
    (d.LogNormalDistributionDescriptor, {"mu": np.log(3), "s": 2}, stats.lognorm(2, scale=3)),
    (d.CauchyDistributionDescriptor, {"t": 2, "s": 3}, stats.cauchy(2, 3)),
    (d.Chi2DistributionDescriptor, {"df": 3}, stats.chi2(3)),
    (d.GompertzDistributionDescriptor, {"eta": 2, "b": 3}, stats.gompertz(2, scale=3)),
    (d.GumbelDistributionDescriptor, {"mu": 2, "beta": 3}, stats.gumbel_r(2, 3)),
    (d.InvGaussDistributionDescriptor, {"mu": 6, "lam": 2}, stats.invgauss(3, scale=2)),
    (d.LaplaceDistributionDescriptor, {"t": 2, "s": 3}, stats.laplace(2, 3)),
    (
        d.HyperbolicDistributionDescriptor,
        {"alpha": 2, "beta": 0.5, "delta": 3, "mu": 4},
        stats.genhyperbolic(1, 6, 1.5, loc=4, scale=3),
    ),
    (
        d.TruncNormDistributionDescriptor,
        {"mean": 3, "var": 4, "a": 1, "b": 7},
        stats.truncnorm(-1, 2, loc=3, scale=2),
    ),
    (d.LogisticDistributionDescriptor, {"t": 2, "s": 3}, stats.logistic(2, 3)),
    (d.RiceDistributionDescriptor, {"nu": 6, "sigma": 2}, stats.rice(3, scale=2)),
    (d.TukeyDistributionDescriptor, {"lam": 0.5}, stats.tukeylambda(0.5)),
    (d.ParetoDistributionDescriptor, {"shape": 2, "scale": 3}, stats.pareto(2, scale=3)),
    (d.InverseGammaDistributionDescriptor, {"alpha": 2, "beta": 3}, stats.invgamma(2, scale=3)),
    (d.LogLogisticDistributionDescriptor, {"alpha": 3, "beta": 2}, stats.fisk(2, scale=3)),
]


@pytest.mark.parametrize("descriptor, values, reference", CASES)
def test_parameter_mapping_and_probability_contract(descriptor, values, reference):
    parameters = descriptor.DEFAULT.parse(values)
    bound = descriptor.bind(parameters)
    x = reference.ppf(np.array([[0.9, 0.1], [0.7, 0.3]]))
    np.testing.assert_allclose(descriptor.cdf(x, parameters), [[0.9, 0.1], [0.7, 0.3]], atol=1e-8)
    for method in ("cdf", "sf", "logcdf", "logsf"):
        actual = getattr(bound, method)(x)
        assert actual.shape == x.shape
        assert actual.dtype == np.float64
        np.testing.assert_allclose(actual, getattr(reference, method)(x), atol=1e-10)
        assert isinstance(getattr(bound, method)(float(x[0, 0])), float)
        assert getattr(bound, method)([]).shape == (0,)
    np.testing.assert_array_equal(bound.cdf([-np.inf, np.inf]), [0, 1])
    assert np.isnan(bound.cdf(np.nan))


@pytest.mark.parametrize(
    "descriptor, values, location, scale",
    [
        (d.LoConNormDistributionDescriptor, {"a": 3}, 3, 1),
        (d.MixConNormDistributionDescriptor, {"a": 3, "b": 2}, 3, 2),
        (d.ScaleConNormDistributionDescriptor, {"b": 2}, 0, 2),
    ],
)
@pytest.mark.parametrize("p", [0, 0.3, 1])
def test_normal_mixtures(descriptor, values, location, scale, p):
    bound = descriptor.bind(descriptor.DEFAULT.parse({"p": p, **values}))
    x = np.array([[-3, 0], [2, 5]])
    for method in ("cdf", "sf"):
        expected = (1 - p) * getattr(stats.norm, method)(x) + p * getattr(
            stats.norm(location, scale), method
        )(x)
        np.testing.assert_allclose(getattr(bound, method)(x), expected)
        np.testing.assert_allclose(getattr(bound, "log" + method)(x), np.log(expected))
    assert np.isfinite(bound.logcdf(-100))
    assert np.isfinite(bound.logsf(100))
    np.testing.assert_array_equal(bound.cdf([-np.inf, np.inf]), [0, 1])


def test_log_normal_alternative_coordinates():
    descriptor = d.LogNormalDistributionDescriptor
    canonical = descriptor.DEFAULT.parse({"mu": np.log(4), "s": 0.7})
    alternative = descriptor.SHAPE_SCALE.parse({"scale": 4, "s": 0.7})
    np.testing.assert_allclose(
        descriptor.cdf([1, 4, 8], canonical), descriptor.cdf([1, 4, 8], alternative)
    )
    assert descriptor.cdf(4, alternative) == pytest.approx(0.5)


def test_binding_rejects_partial_foreign_and_untyped_parameters():
    descriptor = d.NormalDistributionDescriptor
    with pytest.raises(ValueError, match="missing: var"):
        descriptor.bind(descriptor.DEFAULT.parse({"mean": 0}))
    with pytest.raises(ValueError, match="Unsupported"):
        descriptor.bind(d.GammaDistributionDescriptor.DEFAULT.parse({"alfa": 1, "beta": 1}))
    with pytest.raises(TypeError, match="ParameterValues"):
        descriptor.bind({"mean": 0, "var": 1})


@pytest.mark.parametrize(
    "descriptor, values",
    [
        (d.UniformDistributionDescriptor, {"a": 2, "b": 1}),
        (d.TruncNormDistributionDescriptor, {"mean": 0, "var": 1, "a": 2, "b": 2}),
        (d.HyperbolicDistributionDescriptor, {"alpha": 1, "beta": -1, "delta": 1, "mu": 0}),
        (d.RiceDistributionDescriptor, {"nu": -1, "sigma": 1}),
        (d.RiceDistributionDescriptor, {"nu": 1, "sigma": 0}),
        (d.LogNormalDistributionDescriptor, {"mu": 1000, "s": 1}),
        (d.LogNormalDistributionDescriptor, {"mu": -1000, "s": 1}),
    ],
)
def test_binding_rejects_invalid_joint_or_backend_parameters(descriptor, values):
    with pytest.raises(ValueError):
        descriptor.bind(descriptor.DEFAULT.parse(values))


def test_tail_probabilities_do_not_use_subtraction_or_log_of_cdf():
    normal = d.NormalDistributionDescriptor.bind(
        d.NormalDistributionDescriptor.DEFAULT.parse({"mean": 0, "var": 1})
    )
    assert normal.cdf(10) == 1
    assert normal.sf(10) > 0
    assert np.isfinite(normal.logcdf(-40))
    assert np.isfinite(normal.logsf(40))


def test_all_concrete_descriptors_are_covered():
    covered = {case[0] for case in CASES} | {
        d.LoConNormDistributionDescriptor,
        d.MixConNormDistributionDescriptor,
        d.ScaleConNormDistributionDescriptor,
    }
    assert covered == set(d.ContinuousDistributionDescriptor.__subclasses__())


def test_known_values_and_support_without_backend_reference():
    normal = d.NormalDistributionDescriptor
    assert normal.cdf(3, normal.DEFAULT.parse({"mean": 3, "var": 4})) == 0.5
    exponential = d.ExponentialDistributionDescriptor
    parameters = exponential.DEFAULT.parse({"lam": 2})
    np.testing.assert_allclose(
        exponential.cdf([-1, 0, np.log(2) / 2, np.inf], parameters), [0, 0, 0.5, 1]
    )
    uniform = d.UniformDistributionDescriptor
    np.testing.assert_allclose(
        uniform.cdf([-5, 2, 3, 6, 8], uniform.DEFAULT.parse({"a": 2, "b": 6})),
        [0, 0, 0.25, 1, 1],
    )
    truncated = d.TruncNormDistributionDescriptor
    np.testing.assert_allclose(
        truncated.cdf([1, 3, 5], truncated.DEFAULT.parse({"mean": 3, "var": 4, "a": 1, "b": 5})),
        [0, 0.5, 1],
    )


def test_zero_dimensional_input_and_no_parameter_mutation():
    descriptor = d.NormalDistributionDescriptor
    parameters = descriptor.DEFAULT.parse({"mean": 0, "var": 4})
    before = parameters.storage_identity()
    bound = descriptor.bind(parameters)
    assert isinstance(bound.cdf(np.array(0.0)), float)
    assert parameters.storage_identity() == before


@pytest.mark.parametrize(
    "descriptor, values",
    [
        (d.InvGaussDistributionDescriptor, {"mu": 1e-300, "lam": 1e300}),
        (
            d.HyperbolicDistributionDescriptor,
            {"alpha": 1e-300, "beta": 0, "delta": 1e-300, "mu": 0},
        ),
    ],
)
def test_binding_rejects_underflow_in_positive_shape(descriptor, values):
    with pytest.raises(ValueError, match="represented"):
        descriptor.bind(descriptor.DEFAULT.parse(values))
