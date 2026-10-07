"""Numerical references and fixed/free parameter contracts for every uniform statistic."""

import inspect

import numpy as np
import pytest
from scipy import integrate, stats

from pysatl_criterion.distribution.distributions import UniformDistributionDescriptor as Uniform
from pysatl_criterion.hypothesis_testing.limit_distribution.base import (
    MonteCarloLimitDistributionResolver,
)
from pysatl_criterion.statistics.alternative import AlternativeType, TwoSidedAlternative
from pysatl_criterion.statistics.goodness_of_fit import uniform as u
from pysatl_criterion.statistics.hypothesis import GoodnessOfFitHypothesis


FIXED = [
    cls
    for _, cls in inspect.getmembers(u, inspect.isclass)
    if issubclass(cls, u.AbstractUniformGofStatistic)
    and not inspect.isabstract(cls)
    and cls is not u.LillieforsTestUniformGofStatistic
]
SAMPLE = np.array([0.81, 0.03, 0.29, 0.52, 0.17])


@pytest.mark.parametrize("cls", FIXED)
def test_fixed_bounds_contract_and_affine_equivariance(cls):
    standard = cls()
    shifted = cls(a=-3, b=5)
    assert shifted.hypothesis().parameters() == {"a": -3, "b": 5}
    assert cls.supports_hypothesis(shifted.hypothesis())
    for free in ({}, {"a": -3}, {"b": 5}):
        values = Uniform.DEFAULT.parse(free)
        assert not cls.supports_hypothesis(GoodnessOfFitHypothesis(values))
        with pytest.raises(ValueError, match="Unsupported Uniform"):
            cls.from_parameters(values)
    assert cls.from_parameters(shifted.hypothesis().parameter_values).hypothesis().parameters() == {
        "a": -3,
        "b": 5,
    }
    assert shifted.execute_statistic(-3 + 8 * SAMPLE) == pytest.approx(
        standard.execute_statistic(SAMPLE), rel=1e-11, abs=1e-14
    )
    # Holding observations fixed while changing the null interval changes the statistic.
    assert cls(a=-1, b=2).execute_statistic(SAMPLE) != pytest.approx(
        standard.execute_statistic(SAMPLE)
    )


@pytest.mark.parametrize("cls", FIXED)
@pytest.mark.parametrize(("a", "b"), [(np.nan, 1), (0, np.inf), (-np.inf, 1), (2, 1), (1, 1)])
def test_invalid_fixed_bounds(cls, a, b):
    with pytest.raises(ValueError):
        cls(a=a, b=b)


@pytest.mark.parametrize("cls", [*FIXED, u.LillieforsTestUniformGofStatistic])
@pytest.mark.parametrize("sample", [[], [0.1, np.nan], [np.inf, 0.5], [[0.1, 0.2]]])
def test_invalid_samples(cls, sample):
    with pytest.raises(ValueError):
        cls().execute_statistic(sample)


@pytest.mark.parametrize(
    ("alternative", "scipy_alternative"),
    [
        (AlternativeType.TWO_TAILED, "two-sided"),
        (AlternativeType.RIGHT, "greater"),
        (AlternativeType.LEFT, "less"),
    ],
)
def test_ks_reference_with_ties_and_outside_support(alternative, scipy_alternative):
    sample = [-4, -1, -1, 2, 4, 8]
    reference = stats.uniform(loc=-3, scale=8)
    actual = u.KolmogorovSmirnovUniformGofStatistic(-3, 5, alternative).execute_statistic(sample)
    assert actual == pytest.approx(
        stats.kstest(sample, reference.cdf, alternative=scipy_alternative).statistic
    )


def test_edf_statistics_against_integral_definitions():
    sample = np.sort(SAMPLE)
    n = len(sample)

    def discrepancy(x):
        return np.count_nonzero(sample <= x) / n - x

    def integral(f):
        return integrate.quad(f, 0, 1, points=sample, epsabs=1e-12)[0]

    mean = integral(discrepancy)
    assert u.WatsonUniformGofStatistic().execute_statistic(sample) == pytest.approx(
        n * integral(lambda x: (discrepancy(x) - mean) ** 2)
    )
    assert u.CrammerVonMisesUniformGofStatistic().execute_statistic(sample) == pytest.approx(
        stats.cramervonmises(sample, "uniform").statistic
    )
    assert u.AndersonDarlingUniformGofStatistic().execute_statistic(sample) == pytest.approx(
        n * integral(lambda x: discrepancy(x) ** 2 / (x * (1 - x)))
    )
    assert u.KuiperUniformGofStatistic().execute_statistic(sample) == pytest.approx(
        stats.kstest(sample, "uniform", alternative="greater").statistic
        + stats.kstest(sample, "uniform", alternative="less").statistic
    )


def test_lilliefors_fits_bounds_and_rejects_fixed_parameters():
    statistic = u.LillieforsTestUniformGofStatistic()
    assert statistic.hypothesis().parameters() == {}
    assert statistic.supports_hypothesis(statistic.hypothesis())
    assert not statistic.supports_hypothesis(u.KolmogorovSmirnovUniformGofStatistic().hypothesis())
    assert statistic.from_parameters(Uniform.DEFAULT.parse({})).hypothesis().parameters() == {}
    for kwargs in ({"a": 0}, {"b": 1}, {"a": 0, "b": 1}):
        with pytest.raises(TypeError):
            u.LillieforsTestUniformGofStatistic(**kwargs)
    reference = stats.uniform(loc=SAMPLE.min(), scale=np.ptp(SAMPLE))
    expected = stats.kstest(SAMPLE, reference.cdf).statistic
    assert statistic.execute_statistic(SAMPLE) == pytest.approx(expected)
    assert statistic.execute_statistic(4 + 7 * SAMPLE) == pytest.approx(expected)
    assert expected != pytest.approx(
        u.KolmogorovSmirnovUniformGofStatistic().execute_statistic(SAMPLE)
    )
    with pytest.raises(ValueError, match="range"):
        statistic.execute_statistic([0.3, 0.3])


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("A", 3.5389423702359975878),
        ("C", 4.2917566201014437410),
        ("K", 0.4851757595712566718),
    ],
)
def test_zhang_published_formulas(kind, expected):
    # Zhang (2002), evaluated with 50-digit Decimal arithmetic for the five ordered values.
    assert u.ZhangTestsUniformGofStatistic(test_type=kind).execute_statistic(
        SAMPLE
    ) == pytest.approx(expected, rel=1e-13)


@pytest.mark.parametrize(
    "statistic",
    [
        u.AndersonDarlingUniformGofStatistic(),
        *[u.ZhangTestsUniformGofStatistic(test_type=t) for t in "ACK"],
    ],
)
@pytest.mark.parametrize("endpoint", [-1, 0, 1, 2])
def test_logarithmic_statistics_boundary_limits(statistic, endpoint):
    assert statistic.execute_statistic([0.2, endpoint, 0.7]) == np.inf


@pytest.mark.parametrize(
    ("cls", "expected"),
    [
        (u.GreenwoodTestUniformGofStatistic, 0.26),
        (u.ShermanUniformGofStatistic, 0.2),
        (u.QuesenberryMillerUniformGofStatistic, 0.46),
    ],
)
def test_spacing_statistics_hand_calculation(cls, expected):
    # Unit spacings: 0.1, 0.2, 0.4, 0.2, 0.1.
    assert cls(a=2, b=12).execute_statistic([9, 3, 11, 5]) == pytest.approx(expected)


@pytest.mark.parametrize("k", [1, 2, 3, 4, 8])
def test_neyman_against_numpy_legendre_basis(k):
    expected = 0.0
    for j in range(1, k + 1):
        polynomial = np.polynomial.legendre.Legendre.basis(j)
        expected += (2 * j + 1) * np.sum(polynomial(2 * SAMPLE - 1)) ** 2 / len(SAMPLE)
    assert u.NeymanSmoothTestUniformGofStatistic(k=k).execute_statistic(SAMPLE) == pytest.approx(
        expected
    )


@pytest.mark.parametrize("bandwidth", [0.01, 0.15, 2, "auto"])
def test_bickel_rosenblatt_against_adaptive_quadrature(bandwidth):
    h = 1.06 * np.std(SAMPLE) * len(SAMPLE) ** (-0.2) if bandwidth == "auto" else bandwidth

    def integrand(x):
        return (np.mean(stats.norm.pdf(x, loc=SAMPLE, scale=h)) - 1) ** 2

    expected = integrate.quad(integrand, 0, 1, points=np.sort(SAMPLE), epsabs=1e-11)[0]
    actual = u.BickelRosenblattUniformGofStatistic(bandwidth=bandwidth).execute_statistic(SAMPLE)
    assert actual == pytest.approx(expected, rel=1e-10, abs=1e-12)


def test_bickel_rosenblatt_narrow_kernel_and_constant_sample():
    h = 1e-6
    # For a point mass at 0.5, boundary Gaussian tails are negligible at this bandwidth.
    actual = u.BickelRosenblattUniformGofStatistic(bandwidth=h).execute_statistic([0.5, 0.5])
    assert actual == pytest.approx(1 / (2 * np.sqrt(np.pi) * h) - 1)
    with pytest.raises(ValueError, match="bandwidth"):
        u.BickelRosenblattUniformGofStatistic().execute_statistic([0.5, 0.5])


@pytest.mark.parametrize("lambda_", [1, 0, -1, -2, -0.5, 2 / 3])
def test_chi_square_against_scipy(lambda_):
    sample = [0.1, 0.1, 0.4, 0.6, 0.8, 0.9]
    expected = stats.power_divergence([2, 1, 1, 2], [1.5] * 4, lambda_=lambda_).statistic
    actual = u.Chi2PearsonUniformGofStatistic(bins=4, lambda_=lambda_).execute_statistic(sample)
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize(
    ("lambda_", "expected"),
    [
        (1, 6),
        (0, 4 * np.log(4)),
        (-0.5, 8),
        (-1, np.inf),
        (-2, np.inf),
    ],
)
def test_chi_square_empty_bin_limits(lambda_, expected):
    with np.errstate(divide="ignore"):
        actual = u.Chi2PearsonUniformGofStatistic(bins=4, lambda_=lambda_).execute_statistic(
            [0.1, 0.2]
        )
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [(u.NeymanSmoothTestUniformGofStatistic, {"k": k}) for k in [0, -1, 1.5, True]]
    + [
        (u.BickelRosenblattUniformGofStatistic, {"bandwidth": h})
        for h in [0, -1, np.nan, np.inf, "bad"]
    ]
    + [(u.Chi2PearsonUniformGofStatistic, {"bins": b}) for b in [0, 1, 2.5, True, "bad"]],
)
def test_invalid_algorithm_options(cls, kwargs):
    with pytest.raises(ValueError):
        cls(**kwargs)


def test_stein_order_statistic_reference_and_both_tails():
    sample = np.sort(SAMPLE)
    n = len(sample)
    # Equivalent order-statistic expression in Sreedevi and Kattumannil (2023).
    expected = sum((2 * (i - n) + (n - 1) * x) * x for i, x in enumerate(sample, 1)) / (n * (n - 1))
    for cls in (u.SteinUniformGofStatistic, u.CensoredSteinUniformGofStatistic):
        assert isinstance(cls().alternative(), TwoSidedAlternative)
        assert cls().execute_statistic(SAMPLE) == pytest.approx(expected)
        assert cls().execute_statistic([0.5] * 4) < 0
        assert cls().execute_statistic([0, 0, 1, 1]) > 0
        for sample in ([], [0.5]):
            with pytest.raises(ValueError):
                cls().execute_statistic(sample)


def test_censored_stein_hand_calculated_weights_and_sample_denominator():
    statistic = u.CensoredSteinUniformGofStatistic()
    # Event weights 1, 3/2, 3/2, with denominator choose(4,2), not choose(3,2).
    assert statistic.execute_statistic([0.1, 0.2, 0.5, 0.8], [0, 1, 0, 0]) == pytest.approx(
        0.043125
    )
    assert statistic.execute_statistic(SAMPLE, [0] * 5) == pytest.approx(
        u.SteinUniformGofStatistic().execute_statistic(SAMPLE)
    )


def test_censored_stein_ties_are_order_independent():
    sample = np.array([0.1, 0.2, 0.2, 0.8])
    flags = np.array([0, 0, 1, 0])
    statistic = u.CensoredSteinUniformGofStatistic()
    # At the tie the observed event precedes the censoring: K_c(0.2-)=1, K_c(0.8-)=1/2.
    for indices in ([0, 1, 2, 3], [0, 2, 1, 3], [3, 2, 0, 1]):
        assert statistic.execute_statistic(sample[indices], flags[indices]) == pytest.approx(
            0.655 / 6
        )


@pytest.mark.parametrize("flags", [[0], [0, 2, 0, 0, 0], [0, np.nan, 0, 0, 0], [[0] * 5]])
def test_invalid_censoring_flags(flags):
    with pytest.raises(ValueError, match="censoring_indices"):
        u.CensoredSteinUniformGofStatistic().execute_statistic(SAMPLE, flags)


@pytest.mark.parametrize(
    "statistic",
    [
        u.KolmogorovSmirnovUniformGofStatistic(a=2, b=5),
        u.LillieforsTestUniformGofStatistic(),
    ],
)
def test_monte_carlo_uses_the_declared_hypothesis(mocker, statistic):
    parameters = {"a": 0, "b": 1, **statistic.hypothesis().parameters()}
    sample = parameters["a"] + (parameters["b"] - parameters["a"]) * SAMPLE
    sampler = mocker.patch(
        "pysatl_criterion.generator.generators.generate_uniform", return_value=sample
    )
    result = MonteCarloLimitDistributionResolver(2).resolve(statistic, len(sample))
    assert result == pytest.approx([statistic.execute_statistic(sample)] * 2)
    assert (
        sampler.call_args_list
        == [mocker.call(size=len(sample), random_state=None, **parameters)] * 2
    )
