"""Numerical references and fixed/free parameter contracts for every uniform statistic."""

import inspect

import numpy as np
import pytest
from scipy import integrate, stats

from pysatl_criterion.distribution import UniformDistributionDescriptor
from pysatl_criterion.distribution import UniformDistributionDescriptor as Uniform
from pysatl_criterion.hypothesis_testing.limit_distribution.base import (
    MonteCarloLimitDistributionResolver,
)
from pysatl_criterion.statistics.alternative import TwoSidedAlternative
from pysatl_criterion.statistics.goodness_of_fit.binning import EqualProbabilityBins
from pysatl_criterion.statistics.goodness_of_fit.continuous import uniform as u
from pysatl_criterion.statistics.hypothesis import GoodnessOfFitHypothesis
from tests.parameter_cases import parameters_for


FIXED = [
    cls
    for _, cls in inspect.getmembers(u, inspect.isclass)
    if issubclass(cls, u.AbstractUniformGofStatistic) and not inspect.isabstract(cls)
]
SAMPLE = np.array([0.81, 0.03, 0.29, 0.52, 0.17])


@pytest.mark.parametrize("cls", FIXED)
def test_fixed_bounds_contract_and_affine_equivariance(cls):
    standard = cls(parameters_for(cls))
    shifted = cls(parameters_for(cls, a=-3, b=5))
    assert shifted.hypothesis().parameters() == {"a": -3, "b": 5}
    assert cls.supports_hypothesis(shifted.hypothesis())
    for free in ({}, {"a": -3}, {"b": 5}):
        values = Uniform.DEFAULT.parse(free)
        assert not cls.supports_hypothesis(GoodnessOfFitHypothesis(values))
        with pytest.raises(ValueError, match="Unsupported hypothesis or parameterization"):
            cls(values)
    assert cls(shifted.hypothesis().parameter_values).hypothesis().parameters() == {
        "a": -3,
        "b": 5,
    }
    assert shifted.execute_statistic(-3 + 8 * SAMPLE) == pytest.approx(
        standard.execute_statistic(SAMPLE), rel=1e-11, abs=1e-14
    )
    # Holding observations fixed while changing the null interval changes the statistic.
    assert cls(parameters_for(cls, a=-1, b=2)).execute_statistic(SAMPLE) != pytest.approx(
        standard.execute_statistic(SAMPLE)
    )


@pytest.mark.parametrize("cls", FIXED)
@pytest.mark.parametrize(("a", "b"), [(np.nan, 1), (0, np.inf), (-np.inf, 1), (2, 1), (1, 1)])
def test_invalid_fixed_bounds(cls, a, b):
    with pytest.raises(ValueError):
        cls(parameters_for(cls, a=a, b=b))


@pytest.mark.parametrize("cls", FIXED)
@pytest.mark.parametrize("sample", [[], [0.1, np.nan], [np.inf, 0.5], [[0.1, 0.2]]])
def test_invalid_samples(cls, sample):
    with pytest.raises(ValueError):
        cls(parameters_for(cls)).execute_statistic(sample)


@pytest.mark.parametrize(
    ("cls", "scipy_alternative"),
    [
        (u.KolmogorovSmirnovUniformGofStatistic, "two-sided"),
        (u.KolmogorovSmirnovDPlusUniformGofStatistic, "greater"),
        (u.KolmogorovSmirnovDMinusUniformGofStatistic, "less"),
    ],
)
def test_ks_reference_with_ties_and_outside_support(cls, scipy_alternative):
    sample = [-4, -1, -1, 2, 4, 8]
    reference = stats.uniform(loc=-3, scale=8)
    actual = cls(UniformDistributionDescriptor.DEFAULT.parse({"a": -3, "b": 5})).execute_statistic(
        sample
    )
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
    assert u.WatsonUniformGofStatistic(
        UniformDistributionDescriptor.DEFAULT.parse({"a": 0, "b": 1})
    ).execute_statistic(sample) == pytest.approx(
        n * integral(lambda x: (discrepancy(x) - mean) ** 2)
    )
    assert u.CrammerVonMisesUniformGofStatistic(
        UniformDistributionDescriptor.DEFAULT.parse({"a": 0, "b": 1})
    ).execute_statistic(sample) == pytest.approx(stats.cramervonmises(sample, "uniform").statistic)
    assert u.AndersonDarlingUniformGofStatistic(
        UniformDistributionDescriptor.DEFAULT.parse({"a": 0, "b": 1})
    ).execute_statistic(sample) == pytest.approx(
        n * integral(lambda x: discrepancy(x) ** 2 / (x * (1 - x)))
    )
    assert u.KuiperUniformGofStatistic(
        UniformDistributionDescriptor.DEFAULT.parse({"a": 0, "b": 1})
    ).execute_statistic(sample) == pytest.approx(
        stats.kstest(sample, "uniform", alternative="greater").statistic
        + stats.kstest(sample, "uniform", alternative="less").statistic
    )


@pytest.mark.parametrize(
    ("criterion_class", "expected"),
    [
        (u.ZhangAUniformGofStatistic, 3.5389423702359975878),
        (u.ZhangCUniformGofStatistic, 4.2917566201014437410),
        (u.ZhangKUniformGofStatistic, 0.4851757595712566718),
    ],
)
def test_zhang_published_formulas(criterion_class, expected):
    # Zhang (2002), evaluated with 50-digit Decimal arithmetic for the five ordered values.
    assert criterion_class(
        UniformDistributionDescriptor.DEFAULT.parse({"a": 0, "b": 1})
    ).execute_statistic(SAMPLE) == pytest.approx(expected, rel=1e-13)


@pytest.mark.parametrize(
    "statistic",
    [
        u.AndersonDarlingUniformGofStatistic(
            UniformDistributionDescriptor.DEFAULT.parse({"a": 0, "b": 1})
        ),
        *[
            criterion_class(
                UniformDistributionDescriptor.DEFAULT.parse({"a": 0, "b": 1})
            )
            for criterion_class in (
                u.ZhangAUniformGofStatistic,
                u.ZhangCUniformGofStatistic,
                u.ZhangKUniformGofStatistic,
            )
        ],
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
    assert cls(parameters_for(cls, a=2, b=12)).execute_statistic([9, 3, 11, 5]) == pytest.approx(
        expected
    )


@pytest.mark.parametrize("k", [1, 2, 3, 4, 8])
def test_neyman_against_numpy_legendre_basis(k):
    expected = 0.0
    for j in range(1, k + 1):
        polynomial = np.polynomial.legendre.Legendre.basis(j)
        expected += (2 * j + 1) * np.sum(polynomial(2 * SAMPLE - 1)) ** 2 / len(SAMPLE)
    assert u.NeymanSmoothTestUniformGofStatistic(
        UniformDistributionDescriptor.DEFAULT.parse({"a": 0, "b": 1}), k=k
    ).execute_statistic(SAMPLE) == pytest.approx(expected)


@pytest.mark.parametrize("bandwidth", [0.01, 0.15, 2, "auto"])
def test_bickel_rosenblatt_against_adaptive_quadrature(bandwidth):
    h = 1.06 * np.std(SAMPLE) * len(SAMPLE) ** (-0.2) if bandwidth == "auto" else bandwidth

    def integrand(x):
        return (np.mean(stats.norm.pdf(x, loc=SAMPLE, scale=h)) - 1) ** 2

    expected = integrate.quad(integrand, 0, 1, points=np.sort(SAMPLE), epsabs=1e-11)[0]
    actual = u.BickelRosenblattUniformGofStatistic(
        UniformDistributionDescriptor.DEFAULT.parse({"a": 0, "b": 1}), bandwidth=bandwidth
    ).execute_statistic(SAMPLE)
    assert actual == pytest.approx(expected, rel=1e-10, abs=1e-12)


def test_bickel_rosenblatt_narrow_kernel_and_constant_sample():
    h = 1e-6
    # For a point mass at 0.5, boundary Gaussian tails are negligible at this bandwidth.
    actual = u.BickelRosenblattUniformGofStatistic(
        UniformDistributionDescriptor.DEFAULT.parse({"a": 0, "b": 1}), bandwidth=h
    ).execute_statistic([0.5, 0.5])
    assert actual == pytest.approx(1 / (2 * np.sqrt(np.pi) * h) - 1)
    with pytest.raises(ValueError, match="bandwidth"):
        u.BickelRosenblattUniformGofStatistic(
            UniformDistributionDescriptor.DEFAULT.parse({"a": 0, "b": 1})
        ).execute_statistic([0.5, 0.5])


@pytest.mark.parametrize(
    ("cls", "lambda_"),
    [
        (u.Chi2PearsonUniformGofStatistic, 1),
        (u.ModifiedLogLikelihoodUniformGofStatistic, -1),
        (u.NeymanChi2UniformGofStatistic, -2),
        (u.FreemanTukeyUniformGofStatistic, -0.5),
        (u.CressieReadUniformGofStatistic, 2 / 3),
    ],
)
def test_chi_square_against_scipy(cls, lambda_):
    sample = [0.1, 0.1, 0.4, 0.6, 0.8, 0.9]
    expected = stats.power_divergence([2, 1, 1, 2], [1.5] * 4, lambda_=lambda_).statistic
    actual = cls(
        UniformDistributionDescriptor.DEFAULT.parse({"a": 0, "b": 1}),
        binning=EqualProbabilityBins(4),
    ).execute_statistic(sample)
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize(
    ("cls", "expected"),
    [
        (u.Chi2PearsonUniformGofStatistic, 6),
        (u.FreemanTukeyUniformGofStatistic, 8),
        (u.ModifiedLogLikelihoodUniformGofStatistic, np.inf),
        (u.NeymanChi2UniformGofStatistic, np.inf),
    ],
)
def test_chi_square_empty_bin_limits(cls, expected):
    with np.errstate(divide="ignore"):
        actual = cls(
            UniformDistributionDescriptor.DEFAULT.parse({"a": 0, "b": 1}),
            binning=EqualProbabilityBins(4),
        ).execute_statistic([0.1, 0.2])
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [(u.NeymanSmoothTestUniformGofStatistic, {"k": k}) for k in [0, -1, 1.5, True]]
    + [
        (u.BickelRosenblattUniformGofStatistic, {"bandwidth": h})
        for h in [0, -1, np.nan, np.inf, "bad"]
    ],
)
def test_invalid_algorithm_options(cls, kwargs):
    with pytest.raises(ValueError):
        cls(parameters_for(cls), **kwargs)


def test_stein_order_statistic_reference_and_both_tails():
    sample = np.sort(SAMPLE)
    n = len(sample)
    # Equivalent order-statistic expression in Sreedevi and Kattumannil (2023).
    expected = sum((2 * (i - n) + (n - 1) * x) * x for i, x in enumerate(sample, 1)) / (n * (n - 1))
    for cls in (u.SteinUniformGofStatistic, u.CensoredSteinUniformGofStatistic):
        assert isinstance(cls(parameters_for(cls)).alternative(), TwoSidedAlternative)
        assert cls(parameters_for(cls)).execute_statistic(SAMPLE) == pytest.approx(expected)
        assert cls(parameters_for(cls)).execute_statistic([0.5] * 4) < 0
        assert cls(parameters_for(cls)).execute_statistic([0, 0, 1, 1]) > 0
        for sample in ([], [0.5]):
            with pytest.raises(ValueError):
                cls(parameters_for(cls)).execute_statistic(sample)


def test_censored_stein_hand_calculated_weights_and_sample_denominator():
    statistic = u.CensoredSteinUniformGofStatistic(
        UniformDistributionDescriptor.DEFAULT.parse({"a": 0, "b": 1})
    )
    # Event weights 1, 3/2, 3/2, with denominator choose(4,2), not choose(3,2).
    assert statistic.execute_statistic([0.1, 0.2, 0.5, 0.8], [0, 1, 0, 0]) == pytest.approx(
        0.043125
    )
    assert statistic.execute_statistic(SAMPLE, [0] * 5) == pytest.approx(
        u.SteinUniformGofStatistic(
            UniformDistributionDescriptor.DEFAULT.parse({"a": 0, "b": 1})
        ).execute_statistic(SAMPLE)
    )


def test_censored_stein_ties_are_order_independent():
    sample = np.array([0.1, 0.2, 0.2, 0.8])
    flags = np.array([0, 0, 1, 0])
    statistic = u.CensoredSteinUniformGofStatistic(
        UniformDistributionDescriptor.DEFAULT.parse({"a": 0, "b": 1})
    )
    # At the tie the observed event precedes the censoring: K_c(0.2-)=1, K_c(0.8-)=1/2.
    for indices in ([0, 1, 2, 3], [0, 2, 1, 3], [3, 2, 0, 1]):
        assert statistic.execute_statistic(sample[indices], flags[indices]) == pytest.approx(
            0.655 / 6
        )


@pytest.mark.parametrize("flags", [[0], [0, 2, 0, 0, 0], [0, np.nan, 0, 0, 0], [[0] * 5]])
def test_invalid_censoring_flags(flags):
    with pytest.raises(ValueError, match="censoring_indices"):
        u.CensoredSteinUniformGofStatistic(
            UniformDistributionDescriptor.DEFAULT.parse({"a": 0, "b": 1})
        ).execute_statistic(SAMPLE, flags)


@pytest.mark.parametrize(
    "statistic",
    [
        u.KolmogorovSmirnovUniformGofStatistic(
            UniformDistributionDescriptor.DEFAULT.parse({"a": 2, "b": 5})
        ),
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


@pytest.mark.parametrize("cls", FIXED)
def test_uniform_criterion_docstring_has_scientific_reference(cls):
    documentation = inspect.getdoc(cls)
    assert documentation
    assert "References" in documentation or "https://doi.org/" in documentation
    assert "https://" in documentation


@pytest.mark.parametrize("sample", [[0.2, 0.8], [0.2, 0.2, 0.7], SAMPLE, [0, 0.5, 1]])
def test_stein_matches_published_order_statistic_representation(sample):
    # Kattumannil and Sreedevi (2021), Section 2: an O(n) order-statistic
    # representation independently checks the pairwise U-statistic and its scaling.
    ordered = np.sort(sample)
    n = len(ordered)
    ranks = np.arange(1, n + 1)
    reference = np.sum((2 * (ranks - n) + (n - 1) * ordered) * ordered) / (n * (n - 1))
    statistic = u.SteinUniformGofStatistic(Uniform.DEFAULT.parse({"a": 0, "b": 1}))
    assert statistic.execute_statistic(sample) == pytest.approx(reference, abs=1e-15)


def test_neyman_first_four_components_against_published_polynomials():
    # Rayner and Rayner (2001), equation (1); no special-function evaluator.
    x = SAMPLE
    polynomials = [
        np.sqrt(3) * (2 * x - 1),
        np.sqrt(5) * (6 * x**2 - 6 * x + 1),
        np.sqrt(7) * (20 * x**3 - 30 * x**2 + 12 * x - 1),
        3 * (70 * x**4 - 140 * x**3 + 90 * x**2 - 20 * x + 1),
    ]
    for k in range(1, 5):
        reference = sum(np.sum(p) ** 2 for p in polynomials[:k]) / len(x)
        statistic = u.NeymanSmoothTestUniformGofStatistic(
            Uniform.DEFAULT.parse({"a": 0, "b": 1}), k=k
        )
        assert statistic.execute_statistic(x) == pytest.approx(reference, rel=1e-12)
