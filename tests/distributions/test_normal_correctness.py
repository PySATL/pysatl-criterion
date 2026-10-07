"""Independent formula and contract regressions for the normality review."""

import doctest
import inspect
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate, optimize, special, stats

from pysatl_criterion.distribution.distributions import NormalDistributionDescriptor
from pysatl_criterion.hypothesis_testing.limit_distribution.base import (
    MonteCarloLimitDistributionResolver,
    StorageLimitDistributionResolver,
)
from pysatl_criterion.statistics.alternative import AlternativeType
from pysatl_criterion.statistics.goodness_of_fit import normal
from tests.parameter_cases import parameters_for


CLASSES = [
    cls
    for _, cls in inspect.getmembers(normal, inspect.isclass)
    if cls.__module__ == normal.__name__ and not inspect.isabstract(cls)
]
FIXED = (normal.KolmogorovSmirnovNormalityGofStatistic, normal.CramerVonMiseNormalityGofStatistic)
SAMPLE = np.random.default_rng(2307).normal(size=24)


@pytest.mark.parametrize("cls", CLASSES)
def test_scalar_interface_independent_calls_and_input_ownership(cls):
    statistic = cls(parameters_for(cls))
    original = SAMPLE.copy()
    first = statistic.execute_statistic(original, compatibility=True)
    assert isinstance(first, (float, np.float64))
    assert np.isfinite(first)
    np.testing.assert_array_equal(original, SAMPLE)
    second = np.random.default_rng(991).normal(size=32)
    assert statistic.execute_statistic(second) == pytest.approx(
        cls(parameters_for(cls)).execute_statistic(second)
    )
    assert statistic.execute_statistic(original.tolist()) == pytest.approx(first)
    assert statistic.execute_statistic(original) == pytest.approx(first)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("bad", [[], [[1, 2], [3, 4]], [0, np.nan], [1, np.inf], [1 + 2j]])
def test_invalid_samples_raise_value_error(cls, bad):
    with pytest.raises(ValueError):
        cls(parameters_for(cls)).execute_statistic(bad)


@pytest.mark.parametrize("cls", [c for c in CLASSES if c not in FIXED])
def test_constant_samples_are_undefined(cls):
    with pytest.raises(ValueError):
        cls(parameters_for(cls)).execute_statistic(np.ones(32))


@pytest.mark.parametrize("cls", FIXED)
def test_fixed_cdf_accepts_singleton_and_constant(cls):
    assert np.isfinite(cls(parameters_for(cls)).execute_statistic([1]))
    assert np.isfinite(cls(parameters_for(cls)).execute_statistic([2] * 10))


@pytest.mark.parametrize(
    "cls",
    [
        c
        for c in CLASSES
        if c not in FIXED and not issubclass(c, normal.AbstractGraphNormalityGofStatistic)
    ],
)
@pytest.mark.parametrize("scale", [1e-250, 1e250])
def test_family_statistics_have_no_overflow_or_underflow_at_extreme_scales(cls, scale):
    statistic = cls(parameters_for(cls))
    expected = statistic.execute_statistic(SAMPLE)
    assert statistic.execute_statistic(scale * SAMPLE) == pytest.approx(
        expected, rel=2e-8, abs=2e-9
    )


@pytest.mark.parametrize(
    "cls,n",
    [
        (normal.ShapiroWilkNormalityGofStatistic, 3),
        (normal.SkewNormalityGofStatistic, 8),
        (normal.KurtosisNormalityGofStatistic, 5),
        (normal.DAPNormalityGofStatistic, 8),
        (normal.Hosking1NormalityGofStatistic, 4),
        (normal.Hosking2NormalityGofStatistic, 6),
        (normal.Hosking3NormalityGofStatistic, 8),
        (normal.Hosking4NormalityGofStatistic, 10),
        (normal.ZhangQNormalityGofStatistic, 8),
        (normal.ZhangQStarNormalityGofStatistic, 8),
    ],
)
def test_minimum_sample_sizes(cls, n):
    with pytest.raises(ValueError):
        cls(parameters_for(cls)).execute_statistic(np.arange(n - 1))
    assert np.isfinite(cls(parameters_for(cls)).execute_statistic(np.arange(n)))


@pytest.mark.parametrize("cls", FIXED)
@pytest.mark.parametrize("bad", [[1], np.array([1, 2]), True, 1j, "1", None])
def test_fixed_parameters_must_be_real_scalars(cls, bad):
    for name in ("mean", "var"):
        with pytest.raises(ValueError):
            cls(parameters_for(cls, **{name: bad}))


def test_symmetric_skew_is_zero_and_dap_does_not_add_a_spurious_component():
    x = np.arange(-12.0, 13)
    assert normal.SkewNormalityGofStatistic.skew_test(x) == 0
    assert normal.SkewNormalityGofStatistic(
        NormalDistributionDescriptor.DEFAULT.parse({})
    ).execute_statistic(x) == pytest.approx(0, abs=1e-14)
    kurt = stats.kurtosistest(x).statistic
    assert normal.DAPNormalityGofStatistic(
        NormalDistributionDescriptor.DEFAULT.parse({})
    ).execute_statistic(x) == pytest.approx(kurt**2)


@pytest.mark.parametrize(
    "cls,reference",
    [
        (normal.ShapiroWilkNormalityGofStatistic, stats.shapiro),
        (normal.JBNormalityGofStatistic, stats.jarque_bera),
        (normal.SkewNormalityGofStatistic, stats.skewtest),
        (normal.KurtosisNormalityGofStatistic, stats.kurtosistest),
        (normal.DAPNormalityGofStatistic, stats.normaltest),
    ],
)
def test_reference_implementations(cls, reference):
    assert cls(parameters_for(cls)).execute_statistic(SAMPLE) == pytest.approx(
        reference(SAMPLE).statistic, abs=1e-8
    )


def test_lilliefors_refits_each_sample():
    statistic = normal.LillieforsNormalityGofStatistic(
        NormalDistributionDescriptor.DEFAULT.parse({})
    )
    for x in (SAMPLE, np.exp(SAMPLE), SAMPLE[::-1]):
        z = (x - np.mean(x)) / np.std(x, ddof=1)
        assert statistic.execute_statistic(x) == pytest.approx(stats.kstest(z, "norm").statistic)


def test_epps_pulley_matches_characteristic_function_integral():
    z = (SAMPLE - SAMPLE.mean()) / SAMPLE.std()

    def integrand(t):
        delta = np.mean(np.exp(1j * t * z)) - np.exp(-t * t / 2)
        return len(z) * abs(delta) ** 2 * stats.norm.pdf(t)

    expected = integrate.quad(integrand, -12, 12, epsabs=1e-10)[0]
    assert normal.EppsPulleyNormalityGofStatistic(
        NormalDistributionDescriptor.DEFAULT.parse({})
    ).execute_statistic(SAMPLE) == pytest.approx(expected)


def cabana_reference(x, kurtosis):
    """Direct Hermite evaluation and scalar optimization, without polynomial roots."""
    z = (x - np.mean(x)) / np.std(x, ddof=1)

    def h(j, value):
        return special.eval_hermitenorm(j, value) / math.sqrt(math.factorial(j))

    means = {j: np.sum(h(j, z)) / np.sqrt(len(z)) for j in range(3, 9)}

    def process(t):
        if kurtosis:
            series = sum(
                (np.sqrt(j / (j - 1)) * h(j - 2, t) + h(j, t)) * means[j + 3] for j in range(2, 6)
            )
            return (
                -stats.norm.pdf(t) * means[3]
                + (stats.norm.cdf(t) - t * stats.norm.pdf(t)) * means[4]
                - stats.norm.pdf(t) * series
            )
        series = sum(h(j - 1, t) * means[j + 3] / np.sqrt(j) for j in range(1, 6))
        return stats.norm.cdf(t) * means[3] - stats.norm.pdf(t) * series

    grid = np.linspace(-12, 12, 241)
    values = np.abs(process(grid))
    maxima = [abs(means[4 if kurtosis else 3])]
    for i in range(1, len(grid) - 1):
        if values[i] >= values[i - 1] and values[i] >= values[i + 1]:
            result = optimize.minimize_scalar(
                lambda t: -abs(process(t)), bounds=(grid[i - 1], grid[i + 1]), method="bounded"
            )
            maxima.append(-result.fun)
    return max(maxima)


@pytest.mark.parametrize(
    "cls,kurtosis",
    [
        (normal.CabanaCabana1NormalityGofStatistic, False),
        (normal.CabanaCabana2NormalityGofStatistic, True),
    ],
)
@pytest.mark.parametrize("x", [SAMPLE, np.arange(-5.0, 6), np.exp(SAMPLE)])
def test_cabana_continuous_supremum(cls, kurtosis, x):
    assert cls(parameters_for(cls)).execute_statistic(x) == pytest.approx(
        cabana_reference(x, kurtosis), abs=1e-9
    )


def medcouple_reference(x):
    """Enumerate scalar kernels with the median rank convention."""
    median = np.median(x)
    lower, upper = sorted(v for v in x if v <= median), sorted(v for v in x if v >= median)
    ties = sum(v == median for v in x)
    values = []
    for j, b in enumerate(upper):
        for i, a in enumerate(lower):
            if a == b:
                values.append(np.sign(j + i - len(lower) + 1))
            else:
                values.append(((b - median) - (median - a)) / (b - a))
    assert ties <= min(len(lower), len(upper))
    return np.median(values)


def bhs_reference(x):
    x = np.asarray(x)
    median = np.median(x)
    v = np.array(
        [
            medcouple_reference(x),
            -medcouple_reference(x[x < median]),
            medcouple_reference(x[x > median]),
        ]
    ) - [0, 0.198828, 0.198828]
    covariance = np.array(
        [
            [1.24581, 0.322918, -0.322918],
            [0.322918, 2.62068, -0.0123455],
            [-0.322918, -0.0123455, 2.62068],
        ]
    )
    return len(x) * v @ np.linalg.solve(covariance, v)


@pytest.mark.parametrize(
    "x", [SAMPLE[:7], SAMPLE[:10], [-3, -2, -1, 0, 0, 0, 1, 2, 3], [-2, -2, -1, 0, 1, 2, 2]]
)
def test_bhs_exact_kernel_and_strict_halves(x):
    statistic = normal.BHSNormalityGofStatistic(NormalDistributionDescriptor.DEFAULT.parse({}))
    assert statistic.execute_statistic(x) == pytest.approx(bhs_reference(x), rel=1e-12)
    assert statistic.execute_statistic(x) == pytest.approx(
        statistic.execute_statistic(-np.array(x))
    )


def test_bhs_weighted_median_ignores_unused_buffer_tail():
    assert normal.BHSNormalityGofStatistic.whi_med_i([1, 2, 100], [1, 2, 100], 2, [], [], []) == 2


def test_martinez_iglewicz_truncates_outliers():
    x = np.r_[np.arange(-5.0, 6), 100]
    median = np.median(x)
    u = (x - median) / (9 * np.median(abs(x - median)))
    included = abs(u) < 1
    numerator = sum((x[included] - median) ** 2 * (1 - u[included] ** 2) ** 4)
    denominator = sum((1 - u[included] ** 2) * (1 - 5 * u[included] ** 2))
    expected = sum((x - median) ** 2) * denominator**2 / ((len(x) - 1) * len(x) * numerator)
    assert normal.MartinezIglewiczNormalityGofStatistic(
        NormalDistributionDescriptor.DEFAULT.parse({})
    ).execute_statistic(x) == pytest.approx(expected)
    with pytest.raises(ValueError, match="median absolute"):
        normal.MartinezIglewiczNormalityGofStatistic(
            NormalDistributionDescriptor.DEFAULT.parse({})
        ).execute_statistic([0] * 8 + [1, 2])


@pytest.mark.parametrize(
    "cls,reflect",
    [(normal.ZhangQNormalityGofStatistic, False), (normal.ZhangQStarNormalityGofStatistic, True)],
)
def test_zhang_contrasts_match_direct_spacings(cls, reflect):
    x = np.sort(-SAMPLE if reflect else SAMPLE)
    u = stats.norm.ppf((np.arange(1, len(x) + 1) - 0.375) / (len(x) + 0.25))
    q1 = np.mean((x[1:] - x[0]) / (u[1:] - u[0]))
    q2 = np.mean((x[4:] - x[:-4]) / (u[4:] - u[:-4]))
    assert cls(parameters_for(cls)).execute_statistic(x if not reflect else -x) == pytest.approx(
        np.log(q1 / q2)
    )


@pytest.mark.parametrize(
    "cls",
    [
        normal.ZhangWuANormalityGofStatistic,
        normal.ZhangWuCNormalityGofStatistic,
        normal.GlenLeemisBarrNormalityGofStatistic,
    ],
)
def test_probability_tail_rounding_does_not_produce_infinity(cls):
    x = np.r_[np.linspace(-1, 1, 200), 1e6]
    assert np.isfinite(cls(parameters_for(cls)).execute_statistic(x))


def test_desgagne_zero_observation_and_independent_covariance_integration():
    # The fitted score covariance removes the projection on the scale score.
    def raw(z):
        return np.array(
            [
                -0.5 * special.xlogy(z * z, abs(z)),
                -np.log1p(abs(z)),
                -np.log(np.log(math.e + abs(z))),
            ]
        )

    means = integrate.quad_vec(
        lambda z: 2 * stats.norm.pdf(z) * raw(z), 0, 12, epsabs=1e-12, epsrel=1e-12
    )[0]

    def centered(z):
        return raw(z) - means

    covariance = integrate.quad_vec(
        lambda z: 2 * stats.norm.pdf(z) * np.outer(centered(z), centered(z)),
        0,
        12,
        epsabs=1e-12,
        epsrel=1e-12,
    )[0]
    scale = integrate.quad_vec(
        lambda z: 2 * stats.norm.pdf(z) * (z * z - 1) * centered(z),
        0,
        12,
        epsabs=1e-12,
        epsrel=1e-12,
    )[0]
    fitted_covariance = covariance - np.outer(scale, scale) / 2
    x = np.arange(-10.0, 11)
    z = (x - x.mean()) / x.std()
    r = np.mean([centered(v) for v in z], axis=0)
    expected = len(x) * r @ np.linalg.solve(fitted_covariance, r)
    assert normal.DesgagneLafayeNormalityGofStatistic(
        NormalDistributionDescriptor.DEFAULT.parse({})
    ).execute_statistic(x) == pytest.approx(expected, rel=2e-4)


@pytest.mark.parametrize(
    "cls", [c for c in CLASSES if issubclass(c, normal.AbstractGraphNormalityGofStatistic)]
)
def test_graph_summaries_against_exhaustive_small_graph(cls):
    x = np.array([-0.1, 0, 0.03, 0.12, 0.4, 1.0])
    radius = np.ptp(x) / (10 * np.var(x))
    adjacent = abs(x[:, None] - x[None, :]) < radius
    np.fill_diagonal(adjacent, False)
    subsets = [subset for k in range(1, len(x) + 1) for subset in combinations(range(len(x)), k)]
    clique = max(len(g) for g in subsets if all(adjacent[i, j] for i, j in combinations(g, 2)))
    independent = max(
        len(g) for g in subsets if all(not adjacent[i, j] for i, j in combinations(g, 2))
    )
    from scipy.sparse.csgraph import connected_components

    expected = {
        "CLIQUENUMBER": clique,
        "INDEPENDENCENUMBER": independent,
        "EDGESNUMBER": adjacent.sum() / 2,
        "MAXDEGREE": adjacent.sum(axis=1).max(),
        "AVGDEGREE": adjacent.sum(axis=1).mean(),
        "CONNECTEDCOMPONENTS": connected_components(adjacent)[0],
    }
    assert cls(parameters_for(cls)).execute_statistic(x) == expected[cls.short_code()]


def test_clique_at_last_observation_has_no_extra_vertex():
    assert (
        normal.GraphCliqueNumberNormalityGofStatistic(
            NormalDistributionDescriptor.DEFAULT.parse({"var": 1})
        ).execute_statistic([0, 0.01, 0.02])
        == 3.0
    )


@pytest.mark.parametrize(
    "cls,tail",
    [
        (normal.SkewNormalityGofStatistic, AlternativeType.TWO_TAILED),
        (normal.KurtosisNormalityGofStatistic, AlternativeType.TWO_TAILED),
        (normal.DAPNormalityGofStatistic, AlternativeType.RIGHT),
        (normal.SWRGNormalityGofStatistic, AlternativeType.LEFT),
        *[
            (getattr(normal, f"Hosking{i}NormalityGofStatistic"), AlternativeType.RIGHT)
            for i in range(1, 5)
        ],
    ],
)
def test_critical_tails(cls, tail):
    assert cls(parameters_for(cls)).alternative().type() == tail


@pytest.mark.parametrize(
    "statistic",
    [
        normal.KolmogorovSmirnovNormalityGofStatistic(
            NormalDistributionDescriptor.DEFAULT.parse({"mean": 0, "var": 1}),
            alternative_type=AlternativeType.LEFT,
        ),
        normal.RyanJoinerNormalityGofStatistic(
            NormalDistributionDescriptor.DEFAULT.parse({}), weighted=True
        ),
        normal.RyanJoinerNormalityGofStatistic(
            NormalDistributionDescriptor.DEFAULT.parse({}), cte_alpha="1/2"
        ),
    ],
)
def test_unkeyed_options_cannot_use_stored_calibration(statistic, mocker):
    store = mocker.Mock()
    with pytest.raises(ValueError, match="settings"):
        StorageLimitDistributionResolver(store).resolve(statistic, 24)
    store.get.assert_not_called()
    assert np.all(np.isfinite(MonteCarloLimitDistributionResolver(2).resolve(statistic, 24)))


def test_docstring_and_markdown_examples():
    assert doctest.testmod(normal).failed == 0
    folder = Path(__file__).parents[2] / "docs/criteria/goodness-of-fit/normality"
    for page in folder.glob("*.md"):
        for section in page.read_text().split("```python\n")[1:]:
            exec(compile(section.split("```", 1)[0], str(page), "exec"), {})  # noqa: S102


@pytest.mark.parametrize("trim", range(4))
def test_hosking_trimmed_l_moments_by_subsample_enumeration(trim):
    x = np.random.default_rng(123).normal(size=12)
    moments = {}
    for r in (2, 3, 4):
        size = r + 2 * trim
        terms = []
        for group in combinations(x, size):
            ordered = sorted(group)
            terms.append(
                sum((-1) ** k * math.comb(r - 1, k) * ordered[r + trim - k - 1] for k in range(r))
                / r
            )
        moments[r] = np.mean(terms)
    constants = [
        (0.12383, 0.0088038, 0.0049295),
        (0.067077, 0.0081391, 0.0042752),
        (0.044174, 0.008657, 0.0042066),
        (0.03318, 0.0095765, 0.0044609),
    ]
    mu, v3, v4 = constants[trim]
    expected = (moments[3] / moments[2]) ** 2 / v3 + (moments[4] / moments[2] - mu) ** 2 / v4
    cls = getattr(normal, f"Hosking{trim + 1}NormalityGofStatistic")
    assert cls(parameters_for(cls)).execute_statistic(x) == pytest.approx(expected, rel=1e-10)
    if trim:
        with pytest.raises(ValueError, match="L-scale"):
            cls(parameters_for(cls)).execute_statistic([-1] + [0] * 10 + [1])


def test_spiegelhalter_large_n_matches_log_domain_formula():
    x = stats.norm.ppf((np.arange(1, 2001) - 0.5) / 2000)
    n = len(x)
    u = np.ptp(x) / x.std(ddof=1)
    g = sum(abs(x - x.mean())) / (x.std(ddof=1) * np.sqrt(n * (n - 1)))
    a = np.log(2 * n) - special.gammaln(n + 1) / (n - 1) - np.log(u)
    b = -np.log(g)
    expected = np.exp(max(a, b) + np.log1p(np.exp(-(n - 1) * abs(a - b))) / (n - 1))
    assert normal.SpiegelhalterNormalityGofStatistic(
        NormalDistributionDescriptor.DEFAULT.parse({})
    ).execute_statistic(x) == pytest.approx(expected)


def test_coin_rejects_unsupported_normal_score_approximation():
    with pytest.raises(ValueError, match="2000"):
        normal.CoinNormalityGofStatistic(
            NormalDistributionDescriptor.DEFAULT.parse({})
        ).execute_statistic(np.arange(2001))


@pytest.mark.parametrize("kwargs", [{"weighted": 1}, {"cte_alpha": "wrong"}])
def test_ryan_joiner_rejects_invalid_settings(kwargs):
    with pytest.raises((ValueError, TypeError)):
        normal.RyanJoinerNormalityGofStatistic(**kwargs)
