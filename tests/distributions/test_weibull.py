"""Independent numerical references and regressions for the Weibull review."""

import doctest
import inspect
from itertools import pairwise

import numpy as np
import pytest
from scipy import integrate, optimize, stats
from scipy.special import gamma

from pysatl_criterion.statistics import AbstractGoodnessOfFitStatistic
from pysatl_criterion.statistics.alternative import (
    AlternativeType,
    LeftAlternative,
    RightAlternative,
    TwoSidedAlternative,
)
from pysatl_criterion.statistics.goodness_of_fit import weibull as w
from pysatl_criterion.utils.generator import get_hypothesis_generator


SAMPLE = np.array(
    [
        0.92559015,
        0.9993195,
        1.15193844,
        0.84272073,
        0.97535299,
        0.83745092,
        0.92161732,
        1.02751619,
        0.90079826,
        0.79149641,
    ]
)
CLASSES = [
    cls
    for cls in vars(w).values()
    if inspect.isclass(cls)
    and cls.__module__ == w.__name__
    and not inspect.isabstract(cls)
    and issubclass(cls, AbstractGoodnessOfFitStatistic)
]
FIXED = [
    w.KolmogorovSmirnovWeibullGofStatistic,
    w.CrammerVonMisesWeibullGofStatistic,
    w.WatsonWeibullGofStatistic,
    w.MinToshiyukiWeibullGofStatistic,
    w.Chi2PearsonWeibullGofStatistic,
]
COMPOSITE = [cls for cls in CLASSES if cls not in FIXED]
RECORD = w.MahdiDoostparastWeibullGofStatistic


def fitted_logs(sample):
    """SciPy's Gumbel MLE, independent of the module's profile score solver."""
    y = np.sort(np.log(sample))
    location, scale = stats.gumbel_l.fit(y)
    return (y - location) / scale


def ks_reference(u):
    return stats.kstest(u, "uniform").statistic


@pytest.mark.parametrize("cls", CLASSES)
def test_interface_and_statelessness(cls):
    statistic: AbstractGoodnessOfFitStatistic = cls()
    before = SAMPLE.copy()
    result = statistic.execute_statistic(SAMPLE, common_keyword=True)
    assert isinstance(result, (float, np.float64))
    assert np.isfinite(result)
    statistic.execute_statistic(SAMPLE * 3)
    assert statistic.execute_statistic(SAMPLE) == pytest.approx(result)
    np.testing.assert_array_equal(SAMPLE, before)
    assert statistic.code().endswith("_WEIBULL_GOODNESS_OF_FIT")
    if cls is not RECORD:
        assert statistic.execute_statistic(SAMPLE[::-1]) == pytest.approx(result, abs=1e-12)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("sample", [[], 1, [[1, 2]], [1, np.nan], [1, np.inf], [-1, 2], [1j, 2]])
def test_invalid_samples(cls, sample):
    with pytest.raises(ValueError):
        cls().execute_statistic(sample)


@pytest.mark.parametrize("cls", COMPOSITE)
@pytest.mark.parametrize("sample", [[1], [1, 1, 1], [0, 1, 2]])
def test_fitted_degenerate_samples(cls, sample):
    with pytest.raises(ValueError):
        cls().execute_statistic(sample)


@pytest.mark.parametrize("cls", FIXED)
@pytest.mark.parametrize("value", [True, [1], 0, -1, np.inf, np.nan, 1j, "1"])
def test_fixed_parameter_validation(cls, value):
    for parameter in ("a", "k"):
        with pytest.raises(ValueError):
            cls(**{parameter: value})


@pytest.mark.parametrize("cls", COMPOSITE)
def test_composite_hypothesis_and_log_affine_invariance(cls):
    statistic = cls()
    assert statistic.hypothesis().parameters() == {"a": 1, "loc": 0}
    with pytest.raises(TypeError):
        cls(a=2, k=3)
    original = statistic.execute_statistic(SAMPLE)
    for power, scale in [(0.02, 1e100), (2, 1e-100), (10, 3)]:
        assert statistic.execute_statistic(scale * SAMPLE**power) == pytest.approx(
            original, rel=2e-7, abs=1e-9
        )


@pytest.mark.parametrize("a,k", [(1, 1), (2, 0.7), (0.3, 3)])
def test_fixed_edf_references(a, k):
    sample = np.r_[0, SAMPLE, 8]
    u = np.sort(stats.exponweib.cdf(sample, a, k))
    for alternative, scipy_alternative in [
        (AlternativeType.TWO_TAILED, "two-sided"),
        (AlternativeType.LEFT, "less"),
        (AlternativeType.RIGHT, "greater"),
    ]:
        statistic = w.KolmogorovSmirnovWeibullGofStatistic(alternative, a=a, k=k)
        expected = stats.kstest(
            sample, stats.exponweib(a, k).cdf, alternative=scipy_alternative
        ).statistic
        assert statistic.execute_statistic(sample) == pytest.approx(expected)
        assert isinstance(statistic.alternative(), RightAlternative)
        assert statistic.hypothesis().parameters() == {"a": a, "k": k}
    cvm = stats.cramervonmises(sample, stats.exponweib(a, k).cdf).statistic
    assert w.CrammerVonMisesWeibullGofStatistic(a, k).execute_statistic(sample) == pytest.approx(
        cvm
    )
    assert w.WatsonWeibullGofStatistic(a, k).execute_statistic(sample) == pytest.approx(
        cvm - len(u) * (u.mean() - 0.5) ** 2
    )


@pytest.mark.parametrize("sample", [SAMPLE, [0.01, 0.2, 0.3, 3, 5, 10], SAMPLE**0.005])
def test_mle_and_fitted_edf(sample):
    sample = np.asarray(sample)
    z = fitted_logs(sample)
    mle = w.WPPWeibullGofStatistic.MLEst(sample)
    loc, scale = stats.gumbel_l.fit(np.log(sample))
    assert mle["beta"] == pytest.approx(1 / scale, rel=1e-7)
    assert mle["eta"] == pytest.approx(np.exp(loc), rel=1e-7)
    np.testing.assert_allclose(mle["y"], z, rtol=1e-7, atol=1e-7)
    assert np.all(np.diff(mle["y"]) >= 0)
    # Both likelihood equations, including the previously wrong score sign.
    assert np.mean(np.exp(mle["y"])) == pytest.approx(1)
    log_x = np.log(sample)
    power = (sample / mle["eta"]) ** mle["beta"]
    assert np.dot(power, log_x) / power.sum() - log_x.mean() == pytest.approx(1 / mle["beta"])
    u = stats.gumbel_l.cdf(z)
    assert w.LillieforsWeibullGofStatistic().execute_statistic(sample) == pytest.approx(
        ks_reference(u)
    )
    expected_ad = stats.anderson(np.log(sample), dist="gumbel_l").statistic
    assert w.AndersonDarlingWeibullGofStatistic().execute_statistic(sample) == pytest.approx(
        expected_ad
    )
    empirical = (np.arange(1, len(sample) + 1) - 0.5) / len(sample)
    expected_spp = np.max(
        np.abs(2 / np.pi * (np.arcsin(np.sqrt(empirical)) - np.arcsin(np.sqrt(u))))
    )
    assert w.SPPWeibullGofStatistic().execute_statistic(sample) == pytest.approx(expected_spp)


def test_refitting_changes_cdf_not_just_state():
    statistic = w.LillieforsWeibullGofStatistic()
    assert statistic.execute_statistic(SAMPLE) != pytest.approx(
        w.KolmogorovSmirnovWeibullGofStatistic(a=1, k=1).execute_statistic(SAMPLE)
    )
    for sample in [SAMPLE, np.array([0.01, 0.2, 0.4, 0.8, 10])]:
        assert statistic.execute_statistic(sample) == pytest.approx(
            ks_reference(stats.gumbel_l.cdf(fitted_logs(sample)))
        )


@pytest.mark.parametrize(
    "cls", [w.MinToshiyukiWeibullGofStatistic, w.LiaoShimokawaWeibullGofStatistic]
)
def test_weighted_edf(cls):
    u = (
        stats.gumbel_l.cdf(fitted_logs(SAMPLE))
        if cls not in FIXED
        else stats.exponweib.cdf(np.sort(SAMPLE), 1, 1)
    )
    n = len(u)
    expected = sum(
        max((i + 1) / n - x, x - i / n) / np.sqrt(x * (1 - x)) for i, x in enumerate(u)
    ) / np.sqrt(n)
    assert cls().execute_statistic(SAMPLE) == pytest.approx(expected)


def test_boundary_penalties_and_positive_ad_tail():
    assert np.isinf(w.MinToshiyukiWeibullGofStatistic().execute_statistic([0, 1]))
    assert w.MinToshiyukiWeibullGofStatistic().execute_statistic([1000]) == pytest.approx(
        np.exp(500)
    )
    assert np.isfinite(w.AndersonDarlingWeibullGofStatistic().execute_statistic([1e-300, 1, 1e300]))
    assert np.isinf(w.LOSWeibullGofStatistic().execute_statistic([1, 1, 2]))


def test_probability_plot_correlations():
    y = np.log(np.sort(SAMPLE))
    n = len(y)
    for cls, positions, transform in [
        (w.RSBWeibullGofStatistic, np.arange(1, n + 1) / (n + 1), lambda r: n * (1 - r)),
        (w.REJGWeibullGofStatistic, (np.arange(1, n + 1) - 0.3175) / (n + 0.365), lambda r: r),
    ]:
        r2 = stats.pearsonr(y, stats.gumbel_l.ppf(positions)).statistic ** 2
        assert cls().execute_statistic(SAMPLE) == pytest.approx(transform(r2))


def test_spacing_functionals_by_quadrature():
    # Expected gaps retain the explicit Taylor approximation; the AD functional
    # is verified independently as a weighted EDF integral, not its closed sum.
    y = np.log(np.sort(SAMPLE))
    means = w.NormalizeSpaceWeibullGofStatistic.GoFNS(1, len(y), len(y))
    g = np.diff(y) / np.diff(means)
    z = np.cumsum(g)[:-1] / g.sum()
    boundaries = np.r_[0, z, 1]
    expected = sum(
        integrate.quad(lambda u, j=j: len(z) * (j / len(z) - u) ** 2 / (u * (1 - u)), lo, hi)[0]
        for j, (lo, hi) in enumerate(pairwise(boundaries))
    )
    assert w.LOSWeibullGofStatistic().execute_statistic(SAMPLE) == pytest.approx(expected)
    n = len(y)
    assert w.TikuSinghWeibullGofStatistic().execute_statistic(SAMPLE) == pytest.approx(
        2 * sum((n - 1 - i) * g[i - 1] for i in range(1, n - 1)) / ((n - 2) * sum(g))
    )
    assert w.MSFWeibullGofStatistic().execute_statistic(SAMPLE) == pytest.approx(
        sum(g[n // 2 :]) / sum(g)
    )


def test_moment_normalization_by_integration():
    distribution = stats.gumbel_r
    mean, variance, skew, excess = distribution.stats(moments="mvsk")
    kurt = excess + 3

    def expect(function):
        return integrate.quad(
            lambda x: function((x - mean) / np.sqrt(variance)) * distribution.pdf(x),
            -10,
            90,
            epsabs=1e-9,
        )[0]

    def p3(z):
        return z**3 - 3 * z - 1.5 * skew * z**2 + skew / 2

    def p4(z):
        return z**4 - 4 * skew * z - 2 * kurt * z**2 + kurt

    v33, v34, v44 = (
        expect(lambda z: p3(z) ** 2),
        expect(lambda z: p3(z) * p4(z)),
        expect(lambda z: p4(z) ** 2),
    )
    y = -np.log(SAMPLE)
    g = stats.skew(y, bias=True)
    b = stats.kurtosis(y, fisher=False, bias=True)
    assert w.ST1WeibullGofStatistic().execute_statistic(SAMPLE) == pytest.approx(
        len(y) * (g - skew) ** 2 / v33
    )
    assert w.ST2WeibullGofStatistic().execute_statistic(SAMPLE) == pytest.approx(
        len(y) * ((b - kurt) - v34 / v33 * (g - skew)) ** 2 / (v44 - v34**2 / v33)
    )


@pytest.mark.parametrize(
    "cls,kind",
    [(w.LaplaceTransform2WeibullGofStatistic, 2), (w.LaplaceTransform3WeibullGofStatistic, 3)],
)
@pytest.mark.parametrize("m", [7, 100])
def test_laplace_reference_grid_and_fit(cls, kind, m):
    z = fitted_logs(SAMPLE)
    grid = (
        np.arange(-m, 0) / m
        if kind == 2
        else np.arange(np.ceil(-2.5 * m), np.floor(0.49 * m) + 1) / m
    )
    expected = len(z) * sum(
        np.exp(-5 * t - np.exp(-5 * t))
        * (sum(np.exp(-t * y) for y in z) / len(z) - gamma(1 - t)) ** 2
        for t in grid
    )
    assert cls().execute_statistic(SAMPLE, m=m) == pytest.approx(expected, abs=1e-10)


def test_cq_reference():
    z = fitted_logs(SAMPLE)
    v = [np.sqrt(len(z)) * (np.mean(np.exp(-s * z)) - gamma(1 - s)) for s in [-0.1, 0.02]]
    expected = (0.53 * v[0] ** 2 - 2 * 0.91 * v[0] * v[1] + 1.59 * v[1] ** 2) / (
        1.59 * 0.53 - 0.91**2
    )
    assert w.CabanaQuirozWeibullGofStatistic().execute_statistic(SAMPLE) == pytest.approx(expected)


def test_kl_entropy_reference_and_ties():
    z = fitted_logs(SAMPLE)
    m = 2
    extended = np.pad(z, m, mode="edge")
    entropy = sum(
        np.log(len(z) / (2 * m) * (extended[i + 2 * m] - extended[i])) for i in range(len(z))
    ) / len(z)
    expected = -entropy - np.mean(stats.gumbel_l.logpdf(z))
    assert w.KullbackLeiblerWeibullGofStatistic().execute_statistic(SAMPLE, m=m) == pytest.approx(
        expected
    )
    assert np.isinf(w.KullbackLeiblerWeibullGofStatistic().execute_statistic([1, 1, 1, 2, 3], m=1))


@pytest.mark.parametrize("m", [0, -1, True, 1.5, np.inf])
def test_invalid_settings(m):
    with pytest.raises(ValueError):
        w.KullbackLeiblerWeibullGofStatistic().execute_statistic(SAMPLE, m=m)
    with pytest.raises(ValueError):
        w.LaplaceTransform2WeibullGofStatistic().execute_statistic(SAMPLE, m=m)


def test_record_likelihood_and_integral():
    records = np.array([4.0, 2.0, 1.0, 0.5])
    counts = np.array([2, 3, 1, 2])
    sequence = [4, 5, 2, 3, 2.5, 1, 0.5, 0.8]

    def likelihood(theta):
        shape, scale = np.exp(theta)
        dist = stats.weibull_min(shape, scale=scale)
        return -sum(dist.logpdf(records) + (counts - 1) * dist.logsf(records))

    fit = optimize.minimize(
        likelihood, [0.0, 0.0], method="Nelder-Mead", options={"xatol": 1e-11, "fatol": 1e-11}
    )
    assert fit.success
    shape, scale = np.exp(fit.x)
    u = stats.weibull_min.cdf(records[::-1], shape, scale=scale)
    survival = [1.0]
    ordered_counts = counts[::-1]
    for i in range(len(records)):
        survival.append(survival[-1] * (1 - 1 / sum(ordered_counts[i:])))
    boundaries = np.r_[0.0, u, 1.0]
    expected = counts.sum() * sum(
        integrate.quad(lambda t, s=s: (s - (1 - t)) ** 2 / t, lo, hi)[0]
        for s, lo, hi in zip(survival, boundaries[:-1], boundaries[1:], strict=True)
    )
    statistic = RECORD()
    assert statistic.execute_statistic(sequence) == pytest.approx(expected, rel=1e-7)
    assert statistic.execute_statistic(records, record_counts=counts) == pytest.approx(
        expected, rel=1e-7
    )
    with pytest.raises(ValueError, match="two lower records"):
        statistic.execute_statistic([1, 2, 3])
    with pytest.raises(ValueError, match="counts"):
        statistic.execute_statistic(records, record_counts=[1.0, 2.0, 3.0, 4.0])


def test_critical_tails_and_calibration():
    for cls in CLASSES:
        expected = (
            LeftAlternative
            if cls in [w.SbWeibullGofStatistic, w.SBWeibullGofStatistic, w.REJGWeibullGofStatistic]
            else TwoSidedAlternative
            if cls in [w.TikuSinghWeibullGofStatistic, w.OKWeibullGofStatistic]
            else RightAlternative
        )
        assert isinstance(cls().alternative(), expected)
        with pytest.raises(ValueError, match="regenerate"):
            cls()._validate_storage_calibration()
    for cls in COMPOSITE:
        if cls is RECORD:
            with pytest.raises(ValueError, match="Record calibration"):
                get_hypothesis_generator(cls())
        else:
            generator = get_hypothesis_generator(cls())
            assert generator.a == 1 and generator.k == 1
    generator = get_hypothesis_generator(w.KolmogorovSmirnovWeibullGofStatistic(a=3, k=2))
    assert generator.a == 3 and generator.k == 2


def test_documentation_examples_and_class_count():
    assert len(CLASSES) == 24
    assert doctest.testmod(w).failed == 0


def test_monte_carlo_refits_every_replicate(mocker):
    from pysatl_criterion.hypothesis_testing.limit_distribution.base import (
        MonteCarloLimitDistributionResolver,
    )

    samples = [SAMPLE, np.linspace(0.1, 5, len(SAMPLE))]
    sampler = mocker.patch(
        "pysatl_criterion.generator.generators.generate_weibull", side_effect=samples
    )
    result = MonteCarloLimitDistributionResolver(2).resolve(
        w.LillieforsWeibullGofStatistic(), sample_size=len(SAMPLE)
    )
    expected = [ks_reference(stats.gumbel_l.cdf(fitted_logs(sample))) for sample in samples]
    np.testing.assert_allclose(result, expected)
    assert sampler.call_count == 2
    assert all(call.kwargs["a"] == call.kwargs["k"] == 1 for call in sampler.call_args_list)


def test_transform_extremes_preserve_infinity_without_nan():
    # Fitted extreme negative logs make the unweighted positive-t LT overflow;
    # zero weights must not turn the final result into NaN.
    sample = np.r_[1e-300, np.ones(2000)]
    assert np.isinf(w.LaplaceTransform3WeibullGofStatistic().execute_statistic(sample))
    for a in [-1e308, 1e308]:
        value = w.LaplaceTransform3WeibullGofStatistic().execute_statistic(sample, a=a)
        assert np.isfinite(value)
    assert np.isfinite(w.LaplaceTransform2WeibullGofStatistic().execute_statistic(sample))
    assert np.isfinite(w.AndersonDarlingWeibullGofStatistic().execute_statistic(sample))


def test_unsupported_weibull_calibration_is_not_filled_with_defaults():
    from pysatl_criterion.statistics.hypothesis import GoodnessOfFitHypothesis

    class Incomplete(w.KolmogorovSmirnovWeibullGofStatistic):
        def hypothesis(self):
            return GoodnessOfFitHypothesis({"k": 2})

    with pytest.raises(ValueError, match="external calibration"):
        get_hypothesis_generator(Incomplete())
