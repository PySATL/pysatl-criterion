"""Regression checks for the Beta statistics audit, including independent references."""

import numpy as np
import pytest
from scipy import stats

from pysatl_criterion.statistics.alternative import AlternativeType, RightAlternative
from pysatl_criterion.statistics.goodness_of_fit import beta


STATISTICS = [
    beta.KolmogorovSmirnovBetaGofStatistic,
    beta.AndersonDarlingBetaGofStatistic,
    beta.CrammerVonMisesBetaGofStatistic,
    beta.LillieforsTestBetaGofStatistic,
    beta.WatsonBetaGofStatistic,
    beta.KuiperBetaGofStatistic,
    beta.Chi2PearsonBetaGofStatistic,
    beta.MomentBasedBetaGofStatistic,
    beta.SkewnessKurtosisBetaGofStatistic,
    beta.RatioBetaGofStatistic,
    beta.EntropyBetaGofStatistic,
    beta.ModeBetaGofStatistic,
]


@pytest.mark.parametrize("cls", STATISTICS)
@pytest.mark.parametrize("sample", [[], [[0.2, 0.3]], [np.nan] * 8, [np.inf], [-0.1], [1.1]])
def test_invalid_samples(cls, sample):
    with pytest.raises(ValueError):
        statistic = cls() if cls is beta.LillieforsTestBetaGofStatistic else cls(2, 5)
        statistic.execute_statistic(sample)


@pytest.mark.parametrize(
    "cls", [cls for cls in STATISTICS if cls is not beta.LillieforsTestBetaGofStatistic]
)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, [2, 3]])
@pytest.mark.parametrize("parameter", ["alpha", "beta"])
def test_invalid_parameters(cls, value, parameter):
    with pytest.raises(ValueError):
        cls(**{"alpha": 2, "beta": 5, parameter: value})


@pytest.mark.parametrize(
    "direction,scipy_direction",
    [
        (AlternativeType.TWO_TAILED, "two-sided"),
        (AlternativeType.RIGHT, "greater"),
        (AlternativeType.LEFT, "less"),
    ],
)
def test_ks_direction_and_rejection_tail(direction, scipy_direction):
    sample = [0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57]
    statistic = beta.KolmogorovSmirnovBetaGofStatistic(2, 5, direction)
    assert isinstance(statistic.alternative(), RightAlternative)
    expected = stats.kstest(sample, stats.beta(2, 5).cdf, alternative=scipy_direction).statistic
    assert statistic.execute_statistic(sample) == pytest.approx(expected)


def test_lilliefors_fits_both_shapes_and_exposes_composite_null():
    sample = [0.1, 0.2, 0.3, 0.5]
    statistic = beta.LillieforsTestBetaGofStatistic()
    a, b, _, _ = stats.beta.fit(sample, floc=0, fscale=1)
    expected = stats.kstest(sample, stats.beta(a, b).cdf).statistic
    assert statistic.execute_statistic(sample) == pytest.approx(expected)
    assert statistic.execute_statistic(sample) != pytest.approx(
        beta.KolmogorovSmirnovBetaGofStatistic().execute_statistic(sample)
    )
    assert isinstance(statistic.alternative(), RightAlternative)
    assert statistic.hypothesis().parameters() == {}
    assert vars(statistic) == {}


@pytest.mark.parametrize(
    "cls,min_size",
    [
        (beta.MomentBasedBetaGofStatistic, 2),
        (beta.SkewnessKurtosisBetaGofStatistic, 4),
        (beta.EntropyBetaGofStatistic, 3),
        (beta.ModeBetaGofStatistic, 2),
    ],
)
def test_sample_size(cls, min_size):
    with pytest.raises(ValueError, match="at least"):
        cls(2, 5).execute_statistic(np.linspace(0.1, 0.9, min_size - 1))


@pytest.mark.parametrize("cls", [beta.ModeBetaGofStatistic, beta.SkewnessKurtosisBetaGofStatistic])
def test_constant_sample_has_explicit_error(cls):
    with pytest.raises(ValueError, match="nonconstant"):
        cls(2, 5).execute_statistic([0.5] * 8)


@pytest.mark.parametrize("m", [0, -1, 4, 8, 1.5, True, np.nan])
def test_invalid_entropy_window(m):
    with pytest.raises((ValueError, TypeError), match="m must be an integer"):
        beta.EntropyBetaGofStatistic().execute_statistic(np.linspace(0.1, 0.9, 8), m=m)


def test_zero_entropy_spacings_are_not_skipped():
    assert np.isposinf(beta.EntropyBetaGofStatistic().execute_statistic([0.5] * 8))
    assert np.isposinf(beta.EntropyBetaGofStatistic().execute_statistic([0.2] * 5 + [0.8] * 3, m=1))


@pytest.mark.parametrize("n", [3, 8, 71])
def test_entropy_matches_vasicek(n):
    sample = stats.beta(2, 5).ppf((np.arange(n) + 0.5) / n)
    m = min(int(np.sqrt(n) + 0.5), (n - 1) // 2)
    expected = np.sqrt(n) * abs(
        stats.differential_entropy(sample, method="vasicek", window_length=m)
        - stats.beta(2, 5).entropy()
    )
    assert beta.EntropyBetaGofStatistic(2, 5).execute_statistic(sample) == pytest.approx(expected)


def test_entropy_default_window_removes_uniform_fixed_fraction_bias():
    errors = []
    for n in [100, 1000, 10000]:
        sample = (np.arange(n) + 0.5) / n
        errors.append(beta.EntropyBetaGofStatistic().execute_statistic(sample) / np.sqrt(n))
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 0.01


@pytest.mark.parametrize("a,b", [(1, 1), (2, 5), (5, 2), (0.5, 0.7), (100, 200)])
def test_moment_statistics_against_integrated_covariance(a, b):
    distribution = stats.beta(a, b)
    mean, variance = distribution.stats(moments="mv")
    sd = np.sqrt(variance)
    skewness, excess = distribution.stats(moments="sk")
    kurtosis = excess + 3
    sample = distribution.ppf((np.arange(31) + 0.3) / 31)

    def influence(x, kind):
        z = (x - mean) / sd
        if kind == "MB":
            return np.array([x - mean, (x - mean) ** 2 - variance])
        return np.array(
            [
                z**3 - 3 * z - 1.5 * skewness * z**2 + skewness / 2,
                z**4 - 4 * skewness * z - 2 * kurtosis * z**2 + kurtosis,
            ]
        )

    for kind, statistic, difference in [
        (
            "MB",
            beta.MomentBasedBetaGofStatistic(a, b),
            [sample.mean() - mean, sample.var(ddof=1) - variance],
        ),
        (
            "SK",
            beta.SkewnessKurtosisBetaGofStatistic(a, b),
            [
                stats.skew(sample, bias=False) - skewness,
                stats.kurtosis(sample, bias=False) - excess,
            ],
        ),
    ]:
        covariance = np.array(
            [
                [
                    distribution.expect(
                        lambda x, kind=kind, i=i, j=j: influence(x, kind)[i] * influence(x, kind)[j]
                    )
                    for j in range(2)
                ]
                for i in range(2)
            ]
        )
        expected = len(sample) * np.array(difference) @ np.linalg.solve(covariance, difference)
        assert statistic.execute_statistic(sample) == pytest.approx(expected, rel=1e-7)


@pytest.mark.parametrize("center", [0.0025, 0.5, 0.9975])
def test_mode_can_reach_center_and_near_boundaries(center):
    # A narrow, symmetric, unimodal KDE has its mode at the symmetry center.
    sample = center + np.linspace(-0.0001, 0.0001, 31)
    a, b = 1 + center * 4, 1 + (1 - center) * 4
    assert beta.ModeBetaGofStatistic(a, b).execute_statistic(sample) < 1e-7


def test_ad_boundary_infinity_is_preserved():
    assert np.isposinf(beta.AndersonDarlingBetaGofStatistic(2, 5).execute_statistic([0, 0.5, 1]))


def test_pearson_uses_survival_probabilities_in_right_tail():
    # The last CDF difference rounds to zero, while the survival probability does not.
    sample = stats.beta(2, 50).ppf((np.arange(100) + 0.5) / 100)
    statistic = beta.Chi2PearsonBetaGofStatistic(2, 50).execute_statistic(sample)
    assert np.isfinite(statistic)
    assert statistic >= 0


@pytest.mark.parametrize("sample", [[0.2], [0.5] * 8, [0, 0.3, 0.7], [0.2, 0.7, 1]])
def test_lilliefors_rejects_samples_without_finite_interior_mle(sample):
    with pytest.raises(ValueError):
        beta.LillieforsTestBetaGofStatistic().execute_statistic(sample)


def test_lilliefors_does_not_accept_fixed_shape_parameters():
    with pytest.raises(TypeError):
        beta.LillieforsTestBetaGofStatistic(alpha=2, beta=5)


def test_lilliefors_refits_each_sample_through_unified_method():
    statistic = beta.LillieforsTestBetaGofStatistic()
    rng = np.random.default_rng(42)
    samples = [rng.beta(a, b, size=40) for a, b in [(2, 5), (5, 2), (1, 1)]]
    for sample in samples + samples[:1]:
        a, b, _, _ = stats.beta.fit(sample, floc=0, fscale=1)
        expected = stats.kstest(sample, stats.beta(a, b).cdf).statistic
        assert statistic.execute_statistic(sample) == pytest.approx(expected)
        assert statistic.hypothesis().parameters() == {}
        assert vars(statistic) == {}


@pytest.mark.parametrize("cls", STATISTICS)
def test_all_beta_statistics_support_uniform_scalar_execution(cls):
    from pysatl_criterion.statistics import AbstractGoodnessOfFitStatistic

    statistic: AbstractGoodnessOfFitStatistic = cls()
    sample = np.array([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
    original = sample.copy()
    result = statistic.execute_statistic(sample)
    assert isinstance(result, (float, np.float64))
    assert np.isfinite(result)
    assert result >= 0
    np.testing.assert_array_equal(sample, original)
    assert statistic.execute_statistic(sample) == result


def test_composite_beta_cannot_use_default_uniform_calibration():
    from pysatl_criterion.utils.generator import get_hypothesis_generator

    with pytest.raises(ValueError, match="simulation shapes externally"):
        get_hypothesis_generator(beta.LillieforsTestBetaGofStatistic())


@pytest.mark.parametrize("a,b", [(1, 1), (2, 5), (5, 2), (0.5, 0.5)])
def test_fixed_edf_statistics_match_uniform_transform(a, b):
    u = (np.arange(41) + 0.3) / 41
    sample = stats.beta(a, b).ppf(u)
    for cls in [
        beta.KolmogorovSmirnovBetaGofStatistic,
        beta.AndersonDarlingBetaGofStatistic,
        beta.CrammerVonMisesBetaGofStatistic,
        beta.WatsonBetaGofStatistic,
        beta.KuiperBetaGofStatistic,
    ]:
        assert cls(a, b).execute_statistic(sample) == pytest.approx(
            cls().execute_statistic(u), abs=2e-13
        )


def test_beta_docstring_examples():
    import doctest

    result = doctest.testmod(beta, raise_on_error=True)
    assert result.attempted >= 36


@pytest.mark.parametrize("cls", STATISTICS)
def test_each_statistic_documents_formula_hypothesis_and_source_status(cls):
    import inspect

    doc = inspect.getdoc(cls)
    assert "Notes\n-----" in doc
    assert "Examples\n--------" in doc
    assert "Returns\n-------" in inspect.getdoc(cls.execute_statistic)
    if cls in [
        beta.MomentBasedBetaGofStatistic,
        beta.SkewnessKurtosisBetaGofStatistic,
        beta.RatioBetaGofStatistic,
        beta.EntropyBetaGofStatistic,
        beta.ModeBetaGofStatistic,
    ]:
        assert "No publication" in doc
        assert "https://" not in doc
    else:
        assert "References\n----------" in doc
        assert "https://doi.org/" in doc


@pytest.mark.parametrize("cls", STATISTICS)
@pytest.mark.parametrize(
    "sample", [np.array([0.1 + 0.2j, 0.3 + 0.4j]), np.ma.array([0.2, 0.4], mask=[0, 1])]
)
def test_no_silent_loss_of_sample_information(cls, sample):
    with pytest.raises(ValueError, match="real|masked"):
        cls().execute_statistic(sample)


@pytest.mark.parametrize("parameter", ["alpha", "beta"])
def test_complex_shape_is_rejected(parameter):
    with pytest.raises(ValueError):
        beta.KolmogorovSmirnovBetaGofStatistic(**{parameter: 2 + 1j})


@pytest.mark.parametrize("direction", [None, "left", 42])
def test_invalid_ks_direction_is_not_silently_two_sided(direction):
    with pytest.raises(TypeError, match="AlternativeType"):
        beta.KolmogorovSmirnovBetaGofStatistic(alternative_type=direction)


@pytest.mark.parametrize("power", [np.nan, np.inf, [1, 2], 1 + 2j])
def test_power_parameter_validated_at_construction(power):
    with pytest.raises(ValueError, match="finite real scalar"):
        beta.Chi2PearsonBetaGofStatistic(lambda_=power)


def test_skewness_kurtosis_near_constant_sample_matches_affine_equivalent():
    # Exactly representable differences, hence an exact affine equivalent of integers.
    sample = 0.5 + np.arange(8) * np.spacing(0.5)
    g = stats.skew(np.arange(8), bias=False)
    k = stats.kurtosis(np.arange(8), bias=False)
    # Independent Uniform(0,1) influence-function variances from the audit.
    expected = 8 * (g * g / (72 / 35) + (k + 1.2) ** 2 / (1152 / 875))
    assert beta.SkewnessKurtosisBetaGofStatistic().execute_statistic(sample) == pytest.approx(
        expected
    )


def test_ratio_at_subnormal_scale_matches_exact_ratio():
    sample = np.arange(1, 5) * np.nextafter(0.0, 1.0)
    ratio = 24**0.25 / 2.5
    expected = 2 * abs(ratio - 2 / np.e)
    assert beta.RatioBetaGofStatistic().execute_statistic(sample) == pytest.approx(
        expected, abs=1e-12
    )


def test_mode_at_tiny_scale_does_not_underflow_kde_variance():
    # Symmetric unimodal KDE mode is at the midpoint, even when variance underflows.
    sample = np.linspace(1, 2, 9) * 1e-200
    expected = np.sqrt(9) * abs(1.5e-200 - 0.5)
    assert beta.ModeBetaGofStatistic().execute_statistic(sample) == pytest.approx(expected)
