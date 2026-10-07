"""Independent mathematical and interface regressions for Student statistics."""

import inspect
from itertools import pairwise

import numpy as np
import pytest
from scipy import integrate, stats

from pysatl_criterion.distribution.distributions import StudentDistributionDescriptor as Student
from pysatl_criterion.hypothesis_testing.limit_distribution.base import (
    MonteCarloLimitDistributionResolver,
)
from pysatl_criterion.statistics.alternative import AlternativeType
from pysatl_criterion.statistics.goodness_of_fit import student


CLASSES = [
    cls
    for _, cls in inspect.getmembers(student, inspect.isclass)
    if cls.__module__ == student.__name__ and not inspect.isabstract(cls)
]
FIXED = [cls for cls in CLASSES if cls is not student.LillieforsStudentGofStatistic]
SAMPLE = np.array([2.1, -1.3, 0.2, 4.7, -0.8, 0.5, 1.2])


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("bad", [[], 1.0, [[1, 2]], [np.nan], [np.inf], [1j], [True], ["1"]])
def test_sample_validation(cls, bad):
    with pytest.raises(ValueError):
        cls(df=5).execute_statistic(bad)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("bad", [np.nan, np.inf, [5], True, 1j, "5", 0, -1])
def test_df_validation(cls, bad):
    with pytest.raises(ValueError):
        cls(df=bad)


@pytest.mark.parametrize("parameter", ["loc", "scale"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, [1], True, 1j, "1"])
def test_location_scale_validation(parameter, bad):
    with pytest.raises(ValueError):
        student.KuiperStudentGofStatistic(**{parameter: bad})


@pytest.mark.parametrize("cls", CLASSES)
def test_common_interface_no_mutation_and_right_tail(cls):
    x = SAMPLE.copy()
    statistic = cls(df=5)
    result = statistic.execute_statistic(x, unused=True)
    assert isinstance(result, (float, np.float64))
    assert np.isfinite(result)
    np.testing.assert_array_equal(x, SAMPLE)
    assert statistic.alternative().type() is AlternativeType.RIGHT
    assert statistic.supports_hypothesis(statistic.hypothesis())
    reconstructed = cls.from_parameters(statistic.hypothesis().parameter_values)
    assert reconstructed.execute_statistic(x) == pytest.approx(result)


@pytest.mark.parametrize("cls", FIXED)
def test_fixed_hypothesis_affine_equivariance_and_constants(cls):
    base = cls(df=3)
    shifted = cls(df=3, loc=7, scale=2)
    assert shifted.hypothesis().parameters() == {"df": 3, "loc": 7, "scale": 2}
    assert shifted.execute_statistic(2 * SAMPLE + 7) == pytest.approx(
        base.execute_statistic(SAMPLE)
    )
    assert np.isfinite(base.execute_statistic([1]))
    assert np.isfinite(base.execute_statistic([1, 1, 1]))


@pytest.mark.parametrize(
    "direction,scipy_name",
    [
        (AlternativeType.TWO_TAILED, "two-sided"),
        (AlternativeType.RIGHT, "greater"),
        (AlternativeType.LEFT, "less"),
    ],
)
def test_ks_directions_against_scipy(direction, scipy_name):
    reference = stats.kstest(SAMPLE, stats.t(5, loc=1, scale=2).cdf, alternative=scipy_name)
    for option in (direction, direction.value, scipy_name):
        statistic = student.KolmogorovSmirnovStudentGofStatistic(
            df=5, loc=1, scale=2, alternative_type=option
        )
        assert statistic.execute_statistic(SAMPLE) == pytest.approx(reference.statistic)
        assert statistic.alternative().type() is AlternativeType.RIGHT


@pytest.mark.parametrize("bad", ["typo", None, 1, []])
def test_ks_invalid_direction(bad):
    with pytest.raises(ValueError):
        student.KolmogorovSmirnovStudentGofStatistic(alternative_type=bad)


def test_quadratic_edf_statistics_against_integrals():
    u = stats.t.cdf(np.sort(SAMPLE), 5)
    n = len(u)
    boundaries = np.r_[0, u, 1]
    w2 = a2 = mean_error = 0.0
    for j, (left, right) in enumerate(pairwise(boundaries)):
        w2 += n * integrate.quad(lambda t, j=j: (j / n - t) ** 2, left, right)[0]
        a2 += n * integrate.quad(lambda t, j=j: (j / n - t) ** 2 / (t * (1 - t)), left, right)[0]
        mean_error += integrate.quad(lambda t, j=j: j / n - t, left, right)[0]
    assert student.CramerVonMisesStudentGofStatistic(5).execute_statistic(SAMPLE) == pytest.approx(
        w2
    )
    assert student.AndersonDarlingStudentGofStatistic(5).execute_statistic(SAMPLE) == pytest.approx(
        a2
    )
    assert student.WatsonStudentGofStatistic(5).execute_statistic(SAMPLE) == pytest.approx(
        w2 - n * mean_error**2
    )


def test_kuiper_against_two_scipy_distances():
    cdf = stats.t(5).cdf
    expected = sum(stats.kstest(SAMPLE, cdf, alternative=a).statistic for a in ("less", "greater"))
    assert student.KuiperStudentGofStatistic(5).execute_statistic(SAMPLE) == pytest.approx(expected)


def test_zhang_equations_3_2_and_3_3():
    # Known PIT coordinates make the reference independent of the Student CDF.
    u = np.array([0.02, 0.12, 0.3, 0.65, 0.9, 0.99])
    x = stats.t.ppf(u, 4)
    n = len(u)
    zc = sum(np.log((1 / v - 1) / ((n - 0.5) / (i - 0.75) - 1)) ** 2 for i, v in enumerate(u, 1))
    za = -sum(np.log(v) / (n - i + 0.5) + np.log1p(-v) / (i - 0.5) for i, v in enumerate(u, 1))
    assert student.ZhangZcStudentGofStatistic(4).execute_statistic(x) == pytest.approx(zc)
    assert student.ZhangZaStudentGofStatistic(4).execute_statistic(x) == pytest.approx(za)


@pytest.mark.parametrize(
    "cls",
    [
        student.ZhangZaStudentGofStatistic,
        student.ZhangZcStudentGofStatistic,
        student.AndersonDarlingStudentGofStatistic,
    ],
)
def test_log_tails_are_not_clipped_or_computed_by_subtraction(cls):
    x = np.array([-1e7, -1, 0, 1, 1e7])
    assert stats.t.cdf(x[-1], 5) == 1.0
    statistic = cls(5)
    value = statistic.execute_statistic(x)
    assert np.isfinite(value)
    assert value == pytest.approx(statistic.execute_statistic(-x))
    assert value > statistic.execute_statistic([-1e4, -1, 0, 1, 1e4])
    with pytest.raises(ValueError, match="floating-point"):
        statistic.execute_statistic([1e200])


def test_standardization_recovers_overflowing_difference():
    statistic = student.KolmogorovSmirnovStudentGofStatistic(5, loc=-1e308, scale=1e308)
    assert statistic.execute_statistic([1e308]) == pytest.approx(stats.t.cdf(2, 5))
    with pytest.raises(ValueError, match="floating-point"):
        student.KuiperStudentGofStatistic(scale=1e-300).execute_statistic([1e300])


@pytest.mark.parametrize("bad", [True, 1, 0, -1, 2.5, [3], np.inf])
def test_bin_count_validation(bad):
    with pytest.raises(ValueError, match="n_bins"):
        student.ChiSquareStudentGofStatistic(n_bins=bad)


def test_pearson_counts_with_empty_cells_and_boundary_tie():
    x = stats.t.ppf([0.1, 0.2, 0.5, 0.9], 5)
    expected = stats.chisquare([2, 0, 1, 1], [1, 1, 1, 1]).statistic
    assert (
        student.ChiSquareStudentGofStatistic(5, n_bins=np.int64(4)).execute_statistic(x) == expected
    )


@pytest.mark.parametrize("df", [0.5, 1, 5, 30])
def test_lilliefors_quantile_fit_fresh_for_each_call(df):
    statistic = student.LillieforsStudentGofStatistic(df)
    state = vars(statistic).copy()
    for x in (SAMPLE, 7 + 3 * SAMPLE, np.array([-7, -3, -1, 1, 2, 10]), SAMPLE):
        q1, median, q3 = np.quantile(x, [0.25, 0.5, 0.75], method="linear")
        scale = (q3 - q1) / (stats.t.ppf(0.75, df) - stats.t.ppf(0.25, df))
        expected = stats.kstest(x, stats.t(df, loc=median, scale=scale).cdf).statistic
        assert statistic.execute_statistic(x) == pytest.approx(expected)
    assert vars(statistic) == state
    assert statistic.execute_statistic(SAMPLE) == pytest.approx(
        statistic.execute_statistic(7 + 3 * SAMPLE)
    )
    assert statistic.execute_statistic(1e200 * SAMPLE) == pytest.approx(
        statistic.execute_statistic(SAMPLE)
    )


@pytest.mark.parametrize("x", [[1], [1, 1], [0, 0, 0, 0, 1]])
def test_lilliefors_degenerate_quantiles(x):
    with pytest.raises(ValueError):
        student.LillieforsStudentGofStatistic(5).execute_statistic(x)


def test_lilliefors_composite_hypothesis_cannot_use_default_sampler():
    statistic = student.LillieforsStudentGofStatistic.from_parameters(
        Student.DEFAULT.parse({"df": 5})
    )
    assert statistic.hypothesis().parameters() == {"df": 5}
    with pytest.raises(ValueError, match="complete Student parameters"):
        MonteCarloLimitDistributionResolver(2).resolve(statistic, 10)
