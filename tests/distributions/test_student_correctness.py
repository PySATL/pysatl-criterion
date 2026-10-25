"""Independent mathematical and interface regressions for Student statistics."""

import inspect
from itertools import pairwise

import numpy as np
import pytest
from scipy import integrate, stats

from pysatl_criterion.distribution import StudentDistributionDescriptor
from pysatl_criterion.statistics.alternative import AlternativeType
from pysatl_criterion.statistics.goodness_of_fit.binning import EqualProbabilityBins
from pysatl_criterion.statistics.goodness_of_fit.continuous import student
from pysatl_criterion.statistics.goodness_of_fit.continuous.student import student_statistics
from tests.parameter_cases import parameters_for


CLASSES = [
    cls
    for _, cls in inspect.getmembers(student, inspect.isclass)
    if cls.__module__ == student.__name__ and not inspect.isabstract(cls)
]
FIXED = [
    cls
    for cls in CLASSES
    if cls.__name__ in student_statistics._EDF_STATISTICS
    or cls.__name__ in student_statistics._FREQUENCY_STATISTICS
]
SAMPLE = np.array([2.1, -1.3, 0.2, 4.7, -0.8, 0.5, 1.2])


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("bad", [[], 1.0, [[1, 2]], [np.nan], [np.inf], [1j]])
def test_sample_validation(cls, bad):
    with pytest.raises(ValueError):
        cls(parameters_for(cls, df=5)).execute_statistic(bad)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("bad", [np.nan, np.inf, [5], True, 1j, "5", 0, -1])
def test_df_validation(cls, bad):
    with pytest.raises(ValueError):
        cls(parameters_for(cls, df=bad))


@pytest.mark.parametrize("parameter", ["loc", "scale"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, [1], True, 1j, "1"])
def test_location_scale_validation(parameter, bad):
    with pytest.raises(ValueError):
        student.KuiperStudentGofStatistic(
            parameters_for(student.KuiperStudentGofStatistic, **{parameter: bad})
        )


@pytest.mark.parametrize("cls", CLASSES)
def test_common_interface_no_mutation_and_right_tail(cls):
    x = SAMPLE.copy()
    statistic = cls(parameters_for(cls, df=5))
    result = statistic.execute_statistic(x, unused=True)
    assert isinstance(result, (float, np.float64))
    assert np.isfinite(result)
    np.testing.assert_array_equal(x, SAMPLE)
    assert statistic.alternative().type() is AlternativeType.RIGHT
    assert statistic.supports_hypothesis(statistic.hypothesis())
    reconstructed = cls(statistic.hypothesis().parameter_values)
    assert reconstructed.execute_statistic(x) == pytest.approx(result)


@pytest.mark.parametrize("cls", FIXED)
def test_fixed_hypothesis_affine_equivariance_and_constants(cls):
    base = cls(parameters_for(cls, df=3))
    shifted = cls(parameters_for(cls, df=3, loc=7, scale=2))
    assert shifted.hypothesis().parameters() == {"df": 3, "loc": 7, "scale": 2}
    assert shifted.execute_statistic(2 * SAMPLE + 7) == pytest.approx(
        base.execute_statistic(SAMPLE)
    )
    for constant in ([1], [1, 1, 1]):
        value = base.execute_statistic(constant)
        if cls in (
            student.ModifiedLogLikelihoodStudentGofStatistic,
            student.NeymanChi2StudentGofStatistic,
        ):
            assert value == np.inf  # Empty observed cells have infinite penalties.
        else:
            assert np.isfinite(value)


@pytest.mark.parametrize(
    "cls,scipy_name",
    [
        (student.KolmogorovSmirnovStudentGofStatistic, "two-sided"),
        (student.KolmogorovSmirnovDPlusStudentGofStatistic, "greater"),
        (student.KolmogorovSmirnovDMinusStudentGofStatistic, "less"),
    ],
)
def test_ks_directions_against_scipy(cls, scipy_name):
    reference = stats.kstest(SAMPLE, stats.t(5, loc=1, scale=2).cdf, alternative=scipy_name)
    statistic = cls(StudentDistributionDescriptor.DEFAULT.parse({"df": 5, "loc": 1, "scale": 2}))
    assert statistic.execute_statistic(SAMPLE) == pytest.approx(reference.statistic)
    assert statistic.alternative().type() is AlternativeType.RIGHT


def test_quadratic_edf_statistics_against_integrals():
    u = stats.t.cdf(np.sort(SAMPLE), 5)
    n = len(u)
    boundaries = np.r_[0, u, 1]
    w2 = a2 = mean_error = 0.0
    for j, (left, right) in enumerate(pairwise(boundaries)):
        w2 += n * integrate.quad(lambda t, j=j: (j / n - t) ** 2, left, right)[0]
        a2 += n * integrate.quad(lambda t, j=j: (j / n - t) ** 2 / (t * (1 - t)), left, right)[0]
        mean_error += integrate.quad(lambda t, j=j: j / n - t, left, right)[0]
    assert student.CramerVonMisesStudentGofStatistic(
        StudentDistributionDescriptor.DEFAULT.parse({"df": 5, "loc": 0, "scale": 1})
    ).execute_statistic(SAMPLE) == pytest.approx(w2)
    assert student.AndersonDarlingStudentGofStatistic(
        StudentDistributionDescriptor.DEFAULT.parse({"df": 5, "loc": 0, "scale": 1})
    ).execute_statistic(SAMPLE) == pytest.approx(a2)
    assert student.WatsonStudentGofStatistic(
        StudentDistributionDescriptor.DEFAULT.parse({"df": 5, "loc": 0, "scale": 1})
    ).execute_statistic(SAMPLE) == pytest.approx(w2 - n * mean_error**2)


def test_kuiper_against_two_scipy_distances():
    cdf = stats.t(5).cdf
    expected = sum(stats.kstest(SAMPLE, cdf, alternative=a).statistic for a in ("less", "greater"))
    assert student.KuiperStudentGofStatistic(
        StudentDistributionDescriptor.DEFAULT.parse({"df": 5, "loc": 0, "scale": 1})
    ).execute_statistic(SAMPLE) == pytest.approx(expected)


def test_zhang_equations_3_2_and_3_3():
    # Known PIT coordinates make the reference independent of the Student CDF.
    u = np.array([0.02, 0.12, 0.3, 0.65, 0.9, 0.99])
    x = stats.t.ppf(u, 4)
    n = len(u)
    zc = sum(np.log((1 / v - 1) / ((n - 0.5) / (i - 0.75) - 1)) ** 2 for i, v in enumerate(u, 1))
    za = -sum(np.log(v) / (n - i + 0.5) + np.log1p(-v) / (i - 0.5) for i, v in enumerate(u, 1))
    assert student.ZhangZcStudentGofStatistic(
        StudentDistributionDescriptor.DEFAULT.parse({"df": 4, "loc": 0, "scale": 1})
    ).execute_statistic(x) == pytest.approx(zc)
    assert student.ZhangZaStudentGofStatistic(
        StudentDistributionDescriptor.DEFAULT.parse({"df": 4, "loc": 0, "scale": 1})
    ).execute_statistic(x) == pytest.approx(za)


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
    statistic = cls(parameters_for(cls, df=5))
    value = statistic.execute_statistic(x)
    assert np.isfinite(value)
    assert value == pytest.approx(statistic.execute_statistic(-x))
    assert value > statistic.execute_statistic([-1e4, -1, 0, 1, 1e4])
    with pytest.raises(ValueError, match="floating-point"):
        statistic.execute_statistic([1e200])


def test_standardization_recovers_overflowing_difference():
    statistic = student.KolmogorovSmirnovStudentGofStatistic(
        StudentDistributionDescriptor.DEFAULT.parse({"df": 5, "loc": -1e308, "scale": 1e308})
    )
    assert statistic.execute_statistic([1e308]) == pytest.approx(stats.t.cdf(2, 5))
    with pytest.raises(ValueError, match="floating-point"):
        student.KuiperStudentGofStatistic(
            StudentDistributionDescriptor.DEFAULT.parse({"df": 1, "loc": 0, "scale": 1e-300})
        ).execute_statistic([1e300])


def test_pearson_counts_with_empty_cells_and_boundary_tie():
    x = stats.t.ppf([0.1, 0.2, 0.5, 0.9], 5)
    expected = stats.chisquare([2, 0, 1, 1], [1, 1, 1, 1]).statistic
    assert (
        student.ChiSquareStudentGofStatistic(
            StudentDistributionDescriptor.DEFAULT.parse({"df": 5, "loc": 0, "scale": 1}),
            binning=EqualProbabilityBins(count=np.int64(4)),
        ).execute_statistic(x)
        == expected
    )


def test_generated_catalog_and_hypothesis_ownership():
    from pysatl_criterion.statistics.goodness_of_fit.continuous import edf, frequency

    assert "supported_hypotheses" not in vars(student.AbstractStudentGofStatistic)
    for source, catalog, base in (
        (edf, student_statistics._EDF_STATISTICS, edf.AbstractEDFStatistic),
        (frequency, student_statistics._FREQUENCY_STATISTICS, frequency.AbstractFrequencyStatistic),
    ):
        algorithms = {
            cls
            for cls in vars(source).values()
            if inspect.isclass(cls) and issubclass(cls, base) and cls is not base
        }
        assert {spec.algorithm for spec in catalog.values()} == algorithms
        for name in catalog:
            cls = getattr(student, name)
            assert name in student.__all__
            assert "supported_hypotheses" in vars(cls)
            with pytest.raises(ValueError, match="Unsupported hypothesis"):
                cls(StudentDistributionDescriptor.DEFAULT.parse({"df": 5}))
    assert len({cls.code() for cls in CLASSES}) == len(CLASSES)
