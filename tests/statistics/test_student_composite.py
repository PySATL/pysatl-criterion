"""Independent formulas and calibration contracts for composite Student tests."""

import pickle

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gamma
from scipy.stats import t

from pysatl_criterion.distribution import StudentDistributionDescriptor as Student
from pysatl_criterion.statistics.alternative import AlternativeType
from pysatl_criterion.statistics.goodness_of_fit.continuous import student
from pysatl_criterion.statistics.goodness_of_fit.continuous.student import student_statistics
from pysatl_criterion.utils.generator import get_hypothesis_generator


CLASSES = [student.MeintanisStudentGofStatistic, student.CadirciRenyiStudentGofStatistic]
SAMPLE = np.array([-3.7, -1.1, -0.6, -0.2, 0.3, 0.8, 1.9, 4.2, 7.1])


def make(cls, df=5, **kwargs):
    return cls(Student.DEFAULT.parse({"df": df}), **kwargs)


@pytest.mark.parametrize("weight_df", [None, 1])
def test_ecf_against_closed_form_fourier_inversion(weight_df):
    # For nu=3, phi(t)=(1+|t|)exp(-|t|) in the paper's reduced scale.
    # Products with the Cauchy or nu=3 weight have elementary transforms.
    y = (SAMPLE - SAMPLE.mean()) / SAMPLE.std(ddof=0)
    d = y[:, None] - y[None, :]
    if weight_df == 1:
        kernel = 1 / (np.pi * (1 + d**2))
        cross = (2 / (4 + y**2) + (4 - y**2) / (4 + y**2) ** 2) / np.pi
        constant = 17 / (27 * np.pi)
    else:
        kernel = 2 / (np.pi * (1 + d**2) ** 2)
        cross = (
            2 / (4 + y**2) + 2 * (4 - y**2) / (4 + y**2) ** 2 + 2 * (8 - 6 * y**2) / (4 + y**2) ** 3
        ) / np.pi
        constant = 26 / (27 * np.pi)
    expected = kernel.sum() / len(y) - 2 * cross.sum() + len(y) * constant
    actual = make(student.MeintanisStudentGofStatistic, df=3, weight_df=weight_df)
    assert actual.execute_statistic(SAMPLE) == pytest.approx(expected, abs=2e-9)


@pytest.mark.parametrize("df", [2.1, 3, 5, 20])
@pytest.mark.parametrize("k", [1, 3, len(SAMPLE) - 1])
def test_renyi_against_density_integral_and_brute_force_neighbours(df, k):
    q = 1 - 2 / (df + 1)
    scale = np.sqrt((1 - 2 / df) * SAMPLE.var(ddof=1))
    integral = 2 * quad(lambda x: t.pdf(x, df, scale=scale) ** q, 0, np.inf)[0]
    maximum = np.log(integral) / (1 - q)
    distances = np.abs(SAMPLE[:, None] - SAMPLE[None, :])
    np.fill_diagonal(distances, np.inf)
    rho = np.sort(distances, axis=1)[:, k - 1]
    c = (gamma(k) / gamma(k + 1 - q)) ** (1 / (1 - q))
    estimate = np.log(np.mean((2 * (len(SAMPLE) - 1) * c * rho) ** (1 - q))) / (1 - q)
    actual = make(student.CadirciRenyiStudentGofStatistic, df=df, k=k)
    assert actual.execute_statistic(SAMPLE) == pytest.approx(maximum - estimate, abs=2e-7)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("multiplier,offset", [(3, 7), (-2, 100), (1e250, 0), (1e-250, 0)])
def test_affine_invariance_no_mutation_and_no_fitted_state(cls, multiplier, offset):
    statistic = make(cls)
    state = statistic.__dict__.copy()
    data = SAMPLE.copy()
    expected = statistic.execute_statistic(data)
    assert statistic.execute_statistic(multiplier * data + offset) == pytest.approx(expected)
    assert statistic.execute_statistic(data[::-1]) == pytest.approx(expected)
    np.testing.assert_array_equal(data, SAMPLE)
    assert statistic.__dict__ == state
    assert statistic.hypothesis().parameters() == {"df": 5}
    assert statistic.alternative().type() is AlternativeType.RIGHT


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize(
    "bad",
    [
        [],
        [1],
        [1, 1, 1, 1],
        [[1, 2]],
        [1, np.nan],
        [1, np.inf],
        [1j, 2j],
        [True, False],
        ["1", "2"],
    ],
)
def test_invalid_samples(cls, bad):
    with pytest.raises(ValueError):
        make(cls).execute_statistic(bad)


@pytest.mark.parametrize("cls", CLASSES)
def test_masked_sample(cls):
    with pytest.raises(ValueError, match="masks"):
        make(cls).execute_statistic(np.ma.array(SAMPLE, mask=[True] + [False] * 8))


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("df", [0.5, 1, 2])
def test_finite_variance_required(cls, df):
    with pytest.raises(ValueError, match="df > 2"):
        make(cls, df=df)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("values", [{}, {"df": 5, "loc": 0}, {"df": 5, "loc": 0, "scale": 1}])
def test_only_df_fixed_hypothesis_supported(cls, values):
    with pytest.raises(ValueError, match="Unsupported hypothesis"):
        cls(Student.DEFAULT.parse(values))


@pytest.mark.parametrize("bad", [True, 0, -1, 1.5, "3", np.nan])
def test_invalid_k(bad):
    with pytest.raises(ValueError, match="k"):
        make(student.CadirciRenyiStudentGofStatistic, k=bad)


def test_knn_sample_size_and_ties():
    statistic = make(student.CadirciRenyiStudentGofStatistic)
    with pytest.raises(ValueError, match="sample size"):
        statistic.execute_statistic([1, 2, 3])
    with pytest.raises(ValueError, match="distinct"):
        statistic.execute_statistic([1, 1, 2, 3, 4])


@pytest.mark.parametrize("option", ["weight_df", "epsabs", "epsrel", "limit"])
@pytest.mark.parametrize("bad", [True, 0, -1, "1", np.nan, np.inf])
def test_invalid_ecf_options(option, bad):
    with pytest.raises((TypeError, ValueError)):
        make(student.MeintanisStudentGofStatistic, **{option: bad})


def test_quadrature_failure_is_reported():
    with pytest.raises(ValueError, match="quadrature"):
        make(student.MeintanisStudentGofStatistic, limit=1).execute_statistic(SAMPLE)


@pytest.mark.parametrize("cls", CLASSES)
def test_sampler_preserves_df_and_refits_every_replication(cls):
    statistic = make(cls, df=7)
    generator = get_hypothesis_generator(statistic)
    assert generator.parameters() == {"df": 7, "loc": 0, "scale": 1}
    restored_cls = pickle.loads(pickle.dumps(cls))  # noqa: S301 - local class reference
    restored = restored_cls(statistic.hypothesis().parameter_values)
    assert restored.configuration() == statistic.configuration()
    assert restored.execute_statistic(SAMPLE) == pytest.approx(statistic.execute_statistic(SAMPLE))
    for sample in np.random.default_rng(17).standard_t(7, size=(5, 15)):
        assert np.isfinite(statistic.execute_statistic(sample))


def test_configuration_distinguishes_calibrations():
    assert (
        make(student.CadirciRenyiStudentGofStatistic, k=1).configuration()
        != make(student.CadirciRenyiStudentGofStatistic, k=3).configuration()
    )
    assert (
        make(student.MeintanisStudentGofStatistic).configuration()
        != make(student.MeintanisStudentGofStatistic, weight_df=1).configuration()
    )


def test_large_df_entropy_limit_and_negative_values_are_preserved():
    # Shannon nearest-neighbour estimate in the Gaussian/unit-variance limit.
    from scipy.special import digamma

    z = (SAMPLE - SAMPLE.mean()) / SAMPLE.std(ddof=1)
    distances = np.abs(z[:, None] - z[None, :])
    np.fill_diagonal(distances, np.inf)
    rho = np.sort(distances, axis=1)[:, 2]
    shannon = np.log(2 * (len(z) - 1)) - digamma(3) + np.mean(np.log(rho))
    expected = np.log(2 * np.pi * np.e) / 2 - shannon
    value = make(student.CadirciRenyiStudentGofStatistic, df=1e10).execute_statistic(SAMPLE)
    assert value == pytest.approx(expected, abs=1e-8)
    assert (
        make(student.CadirciRenyiStudentGofStatistic).execute_statistic(np.linspace(-2, 2, 20)) < 0
    )


@pytest.mark.parametrize("df", [1, 3, 5, 1000])
def test_characteristic_function_against_density_fourier_integral(df):
    for u in (0.0, 0.01, 0.4, 2.0):
        expected = 2 * quad(lambda x: t.pdf(x, df), 0, np.inf, weight="cos", wvar=u)[0]
        assert student_statistics._student_cf(u, df) == pytest.approx(expected, abs=2e-8)


@pytest.mark.parametrize("cls", CLASSES)
def test_registry_and_calibration_identity(cls):
    from pysatl_criterion.persistence.models.distribution_key import DistributionKey
    from pysatl_criterion.utils.statistic import get_available_criteria, get_criterion_by_code

    assert get_criterion_by_code(cls.code()) is cls
    assert get_available_criteria(Student.type()).count(cls) == 1
    assert cls.__name__ in student.__all__
    first = DistributionKey.from_statistic(make(cls, df=5), 30)
    other_df = DistributionKey.from_statistic(make(cls, df=7), 30)
    options = {"k": 1} if cls is student.CadirciRenyiStudentGofStatistic else {"weight_df": 1}
    other_settings = DistributionKey.from_statistic(make(cls, **options), 30)
    assert first != other_df
    assert first != other_settings
