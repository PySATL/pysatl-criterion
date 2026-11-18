"""Lequesne (2015), equations (7), (8) and (10), and bootstrap restrictions."""

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


Lequesne = student.LequesneRenyiStudentGofStatistic
SAMPLE = np.array([-3.7, -1.1, -0.6, -0.2, 0.3, 0.8, 1.9, 4.2, 7.1])


def reference(sample, df, scale, k):
    q = (df - 1) / (df + 1)
    integral = 2 * quad(lambda x: t.pdf(x, df, scale=scale) ** q, 0, np.inf)[0]
    theoretical = np.log(integral) / (1 - q)
    distances = np.abs(sample[:, None] - sample[None, :])
    np.fill_diagonal(distances, np.inf)
    rho = np.sort(distances, axis=1)[:, k - 1]
    g = (gamma(k) / gamma(k + 1 - q)) ** (1 / (1 - q))
    estimated = np.log(np.mean((2 * (len(sample) - 1) * g * rho) ** (1 - q))) / (1 - q)
    return theoretical - estimated


@pytest.mark.parametrize("df", [2.5, 5, 10])
@pytest.mark.parametrize("k", [1, 3, 7])
def test_t1_matches_equations_7_and_10(df, k):
    criterion = Lequesne(Student.DEFAULT.parse({"df": df}), k=k)
    scale = np.sqrt((df - 2) / df * SAMPLE.var(ddof=1))
    assert criterion.execute_statistic(SAMPLE) == pytest.approx(reference(SAMPLE, df, scale, k))
    cadirci = student.CadirciRenyiStudentGofStatistic(criterion.hypothesis().parameter_values, k=k)
    assert criterion.execute_statistic(SAMPLE) == cadirci.execute_statistic(SAMPLE)


@pytest.mark.parametrize("fitted_df,fitted_scale", [(3, 0.8), (7, 1.3)])
def test_t2_uses_fitted_q_and_mle_scale(monkeypatch, fitted_df, fitted_scale):
    def fit(sample, *args, **kwargs):
        assert sample.std(ddof=1) == pytest.approx(1)
        assert callable(kwargs["optimizer"])
        return fitted_df, 0.2, fitted_scale

    monkeypatch.setattr(student_statistics.scipy_stats.t, "fit", fit)
    standardized = (SAMPLE - SAMPLE.mean()) / SAMPLE.std(ddof=1)
    criterion = Lequesne(Student.DEFAULT.parse({}), k=2)
    assert criterion.execute_statistic(SAMPLE) == pytest.approx(
        reference(standardized, fitted_df, fitted_scale, 2)
    )
    assert criterion.hypothesis().parameters() == {}


def test_t2_real_fit_against_scipy_and_affine_invariance():
    sample = np.random.default_rng(1729).standard_t(5, 250)
    df, _, scale = t.fit(sample, 5)
    assert 2 < df < 20
    expected = reference(sample, df, scale, 3)
    criterion = Lequesne(Student.DEFAULT.parse({}))
    before = sample.copy()
    state = criterion.__dict__.copy()
    value = criterion.execute_statistic(sample)
    assert value == pytest.approx(expected, abs=2e-5)
    assert criterion.execute_statistic(7 - 3 * sample) == pytest.approx(value, abs=1e-7)
    assert criterion.execute_statistic(sample[::-1]) == pytest.approx(value, abs=1e-10)
    np.testing.assert_array_equal(sample, before)
    assert criterion.__dict__ == state
    assert criterion.alternative().type() is AlternativeType.RIGHT


@pytest.mark.parametrize("fit", [(2, 0, 1), (1, 0, 1), (5, 0, 0), (np.inf, 0, 1)])
def test_invalid_fitted_parameters_are_not_clamped(monkeypatch, fit):
    monkeypatch.setattr(student_statistics.scipy_stats.t, "fit", lambda *args, **kwargs: fit)
    with pytest.raises(ValueError, match="finite df > 2"):
        Lequesne(Student.DEFAULT.parse({})).execute_statistic(SAMPLE)


def test_nonconverged_optimizer_is_reported(monkeypatch):
    monkeypatch.setattr(
        student_statistics, "fmin", lambda *args, **kwargs: (np.ones(3), 1.0, 10, 20, 1)
    )
    with pytest.raises(ValueError, match="did not converge"):
        Lequesne(Student.DEFAULT.parse({})).execute_statistic(SAMPLE)


@pytest.mark.parametrize("values", [{"df": 5}, {}])
def test_sample_and_neighbour_validation(values):
    criterion = Lequesne(Student.DEFAULT.parse(values))
    for sample in ([], [1, 2, 3], [1, 1, 2, 3], [1, 2, np.inf, 4], [[1, 2], [3, 4]]):
        with pytest.raises(ValueError):
            criterion.execute_statistic(sample)
    for k in (True, 0, -1, 2.5):
        with pytest.raises(ValueError, match="k"):
            Lequesne(Student.DEFAULT.parse(values), k=k)
    with pytest.raises(ValueError, match="four observations"):
        Lequesne(Student.DEFAULT.parse({}), k=1).execute_statistic([1, 2, 3])


def test_known_df_domain_and_negative_values():
    for df in (1, 2):
        with pytest.raises(ValueError, match="df > 2"):
            Lequesne(Student.DEFAULT.parse({"df": df}))
    criterion = Lequesne(Student.DEFAULT.parse({"df": 5}))
    assert criterion.execute_statistic(np.linspace(-2, 2, 20)) < 0


def test_calibration_contracts():
    known = Lequesne(Student.DEFAULT.parse({"df": 7}))
    assert get_hypothesis_generator(known).parameters() == {"df": 7, "loc": 0, "scale": 1}
    known._validate_storage_calibration()
    unknown = Lequesne(Student.DEFAULT.parse({}))
    with pytest.raises(ValueError, match="external parametric bootstrap"):
        get_hypothesis_generator(unknown)
    with pytest.raises(ValueError, match="external parametric bootstrap"):
        unknown._validate_storage_calibration()
    assert known.configuration() != unknown.configuration()
    assert unknown.configuration() != Lequesne(Student.DEFAULT.parse({}), k=1).configuration()


def test_registration_and_explicit_hypotheses():
    from pysatl_criterion.utils.statistic import get_available_criteria, get_criterion_by_code

    assert "supported_hypotheses" in Lequesne.__dict__
    assert "supported_hypotheses" not in student.AbstractStudentGofStatistic.__dict__
    assert Lequesne.__name__ in student.__all__
    assert get_criterion_by_code(Lequesne.code()) is Lequesne
    assert get_available_criteria(Student.type()).count(Lequesne) == 1
    with pytest.raises(ValueError, match="Unsupported hypothesis"):
        Lequesne(Student.DEFAULT.parse({"df": 5, "loc": 0, "scale": 1}))
