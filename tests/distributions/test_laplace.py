import pytest

from pysatl_criterion import DistributionType
from pysatl_criterion.statistics.goodness_of_fit.laplace import (
    AbstractLaplaceGofStatistic,
    AndersonDarlingLaplaceGofStatistic,
    CramerVonMisesLaplaceGofStatistic,
    KolmogorovSmirnovLaplaceGofStatistic,
)


_SAMPLE = [0.1, 0.2, 0.3]
_LARGE_SAMPLE = [
    -3.2,
    -2.7,
    -2.1,
    -1.8,
    -1.5,
    -1.3,
    -1.1,
    -0.9,
    -0.7,
    -0.5,
    -0.4,
    -0.3,
    -0.2,
    -0.1,
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.7,
    0.9,
    1.1,
    1.3,
    1.5,
    1.8,
    2.1,
    2.4,
    2.8,
    3.3,
]
_LOCATION = 0.0
_SCALE = 1.0


def test_laplace_base_code():
    assert AbstractLaplaceGofStatistic.code() == "LAPLACE_GOODNESS_OF_FIT"


def test_laplace_hypothesis():
    statistic = KolmogorovSmirnovLaplaceGofStatistic(t=_LOCATION, s=_SCALE)

    hypothesis = statistic.hypothesis()

    assert hypothesis.params == {"t": _LOCATION, "s": _SCALE}


def test_kolmogorov_smirnov_laplace_statistic():
    statistic = KolmogorovSmirnovLaplaceGofStatistic(
        t=_LOCATION,
        s=_SCALE,
    )

    result = statistic.execute_statistic(_SAMPLE)

    assert result == pytest.approx(0.5475812909820202)


def test_cramer_von_mises_laplace_statistic():
    statistic = CramerVonMisesLaplaceGofStatistic(
        t=_LOCATION,
        s=_SCALE,
    )

    result = statistic.execute_statistic(_SAMPLE)

    assert result == pytest.approx(0.2225993471193351)


def test_anderson_darling_laplace_statistic():
    statistic = AndersonDarlingLaplaceGofStatistic(
        t=_LOCATION,
        s=_SCALE,
    )

    result = statistic.execute_statistic(_SAMPLE)

    assert result == pytest.approx(1.0445557661164182)


@pytest.mark.parametrize(
    ("statistic_class", "expected_value"),
    [
        (KolmogorovSmirnovLaplaceGofStatistic, 0.10023112481762697),
        (CramerVonMisesLaplaceGofStatistic, 0.06360683737580435),
        (AndersonDarlingLaplaceGofStatistic, 0.4993990825244552),
    ],
)
def test_laplace_statistics_on_large_sample(statistic_class, expected_value):
    statistic = statistic_class(t=_LOCATION, s=_SCALE)

    result = statistic.execute_statistic(_LARGE_SAMPLE)

    assert result == pytest.approx(expected_value)


@pytest.mark.parametrize(
    ("statistic_class", "expected_code"),
    [
        (
            KolmogorovSmirnovLaplaceGofStatistic,
            "KS_LAPLACE_GOODNESS_OF_FIT",
        ),
        (
            CramerVonMisesLaplaceGofStatistic,
            "CVM_LAPLACE_GOODNESS_OF_FIT",
        ),
        (
            AndersonDarlingLaplaceGofStatistic,
            "AD_LAPLACE_GOODNESS_OF_FIT",
        ),
    ],
)
def test_laplace_statistic_codes(statistic_class, expected_code):
    assert statistic_class.code() == expected_code


def test_laplace_distribution_type():
    assert AbstractLaplaceGofStatistic.distribution() == DistributionType.LAPLACE


@pytest.mark.parametrize("scale", [0.0, -1.0])
def test_laplace_positive_scale_required(scale):
    with pytest.raises(ValueError, match="Scale must be positive."):
        KolmogorovSmirnovLaplaceGofStatistic(s=scale)


def test_laplace_parameters_affect_statistic():
    default_statistic = KolmogorovSmirnovLaplaceGofStatistic(t=0.0, s=1.0)
    shifted_statistic = KolmogorovSmirnovLaplaceGofStatistic(t=1.0, s=2.0)

    default_result = default_statistic.execute_statistic(_SAMPLE)
    shifted_result = shifted_statistic.execute_statistic(_SAMPLE)

    assert default_result != pytest.approx(shifted_result)
