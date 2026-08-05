import numpy as np
import pytest
import scipy.stats as scipy_stats

from pysatl_criterion import DistributionType
from pysatl_criterion.statistics.alternative import RightAlternative
from pysatl_criterion.statistics.goodness_of_fit.laplace import (
    AbstractLaplaceGofStatistic,
    AndersonDarlingLaplaceGofStatistic,
    CramerVonMisesLaplaceGofStatistic,
    GreenwoodLaplaceGofStatistic,
    KolmogorovSmirnovLaplaceGofStatistic,
    KuiperLaplaceGofStatistic,
    WatsonLaplaceGofStatistic,
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
    """Ensure Laplace GOF statistics expose the expected code identifier."""

    assert AbstractLaplaceGofStatistic.code() == "LAPLACE_GOODNESS_OF_FIT"


def test_laplace_hypothesis():
    """Ensure the Laplace location and scale are stored in the hypothesis."""

    statistic = KolmogorovSmirnovLaplaceGofStatistic(t=_LOCATION, s=_SCALE)

    hypothesis = statistic.hypothesis()

    assert hypothesis.params == {"t": _LOCATION, "s": _SCALE}


def test_kolmogorov_smirnov_laplace_statistic():
    """Verify the Kolmogorov--Smirnov statistic for a Laplace sample."""

    statistic = KolmogorovSmirnovLaplaceGofStatistic(
        t=_LOCATION,
        s=_SCALE,
    )

    result = statistic.execute_statistic(_SAMPLE)

    assert result == pytest.approx(0.5475812909820202)


def test_cramer_von_mises_laplace_statistic():
    """Verify the Cramer--von Mises statistic for a Laplace sample."""

    statistic = CramerVonMisesLaplaceGofStatistic(
        t=_LOCATION,
        s=_SCALE,
    )

    result = statistic.execute_statistic(_SAMPLE)

    assert result == pytest.approx(0.2225993471193351)


def test_anderson_darling_laplace_statistic():
    """Verify the Anderson--Darling statistic for a Laplace sample."""

    statistic = AndersonDarlingLaplaceGofStatistic(
        t=_LOCATION,
        s=_SCALE,
    )

    result = statistic.execute_statistic(_SAMPLE)

    assert result == pytest.approx(1.0445557661164182)


@pytest.mark.parametrize(
    ("sample", "location", "scale"),
    [
        ([0.1], 0.0, 1.0),
        ([2.0, -1.0, 0.5, 0.5], 0.0, 1.0),
        ([-5.0, -2.0, 3.0, 8.0], 1.0, 2.5),
        (np.array([1, -2, 3, 0], dtype=np.int64), -0.5, 1.5),
        ([-1_000.0, -100.0, 0.0, 100.0, 1_000.0], 0.0, 1.0),
        ([], 0.0, 1.0),
    ],
)
def test_anderson_darling_laplace_matches_scipy_reference(sample, location, scale):
    """Optimized Anderson--Darling implementation should match SciPy."""
    sorted_sample = np.sort(np.asarray(sample, dtype=np.float64))
    n = len(sorted_sample)

    if n == 0:
        expected = 0.0
    else:
        log_cdf = scipy_stats.laplace.logcdf(
            sorted_sample,
            loc=location,
            scale=scale,
        )
        log_sf = scipy_stats.laplace.logsf(
            sorted_sample,
            loc=location,
            scale=scale,
        )
        i = np.arange(1, n + 1)
        expected = -n - np.sum((2 * i - 1.0) / n * (log_cdf + log_sf[::-1]))

    result = AndersonDarlingLaplaceGofStatistic(
        t=location,
        s=scale,
    ).execute_statistic(sample)

    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    ("sample", "location", "scale"),
    [
        ([0.1], 0.0, 1.0),
        ([2.0, -1.0, 0.5, 0.5], 0.0, 1.0),
        ([-5.0, -2.0, 3.0, 8.0], 1.0, 2.5),
        (np.array([1, -2, 3, 0], dtype=np.int64), -0.5, 1.5),
        ([-1_000.0, -100.0, 0.0, 100.0, 1_000.0], 0.0, 1.0),
    ],
)
def test_watson_laplace_matches_scipy_reference(sample, location, scale):
    """Optimized Watson implementation should match the SciPy reference."""
    sorted_sample = np.sort(np.asarray(sample, dtype=np.float64))
    n = len(sorted_sample)

    cdf_values = scipy_stats.laplace.cdf(
        sorted_sample,
        loc=location,
        scale=scale,
    )
    expected_cdf = (2 * np.arange(1, n + 1) - 1) / (2 * n)
    differences = cdf_values - expected_cdf

    w_squared = 1.0 / (12 * n) + np.sum(differences**2)
    mean_adjustment = np.sum(cdf_values) - n / 2
    expected = w_squared - mean_adjustment**2 / n

    result = WatsonLaplaceGofStatistic(
        t=location,
        s=scale,
    ).execute_statistic(sample)

    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    ("statistic_class", "expected_value"),
    [
        (KolmogorovSmirnovLaplaceGofStatistic, 0.10023112481762697),
        (CramerVonMisesLaplaceGofStatistic, 0.06360683737580435),
        (AndersonDarlingLaplaceGofStatistic, 0.4993990825244552),
        (GreenwoodLaplaceGofStatistic, 0.03654367524062204),
        (KuiperLaplaceGofStatistic, 0.16712891630192053),
        (WatsonLaplaceGofStatistic, 0.05656030222478137),
    ],
)
def test_laplace_statistics_on_large_sample(statistic_class, expected_value):
    """Check all Laplace EDF statistics against a larger reference sample."""

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
        (
            GreenwoodLaplaceGofStatistic,
            "GRW_LAPLACE_GOODNESS_OF_FIT",
        ),
        (
            KuiperLaplaceGofStatistic,
            "KUI_LAPLACE_GOODNESS_OF_FIT",
        ),
        (
            WatsonLaplaceGofStatistic,
            "WAT_LAPLACE_GOODNESS_OF_FIT",
        ),
    ],
)
def test_laplace_statistic_codes(statistic_class, expected_code):
    """Ensure every Laplace statistic exposes a stable code identifier."""

    assert statistic_class.code() == expected_code


def test_greenwood_laplace_alternative():
    """Ensure Greenwood uses a right-tailed alternative."""

    assert isinstance(GreenwoodLaplaceGofStatistic().alternative(), RightAlternative)


def test_watson_laplace_alternative():
    """Ensure Watson uses a right-tailed alternative."""
    assert isinstance(WatsonLaplaceGofStatistic().alternative(), RightAlternative)


def test_laplace_distribution_type():
    """Ensure Laplace statistics report the Laplace distribution type."""

    assert AbstractLaplaceGofStatistic.distribution() == DistributionType.LAPLACE


@pytest.mark.parametrize("scale", [0.0, -1.0])
def test_laplace_positive_scale_required(scale):
    """Laplace statistic constructors should reject non-positive scales."""

    with pytest.raises(ValueError, match="Scale must be positive."):
        KolmogorovSmirnovLaplaceGofStatistic(s=scale)


def test_laplace_parameters_affect_statistic():
    """Changing location and scale should change the resulting KS statistic."""

    default_statistic = KolmogorovSmirnovLaplaceGofStatistic(t=0.0, s=1.0)
    shifted_statistic = KolmogorovSmirnovLaplaceGofStatistic(t=1.0, s=2.0)

    default_result = default_statistic.execute_statistic(_SAMPLE)
    shifted_result = shifted_statistic.execute_statistic(_SAMPLE)

    assert default_result != pytest.approx(shifted_result)


@pytest.mark.parametrize(
    "statistic_class",
    [
        KuiperLaplaceGofStatistic,
        WatsonLaplaceGofStatistic,
        GreenwoodLaplaceGofStatistic,
    ],
)
def test_additional_laplace_statistics_require_non_empty_sample(statistic_class):
    """Additional Laplace statistics should reject an empty sample."""

    with pytest.raises(ValueError, match="At least one observation is required"):
        statistic_class(t=_LOCATION, s=_SCALE).execute_statistic([])


@pytest.mark.parametrize(
    ("sample", "location", "scale"),
    [
        ([0.1], 0.0, 1.0),
        ([2.0, -1.0, 0.5, 0.5], 0.0, 1.0),
        ([-5.0, -2.0, 3.0, 8.0], 1.0, 2.5),
        (np.array([1, -2, 3, 0], dtype=np.int64), -0.5, 1.5),
    ],
)
def test_greenwood_laplace_matches_scipy_reference(sample, location, scale):
    """Optimized Greenwood implementation should match the SciPy reference."""

    sorted_sample = np.sort(np.asarray(sample, dtype=np.float64))
    cdf_values = scipy_stats.laplace.cdf(
        sorted_sample,
        loc=location,
        scale=scale,
    )
    spacings = np.diff(
        np.concatenate(
            (
                [0.0],
                cdf_values,
                [1.0],
            )
        )
    )
    expected = np.sum(spacings**2)

    statistic = GreenwoodLaplaceGofStatistic(
        t=location,
        s=scale,
    )

    result = statistic.execute_statistic(sample)

    assert result == pytest.approx(expected)
