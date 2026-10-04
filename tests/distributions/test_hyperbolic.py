from collections import Counter

import numpy as np
import pytest
import scipy.stats as scipy_stats

from pysatl_criterion import DistributionType
from pysatl_criterion.distribution.distributions import HyperbolicDistributionDescriptor
from pysatl_criterion.statistics.alternative import (
    AlternativeType,
    LeftAlternative,
    RightAlternative,
    TwoSidedAlternative,
)
from pysatl_criterion.statistics.goodness_of_fit.hyperbolic import (
    AbstractHyperbolicGofStatistic,
    AndersonDarlingHyperbolicGofStatistic,
    CramerVonMisesHyperbolicGofStatistic,
    KolmogorovSmirnovHyperbolicGofStatistic,
    KuiperHyperbolicGofStatistic,
    WatsonHyperbolicGofStatistic,
)
from pysatl_criterion.utils.distribution import get_available_distribution_descriptor
from pysatl_criterion.utils.statistic import get_available_criteria_codes


_SAMPLE = np.array([-2.4, 1.3, -0.2, 0.0, 2.1, -1.0, 0.7], dtype=np.float64)


def _distribution_parameters(
    alpha: float,
    beta: float,
    delta: float,
    mu: float,
) -> dict[str, float]:
    """Translate the classical hyperbolic parameters to SciPy parameters."""
    return {
        "p": 1.0,
        "a": alpha * delta,
        "b": beta * delta,
        "loc": mu,
        "scale": delta,
    }


def _cdf_values(
    sample,
    alpha: float,
    beta: float,
    delta: float,
    mu: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a sorted sample and its SciPy hyperbolic CDF values."""
    sorted_sample = np.sort(np.asarray(sample, dtype=np.float64))
    cdf_values = scipy_stats.genhyperbolic.cdf(
        sorted_sample,
        **_distribution_parameters(alpha, beta, delta, mu),
    )
    return sorted_sample, np.asarray(cdf_values, dtype=np.float64)


def test_hyperbolic_base_metadata():
    """The base class should expose stable distribution metadata."""
    statistic = CramerVonMisesHyperbolicGofStatistic(
        alpha=1.5,
        beta=0.25,
        delta=2.0,
        mu=-0.5,
    )

    assert AbstractHyperbolicGofStatistic.code() == "HYPERBOLIC_GOODNESS_OF_FIT"
    assert statistic.distribution() == DistributionType.HYPERBOLIC
    assert statistic.hypothesis().params == {
        "alpha": 1.5,
        "beta": 0.25,
        "delta": 2.0,
        "mu": -0.5,
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"alpha": 0.0}, "Alpha must be finite and positive"),
        ({"alpha": np.inf}, "Alpha must be finite and positive"),
        ({"alpha": 1.0, "beta": 1.0}, r"abs\(beta\) < alpha"),
        ({"alpha": 1.0, "beta": -1.0}, r"abs\(beta\) < alpha"),
        ({"beta": np.nan}, "Beta must be finite"),
        ({"delta": 0.0}, "Delta must be finite and positive"),
        ({"delta": np.inf}, "Delta must be finite and positive"),
        ({"mu": np.nan}, "Mu must be finite"),
    ],
)
def test_hyperbolic_parameter_validation(kwargs, message):
    """Invalid hyperbolic parameters should be rejected."""
    with pytest.raises(ValueError, match=message):
        CramerVonMisesHyperbolicGofStatistic(**kwargs)


@pytest.mark.parametrize("sample", [[], np.ones((2, 2))])
def test_hyperbolic_sample_validation(sample):
    """Statistics should require a non-empty one-dimensional sample."""
    with pytest.raises(ValueError):
        CramerVonMisesHyperbolicGofStatistic().execute_statistic(sample)


@pytest.mark.parametrize(
    ("sample", "alpha", "beta", "delta", "mu"),
    [
        (_SAMPLE, 1.5, 0.25, 1.0, 0.0),
        ([3, -2, 1, 1, 0], 2.0, -0.5, 0.75, 1.0),
        ([-8.0, -2.0, 0.0, 5.0, 12.0], 0.8, 0.2, 2.5, -1.5),
    ],
)
def test_hyperbolic_edf_statistics_match_scipy_reference(
    sample,
    alpha,
    beta,
    delta,
    mu,
):
    """Numba statistic kernels should match independent NumPy/SciPy formulas."""
    sorted_sample, cdf_values = _cdf_values(sample, alpha, beta, delta, mu)
    n = len(sorted_sample)
    positions = np.arange(n, dtype=np.float64)
    expected_cdf = (2.0 * positions + 1.0) / (2.0 * n)
    d_plus = np.max((positions + 1.0) / n - cdf_values)
    d_minus = np.max(cdf_values - positions / n)
    expected_cvm = 1.0 / (12.0 * n) + np.sum((expected_cdf - cdf_values) ** 2)
    expected_kuiper = d_plus + d_minus
    expected_watson = expected_cvm - n * (np.mean(cdf_values) - 0.5) ** 2

    parameters = {
        "alpha": alpha,
        "beta": beta,
        "delta": delta,
        "mu": mu,
    }

    assert KolmogorovSmirnovHyperbolicGofStatistic(**parameters).execute_statistic(
        sample
    ) == pytest.approx(max(d_plus, d_minus))
    assert CramerVonMisesHyperbolicGofStatistic(**parameters).execute_statistic(
        sample
    ) == pytest.approx(expected_cvm)
    assert KuiperHyperbolicGofStatistic(**parameters).execute_statistic(sample) == pytest.approx(
        expected_kuiper
    )
    assert WatsonHyperbolicGofStatistic(**parameters).execute_statistic(sample) == pytest.approx(
        expected_watson
    )


@pytest.mark.parametrize(
    ("alpha", "beta", "delta", "mu"),
    [
        (1.5, 0.25, 1.0, 0.0),
        (2.0, -0.5, 0.75, 1.0),
    ],
)
def test_anderson_darling_hyperbolic_matches_scipy_reference(alpha, beta, delta, mu):
    """Anderson--Darling should use stable SciPy tail probabilities."""
    sorted_sample = np.sort(_SAMPLE)
    parameters = _distribution_parameters(alpha, beta, delta, mu)
    log_cdf = scipy_stats.genhyperbolic.logcdf(sorted_sample, **parameters)
    log_sf = scipy_stats.genhyperbolic.logsf(sorted_sample, **parameters)
    n = len(sorted_sample)
    weights = (2.0 * np.arange(n) + 1.0) / n
    expected = -n - np.sum(weights * (log_cdf + log_sf[::-1]))

    result = AndersonDarlingHyperbolicGofStatistic(
        alpha=alpha,
        beta=beta,
        delta=delta,
        mu=mu,
    ).execute_statistic(_SAMPLE)

    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    ("alternative_type", "alternative_class", "expected_side"),
    [
        (AlternativeType.TWO_TAILED, TwoSidedAlternative, "two-sided"),
        (AlternativeType.RIGHT, RightAlternative, "right"),
        (AlternativeType.LEFT, LeftAlternative, "left"),
    ],
)
def test_kolmogorov_smirnov_hyperbolic_alternatives(
    alternative_type,
    alternative_class,
    expected_side,
):
    """KS should calculate the deviation selected by its alternative."""
    _, cdf_values = _cdf_values(_SAMPLE, 1.5, 0.25, 1.0, 0.0)
    n = len(cdf_values)
    positions = np.arange(n, dtype=np.float64)
    d_plus = np.max((positions + 1.0) / n - cdf_values)
    d_minus = np.max(cdf_values - positions / n)
    expected = {
        "two-sided": max(d_plus, d_minus),
        "right": d_plus,
        "left": d_minus,
    }[expected_side]
    statistic = KolmogorovSmirnovHyperbolicGofStatistic(
        alternative_type=alternative_type,
        alpha=1.5,
        beta=0.25,
    )

    assert isinstance(statistic.alternative(), alternative_class)
    assert statistic.execute_statistic(_SAMPLE) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("statistic", "expected_code"),
    [
        (KolmogorovSmirnovHyperbolicGofStatistic, "KS_HYPERBOLIC_GOODNESS_OF_FIT"),
        (CramerVonMisesHyperbolicGofStatistic, "CVM_HYPERBOLIC_GOODNESS_OF_FIT"),
        (AndersonDarlingHyperbolicGofStatistic, "AD_HYPERBOLIC_GOODNESS_OF_FIT"),
        (KuiperHyperbolicGofStatistic, "KUI_HYPERBOLIC_GOODNESS_OF_FIT"),
        (WatsonHyperbolicGofStatistic, "WAT_HYPERBOLIC_GOODNESS_OF_FIT"),
    ],
)
def test_hyperbolic_statistic_codes(statistic, expected_code):
    """Every hyperbolic statistic should have a stable unique code."""
    assert statistic.code() == expected_code


@pytest.mark.parametrize(
    "statistic",
    [
        CramerVonMisesHyperbolicGofStatistic(),
        AndersonDarlingHyperbolicGofStatistic(),
        KuiperHyperbolicGofStatistic(),
        WatsonHyperbolicGofStatistic(),
    ],
)
def test_hyperbolic_statistics_use_right_alternative(statistic):
    """Quadratic and supremum statistics should use a right-tailed alternative."""
    assert isinstance(statistic.alternative(), RightAlternative)


def test_hyperbolic_criteria_are_discoverable():
    """The public criterion registry should discover all hyperbolic statistics."""
    codes = get_available_criteria_codes(DistributionType.HYPERBOLIC)
    assert Counter(codes) == Counter(["KS", "CVM", "AD", "KUI", "WAT"])


def test_hyperbolic_distribution_descriptor():
    """The distribution registry should expose hyperbolic parameter metadata."""
    descriptor = get_available_distribution_descriptor(DistributionType.HYPERBOLIC)

    assert isinstance(descriptor, HyperbolicDistributionDescriptor)
    assert descriptor.type() == DistributionType.HYPERBOLIC
    assert [parameter.name for parameter in descriptor.parameters()] == [
        "alpha",
        "beta",
        "delta",
        "mu",
    ]


def test_hyperbolic_nan_sample_propagates():
    """An undefined CDF value should produce an undefined statistic."""
    result = KuiperHyperbolicGofStatistic().execute_statistic([0.0, np.nan, 1.0])
    assert np.isnan(result)


def test_kolmogorov_smirnov_requires_precomputed_cdf():
    """The low-level KS entry point should require its probability array."""
    with pytest.raises(ValueError, match="CDF values are required"):
        KolmogorovSmirnovHyperbolicGofStatistic().do_execute_statistic(_SAMPLE)


def test_anderson_darling_requires_log_probabilities():
    """The low-level AD entry point should require both tail arrays."""
    with pytest.raises(ValueError, match="Log-CDF and log-survival values are required"):
        AndersonDarlingHyperbolicGofStatistic().do_execute_statistic(_SAMPLE)
