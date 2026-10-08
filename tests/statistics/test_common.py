import numpy as np
import pytest
from scipy import stats

from pysatl_criterion.distribution.distributions import BetaDistributionDescriptor as Beta
from pysatl_criterion.distribution.distributions import (
    ExponentiatedWeibullDistributionDescriptor as ExponentiatedWeibull,
)
from pysatl_criterion.distribution.distributions import (
    GammaDistributionDescriptor,
    UniformDistributionDescriptor,
)
from pysatl_criterion.hypothesis_testing.alternative_factory.alternative_factories import (
    AbstractAlternativeFactory,
)
from pysatl_criterion.statistics.alternative import AlternativeType
from pysatl_criterion.statistics.goodness_of_fit.beta import (
    Chi2PearsonBetaGofStatistic,
    EbnerLiebenbergBetaGofStatistic,
    KolmogorovSmirnovBetaGofStatistic,
)
from pysatl_criterion.statistics.goodness_of_fit.common import Chi2Statistic
from pysatl_criterion.statistics.goodness_of_fit.exponentiated_weibull import (
    Chi2PearsonExponentiatedWeibullGofStatistic,
)
from pysatl_criterion.statistics.goodness_of_fit.gamma import (
    CressieReadGammaGofStatistic,
    MinToshiyukiGammaGofStatistic,
)
from pysatl_criterion.statistics.goodness_of_fit.uniform import Chi2PearsonUniformGofStatistic


@pytest.mark.parametrize(
    ("direction", "scipy_direction"),
    [
        (AlternativeType.RIGHT, "greater"),
        (AlternativeType.LEFT, "less"),
        (AlternativeType.TWO_TAILED, "two-sided"),
    ],
)
def test_ks_direction_selects_deviation_but_always_uses_right_tail(direction, scipy_direction):
    sample = np.array([0.1, 0.15, 0.3, 0.65])
    statistic = KolmogorovSmirnovBetaGofStatistic(
        Beta.DEFAULT.parse({"a": 1, "b": 1}), alternative_type=direction
    )
    reference = stats.kstest(sample, stats.uniform.cdf, alternative=scipy_direction)
    assert statistic.execute_statistic(sample) == pytest.approx(reference.statistic)
    assert statistic.alternative().type() == AlternativeType.RIGHT


@pytest.mark.parametrize(
    "statistic",
    [
        KolmogorovSmirnovBetaGofStatistic(
            Beta.DEFAULT.parse({"a": 1, "b": 1}), alternative_type=direction
        )
        for direction in AlternativeType
    ]
    + [
        EbnerLiebenbergBetaGofStatistic(Beta.DEFAULT.parse({})),
        MinToshiyukiGammaGofStatistic(
            GammaDistributionDescriptor.DEFAULT.parse({"alfa": 1, "beta": 1})
        ),
    ],
)
def test_distance_statistics_use_upper_tail_p_values(statistic):
    factory = AbstractAlternativeFactory.get_concrete_factory(statistic.alternative().type())
    calculator = factory.get_p_value_calculator()
    null_distribution = [0.1, 0.2, 0.3, 0.4]
    assert calculator.calculate(null_distribution, 0.05) == 1.0
    assert calculator.calculate(null_distribution, 0.35) == 0.25
    assert calculator.calculate(null_distribution, 0.5) == 0.0


@pytest.mark.parametrize("power", [1, 0, -1, -2, -0.5, 2 / 3, 2])
def test_power_divergence_matches_scipy_for_positive_counts(power):
    observed, expected = [2, 7, 11], [5, 6, 9]
    actual = Chi2Statistic.do_execute_statistic(object(), observed, expected, power)
    reference = stats.power_divergence(observed, expected, lambda_=power).statistic
    assert actual == pytest.approx(reference)


@pytest.mark.parametrize(
    ("power", "expected"),
    [
        (1, 10),
        (0, 20 * np.log(2)),
        (-0.5, 80 - 40 * np.sqrt(2)),
        (2 / 3, 18 * (2 ** (2 / 3) - 1)),
        (-1, np.inf),
        (-1.5, np.inf),
        (-2, np.inf),
    ],
)
def test_zero_observations_use_limits_without_invalid_operations(power, expected):
    with np.errstate(divide="raise", invalid="raise"):
        actual = Chi2Statistic.do_execute_statistic(object(), [0, 10], [5, 5], power)
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize(
    ("observed", "expected", "message"),
    [
        ([], [], "nonempty 1D"),
        ([1, 2], [3], "same shape"),
        ([[1, 2]], [[1, 2]], "1D"),
        (1, 1, "1D"),
        ([np.nan, 1], [1, 1], "finite"),
        ([1, 1], [np.inf, 1], "finite"),
        ([-1, 3], [1, 1], "nonnegative"),
        ([0, 2], [0, 2], "positive"),
        ([1, 1], [-1, 3], "positive"),
        ([1, 2], [1, 3], "equal sums"),
        ([0, 0], [1, 1], "equal sums"),
    ],
)
def test_invalid_frequencies_are_rejected(observed, expected, message):
    with pytest.raises(ValueError, match=message):
        Chi2Statistic.do_execute_statistic(object(), observed, expected, 1)


@pytest.mark.parametrize("power", [np.nan, np.inf, -np.inf, [1, 2]])
def test_invalid_power_is_rejected(power):
    with pytest.raises(ValueError, match="finite scalar"):
        Chi2Statistic.do_execute_statistic(object(), [1, 1], [1, 1], power)


def test_frequency_total_tolerance_allows_roundoff():
    actual = Chi2Statistic.do_execute_statistic(object(), [2, 3], [2, 3 + 1e-12], 1)
    assert actual == pytest.approx(0, abs=1e-20)


@pytest.mark.parametrize("power", [-0.5, -1, -2])
@pytest.mark.parametrize("distribution", ["beta", "gamma", "uniform"])
def test_binned_statistics_inherit_zero_count_limits(distribution, power):
    if distribution == "beta":
        statistic = Chi2PearsonBetaGofStatistic(Beta.DEFAULT.parse({"a": 1, "b": 1}), lambda_=power)
    elif distribution == "gamma":
        statistic = CressieReadGammaGofStatistic(
            GammaDistributionDescriptor.DEFAULT.parse({"alfa": 1, "beta": 1}), power=power, bins=2
        )
    else:
        statistic = Chi2PearsonUniformGofStatistic(
            UniformDistributionDescriptor.DEFAULT.parse({"a": 0, "b": 1}), lambda_=power, bins=2
        )
    # Four equal observations occupy one of two equiprobable bins.
    sample = [0.2] * 4
    expected = 4 * (2 + (2 - np.sqrt(2)) ** 2) if power == -0.5 else np.inf
    with np.errstate(divide="raise", invalid="raise"):
        assert statistic.execute_statistic(sample) == pytest.approx(expected)


@pytest.mark.parametrize(("a", "k"), [(1, 1), (1, 2), (2, 3)])
def test_weibull_pearson_uses_counts_and_includes_both_tails(a, k):
    # Nine points in three quantile bins: counts [2, 3, 4], expected [3, 3, 3].
    probabilities = [0.001, 0.1, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999]
    sample = stats.exponweib.ppf(probabilities, a, k)
    statistic = Chi2PearsonExponentiatedWeibullGofStatistic(
        ExponentiatedWeibull.DEFAULT.parse({"exponent": a, "shape": k, "scale": 1})
    )
    reference = stats.chisquare([2, 3, 4], [3, 3, 3]).statistic
    assert statistic.execute_statistic(sample) == pytest.approx(reference)
    assert statistic.execute_statistic(sample[::-1]) == pytest.approx(reference)


def test_weibull_pearson_handles_empty_bins_and_rounded_cdf_endpoints():
    statistic = Chi2PearsonExponentiatedWeibullGofStatistic(
        ExponentiatedWeibull.DEFAULT.parse({"exponent": 1, "shape": 1, "scale": 1})
    )
    assert statistic.execute_statistic([0.0] * 4) == pytest.approx(4)
    assert statistic.execute_statistic([1000.0] * 4) == pytest.approx(4)
    assert statistic.execute_statistic([0.0, 0.0, 1000.0, 1000.0]) == pytest.approx(0)
    assert statistic.execute_statistic([1.0]) == pytest.approx(0)


@pytest.mark.parametrize("sample", [[], [[1, 2]], 1, [-1, 2], [np.nan], [np.inf]])
def test_weibull_pearson_rejects_invalid_samples(sample):
    with pytest.raises(ValueError, match="Sample"):
        Chi2PearsonExponentiatedWeibullGofStatistic(
            ExponentiatedWeibull.DEFAULT.parse({"exponent": 1, "shape": 1, "scale": 1})
        ).execute_statistic(sample)


@pytest.mark.parametrize("parameter", ["a", "k"])
@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, [1, 2]])
def test_weibull_pearson_rejects_invalid_parameters(parameter, value):
    with pytest.raises(ValueError, match="Invalid value for"):
        Chi2PearsonExponentiatedWeibullGofStatistic(
            ExponentiatedWeibull.DEFAULT.parse(
                {
                    "exponent": 1,
                    "shape": 1,
                    "scale": 1,
                    {"a": "exponent", "k": "shape"}[parameter]: value,
                }
            )
        ).execute_statistic([1, 2])
