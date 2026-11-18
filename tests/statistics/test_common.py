import numpy as np
import pytest
from scipy import stats

from pysatl_criterion.distribution import BetaDistributionDescriptor as Beta
from pysatl_criterion.distribution import (
    ExponentiatedWeibullDistributionDescriptor as ExponentiatedWeibull,
)
from pysatl_criterion.distribution import (
    GammaDistributionDescriptor,
)
from pysatl_criterion.hypothesis_testing.alternative_factory.alternative_factories import (
    AbstractAlternativeFactory,
)
from pysatl_criterion.statistics.alternative import AlternativeType
from pysatl_criterion.statistics.goodness_of_fit.binning import EqualProbabilityBins
from pysatl_criterion.statistics.goodness_of_fit.continuous.beta import (
    EbnerLiebenbergBetaGofStatistic,
    KolmogorovSmirnovBetaGofStatistic,
    KolmogorovSmirnovDMinusBetaGofStatistic,
    KolmogorovSmirnovDPlusBetaGofStatistic,
)
from pysatl_criterion.statistics.goodness_of_fit.continuous.exponentiated_weibull import (
    Chi2PearsonExponentiatedWeibullGofStatistic,
)
from pysatl_criterion.statistics.goodness_of_fit.continuous.gamma import (
    LiaoShimokawaGammaGofStatistic,
)
from pysatl_criterion.statistics.goodness_of_fit.edf import (
    AbstractKSDMinusStatistic,
    AbstractKSDPlusStatistic,
    AbstractKSStatistic,
)


@pytest.mark.parametrize(
    ("statistic_class", "scipy_direction"),
    [
        (AbstractKSDPlusStatistic, "greater"),
        (AbstractKSDMinusStatistic, "less"),
        (AbstractKSStatistic, "two-sided"),
    ],
)
@pytest.mark.parametrize(
    "sample", [[0.1, 0.15, 0.3, 0.65], [0.35, 0.7, 0.85, 0.9], [0.4], [0, 0.5, 0.5, 1]]
)
def test_compiled_ks_matches_scipy(statistic_class, scipy_direction, sample):
    sample = np.asarray(sample)
    actual = statistic_class.do_execute_statistic(stats.uniform.cdf(sample))
    reference = stats.kstest(sample, stats.uniform.cdf, alternative=scipy_direction)
    assert actual == pytest.approx(reference.statistic)
    assert statistic_class.alternative(object()).type() == AlternativeType.RIGHT
    assert statistic_class.do_execute_statistic.nopython_signatures


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
    cls = {
        AlternativeType.RIGHT: KolmogorovSmirnovDPlusBetaGofStatistic,
        AlternativeType.LEFT: KolmogorovSmirnovDMinusBetaGofStatistic,
        AlternativeType.TWO_TAILED: KolmogorovSmirnovBetaGofStatistic,
    }[direction]
    statistic = cls(Beta.DEFAULT.parse({"a": 1, "b": 1}))
    reference = stats.kstest(sample, stats.uniform.cdf, alternative=scipy_direction)
    assert statistic.execute_statistic(sample) == pytest.approx(reference.statistic)
    assert statistic.alternative().type() == AlternativeType.RIGHT


@pytest.mark.parametrize(
    "statistic",
    [
        cls(Beta.DEFAULT.parse({"a": 1, "b": 1}))
        for cls in (
            KolmogorovSmirnovBetaGofStatistic,
            KolmogorovSmirnovDPlusBetaGofStatistic,
            KolmogorovSmirnovDMinusBetaGofStatistic,
        )
    ]
    + [
        EbnerLiebenbergBetaGofStatistic(Beta.DEFAULT.parse({})),
        LiaoShimokawaGammaGofStatistic(
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


@pytest.mark.parametrize(("a", "k"), [(1, 1), (1, 2), (2, 3)])
def test_weibull_pearson_uses_counts_and_includes_both_tails(a, k):
    # Nine points in three quantile bins: counts [2, 3, 4], expected [3, 3, 3].
    probabilities = [0.001, 0.1, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999]
    sample = stats.exponweib.ppf(probabilities, a, k)
    statistic = Chi2PearsonExponentiatedWeibullGofStatistic(
        ExponentiatedWeibull.DEFAULT.parse({"exponent": a, "shape": k, "scale": 1}),
        binning=EqualProbabilityBins(3),
    )
    reference = stats.chisquare([2, 3, 4], [3, 3, 3]).statistic
    assert statistic.execute_statistic(sample) == pytest.approx(reference)
    assert statistic.execute_statistic(sample[::-1]) == pytest.approx(reference)


def test_weibull_pearson_handles_empty_bins_and_rounded_cdf_endpoints():
    statistic = Chi2PearsonExponentiatedWeibullGofStatistic(
        ExponentiatedWeibull.DEFAULT.parse({"exponent": 1, "shape": 1, "scale": 1}),
        binning=EqualProbabilityBins(2),
    )
    assert statistic.execute_statistic([0.0] * 4) == pytest.approx(4)
    assert statistic.execute_statistic([1000.0] * 4) == pytest.approx(4)
    assert statistic.execute_statistic([0.0, 0.0, 1000.0, 1000.0]) == pytest.approx(0)
    assert statistic.execute_statistic([1.0]) == pytest.approx(1)


@pytest.mark.parametrize("sample", [[], [[1, 2]], 1, [-1, 2], [np.nan], [np.inf]])
def test_weibull_pearson_rejects_invalid_samples(sample):
    with pytest.raises(ValueError, match="[Ss]ample|nonnegative"):
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
