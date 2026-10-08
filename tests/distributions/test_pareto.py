import math

import numpy as np
import pytest
import scipy.stats as scipy_stats

from pysatl_criterion.distribution.distributions import ParetoDistributionDescriptor
from pysatl_criterion.statistics.alternative import AlternativeType
from pysatl_criterion.statistics.goodness_of_fit.pareto import (
    AbstractParetoGofStatistic,
    AndersonDarlingParetoGofStatistic,
    CramerVonMisesParetoGofStatistic,
    GreenwoodParetoGofStatistic,
    KolmogorovSmirnovParetoGofStatistic,
    LequesneKlParetoGofStatistic,
    MinToshiyukiParetoGofStatistic,
    ObradovicParetoGofStatistic,
)


def test_abstract_pareto_criterion_code():
    assert "PARETO_GOODNESS_OF_FIT" == AbstractParetoGofStatistic.code()


def test_abstract_pareto_distribution():
    from pysatl_criterion import DistributionType

    assert AbstractParetoGofStatistic.distribution().type() == DistributionType.PARETO


def test_abstract_pareto_invalid_shape():
    with pytest.raises(ValueError, match="Invalid value for shape"):
        KolmogorovSmirnovParetoGofStatistic(
            ParetoDistributionDescriptor.DEFAULT.parse({"shape": 0, "scale": 1})
        )
    with pytest.raises(ValueError, match="Invalid value for shape"):
        KolmogorovSmirnovParetoGofStatistic(
            ParetoDistributionDescriptor.DEFAULT.parse({"shape": -1, "scale": 1})
        )


def test_abstract_pareto_invalid_scale():
    with pytest.raises(ValueError, match="Invalid value for s"):
        KolmogorovSmirnovParetoGofStatistic(
            ParetoDistributionDescriptor.DEFAULT.parse({"shape": 2, "scale": 0})
        )
    with pytest.raises(ValueError, match="Invalid value for s"):
        KolmogorovSmirnovParetoGofStatistic(
            ParetoDistributionDescriptor.DEFAULT.parse({"shape": 2, "scale": -1})
        )


def test_abstract_pareto_hypothesis():
    stat = KolmogorovSmirnovParetoGofStatistic(
        ParetoDistributionDescriptor.DEFAULT.parse({"shape": 2.5, "scale": 1.5})
    )
    hypothesis = stat.hypothesis()
    assert hypothesis.params is not None
    assert hypothesis.parameters() == {"scale": 1.5, "shape": 2.5}


@pytest.mark.parametrize(
    ("data", "shape", "scale", "expected"),
    [
        ([1.5, 2.0, 2.5, 3.0, 4.0], 2.0, 1.0, 0.55556),
        ([1.1, 1.5, 2.0, 3.0, 5.0], 2.5, 1.0, 0.43711),
        ([2.0, 3.0, 4.0, 5.0, 6.0], 3.0, 2.0, 0.50370),
        ([1.1, 1.5, 2.0, 3.0, 5.0, 8.0], 2.0, 1.0, 0.41667),
        ([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 3.0, 2.0, 0.58929),
    ],
)
def test_ks_pareto_criterion(data, shape, scale, expected):
    statistic = KolmogorovSmirnovParetoGofStatistic(
        ParetoDistributionDescriptor.DEFAULT.parse({"shape": shape, "scale": scale})
    ).execute_statistic(data)
    assert expected == pytest.approx(statistic, rel=0.01)


def test_ks_pareto_criterion_code():
    assert (
        "KS_PARETO_GOODNESS_OF_FIT"
        == KolmogorovSmirnovParetoGofStatistic(
            ParetoDistributionDescriptor.DEFAULT.parse({"shape": 1, "scale": 1})
        ).code()
    )


def test_ks_pareto_alternatives():
    data = [1.5, 2.0, 2.5, 3.0, 4.0]

    stat_two = KolmogorovSmirnovParetoGofStatistic(
        ParetoDistributionDescriptor.DEFAULT.parse({"shape": 2.0, "scale": 1.0}),
        alternative_type=AlternativeType.TWO_TAILED,
    )
    stat_left = KolmogorovSmirnovParetoGofStatistic(
        ParetoDistributionDescriptor.DEFAULT.parse({"shape": 2.0, "scale": 1.0}),
        alternative_type=AlternativeType.LEFT,
    )
    stat_right = KolmogorovSmirnovParetoGofStatistic(
        ParetoDistributionDescriptor.DEFAULT.parse({"shape": 2.0, "scale": 1.0}),
        alternative_type=AlternativeType.RIGHT,
    )

    result_two = stat_two.execute_statistic(data)
    result_left = stat_left.execute_statistic(data)
    result_right = stat_right.execute_statistic(data)

    assert result_two >= result_left and result_two >= result_right


def test_ks_pareto_with_scipy():
    np.random.seed(42)
    sample = scipy_stats.pareto.rvs(b=2.0, scale=1.0, size=100)

    stat = KolmogorovSmirnovParetoGofStatistic(
        ParetoDistributionDescriptor.DEFAULT.parse({"shape": 2.0, "scale": 1.0})
    )
    our_result = stat.execute_statistic(sample)

    scipy_result = scipy_stats.kstest(sample, scipy_stats.pareto(b=2.0, scale=1.0).cdf)
    assert our_result == pytest.approx(scipy_result.statistic, rel=0.01)


def test_ks_pareto_non_positive_values():
    stat = KolmogorovSmirnovParetoGofStatistic(
        ParetoDistributionDescriptor.DEFAULT.parse({"shape": 2.0, "scale": 1.0})
    )

    for values in ([0, 1, 2, 3], [-1, 1, 2, 3]):
        with pytest.raises(ValueError, match="strictly positive"):
            stat.execute_statistic(values)


@pytest.mark.parametrize(
    ("data", "shape", "scale", "expected"),
    [
        ([1.5, 2.0, 2.5, 3.0, 4.0], 2.0, 1.0, 2.8515060009180715),
        ([1.1, 1.5, 2.0, 3.0, 5.0], 2.5, 1.0, 1.9340733401157904),
        ([2.0, 3.0, 4.0, 5.0, 6.0], 3.0, 2.0, math.inf),
        ([1.1, 1.5, 2.0, 3.0, 5.0, 8.0], 2.0, 1.0, 2.369839910981966),
    ],
)
def test_ad_pareto_criterion(data, shape, scale, expected):
    statistic = AndersonDarlingParetoGofStatistic(
        ParetoDistributionDescriptor.DEFAULT.parse({"shape": shape, "scale": scale})
    ).execute_statistic(data)
    if math.isinf(expected):
        assert math.isinf(statistic)
    else:
        assert expected == pytest.approx(statistic, rel=0.05)


def test_ad_pareto_criterion_code():
    assert (
        "AD_PARETO_GOODNESS_OF_FIT"
        == AndersonDarlingParetoGofStatistic(
            ParetoDistributionDescriptor.DEFAULT.parse({"shape": 1, "scale": 1})
        ).code()
    )


def test_ad_pareto_positive():
    np.random.seed(42)
    sample = scipy_stats.pareto.rvs(b=2.0, scale=1.0, size=50)
    stat = AndersonDarlingParetoGofStatistic(
        ParetoDistributionDescriptor.DEFAULT.parse({"shape": 2.0, "scale": 1.0})
    )
    result = stat.execute_statistic(sample)
    assert result > 0


@pytest.mark.parametrize(
    ("data", "shape", "scale", "expected"),
    [
        ([1.5, 2.0, 2.5, 3.0, 4.0], 2.0, 1.0, 0.5793827932098767),
        ([1.1, 1.5, 2.0, 3.0, 5.0], 2.5, 1.0, 0.30969962012912966),
        ([2.0, 3.0, 4.0, 5.0, 6.0], 3.0, 2.0, 0.38992868175582995),
        ([1.1, 1.5, 2.0, 3.0, 5.0, 8.0], 2.0, 1.0, 0.36855253145583944),
    ],
)
def test_cvm_pareto_criterion(data, shape, scale, expected):
    statistic = CramerVonMisesParetoGofStatistic(
        ParetoDistributionDescriptor.DEFAULT.parse({"shape": shape, "scale": scale})
    ).execute_statistic(data)
    assert expected == pytest.approx(statistic, rel=0.05)


def test_cvm_pareto_criterion_code():
    assert (
        "CVM_PARETO_GOODNESS_OF_FIT"
        == CramerVonMisesParetoGofStatistic(
            ParetoDistributionDescriptor.DEFAULT.parse({"shape": 1, "scale": 1})
        ).code()
    )


def test_cvm_pareto_positive():
    np.random.seed(42)
    sample = scipy_stats.pareto.rvs(b=2.0, scale=1.0, size=50)
    stat = CramerVonMisesParetoGofStatistic(
        ParetoDistributionDescriptor.DEFAULT.parse({"shape": 2.0, "scale": 1.0})
    )
    result = stat.execute_statistic(sample)
    assert result > 0


def test_mt_pareto_criterion_code():
    assert (
        "MT_PARETO_GOODNESS_OF_FIT"
        == MinToshiyukiParetoGofStatistic(
            ParetoDistributionDescriptor.DEFAULT.parse({"shape": 1, "scale": 1})
        ).code()
    )


def test_mt_pareto_positive():
    np.random.seed(42)
    sample = scipy_stats.pareto.rvs(b=2.0, scale=1.0, size=50)
    stat = MinToshiyukiParetoGofStatistic(
        ParetoDistributionDescriptor.DEFAULT.parse({"shape": 2.0, "scale": 1.0})
    )
    result = stat.execute_statistic(sample)
    assert result > 0


def test_mt_pareto_finite():
    np.random.seed(42)
    sample = scipy_stats.pareto.rvs(b=2.0, scale=1.0, size=50)
    stat = MinToshiyukiParetoGofStatistic(
        ParetoDistributionDescriptor.DEFAULT.parse({"shape": 2.0, "scale": 1.0})
    )
    result = stat.execute_statistic(sample)
    assert np.isfinite(result)


@pytest.mark.parametrize(
    ("data", "shape", "scale"),
    [
        ([1.1, 1.5, 2.0, 3.0, 5.0], 2.0, 1.0),
        ([1.5, 2.0, 2.5, 3.0, 4.0, 5.0], 2.5, 1.0),
        ([2.0, 3.0, 4.0, 5.0, 6.0], 3.0, 2.0),
    ],
)
def test_mt_pareto_finite_values(data, shape, scale):
    stat = MinToshiyukiParetoGofStatistic(
        ParetoDistributionDescriptor.DEFAULT.parse({"shape": shape, "scale": scale})
    )
    result = stat.execute_statistic(data)
    if min(data) == scale:
        assert result == np.inf
    else:
        assert np.isfinite(result)
    assert result > 0


def test_obradovic_pareto_criterion_code():
    assert (
        "OJM_Integral_PARETO_GOODNESS_OF_FIT"
        == ObradovicParetoGofStatistic(
            ParetoDistributionDescriptor.DEFAULT.parse({"scale": 1})
        ).code()
    )


def test_obradovic_pareto_finite():
    np.random.seed(42)
    sample = scipy_stats.pareto.rvs(b=2.0, scale=1.0, size=50)
    stat = ObradovicParetoGofStatistic(ParetoDistributionDescriptor.DEFAULT.parse({"scale": 1.0}))
    result = stat.execute_statistic(sample)
    assert np.isfinite(result)


def test_obradovic_pareto_small_sample():
    np.random.seed(42)
    sample = scipy_stats.pareto.rvs(b=2.0, scale=1.0, size=5)
    stat = ObradovicParetoGofStatistic(ParetoDistributionDescriptor.DEFAULT.parse({"scale": 1.0}))
    result = stat.execute_statistic(sample)
    assert np.isfinite(result)


def test_obradovic_pareto_too_small_sample():
    stat = ObradovicParetoGofStatistic(ParetoDistributionDescriptor.DEFAULT.parse({"scale": 1.0}))
    with pytest.raises(ValueError, match="At least 2 observations are required"):
        stat.execute_statistic([1.0])


def test_obradovic_pareto_non_positive_values():
    stat = ObradovicParetoGofStatistic(ParetoDistributionDescriptor.DEFAULT.parse({"scale": 1.0}))
    with pytest.raises(ValueError, match="Sample values must be strictly positive"):
        stat.execute_statistic([0, 1, 2, 3, 4])
    with pytest.raises(ValueError, match="Sample values must be strictly positive"):
        stat.execute_statistic([-1, 1, 2, 3, 4])


def test_greenwood_pareto_criterion_code():
    assert (
        "Greenwood_PARETO_GOODNESS_OF_FIT"
        == GreenwoodParetoGofStatistic(ParetoDistributionDescriptor.DEFAULT.parse({})).code()
    )


def test_greenwood_pareto_range():
    np.random.seed(42)
    sample = scipy_stats.pareto.rvs(b=2.0, scale=1.0, size=50)
    stat = GreenwoodParetoGofStatistic(ParetoDistributionDescriptor.DEFAULT.parse({}))
    result = stat.execute_statistic(sample)
    assert 0 <= result <= 1


def test_greenwood_pareto_constant_sample():
    stat = GreenwoodParetoGofStatistic(ParetoDistributionDescriptor.DEFAULT.parse({}))
    with pytest.raises(ValueError, match="nonconstant"):
        stat.execute_statistic(np.ones(10) * 2.0)


def test_greenwood_pareto_too_small_sample():
    stat = GreenwoodParetoGofStatistic(ParetoDistributionDescriptor.DEFAULT.parse({}))
    with pytest.raises(ValueError, match="At least 3 observations are required"):
        stat.execute_statistic([1.0])


def test_greenwood_pareto_non_positive_values():
    stat = GreenwoodParetoGofStatistic(ParetoDistributionDescriptor.DEFAULT.parse({}))
    with pytest.raises(ValueError, match="Sample values must be strictly positive"):
        stat.execute_statistic([0, 1, 2, 3, 4])
    with pytest.raises(ValueError, match="Sample values must be strictly positive"):
        stat.execute_statistic([-1, 1, 2, 3, 4])


def test_lequesne_pareto_criterion_code():
    assert (
        "Lequesne_KL_PARETO_GOODNESS_OF_FIT"
        == LequesneKlParetoGofStatistic(ParetoDistributionDescriptor.DEFAULT.parse({})).code()
    )


def test_lequesne_pareto_non_negative():
    np.random.seed(42)
    sample = scipy_stats.pareto.rvs(b=2.0, scale=1.0, size=50)
    stat = LequesneKlParetoGofStatistic(ParetoDistributionDescriptor.DEFAULT.parse({}))
    result = stat.execute_statistic(sample)
    assert result >= 0


def test_lequesne_pareto_custom_window():
    np.random.seed(42)
    sample = scipy_stats.pareto.rvs(b=2.0, scale=1.0, size=50)
    stat = LequesneKlParetoGofStatistic(ParetoDistributionDescriptor.DEFAULT.parse({}))

    result_default = stat.execute_statistic(sample)
    result_custom = stat.execute_statistic(sample, m=5)

    assert np.isfinite(result_custom)
    assert result_default >= 0
    assert result_custom >= 0


def test_lequesne_pareto_small_sample():
    stat = LequesneKlParetoGofStatistic(ParetoDistributionDescriptor.DEFAULT.parse({}))
    with pytest.raises(ValueError, match="At least 4 observations"):
        stat.execute_statistic([1.0, 2.0, 3.0])


def test_lequesne_pareto_non_positive_values():
    stat = LequesneKlParetoGofStatistic(ParetoDistributionDescriptor.DEFAULT.parse({}))
    with pytest.raises(ValueError, match="Sample values must be strictly positive"):
        stat.execute_statistic([0, 1, 2, 3, 4])
    with pytest.raises(ValueError, match="Sample values must be strictly positive"):
        stat.execute_statistic([-1, 1, 2, 3, 4])


def test_all_pareto_statistics_run():
    np.random.seed(42)
    sample = scipy_stats.pareto.rvs(b=2.0, scale=1.0, size=100)

    statistics = [
        KolmogorovSmirnovParetoGofStatistic(
            ParetoDistributionDescriptor.DEFAULT.parse({"shape": 2.0, "scale": 1.0})
        ),
        AndersonDarlingParetoGofStatistic(
            ParetoDistributionDescriptor.DEFAULT.parse({"shape": 2.0, "scale": 1.0})
        ),
        CramerVonMisesParetoGofStatistic(
            ParetoDistributionDescriptor.DEFAULT.parse({"shape": 2.0, "scale": 1.0})
        ),
        MinToshiyukiParetoGofStatistic(
            ParetoDistributionDescriptor.DEFAULT.parse({"shape": 2.0, "scale": 1.0})
        ),
        ObradovicParetoGofStatistic(ParetoDistributionDescriptor.DEFAULT.parse({"scale": 1.0})),
        GreenwoodParetoGofStatistic(ParetoDistributionDescriptor.DEFAULT.parse({})),
        LequesneKlParetoGofStatistic(ParetoDistributionDescriptor.DEFAULT.parse({})),
    ]

    for stat in statistics:
        result = stat.execute_statistic(sample)
        assert np.isfinite(result), f"{stat.code()} returned non-finite value"


def test_pareto_statistics_with_different_parameters():
    np.random.seed(42)
    sample = scipy_stats.pareto.rvs(b=3.0, scale=2.0, size=100)

    statistics = [
        KolmogorovSmirnovParetoGofStatistic(
            ParetoDistributionDescriptor.DEFAULT.parse({"shape": 3.0, "scale": 2.0})
        ),
        AndersonDarlingParetoGofStatistic(
            ParetoDistributionDescriptor.DEFAULT.parse({"shape": 3.0, "scale": 2.0})
        ),
        CramerVonMisesParetoGofStatistic(
            ParetoDistributionDescriptor.DEFAULT.parse({"shape": 3.0, "scale": 2.0})
        ),
        GreenwoodParetoGofStatistic(ParetoDistributionDescriptor.DEFAULT.parse({})),
    ]

    for stat in statistics:
        result = stat.execute_statistic(sample)
        assert np.isfinite(result), f"{stat.code()} returned non-finite value"


def test_pareto_distance_for_one_supported_alternative_sample():
    np.random.seed(42)
    pareto_data = scipy_stats.pareto.rvs(b=2.0, scale=1.0, size=1000)
    normal_data = 1 + np.abs(np.random.normal(2, 1, size=1000))

    stat = KolmogorovSmirnovParetoGofStatistic(
        ParetoDistributionDescriptor.DEFAULT.parse({"shape": 2.0, "scale": 1.0})
    )

    result_pareto = stat.execute_statistic(pareto_data)
    result_normal = stat.execute_statistic(normal_data)

    assert result_pareto < result_normal


def test_pareto_statistics_with_large_sample():
    np.random.seed(42)
    large_sample = scipy_stats.pareto.rvs(b=2.0, scale=1.0, size=10000)

    stat = KolmogorovSmirnovParetoGofStatistic(
        ParetoDistributionDescriptor.DEFAULT.parse({"shape": 2.0, "scale": 1.0})
    )
    result = stat.execute_statistic(large_sample)
    assert np.isfinite(result)
    assert result < 0.1


def test_pareto_statistics_extreme_values():
    data = [1.001, 1.01, 1.1, 10, 100, 1000]
    stat = KolmogorovSmirnovParetoGofStatistic(
        ParetoDistributionDescriptor.DEFAULT.parse({"shape": 2.0, "scale": 1.0})
    )
    result = stat.execute_statistic(data)
    assert np.isfinite(result)
    assert 0 <= result <= 1
