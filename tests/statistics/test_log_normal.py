import inspect
import numpy as np
import pytest
import scipy.stats as scipy_stats

from pysatl_criterion.statistics.log_normal import (
    AbstractLogNormalGofStatistic,
    KolmogorovSmirnovLogNormalGofStatistic,
    CramerVonMiseLogNormalGofStatistic,
    QuesenberryMillerLogNormalGofStatistic,
)


@pytest.mark.parametrize(
    ("data", "result"),
    [
        (
            [
                0.279,
                1.455,
                0.323,
                2.981,
                0.732,
                0.654,
                1.121,
                0.485,
                2.341,
                0.892,
            ],
            0.1545,
        ),
    ],
)
def test_ks_lognormal_criterion(data, result):
    statistic = KolmogorovSmirnovLogNormalGofStatistic(s=1, scale=1).execute_statistic(data)
    assert result == pytest.approx(statistic, 0.001)


def test_ks_lognormal_criterion_code():
    assert "KS_LOGNORMAL_GOODNESS_OF_FIT" == KolmogorovSmirnovLogNormalGofStatistic().code()


def test_ks_lognormal_negative_data():
    np.random.seed(42)
    s, scale = 0.8, 1.5
    data = scipy_stats.lognorm.rvs(s=s, scale=scale, size=50)
    data = np.append(data, -1)

    our_stat = KolmogorovSmirnovLogNormalGofStatistic(s=s, scale=scale).execute_statistic(data)

    scipy_stat, _ = scipy_stats.kstest(data, "lognorm", args=(s, 0, scale))

    assert our_stat == pytest.approx(scipy_stat, 1e-10)


@pytest.mark.parametrize(
    ("data", "result"),
    [
        (
            [
                0.279,
                1.455,
                0.323,
                2.981,
                0.732,
                0.654,
                1.121,
                0.485,
                2.341,
                0.892,
            ],
            0.05776,
        ),
    ],
)
def test_cvm_lognormal_criterion(data, result):
    statistic = CramerVonMiseLogNormalGofStatistic(s=1, scale=1).execute_statistic(data)
    assert result == pytest.approx(statistic, 0.001)


def test_cvm_lognormal_criterion_code():
    assert "CVM_LOGNORMAL_GOODNESS_OF_FIT" == CramerVonMiseLogNormalGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        # 3 points mapping to CDF 0.25, 0.5, 0.75 for LogNorm(s=1, scale=1)
        (
            [0.50941628, 1.0, 1.96303108],
            0.4375,
        ),
        # 4 points mapping to CDF 0.2, 0.4, 0.6, 0.8
        (
            [0.431011, 0.776198, 1.288330, 2.320149],
            0.36,
        ),
    ],
)
def test_quesenberry_miller_lognormal_criterion(data, result):
    statistic = QuesenberryMillerLogNormalGofStatistic(s=1, scale=1).execute_statistic(data)
    assert result == pytest.approx(statistic, 0.0001)


def test_quesenberry_miller_lognormal_criterion_code():
    assert (
        "QUESENBERRY_MILLER_LOGNORMAL_GOODNESS_OF_FIT"
        == QuesenberryMillerLogNormalGofStatistic().code()
    )


def get_dynamic_lognormal_classes():
    from pysatl_criterion.statistics import log_normal as log_normal_module
    from pysatl_criterion.statistics import normal as normal_module

    excluded_classes = {
        "KolmogorovSmirnovLogNormalGofStatistic",
        "CramerVonMiseLogNormalGofStatistic",
        "QuesenberryMillerLogNormalGofStatistic",
        "AbstractLogNormalGofStatistic",
    }

    dynamic_pairs = []

    for name, obj in inspect.getmembers(log_normal_module, inspect.isclass):
        if not name.endswith("LogNormalGofStatistic"):
            continue

        if name in excluded_classes:
            continue

        normal_name = name.replace("LogNormal", "Normality")

        try:
            normal_cls = getattr(normal_module, normal_name)
        except AttributeError:
            continue

        if hasattr(obj, "execute_statistic") and hasattr(normal_cls, "execute_statistic"):
            dynamic_pairs.append((obj, normal_cls))

    return dynamic_pairs


DYNAMIC_CLASSES = get_dynamic_lognormal_classes()


@pytest.mark.parametrize("log_normal_cls, normal_cls", DYNAMIC_CLASSES)
def test_dynamic_lognormal_equivalence(log_normal_cls, normal_cls):
    np.random.seed(42)
    s = 0.8
    scale = 2.0
    z_data = np.random.normal(loc=0, scale=1, size=50)

    x_data = np.exp(z_data * s + np.log(scale))

    ln_stat_obj = log_normal_cls(s=s, scale=scale)
    ln_val = ln_stat_obj.execute_statistic(x_data)

    n_stat_obj = normal_cls()
    n_val = n_stat_obj.execute_statistic(z_data)

    print(log_normal_cls.code())
    assert ln_val == pytest.approx(n_val, rel=1e-5)


@pytest.mark.parametrize("log_normal_cls, normal_cls", DYNAMIC_CLASSES)
def test_dynamic_lognormal_negative_data_handling(log_normal_cls, normal_cls):
    data = [1.0, 2.0, -0.5, 3.0]
    stat = log_normal_cls(s=1, scale=1).execute_statistic(data)
    assert stat == float("inf")

    data_zero = [1.0, 2.0, 0.0, 3.0]
    stat_z = log_normal_cls(s=1, scale=1).execute_statistic(data_zero)
    print(log_normal_cls.code())
    assert stat_z == float("inf")


def test_dynamic_lognormal_example_code_method():
    from pysatl_criterion.statistics.log_normal import (
        ShapiroWilkLogNormalGofStatistic,
        LillieforsLogNormalGofStatistic,
    )

    print("!")
    assert "SW_LOGNORMAL_GOODNESS_OF_FIT" == ShapiroWilkLogNormalGofStatistic().code()
    assert "LILLIE_LOGNORMAL_GOODNESS_OF_FIT" == LillieforsLogNormalGofStatistic().code()
