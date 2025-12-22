import inspect

import numpy as np
import pytest
import scipy.stats as scipy_stats

from pysatl_criterion.statistics.log_normal import (
    CramerVonMiseLogNormalGofStatistic,
    KLIntegralLogNormalGoFStatistic,
    KLSupremumLogNormalGoFStatistic,
    KolmogorovSmirnovLogNormalGofStatistic,
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


@pytest.mark.parametrize(
    ("data", "result"),
    [
        (
            [
                0.746,
                0.357,
                0.376,
                0.327,
                0.485,
                1.741,
                0.241,
                0.777,
                0.768,
                0.409,
                0.252,
                0.512,
                0.534,
                1.656,
                0.742,
                0.378,
                0.714,
                1.121,
                0.597,
                0.231,
                0.541,
                0.805,
                0.682,
                0.418,
                0.506,
                0.501,
                0.247,
                0.922,
                0.880,
                0.344,
                0.519,
                1.302,
                0.275,
                0.601,
                0.388,
                0.450,
                0.845,
                0.319,
                0.486,
                0.529,
                1.547,
                0.690,
                0.676,
                0.314,
                0.736,
                0.643,
                0.483,
                0.352,
                0.636,
                1.080,
            ],
            0.9521,  # значение из статьи
        ),
    ],
)
def test_kl_supremum_lognormal_criterion(data, result):
    statistic = KLSupremumLogNormalGoFStatistic().execute_statistic(data)

    assert result == pytest.approx(statistic, 0.0001)


def test_kl_supremum_lognormal_criterion_code():
    assert "KL_SUP_LOGNORMAL_GOODNESS_OF_FIT" == KLSupremumLogNormalGoFStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        (
            [
                0.746,
                0.357,
                0.376,
                0.327,
                0.485,
                1.741,
                0.241,
                0.777,
                0.768,
                0.409,
                0.252,
                0.512,
                0.534,
                1.656,
                0.742,
                0.378,
                0.714,
                1.121,
                0.597,
                0.231,
                0.541,
                0.805,
                0.682,
                0.418,
                0.506,
                0.501,
                0.247,
                0.922,
                0.880,
                0.344,
                0.519,
                1.302,
                0.275,
                0.601,
                0.388,
                0.450,
                0.845,
                0.319,
                0.486,
                0.529,
                1.547,
                0.690,
                0.676,
                0.314,
                0.736,
                0.643,
                0.483,
                0.352,
                0.636,
                1.080,
            ],
            1.28216,  # значение из статьи
        ),
    ],
)
def test_kl_integral_lognormal_criterion(data, result):
    statistic = KLIntegralLogNormalGoFStatistic().execute_statistic(data)

    assert result == pytest.approx(statistic, 0.01)


def test_kl_supremum_critical_value_n50():
    statistic = KLSupremumLogNormalGoFStatistic()
    n = 50
    alpha = 0.05
    expected = 2.1397  # Из статьи: c1,0.05(50) = 2.1397

    calculated = statistic.calculate_critical_value(n, alpha)

    assert abs(calculated - expected) < 1e-4


def test_kl_integral_critical_value_n50():
    statistic = KLIntegralLogNormalGoFStatistic()
    n = 50
    alpha = 0.05
    expected = 5.7369  # Из статьи: c2,0.05(50) = 5.7369

    calculated = statistic.calculate_critical_value(n, alpha)

    assert abs(calculated - expected) < 1e-4


def test_kl_supremum_critical_value_n90():
    statistic = KLSupremumLogNormalGoFStatistic()
    n = 90
    alpha = 0.05
    expected = 2.2708  # Из статьи: c1,0.05(90) = 2.2708

    calculated = statistic.calculate_critical_value(n, alpha)

    assert abs(calculated - expected) < 1e-4


def test_kl_integral_critical_value_n90():
    statistic = KLIntegralLogNormalGoFStatistic()
    n = 90
    alpha = 0.05
    expected = 6.6890  # Из статьи: c2,0.05(90) = 6.6890

    calculated = statistic.calculate_critical_value(n, alpha)

    assert abs(calculated - expected) < 1e-4


@pytest.mark.parametrize("n", [20, 40, 60, 80, 100])
def test_kl_sup_critical_values(n):
    stat = KLSupremumLogNormalGoFStatistic()

    # Test alpha = 0.05
    cv_05 = stat.calculate_critical_value(n, 0.05)
    assert cv_05 > 0

    # Test alpha = 0.01
    cv_01 = stat.calculate_critical_value(n, 0.01)
    assert cv_01 > cv_05


@pytest.mark.parametrize("n", [20, 40, 60, 80, 100])
def test_kl_int_critical_values(n):
    stat = KLIntegralLogNormalGoFStatistic()

    # Test alpha = 0.05
    cv_05 = stat.calculate_critical_value(n, 0.05)
    assert cv_05 > 0

    # Test alpha = 0.01
    cv_01 = stat.calculate_critical_value(n, 0.01)
    assert cv_01 > cv_05


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([-1.0, 0.5, 2.0, 3.0, -0.1], float("inf")),
        ([0.0, 1.0, 2.0, 3.0, 4.0], float("inf")),
    ],
)
def test_kl_supremum_non_positive_data(data, result):
    statistic = KLSupremumLogNormalGoFStatistic().execute_statistic(data)
    assert statistic == result


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([-1.0, 0.5, 2.0, 3.0, -0.1], float("inf")),
        ([0.0, 1.0, 2.0, 3.0, 4.0], float("inf")),
    ],
)
def test_kl_integral_non_positive_data(data, result):
    statistic = KLIntegralLogNormalGoFStatistic().execute_statistic(data)
    assert statistic == result


def test_kl_sup_lognormal_criterion_code():
    stat = KLSupremumLogNormalGoFStatistic()
    assert stat.code() == "KL_SUP_LOGNORMAL_GOODNESS_OF_FIT"


def test_kl_int_lognormal_criterion_code():
    stat = KLIntegralLogNormalGoFStatistic()
    assert stat.code() == "KL_INT_LOGNORMAL_GOODNESS_OF_FIT"


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

    assert ln_val == pytest.approx(n_val, rel=1e-5)


@pytest.mark.parametrize("log_normal_cls, normal_cls", DYNAMIC_CLASSES)
def test_dynamic_lognormal_negative_data_handling(log_normal_cls, normal_cls):
    data = [1.0, 2.0, -0.5, 3.0]
    stat = log_normal_cls(s=1, scale=1).execute_statistic(data)
    assert stat == float("inf")

    data_zero = [1.0, 2.0, 0.0, 3.0]
    stat_z = log_normal_cls(s=1, scale=1).execute_statistic(data_zero)
    assert stat_z == float("inf")


def test_dynamic_lognormal_example_code_method():
    from pysatl_criterion.statistics.log_normal import (
        LillieforsLogNormalGofStatistic,
        ShapiroWilkLogNormalGofStatistic,
    )

    assert "SW_LOGNORMAL_GOODNESS_OF_FIT" == ShapiroWilkLogNormalGofStatistic().code()
    assert "LILLIE_LOGNORMAL_GOODNESS_OF_FIT" == LillieforsLogNormalGofStatistic().code()
