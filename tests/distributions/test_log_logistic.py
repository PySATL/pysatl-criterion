import numpy as np
import pytest

from pysatl_criterion.statistics.goodness_of_fit.log_logistic import (
    AbstractLogLogisticGofStatistic,
    AndersonDarlingLogLogisticGofStatistic,
    Chi2PearsonLogLogisticGofStatistic,
    CramerVonMisesLogLogisticGofStatistic,
    KolmogorovSmirnovLogLogisticGofStatistic,
    MirvalievLogLogisticGofStatistic,
    NikulinLogLogisticGofStatistic,
)


_SAMPLE = [
    0.42,
    1.38,
    2.65,
    0.95,
    1.72,
    3.48,
    2.18,
    1.04,
    2.91,
    1.56,
]
_ALPHA = 1.5
_BETA = 2.0


def test_loglogistic_base_code():
    """Ensure LogLogistic GOF statistics expose the expected code identifier."""
    assert AbstractLogLogisticGofStatistic.code() == "LOG_LOGISTIC_GOODNESS_OF_FIT"


def test_loglogistic_positive_alpha_required():
    """Stat constructors should reject non-positive alpha parameters."""
    with pytest.raises(ValueError, match="Alpha parameter must be strictly greater than zero."):
        KolmogorovSmirnovLogLogisticGofStatistic(alpha=0.0, beta=1.0)


def test_loglogistic_positive_beta_required():
    """Stat constructors should reject non-positive beta parameters."""
    with pytest.raises(ValueError, match="Beta parameter must be strictly greater than zero."):
        KolmogorovSmirnovLogLogisticGofStatistic(alpha=1.0, beta=-1.0)


def test_kolmogorov_smirnov_loglogistic_statistic():
    """KS statistic for LogLogistic(alpha=1.5, beta=2.0)."""
    statistic = KolmogorovSmirnovLogLogisticGofStatistic(
        alpha=_ALPHA, beta=_BETA
    ).execute_statistic(_SAMPLE)
    assert statistic == pytest.approx(0.18628072957969863, rel=1e-9)


def test_anderson_darling_loglogistic_statistic():
    """AD statistic for LogLogistic(alpha=1.5, beta=2.0)."""
    statistic = AndersonDarlingLogLogisticGofStatistic(alpha=_ALPHA, beta=_BETA).execute_statistic(
        _SAMPLE
    )
    assert statistic == pytest.approx(0.42348000198919955, rel=1e-9)


def test_cramervonmises_loglogistic_statistic():
    """CVM statistic for LogLogistic(alpha=1.5, beta=2.0)."""
    statistic = CramerVonMisesLogLogisticGofStatistic(alpha=_ALPHA, beta=_BETA).execute_statistic(
        _SAMPLE
    )
    assert statistic == pytest.approx(0.06576214259270306, rel=1e-9)


def test_chi2_pearson_loglogistic_statistic():
    """Chi-square statistic for LogLogistic(alpha=1.5, beta=2.0)."""
    statistic = Chi2PearsonLogLogisticGofStatistic(
        bins=5,
        alpha=_ALPHA,
        beta=_BETA,
    ).execute_statistic(_SAMPLE)
    assert statistic == pytest.approx(2.0, rel=1e-9)


def test_nikulin_loglogistic_statistic():
    """Nikulin statistic for LogLogistic(alpha=1.5, beta=2.0)."""
    times = np.array(_SAMPLE)
    censored = np.ones_like(times)

    statistic = NikulinLogLogisticGofStatistic(n_intervals=5).execute_statistic((times, censored))
    assert statistic == pytest.approx(97.26713859725925, rel=1e-9)


def test_nikulin_loglogistic_with_censored_data():
    """Nikulin test with right-censored data."""
    np.random.seed(42)
    n = 50
    alpha_true, beta_true = 2.0, 1.5

    u = np.random.uniform(0, 1, n)
    times = alpha_true * (u / (1 - u)) ** (1 / beta_true)

    censoring_time = np.percentile(times, 70)
    censored = np.array([1 if t <= censoring_time else 0 for t in times])
    times_cens = np.minimum(times, censoring_time)

    statistic = NikulinLogLogisticGofStatistic(n_intervals=5).execute_statistic(
        (times_cens, censored)
    )

    assert isinstance(statistic, float)
    assert statistic >= 0
    assert not np.isnan(statistic)
    assert not np.isinf(statistic)


def test_mirvaliev_loglogistic_statistic():
    """Mirvaliev statistic for LogLogistic(alpha=1.5, beta=2.0)."""
    times = np.array(_SAMPLE)
    censored = np.ones_like(times)

    statistic = MirvalievLogLogisticGofStatistic(n_intervals=8).execute_statistic((times, censored))
    assert statistic == pytest.approx(787268848.3681092, rel=1e-9)


def test_mirvaliev_with_censored_data():
    """Mirvaliev test with right-censored data."""
    np.random.seed(42)
    n = 50
    alpha_true, beta_true = 2.0, 1.5

    u = np.random.uniform(0, 1, n)
    times = alpha_true * (u / (1 - u)) ** (1 / beta_true)

    censoring_time = np.percentile(times, 70)
    censored = np.array([1 if t <= censoring_time else 0 for t in times])
    times_cens = np.minimum(times, censoring_time)

    statistic = MirvalievLogLogisticGofStatistic(n_intervals=8).execute_statistic(
        (times_cens, censored)
    )

    assert isinstance(statistic, float)
    assert statistic >= 0
    assert not np.isnan(statistic)
    assert not np.isinf(statistic)


@pytest.mark.parametrize(
    ("stat_class", "expected_code"),
    [
        (KolmogorovSmirnovLogLogisticGofStatistic, "KS_LOG_LOGISTIC_GOODNESS_OF_FIT"),
        (
            AndersonDarlingLogLogisticGofStatistic,
            "AD_LOG_LOGISTIC_GOODNESS_OF_FIT",
        ),
        (
            CramerVonMisesLogLogisticGofStatistic,
            "CVM_LOG_LOGISTIC_GOODNESS_OF_FIT",
        ),
        (
            Chi2PearsonLogLogisticGofStatistic,
            "CHI2_PEARSON_LOG_LOGISTIC_GOODNESS_OF_FIT",
        ),
        (NikulinLogLogisticGofStatistic, "BN_GOF_LOG_LOGISTIC_GOODNESS_OF_FIT"),
        (MirvalievLogLogisticGofStatistic, "MIRVALIEV_LOG_LOGISTIC_GOODNESS_OF_FIT"),
    ],
)
def test_loglogistic_statistic_codes(stat_class, expected_code):
    """Ensure every LogLogistic statistic exposes a stable `code` identifier."""
    assert stat_class.code() == expected_code


def test_loglogistic_statistics_require_observations():
    """Statistics that rely on EDF should reject empty samples."""
    statistic = KolmogorovSmirnovLogLogisticGofStatistic(alpha=_ALPHA, beta=_BETA)
    with pytest.raises(ValueError):
        statistic.execute_statistic([])


def test_loglogistic_negative_data_handling():
    """Nikulin statistic should raise ValueError for non-positive data."""
    times = np.array([-1.0, 0.5, 2.0, 3.0, -0.1])
    censored = np.ones_like(times)

    statistic = NikulinLogLogisticGofStatistic(n_intervals=5)
    with pytest.raises(ValueError, match="All times must be positive."):
        statistic.execute_statistic((times, censored))


def test_loglogistic_zero_data_handling():
    """Nikulin statistic should raise ValueError for zero data."""
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    censored = np.ones_like(times)

    statistic = NikulinLogLogisticGofStatistic(n_intervals=5)
    with pytest.raises(ValueError, match="All times must be positive."):
        statistic.execute_statistic((times, censored))


def test_loglogistic_binned_statistics_validate_bin_count():
    """Binned statistics require at least two histogram bins."""
    with pytest.raises(ValueError, match="At least two bins"):
        Chi2PearsonLogLogisticGofStatistic(bins=1, alpha=_ALPHA, beta=_BETA)


def test_loglogistic_binned_statistics_require_sample():
    """Histogram-based tests must receive observations."""
    statistic = Chi2PearsonLogLogisticGofStatistic(bins=5, alpha=_ALPHA, beta=_BETA)
    with pytest.raises(ValueError, match="At least one observation"):
        statistic.execute_statistic([])


def test_mirvaliev_matrices_dimensions():
    """Check that Mirvaliev matrices have correct dimensions."""
    np.random.seed(42)
    n = 50
    alpha_true, beta_true = 2.0, 1.5

    u = np.random.uniform(0, 1, n)
    times = alpha_true * (u / (1 - u)) ** (1 / beta_true)
    censored = np.ones_like(times)

    stat = MirvalievLogLogisticGofStatistic(n_intervals=8)

    alpha_hat, beta_hat = stat._fit_mle(times, censored)
    bounds = stat._build_intervals(times, alpha_hat, beta_hat)
    U_j, e_j = stat._compute_frequencies(times, censored, bounds, alpha_hat, beta_hat)

    Z, A_inv, C, G, G_inv = stat._compute_mirvaliev_matrices(
        times, censored, U_j, e_j, bounds, alpha_hat, beta_hat
    )

    r = stat.n_intervals
    m = 2

    assert Z.shape == (r,)
    assert A_inv.shape == (r, r)
    assert C.shape == (m, r)
    assert G.shape == (m, m)
    assert G_inv.shape == (m, m)


def test_mirvaliev_statistic_non_negative():
    """Mirvaliev statistic must be non-negative."""
    np.random.seed(42)
    n = 100
    alpha_true, beta_true = 2.0, 1.5

    u = np.random.uniform(0, 1, n)
    times = alpha_true * (u / (1 - u)) ** (1 / beta_true)
    censored = np.ones_like(times)

    stat = MirvalievLogLogisticGofStatistic(n_intervals=8)
    S_n = stat.execute_statistic((times, censored))

    assert S_n >= 0
    assert not np.isnan(S_n)


def test_nikulin_statistic_non_negative():
    """Nikulin statistic must be non-negative."""
    np.random.seed(42)
    n = 100
    alpha_true, beta_true = 2.0, 1.5

    u = np.random.uniform(0, 1, n)
    times = alpha_true * (u / (1 - u)) ** (1 / beta_true)
    censored = np.ones_like(times)

    stat = NikulinLogLogisticGofStatistic(n_intervals=5)
    Y2 = stat.execute_statistic((times, censored))

    assert Y2 >= 0
    assert not np.isnan(Y2)


def test_nikulin_with_known_data():
    """Nikulin statistic with known data."""
    times = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
    censored = np.ones_like(times)

    stat = NikulinLogLogisticGofStatistic(n_intervals=5)
    Y2 = stat.execute_statistic((times, censored))

    assert Y2 > 0
    assert not np.isnan(Y2)
    assert not np.isinf(Y2)


def test_loglogistic_ks_single_observation():
    """KS statistic for single observation."""
    data = [1.0]
    stat = KolmogorovSmirnovLogLogisticGofStatistic(alpha=1.0, beta=1.0).execute_statistic(data)
    assert 0 <= stat <= 1


def test_loglogistic_ad_single_observation():
    """AD statistic for single observation."""
    data = [1.0]
    stat = AndersonDarlingLogLogisticGofStatistic(alpha=1.0, beta=1.0).execute_statistic(data)
    assert stat >= 0


def test_loglogistic_cvm_single_observation():
    """CVM statistic for single observation."""
    data = [1.0]
    stat = CramerVonMisesLogLogisticGofStatistic(alpha=1.0, beta=1.0).execute_statistic(data)
    assert stat >= 0


def test_loglogistic_chi2_single_observation():
    """Chi2 statistic for single observation."""
    data = [1.0]
    stat = Chi2PearsonLogLogisticGofStatistic(bins=5, alpha=1.0, beta=1.0).execute_statistic(data)
    assert stat >= 0
