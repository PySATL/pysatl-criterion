import pytest

from criterion import (
    ADWeibullTest,
    KSWeibullTest,
    LOSWeibullTestStatistic,
    MSFWeibullTestStatistic,
    OKWeibullTestStatistic,
    REJGWeibullTestStatistic,
    RSBWeibullTestStatistic,
    SBWeibullTestStatistic,
    SPPWeibullTestStatistic,
    ST1WeibullTestStatistic,
    ST2WeibullTestStatistic,
    TSWeibullTestStatistic,
)


@pytest.mark.parametrize(
    ("data", "result"),
    [
        # Weibull
        (
            [
                0.38323312,
                -1.10386561,
                0.75226465,
                -2.23024566,
                -0.27247827,
                0.95926434,
                0.42329541,
                -0.11820711,
                0.90892169,
                -0.29045373,
            ],
            0.686501978410317,
        ),
    ],
)
def test_ks_weibull_criterion(data, result):
    statistic = KSWeibullTest().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_ks_weibull_criterion_code():
    assert "KS_WEIBULL_GOODNESS_OF_FIT" == KSWeibullTest().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        # Weibull
        (
            [
                0.38323312,
                -1.10386561,
                0.75226465,
                -2.23024566,
                -0.27247827,
                0.95926434,
                0.42329541,
                -0.11820711,
                0.90892169,
                -0.29045373,
            ],
            0.686501978410317,
        ),
    ],
)
@pytest.mark.skip(reason="no way of currently testing this")
def test_ad_weibull_criterion(data, result):
    statistic = ADWeibullTest().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_ad_weibull_criterion_code():
    assert "AD_WEIBULL_GOODNESS_OF_FIT" == ADWeibullTest().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        # Weibull
        (
            [
                0.92559015,
                0.9993195,
                1.15193844,
                0.84272073,
                0.97535299,
                0.83745092,
                0.92161732,
                1.02751619,
                0.90079826,
                0.79149641,
            ],
            1.1845,
        ),
    ],
)
def test_los_weibull_criterion(data, result):
    statistic = LOSWeibullTestStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.1)


def test_los_weibull_criterion_code():
    assert "LOS_WEIBULL_GOODNESS_OF_FIT" == LOSWeibullTestStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        # Weibull
        (
            [
                0.92559015,
                0.9993195,
                1.15193844,
                0.84272073,
                0.97535299,
                0.83745092,
                0.92161732,
                1.02751619,
                0.90079826,
                0.79149641,
            ],
            0.67173,
        ),
    ],
)
def test_msf_weibull_criterion(data, result):
    statistic = MSFWeibullTestStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.01)


def test_msf_weibull_criterion_code():
    assert "MSF_WEIBULL_GOODNESS_OF_FIT" == MSFWeibullTestStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        # Weibull
        (
            [
                0.92559015,
                0.9993195,
                1.15193844,
                0.84272073,
                0.97535299,
                0.83745092,
                0.92161732,
                1.02751619,
                0.90079826,
                0.79149641,
            ],
            1.8927,
        ),
    ],
)
def test_ok_weibull_criterion(data, result):
    statistic = OKWeibullTestStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.0001)


def test_ok_weibull_criterion_code():
    assert "OK_WEIBULL_GOODNESS_OF_FIT" == OKWeibullTestStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        # Weibull
        (
            [
                0.92559015,
                0.9993195,
                1.15193844,
                0.84272073,
                0.97535299,
                0.83745092,
                0.92161732,
                1.02751619,
                0.90079826,
                0.79149641,
            ],
            0.84064,
        ),
    ],
)
def test_rejg_weibull_criterion(data, result):
    statistic = REJGWeibullTestStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_rejg_weibull_criterion_code():
    assert "REJG_WEIBULL_GOODNESS_OF_FIT" == REJGWeibullTestStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        # Weibull
        (
            [
                0.92559015,
                0.9993195,
                1.15193844,
                0.84272073,
                0.97535299,
                0.83745092,
                0.92161732,
                1.02751619,
                0.90079826,
                0.79149641,
            ],
            8.4755,
        ),
    ],
)
def test_rsb_weibull_criterion(data, result):
    statistic = RSBWeibullTestStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.0001)


def test_rsb_weibull_criterion_code():
    assert "RSB_WEIBULL_GOODNESS_OF_FIT" == RSBWeibullTestStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        # Weibull
        (
            [
                0.92559015,
                0.9993195,
                1.15193844,
                0.84272073,
                0.97535299,
                0.83745092,
                0.92161732,
                1.02751619,
                0.90079826,
                0.79149641,
            ],
            1.0644,
        ),
    ],
)
def test_sb_weibull_criterion(data, result):
    statistic = SBWeibullTestStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.01)


def test_sb_weibull_criterion_code():
    assert "SB_WEIBULL_GOODNESS_OF_FIT" == SBWeibullTestStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        # Weibull
        (
            [
                0.92559015,
                0.9993195,
                1.15193844,
                0.84272073,
                0.97535299,
                0.83745092,
                0.92161732,
                1.02751619,
                0.90079826,
                0.79149641,
            ],
            0.78178,
        ),
    ],
)
def test_spp_weibull_criterion(data, result):
    statistic = SPPWeibullTestStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.1)


def test_spp_weibull_criterion_code():
    assert "SPP_WEIBULL_GOODNESS_OF_FIT" == SPPWeibullTestStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        # Weibull
        (
            [
                0.92559015,
                0.9993195,
                1.15193844,
                0.84272073,
                0.97535299,
                0.83745092,
                0.92161732,
                1.02751619,
                0.90079826,
                0.79149641,
            ],
            1.1202,
        ),
    ],
)
def test_st1_weibull_criterion(data, result):
    statistic = ST1WeibullTestStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.001)


def test_st1_weibull_criterion_code():
    assert "ST1_WEIBULL_GOODNESS_OF_FIT" == ST1WeibullTestStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        # Weibull
        (
            [
                0.92559015,
                0.9993195,
                1.15193844,
                0.84272073,
                0.97535299,
                0.83745092,
                0.92161732,
                1.02751619,
                0.90079826,
                0.79149641,
            ],
            3.218,
        ),
    ],
)
def test_st2_weibull_criterion(data, result):
    statistic = ST2WeibullTestStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_st2_weibull_criterion_code():
    assert "ST2_WEIBULL_GOODNESS_OF_FIT" == ST2WeibullTestStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        # Weibull
        (
            [
                0.92559015,
                0.9993195,
                1.15193844,
                0.84272073,
                0.97535299,
                0.83745092,
                0.92161732,
                1.02751619,
                0.90079826,
                0.79149641,
            ],
            0.71566,
        ),
    ],
)
def test_ts_weibull_criterion(data, result):
    statistic = TSWeibullTestStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.01)


def test_ts_weibull_criterion_code():
    assert "TS_WEIBULL_GOODNESS_OF_FIT" == TSWeibullTestStatistic().code()
