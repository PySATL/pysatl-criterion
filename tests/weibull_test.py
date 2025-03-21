import pytest

from criterion import (
    AndersonDarlingWeibullGofStatistic,
    CrammerVonMisesWeibullGofStatistic,
    KolmogorovSmirnovWeibullGofStatistic,
    LillieforsWeibullGofStatistic,
    LOSWeibullGofStatistic,
    MinToshiyukiWeibullGofStatistic,
    MSFWeibullGofStatistic,
    OKWeibullGofStatistic,
    REJGWeibullGofStatistic,
    RSBWeibullGofStatistic,
    SBWeibullGofStatistic,
    SPPWeibullGofStatistic,
    ST1WeibullGofStatistic,
    ST2WeibullGofStatistic,
    TikuSinghWeibullGofStatistic,
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
    statistic = KolmogorovSmirnovWeibullGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_ks_weibull_criterion_code():
    assert "KS_WEIBULL_GOODNESS_OF_FIT" == KolmogorovSmirnovWeibullGofStatistic().code()


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
    statistic = AndersonDarlingWeibullGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_ad_weibull_criterion_code():
    assert "AD_WEIBULL_GOODNESS_OF_FIT" == AndersonDarlingWeibullGofStatistic().code()


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
    statistic = LOSWeibullGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.1)


def test_los_weibull_criterion_code():
    assert "LOS_WEIBULL_GOODNESS_OF_FIT" == LOSWeibullGofStatistic().code()


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
    statistic = MSFWeibullGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.01)


def test_msf_weibull_criterion_code():
    assert "MSF_WEIBULL_GOODNESS_OF_FIT" == MSFWeibullGofStatistic().code()


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
    statistic = OKWeibullGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.0001)


def test_ok_weibull_criterion_code():
    assert "OK_WEIBULL_GOODNESS_OF_FIT" == OKWeibullGofStatistic().code()


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
    statistic = REJGWeibullGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_rejg_weibull_criterion_code():
    assert "REJG_WEIBULL_GOODNESS_OF_FIT" == REJGWeibullGofStatistic().code()


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
    statistic = RSBWeibullGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.0001)


def test_rsb_weibull_criterion_code():
    assert "RSB_WEIBULL_GOODNESS_OF_FIT" == RSBWeibullGofStatistic().code()


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
    statistic = SBWeibullGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.01)


def test_sb_weibull_criterion_code():
    assert "SB_WEIBULL_GOODNESS_OF_FIT" == SBWeibullGofStatistic().code()


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
    statistic = SPPWeibullGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.1)


def test_spp_weibull_criterion_code():
    assert "SPP_WEIBULL_GOODNESS_OF_FIT" == SPPWeibullGofStatistic().code()


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
    statistic = ST1WeibullGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.001)


def test_st1_weibull_criterion_code():
    assert "ST1_WEIBULL_GOODNESS_OF_FIT" == ST1WeibullGofStatistic().code()


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
    statistic = ST2WeibullGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_st2_weibull_criterion_code():
    assert "ST2_WEIBULL_GOODNESS_OF_FIT" == ST2WeibullGofStatistic().code()


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
    statistic = TikuSinghWeibullGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.01)


def test_mt_weibull_criterion_code():
    assert "MT_WEIBULL_GOODNESS_OF_FIT" == MinToshiyukiWeibullGofStatistic().code()


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
@pytest.mark.skip(reason="no way of currently testing this")
def test_mt_weibull_criterion(data, result):
    statistic = MinToshiyukiWeibullGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_lillie_weibull_criterion_code():
    assert "LILLIE_WEIBULL_GOODNESS_OF_FIT" == LillieforsWeibullGofStatistic().code()


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
            0.5468,
        ),
    ],
)
def test_lillie_weibull_criterion(data, result):
    statistic = LillieforsWeibullGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.001)


def test_cvm_weibull_criterion_code():
    assert "CVM_WEIBULL_GOODNESS_OF_FIT" == CrammerVonMisesWeibullGofStatistic().code()


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
            0.745,
        ),
    ],
)
def test_cvm_weibull_criterion(data, result):
    statistic = CrammerVonMisesWeibullGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.001)
