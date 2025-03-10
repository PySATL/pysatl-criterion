import pytest as pytest
from scipy.stats import kstest

from criterion import (
    AHSTestExp,
    ATKTestExp,
    HG1TestExp,
    HG2TestExp,
    HMTestExp,
    HPTestExp,
    KCTestExp,
    KMTestExp,
    KSTestExp,
    LZTestExp,
    MNTestExp,
    PTTestExp,
    RSTestExp,
    SWTestExp,
    WETestExp,
    WWTestExp,
)
from criterion.exponent import (
    AbstractExponentialityTestStatistic,
    COTestExp,
    CVMTestExp,
    DSPTestExp,
    EPSTestExp,
    EPTestExp,
    FZTestExp,
    GDTestExp,
    GiniTestExp,
    GraphEdgesNumberExpTest,
    GraphMaxDegreeExpTest,
)


def test_abstract_exponentiality_criterion_code():
    assert "EXPONENTIALITY_GOODNESS_OF_FIT" == AbstractExponentialityTestStatistic.code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([-0.5, 1.7, 1.2, 2.2, 0, -3.2768, 0.42], -0.37900874635568516),
        ([1.5, 2.7, -3.8, 4.6, -0.5, -0.6, 0.7, 0.8, -0.9, 10], -0.41),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.33),
    ],
)
def test_ahs_exponentiality_criterion(data, result):
    statistic = AHSTestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_ahs_exponentiality_criterion_code():
    assert "AHS_EXPONENTIALITY_GOODNESS_OF_FIT" == AHSTestExp().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.0075643),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.0085804),
    ],
)
def test_atk_exponentiality_criterion(data, result):
    statistic = ATKTestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_atk_exponentiality_criterion_code():
    assert "ATK_EXPONENTIALITY_GOODNESS_OF_FIT" == ATKTestExp().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 4.8636),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 6.5439),
    ],
)
def test_co_exponentiality_criterion(data, result):
    statistic = COTestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_co_exponentiality_criterion_code():
    assert "CO_EXPONENTIALITY_GOODNESS_OF_FIT" == COTestExp().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.12851),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.15712),
    ],
)
def test_cvm_exponentiality_criterion(data, result):
    statistic = CVMTestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.001)


def test_cvm_exponentiality_criterion_code():
    assert "CVM_EXPONENTIALITY_GOODNESS_OF_FIT" == CVMTestExp().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.78571),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.78889),
    ],
)
def test_dsp_exponentiality_criterion(data, result):
    statistic = DSPTestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_dsp_exponentiality_criterion_code():
    assert "DSP_EXPONENTIALITY_GOODNESS_OF_FIT" == DSPTestExp().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], -1.5476370552787166),
        ([1, 2, -3, 4, -5, -6, 7, 8, -9, 10], 50597.27324595228),
    ],
)
def test_ep_exponentiality_criterion(data, result):
    statistic = EPTestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_ep_exponentiality_criterion_code():
    assert "EP_EXPONENTIALITY_GOODNESS_OF_FIT" == EPTestExp().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 1.9806),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3.2841),
    ],
)
def test_eps_exponentiality_criterion(data, result):
    statistic = EPSTestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.001)


def test_eps_exponentiality_criterion_code():
    assert "EPS_EXPONENTIALITY_GOODNESS_OF_FIT" == EPSTestExp().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.30743),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.35194),
    ],
)
def test_fz_exponentiality_criterion(data, result):
    statistic = FZTestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.001)


def test_fz_exponentiality_criterion_code():
    assert "FZ_EXPONENTIALITY_GOODNESS_OF_FIT" == FZTestExp().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 2.75),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 2.6667),
    ],
)
def test_gdt_exponentiality_criterion(data, result):
    statistic = GDTestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.001)


def test_gdt_exponentiality_criterion_code():
    assert "GD_EXPONENTIALITY_GOODNESS_OF_FIT" == GDTestExp().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.33333),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.33333),
    ],
)
def test_gini_exponentiality_criterion(data, result):
    statistic = GiniTestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_gini_exponentiality_criterion_code():
    assert "GINI_EXPONENTIALITY_GOODNESS_OF_FIT" == GiniTestExp().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([0.713, 0.644, 2.625, 0.740, 0.501, 0.185, 0.982, 1.028, 1.152, 0.267], 12),
        ([0.039, 3.036, 0.626, 1.107, 0.139, 1.629, 0.050, 0.118, 0.978, 2.699], 7),
    ],
)
def test_graph_edges_number_exponentiality_criterion(data, result):
    statistic = GraphEdgesNumberExpTest().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_graph_edges_number_exponentiality_criterion_code():
    assert "EdgesNumber_GOODNESS_OF_FIT" == GraphEdgesNumberExpTest().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([0.713, 0.644, 2.625, 0.740, 0.501, 0.185, 0.982, 1.028, 1.152, 0.267], 4),
        ([0.039, 3.036, 0.626, 1.107, 0.139, 1.629, 0.050, 0.118, 0.978, 2.699], 3),
    ],
)
def test_graph_max_degree_exponentiality_criterion(data, result):
    statistic = GraphMaxDegreeExpTest().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_graph_max_degree_exponentiality_criterion_code():
    assert "MaxDegree_GOODNESS_OF_FIT" == GraphMaxDegreeExpTest().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 3.1384),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 4.6125),
    ],
)
def test_hg1_exponentiality_criterion(data, result):
    statistic = HG1TestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.001)


def test_hg1_exponentiality_criterion_code():
    assert "HG1_EXPONENTIALITY_GOODNESS_OF_FIT" == HG1TestExp().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 11.81),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 26.207),
    ],
)
def test_hg2_exponentiality_criterion(data, result):
    statistic = HG2TestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.001)


def test_hg2_exponentiality_criterion_code():
    assert "HG2_EXPONENTIALITY_GOODNESS_OF_FIT" == HG2TestExp().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 1),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1),
    ],
)
def test_hm_exponentiality_criterion(data, result):
    statistic = HMTestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_hm_exponentiality_criterion_code():
    assert "HM_EXPONENTIALITY_GOODNESS_OF_FIT" == HMTestExp().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.12381),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.13889),
    ],
)
def test_hp_exponentiality_criterion(data, result):
    statistic = HPTestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_hp_exponentiality_criterion_code():
    assert "HP_EXPONENTIALITY_GOODNESS_OF_FIT" == HPTestExp().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 2.4073),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 2.592),
    ],
)
def test_kc_exponentiality_criterion(data, result):
    statistic = KCTestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.001)


def test_kc_exponentiality_criterion_code():
    assert "KC_EXPONENTIALITY_GOODNESS_OF_FIT" == KCTestExp().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.13948),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.1238),
    ],
)
def test_km_exponentiality_criterion(data, result):
    statistic = KMTestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.001)


def test_km_exponentiality_criterion_code():
    assert "KM_EXPONENTIALITY_GOODNESS_OF_FIT" == KMTestExp().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.72180),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.764664),
    ],
)
def test_ks_exponentiality_criterion(data, result):
    kstest(data, "expon")
    statistic = KSTestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.001)


def test_ks_exponentiality_criterion_code():
    assert "KS_EXPONENTIALITY_GOODNESS_OF_FIT" == KSTestExp().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.21429),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.27273),
    ],
)
def test_lz_exponentiality_criterion(data, result):
    statistic = LZTestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.001)


def test_lz_exponentiality_criterion_code():
    assert "LZ_EXPONENTIALITY_GOODNESS_OF_FIT" == LZTestExp().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.4088),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.38291),
    ],
)
def test_mn_exponentiality_criterion(data, result):
    statistic = MNTestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_mn_exponentiality_criterion_code():
    assert "MN_EXPONENTIALITY_GOODNESS_OF_FIT" == MNTestExp().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.21429),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.22727),
    ],
)
def test_pt_exponentiality_criterion(data, result):
    statistic = PTTestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.001)


def test_pt_exponentiality_criterion_code():
    assert "PT_EXPONENTIALITY_GOODNESS_OF_FIT" == PTTestExp().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.095238),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.091667),
    ],
)
def test_rs_exponentiality_criterion(data, result):
    statistic = RSTestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_rs_exponentiality_criterion_code():
    assert "RS_EXPONENTIALITY_GOODNESS_OF_FIT" == RSTestExp().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.375),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.27273),
    ],
)
def test_sw_exponentiality_criterion(data, result):
    statistic = SWTestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_sw_exponentiality_criterion_code():
    assert "SW_EXPONENTIALITY_GOODNESS_OF_FIT" == SWTestExp().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.035714),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.027273),
    ],
)
def test_we_exponentiality_criterion(data, result):
    statistic = WETestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, abs=0.01)


def test_we_exponentiality_criterion_code():
    assert "WE_EXPONENTIALITY_GOODNESS_OF_FIT" == WETestExp().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 7),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 10),
    ],
)
def test_ww_exponentiality_criterion(data, result):
    statistic = WWTestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_ww_exponentiality_criterion_code():
    assert "WW_EXPONENTIALITY_GOODNESS_OF_FIT" == WWTestExp().code()
