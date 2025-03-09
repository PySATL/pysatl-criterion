import pytest as pytest

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


# TODO: actual test (7; 10)
@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([-0.5, 1.7, 1.2, 2.2, 0, -3.2768, 0.42], -0.37900874635568516),
        ([1.5, 2.7, -3.8, 4.6, -0.5, -0.6, 0.7, 0.8, -0.9, 10], -0.41),
    ],
)
@pytest.mark.skip(reason="fix test and check")
def test_ahs_exponentiality_criterion(data, result):
    statistic = AHSTestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_ahs_exponentiality_criterion_code():
    assert "AHS_EXPONENTIALITY_GOODNESS_OF_FIT" == AHSTestExp().code()


# TODO: actual test (7; 10)
@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.007564336567134994),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.00858038457884382),
    ],
)
@pytest.mark.skip(reason="fix test and check")
def test_atk_exponentiality_criterion(data, result):
    statistic = ATKTestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_atk_exponentiality_criterion_code():
    assert "ATK_EXPONENTIALITY_GOODNESS_OF_FIT" == ATKTestExp().code()


@pytest.mark.parametrize(  # TODO: actual test (7; 10)
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.17377394345044517),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.16232061118184815),
    ],
)
@pytest.mark.skip(reason="fix test and check")
def test_co_exponentiality_criterion(data, result):
    statistic = COTestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_co_exponentiality_criterion_code():
    assert "CO_EXPONENTIALITY_GOODNESS_OF_FIT" == COTestExp().code()


@pytest.mark.parametrize(  # TODO: actual test (7; 10)
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.17377394345044517),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.16232061118184815),
    ],
)
@pytest.mark.skip(reason="fix test and check")
def test_cvm_exponentiality_criterion(data, result):
    statistic = CVMTestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_cvm_exponentiality_criterion_code():
    assert "CVM_EXPONENTIALITY_GOODNESS_OF_FIT" == CVMTestExp().code()


@pytest.mark.parametrize(  # TODO: actual test (7; 10)
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.17377394345044517),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.16232061118184815),
    ],
)
@pytest.mark.skip(reason="fix test and check")
def test_dsp_exponentiality_criterion(data, result):
    statistic = DSPTestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_dsp_exponentiality_criterion_code():
    assert "DSP_EXPONENTIALITY_GOODNESS_OF_FIT" == DSPTestExp().code()


@pytest.mark.parametrize(  # TODO: actual test (7; 10)
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], -1.5476370552787166),
        ([1, 2, -3, 4, -5, -6, 7, 8, -9, 10], 50597.27324595228),
    ],
)
@pytest.mark.skip(reason="fix test and check")
def test_ep_exponentiality_criterion(data, result):
    statistic = EPTestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_ep_exponentiality_criterion_code():
    assert "EP_EXPONENTIALITY_GOODNESS_OF_FIT" == EPTestExp().code()


@pytest.mark.parametrize(  # TODO: actual test (7; 10)
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.17377394345044517),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.16232061118184815),
    ],
)
@pytest.mark.skip(reason="fix test and check")
def test_eps_exponentiality_criterion(data, result):
    statistic = EPSTestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_eps_exponentiality_criterion_code():
    assert "EPS_EXPONENTIALITY_GOODNESS_OF_FIT" == EPSTestExp().code()


@pytest.mark.parametrize(  # TODO: actual test (7; 10)
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.17377394345044517),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.16232061118184815),
    ],
)
@pytest.mark.skip(reason="fix test and check")
def test_fz_exponentiality_criterion(data, result):
    statistic = FZTestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_fz_exponentiality_criterion_code():
    assert "FZ_EXPONENTIALITY_GOODNESS_OF_FIT" == FZTestExp().code()


@pytest.mark.parametrize(  # TODO: actual test (7; 10)
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.17377394345044517),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.16232061118184815),
    ],
)
@pytest.mark.skip(reason="fix test and check")
def test_gdt_exponentiality_criterion(data, result):
    statistic = GDTestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_gdt_exponentiality_criterion_code():
    assert "GD_EXPONENTIALITY_GOODNESS_OF_FIT" == GDTestExp().code()


@pytest.mark.parametrize(  # TODO: actual test (7; 10)
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.17377394345044517),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.16232061118184815),
    ],
)
@pytest.mark.skip(reason="fix test and check")
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


@pytest.mark.parametrize(  # TODO: actual test (7; 10)
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.17377394345044517),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.16232061118184815),
    ],
)
@pytest.mark.skip(reason="fix test and check")
def test_hg1_exponentiality_criterion(data, result):
    statistic = HG1TestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_hg1_exponentiality_criterion_code():
    assert "HG1_EXPONENTIALITY_GOODNESS_OF_FIT" == HG1TestExp().code()


@pytest.mark.parametrize(  # TODO: actual test (7; 10)
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.17377394345044517),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.16232061118184815),
    ],
)
@pytest.mark.skip(reason="fix test and check")
def test_hg2_exponentiality_criterion(data, result):
    statistic = HG2TestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_hg2_exponentiality_criterion_code():
    assert "HG2_EXPONENTIALITY_GOODNESS_OF_FIT" == HG2TestExp().code()


@pytest.mark.parametrize(  # TODO: actual test (7; 10)
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.17377394345044517),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.16232061118184815),
    ],
)
@pytest.mark.skip(reason="fix test and check")
def test_hm_exponentiality_criterion(data, result):
    statistic = HMTestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_hm_exponentiality_criterion_code():
    assert "HM_EXPONENTIALITY_GOODNESS_OF_FIT" == HMTestExp().code()


@pytest.mark.parametrize(  # TODO: actual test (7; 10)
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.17377394345044517),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.16232061118184815),
    ],
)
@pytest.mark.skip(reason="fix test and check")
def test_hp_exponentiality_criterion(data, result):
    statistic = HPTestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_hp_exponentiality_criterion_code():
    assert "HP_EXPONENTIALITY_GOODNESS_OF_FIT" == HPTestExp().code()


@pytest.mark.parametrize(  # TODO: actual test (7; 10)
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.17377394345044517),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.16232061118184815),
    ],
)
@pytest.mark.skip(reason="fix test and check")
def test_kc_exponentiality_criterion(data, result):
    statistic = KCTestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_kc_exponentiality_criterion_code():
    assert "KC_EXPONENTIALITY_GOODNESS_OF_FIT" == KCTestExp().code()


@pytest.mark.parametrize(  # TODO: actual test (7; 10)
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.17378394345044517),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.16282061118184815),
    ],
)
@pytest.mark.skip(reason="fix test and check")
def test_km_exponentiality_criterion(data, result):
    statistic = KMTestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_km_exponentiality_criterion_code():
    assert "KM_EXPONENTIALITY_GOODNESS_OF_FIT" == KMTestExp().code()


@pytest.mark.parametrize(  # TODO: actual test (7; 10)
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.17377394345044517),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.16232061118184815),
    ],
)
@pytest.mark.skip(reason="fix test and check")
def test_ks_exponentiality_criterion(data, result):
    statistic = KSTestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_ks_exponentiality_criterion_code():
    assert "KS_EXPONENTIALITY_GOODNESS_OF_FIT" == KSTestExp().code()


@pytest.mark.parametrize(  # TODO: actual test (7; 10)
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.17377394345044517),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.16232061118184815),
    ],
)
@pytest.mark.skip(reason="fix test and check")
def test_lz_exponentiality_criterion(data, result):
    statistic = LZTestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_lz_exponentiality_criterion_code():
    assert "LZ_EXPONENTIALITY_GOODNESS_OF_FIT" == LZTestExp().code()


@pytest.mark.parametrize(  # TODO: actual test (7; 10)
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.17377394345044517),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.16232061118184815),
    ],
)
@pytest.mark.skip(reason="fix test and check")
def test_mn_exponentiality_criterion(data, result):
    statistic = MNTestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_mn_exponentiality_criterion_code():
    assert "MN_EXPONENTIALITY_GOODNESS_OF_FIT" == MNTestExp().code()


@pytest.mark.parametrize(  # TODO: actual test (7; 10)
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.17377394345044517),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.16232061118184815),
    ],
)
@pytest.mark.skip(reason="fix test and check")
def test_pt_exponentiality_criterion(data, result):
    statistic = PTTestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_pt_exponentiality_criterion_code():
    assert "PT_EXPONENTIALITY_GOODNESS_OF_FIT" == PTTestExp().code()


@pytest.mark.parametrize(  # TODO: actual test (7; 10)
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.17377394345044517),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.16232061118184815),
    ],
)
@pytest.mark.skip(reason="fix test and check")
def test_rs_exponentiality_criterion(data, result):
    statistic = RSTestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_rs_exponentiality_criterion_code():
    assert "RS_EXPONENTIALITY_GOODNESS_OF_FIT" == RSTestExp().code()


@pytest.mark.parametrize(  # TODO: actual test (7; 10)
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.17377394345044517),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.16232061118184815),
    ],
)
@pytest.mark.skip(reason="fix test and check")
def test_sw_exponentiality_criterion(data, result):
    statistic = SWTestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_sw_exponentiality_criterion_code():
    assert "SW_EXPONENTIALITY_GOODNESS_OF_FIT" == SWTestExp().code()


@pytest.mark.parametrize(  # TODO: actual test (7; 10)
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.17377394345044517),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.16232061118184815),
    ],
)
@pytest.mark.skip(reason="fix test and check")
def test_we_exponentiality_criterion(data, result):
    statistic = WETestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_we_exponentiality_criterion_code():
    assert "WE_EXPONENTIALITY_GOODNESS_OF_FIT" == WETestExp().code()


@pytest.mark.parametrize(  # TODO: actual test (7; 10)
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.17377394345044517),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.16232061118184815),
    ],
)
@pytest.mark.skip(reason="fix test and check")
def test_ww_exponentiality_criterion(data, result):
    statistic = WWTestExp().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_ww_exponentiality_criterion_code():
    assert "WW_EXPONENTIALITY_GOODNESS_OF_FIT" == WWTestExp().code()
