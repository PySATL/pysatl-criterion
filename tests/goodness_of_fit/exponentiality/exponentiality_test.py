import numpy as np
import pytest as pytest
from numpy import inf

from pysatl_criterion import (
    AhsanullahExponentialityGofStatistic,
    AtkinsonExponentialityGofStatistic,
    HarrisExponentialityGofStatistic,
    HegazyGreen1ExponentialityGofStatistic,
    HegazyGreen2ExponentialityGofStatistic,
    HollanderProshanExponentialityGofStatistic,
    KimberMichaelExponentialityGofStatistic,
    KocharExponentialityGofStatistic,
    KolmogorovSmirnovExponentialityGofStatistic,
    LorenzExponentialityGofStatistic,
    MoranExponentialityGofStatistic,
    PietraExponentialityGofStatistic,
    RossbergExponentialityGofStatistic,
    ShapiroWilkExponentialityGofStatistic,
    WeExponentialityGofStatistic,
    WongWongExponentialityGofStatistic,
)
from pysatl_criterion.exponent import (
    AbstractExponentialityGofStatistic,
    CoxOakesExponentialityGofStatistic,
    CramerVonMisesExponentialityGofStatistic,
    DeshpandeExponentialityGofStatistic,
    EppsPulleyExponentialityGofStatistic,
    EpsteinExponentialityGofStatistic,
    FroziniExponentialityGofStatistic,
    GiniExponentialityGofStatistic,
    GnedenkoExponentialityGofStatistic,
    GraphEdgesNumberExponentialityGofStatistic,
    GraphMaxDegreeExponentialityGofStatistic,
)


def test_abstract_exponentiality_criterion_code():
    assert "EXPONENTIALITY_GOODNESS_OF_FIT" == AbstractExponentialityGofStatistic.code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([-0.5, 1.7, 1.2, 2.2, 0, -3.2768, 0.42], -0.37900874635568516),
        ([1.5, 2.7, -3.8, 4.6, -0.5, -0.6, 0.7, 0.8, -0.9, 10], -0.41),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.33),
        ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0),  # Zero exception test
        ([-4, -1, -6, -8, -4, -2, 0, -2, 0, -3], -0.811),  # Negative values test
    ],
)
def test_ahs_exponentiality_criterion(data, result):
    statistic = AhsanullahExponentialityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_ahs_exponentiality_criterion_code():
    assert "AHS_EXPONENTIALITY_GOODNESS_OF_FIT" == AhsanullahExponentialityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.0075643),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.0085804),
    ],
)
def test_atk_exponentiality_criterion(data, result):
    statistic = AtkinsonExponentialityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


@pytest.mark.parametrize(
    "data",
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Zero exception test
        [-4, -1, -6, -8, -4, -2, 0, -2, 0, -3],  # Negative values test
    ],
)
def test_nan_atk_exponentiality_criterion(data):
    statistic = AtkinsonExponentialityGofStatistic().execute_statistic(data)
    assert np.isnan(statistic)


def test_atk_exponentiality_criterion_code():
    assert "ATK_EXPONENTIALITY_GOODNESS_OF_FIT" == AtkinsonExponentialityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 4.8636),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 6.5439),
        ([-4, -1, -6, -8, -4, -2, 0, -2, 0, -3], -inf),  # Negative values test
    ],
)
def test_co_exponentiality_criterion(data, result):
    statistic = CoxOakesExponentialityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


@pytest.mark.parametrize(
    "data",
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Zero exception test
    ],
)
def test_nan_co_exponentiality_criterion(data):
    statistic = CoxOakesExponentialityGofStatistic().execute_statistic(data)
    assert np.isnan(statistic)


def test_co_exponentiality_criterion_code():
    assert "CO_EXPONENTIALITY_GOODNESS_OF_FIT" == CoxOakesExponentialityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.12851),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.15712),
        ([-4, -1, -6, -8, -4, -2, 0, -2, 0, -3], 0.06943545873682137),  # Negative values test
    ],
)
def test_cvm_exponentiality_criterion(data, result):
    statistic = CramerVonMisesExponentialityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.001)


@pytest.mark.parametrize(
    "data",
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Zero exception test
    ],
)
def test_nan_cvm_exponentiality_criterion(data):
    statistic = CramerVonMisesExponentialityGofStatistic().execute_statistic(data)
    assert np.isnan(statistic)


def test_cvm_exponentiality_criterion_code():
    assert "CVM_EXPONENTIALITY_GOODNESS_OF_FIT" == CramerVonMisesExponentialityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.78571),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.78889),
        ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0),  # Zero exception test
        ([-4, -1, -6, -8, -4, -2, 0, -2, 0, -3], 0.28888888888888886),  # Negative values test
    ],
)
def test_dsp_exponentiality_criterion(data, result):
    statistic = DeshpandeExponentialityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_dsp_exponentiality_criterion_code():
    assert "DSP_EXPONENTIALITY_GOODNESS_OF_FIT" == DeshpandeExponentialityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], -1.5476370552787166),
        ([1, 2, -3, 4, -5, -6, 7, 8, -9, 10], 50597.27324595228),
        ([-4, -1, -6, -8, -4, -2, 0, -2, 0, -3], -0.3434045153356423),  # Negative values test
    ],
)
def test_ep_exponentiality_criterion(data, result):
    statistic = EppsPulleyExponentialityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


@pytest.mark.parametrize(
    "data",
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Zero exception test
    ],
)
def test_nan_ep_exponentiality_criterion(data):
    statistic = EppsPulleyExponentialityGofStatistic().execute_statistic(data)
    assert np.isnan(statistic)


def test_ep_exponentiality_criterion_code():
    assert "EP_EXPONENTIALITY_GOODNESS_OF_FIT" == EppsPulleyExponentialityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 1.9806),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3.2841),
    ],
)
def test_eps_exponentiality_criterion(data, result):
    statistic = EpsteinExponentialityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.001)


@pytest.mark.parametrize(
    "data",
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Zero exception test
        [-4, -1, -6, -8, -4, -2, 0, -2, 0, -3],  # Negative values test
    ],
)
def test_nan_eps_exponentiality_criterion(data):
    statistic = EpsteinExponentialityGofStatistic().execute_statistic(data)
    assert np.isnan(statistic)


def test_eps_exponentiality_criterion_code():
    assert "EPS_EXPONENTIALITY_GOODNESS_OF_FIT" == EpsteinExponentialityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.30743),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.35194),
        ([-4, -1, -6, -8, -4, -2, 0, -2, 0, -3], 1.6265090069613914),  # Negative values test
    ],
)
def test_fz_exponentiality_criterion(data, result):
    statistic = FroziniExponentialityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.001)


@pytest.mark.parametrize(
    "data",
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Zero exception test
    ],
)
def test_nan_fz_exponentiality_criterion(data):
    statistic = FroziniExponentialityGofStatistic().execute_statistic(data)
    assert np.isnan(statistic)


def test_fz_exponentiality_criterion_code():
    assert "FZ_EXPONENTIALITY_GOODNESS_OF_FIT" == FroziniExponentialityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 2.75),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 2.6667),
        ([-4, -1, -6, -8, -4, -2, 0, -2, 0, -3], 7.285714285714286),  # Negative values test
    ],
)
def test_gd_exponentiality_criterion(data, result):
    statistic = GnedenkoExponentialityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.001)


@pytest.mark.parametrize(
    "data",
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Zero exception test
    ],
)
def test_nan_gd_exponentiality_criterion(data):
    statistic = GnedenkoExponentialityGofStatistic().execute_statistic(data)
    assert np.isnan(statistic)


def test_gd_exponentiality_criterion_code():
    assert "GD_EXPONENTIALITY_GOODNESS_OF_FIT" == GnedenkoExponentialityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.33333),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.33333),
        ([-4, -1, -6, -8, -4, -2, 0, -2, 0, -3], -0.5037037037037037),  # Negative values tes
    ],
)
def test_gini_exponentiality_criterion(data, result):
    statistic = GiniExponentialityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


@pytest.mark.parametrize(
    "data",
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Zero exception test
    ],
)
def test_nan_gini_exponentiality_criterion(data):
    statistic = GiniExponentialityGofStatistic().execute_statistic(data)
    assert np.isnan(statistic)


def test_gini_exponentiality_criterion_code():
    assert "GINI_EXPONENTIALITY_GOODNESS_OF_FIT" == GiniExponentialityGofStatistic().code()


# TODO: more graph tests?
@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([0.713, 0.644, 2.625, 0.740, 0.501, 0.185, 0.982, 1.028, 1.152, 0.267], 12),
        ([0.039, 3.036, 0.626, 1.107, 0.139, 1.629, 0.050, 0.118, 0.978, 2.699], 7),
    ],
)
def test_graph_edges_number_exponentiality_criterion(data, result):
    statistic = GraphEdgesNumberExponentialityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_graph_edges_number_exponentiality_criterion_code():
    assert "EdgesNumber_GOODNESS_OF_FIT" == GraphEdgesNumberExponentialityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([0.713, 0.644, 2.625, 0.740, 0.501, 0.185, 0.982, 1.028, 1.152, 0.267], 4),
        ([0.039, 3.036, 0.626, 1.107, 0.139, 1.629, 0.050, 0.118, 0.978, 2.699], 3),
    ],
)
def test_graph_max_degree_exponentiality_criterion(data, result):
    statistic = GraphMaxDegreeExponentialityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_graph_max_degree_exponentiality_criterion_code():
    assert "MaxDegree_GOODNESS_OF_FIT" == GraphMaxDegreeExponentialityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 3.1384),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 4.6125),
        ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0.8874540154908189),  # Zero exception test
        ([-4, -1, -6, -8, -4, -2, 0, -2, 0, -3], 3.8874540154908193),  # Negative values test
    ],
)
def test_hg1_exponentiality_criterion(data, result):
    statistic = HegazyGreen1ExponentialityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.001)


def test_hg1_exponentiality_criterion_code():
    assert "HG1_EXPONENTIALITY_GOODNESS_OF_FIT" == HegazyGreen1ExponentialityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 11.81),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 26.207),
        ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 1.2711662185480854),  # Zero exception test
        ([-4, -1, -6, -8, -4, -2, 0, -2, 0, -3], 18.62437981246421),  # Negative values test
    ],
)
def test_hg2_exponentiality_criterion(data, result):
    statistic = HegazyGreen2ExponentialityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.001)


def test_hg2_exponentiality_criterion_code():
    assert "HG2_EXPONENTIALITY_GOODNESS_OF_FIT" == HegazyGreen2ExponentialityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 1),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1),
        ([-4, -1, -6, -8, -4, -2, 0, -2, 0, -3], 2.85),  # Negative values test
    ],
)
def test_hm_exponentiality_criterion(data, result):
    statistic = HarrisExponentialityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


@pytest.mark.parametrize(
    "data",
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Zero exception test
    ],
)
def test_nan_hm_exponentiality_criterion(data):
    statistic = HarrisExponentialityGofStatistic().execute_statistic(data)
    assert np.isnan(statistic)


def test_hm_exponentiality_criterion_code():
    assert "HM_EXPONENTIALITY_GOODNESS_OF_FIT" == HarrisExponentialityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.12381),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.13889),
        ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0.0),  # Zero exception test
        ([-4, -1, -6, -8, -4, -2, 0, -2, 0, -3], 0.7000000000000001),  # Negative values test
    ],
)
def test_hp_exponentiality_criterion(data, result):
    statistic = HollanderProshanExponentialityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_hp_exponentiality_criterion_code():
    assert (
        "HP_EXPONENTIALITY_GOODNESS_OF_FIT" == HollanderProshanExponentialityGofStatistic().code()
    )


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 2.4073),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 2.592),
        ([-4, -1, -6, -8, -4, -2, 0, -2, 0, -3], 6.767360569535781),  # Negative values test
    ],
)
def test_kc_exponentiality_criterion(data, result):
    statistic = KocharExponentialityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.001)


@pytest.mark.parametrize(
    "data",
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Zero exception test
    ],
)
def test_nan_kc_exponentiality_criterion(data):
    statistic = KocharExponentialityGofStatistic().execute_statistic(data)
    assert np.isnan(statistic)


def test_kc_exponentiality_criterion_code():
    assert "KC_EXPONENTIALITY_GOODNESS_OF_FIT" == KocharExponentialityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.13948),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.1238),
        ([-4, -1, -6, -8, -4, -2, 0, -2, 0, -3], 0.8564337068712936),  # Negative values test
    ],
)
def test_km_exponentiality_criterion(data, result):
    statistic = KimberMichaelExponentialityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.001)


@pytest.mark.parametrize(
    "data",
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Zero exception test
    ],
)
def test_nan_km_exponentiality_criterion(data):
    statistic = KimberMichaelExponentialityGofStatistic().execute_statistic(data)
    assert np.isnan(statistic)


def test_km_exponentiality_criterion_code():
    assert "KM_EXPONENTIALITY_GOODNESS_OF_FIT" == KimberMichaelExponentialityGofStatistic().code()


# TODO: ([4, 1, 6, -8, 4, 2, 0, -2, 0, 3], 2980.057987041728), - some garbage values test?
@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.72180),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.764664),
        ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 1.0),  # Zero exception test
        ([-4, -1, -6, -8, -4, -2, 0, -2, 0, -3], 1.0),  # Negative values test
    ],
)
def test_ks_exponentiality_criterion(data, result):
    statistic = KolmogorovSmirnovExponentialityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.001)


def test_ks_exponentiality_criterion_code():
    assert (
        "KS_EXPONENTIALITY_GOODNESS_OF_FIT" == KolmogorovSmirnovExponentialityGofStatistic().code()
    )


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.21429),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.27273),
        ([-4, -1, -6, -8, -4, -2, 0, -2, 0, -3], 0.8333333333333334),  # Negative values test
    ],
)
def test_lz_exponentiality_criterion(data, result):
    statistic = LorenzExponentialityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.001)


@pytest.mark.parametrize(
    "data",
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Zero exception test
    ],
)
def test_exc_lz_exponentiality_criterion(data):
    with pytest.raises(ZeroDivisionError):
        statistic = LorenzExponentialityGofStatistic().execute_statistic(data)
        pytest.approx(statistic, 0.00001)


def test_lz_exponentiality_criterion_code():
    assert "LZ_EXPONENTIALITY_GOODNESS_OF_FIT" == LorenzExponentialityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.4088),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.38291),
        ([-4, -1, -6, -8, -4, -2, 0, -2, 0, -3], -inf),  # Negative values test
    ],
)
def test_mn_exponentiality_criterion(data, result):
    statistic = MoranExponentialityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


@pytest.mark.parametrize(
    "data",
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Zero exception test
    ],
)
def test_nan_mn_exponentiality_criterion(data):
    statistic = MoranExponentialityGofStatistic().execute_statistic(data)
    assert np.isnan(statistic)


def test_mn_exponentiality_criterion_code():
    assert "MN_EXPONENTIALITY_GOODNESS_OF_FIT" == MoranExponentialityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.21429),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.22727),
        ([-4, -1, -6, -8, -4, -2, 0, -2, 0, -3], -0.3333333333333333),  # Negative values test
    ],
)
def test_pt_exponentiality_criterion(data, result):
    statistic = PietraExponentialityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.001)


@pytest.mark.parametrize(
    "data",
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Zero exception test
    ],
)
def test_nan_pt_exponentiality_criterion(data):
    statistic = PietraExponentialityGofStatistic().execute_statistic(data)
    assert np.isnan(statistic)


def test_pt_exponentiality_criterion_code():
    assert "PT_EXPONENTIALITY_GOODNESS_OF_FIT" == PietraExponentialityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.095238),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.091667),
        ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0),  # Zero exception test
        ([-4, -1, -6, -8, -4, -2, 0, -2, 0, -3], -0.6066666666666667),  # Negative values test
    ],
)
def test_rs_exponentiality_criterion(data, result):
    statistic = RossbergExponentialityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_rs_exponentiality_criterion_code():
    assert "RS_EXPONENTIALITY_GOODNESS_OF_FIT" == RossbergExponentialityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.375),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.27273),
        ([-4, -1, -6, -8, -4, -2, 0, -2, 0, -3], 0.46296296296296297),  # Negative values test
    ],
)
def test_sw_exponentiality_criterion(data, result):
    statistic = ShapiroWilkExponentialityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


@pytest.mark.parametrize(
    "data",
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Zero exception test
    ],
)
def test_nan_sw_exponentiality_criterion(data):
    statistic = ShapiroWilkExponentialityGofStatistic().execute_statistic(data)
    assert np.isnan(statistic)


def test_sw_exponentiality_criterion_code():
    assert "SW_EXPONENTIALITY_GOODNESS_OF_FIT" == ShapiroWilkExponentialityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 0.035714),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.027273),
        ([-4, -1, -6, -8, -4, -2, 0, -2, 0, -3], 0.06),  # Negative values test
    ],
)
def test_we_exponentiality_criterion(data, result):
    statistic = WeExponentialityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, abs=0.01)


@pytest.mark.parametrize(
    "data",
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Zero exception test
    ],
)
def test_nan_we_exponentiality_criterion(data):
    statistic = WeExponentialityGofStatistic().execute_statistic(data)
    assert np.isnan(statistic)


def test_we_exponentiality_criterion_code():
    assert "WE_EXPONENTIALITY_GOODNESS_OF_FIT" == WeExponentialityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([1, 2, 3, 4, 5, 6, 7], 7),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 10),
        ([-4, -1, -6, -8, -4, -2, 0, -2, 0, -3], -0.0),  # Negative values test
    ],
)
def test_ww_exponentiality_criterion(data, result):
    statistic = WongWongExponentialityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


@pytest.mark.parametrize(
    "data",
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Zero exception test
    ],
)
def test_exc_ww_exponentiality_criterion(data):
    with pytest.raises(ZeroDivisionError):
        statistic = WongWongExponentialityGofStatistic().execute_statistic(data)
        pytest.approx(statistic, 0.00001)


def test_ww_exponentiality_criterion_code():
    assert "WW_EXPONENTIALITY_GOODNESS_OF_FIT" == WongWongExponentialityGofStatistic().code()
