import pytest as pytest

from pysatl_criterion.normal import (
    AndersonDarlingNormalityGofStatistic,
    BHSNormalityGofStatistic,
    BonettSeierNormalityGofStatistic,
    BontempsMeddahi1NormalityGofStatistic,
    BontempsMeddahi2NormalityGofStatistic,
    CabanaCabana1NormalityGofStatistic,
    CabanaCabana2NormalityGofStatistic,
    ChenShapiroNormalityGofStatistic,
    CoinNormalityGofStatistic,
    DagostinoNormalityGofStatistic,
    DAPNormalityGofStatistic,
    DesgagneLafayeNormalityGofStatistic,
    DoornikHansenNormalityGofStatistic,
    EppsPulleyNormalityGofStatistic,
    FilliNormalityGofStatistic,
    GlenLeemisBarrNormalityGofStatistic,
    GMGNormalityGofStatistic,
    GraphEdgesNumberNormalityGofStatistic,
    GraphMaxDegreeNormalityGofStatistic,
    Hosking1NormalityGofStatistic,
    Hosking2NormalityGofStatistic,
    Hosking3NormalityGofStatistic,
    Hosking4NormalityGofStatistic,
    JBNormalityGofStatistic,
    KolmogorovSmirnovNormalityGofStatistic,
    KurtosisNormalityGofStatistic,
    LillieforsNormalityGofStatistic,
    LooneyGulledgeNormalityGofStatistic,
    MartinezIglewiczNormalityGofStatistic,
    RobustJarqueBeraNormalityGofStatistic,
    RyanJoinerNormalityGofStatistic,
    SFNormalityGofStatistic,
    ShapiroWilkNormalityGofStatistic,
    SkewNormalityGofStatistic,
    SpiegelhalterNormalityGofStatistic,
    SWRGNormalityGofStatistic,
    ZhangQNormalityGofStatistic,
    ZhangQStarNormalityGofStatistic,
    ZhangWuANormalityGofStatistic,
    ZhangWuCNormalityGofStatistic,
)


@pytest.mark.parametrize(
    ("data", "result"),
    [
        # Normal with mean = 0, variance = 1
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
            0.18573457378941832,
        ),
        # Normal with mean = 11, variance = 1
        (
            [
                -0.46869863,
                -0.22452687,
                -1.7674444,
                -0.727139,
                1.09089112,
                -0.01319041,
                0.38578004,
                1.47354665,
                0.95253258,
                -1.17323879,
            ],
            0.12958652448618313,
        ),
        # Normal with mean = 0, variance = 5
        (
            [
                -0.46869863,
                -0.22452687,
                -1.7674444,
                -0.727139,
                1.09089112,
                -0.01319041,
                0.38578004,
                1.47354665,
                0.95253258,
                -1.17323879,
            ],
            0.12958652448618313,
        ),
        # Normal with mean = 11, variance = 5
        (
            [
                -0.46869863,
                -0.22452687,
                -1.7674444,
                -0.727139,
                1.09089112,
                -0.01319041,
                0.38578004,
                1.47354665,
                0.95253258,
                -1.17323879,
            ],
            0.12958652448618313,
        ),
    ],
)
def test_ks_normality_criterion(data, result):
    statistic = KolmogorovSmirnovNormalityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_ks_normality_criterion_code():
    assert "KS_NORMALITY_GOODNESS_OF_FIT" == KolmogorovSmirnovNormalityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        # Normal with mean = 0, variance = 1
        ([16, 18, 16, 14, 12, 12, 16, 18, 16, 14, 12, 12], 0.6822883),
        (
            [
                1.0329650,
                -0.2861944,
                0.1488185,
                0.9907514,
                -0.3244450,
                0.4430822,
                -0.1238494,
            ],
            0.3753546,
        ),
        (
            [
                -0.21999313,
                0.48724826,
                0.87227246,
                -0.08396081,
                -0.12506021,
                -2.54337169,
                0.50740722,
                -0.15209779,
                -0.12694116,
                -1.09978690,
            ],
            0.7747652,
        ),
    ],
)
def test_ad_normality_criterion(data, result):
    statistic = AndersonDarlingNormalityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_ad_normality_criterion_code():
    assert "AD_NORMALITY_GOODNESS_OF_FIT" == AndersonDarlingNormalityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        (
            [
                -1.3662750,
                -1.2772617,
                1.2341902,
                0.4943849,
                -0.6015152,
                0.7927679,
                -2.6990387,
            ],
            0.9122889,
        ),
        (
            [
                2.39179539,
                1.00572055,
                0.55561602,
                0.49246060,
                -0.43379600,
                0.03081284,
                0.31172966,
                0.40097292,
                0.46238934,
                -0.29856372,
            ],
            1.239515,
        ),
    ],
)
# TODO: remove skip
@pytest.mark.skip(reason="no way of currently testing this")
def test_bhs_normality_criterion(data, result):
    statistic = BHSNormalityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_bhs_normality_criterion_code():
    assert "BHS_NORMALITY_GOODNESS_OF_FIT" == BHSNormalityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        (
            [
                -0.7826974,
                -1.1840876,
                1.0317606,
                -0.7335811,
                -0.0862771,
                -1.1437829,
                0.4685360,
            ],
            -1.282991,
        ),
        (
            [
                0.73734236,
                0.40517722,
                0.09825027,
                0.27044629,
                0.93485784,
                -0.41404827,
                -0.01128772,
                0.41428093,
                0.18568170,
                -0.89367267,
            ],
            0.6644447,
        ),
    ],
)
def test_bonett_seier_normality_criterion(data, result):
    statistic = BonettSeierNormalityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_bonett_seier_normality_criterion_code():
    assert "BS_NORMALITY_GOODNESS_OF_FIT" == BonettSeierNormalityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        (
            [
                0.5803773,
                -1.3941189,
                1.4266496,
                0.4229516,
                1.2052829,
                -0.7798392,
                -0.2476446,
            ],
            0.2506808,
        ),
        (
            [
                0.05518574,
                -0.09918900,
                -0.25097539,
                0.45345120,
                1.01584731,
                0.45844901,
                0.79256755,
                0.36811349,
                -0.56170844,
                3.15364608,
            ],
            4.814269,
        ),
    ],
)
def test_bontemps_meddahi1_normality_criterion(data, result):
    statistic = BontempsMeddahi1NormalityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_bontemps_meddahi1_normality_criterion_code():
    assert "BM1_NORMALITY_GOODNESS_OF_FIT" == BontempsMeddahi1NormalityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        (
            [
                -1.16956851,
                -1.88725716,
                -0.09051621,
                -0.84191408,
                -0.65989921,
                -0.22018994,
                -0.12274684,
            ],
            1.155901,
        ),
        (
            [
                -2.1291160,
                -1.2046194,
                -0.9706029,
                0.1458201,
                0.5181943,
                -0.9769141,
                -0.8174199,
                0.2369553,
                0.4190111,
                0.6978357,
            ],
            1.170676,
        ),
    ],
)
def test_bontemps_meddahi2_normality_criterion(data, result):
    statistic = BontempsMeddahi2NormalityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_bontemps_meddahi2_normality_criterion_code():
    assert "BM2_NORMALITY_GOODNESS_OF_FIT" == BontempsMeddahi2NormalityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        (
            [
                -0.33234073,
                -1.73762000,
                -0.08110214,
                1.13414679,
                0.09228884,
                -0.69521329,
                0.10573062,
            ],
            0.2897665,
        ),
        (
            [
                0.99880346,
                -0.07557944,
                0.25368407,
                -1.20830967,
                -0.15914329,
                0.16900984,
                0.99395022,
                -0.28167969,
                0.11683112,
                0.68954236,
            ],
            0.5265257,
        ),
    ],
)
def test_cabana_cabana_1_normality_criterion(data, result):
    statistic = CabanaCabana1NormalityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_cabana_cabana_1_normality_criterion_code():
    assert "CC1_NORMALITY_GOODNESS_OF_FIT" == CabanaCabana1NormalityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        (
            [
                -0.2115733,
                -0.8935314,
                -0.1916746,
                0.2805032,
                1.3372893,
                -0.4324158,
                2.8578810,
            ],
            0.2497146,
        ),
        (
            [
                0.99880346,
                -0.07557944,
                0.25368407,
                -1.20830967,
                -0.15914329,
                0.16900984,
                0.99395022,
                -0.28167969,
                0.11683112,
                0.68954236,
            ],
            0.1238103,
        ),
    ],
)
def test_cabana_cabana_2_normality_criterion(data, result):
    statistic = CabanaCabana2NormalityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_cabana_cabana_2_normality_criterion_code():
    assert "CC2_NORMALITY_GOODNESS_OF_FIT" == CabanaCabana2NormalityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        (
            [
                0.93797412,
                -0.33927015,
                -0.57280736,
                0.03294079,
                0.48674056,
                -0.52471379,
                1.15231162,
            ],
            -0.07797202,
        ),
        (
            [
                -0.8732478,
                0.6104841,
                1.1886920,
                0.3229907,
                1.4729158,
                0.5256972,
                -0.4902668,
                -0.8249011,
                -0.7751734,
                -1.8370833,
            ],
            -0.1217789,
        ),
    ],
)
def test_chen_shapiro_normality_criterion(data, result):
    statistic = ChenShapiroNormalityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_chen_shapiro_normality_criterion_code():
    assert "CS_NORMALITY_GOODNESS_OF_FIT" == ChenShapiroNormalityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        (
            [
                0.5171378,
                1.1163130,
                -1.3117699,
                -0.1739053,
                -0.6385798,
                -0.2520054,
                0.3237990,
            ],
            0.01249513,
        ),
        (
            [
                0.92728608,
                -0.75756591,
                -0.07266914,
                0.09636470,
                -1.13792085,
                -0.91534895,
                1.57469227,
                0.28462605,
                0.22804695,
                -0.29829152,
            ],
            0.0009059856,
        ),
    ],
)
def test_coin_normality_criterion(data, result):
    statistic = CoinNormalityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_coin_normality_criterion_code():
    assert "COIN_NORMALITY_GOODNESS_OF_FIT" == CoinNormalityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        (
            [
                -0.4419019,
                -0.3962638,
                -0.9919951,
                -1.7636001,
                1.0433300,
                -0.6375415,
                1.2467400,
            ],
            -0.7907805,
        ),
        (
            [
                0.67775366,
                -0.07238245,
                1.87603589,
                0.46277364,
                1.10585543,
                -0.95274655,
                -1.47549650,
                0.42478574,
                0.91713384,
                0.24491208,
            ],
            -0.7608445,
        ),
    ],
)
def test_dagostino_normality_criterion(data, result):
    statistic = DagostinoNormalityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_dagostino_normality_criterion_code():
    assert "D_NORMALITY_GOODNESS_OF_FIT" == DagostinoNormalityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([148, 154, 158, 160, 161, 162, 166, 170, 182, 195, 236], 13.034263121192582),
        ([16, 18, 16, 14, 12, 12, 16, 18, 16, 14, 12, 12], 2.5224),
    ],
)
def test_dap_normality_criterion(data, result):
    statistic = DAPNormalityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_dap_normality_criterion_code():
    assert "DAP_NORMALITY_GOODNESS_OF_FIT" == DAPNormalityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        (
            [
                0.7258605,
                -1.0666939,
                -1.3391837,
                0.1826556,
                -0.9695532,
                0.5618815,
                -1.5228123,
            ],
            5.583199,
        ),
        (
            [
                -0.59336721,
                1.31220820,
                -0.82065801,
                -1.68778329,
                1.96735245,
                -0.43180098,
                -0.63682878,
                -1.34366222,
                -0.03375564,
                1.30610658,
            ],
            0.8639117,
        ),
    ],
)
def test_desgagne_lafaye_normality_criterion(data, result):
    statistic = DesgagneLafayeNormalityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_desgagne_lafaye_normality_criterion_code():
    assert "DLDMZEPD_NORMALITY_GOODNESS_OF_FIT" == DesgagneLafayeNormalityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        (
            [
                -1.09468228,
                -0.31322754,
                -0.78294147,
                -0.58466218,
                0.09357476,
                0.35397261,
                -2.77320261,
                0.29275119,
                -0.66726297,
                2.17449000,
            ],
            7.307497,
        ),
        (
            [
                -0.67054898,
                -0.96828029,
                -0.84417791,
                0.06829821,
                1.52624840,
                1.72143189,
                1.50767670,
                -0.08592902,
                -0.46234996,
                0.29561229,
                0.32708351,
            ],
            2.145117,
        ),
    ],
)
def test_doornik_hansen_normality_criterion(data, result):
    statistic = DoornikHansenNormalityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_doornik_hansen_normality_criterion_code():
    assert "DH_NORMALITY_GOODNESS_OF_FIT" == DoornikHansenNormalityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        (
            [
                5.50,
                5.55,
                5.57,
                5.34,
                5.42,
                5.30,
                5.61,
                5.36,
                5.53,
                5.79,
                5.47,
                5.75,
                4.88,
                5.29,
                5.62,
                5.10,
                5.63,
                5.68,
                5.07,
                5.58,
                5.29,
                5.27,
                5.34,
                5.85,
                5.26,
                5.65,
                5.44,
                5.39,
                5.46,
            ],
            0.05191694742233466,
        ),
    ],
)
def test_ep_normality_criterion(data, result):
    statistic = EppsPulleyNormalityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_ep_normality_criterion_code():
    assert "EP_NORMALITY_GOODNESS_OF_FIT" == EppsPulleyNormalityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([-4, -2, 0, 1, 5, 6, 8], 0.9854095718708367),
    ],
)
def test_filli_normality_criterion(data, result):
    statistic = FilliNormalityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_filli_normality_criterion_code():
    assert "FILLI_NORMALITY_GOODNESS_OF_FIT" == FilliNormalityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        (
            [
                -0.2326386,
                0.5440749,
                0.9742477,
                0.1979832,
                1.7937705,
                -0.5430379,
                1.5229193,
            ],
            5.019177,
        ),
        (
            [
                0.43880197,
                0.01657893,
                0.11190411,
                0.22168051,
                -0.83993220,
                -1.85572181,
                0.07311574,
                -0.69846684,
                0.54829821,
                -0.45549464,
            ],
            12.54511,
        ),
    ],
)
def test_glen_leemis_barr_normality_criterion(data, result):
    statistic = GlenLeemisBarrNormalityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_glen_leemis_barr_normality_criterion_code():
    assert "GLB_NORMALITY_GOODNESS_OF_FIT" == GlenLeemisBarrNormalityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        (
            [
                -0.01686868,
                1.98378809,
                1.34831025,
                0.38120500,
                -0.35364982,
                -0.65345851,
                0.05968902,
            ],
            1.033118,
        ),
        (
            [
                1.00488088,
                -1.71519143,
                0.48269944,
                -0.10380093,
                -0.02961192,
                -0.42891128,
                0.07543129,
                -0.03098110,
                -0.72435341,
                -0.90046224,
            ],
            1.066354,
        ),
    ],
)
def test_gmg_normality_criterion(data, result):
    statistic = GMGNormalityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_gmg_normality_criterion_code():
    assert "GMG_NORMALITY_GOODNESS_OF_FIT" == GMGNormalityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        (
            [-0.264, 0.031, 0.919, 1.751, -0.038, 0.133, 0.643, -0.480, 0.094, -0.527],
            18,
        ),
        ([1.311, 3.761, 0.415, 0.764, 0.100, -0.028, -1.516, -0.108, 2.248, 0.229], 6),
    ],
)
def test_graph_edges_number_normality_criterion(data, result):
    statistic = GraphEdgesNumberNormalityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_graph_edges_number_normality_criterion_code():
    assert "EdgesNumber_NORMALITY_GOODNESS_OF_FIT" == GraphEdgesNumberNormalityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([-0.264, 0.031, 0.919, 1.751, -0.038, 0.133, 0.643, -0.480, 0.094, -0.527], 6),
        ([1.311, 3.761, 0.415, 0.764, 0.100, -0.028, -1.516, -0.108, 2.248, 0.229], 3),
    ],
)
def test_graph_max_degree_normality_criterion(data, result):
    statistic = GraphMaxDegreeNormalityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_graph_max_degree_normality_criterion_code():
    assert "MaxDegree_NORMALITY_GOODNESS_OF_FIT" == GraphMaxDegreeNormalityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        (
            [
                -0.17746565,
                0.36878289,
                0.14037068,
                -0.56292004,
                2.02953048,
                0.63754044,
                -0.05821471,
            ],
            31.26032,
        ),
        (
            [
                0.07532721,
                0.17561663,
                -0.45442472,
                -0.31402998,
                -0.36055484,
                0.46426559,
                0.18860127,
                -0.18712276,
                0.12134652,
                0.25866486,
            ],
            3.347533,
        ),
    ],
)
def test_hosking_1normality_criterion(data, result):
    statistic = Hosking1NormalityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_hosking_1normality_criterion_code():
    assert "HOSKING1_NORMALITY_GOODNESS_OF_FIT" == Hosking1NormalityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        (
            [
                -1.52179225,
                -0.07079375,
                0.96500281,
                0.65256255,
                0.71376323,
                0.18295605,
                0.31265772,
            ],
            13.27192,
        ),
        (
            [
                -0.9352061,
                -1.0456637,
                0.7134893,
                1.6715891,
                1.7931811,
                -0.1422531,
                0.9682729,
                0.2980237,
                0.8548988,
                -0.8224675,
            ],
            1.693243,
        ),
    ],
)
def test_hosking_2_normality_criterion(data, result):
    statistic = Hosking2NormalityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_hosking_2_normality_criterion_code():
    assert "HOSKING2_NORMALITY_GOODNESS_OF_FIT" == Hosking2NormalityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        (
            [
                -0.9515396,
                0.4302541,
                0.1149620,
                1.7218222,
                -0.4061157,
                -0.2528552,
                0.7840704,
                -1.6576825,
            ],
            41.33229,
        ),
        (
            [
                -1.4387336,
                1.2636724,
                -1.9232885,
                0.5963312,
                0.1208620,
                -1.1269378,
                0.5032659,
                0.3810323,
                0.8924223,
                1.8037073,
            ],
            117.5835,
        ),
    ],
)
def test_hosking_3_normality_criterion(data, result):
    statistic = Hosking3NormalityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_hosking_3_normality_criterion_code():
    assert "HOSKING3_NORMALITY_GOODNESS_OF_FIT" == Hosking3NormalityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        (
            [
                -0.93804525,
                -0.85812989,
                -1.35114261,
                0.16821566,
                2.05324842,
                0.72370964,
                1.58014787,
                0.07116436,
                -0.20992477,
                0.37184699,
                -0.41287789,
            ],
            1.737481,
        ),
        (
            [
                -0.18356827,
                0.42145728,
                -1.30305510,
                1.65498056,
                0.16475340,
                0.68201228,
                -0.26179821,
                -0.03263223,
                1.57505463,
                -0.34043549,
            ],
            3.111041,
        ),
    ],
)
def test_hosking_4_normality_criterion(data, result):
    statistic = Hosking4NormalityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_hosking_4_normality_criterion_code():
    assert "HOSKING4_NORMALITY_GOODNESS_OF_FIT" == Hosking4NormalityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([148, 154, 158, 160, 161, 162, 166, 170, 182, 195, 236], 6.982848237344646),
        (
            [
                0.30163062,
                -1.17676177,
                -0.883211,
                0.55872679,
                2.04829646,
                0.66029436,
                0.83445286,
                0.72505429,
                1.25012578,
                -1.11276931,
            ],
            0.44334632590843914,
        ),
    ],
)
def test_jb_normality_criterion(data, result):
    statistic = JBNormalityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_jb_normality_criterion_code():
    assert "JB_NORMALITY_GOODNESS_OF_FIT" == JBNormalityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([148, 154, 158, 160, 161, 162, 166, 170, 182, 195, 236], 2.3048235214240873),
    ],
)
def test_kurtosis_normality_criterion(data, result):
    statistic = KurtosisNormalityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_kurtosis_normality_criterion_code():
    assert "KURTOSIS_NORMALITY_GOODNESS_OF_FIT" == KurtosisNormalityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([148, 154, 158, 160, 161, 162, 166, 170, 170, 182, 195], 0.956524208286),
    ],
)
def test_looney_gulledge_normality_criterion(data, result):
    statistic = LooneyGulledgeNormalityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_looney_gulledge_normality_criterion_code():
    assert "LG_NORMALITY_GOODNESS_OF_FIT" == LooneyGulledgeNormalityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([-1, 0, 1], 0.17467808),
        (
            [
                0.8366388,
                1.1972029,
                0.4660834,
                -1.8013118,
                0.8941450,
                -0.2602227,
                0.8496448,
            ],
            0.2732099,
        ),
        (
            [
                0.72761915,
                -0.02049438,
                -0.13595651,
                -0.12371837,
                -0.11037662,
                0.46608165,
                1.25378956,
                -0.64296653,
                0.25356762,
                0.23345769,
            ],
            0.1695222,
        ),
    ],
)
def test_lilliefors_normality_criterion(data, result):
    statistic = LillieforsNormalityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_lilliefors_normality_criterion_code():
    assert "LILLIE_NORMALITY_GOODNESS_OF_FIT" == LillieforsNormalityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        (
            [
                0.42240539,
                -1.05926060,
                1.38703979,
                -0.69969283,
                -0.58799872,
                0.45095572,
                0.07361136,
            ],
            1.081138,
        ),
        (
            [
                -0.6930954,
                -0.1279816,
                0.7552798,
                -1.1526064,
                0.8638102,
                -0.5517623,
                0.3070847,
                -1.6807102,
                -1.7846244,
                -0.3949447,
            ],
            1.020476,
        ),
    ],
)
def test_martinez_iglewicz_normality_criterion(data, result):
    statistic = MartinezIglewiczNormalityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_martinez_iglewicz_normality_criterion_code():
    assert "MI_NORMALITY_GOODNESS_OF_FIT" == MartinezIglewiczNormalityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([148, 154, 158, 160, 161, 162, 166, 170, 170, 182, 195], 0.9565242082866772),
        ([6, 1, -4, 8, -2, 5, 0], 0.9844829186140105),
    ],
)
def test_ryan_joiner_normality_criterion(data, result):
    statistic = RyanJoinerNormalityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_ryan_joiner_normality_criterion_code():
    assert "RJ_NORMALITY_GOODNESS_OF_FIT" == RyanJoinerNormalityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        (
            [
                1.2318068,
                -0.3417207,
                -1.2044307,
                -0.7724564,
                -0.2145365,
                -1.0119879,
                0.2222634,
            ],
            0.8024895,
        ),
        (
            [
                -1.0741031,
                1.3157369,
                2.7003935,
                0.8843286,
                -0.4361445,
                -0.3000996,
                -0.2710125,
                -0.6915687,
                -1.7699595,
                1.3740694,
            ],
            0.4059704,
        ),
    ],
)
def test_robust_jarque_bera_normality_criterion(data, result):
    statistic = RobustJarqueBeraNormalityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_robust_jarque_bera_normality_criterion_code():
    assert "RJB_NORMALITY_GOODNESS_OF_FIT" == RobustJarqueBeraNormalityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        (
            [
                -1.5461357,
                0.8049704,
                -1.2676556,
                0.1912453,
                1.4391551,
                0.5352138,
                -1.6212611,
                0.1015035,
                -0.2571793,
                0.8756286,
            ],
            0.93569,
        ),
    ],
)
def test_sf_normality_criterion(data, result):
    statistic = SFNormalityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_sf_normality_criterion_code():
    assert "SF_NORMALITY_GOODNESS_OF_FIT" == SFNormalityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([148, 154, 158, 160, 161, 162, 166, 170, 182, 195, 236], 2.7788579769903414),
    ],
)
def test_skew_normality_criterion(data, result):
    statistic = SkewNormalityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_skew_normality_criterion_code():
    assert "SKEW_NORMALITY_GOODNESS_OF_FIT" == SkewNormalityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        (
            [
                -1.31669223,
                2.22819380,
                -0.27391944,
                -1.57616900,
                -2.21675399,
                -0.01497801,
                -1.65492071,
            ],
            1.328315,
        ),
        (
            [
                -1.6412500,
                -1.1946111,
                1.1054937,
                -0.4210709,
                -1.1736754,
                -1.1750840,
                1.3267088,
                -0.3299987,
                -0.5767829,
                -1.4114579,
            ],
            1.374628,
        ),
    ],
)
def test_spiegelhalter_normality_criterion(data, result):
    statistic = SpiegelhalterNormalityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_spiegelhalter_normality_criterion_code():
    assert "SH_NORMALITY_GOODNESS_OF_FIT" == SpiegelhalterNormalityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([16, 18, 16], 0.75),
        ([16, 18, 16, 14], 0.9446643),
        ([16, 18, 16, 14, 15], 0.955627),
        ([38.7, 41.5, 43.8, 44.5, 45.5, 46.0, 47.7, 58.0], 0.872973),
    ],
)
def test_sw_normality_criterion(data, result):
    statistic = ShapiroWilkNormalityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_sw_normality_criterion_code():
    assert "SW_NORMALITY_GOODNESS_OF_FIT" == ShapiroWilkNormalityGofStatistic().code()


"""
@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([12, 12, 12, 12, 14, 14, 16, 16, 16, 16, 18], 3.66666666),
        ([18, 16, 16, 16, 16, 14, 14, 12, 12, 12, 12], 3.66666666),
        ([-1.71228079, -0.86710019, 0.29950617, 1.18632683, -0.13929811, -1.47008114,
          -1.29073683, 1.18998087, 0.80807576, 0.45558552], 0.07474902435493411),
    ],
)
def test_swm_normality_criterion(data, result):
    statistic = SWMNormalityTest().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)
"""


@pytest.mark.parametrize(
    ("data", "result"),
    [
        (
            [
                0.2361394,
                0.3353304,
                1.0837427,
                1.7284429,
                -0.5890293,
                -0.2175157,
                -0.3615631,
            ],
            0.9120115,
        ),
        (
            [
                -0.38611393,
                0.40744855,
                0.01385485,
                -0.80707299,
                -1.33020278,
                0.53527066,
                0.35588475,
                -0.44262575,
                0.28699128,
                -0.66855218,
            ],
            0.9092612,
        ),
    ],
)
def test_swrg_normality_criterion(data, result):
    statistic = SWRGNormalityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_swrg_normality_criterion_code():
    assert "SWRG_NORMALITY_GOODNESS_OF_FIT" == SWRGNormalityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        (
            [
                1.84917102,
                -0.11520399,
                -0.99425682,
                -0.02146024,
                1.90311564,
                1.10073929,
                0.26444036,
                1.46165119,
                1.65611589,
                0.14613976,
                0.29509227,
            ],
            0.03502227,
        ),
        (
            [
                -1.21315374,
                -0.19765354,
                0.46560179,
                -1.48894141,
                -0.57958644,
                -0.87905499,
                2.25757863,
                -0.83696957,
                0.01074617,
                -0.34492908,
            ],
            -0.2811746,
        ),
    ],
)
def test_zhang_q_normality_criterion(data, result):
    statistic = ZhangQNormalityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_zhang_q_normality_criterion_code():
    assert "ZQ_NORMALITY_GOODNESS_OF_FIT" == ZhangQNormalityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        (
            [
                0.002808107,
                -0.366738714,
                0.627663491,
                0.459724293,
                0.044694653,
                1.096110474,
                -0.492341832,
                0.708343932,
                0.247694191,
                0.523295664,
                0.234479385,
            ],
            0.0772938,
        ),
        (
            [
                0.3837459,
                -2.4917339,
                0.6754353,
                -0.5634646,
                -1.3273973,
                0.4896063,
                1.0786708,
                -0.1585859,
                -1.0140335,
                1.0448026,
            ],
            -0.5880094,
        ),
    ],
)
def test_zhang_q_tar_normality_criterion(data, result):
    statistic = ZhangQStarNormalityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_zhang_q_tar_normality_criterion_code():
    assert "ZQS_NORMALITY_GOODNESS_OF_FIT" == ZhangQStarNormalityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        (
            [
                -1.0968987,
                1.7392081,
                0.9674481,
                -0.3418871,
                -0.5659707,
                1.0234917,
                1.0958103,
            ],
            1.001392,
        ),
        (
            [
                0.31463996,
                0.17626475,
                -0.01481709,
                0.25539075,
                0.64605810,
                0.64965352,
                -0.36176169,
                -0.59318222,
                -0.44131251,
                0.41216214,
            ],
            1.225743,
        ),
    ],
)
@pytest.mark.skip(reason="no way of currently testing this")
def test_zhang_wu_a_normality_criterion(data, result):
    statistic = ZhangWuANormalityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_zhang_wu_a_normality_criterion_code():
    assert "ZWA_NORMALITY_GOODNESS_OF_FIT" == ZhangWuANormalityGofStatistic().code()


@pytest.mark.parametrize(
    ("data", "result"),
    [
        (
            [
                -0.1750217,
                -1.8522318,
                -0.3997543,
                -0.3326987,
                -0.9864067,
                -0.4701900,
                2.7674965,
            ],
            5.194459,
        ),
        (
            [
                -0.12344697,
                -0.74936974,
                1.12023439,
                1.09091550,
                -0.05204564,
                -0.35421236,
                -0.70361281,
                2.38810563,
                -0.70401541,
                1.16743393,
            ],
            5.607312,
        ),
    ],
)
def test_zhang_wu_c_normality_criterion(data, result):
    statistic = ZhangWuCNormalityGofStatistic().execute_statistic(data)
    assert result == pytest.approx(statistic, 0.00001)


def test_zhang_wu_c_normality_criterion_code():
    assert "ZWC_NORMALITY_GOODNESS_OF_FIT" == ZhangWuCNormalityGofStatistic().code()
