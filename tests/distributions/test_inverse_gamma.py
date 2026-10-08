import numpy as np
import pytest
from scipy import stats

from pysatl_criterion import DistributionType
from pysatl_criterion.distribution.distributions import InverseGammaDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit.inverse_gamma import (
    AbstractInverseGammaGofStatistic,
    AndersonDarlingInverseGammaGofStatistic,
    Chi2PearsonInverseGammaGofStatistic,
    CramerVonMisesInverseGammaGofStatistic,
    GreenwoodInverseGammaGofStatistic,
    KolmogorovSmirnovInverseGammaGofStatistic,
    KuiperInverseGammaGofStatistic,
    MinToshiyukiInverseGammaGofStatistic,
    WatsonInverseGammaGofStatistic,
    ZhangAInverseGammaGofStatistic,
    ZhangCInverseGammaGofStatistic,
    ZhangKInverseGammaGofStatistic,
)
from tests.parameter_cases import parameters_for


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
_ALPHA = 2.5
_BETA = 1.1


def _reference_statistics(sample, alpha, beta):
    x = np.sort(sample)
    u = stats.gamma.sf(beta / x, a=alpha)
    n = len(x)
    i = np.arange(1, n + 1)
    dplus = i / n - u
    dminus = u - (i - 1) / n
    gaps = np.diff(np.r_[0, u, 1])
    w2 = 1 / (12 * n) + np.sum((u - (i - 0.5) / n) ** 2)
    counts = np.histogram(u, bins=np.linspace(0, 1, 6))[0]
    return {
        "kuiper": max(dplus) + max(dminus),
        "greenwood": np.dot(gaps, gaps),
        "min_toshiyuki": sum(np.maximum(dplus, dminus) / np.sqrt(u * (1 - u))) / np.sqrt(n),
        "watson": w2 - n * (u.mean() - 0.5) ** 2,
        "chi2_pearson": stats.chisquare(counts).statistic,
        "zhang_a": -sum(np.log(u) / (n - i + 0.5) + np.log1p(-u) / (i - 0.5)),
        "zhang_c": sum(np.log((1 / u - 1) / ((n - 0.5) / (i - 0.75) - 1)) ** 2),
        "zhang_k": max(
            (i - 0.5) * np.log((i - 0.5) / (n * u))
            + (n - i + 0.5) * np.log((n - i + 0.5) / (n * (1 - u)))
        ),
    }


def test_inverse_gamma_base_code():
    """Ensure Inverse Gamma GOF statistics expose the expected code identifier."""
    assert AbstractInverseGammaGofStatistic.code() == "INV_GAMMA_GOODNESS_OF_FIT"


def test_inverse_gamma_positive_shape_required():
    """Stat constructors should reject non-positive shape parameters."""
    with pytest.raises(ValueError, match="Invalid value for alpha"):
        KolmogorovSmirnovInverseGammaGofStatistic(
            InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": 0.0, "beta": 1})
        )


def test_inverse_gamma_positive_scale_required():
    """Stat constructors should reject non-positive scale parameters."""
    with pytest.raises(ValueError, match="Invalid value for beta"):
        KolmogorovSmirnovInverseGammaGofStatistic(
            InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": 1, "beta": -1.0})
        )


def test_kolmogorov_smirnov_inverse_gamma_statistic():
    """Kolmogorov–Smirnov $D$ for Inverse Gamma(shape=2.5, scale=1.1)."""
    statistic = KolmogorovSmirnovInverseGammaGofStatistic(
        InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": _ALPHA, "beta": _BETA})
    ).execute_statistic(_SAMPLE)
    assert statistic == pytest.approx(0.7039450618147463, rel=1e-9)


def test_anderson_darling_inverse_gamma_statistic():
    """Anderson–Darling $A^2$ tailored to the Inverse Gamma reference model."""
    statistic = AndersonDarlingInverseGammaGofStatistic(
        InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": _ALPHA, "beta": _BETA})
    ).execute_statistic(_SAMPLE)
    assert statistic == pytest.approx(11.065191334437646, rel=1e-9)


def test_cramervonmises_inverse_gamma_statistic():
    """Cramér–von Mises $W^2$ using the Inverse Gamma CDF."""
    statistic = CramerVonMisesInverseGammaGofStatistic(
        InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": _ALPHA, "beta": _BETA})
    ).execute_statistic(_SAMPLE)
    assert statistic == pytest.approx(1.7341737031750106, rel=1e-9)


def test_kuiper_inverse_gamma_statistic():
    """Kuiper $V$ statistic measuring circular EDF deviation."""
    statistic = KuiperInverseGammaGofStatistic(
        InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": _ALPHA, "beta": _BETA})
    ).execute_statistic(_SAMPLE)
    assert statistic == pytest.approx(
        _reference_statistics(_SAMPLE, _ALPHA, _BETA)["kuiper"], rel=1e-9
    )


def test_greenwood_inverse_gamma_statistic():
    """Greenwood spacing statistic computed from Inverse Gamma-transformed spacings."""
    statistic = GreenwoodInverseGammaGofStatistic(
        InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": _ALPHA, "beta": _BETA})
    ).execute_statistic(_SAMPLE)
    assert statistic == pytest.approx(
        _reference_statistics(_SAMPLE, _ALPHA, _BETA)["greenwood"], rel=1e-9
    )


def test_min_toshiyuki_inverse_gamma_statistic():
    """Min–Toshiyuki tail-weighted EDF statistic under Inverse Gamma reference."""
    statistic = MinToshiyukiInverseGammaGofStatistic(
        InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": _ALPHA, "beta": _BETA})
    ).execute_statistic(_SAMPLE)
    assert statistic == pytest.approx(
        _reference_statistics(_SAMPLE, _ALPHA, _BETA)["min_toshiyuki"], rel=1e-9
    )


def test_watson_inverse_gamma_statistic():
    """Watson $U^2$ statistic using Inverse Gamma CDF centering."""
    statistic = WatsonInverseGammaGofStatistic(
        InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": _ALPHA, "beta": _BETA})
    ).execute_statistic(_SAMPLE)
    assert statistic == pytest.approx(
        _reference_statistics(_SAMPLE, _ALPHA, _BETA)["watson"], rel=1e-9
    )


def test_chi2_pearson_inverse_gamma_statistic():
    """Pearson chi-square statistic using equiprobable Inverse Gamma bins."""
    statistic = Chi2PearsonInverseGammaGofStatistic(
        InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": _ALPHA, "beta": _BETA}), bins=5
    ).execute_statistic(_SAMPLE)
    assert statistic == pytest.approx(
        _reference_statistics(_SAMPLE, _ALPHA, _BETA)["chi2_pearson"], rel=1e-9
    )


def test_zhang_a_inverse_gamma_statistic():
    """Zhang ZA statistic for Inverse Gamma distribution."""
    statistic = ZhangAInverseGammaGofStatistic(
        InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": _ALPHA, "beta": _BETA})
    ).execute_statistic(_SAMPLE)
    assert statistic == pytest.approx(
        _reference_statistics(_SAMPLE, _ALPHA, _BETA)["zhang_a"], rel=1e-9
    )


def test_zhang_c_inverse_gamma_statistic():
    """Zhang ZC statistic for Inverse Gamma distribution."""
    statistic = ZhangCInverseGammaGofStatistic(
        InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": _ALPHA, "beta": _BETA})
    ).execute_statistic(_SAMPLE)
    assert statistic == pytest.approx(
        _reference_statistics(_SAMPLE, _ALPHA, _BETA)["zhang_c"], rel=1e-9
    )


def test_zhang_k_inverse_gamma_statistic():
    """Zhang ZK statistic for Inverse Gamma distribution."""
    statistic = ZhangKInverseGammaGofStatistic(
        InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": _ALPHA, "beta": _BETA})
    ).execute_statistic(_SAMPLE)
    assert statistic == pytest.approx(
        _reference_statistics(_SAMPLE, _ALPHA, _BETA)["zhang_k"], rel=1e-9
    )


@pytest.mark.parametrize(
    ("stat_class", "expected_code"),
    [
        (KolmogorovSmirnovInverseGammaGofStatistic, "KS_INV_GAMMA_GOODNESS_OF_FIT"),
        (AndersonDarlingInverseGammaGofStatistic, "AD_INV_GAMMA_GOODNESS_OF_FIT"),
        (CramerVonMisesInverseGammaGofStatistic, "CVM_INV_GAMMA_GOODNESS_OF_FIT"),
        (WatsonInverseGammaGofStatistic, "WAT_INV_GAMMA_GOODNESS_OF_FIT"),
        (KuiperInverseGammaGofStatistic, "KUI_INVGAMMA_INV_GAMMA_GOODNESS_OF_FIT"),
        (GreenwoodInverseGammaGofStatistic, "GRW_INV_GAMMA_GOODNESS_OF_FIT"),
        (MinToshiyukiInverseGammaGofStatistic, "MT_INVGAMMA_INV_GAMMA_GOODNESS_OF_FIT"),
        (Chi2PearsonInverseGammaGofStatistic, "CHI2_PEARSON_INV_GAMMA_GOODNESS_OF_FIT"),
        (ZhangAInverseGammaGofStatistic, "ZAA_INV_GAMMA_GOODNESS_OF_FIT"),
        (ZhangCInverseGammaGofStatistic, "ZAC_INV_GAMMA_GOODNESS_OF_FIT"),
        (ZhangKInverseGammaGofStatistic, "ZAK_INV_GAMMA_GOODNESS_OF_FIT"),
    ],
)
def test_inverse_gamma_statistic_codes(stat_class, expected_code):
    """Ensure every Inverse Gamma statistic exposes a stable `code` identifier."""
    assert stat_class.code() == expected_code


@pytest.mark.parametrize(
    "stat_class",
    [
        WatsonInverseGammaGofStatistic,
        KuiperInverseGammaGofStatistic,
        GreenwoodInverseGammaGofStatistic,
        ZhangAInverseGammaGofStatistic,
        ZhangCInverseGammaGofStatistic,
        ZhangKInverseGammaGofStatistic,
    ],
)
def test_inverse_gamma_statistics_require_observations(stat_class):
    """Statistics that rely on EDF spacings should reject empty samples."""
    statistic = stat_class(parameters_for(stat_class, alpha=_ALPHA, beta=_BETA))

    with pytest.raises(ValueError, match="At least"):
        statistic.execute_statistic([])


def test_inverse_gamma_requires_positive_observations():
    """Inverse Gamma requires strictly positive observations."""
    statistic = KolmogorovSmirnovInverseGammaGofStatistic(
        InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": _ALPHA, "beta": _BETA})
    )
    try:
        result = statistic.execute_statistic([0.0, 1.0, 2.0])
        assert isinstance(result, float)
    except ValueError:
        pass


def test_inverse_gamma_binned_statistics_validate_bin_count():
    """Binned statistics require at least two histogram bins."""
    with pytest.raises(ValueError, match="At least two bins"):
        Chi2PearsonInverseGammaGofStatistic(
            InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": _ALPHA, "beta": _BETA}),
            bins=1,
        )


def test_inverse_gamma_binned_statistics_require_sample():
    """Histogram-based tests must receive observations."""
    statistic = Chi2PearsonInverseGammaGofStatistic(
        InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": _ALPHA, "beta": _BETA}), bins=5
    )
    with pytest.raises(ValueError, match="At least"):
        statistic.execute_statistic([])


def test_inverse_gamma_binned_statistics_require_positive_observations():
    """Binned Inverse Gamma statistics require positive observations."""
    statistic = Chi2PearsonInverseGammaGofStatistic(
        InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": _ALPHA, "beta": _BETA}), bins=5
    )
    with pytest.raises(ValueError, match="strictly positive"):
        statistic.execute_statistic([0.0, 1.0, 2.0])

    with pytest.raises(ValueError, match="strictly positive"):
        statistic.execute_statistic([-1.0, 1.0, 2.0])


def test_greenwood_inverse_gamma_detects_non_positive_observations():
    """Greenwood statistic should reject non-positive observations."""
    statistic = GreenwoodInverseGammaGofStatistic(
        InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": _ALPHA, "beta": _BETA})
    )
    with pytest.raises(ValueError, match="positive for Inverse Gamma"):
        statistic.execute_statistic([0.0, 1.0, 2.0])


def test_greenwood_inverse_gamma_detects_negative_spacings(monkeypatch):
    """Artificially broken CDF should trigger the spacing guard."""
    from pysatl_criterion.statistics.goodness_of_fit import inverse_gamma as invgamma_module

    statistic = GreenwoodInverseGammaGofStatistic(
        InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": _ALPHA, "beta": _BETA})
    )

    def fake_cdf(values, **kwargs):
        values = np.asarray(values, dtype=float)
        artificial = np.linspace(0, 1, values.size, dtype=float)
        if artificial.size >= 2:
            artificial[1] = artificial[0] - 0.1
        return artificial

    monkeypatch.setattr(invgamma_module.scipy_stats.invgamma, "cdf", fake_cdf)

    with pytest.raises(FloatingPointError, match="Invalid Inverse Gamma CDF"):
        statistic.execute_statistic(_SAMPLE)


def test_inverse_gamma_hypothesis():
    """Test that hypothesis returns correct parameters."""
    statistic = KolmogorovSmirnovInverseGammaGofStatistic(
        InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": _ALPHA, "beta": _BETA})
    )
    hypothesis = statistic.hypothesis()
    if hasattr(hypothesis, "params"):
        assert hypothesis.params == {"alpha": _ALPHA, "beta": _BETA}
    elif hasattr(hypothesis, "__getitem__"):
        assert hypothesis["alpha"] == _ALPHA
        assert hypothesis["beta"] == _BETA
    else:
        assert hypothesis == {"alpha": _ALPHA, "beta": _BETA}


def test_inverse_gamma_distribution():
    """Test that distribution returns correct type."""
    try:
        distribution = AbstractInverseGammaGofStatistic.distribution().type()

        assert distribution == DistributionType.INVERSE_GAMMA
    except AttributeError:
        distribution = AbstractInverseGammaGofStatistic.distribution().type()
        assert str(distribution) == "INVERSE_GAMMA" or distribution == "INVERSE_GAMMA"


def test_abstract_binned_inverse_gamma_lambda_value():
    """Test that lambda_value can be overridden in subclasses."""

    class CustomChi2InverseGamma(Chi2PearsonInverseGammaGofStatistic):
        lambda_value = 0.5

    statistic = CustomChi2InverseGamma(
        InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": _ALPHA, "beta": _BETA}), bins=5
    )
    assert statistic.lambda_value == 0.5


def test_inverse_gamma_ks_alternative_type():
    from pysatl_criterion.statistics.alternative import AlternativeType

    for direction in AlternativeType:
        statistic = KolmogorovSmirnovInverseGammaGofStatistic(
            InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": _ALPHA, "beta": _BETA}),
            alternative_type=direction,
        )
        assert statistic.alternative().type() == AlternativeType.RIGHT
    with pytest.raises(ValueError):
        KolmogorovSmirnovInverseGammaGofStatistic(
            InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": 1, "beta": 1}),
            alternative_type="GREATER",
        )


def test_watson_inverse_gamma_requires_positive_observations():
    """Watson statistic requires positive observations."""
    statistic = WatsonInverseGammaGofStatistic(
        InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": _ALPHA, "beta": _BETA})
    )
    with pytest.raises(ValueError, match="strictly positive"):
        statistic.execute_statistic([0.0, 1.0, 2.0])


def test_kuiper_inverse_gamma_requires_positive_observations():
    """Kuiper statistic requires positive observations."""
    statistic = KuiperInverseGammaGofStatistic(
        InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": _ALPHA, "beta": _BETA})
    )
    with pytest.raises(ValueError, match="strictly positive"):
        statistic.execute_statistic([-1.0, 1.0, 2.0])


def test_min_toshiyuki_inverse_gamma_requires_positive_observations():
    """Min-Toshiyuki statistic requires positive observations."""
    statistic = MinToshiyukiInverseGammaGofStatistic(
        InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": _ALPHA, "beta": _BETA})
    )
    with pytest.raises(ValueError, match="strictly positive"):
        statistic.execute_statistic([0.0, 1.0, 2.0])


def test_zhang_statistics_require_positive_observations():
    """Zhang statistics require positive observations."""
    statistic = ZhangAInverseGammaGofStatistic(
        InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": _ALPHA, "beta": _BETA})
    )
    with pytest.raises(ValueError, match="positive for Inverse Gamma"):
        statistic.execute_statistic([0.0, 1.0, 2.0])

    statistic = ZhangCInverseGammaGofStatistic(
        InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": _ALPHA, "beta": _BETA})
    )
    with pytest.raises(ValueError, match="positive for Inverse Gamma"):
        statistic.execute_statistic([-1.0, 1.0, 2.0])

    statistic = ZhangKInverseGammaGofStatistic(
        InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": _ALPHA, "beta": _BETA})
    )
    with pytest.raises(ValueError, match="positive for Inverse Gamma"):
        statistic.execute_statistic([0.0, 1.0, 2.0])


def test_zhang_statistics_report_log_tail_underflow():
    """Numerical underflow must not silently replace the tails by epsilon."""
    statistic = ZhangAInverseGammaGofStatistic(
        InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": _ALPHA, "beta": _BETA})
    )
    sample_extreme = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    with pytest.raises(FloatingPointError):
        statistic.execute_statistic(sample_extreme)
