import numpy as np
import pytest

from pysatl_criterion.distribution.distributions import GammaDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit.gamma import (
    AbstractGammaGofStatistic,
    AndersonDarlingGammaGofStatistic,
    Chi2PearsonGammaGofStatistic,
    CramerVonMisesGammaGofStatistic,
    CressieReadGammaGofStatistic,
    GraphAverageDegreeGammaGofStatistic,
    GraphCliqueNumberGammaGofStatistic,
    GraphConnectedComponentsGammaGofStatistic,
    GraphEdgesNumberGammaGofStatistic,
    GraphIndependenceNumberGammaGofStatistic,
    GraphMaxDegreeGammaGofStatistic,
    GreenwoodGammaGofStatistic,
    KolmogorovSmirnovGammaGofStatistic,
    KuiperGammaGofStatistic,
    LikelihoodRatioGammaGofStatistic,
    MinToshiyukiGammaGofStatistic,
    MoranGammaGofStatistic,
    ProbabilityPlotCorrelationGammaGofStatistic,
    WatsonGammaGofStatistic,
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
_SHAPE = 2.5
_SCALE = 1.1


def test_gamma_base_code():
    """Ensure Gamma GOF statistics expose the expected code identifier."""

    assert AbstractGammaGofStatistic.code() == "GAMMA_GOODNESS_OF_FIT"


def test_gamma_positive_shape_required():
    """Stat constructors should reject non-positive shape parameters."""

    with pytest.raises(ValueError, match="Invalid value for alfa"):
        KolmogorovSmirnovGammaGofStatistic(
            GammaDistributionDescriptor.DEFAULT.parse({"alfa": 0.0, "beta": 1})
        )


def test_gamma_positive_scale_required():
    """Stat constructors should reject non-positive scale parameters."""

    with pytest.raises(ValueError, match="Invalid value for beta"):
        KolmogorovSmirnovGammaGofStatistic(
            GammaDistributionDescriptor.DEFAULT.parse({"alfa": 1, "beta": -1.0})
        )


def test_kolmogorov_smirnov_gamma_statistic():
    """Kolmogorov–Smirnov $D$ for Gamma(shape=2.5, scale=1.1).

    Parameters
    ----------
    sample : list[float]
        Ordered observations used to compute the EDF against the Gamma CDF.

    Returns
    -------
    float
        Maximum absolute deviation (expected 0.2814182084684763) as
        reported in Kolmogorov (1933) and Smirnov (1948).
    """

    statistic = KolmogorovSmirnovGammaGofStatistic(
        GammaDistributionDescriptor.DEFAULT.parse({"alfa": _SHAPE, "beta": 1 / _SCALE})
    ).execute_statistic(_SAMPLE)
    assert statistic == pytest.approx(0.2814182084684763, rel=1e-9)


def test_anderson_darling_gamma_statistic():
    """Anderson–Darling $A^2$ tailored to the Gamma reference model.

    Parameters
    ----------
    sample : list[float]
        Synthetic Gamma observations with shape=2.5 and scale=1.1.

    Returns
    -------
    float
        Weighted EDF integral (expected 1.5834952876091002) following
        Anderson & Darling (1952).
    """

    statistic = AndersonDarlingGammaGofStatistic(
        GammaDistributionDescriptor.DEFAULT.parse({"alfa": _SHAPE, "beta": 1 / _SCALE})
    ).execute_statistic(_SAMPLE)
    assert statistic == pytest.approx(1.5834952876091002, rel=1e-9)


def test_cramervonmises_gamma_statistic():
    """Cramér–von Mises $W^2$ using the Gamma CDF.

    Parameters
    ----------
    sample : list[float]
        Sample used to form the quadratic EDF functional.

    Returns
    -------
    float
        Expected value 0.3047570587738219 per Cramér (1928) and von Mises (1931).
    """

    statistic = CramerVonMisesGammaGofStatistic(
        GammaDistributionDescriptor.DEFAULT.parse({"alfa": _SHAPE, "beta": 1 / _SCALE})
    ).execute_statistic(_SAMPLE)
    assert statistic == pytest.approx(0.3047570587738219, rel=1e-9)


def test_kuiper_gamma_statistic():
    """Kuiper $V$ statistic measuring circular EDF deviation.

    Parameters
    ----------
    sample : list[float]
        Ordered sample used to compute $D^+$ and $D^-$ terms via Gamma CDF values.

    Returns
    -------
    float
        Sum of extreme deviations (expected 0.3021236878101907) after Kuiper (1960).
    """

    statistic = KuiperGammaGofStatistic(
        GammaDistributionDescriptor.DEFAULT.parse({"alfa": _SHAPE, "beta": 1 / _SCALE})
    ).execute_statistic(_SAMPLE)
    assert statistic == pytest.approx(0.3021236878101907, rel=1e-9)


def test_greenwood_gamma_statistic():
    """Greenwood spacing statistic computed from Gamma-transformed spacings.

    Parameters
    ----------
    sample : list[float]
        Sample transformed through the Gamma CDF to obtain uniform spacings.

    Returns
    -------
    float
        Sum of squared spacings (expected 0.14183310386065612) from Greenwood (1946).
    """

    statistic = GreenwoodGammaGofStatistic(
        GammaDistributionDescriptor.DEFAULT.parse({"alfa": _SHAPE, "beta": 1 / _SCALE})
    ).execute_statistic(_SAMPLE)
    assert statistic == pytest.approx(0.14183310386065612, rel=1e-9)


def test_min_toshiyuki_gamma_statistic():
    """Min–Toshiyuki tail-weighted EDF statistic under Gamma reference.

    Parameters
    ----------
    sample : list[float]
        Observations mapped onto the Gamma CDF to emphasize tail deviations.

    Returns
    -------
    float
        Tail-sensitive score (expected 1.586859983429235) from direct evaluation
        of the documented local formula.
    """

    statistic = MinToshiyukiGammaGofStatistic(
        GammaDistributionDescriptor.DEFAULT.parse({"alfa": _SHAPE, "beta": 1 / _SCALE})
    ).execute_statistic(_SAMPLE)
    assert statistic == pytest.approx(1.586859983429235, rel=1e-9)


def test_watson_gamma_statistic():
    """Watson $U^2$ statistic using Gamma CDF centering.

    Parameters
    ----------
    sample : list[float]
        Sample leveraged to compute the rotation-invariant EDF score.

    Returns
    -------
    float
        Expected value 0.06149897222339809 after Watson (1961).
    """

    statistic = WatsonGammaGofStatistic(
        GammaDistributionDescriptor.DEFAULT.parse({"alfa": _SHAPE, "beta": 1 / _SCALE})
    ).execute_statistic(_SAMPLE)
    assert statistic == pytest.approx(0.06149897222339809, rel=1e-9)


def test_moran_gamma_statistic():
    """Moran log-spacing statistic computed from Gamma-transformed spacings.

    Parameters
    ----------
    sample : list[float]
        Observations mapped through the Gamma CDF to obtain strictly positive spacings.

    Returns
    -------
    float
        Sum of negative log-scaled spacings (expected 3.906045589439027)
        with the documented n scaling.
    """

    statistic = MoranGammaGofStatistic(
        GammaDistributionDescriptor.DEFAULT.parse({"alfa": _SHAPE, "beta": 1 / _SCALE})
    ).execute_statistic(_SAMPLE)
    assert statistic == pytest.approx(3.906045589439027, rel=1e-9)


def test_chi2_pearson_gamma_statistic():
    """Pearson chi-square statistic using equiprobable Gamma bins.

    Parameters
    ----------
    sample : list[float]
        Sample histogrammed with Gamma-quantile binning (five bins).

    Returns
    -------
    float
        Discrepancy score (expected 3.0) as in Pearson (1900).
    """

    statistic = Chi2PearsonGammaGofStatistic(
        GammaDistributionDescriptor.DEFAULT.parse({"alfa": _SHAPE, "beta": 1 / _SCALE}), bins=5
    ).execute_statistic(_SAMPLE)
    assert statistic == pytest.approx(3.0, rel=1e-9)


def test_likelihood_ratio_gamma_statistic():
    """Likelihood-ratio ($G$) statistic formed on Gamma quantile bins."""

    statistic = LikelihoodRatioGammaGofStatistic(
        GammaDistributionDescriptor.DEFAULT.parse({"alfa": _SHAPE, "beta": 1 / _SCALE}), bins=5
    ).execute_statistic(_SAMPLE)
    assert statistic == pytest.approx(4.865581297297973, rel=1e-9)


def test_cressie_read_gamma_statistic():
    """Cressie–Read power divergence with the default $\\lambda=2/3$."""

    statistic = CressieReadGammaGofStatistic(
        GammaDistributionDescriptor.DEFAULT.parse({"alfa": _SHAPE, "beta": 1 / _SCALE}),
        power=2 / 3,
        bins=5,
    ).execute_statistic(_SAMPLE)
    assert statistic == pytest.approx(3.352003528728041, rel=1e-9)


def test_probability_plot_correlation_gamma_statistic():
    """Probability-plot correlation coefficient deviation under Gamma fit."""

    statistic = ProbabilityPlotCorrelationGammaGofStatistic(
        GammaDistributionDescriptor.DEFAULT.parse({"alfa": _SHAPE, "beta": 1 / _SCALE})
    ).execute_statistic(_SAMPLE)
    assert statistic == pytest.approx(0.012272781981296887, rel=1e-9)


def test_graph_edges_number_gamma_statistic():
    """Graph edges count on Gamma-CDF transformed sample."""

    statistic = GraphEdgesNumberGammaGofStatistic(
        GammaDistributionDescriptor.DEFAULT.parse({"alfa": _SHAPE, "beta": 1 / _SCALE})
    ).execute_statistic(_SAMPLE)
    assert statistic == pytest.approx(4.0, rel=1e-12)


def test_graph_max_degree_gamma_statistic():
    """Maximum node degree observed in the Gamma proximity graph."""

    statistic = GraphMaxDegreeGammaGofStatistic(
        GammaDistributionDescriptor.DEFAULT.parse({"alfa": _SHAPE, "beta": 1 / _SCALE})
    ).execute_statistic(_SAMPLE)
    assert statistic == pytest.approx(2.0, rel=1e-12)


def test_graph_average_degree_gamma_statistic():
    """Average vertex degree after Gamma probability integral transform."""

    statistic = GraphAverageDegreeGammaGofStatistic(
        GammaDistributionDescriptor.DEFAULT.parse({"alfa": _SHAPE, "beta": 1 / _SCALE})
    ).execute_statistic(_SAMPLE)
    assert statistic == pytest.approx(0.8, rel=1e-12)


def test_graph_connected_components_gamma_statistic():
    """Connected components count on the Gamma-derived proximity graph."""

    statistic = GraphConnectedComponentsGammaGofStatistic(
        GammaDistributionDescriptor.DEFAULT.parse({"alfa": _SHAPE, "beta": 1 / _SCALE})
    ).execute_statistic(_SAMPLE)
    assert statistic == pytest.approx(6.0, rel=1e-12)


def test_graph_clique_number_gamma_statistic():
    """Largest clique size after transforming the sample via the Gamma CDF."""

    statistic = GraphCliqueNumberGammaGofStatistic(
        GammaDistributionDescriptor.DEFAULT.parse({"alfa": _SHAPE, "beta": 1 / _SCALE})
    ).execute_statistic(_SAMPLE)
    assert statistic == pytest.approx(2.0, rel=1e-12)


def test_graph_independence_number_gamma_statistic():
    """Independence number computed on the Gamma-induced proximity graph."""

    statistic = GraphIndependenceNumberGammaGofStatistic(
        GammaDistributionDescriptor.DEFAULT.parse({"alfa": _SHAPE, "beta": 1 / _SCALE})
    ).execute_statistic(_SAMPLE)
    assert statistic == pytest.approx(7.0, rel=1e-12)


@pytest.mark.parametrize(
    ("stat_class", "expected_code"),
    [
        (KolmogorovSmirnovGammaGofStatistic, "KS_GAMMA_GOODNESS_OF_FIT"),
        (AndersonDarlingGammaGofStatistic, "AD_GAMMA_GOODNESS_OF_FIT"),
        (CramerVonMisesGammaGofStatistic, "CVM_GAMMA_GOODNESS_OF_FIT"),
        (WatsonGammaGofStatistic, "WAT_GAMMA_GOODNESS_OF_FIT"),
        (KuiperGammaGofStatistic, "KUI_GAMMA_GOODNESS_OF_FIT"),
        (GreenwoodGammaGofStatistic, "GRW_GAMMA_GOODNESS_OF_FIT"),
        (MoranGammaGofStatistic, "MOR_GAMMA_GOODNESS_OF_FIT"),
        (MinToshiyukiGammaGofStatistic, "MT_GAMMA_GOODNESS_OF_FIT"),
        (Chi2PearsonGammaGofStatistic, "CHI2_PEARSON_GAMMA_GOODNESS_OF_FIT"),
        (LikelihoodRatioGammaGofStatistic, "G_TEST_GAMMA_GOODNESS_OF_FIT"),
        (CressieReadGammaGofStatistic, "CRESSIE_READ_GAMMA_GOODNESS_OF_FIT"),
        (
            ProbabilityPlotCorrelationGammaGofStatistic,
            "PPCC_GAMMA_GOODNESS_OF_FIT",
        ),
        (
            GraphEdgesNumberGammaGofStatistic,
            "EDGESNUMBER_GRAPH_GAMMA_GOODNESS_OF_FIT",
        ),
        (
            GraphMaxDegreeGammaGofStatistic,
            "MAXDEGREE_GRAPH_GAMMA_GOODNESS_OF_FIT",
        ),
        (
            GraphAverageDegreeGammaGofStatistic,
            "AVGDEGREE_GRAPH_GAMMA_GOODNESS_OF_FIT",
        ),
        (
            GraphConnectedComponentsGammaGofStatistic,
            "CONNECTEDCOMPONENTS_GRAPH_GAMMA_GOODNESS_OF_FIT",
        ),
        (
            GraphCliqueNumberGammaGofStatistic,
            "CLIQUENUMBER_GRAPH_GAMMA_GOODNESS_OF_FIT",
        ),
        (
            GraphIndependenceNumberGammaGofStatistic,
            "INDEPENDENCENUMBER_GRAPH_GAMMA_GOODNESS_OF_FIT",
        ),
    ],
)
def test_gamma_statistic_codes(stat_class, expected_code):
    """Ensure every Gamma statistic exposes a stable `code` identifier."""

    assert stat_class.code() == expected_code


@pytest.mark.parametrize(
    "stat_class",
    [WatsonGammaGofStatistic, KuiperGammaGofStatistic, MoranGammaGofStatistic],
)
def test_gamma_statistics_require_observations(stat_class):
    """Statistics that rely on EDF spacings should reject empty samples."""

    statistic = stat_class(parameters_for(stat_class, alfa=_SHAPE, beta=1 / _SCALE))
    with pytest.raises(ValueError, match="At least one observation"):
        statistic.execute_statistic([])


def test_moran_gamma_detects_non_positive_spacings():
    """Duplicate-valued samples give positive infinity by the log-spacing limit."""

    statistic = MoranGammaGofStatistic(
        GammaDistributionDescriptor.DEFAULT.parse({"alfa": _SHAPE, "beta": 1 / _SCALE})
    )
    assert statistic.execute_statistic([1.0, 1.0, 1.0]) == np.inf


def test_gamma_binned_statistics_validate_bin_count():
    """Binned statistics require at least two histogram bins."""

    with pytest.raises(ValueError, match="At least two bins"):
        Chi2PearsonGammaGofStatistic(
            GammaDistributionDescriptor.DEFAULT.parse({"alfa": _SHAPE, "beta": 1 / _SCALE}), bins=1
        )


def test_gamma_binned_statistics_require_sample():
    """Histogram-based tests must receive observations."""

    statistic = Chi2PearsonGammaGofStatistic(
        GammaDistributionDescriptor.DEFAULT.parse({"alfa": _SHAPE, "beta": 1 / _SCALE}), bins=5
    )
    with pytest.raises(ValueError, match="At least one observation"):
        statistic.execute_statistic([])


def test_probability_plot_gamma_requires_minimum_sample():
    """PPCC metric needs at least two points to compute a correlation."""

    statistic = ProbabilityPlotCorrelationGammaGofStatistic(
        GammaDistributionDescriptor.DEFAULT.parse({"alfa": _SHAPE, "beta": 1 / _SCALE})
    )
    with pytest.raises(ValueError, match="At least two observations"):
        statistic.execute_statistic([1.0])


def test_probability_plot_gamma_detects_degenerate_sample():
    """Identical points yield zero variance and should fail PPCC computation."""

    statistic = ProbabilityPlotCorrelationGammaGofStatistic(
        GammaDistributionDescriptor.DEFAULT.parse({"alfa": _SHAPE, "beta": 1 / _SCALE})
    )
    with pytest.raises(ValueError, match="Degenerate data"):
        statistic.execute_statistic([1.0, 1.0, 1.0])


def test_graph_gamma_statistics_require_sample():
    """Graph-based Gamma tests should reject empty datasets."""

    statistic = GraphEdgesNumberGammaGofStatistic(
        GammaDistributionDescriptor.DEFAULT.parse({"alfa": _SHAPE, "beta": 1 / _SCALE})
    )
    with pytest.raises(ValueError, match="At least one observation"):
        statistic.execute_statistic([])


def test_greenwood_gamma_detects_negative_spacings(monkeypatch):
    """Artificially broken CDF should trigger the spacing guard."""

    from pysatl_criterion.statistics.goodness_of_fit import gamma as gamma_module

    statistic = GreenwoodGammaGofStatistic(
        GammaDistributionDescriptor.DEFAULT.parse({"alfa": _SHAPE, "beta": 1 / _SCALE})
    )

    def fake_cdf(values, **kwargs):
        values = np.asarray(values, dtype=float)
        artificial = np.linspace(0, 1, values.size, dtype=float)
        if artificial.size >= 2:
            artificial[1] = artificial[0] - 0.1  # force a negative spacing
        return artificial

    monkeypatch.setattr(gamma_module.scipy_stats.gamma, "cdf", fake_cdf)

    with pytest.raises(ValueError, match="Spacings must be non-negative"):
        statistic.execute_statistic(_SAMPLE)
