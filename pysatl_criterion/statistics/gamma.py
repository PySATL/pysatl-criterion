from __future__ import annotations

from abc import ABC

import numpy as np
import scipy.stats as scipy_stats
from typing_extensions import override

from pysatl_criterion.statistics.common import (
    ADStatistic,
    Chi2Statistic,
    CrammerVonMisesStatistic,
    KSStatistic,
    LillieforsTest,
    MinToshiyukiStatistic,
)
from pysatl_criterion.statistics.goodness_of_fit import AbstractGoodnessOfFitStatistic
from pysatl_criterion.statistics.graph_goodness_of_fit import (
    AbstractGraphTestStatistic,
    GraphAverageDegreeTestStatistic,
    GraphCliqueNumberTestStatistic,
    GraphConnectedComponentsTestStatistic,
    GraphEdgesNumberTestStatistic,
    GraphIndependenceNumberTestStatistic,
    GraphMaxDegreeTestStatistic,
)


class AbstractGammaGofStatistic(AbstractGoodnessOfFitStatistic, ABC):
    """
    Abstract base class for Gamma distribution goodness-of-fit statistics.
    """

    def __init__(self, shape: float = 1.0, scale: float = 1.0):
        """
        Initialize Gamma distribution goodness-of-fit statistic.

        :param shape: shape parameter (alpha) > 0.
        :param scale: scale parameter (theta) > 0.
        :raises ValueError: if shape or scale is not positive.
        """
        if shape <= 0:
            raise ValueError("Shape must be positive.")
        if scale <= 0:
            raise ValueError("Scale must be positive.")
        self.shape = shape
        self.scale = scale

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for Gamma distribution statistics.

        :return: string code in format "GAMMA_{parent_code}".
        """
        return f"GAMMA_{AbstractGoodnessOfFitStatistic.code()}"


class KolmogorovSmirnovGammaGofStatistic(AbstractGammaGofStatistic, KSStatistic):
    """
    Kolmogorov–Smirnov EDF test computed with the Gamma reference CDF.

    Compares the empirical distribution function with the theoretical Gamma
    distribution function. Sensitive to differences in both location and shape.
    """

    @override
    def __init__(
        self,
        alternative="two-sided",
        mode="auto",
        shape: float = 1.0,
        scale: float = 1.0,
    ):
        AbstractGammaGofStatistic.__init__(self, shape=shape, scale=scale)
        KSStatistic.__init__(self, alternative=alternative, mode=mode)

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "KS".
        """
        return "KS"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "KS_GAMMA_{parent_code}".
        """
        short_code = KolmogorovSmirnovGammaGofStatistic.short_code()
        return f"{short_code}_{AbstractGammaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Kolmogorov-Smirnov test statistic for Gamma distribution.

        :param rvs: array of observations assumed to follow Gamma(shape, scale).
        :return: Kolmogorov–Smirnov D statistic computed with the Gamma CDF.
        """

        sorted_rvs = np.sort(np.asarray(rvs))
        cdf_vals = scipy_stats.gamma.cdf(sorted_rvs, a=self.shape, scale=self.scale)
        return KSStatistic.execute_statistic(self, sorted_rvs, cdf_vals)


class LillieforsGammaGofStatistic(AbstractGammaGofStatistic, LillieforsTest):
    """
    Lilliefors correction that re-estimates Gamma parameters before KS.

    Modification of Kolmogorov-Smirnov test for the case when Gamma distribution
    parameters are estimated from the data rather than specified a priori.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "LILLIE".
        """
        return "LILLIE"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "LILLIE_GAMMA_{parent_code}".
        """
        short_code = LillieforsGammaGofStatistic.short_code()
        return f"{short_code}_{AbstractGammaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Lilliefors test statistic for Gamma distribution.

        :param rvs: array of observations assumed to follow Gamma(shape, scale).
        :return: Lilliefors-adjusted Kolmogorov–Smirnov statistic with estimated Gamma parameters.
        :raises ValueError: if sample is empty or mean/variance is not positive.
        """

        sample = np.asarray(rvs, dtype=float)
        n = sample.size
        if n == 0:
            raise ValueError("At least one observation is required for the Lilliefors statistic.")

        mean = np.mean(sample)
        var = np.var(sample, ddof=1)
        if mean <= 0 or var <= 0:
            raise ValueError(
                "Sample mean and variance must be positive for Gamma parameter estimation."
            )

        shape_hat = mean**2 / var
        scale_hat = var / mean
        sorted_sample = np.sort(sample)
        cdf_vals = scipy_stats.gamma.cdf(sorted_sample, a=shape_hat, scale=scale_hat)
        return super(LillieforsTest, self).execute_statistic(sorted_sample, cdf_vals)


class AndersonDarlingGammaGofStatistic(AbstractGammaGofStatistic, ADStatistic):
    """
    Anderson–Darling EDF statistic fitted to the Gamma distribution.

    Modification of Kolmogorov-Smirnov test that gives more weight to the tails
    of the distribution, making it more sensitive to tail deviations.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "AD".
        """
        return "AD"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "AD_GAMMA_{parent_code}".
        """
        short_code = AndersonDarlingGammaGofStatistic.short_code()
        return f"{short_code}_{AbstractGammaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Anderson-Darling test statistic for Gamma distribution.

        :param rvs: array of observations assumed to follow Gamma(shape, scale).
        :return: Anderson–Darling A^2 statistic tailored to the Gamma model.
        """

        sorted_rvs = np.sort(np.asarray(rvs))
        log_cdf = scipy_stats.gamma.logcdf(sorted_rvs, a=self.shape, scale=self.scale)
        log_sf = scipy_stats.gamma.logsf(sorted_rvs, a=self.shape, scale=self.scale)
        return super().execute_statistic(sorted_rvs, log_cdf=log_cdf, log_sf=log_sf)


class CramerVonMisesGammaGofStatistic(AbstractGammaGofStatistic, CrammerVonMisesStatistic):
    """
    Cramér–von Mises quadratic EDF test specialized for Gamma samples.

    Goodness-of-fit test that measures the integrated squared difference between
    the empirical and theoretical cumulative distribution functions.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "CVM".
        """
        return "CVM"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "CVM_GAMMA_{parent_code}".
        """
        short_code = CramerVonMisesGammaGofStatistic.short_code()
        return f"{short_code}_{AbstractGammaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Cramér-von Mises test statistic for Gamma distribution.

        :param rvs: array of observations assumed to follow Gamma(shape, scale).
        :return: Cramér–von Mises W^2 statistic using the Gamma CDF.
        """

        sorted_rvs = np.sort(np.asarray(rvs))
        cdf_vals = scipy_stats.gamma.cdf(sorted_rvs, a=self.shape, scale=self.scale)
        return CrammerVonMisesStatistic.execute_statistic(self, sorted_rvs, cdf_vals)


class WatsonGammaGofStatistic(AbstractGammaGofStatistic):
    """
    Watson's rotation-invariant EDF statistic using Gamma CDF values.

    Modification of Cramér-von Mises test that is invariant under location changes,
    making it suitable for circular data analysis.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "WAT".
        """
        return "WAT"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "WAT_GAMMA_{parent_code}".
        """
        short_code = WatsonGammaGofStatistic.short_code()
        return f"{short_code}_{AbstractGammaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Watson test statistic for Gamma distribution.

        :param rvs: array of observations assumed to follow Gamma(shape, scale).
        :return: Watson U^2 statistic derived from the Gamma CDF values.
        :raises ValueError: if sample is empty.
        """

        sorted_rvs = np.sort(np.asarray(rvs))
        n = len(sorted_rvs)
        if n == 0:
            raise ValueError(
                "At least one observation is required to compute the Watson statistic."
            )

        cdf_vals = scipy_stats.gamma.cdf(sorted_rvs, a=self.shape, scale=self.scale)
        u = (2 * np.arange(1, n + 1) - 1) / (2 * n)
        diff = cdf_vals - u
        w_squared = 1.0 / (12 * n) + np.sum(diff**2)
        mean_adj = np.sum(cdf_vals) - n / 2
        return float(w_squared - (mean_adj**2) / n)


class KuiperGammaGofStatistic(AbstractGammaGofStatistic):
    """
    Kuiper's circular EDF statistic after Gamma probability transform.

    Variant of Kolmogorov-Smirnov test that sums the maximum positive and negative
    deviations, making it equally sensitive to deviations in both tails.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "KUI".
        """
        return "KUI"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "KUI_GAMMA_{parent_code}".
        """
        short_code = KuiperGammaGofStatistic.short_code()
        return f"{short_code}_{AbstractGammaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Kuiper test statistic for Gamma distribution.

        :param rvs: array of observations assumed to follow Gamma(shape, scale).
        :return: Kuiper V = D+ + D- statistic after the Gamma probability integral transform.
        :raises ValueError: if sample is empty.
        """

        sorted_rvs = np.sort(np.asarray(rvs))
        cdf_vals = scipy_stats.gamma.cdf(sorted_rvs, a=self.shape, scale=self.scale)

        n = len(sorted_rvs)
        if n == 0:
            raise ValueError(
                "At least one observation is required to compute the Kuiper statistic."
            )

        i = np.arange(1, n + 1)
        d_plus = np.max(i / n - cdf_vals)
        d_minus = np.max(cdf_vals - (i - 1) / n)
        return d_plus + d_minus


class GreenwoodGammaGofStatistic(AbstractGammaGofStatistic):
    """
    Greenwood spacing statistic measuring uniformized Gamma gaps.

    Test based on sum of squared spacings between consecutive Gamma CDF values.
    Sensitive to clustering of observations.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "GRW".
        """
        return "GRW"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "GRW_GAMMA_{parent_code}".
        """
        short_code = GreenwoodGammaGofStatistic.short_code()
        return f"{short_code}_{AbstractGammaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Greenwood test statistic for Gamma distribution.

        :param rvs: array of observations assumed to follow Gamma(shape, scale).
        :return: Greenwood spacing statistic G = sum(D_i^2) where spacings D_i are computed
        from Gamma CDF values.
        :raises ValueError: if spacings are negative.
        """

        sorted_rvs = np.sort(np.asarray(rvs))
        cdf_vals = scipy_stats.gamma.cdf(sorted_rvs, a=self.shape, scale=self.scale)
        spacings = np.diff(np.concatenate(([0.0], cdf_vals, [1.0])))
        if np.any(spacings < 0):
            raise ValueError("Spacings must be non-negative; check input data ordering.")
        return float(np.sum(spacings**2))


class MoranGammaGofStatistic(AbstractGammaGofStatistic):
    """
    Moran log-spacing statistic applied to Gamma-transformed uniforms.

    Test based on sum of log-transformed spacings between consecutive Gamma CDF values.
    Sensitive to uniformity of probability integral transform.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "MOR".
        """
        return "MOR"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "MOR_GAMMA_{parent_code}".
        """
        short_code = MoranGammaGofStatistic.short_code()
        return f"{short_code}_{AbstractGammaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Moran test statistic for Gamma distribution.

        :param rvs: array of observations assumed to follow Gamma(shape, scale).
        :return: Moran spacing statistic M = -sum(log(n * D_i)) based on Gamma CDF spacings.
        :raises ValueError: if sample is empty or spacings are not strictly positive.
        """

        sorted_rvs = np.sort(np.asarray(rvs))
        n = len(sorted_rvs)
        if n == 0:
            raise ValueError("At least one observation is required to compute the Moran statistic.")

        cdf_vals = scipy_stats.gamma.cdf(sorted_rvs, a=self.shape, scale=self.scale)
        spacings = np.diff(np.concatenate(([0.0], cdf_vals, [1.0])))
        if np.any(spacings <= 0):
            raise ValueError("Spacings must be strictly positive for the Moran statistic.")

        scaled_spacings = n * spacings
        return float(-np.sum(np.log(scaled_spacings)))


class MinToshiyukiGammaGofStatistic(AbstractGammaGofStatistic, MinToshiyukiStatistic):
    """
    Min–Toshiyuki tail-sensitive EDF statistic under a Gamma model.

    EDF statistic with adaptive tail sensitivity that up-weights deviations
    near the distribution tails using Gamma CDF values.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "MT".
        """
        return "MT"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "MT_GAMMA_{parent_code}".
        """
        short_code = MinToshiyukiGammaGofStatistic.short_code()
        return f"{short_code}_{AbstractGammaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Min-Toshiyuki test statistic for Gamma distribution.

        :param rvs: array of observations assumed to follow Gamma(shape, scale).
        :return: Min–Toshiyuki statistic that up-weights EDF deviations near the distribution tails
        using Gamma CDF values.
        """

        sorted_rvs = np.sort(np.asarray(rvs))
        cdf_vals = scipy_stats.gamma.cdf(sorted_rvs, a=self.shape, scale=self.scale)
        return MinToshiyukiStatistic.execute_statistic(self, cdf_vals)


class AbstractBinnedGammaGofStatistic(AbstractGammaGofStatistic, Chi2Statistic, ABC):
    """
    Base class for Gamma GOF tests built on equiprobable histogram bins.

    Provides common infrastructure for chi-squared type tests that bin data
    using Gamma quantile function to ensure equal theoretical probability per bin.
    """

    lambda_value: float = 1.0

    def __init__(self, bins: int = 8, shape: float = 1.0, scale: float = 1.0):
        if bins < 2:
            raise ValueError("At least two bins are required for binned Gamma statistics.")
        self.bins = bins
        super().__init__(shape=shape, scale=scale)
        self.lambda_value = getattr(self, "lambda_value", 1.0)

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the binned chi-squared test statistic for Gamma distribution.

        :param rvs: array of observations assumed to follow Gamma(shape, scale).
        :return: chi-squared test statistic value computed on equiprobable Gamma bins.
        :raises ValueError: if sample is empty.
        """
        counts, expected = self._counts_and_expected(rvs)
        return float(
            Chi2Statistic.execute_statistic(self, counts, expected, lambda_=self.lambda_value)
        )


class Chi2PearsonGammaGofStatistic(AbstractBinnedGammaGofStatistic):
    """
    Pearson chi-square frequency test based on Gamma equiprobable bins.

    Implements Karl Pearson's (1900) frequency test by binning via the Gamma
    quantile function so each bin has equal theoretical probability.
    """

    lambda_value = 1.0

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "CHI2_PEARSON".
        """
        return "CHI2_PEARSON"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "CHI2_PEARSON_GAMMA_{parent_code}".
        """
        short_code = Chi2PearsonGammaGofStatistic.short_code()
        return f"{short_code}_{AbstractGammaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute Pearson chi-square statistic for Gamma distribution.

        :param rvs: array of observations assumed to follow Gamma(shape, scale).
        :return: Pearson chi-square statistic computed on equiprobable Gamma bins.
        """

        return super().execute_statistic(rvs, **kwargs)


class LikelihoodRatioGammaGofStatistic(AbstractBinnedGammaGofStatistic):
    """
    Log-likelihood ratio (G-test) for Gamma reference distribution.

    Follows the classical G-test described by S. S. Wilks (1935) and tests
    histogram counts against expected Gamma frequencies.
    """

    lambda_value = 0.0

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "G_TEST".
        """
        return "G_TEST"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "G_TEST_GAMMA_{parent_code}".
        """
        short_code = LikelihoodRatioGammaGofStatistic.short_code()
        return f"{short_code}_{AbstractGammaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute likelihood-ratio statistic for Gamma distribution.

        :param rvs: array of observations assumed to follow Gamma(shape, scale).
        :return: likelihood-ratio statistic using equiprobable Gamma quantile bins.
        """

        return super().execute_statistic(rvs, **kwargs)


class CressieReadGammaGofStatistic(AbstractBinnedGammaGofStatistic):
    """
    Cressie–Read power-divergence statistic for Gamma data.

    Power-divergence statistic bridging Pearson (lambda=1) and G-tests (lambda=0).
    Defaults to the recommended lambda=2/3 value from Read & Cressie (1988).
    """

    def __init__(
        self,
        power: float = 2 / 3,
        bins: int = 8,
        shape: float = 1.0,
        scale: float = 1.0,
    ):
        self.lambda_value = power
        super().__init__(bins=bins, shape=shape, scale=scale)

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "CRESSIE_READ".
        """
        return "CRESSIE_READ"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "CRESSIE_READ_GAMMA_{parent_code}".
        """
        short_code = CressieReadGammaGofStatistic.short_code()
        return f"{short_code}_{AbstractGammaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute Cressie-Read power-divergence statistic for Gamma distribution.

        :param rvs: array of observations assumed to follow Gamma(shape, scale).
        :return: power-divergence statistic bridging Pearson and G-tests.
        """

        return super().execute_statistic(rvs, **kwargs)


class ProbabilityPlotCorrelationGammaGofStatistic(AbstractGammaGofStatistic):
    """
    Filliben-style PPCC statistic comparing Gamma quantiles to the sample.

    Probability plot correlation coefficient test that measures linear alignment
    between ordered sample values and theoretical Gamma quantiles.
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "PPCC".
        """
        return "PPCC"

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "PPCC_GAMMA_{parent_code}".
        """
        short_code = ProbabilityPlotCorrelationGammaGofStatistic.short_code()
        return f"{short_code}_{AbstractGammaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the probability plot correlation coefficient test for Gamma distribution.

        :param rvs: array of observations assumed to follow Gamma(shape, scale).
        :return: one minus the probability-plot correlation coefficient. Values near zero
        indicate strong linear alignment with theoretical Gamma quantiles.
        :raises ValueError: if sample has fewer than 2 observations or data is degenerate.
        """

        sample = np.sort(np.asarray(rvs, dtype=float))
        n = sample.size
        if n < 2:
            raise ValueError("At least two observations are required for the PPCC statistic.")

        plotting_positions = (np.arange(1, n + 1) - 0.375) / (n + 0.25)
        expected = scipy_stats.gamma.ppf(plotting_positions, a=self.shape, scale=self.scale)

        sample_centered = sample - np.mean(sample)
        expected_centered = expected - np.mean(expected)
        numerator = np.sum(sample_centered * expected_centered)
        denominator = np.sqrt(np.sum(sample_centered**2) * np.sum(expected_centered**2))
        if denominator == 0:
            raise ValueError("Degenerate data encountered while computing PPCC statistic.")

        corr = numerator / denominator
        return float(1.0 - corr)


class AbstractGraphGammaGofStatistic(AbstractGammaGofStatistic, AbstractGraphTestStatistic):
    """
    Base class for Gamma graph-based GOF statistics using EDF transforms.

    Combines Gamma distribution testing with graph-theoretic statistics by
    transforming data through Gamma CDF and analyzing resulting structure.
    """

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for graph-based Gamma statistics.

        :return: string code in format "GRAPH_GAMMA_{parent_code}".
        """
        parent_code = AbstractGammaGofStatistic.code()
        return f"GRAPH_{parent_code}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the graph-based test statistic for Gamma distribution.

        :param rvs: array of observations assumed to follow Gamma(shape, scale).
        :return: graph-based test statistic value computed on Gamma-CDF transformed data.
        :raises ValueError: if sample is empty.
        """
        transformed_sample = self._transform_sample(rvs)
        return self._evaluate_graph_statistic(transformed_sample, **kwargs)


class GraphEdgesNumberGammaGofStatistic(
    AbstractGraphGammaGofStatistic, GraphEdgesNumberTestStatistic
):
    """
    Counts edges in the proximity graph built on Gamma-CDF spacings.

    Graph-based test that counts edges in proximity graph constructed from
    Gamma probability integral transform of the data.
    """

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "{short_code}_GRAPH_GAMMA_{parent_code}".
        """
        parent_code = AbstractGraphGammaGofStatistic.code()
        short_code = GraphEdgesNumberGammaGofStatistic.short_code()
        return f"{short_code}_{parent_code}"


class GraphMaxDegreeGammaGofStatistic(AbstractGraphGammaGofStatistic, GraphMaxDegreeTestStatistic):
    """
    Maximum degree in the Gamma-induced proximity graph.

    Graph-based test that computes maximum vertex degree in proximity graph
    constructed from Gamma probability integral transform of the data.
    """

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "{short_code}_GRAPH_GAMMA_{parent_code}".
        """
        parent_code = AbstractGraphGammaGofStatistic.code()
        short_code = GraphMaxDegreeGammaGofStatistic.short_code()
        return f"{short_code}_{parent_code}"


class GraphAverageDegreeGammaGofStatistic(
    AbstractGraphGammaGofStatistic, GraphAverageDegreeTestStatistic
):
    """
    Average vertex degree of the Gamma proximity graph.

    Graph-based test that computes average vertex degree in proximity graph
    constructed from Gamma probability integral transform of the data.
    """

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "{short_code}_GRAPH_GAMMA_{parent_code}".
        """
        parent_code = AbstractGraphGammaGofStatistic.code()
        short_code = GraphAverageDegreeGammaGofStatistic.short_code()
        return f"{short_code}_{parent_code}"


class GraphConnectedComponentsGammaGofStatistic(
    AbstractGraphGammaGofStatistic, GraphConnectedComponentsTestStatistic
):
    """
    Number of connected components in the Gamma proximity graph.

    Graph-based test that counts connected components in proximity graph
    constructed from Gamma probability integral transform of the data.
    """

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "{short_code}_GRAPH_GAMMA_{parent_code}".
        """
        parent_code = AbstractGraphGammaGofStatistic.code()
        short_code = GraphConnectedComponentsGammaGofStatistic.short_code()
        return f"{short_code}_{parent_code}"


class GraphCliqueNumberGammaGofStatistic(
    AbstractGraphGammaGofStatistic, GraphCliqueNumberTestStatistic
):
    """
    Largest clique observed in the Gamma proximity graph.

    Graph-based test that computes maximum clique size in proximity graph
    constructed from Gamma probability integral transform of the data.
    """

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "{short_code}_GRAPH_GAMMA_{parent_code}".
        """
        parent_code = AbstractGraphGammaGofStatistic.code()
        short_code = GraphCliqueNumberGammaGofStatistic.short_code()
        return f"{short_code}_{parent_code}"


class GraphIndependenceNumberGammaGofStatistic(
    AbstractGraphGammaGofStatistic, GraphIndependenceNumberTestStatistic
):
    """
    Independence number of the Gamma proximity graph.

    Graph-based test that computes maximum independent set size in proximity graph
    constructed from Gamma probability integral transform of the data.
    """

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for this test.

        :return: string code in format "{short_code}_GRAPH_GAMMA_{parent_code}".
        """
        parent_code = AbstractGraphGammaGofStatistic.code()
        short_code = GraphIndependenceNumberGammaGofStatistic.short_code()
        return f"{short_code}_{parent_code}"
