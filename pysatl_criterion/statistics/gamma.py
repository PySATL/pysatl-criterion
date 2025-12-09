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
    """Base class for Gamma goodness-of-fit statistics."""

    def __init__(self, shape: float = 1.0, scale: float = 1.0):
        if shape <= 0:
            raise ValueError("Shape must be positive.")
        if scale <= 0:
            raise ValueError("Scale must be positive.")
        self.shape = shape
        self.scale = scale

    @staticmethod
    @override
    def code():
        return f"GAMMA_{AbstractGoodnessOfFitStatistic.code()}"


class KolmogorovSmirnovGammaGofStatistic(AbstractGammaGofStatistic, KSStatistic):
    """Kolmogorov–Smirnov EDF test computed with the Gamma reference CDF.

    References
    ----------
    .. [1] Kolmogorov, A. N. (1933). "On the empirical determination of a
           distribution law". *Giornale dell'Istituto Italiano degli Attuari*,
           4, 83–91.
    .. [2] Smirnov, N. V. (1948). "Table for estimating the goodness of fit of
        empirical distributions". *Annals of Mathematical Statistics*, 19(2),
        279–281.
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
    def code():
        return f"KS_{AbstractGammaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Parameters
        ----------
        rvs : array_like
            Observations assumed to follow Gamma(shape, scale).

        Returns
        -------
        float
            Kolmogorov–Smirnov $D$ statistic computed with the Gamma CDF.
        """

        sorted_rvs = np.sort(np.asarray(rvs))
        cdf_vals = scipy_stats.gamma.cdf(sorted_rvs, a=self.shape, scale=self.scale)
        return KSStatistic.execute_statistic(self, sorted_rvs, cdf_vals)


class LillieforsGammaGofStatistic(AbstractGammaGofStatistic, LillieforsTest):
    """Lilliefors correction that re-estimates Gamma parameters before KS.

    References
    ----------
    .. [1] Lilliefors, H. W. (1967). "On the Kolmogorov–Smirnov test for
           normality with mean and variance unknown". *Journal of the American
           Statistical Association*, 62(318), 399–402.
    """

    @staticmethod
    @override
    def code():
        return f"LILLIE_{AbstractGammaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Parameters
        ----------
        rvs : array_like
            Observations assumed to follow Gamma(shape, scale).

        Returns
        -------
        float
            Lilliefors-adjusted Kolmogorov–Smirnov statistic with estimated
            Gamma parameters.
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
    """Anderson–Darling EDF statistic fitted to the Gamma distribution.

    References
    ----------
    .. [1] Anderson, T. W., & Darling, D. A. (1952). "Asymptotic theory of
           certain goodness-of-fit criteria based on stochastic processes".
           *Annals of Mathematical Statistics*, 23(2), 193–212.
    """

    @staticmethod
    @override
    def code():
        return f"AD_{AbstractGammaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Parameters
        ----------
        rvs : array_like
            Observations assumed to follow Gamma(shape, scale).

        Returns
        -------
        float
            Anderson–Darling $A^{2}$ statistic tailored to the Gamma model.
        """

        sorted_rvs = np.sort(np.asarray(rvs))
        log_cdf = scipy_stats.gamma.logcdf(sorted_rvs, a=self.shape, scale=self.scale)
        log_sf = scipy_stats.gamma.logsf(sorted_rvs, a=self.shape, scale=self.scale)
        return super().execute_statistic(sorted_rvs, log_cdf=log_cdf, log_sf=log_sf)


class CramerVonMisesGammaGofStatistic(AbstractGammaGofStatistic, CrammerVonMisesStatistic):
    """Cramér–von Mises quadratic EDF test specialized for Gamma samples.

    References
    ----------
    .. [1] Cramér, H. (1928). "On the composition of elementary errors."
        *Scandinavian Actuarial Journal*, 11(1), 13–74.
    .. [2] von Mises, R. (1931). "Probability calculus and its application in
           statistics and theoretical physics". Leipzig: F. Deuticke.
    """

    @staticmethod
    @override
    def code():
        return f"CVM_{AbstractGammaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Parameters
        ----------
        rvs : array_like
            Observations assumed to follow Gamma(shape, scale).

        Returns
        -------
        float
            Cramér–von Mises $W^{2}$ statistic using the Gamma CDF.
        """

        sorted_rvs = np.sort(np.asarray(rvs))
        cdf_vals = scipy_stats.gamma.cdf(sorted_rvs, a=self.shape, scale=self.scale)
        return CrammerVonMisesStatistic.execute_statistic(self, sorted_rvs, cdf_vals)


class WatsonGammaGofStatistic(AbstractGammaGofStatistic):
    """Watson's rotation-invariant EDF statistic using Gamma CDF values.

    References
    ----------
    .. [1] Watson, G. S. (1961). "Goodness-of-fit tests on a circle".
           *Biometrika*, 48(1/2), 109–114.
    """

    @staticmethod
    @override
    def code():
        return f"WAT_{AbstractGammaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Parameters
        ----------
        rvs : array_like
            Observations assumed to follow Gamma(shape, scale).

        Returns
        -------
        float
            Watson $U^{2}$ statistic derived from the Gamma CDF values.
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
    """Kuiper's circular EDF statistic after Gamma probability transform.

    References
    ----------
    .. [1] Kuiper, N. H. (1960). "Tests concerning random points on a circle".
           *Ned. Akad. Wetensch. Proc. Ser. A*, 63, 38–47.
    """

    @staticmethod
    @override
    def code():
        return f"KUI_{AbstractGammaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Parameters
        ----------
        rvs : array_like
            Observations assumed to follow Gamma(shape, scale).

        Returns
        -------
        float
            Kuiper $V = D^{+} + D^{-}$ statistic after the Gamma probability
            integral transform.
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
    """Greenwood spacing statistic measuring uniformized Gamma gaps.

    References
    ----------
    .. [1] Greenwood, M. (1946). "The statistical study of infectious disease".
           *Journal of the Royal Statistical Society. Series A*, 109(1), 85–110.
    """

    @staticmethod
    @override
    def code():
        return f"GRW_{AbstractGammaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Parameters
        ----------
        rvs : array_like
            Observations assumed to follow Gamma(shape, scale).

        Returns
        -------
        float
            Greenwood spacing statistic $G = \\sum_{i=1}^{n+1} D_i^2$ where
            spacings $D_i$ are computed from Gamma CDF values.
        """

        sorted_rvs = np.sort(np.asarray(rvs))
        cdf_vals = scipy_stats.gamma.cdf(sorted_rvs, a=self.shape, scale=self.scale)
        spacings = np.diff(np.concatenate(([0.0], cdf_vals, [1.0])))
        if np.any(spacings < 0):
            raise ValueError("Spacings must be non-negative; check input data ordering.")
        return float(np.sum(spacings**2))


class MoranGammaGofStatistic(AbstractGammaGofStatistic):
    """Moran log-spacing statistic applied to Gamma-transformed uniforms.

    References
    ----------
    .. [1] Moran, P. A. P. (1950). "A test for serial independence of residuals".
           *Biometrika*, 37(1/2), 178–181.
    """

    @staticmethod
    @override
    def code():
        return f"MOR_{AbstractGammaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Parameters
        ----------
        rvs : array_like
            Observations assumed to follow Gamma(shape, scale).

        Returns
        -------
        float
            Moran spacing statistic $M = -\\sum \\log(n D_i)$ based on Gamma CDF spacings.
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
    """Min–Toshiyuki tail-sensitive EDF statistic under a Gamma model.

    References
    ----------
    .. [1] Min, C., & Toshiyuki, T. (2015). "An EDF statistic with adaptive
           tail sensitivity". *Communications in Statistics – Simulation and
           Computation*, 44(7), 1731–1749.
    """

    @staticmethod
    @override
    def code():
        return f"MT_{AbstractGammaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Parameters
        ----------
        rvs : array_like
            Observations assumed to follow Gamma(shape, scale).

        Returns
        -------
        float
            Min–Toshiyuki statistic that up-weights EDF deviations near the
            distribution tails using Gamma CDF values.
        """

        sorted_rvs = np.sort(np.asarray(rvs))
        cdf_vals = scipy_stats.gamma.cdf(sorted_rvs, a=self.shape, scale=self.scale)
        return MinToshiyukiStatistic.execute_statistic(self, cdf_vals)


class AbstractBinnedGammaGofStatistic(AbstractGammaGofStatistic, Chi2Statistic, ABC):
    """Base class for Gamma GOF tests built on equiprobable histogram bins."""

    lambda_value: float = 1.0

    def __init__(self, bins: int = 8, shape: float = 1.0, scale: float = 1.0):
        if bins < 2:
            raise ValueError("At least two bins are required for binned Gamma statistics.")
        self.bins = bins
        super().__init__(shape=shape, scale=scale)
        self.lambda_value = getattr(self, "lambda_value", 1.0)

    def _counts_and_expected(self, rvs):
        sample = np.asarray(rvs)
        n = sample.size
        if n == 0:
            raise ValueError("At least one observation is required for binned Gamma statistics.")

        quantiles = np.linspace(0.0, 1.0, self.bins + 1)
        edges = scipy_stats.gamma.ppf(quantiles, a=self.shape, scale=self.scale)
        edges[0] = -np.inf
        edges[-1] = np.inf
        counts, _ = np.histogram(sample, bins=edges)
        expected = np.full(self.bins, n / self.bins)
        return counts, expected

    @override
    def execute_statistic(self, rvs, **kwargs):
        counts, expected = self._counts_and_expected(rvs)
        return float(
            Chi2Statistic.execute_statistic(self, counts, expected, lambda_=self.lambda_value)
        )


class Chi2PearsonGammaGofStatistic(AbstractBinnedGammaGofStatistic):
    """Pearson chi-square frequency test based on Gamma equiprobable bins.

    References
    ----------
    .. [1] Pearson, K. (1900). "On the criterion that a given system of
           deviations from the probable in the case of a correlated system of
           variables is such that it can be reasonably supposed to have arisen
           from random sampling". *Philosophical Magazine*, 50(302), 157–175.
    """

    lambda_value = 1.0

    @staticmethod
    @override
    def code():
        return f"CHI2_PEARSON_{AbstractGammaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Pearson chi-square statistic computed on equiprobable Gamma bins.

        Implements Karl Pearson's (1900) frequency test by binning via the
        Gamma quantile function so each bin has equal theoretical probability.
        """

        return super().execute_statistic(rvs, **kwargs)


class LikelihoodRatioGammaGofStatistic(AbstractBinnedGammaGofStatistic):
    """Log-likelihood ratio ($G$-test) for Gamma reference distribution.

    References
    ----------
    .. [1] Wilks, S. S. (1935). "The likelihood test of independence in
           contingency tables". *Annals of Mathematical Statistics*, 6(4), 190–196.
    """

    lambda_value = 0.0

    @staticmethod
    @override
    def code():
        return f"G_TEST_{AbstractGammaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Likelihood-ratio statistic using equiprobable Gamma quantile bins.

        Follows the classical $G$-test described by S. S. Wilks (1935) and
        tests histogram counts against expected Gamma frequencies.
        """

        return super().execute_statistic(rvs, **kwargs)


class CressieReadGammaGofStatistic(AbstractBinnedGammaGofStatistic):
    """Cressie–Read power-divergence statistic for Gamma data.

    References
    ----------
    .. [1] Read, T. R. C., & Cressie, N. A. C. (1988). *Goodness-of-Fit
           Statistics for Discrete Multivariate Data*. Springer.
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
    def code():
        return f"CRESSIE_READ_{AbstractGammaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Power-divergence statistic bridging Pearson ($\\lambda=1$) and $G$-tests.

        Defaults to the recommended $\\lambda=2/3$ value from Read & Cressie
        (1988) but allows custom power parameters.
        """

        return super().execute_statistic(rvs, **kwargs)


class ProbabilityPlotCorrelationGammaGofStatistic(AbstractGammaGofStatistic):
    """Filliben-style PPCC statistic comparing Gamma quantiles to the sample.

    References
    ----------
    .. [1] Filliben, J. J. (1975). "The probability plot correlation
           coefficient test for normality". *Technometrics*, 17(1), 111–117.
    """

    @staticmethod
    @override
    def code():
        return f"PPCC_{AbstractGammaGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Parameters
        ----------
        rvs : array_like
            Observations assumed to follow Gamma(shape, scale).

        Returns
        -------
        float
            One minus the probability-plot correlation coefficient. Values
            near zero indicate a strong linear alignment with the theoretical
            Gamma quantiles (Filliben, 1975).
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
    """Base class for Gamma graph-based GOF statistics using EDF transforms."""

    @staticmethod
    @override
    def code():
        parent_code = AbstractGammaGofStatistic.code()
        return f"GRAPH_{parent_code}"

    def _transform_sample(self, rvs):
        sample = np.asarray(rvs, dtype=float)
        if sample.size == 0:
            raise ValueError(
                "At least one observation is required to compute Gamma graph statistics."
            )

        sorted_sample = np.sort(sample)
        uniformized = scipy_stats.gamma.cdf(sorted_sample, a=self.shape, scale=self.scale)
        return uniformized.tolist()

    def _evaluate_graph_statistic(self, transformed_sample, **kwargs):
        """Delegate graph statistic evaluation to the generic adjacency-based logic."""

        return AbstractGraphTestStatistic.execute_statistic(self, transformed_sample, **kwargs)

    @override
    def execute_statistic(self, rvs, **kwargs):
        transformed_sample = self._transform_sample(rvs)
        return self._evaluate_graph_statistic(transformed_sample, **kwargs)


class GraphEdgesNumberGammaGofStatistic(
    AbstractGraphGammaGofStatistic, GraphEdgesNumberTestStatistic
):
    """Counts edges in the proximity graph built on Gamma-CDF spacings."""

    @staticmethod
    @override
    def code():
        parent_code = AbstractGraphGammaGofStatistic.code()
        stat_name = GraphEdgesNumberGammaGofStatistic.get_stat_name()
        return f"{stat_name}_{parent_code}"


class GraphMaxDegreeGammaGofStatistic(AbstractGraphGammaGofStatistic, GraphMaxDegreeTestStatistic):
    """Maximum degree in the Gamma-induced proximity graph."""

    @staticmethod
    @override
    def code():
        parent_code = AbstractGraphGammaGofStatistic.code()
        stat_name = GraphMaxDegreeGammaGofStatistic.get_stat_name()
        return f"{stat_name}_{parent_code}"


class GraphAverageDegreeGammaGofStatistic(
    AbstractGraphGammaGofStatistic, GraphAverageDegreeTestStatistic
):
    """Average vertex degree of the Gamma proximity graph."""

    @staticmethod
    @override
    def code():
        parent_code = AbstractGraphGammaGofStatistic.code()
        stat_name = GraphAverageDegreeGammaGofStatistic.get_stat_name()
        return f"{stat_name}_{parent_code}"


class GraphConnectedComponentsGammaGofStatistic(
    AbstractGraphGammaGofStatistic, GraphConnectedComponentsTestStatistic
):
    """Number of connected components in the Gamma proximity graph."""

    @staticmethod
    @override
    def code():
        parent_code = AbstractGraphGammaGofStatistic.code()
        stat_name = GraphConnectedComponentsGammaGofStatistic.get_stat_name()
        return f"{stat_name}_{parent_code}"


class GraphCliqueNumberGammaGofStatistic(
    AbstractGraphGammaGofStatistic, GraphCliqueNumberTestStatistic
):
    """Largest clique observed in the Gamma proximity graph."""

    @staticmethod
    @override
    def code():
        parent_code = AbstractGraphGammaGofStatistic.code()
        stat_name = GraphCliqueNumberGammaGofStatistic.get_stat_name()
        return f"{stat_name}_{parent_code}"

    def _evaluate_graph_statistic(self, transformed_sample, **kwargs):
        return GraphCliqueNumberTestStatistic.execute_statistic(self, transformed_sample, **kwargs)


class GraphIndependenceNumberGammaGofStatistic(
    AbstractGraphGammaGofStatistic, GraphIndependenceNumberTestStatistic
):
    """Independence number of the Gamma proximity graph."""

    @staticmethod
    @override
    def code():
        parent_code = AbstractGraphGammaGofStatistic.code()
        stat_name = GraphIndependenceNumberGammaGofStatistic.get_stat_name()
        return f"{stat_name}_{parent_code}"

    def _evaluate_graph_statistic(self, transformed_sample, **kwargs):
        return GraphIndependenceNumberTestStatistic.execute_statistic(
            self, transformed_sample, **kwargs
        )
