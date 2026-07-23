from abc import ABC

import numpy as np
import scipy.stats as scipy_stats
from scipy.optimize import minimize
from typing_extensions import override

from pysatl_criterion import DistributionType
from pysatl_criterion.statistics import AbstractGoodnessOfFitStatistic
from pysatl_criterion.statistics.alternative import AlternativeType
from pysatl_criterion.statistics.goodness_of_fit.common import (
    ADStatistic,
    Chi2Statistic,
    CrammerVonMisesStatistic,
    KSStatistic,
)
from pysatl_criterion.statistics.hypothesis import GoodnessOfFitHypothesis


class AbstractLogLogisticGofStatistic(AbstractGoodnessOfFitStatistic, ABC):
    def __init__(self, alpha=1, beta=1):
        if alpha <= 0:
            raise ValueError("Alpha parameter must be strictly greater than zero.")
        if beta <= 0:
            raise ValueError("Beta parameter must be strictly greater than zero.")

        self.alpha = alpha
        self.beta = beta

    @override
    def hypothesis(self) -> GoodnessOfFitHypothesis:
        return GoodnessOfFitHypothesis({"alpha": self.alpha, "beta": self.beta})

    @staticmethod
    @override
    def distribution() -> DistributionType:
        return DistributionType.LOG_LOGISTIC

    @staticmethod
    @override
    def code():
        return f"LOG_LOGISTIC_{AbstractGoodnessOfFitStatistic.code()}"


class KolmogorovSmirnovLogLogisticGofStatistic(
    AbstractLogLogisticGofStatistic, KSStatistic
):
    def __init__(
        self,
        alternative_type: AlternativeType = AlternativeType.TWO_TAILED,
        alpha=1.0,
        beta=1.0,
    ):
        AbstractLogLogisticGofStatistic.__init__(self, alpha=alpha, beta=beta)
        KSStatistic.__init__(self, alternative_type)

    @staticmethod
    @override
    def short_code() -> str:
        return "KS"

    @staticmethod
    @override
    def code() -> str:
        short_code = KolmogorovSmirnovLogLogisticGofStatistic.short_code()
        return f"{short_code}_{AbstractLogLogisticGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        rvs = np.sort(rvs)

        cdf_vals = scipy_stats.fisk.cdf(rvs, c=self.beta, scale=self.alpha)
        return KSStatistic.do_execute_statistic(self, rvs, cdf_vals)


class AndersonDarlingLogLogisticGofStatistic(
    AbstractLogLogisticGofStatistic, ADStatistic
):
    @staticmethod
    @override
    def short_code() -> str:
        return "AD"

    @staticmethod
    @override
    def code() -> str:
        short_code = AndersonDarlingLogLogisticGofStatistic.short_code()
        return f"{short_code}_{AbstractLogLogisticGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        sorted_rvs = np.sort(np.asarray(rvs))

        log_cdf = scipy_stats.fisk.logcdf(sorted_rvs, c=self.beta, scale=self.alpha)
        log_sf = scipy_stats.fisk.logsf(sorted_rvs, c=self.beta, scale=self.alpha)
        return super().do_execute_statistic(
            sorted_rvs, log_cdf=log_cdf, log_sf=log_sf
        )


class CramerVonMisesLogLogisticGofStatistic(
    AbstractLogLogisticGofStatistic, CrammerVonMisesStatistic
):
    @staticmethod
    @override
    def short_code() -> str:
        return "CVM"

    @staticmethod
    @override
    def code() -> str:
        short_code = CramerVonMisesLogLogisticGofStatistic.short_code()
        return f"{short_code}_{AbstractLogLogisticGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        sorted_rvs = np.sort(np.asarray(rvs))

        cdf_vals = scipy_stats.fisk.cdf(sorted_rvs, c=self.beta, scale=self.alpha)
        return CrammerVonMisesStatistic.do_execute_statistic(
            self, sorted_rvs, cdf_vals
        )


class AbstractBinnedLogLogisticGofStatistic(
    AbstractLogLogisticGofStatistic, Chi2Statistic, ABC
):
    lambda_value: float = 1.0

    def __init__(self, bins: int = 8, alpha: float = 1.0, beta: float = 2.0):
        if bins < 2:
            raise ValueError(
                "At least two bins are required for binned Log-Logistic statistics."
            )
        self.bins = bins
        AbstractLogLogisticGofStatistic.__init__(self, alpha=alpha, beta=beta)
        self.lambda_value = getattr(self, "lambda_value", 1.0)

    def _counts_and_expected(self, rvs):
        sample = np.asarray(rvs)
        n = sample.size
        if n == 0:
            raise ValueError(
                "At least one observation is required for binned "
                "Log-Logistic statistics."
            )

        quantiles = np.linspace(0.0, 1.0, self.bins + 1)

        edges = scipy_stats.fisk.ppf(quantiles, c=self.beta, scale=self.alpha)

        edges[0] = -np.inf
        edges[-1] = np.inf

        counts, _ = np.histogram(sample, bins=edges)
        expected = np.full(self.bins, n / self.bins)
        return counts, expected

    @override
    def execute_statistic(self, rvs):
        counts, expected = self._counts_and_expected(rvs)
        return float(
            Chi2Statistic.do_execute_statistic(
                self, counts, expected, lambda_=self.lambda_value
            )
        )


class Chi2PearsonLogLogisticGofStatistic(AbstractBinnedLogLogisticGofStatistic):
    lambda_value = 1.0

    @staticmethod
    @override
    def short_code() -> str:
        return "CHI2_PEARSON"

    @staticmethod
    @override
    def code() -> str:
        short_code = Chi2PearsonLogLogisticGofStatistic.short_code()
        return f"{short_code}_{AbstractLogLogisticGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs):
        return super().execute_statistic(rvs)


class NikulinLogLogisticGofStatistic(Chi2PearsonLogLogisticGofStatistic):
    """
    Bogdanavicius-Nikulin goodness-of-fit test (BN-GOF)
    for the log-logistic distribution with censored data.
    Computes the statistic Y² = X² + QF, where:

    X² is the classical Pearson statistic (from the parent class)

    QF is a correction term accounting for parameter estimation
    """

    def __init__(self, n_intervals: int = 5):
        super().__init__()
        self.n_intervals = n_intervals

    @staticmethod
    @override
    def short_code() -> str:
        return "BN_GOF"

    @staticmethod
    @override
    def code() -> str:
        short_code = NikulinLogLogisticGofStatistic.short_code()
        return f"{short_code}_{AbstractLogLogisticGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs) -> float:
        chi2_result = super().execute_statistic(rvs)
        X2 = (
            chi2_result.get("statistic", 0.0)
            if isinstance(chi2_result, dict)
            else float(chi2_result)
        )

        times, censored = self._extract_data(rvs)
        alpha_hat, beta_hat = self._fit_mle(times, censored)
        bounds = self._build_intervals(times, alpha_hat, beta_hat)
        U_j, e_j = self._compute_frequencies(
            times, censored, bounds, alpha_hat, beta_hat
        )
        QF = self._compute_qf(
            times, censored, U_j, e_j, bounds, alpha_hat, beta_hat
        )
        return X2 + QF

    def _extract_data(self, rvs) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(rvs, tuple) and len(rvs) == 2:
            times, censored = rvs
            times = np.asarray(times)
            censored = np.asarray(censored)
        else:
            times = np.asarray(rvs)
            censored = np.ones_like(times)

        if np.any(times <= 0):
            raise ValueError("All times must be positive.")

        return times, censored

    def _fit_mle(self, times: np.ndarray, censored: np.ndarray) -> tuple[float, float]:
        def neg_log_likelihood(params):
            alpha, beta = params
            if alpha <= 0 or beta <= 0:
                return 1e10

            log_lik = 0.0
            for t, delta in zip(times, censored, strict=False):
                z = (t / alpha) ** beta
                log_s = -np.log(1 + z)

                if delta == 1:
                    log_f = (
                        np.log(beta)
                        - np.log(alpha)
                        + (beta - 1) * np.log(t / alpha)
                        - 2 * np.log(1 + z)
                    )
                    log_lik += log_f
                else:
                    log_lik += log_s

            return -log_lik

        event_times = times[censored == 1]
        if len(event_times) > 0:
            log_times = np.log(event_times)
            alpha_init = np.exp(np.mean(log_times))
            beta_init = np.sqrt(3) / (np.std(log_times) + 1e-10)
        else:
            alpha_init = np.median(times)
            beta_init = 1.0

        result = minimize(
            neg_log_likelihood,
            [alpha_init, beta_init],
            bounds=[(1e-6, None), (1e-6, None)],
            method="L-BFGS-B",
        )

        return result.x[0], result.x[1]

    def _build_intervals(
        self, times: np.ndarray, alpha: float, beta: float
    ) -> np.ndarray:
        tau = np.max(times)

        quantiles = np.linspace(0.001, 0.95, self.n_intervals + 1)
        bounds = alpha * (quantiles / (1 - quantiles)) ** (1 / beta)

        bounds[0] = 0.0
        bounds[-1] = tau

        return bounds

    def _compute_frequencies(
        self,
        times: np.ndarray,
        censored: np.ndarray,
        bounds: np.ndarray,
        alpha: float,
        beta: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        r = self.n_intervals
        U_j = np.zeros(r)
        e_j = np.zeros(r)

        def hazard(t):
            z = (t / alpha) ** beta
            return (beta / alpha) * (z / (1 + z))

        for j in range(r):
            a_l, a_u = bounds[j], bounds[j + 1]

            mask = (times > a_l) & (times <= a_u) & (censored == 1)
            U_j[j] = np.sum(mask)

            points = np.linspace(a_l, a_u, 100)
            integral = 0.0
            for u in points:
                Y_u = np.sum(times >= u)
                integral += hazard(u) * Y_u
            e_j[j] = integral * (a_u - a_l) / len(points)

        return U_j, e_j

    def _compute_qf(
        self,
        times: np.ndarray,
        censored: np.ndarray,
        U_j: np.ndarray,
        e_j: np.ndarray,
        bounds: np.ndarray,
        alpha: float,
        beta: float,
    ) -> float:
        n = len(times)
        r = self.n_intervals
        m = 2

        Z = (U_j - e_j) / np.sqrt(n)

        A = U_j / n
        A[A < 1e-10] = 1e-10

        def log_hazard_derivatives(t):
            z = (t / alpha) ** beta

            d_alpha = -beta / alpha + (beta / alpha) * (z / (1 + z))
            d_beta = 1.0 / beta + np.log(t / alpha) * (1 - z / (1 + z))

            return np.array([d_alpha, d_beta])

        C = np.zeros((m, r))
        for j in range(r):
            a_l, a_u = bounds[j], bounds[j + 1]
            mask = (times > a_l) & (times <= a_u) & (censored == 1)

            if np.any(mask):
                for i in np.where(mask)[0]:
                    derivs = log_hazard_derivatives(times[i])
                    C[:, j] += derivs / n

        information_matrix = np.zeros((m, m))
        for i in range(n):
            if censored[i] == 1:
                derivs = log_hazard_derivatives(times[i])
                information_matrix += np.outer(derivs, derivs) / n

        A_inv = np.diag(1.0 / A)
        G = information_matrix - C @ A_inv @ C.T
        G += np.eye(m) * 1e-8
        W = C @ A_inv @ Z
        try:
            G_inv = np.linalg.inv(G)
            QF = W @ G_inv @ W
        except np.linalg.LinAlgError:
            G_pinv = np.linalg.pinv(G)
            QF = W @ G_pinv @ W

        return QF


class MirvalievLogLogisticGofStatistic(NikulinLogLogisticGofStatistic):
    """
    Mirvaliev's test for the log-logistic distribution.
    Statistic: S_n = U_n² + W_n².
    """

    def __init__(self, n_intervals: int = 8):
        super().__init__(n_intervals=n_intervals)

    @staticmethod
    @override
    def short_code() -> str:
        return "MIRVALIEV"

    @staticmethod
    @override
    def code() -> str:
        return (
            f"{MirvalievLogLogisticGofStatistic.short_code()}"
            f"_{AbstractLogLogisticGofStatistic.code()}"
        )

    @override
    def execute_statistic(self, rvs) -> float:
        times, censored = self._extract_data(rvs)
        alpha_hat, beta_hat = self._fit_mle(times, censored)
        bounds = self._build_intervals(times, alpha_hat, beta_hat)
        U_j, e_j = self._compute_frequencies(
            times, censored, bounds, alpha_hat, beta_hat
        )
        Z, A_inv, C, G, G_inv = self._compute_mirvaliev_matrices(
            times, censored, U_j, e_j, bounds, alpha_hat, beta_hat
        )
        W_n2 = Z @ A_inv @ C.T @ G_inv @ C @ A_inv @ Z
        U_n2 = Z @ A_inv @ Z - W_n2

        if U_n2 < 0 and U_n2 > -1e-8:
            U_n2 = 0.0

        return U_n2 + W_n2

    def _compute_mirvaliev_matrices(
        self,
        times: np.ndarray,
        censored: np.ndarray,
        U_j: np.ndarray,
        e_j: np.ndarray,
        bounds: np.ndarray,
        alpha: float,
        beta: float,
    ):
        n = len(times)
        r = self.n_intervals
        m = 2
        Z = (U_j - e_j) / np.sqrt(n)
        A = U_j / n
        A[A < 1e-10] = 1e-10
        A_inv = np.diag(1.0 / A)

        def log_hazard_derivatives(t):
            z = (t / alpha) ** beta
            d_alpha = -beta / alpha + (beta / alpha) * (z / (1 + z))
            d_beta = 1.0 / beta + np.log(t / alpha) * (1 - z / (1 + z))
            return np.array([d_alpha, d_beta])

        C = np.zeros((m, r))
        for j in range(r):
            a_l, a_u = bounds[j], bounds[j + 1]
            mask = (times > a_l) & (times <= a_u) & (censored == 1)
            if np.any(mask):
                for i in np.where(mask)[0]:
                    C[:, j] += log_hazard_derivatives(times[i]) / n

        information_matrix = np.zeros((m, m))
        for i in range(n):
            if censored[i] == 1:
                d = log_hazard_derivatives(times[i])
                information_matrix += np.outer(d, d) / n

        G = information_matrix - C @ A_inv @ C.T
        G += np.eye(m) * 1e-8

        try:
            G_inv = np.linalg.inv(G)
        except np.linalg.LinAlgError:
            G_inv = np.linalg.pinv(G)

        return Z, A_inv, C, G, G_inv