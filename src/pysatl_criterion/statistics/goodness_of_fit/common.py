from abc import ABC

import numpy as np
from scipy import special
from typing_extensions import override

from pysatl_criterion.statistics.alternative import (
    Alternative,
    AlternativeType,
    RightAlternative,
)
from pysatl_criterion.statistics.statistic import AbstractStatistic


class KSStatistic(AbstractStatistic, ABC):
    def __init__(self, alternative_type: AlternativeType = AlternativeType.TWO_TAILED, mode="auto"):
        self.alternative_type = alternative_type
        if mode == "auto":  # Always select exact
            mode = "exact"
        self.mode = mode

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    def do_execute_statistic(self, rvs, cdf_vals=None):
        """Compute D, D+ (RIGHT), or D- (LEFT) from aligned, sorted data and CDF values.

        ``alternative_type`` selects the direction of the empirical CDF deviation;
        all three statistics reject for large values, so their test tail is RIGHT.
        ``mode`` is retained for compatibility and does not affect the statistic
        or calculate a p-value here.
        """

        d_minus, _ = KSStatistic.__compute_dminus(cdf_vals, rvs)

        if self.alternative_type == AlternativeType.RIGHT:
            d_plus, _d_location = KSStatistic.__compute_dplus(cdf_vals, rvs)
            return d_plus
        if self.alternative_type == AlternativeType.LEFT:
            d_minus, _d_location = KSStatistic.__compute_dminus(cdf_vals, rvs)
            return d_minus

        # alternative == 'two-sided':
        d_plus, _d_plus_location = KSStatistic.__compute_dplus(cdf_vals, rvs)
        d_minus, _d_minus_location = KSStatistic.__compute_dminus(cdf_vals, rvs)
        if d_plus > d_minus:
            D = d_plus
            # d_location = d_plus_location
            # d_sign = 1
        else:
            D = d_minus
            # d_location = d_minus_location
            # d_sign = -1
        return D

    @staticmethod
    def __compute_dplus(cdf_vals, rvs):
        n = len(cdf_vals)
        d_plus = np.arange(1.0, n + 1) / n - cdf_vals
        a_max = d_plus.argmax()
        loc_max = rvs[a_max]
        return d_plus[a_max], loc_max

    @staticmethod
    def __compute_dminus(cdf_vals, rvs):
        n = len(cdf_vals)
        d_minus = cdf_vals - np.arange(0.0, n) / n
        a_max = d_minus.argmax()
        loc_max = rvs[a_max]
        return d_minus[a_max], loc_max


class ADStatistic(AbstractStatistic, ABC):
    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def code():
        raise NotImplementedError("Method is not implemented")

    def do_execute_statistic(self, rvs, log_cdf=None, log_sf=None):
        """
        Title: The Anderson-Darling test Ref. (book or article): See package nortest and also
        Table 4.9 p. 127 in M.
        A. Stephens, “Tests Based on EDF Statistics,” In: R. B. D’Agostino and M. A. Stephens, Eds.,
        Goodness-of-Fit Techniques, Marcel Dekker, New York, 1986, pp. 97-193.

        :param rvs:
        :return:
        """
        n = len(rvs)

        i = np.arange(1, n + 1)
        A2 = -n - np.sum((2 * i - 1.0) / n * (log_cdf + log_sf[::-1]), axis=0)
        return A2


class CrammerVonMisesStatistic(AbstractStatistic, ABC):
    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def code():
        return "CVM"

    def do_execute_statistic(self, rvs, cdf_vals):
        n = len(rvs)

        u = (2 * np.arange(1, n + 1) - 1) / (2 * n)
        w = 1 / (12 * n) + np.sum((u - cdf_vals) ** 2)

        return w


class Chi2Statistic(AbstractStatistic, ABC):
    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    def _m_sum(a, *, axis, preserve_mask, xp):
        if np.ma.isMaskedArray(a):
            s = a.sum(axis)
            return s if preserve_mask else np.asarray(s)
        return xp.sum(a, axis=axis)

    @staticmethod
    def _validate_frequencies(f_obs, f_exp):
        f_obs = np.asarray(f_obs, dtype=float)
        f_exp = np.asarray(f_exp, dtype=float)
        if f_obs.ndim != 1 or f_exp.ndim != 1 or f_obs.shape != f_exp.shape or not f_obs.size:
            raise ValueError("Frequencies must be nonempty 1D arrays of the same shape")
        if not np.all(np.isfinite(f_obs)) or not np.all(np.isfinite(f_exp)):
            raise ValueError("Frequencies must be finite")
        if np.any(f_obs < 0) or np.any(f_exp <= 0):
            raise ValueError("Observed frequencies must be nonnegative and expected positive")
        observed_total, expected_total = f_obs.sum(), f_exp.sum()
        if not np.isfinite(observed_total) or not np.isfinite(expected_total):
            raise ValueError("Frequency totals must be finite")
        if not np.isclose(
            observed_total, expected_total, rtol=np.sqrt(np.finfo(float).eps), atol=0
        ):
            raise ValueError("Observed and expected frequencies must have equal sums")
        return f_obs, f_exp

    def do_execute_statistic(self, f_obs, f_exp, lambda_):
        """Compute power divergence for matching observed and expected frequencies.

        Zero observations use their mathematical limits: a finite contribution
        for lambda_ > -1, and positive infinity for lambda_ <= -1.
        Expected frequencies must be strictly positive; totals must agree within
        a relative tolerance of sqrt(float64 epsilon).
        """
        f_obs, f_exp = Chi2Statistic._validate_frequencies(f_obs, f_exp)
        if np.ndim(lambda_) != 0 or not np.isfinite(lambda_):
            raise ValueError("lambda_ must be a finite scalar")
        if lambda_ <= -1 and np.any(f_obs == 0):
            return np.inf

        if lambda_ == 1:
            # Pearson's chi-squared statistic
            terms = (f_obs - f_exp) ** 2 / f_exp
        elif lambda_ == 0:
            # Log-likelihood ratio (i.e. G-test)
            terms = 2.0 * special.xlogy(f_obs, f_obs / f_exp)
        elif lambda_ == -1:
            # Modified log-likelihood ratio
            terms = 2.0 * special.xlogy(f_exp, f_exp / f_obs)
        else:
            # For lambda_ > -1, zero observations contribute zero to this form.
            positive = f_obs > 0
            observed, expected = f_obs[positive], f_exp[positive]
            terms = observed * ((observed / expected) ** lambda_ - 1)
            terms /= 0.5 * lambda_ * (lambda_ + 1)

        return terms.sum()


class MinToshiyukiStatistic(AbstractStatistic, ABC):
    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    def do_execute_statistic(self, cdf_vals):
        n = len(cdf_vals)
        d_plus = np.arange(1.0, n + 1) / n - cdf_vals
        d_minus = cdf_vals - np.arange(0.0, n) / n
        d = np.maximum.reduce([d_plus, d_minus])

        fi = 1 / (cdf_vals * (1 - cdf_vals))

        s = np.sum(d * np.sqrt(fi))
        return s / np.sqrt(n)
