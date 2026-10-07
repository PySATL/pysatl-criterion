from abc import ABC
from numbers import Integral

import numpy as np
import scipy.stats as scipy_stats
from typing_extensions import override

from pysatl_criterion.distribution.distributions import (
    StudentDistributionDescriptor as Distribution,
)
from pysatl_criterion.distribution.distributions import StudentDistributionDescriptor as Student
from pysatl_criterion.distribution.parameters import HypothesisSupport, ParameterValues
from pysatl_criterion.statistics import AbstractGoodnessOfFitStatistic
from pysatl_criterion.statistics.alternative import Alternative, AlternativeType, RightAlternative
from pysatl_criterion.statistics.goodness_of_fit.common import (
    ADStatistic,
    CrammerVonMisesStatistic,
    KSStatistic,
    LillieforsTest,
)


class AbstractStudentGofStatistic(AbstractGoodnessOfFitStatistic, ABC):
    """
    Abstract base class for Student's t-distribution goodness-of-fit statistics.
    """

    @property
    def df(self) -> float:
        """Read df by its stable parameter identity."""
        return self._parameters[Distribution.DF]

    @property
    def loc(self) -> float:
        """Read loc by its stable parameter identity."""
        return self._parameters[Distribution.LOCATION]

    @property
    def scale(self) -> float:
        """Read scale by its stable parameter identity."""
        return self._parameters[Distribution.SCALE]

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Student.DEFAULT, frozenset(Student.DEFAULT.parameters)),)

    @staticmethod
    @override
    def distribution() -> type[Student]:
        """Return the distribution descriptor class."""
        return Student

    @classmethod
    @override
    def code(cls) -> str:
        """Return the family identifier or the concrete statistic's full identifier."""
        family_code = f"STUDENT_{AbstractGoodnessOfFitStatistic.code()}"
        if "short_code" in cls.__abstractmethods__:
            return family_code
        return f"{cls.short_code()}_{family_code}"

    @staticmethod
    def _validate_sample(rvs):
        sample = np.asarray(rvs)
        if sample.dtype.kind not in "iuf":
            raise ValueError("Sample must contain finite real numbers")
        sample = np.asarray(sample, dtype=float)
        if sample.ndim != 1 or sample.size == 0 or not np.all(np.isfinite(sample)):
            raise ValueError("Sample must be nonempty, one-dimensional and finite")
        return np.sort(sample)

    def _standardize(self, sample):
        # Recover representable differences when subtraction alone overflows.
        with np.errstate(over="ignore", invalid="ignore"):
            difference = sample - self.loc
            z = difference / self.scale
            overflow = np.isinf(difference)
            z[overflow] = sample[overflow] / self.scale - self.loc / self.scale
        if not np.all(np.isfinite(z)):
            raise ValueError("Standardized observations exceed floating-point range")
        return z

    def _log_probabilities(self, standardized):
        logcdf = scipy_stats.t.logcdf(standardized, self.df)
        logsf = scipy_stats.t.logsf(standardized, self.df)
        if not np.all(np.isfinite(logcdf)) or not np.all(np.isfinite(logsf)):
            raise ValueError("Student tail probabilities exceed floating-point range")
        return logcdf, logsf


class KolmogorovSmirnovStudentGofStatistic(AbstractStudentGofStatistic, KSStatistic):
    """Kolmogorov-Smirnov distance to a specified Student t CDF.

    Parameters
    ----------
    parameters : ParameterValues
        Values with df, loc, scale fixed; omitted parameters are unknown.
    alternative_type : AlternativeType or str, optional
        CDF deviation direction; default TWO_TAILED. Also accepts enum values
        'two_tailed', 'right', 'left' and SciPy names 'two-sided', 'greater',
        'less'. All choices reject for large statistic values.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar discrepancy without changing the input or fitting state.
    hypothesis()
        Return the fixed parameters of the null hypothesis.
    alternative()
        Return the right tail of the statistic's distribution.

    Notes
    -----
    The null fixes df, loc and scale on the whole real line. Write
    u_i = t_df.cdf((x_(i)-loc)/scale) for sorted observations, i=1,...,n.
    D+ = max(i/n-u_i), D- = max(u_i-(i-1)/n). Return max(D+,D-)
    for two-sided deviations, D+ for RIGHT/greater and D- for LEFT/less.
    No sqrt(n) scaling is applied.
    Large values reject. The probability integral transform removes all
    distribution parameters from the null law, not from the hypothesis.
    The reference describes a general statistic, applied here through the
    specified Student CDF, not a separately derived Student-specific test.

    References
    ----------
    .. [1] M. A. Stephens (1974), EDF Statistics for Goodness of Fit and Some
    Comparisons, JASA 69, 730-737, https://doi.org/10.1080/01621459.1974.10480196.

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'df': 5, 'loc': 0, 'scale': 1})
    >>> statistic = KolmogorovSmirnovStudentGofStatistic(parameters)
    >>> value = statistic.execute_statistic([-2., -0.7, 0., 0.4, 1.8])
    >>> bool(np.isfinite(value))
    True
    """

    def __init__(
        self,
        parameters: ParameterValues,
        *,
        alternative_type: AlternativeType = AlternativeType.TWO_TAILED,
    ):
        AbstractStudentGofStatistic.__init__(self, parameters)
        aliases = {
            "two-sided": AlternativeType.TWO_TAILED,
            "greater": AlternativeType.RIGHT,
            "less": AlternativeType.LEFT,
        }
        if isinstance(alternative_type, str):
            alternative_type = aliases.get(alternative_type, alternative_type)
            try:
                alternative_type = AlternativeType(alternative_type)
            except ValueError as exc:
                raise ValueError("Invalid KS alternative_type") from exc
        if alternative_type not in tuple(AlternativeType):
            raise ValueError("Invalid KS alternative_type")
        KSStatistic.__init__(self, alternative_type)

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "KS".
        """
        return "KS"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute kolmogorov-Smirnov distance to a specified Student t CDF.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real sample on the whole real line, with
            at least one observation; ties and constants are allowed.
            The input is not modified.
        **kwargs : dict
            Accepted for the common interface; no call-time options are used.

        Returns
        -------
        float or numpy.float64
            Unscaled statistic; large values reject the null.

        Raises
        ------
        ValueError
            If the sample is invalid or required numerical transformations
            cannot be represented in floating-point arithmetic.
        """
        rvs = self._validate_sample(rvs)
        # Standardize the data
        standardized = self._standardize(rvs)
        cdf_vals = scipy_stats.t.cdf(standardized, self.df)
        return KSStatistic.do_execute_statistic(self, rvs, cdf_vals)


class AndersonDarlingStudentGofStatistic(AbstractStudentGofStatistic, ADStatistic):
    """Anderson-Darling A-squared for a specified Student t CDF.

    Parameters
    ----------
    parameters : ParameterValues
        Values with df, loc, scale fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar discrepancy without changing the input or fitting state.
    hypothesis()
        Return the fixed parameters of the null hypothesis.
    alternative()
        Return the right tail of the statistic's distribution.

    Notes
    -----
    The null fixes df, loc and scale on the whole real line. Write
    u_i = t_df.cdf((x_(i)-loc)/scale) for sorted observations, i=1,...,n.
    A2 = -n - sum((2*i-1)*(log(u_i)+log(1-u_(n+1-i))))/n.
    No fitted-parameter correction is applied.
    Large values reject. The probability integral transform removes all
    distribution parameters from the null law, not from the hypothesis.
    The reference describes a general statistic, applied here through the
    specified Student CDF, not a separately derived Student-specific test.
    Log-CDF and log-survival are evaluated separately without clipping.
    True probabilities are interior for finite data; numerical tail
    underflow raises ValueError instead of fabricating a finite statistic.

    References
    ----------
    .. [1] J. Zhang (2002), Powerful goodness-of-fit tests based on the likelihood
    ratio, JRSS B 64, 281-294, https://doi.org/10.1111/1467-9868.00337. Section 2.2.

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'df': 5, 'loc': 0, 'scale': 1})
    >>> statistic = AndersonDarlingStudentGofStatistic(parameters)
    >>> value = statistic.execute_statistic([-2., -0.7, 0., 0.4, 1.8])
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "AD".
        """
        return "AD"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute anderson-Darling A-squared for a specified Student t CDF.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real sample on the whole real line, with
            at least one observation; ties and constants are allowed.
            The input is not modified.
        **kwargs : dict
            Accepted for the common interface; no call-time options are used.

        Returns
        -------
        float or numpy.float64
            Unscaled statistic; large values reject the null.

        Raises
        ------
        ValueError
            If the sample is invalid or required numerical transformations
            cannot be represented in floating-point arithmetic.
        """
        y = self._validate_sample(rvs)
        # Standardize the data
        standardized = self._standardize(y)
        logcdf, logsf = self._log_probabilities(standardized)
        return ADStatistic.do_execute_statistic(self, y, log_cdf=logcdf, log_sf=logsf)


class CramerVonMisesStudentGofStatistic(AbstractStudentGofStatistic, CrammerVonMisesStatistic):
    """Cramer-von Mises W-squared for a specified Student t CDF.

    Parameters
    ----------
    parameters : ParameterValues
        Values with df, loc, scale fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar discrepancy without changing the input or fitting state.
    hypothesis()
        Return the fixed parameters of the null hypothesis.
    alternative()
        Return the right tail of the statistic's distribution.

    Notes
    -----
    The null fixes df, loc and scale on the whole real line. Write
    u_i = t_df.cdf((x_(i)-loc)/scale) for sorted observations, i=1,...,n.
    W2 = 1/(12*n) + sum((u_i-(i-0.5)/n)**2).
    No finite-sample correction is applied.
    Large values reject. The probability integral transform removes all
    distribution parameters from the null law, not from the hypothesis.
    The reference describes a general statistic, applied here through the
    specified Student CDF, not a separately derived Student-specific test.

    References
    ----------
    .. [1] J. Zhang (2002), Powerful goodness-of-fit tests based on the likelihood
    ratio, JRSS B 64, 281-294, https://doi.org/10.1111/1467-9868.00337. Section 2.3.

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'df': 5, 'loc': 0, 'scale': 1})
    >>> statistic = CramerVonMisesStudentGofStatistic(parameters)
    >>> value = statistic.execute_statistic([-2., -0.7, 0., 0.4, 1.8])
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "CVM".
        """
        return "CVM"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute cramer-von Mises W-squared for a specified Student t CDF.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real sample on the whole real line, with
            at least one observation; ties and constants are allowed.
            The input is not modified.
        **kwargs : dict
            Accepted for the common interface; no call-time options are used.

        Returns
        -------
        float or numpy.float64
            Unscaled statistic; large values reject the null.

        Raises
        ------
        ValueError
            If the sample is invalid or required numerical transformations
            cannot be represented in floating-point arithmetic.
        """
        rvs = self._validate_sample(rvs)
        # Standardize the data
        standardized = self._standardize(rvs)
        cdf_vals = scipy_stats.t.cdf(standardized, self.df)
        return CrammerVonMisesStatistic.do_execute_statistic(self, rvs, cdf_vals)


class KuiperStudentGofStatistic(AbstractStudentGofStatistic):
    """Kuiper V for a specified Student t CDF.

    Parameters
    ----------
    parameters : ParameterValues
        Values with df, loc, scale fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar discrepancy without changing the input or fitting state.
    hypothesis()
        Return the fixed parameters of the null hypothesis.
    alternative()
        Return the right tail of the statistic's distribution.

    Notes
    -----
    The null fixes df, loc and scale on the whole real line. Write
    u_i = t_df.cdf((x_(i)-loc)/scale) for sorted observations, i=1,...,n.
    V = max(i/n-u_i) + max(u_i-(i-1)/n).
    This is the unscaled sum of the two one-sided EDF distances.
    Large values reject. The probability integral transform removes all
    distribution parameters from the null law, not from the hypothesis.
    The reference describes a general statistic, applied here through the
    specified Student CDF, not a separately derived Student-specific test.

    References
    ----------
    .. [1] M. A. Stephens (1974), EDF Statistics for Goodness of Fit and Some
    Comparisons, JASA 69, 730-737, https://doi.org/10.1080/01621459.1974.10480196.

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'df': 5, 'loc': 0, 'scale': 1})
    >>> statistic = KuiperStudentGofStatistic(parameters)
    >>> value = statistic.execute_statistic([-2., -0.7, 0., 0.4, 1.8])
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "KUIPER".
        """
        return "KUIPER"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute kuiper V for a specified Student t CDF.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real sample on the whole real line, with
            at least one observation; ties and constants are allowed.
            The input is not modified.
        **kwargs : dict
            Accepted for the common interface; no call-time options are used.

        Returns
        -------
        float or numpy.float64
            Unscaled statistic; large values reject the null.

        Raises
        ------
        ValueError
            If the sample is invalid or required numerical transformations
            cannot be represented in floating-point arithmetic.
        """
        rvs = self._validate_sample(rvs)
        n = len(rvs)
        # Standardize the data
        standardized = self._standardize(rvs)
        cdf_vals = scipy_stats.t.cdf(standardized, self.df)

        # D+ and D-
        d_plus = np.max(np.arange(1.0, n + 1) / n - cdf_vals)
        d_minus = np.max(cdf_vals - np.arange(0.0, n) / n)

        return d_plus + d_minus


class WatsonStudentGofStatistic(AbstractStudentGofStatistic):
    """Watson U-squared for a specified Student t CDF.

    Parameters
    ----------
    parameters : ParameterValues
        Values with df, loc, scale fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar discrepancy without changing the input or fitting state.
    hypothesis()
        Return the fixed parameters of the null hypothesis.
    alternative()
        Return the right tail of the statistic's distribution.

    Notes
    -----
    The null fixes df, loc and scale on the whole real line. Write
    u_i = t_df.cdf((x_(i)-loc)/scale) for sorted observations, i=1,...,n.
    U2 = W2 - n*(mean(u)-0.5)**2, where
    W2 = 1/(12*n) + sum((u_i-(i-0.5)/n)**2). Centered residuals
    implement the same formula without cancellation. No correction is applied.
    Large values reject. The probability integral transform removes all
    distribution parameters from the null law, not from the hypothesis.
    The reference describes a general statistic, applied here through the
    specified Student CDF, not a separately derived Student-specific test.

    References
    ----------
    .. [1] G. S. Watson (1961), Goodness-of-fit tests on a circle, Biometrika
    48, 109-114, https://doi.org/10.1093/biomet/48.1-2.109.

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'df': 5, 'loc': 0, 'scale': 1})
    >>> statistic = WatsonStudentGofStatistic(parameters)
    >>> value = statistic.execute_statistic([-2., -0.7, 0., 0.4, 1.8])
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "WATSON".
        """
        return "WATSON"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute watson U-squared for a specified Student t CDF.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real sample on the whole real line, with
            at least one observation; ties and constants are allowed.
            The input is not modified.
        **kwargs : dict
            Accepted for the common interface; no call-time options are used.

        Returns
        -------
        float or numpy.float64
            Unscaled statistic; large values reject the null.

        Raises
        ------
        ValueError
            If the sample is invalid or required numerical transformations
            cannot be represented in floating-point arithmetic.
        """
        rvs = self._validate_sample(rvs)
        n = len(rvs)
        # Standardize the data
        standardized = self._standardize(rvs)
        cdf_vals = scipy_stats.t.cdf(standardized, self.df)

        # Center residuals before squaring to avoid subtracting large terms.
        residual = cdf_vals - (np.arange(1, n + 1) - 0.5) / n
        return float(1 / (12 * n) + np.sum((residual - np.mean(residual)) ** 2))


class ZhangZcStudentGofStatistic(AbstractStudentGofStatistic):
    """Zhang Z_C for a specified Student t CDF.

    Parameters
    ----------
    parameters : ParameterValues
        Values with df, loc, scale fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar discrepancy without changing the input or fitting state.
    hypothesis()
        Return the fixed parameters of the null hypothesis.
    alternative()
        Return the right tail of the statistic's distribution.

    Notes
    -----
    The null fixes df, loc and scale on the whole real line. Write
    u_i = t_df.cdf((x_(i)-loc)/scale) for sorted observations, i=1,...,n.
    Z_C = sum((log((1-u_i)/u_i) - log((n-i+0.25)/(i-0.75)))**2).
    This implements equation (3.3), not the exact integrated likelihood
    ratio from which Zhang derives that formula approximately.
    Large values reject. The probability integral transform removes all
    distribution parameters from the null law, not from the hypothesis.
    The reference describes a general statistic, applied here through the
    specified Student CDF, not a separately derived Student-specific test.
    Log-CDF and log-survival are evaluated separately without clipping.
    True probabilities are interior for finite data; numerical tail
    underflow raises ValueError instead of fabricating a finite statistic.
    Recompute stored null distributions made with the previous clipped code.

    References
    ----------
    .. [1] J. Zhang (2002), Powerful goodness-of-fit tests based on the likelihood
    ratio, JRSS B 64, 281-294, https://doi.org/10.1111/1467-9868.00337. Section 3.3.

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'df': 5, 'loc': 0, 'scale': 1})
    >>> statistic = ZhangZcStudentGofStatistic(parameters)
    >>> value = statistic.execute_statistic([-2., -0.7, 0., 0.4, 1.8])
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "ZHANG_ZC".
        """
        return "ZHANG_ZC"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute zhang Z_C for a specified Student t CDF.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real sample on the whole real line, with
            at least one observation; ties and constants are allowed.
            The input is not modified.
        **kwargs : dict
            Accepted for the common interface; no call-time options are used.

        Returns
        -------
        float or numpy.float64
            Unscaled statistic; large values reject the null.

        Raises
        ------
        ValueError
            If the sample is invalid or required numerical transformations
            cannot be represented in floating-point arithmetic.
        """
        rvs = self._validate_sample(rvs)
        n = len(rvs)
        # Standardize the data
        standardized = self._standardize(rvs)
        logcdf, logsf = self._log_probabilities(standardized)
        i = np.arange(1, n + 1)
        log_rank_odds = np.log(n - i + 0.25) - np.log(i - 0.75)
        return float(np.sum((logsf - logcdf - log_rank_odds) ** 2))


class ZhangZaStudentGofStatistic(AbstractStudentGofStatistic):
    """Zhang Z_A for a specified Student t CDF.

    Parameters
    ----------
    parameters : ParameterValues
        Values with df, loc, scale fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar discrepancy without changing the input or fitting state.
    hypothesis()
        Return the fixed parameters of the null hypothesis.
    alternative()
        Return the right tail of the statistic's distribution.

    Notes
    -----
    The null fixes df, loc and scale on the whole real line. Write
    u_i = t_df.cdf((x_(i)-loc)/scale) for sorted observations, i=1,...,n.
    Z_A = -sum(log(u_i)/(n-i+0.5) + log(1-u_i)/(i-0.5)).
    No additional normalization is applied.
    Large values reject. The probability integral transform removes all
    distribution parameters from the null law, not from the hypothesis.
    The reference describes a general statistic, applied here through the
    specified Student CDF, not a separately derived Student-specific test.
    Log-CDF and log-survival are evaluated separately without clipping.
    True probabilities are interior for finite data; numerical tail
    underflow raises ValueError instead of fabricating a finite statistic.
    Recompute stored null distributions made with the previous clipped code.

    References
    ----------
    .. [1] J. Zhang (2002), Powerful goodness-of-fit tests based on the likelihood
    ratio, JRSS B 64, 281-294, https://doi.org/10.1111/1467-9868.00337. Equation (3.2).

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'df': 5, 'loc': 0, 'scale': 1})
    >>> statistic = ZhangZaStudentGofStatistic(parameters)
    >>> value = statistic.execute_statistic([-2., -0.7, 0., 0.4, 1.8])
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "ZHANG_ZA".
        """
        return "ZHANG_ZA"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute zhang Z_A for a specified Student t CDF.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real sample on the whole real line, with
            at least one observation; ties and constants are allowed.
            The input is not modified.
        **kwargs : dict
            Accepted for the common interface; no call-time options are used.

        Returns
        -------
        float or numpy.float64
            Unscaled statistic; large values reject the null.

        Raises
        ------
        ValueError
            If the sample is invalid or required numerical transformations
            cannot be represented in floating-point arithmetic.
        """
        rvs = self._validate_sample(rvs)
        n = len(rvs)
        # Standardize the data
        standardized = self._standardize(rvs)
        logcdf, logsf = self._log_probabilities(standardized)
        i = np.arange(1, n + 1)
        return float(-np.sum(logcdf / (n - i + 0.5) + logsf / (i - 0.5)))


class LillieforsStudentGofStatistic(AbstractStudentGofStatistic, LillieforsTest):
    """Fitted KS distance for Student t with known degrees of freedom.

    Parameters
    ----------
    parameters : ParameterValues
        Values with df fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar discrepancy without changing the input or fitting state.
    hypothesis()
        Return the fixed parameters of the null hypothesis.
    alternative()
        Return the right tail of the statistic's distribution.

    Notes
    -----
    The null is t_df with unknown location and positive scale. Each call
    estimates location by the sample median and scale by
    (Q_0.75-Q_0.25)/(2*t_df.ppf(0.75)), using linear sample quantiles.
    Return max(max(i/n-u_i), max(u_i-(i-1)/n)) for the fitted CDF.
    This quantile estimator works without finite moments, including df <= 2;
    it is not maximum likelihood. Samples need n >= 2 and positive IQR.
    Ties are accepted if the IQR remains positive. Estimates are not stored.

    This is a locally specified Lilliefors-type extension. A search did not
    identify a primary paper establishing this exact Student/quantile version.
    The normal Lilliefors paper does not validate it. Large values reject.
    Location/scale equivariance removes those nuisance parameters; the null
    law still depends on df, sample size and the estimator. Refit each
    Monte Carlo replicate; ordinary KS and normal Lilliefors tables do not
    apply. The built-in sampler requires complete Student parameters and
    rejects this composite hypothesis. Calibrate externally at the fixed df
    (location 0, scale 1 is justified by equivariance), calling this method
    on every replicate. Previously stored LILLIE distributions are invalid.

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'df': 5})
    >>> statistic = LillieforsStudentGofStatistic(parameters)
    >>> value = statistic.execute_statistic([-2., -0.7, 0., 0.4, 1.8])
    >>> bool(np.isfinite(value))
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (HypothesisSupport(Student.DEFAULT, frozenset({Student.DF})),)

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "LILLIE".
        """
        return "LILLIE"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute fitted KS distance for Student t with known degrees of freedom.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real sample on the whole real line, with
            at least two observations and positive interquartile range.
            The input is not modified.
        **kwargs : dict
            Accepted for the common interface; no call-time options are used.

        Returns
        -------
        float or numpy.float64
            Unscaled statistic; large values reject the null.

        Raises
        ------
        ValueError
            If the sample is invalid or required numerical transformations
            cannot be represented in floating-point arithmetic.
            Also raised for fewer than two observations or zero IQR.
        """
        sample = self._validate_sample(rvs)
        if sample.size < 2:
            raise ValueError("At least two observations are required")
        # Normalize first, so quantile interpolation cannot overflow.
        magnitude = np.max(np.abs(sample))
        if magnitude == 0:
            raise ValueError("Sample interquartile range must be positive")
        sample = sample / magnitude
        q1, median, q3 = np.quantile(sample, [0.25, 0.5, 0.75], method="linear")
        reference = scipy_stats.t.ppf(0.75, self.df)
        fitted_scale = ((q3 - q1) / 2) / reference
        if not np.isfinite(fitted_scale) or fitted_scale <= 0:
            raise ValueError("Sample interquartile range must define a positive finite scale")
        with np.errstate(over="ignore"):
            standardized = (sample - median) / fitted_scale
        if not np.all(np.isfinite(standardized)):
            raise ValueError("Fitted observations exceed floating-point range")
        cdf_vals = scipy_stats.t.cdf(standardized, self.df)
        return float(LillieforsTest.do_execute_statistic(self, sample, cdf_vals))


class ChiSquareStudentGofStatistic(AbstractStudentGofStatistic):
    """Pearson chi-squared for equiprobable Student t bins.

    Parameters
    ----------
    parameters : ParameterValues
        Values with df, loc, scale fixed; omitted parameters are unknown.
    n_bins : int, default: 10
        Fixed number of equiprobable bins, at least 2; booleans are rejected.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar discrepancy without changing the input or fitting state.
    hypothesis()
        Return the fixed parameters of the null hypothesis.
    alternative()
        Return the right tail of the statistic's distribution.

    Notes
    -----
    The null fixes df, loc and scale on the whole real line. Write
    u_i = t_df.cdf((x_(i)-loc)/scale) for sorted observations, i=1,...,n.
    For k=n_bins, use edges t_df.ppf(j/k), j=0,...,k on standardized
    data, with infinite outer edges. Internal boundary ties enter the right
    bin. Return sum((O_j-n/k)**2/(n/k)), where O_j are observed counts.
    Empty cells contribute n/k. Under the null, counts are multinomial with
    probabilities 1/k. The chi-squared(k-1) law is only asymptotic and needs
    sufficiently large expected counts. Monte Carlo must retain n_bins.
    Large values reject. The probability integral transform removes all
    distribution parameters from the null law, not from the hypothesis.
    The reference describes a general statistic, applied here through the
    specified Student CDF, not a separately derived Student-specific test.

    References
    ----------
    .. [1] K. Pearson (1900), On the criterion that a given system of deviations
    from the probable ... , https://doi.org/10.1080/14786440009463897.

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'df': 5, 'loc': 0, 'scale': 1})
    >>> statistic = ChiSquareStudentGofStatistic(parameters)
    >>> value = statistic.execute_statistic([-2., -0.7, 0., 0.4, 1.8])
    >>> bool(np.isfinite(value))
    True
    """

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    def __init__(self, parameters: ParameterValues, *, n_bins: int = 10):
        super().__init__(parameters)
        if isinstance(n_bins, bool) or not isinstance(n_bins, Integral) or n_bins < 2:
            raise ValueError("n_bins must be an integer of at least 2")
        self.n_bins = int(n_bins)

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "CHI2".
        """
        return "CHI2"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute pearson chi-squared for equiprobable Student t bins.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real sample on the whole real line, with
            at least one observation; ties and constants are allowed.
            The input is not modified.
        **kwargs : dict
            Accepted for the common interface; no call-time options are used.

        Returns
        -------
        float or numpy.float64
            Unscaled statistic; large values reject the null.

        Raises
        ------
        ValueError
            If the sample is invalid or required numerical transformations
            cannot be represented in floating-point arithmetic.
        """
        rvs = self._validate_sample(rvs)
        n = len(rvs)
        standardized = self._standardize(rvs)

        # Create bin edges based on quantiles of the t-distribution
        bin_edges = scipy_stats.t.ppf(np.linspace(0, 1, self.n_bins + 1), self.df)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        if not np.all(np.isfinite(bin_edges[1:-1])) or np.any(np.diff(bin_edges) <= 0):
            raise ValueError("Student quantile bins cannot be represented distinctly")

        # Observed frequencies
        observed, _ = np.histogram(standardized, bins=bin_edges)

        # Expected frequencies (uniform for equiprobable bins)
        expected = np.ones(self.n_bins) * n / self.n_bins

        # Chi-square statistic
        chi2 = np.sum((observed - expected) ** 2 / expected)

        return chi2
