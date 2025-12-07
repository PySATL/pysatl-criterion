"""
Goodness-of-fit test statistics for the Student's t-distribution.

This module provides implementations of various goodness-of-fit test statistics
specifically designed for testing whether a sample comes from a Student's t-distribution.

The Student's t-distribution is a probability distribution that arises when estimating
the mean of a normally distributed population in situations where the sample size is small
and the population standard deviation is unknown.

References
----------
.. [1] D'Agostino, R. B., & Stephens, M. A. (1986). Goodness-of-fit techniques.
       Marcel Dekker, New York.
.. [2] Stephens, M. A. (1979). Tests of fit for the logistic distribution based on the
       empirical distribution function. Biometrika, 66(3), 591-595.
"""

from abc import ABC

import numpy as np
import scipy.stats as scipy_stats
from typing_extensions import override

from pysatl_criterion.statistics.common import ADStatistic, CrammerVonMisesStatistic, KSStatistic
from pysatl_criterion.statistics.goodness_of_fit import AbstractGoodnessOfFitStatistic


class AbstractStudentGofStatistic(AbstractGoodnessOfFitStatistic, ABC):
    """
    Abstract base class for Student's t-distribution goodness-of-fit statistics.

    This class serves as the base for all goodness-of-fit test statistics that test
    whether a sample comes from a Student's t-distribution.

    Parameters
    ----------
    df : float
        Degrees of freedom for the Student's t-distribution. Must be positive.
    loc : float, optional
        Location parameter (mean) of the distribution. Default is 0.
    scale : float, optional
        Scale parameter of the distribution. Must be positive. Default is 1.

    Attributes
    ----------
    df : float
        Degrees of freedom parameter.
    loc : float
        Location parameter.
    scale : float
        Scale parameter.

    Notes
    -----
    The Student's t-distribution with df degrees of freedom has PDF:
    .. math::

        f(x; \\nu) = \\frac{\\Gamma((\\nu+1)/2)}{\\sqrt{\\nu\\pi}\\Gamma(\\nu/2)}
                    \\left(1 + \\frac{x^2}{\\nu}\\right)^{-(\\nu+1)/2}

    where :math:`\\nu` is the degrees of freedom and :math:`\\Gamma` is the gamma function.
    """

    def __init__(self, df: float = 1, loc: float = 0, scale: float = 1):
        """
        Initialize the Student's t-distribution goodness-of-fit statistic.

        Parameters
        ----------
        df : float, optional
            Degrees of freedom for the Student's t-distribution.
            Must be positive. Default is 1.
        loc : float, optional
            Location parameter (mean) of the distribution. Default is 0.
        scale : float, optional
            Scale parameter of the distribution. Must be positive. Default is 1.

        Raises
        ------
        ValueError
            If df <= 0 or scale <= 0.
        """
        if df <= 0:
            raise ValueError("Degrees of freedom must be positive")
        if scale <= 0:
            raise ValueError("Scale must be positive")
        self.df = df
        self.loc = loc
        self.scale = scale

    @staticmethod
    @override
    def code():
        """
        Return the unique identifier code for Student's t-distribution GoF statistics.

        Returns
        -------
        str
            The code string "STUDENT_GOODNESS_OF_FIT".
        """
        return f"STUDENT_{AbstractGoodnessOfFitStatistic.code()}"


class KolmogorovSmirnovStudentGofStatistic(AbstractStudentGofStatistic, KSStatistic):
    """
    Kolmogorov-Smirnov test statistic for the Student's t-distribution.

    The Kolmogorov-Smirnov statistic quantifies the distance between the empirical
    distribution function of the sample and the cumulative distribution function
    of the reference Student's t-distribution.

    Parameters
    ----------
    df : float
        Degrees of freedom for the Student's t-distribution. Must be positive.
    loc : float, optional
        Location parameter of the distribution. Default is 0.
    scale : float, optional
        Scale parameter of the distribution. Default is 1.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis. Default is 'two-sided'.

    Notes
    -----
    The test statistic is defined as:

    .. math::

        D_n = \\sup_x |F_n(x) - F(x)|

    where :math:`F_n(x)` is the empirical distribution function and
    :math:`F(x)` is the CDF of the Student's t-distribution.

    References
    ----------
    .. [1] Stephens, M. A. (1974). EDF statistics for goodness of fit and some comparisons.
           Journal of the American Statistical Association, 69(347), 730-737.
    .. [2] D'Agostino, R. B., & Stephens, M. A. (1986). Goodness-of-fit techniques.
           Marcel Dekker, New York.
    """

    def __init__(
        self,
        df: float = 1,
        loc: float = 0,
        scale: float = 1,
        alternative: str = "two-sided",
    ):
        """
        Initialize the Kolmogorov-Smirnov test for Student's t-distribution.

        Parameters
        ----------
        df : float, optional
            Degrees of freedom for the Student's t-distribution.
            Must be positive. Default is 1.
        loc : float, optional
            Location parameter of the distribution. Default is 0.
        scale : float, optional
            Scale parameter of the distribution. Default is 1.
        alternative : {'two-sided', 'less', 'greater'}, optional
            Defines the alternative hypothesis. Default is 'two-sided'.
        """
        AbstractStudentGofStatistic.__init__(self, df, loc, scale)
        KSStatistic.__init__(self, alternative)

    @staticmethod
    @override
    def code():
        """
        Return the unique identifier code for this statistic.

        Returns
        -------
        str
            The code string "KS_STUDENT_GOODNESS_OF_FIT".
        """
        return f"KS_{AbstractStudentGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Calculate the Kolmogorov-Smirnov statistic for testing fit to Student's t-distribution.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.
        **kwargs : dict
            Additional keyword arguments (not used).

        Returns
        -------
        float
            The Kolmogorov-Smirnov test statistic value.

        Examples
        --------
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> data = np.random.standard_t(df=5, size=100)
        >>> stat = KolmogorovSmirnovStudentGofStatistic(df=5)
        >>> ks_value = stat.execute_statistic(data)
        """
        rvs = np.sort(rvs)
        # Standardize the data
        standardized = (rvs - self.loc) / self.scale
        cdf_vals = scipy_stats.t.cdf(standardized, self.df)
        return KSStatistic.execute_statistic(self, rvs, cdf_vals)


class AndersonDarlingStudentGofStatistic(AbstractStudentGofStatistic, ADStatistic):
    """
    Anderson-Darling test statistic for the Student's t-distribution.

    The Anderson-Darling test is a modification of the Kolmogorov-Smirnov test that
    gives more weight to the tails of the distribution. It is generally more powerful
    for detecting departures in the tails.

    Parameters
    ----------
    df : float
        Degrees of freedom for the Student's t-distribution. Must be positive.
    loc : float, optional
        Location parameter of the distribution. Default is 0.
    scale : float, optional
        Scale parameter of the distribution. Default is 1.

    Notes
    -----
    The Anderson-Darling statistic is defined as:

    .. math::

        A^2 = -n - \\frac{1}{n}\\sum_{i=1}^{n}(2i-1)[\\ln F(X_{(i)}) + \\ln(1-F(X_{(n+1-i)}))]

    where :math:`X_{(i)}` are the order statistics and :math:`F` is the CDF.

    References
    ----------
    .. [1] Anderson, T. W., & Darling, D. A. (1954). A test of goodness of fit.
           Journal of the American Statistical Association, 49(268), 765-769.
    .. [2] Stephens, M. A. (1986). Tests Based on EDF Statistics. In Goodness-of-Fit
           Techniques (pp. 97-193). Marcel Dekker.
    """

    @staticmethod
    @override
    def code():
        """
        Return the unique identifier code for this statistic.

        Returns
        -------
        str
            The code string "AD_STUDENT_GOODNESS_OF_FIT".
        """
        return f"AD_{AbstractStudentGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Calculate the Anderson-Darling statistic for testing fit to Student's t-distribution.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.
        **kwargs : dict
            Additional keyword arguments (not used).

        Returns
        -------
        float
            The Anderson-Darling test statistic value.

        Examples
        --------
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> data = np.random.standard_t(df=5, size=100)
        >>> stat = AndersonDarlingStudentGofStatistic(df=5)
        >>> ad_value = stat.execute_statistic(data)
        """
        y = np.sort(rvs)
        # Standardize the data
        standardized = (y - self.loc) / self.scale
        logcdf = scipy_stats.t.logcdf(standardized, self.df)
        logsf = scipy_stats.t.logsf(standardized, self.df)
        return ADStatistic.execute_statistic(self, y, log_cdf=logcdf, log_sf=logsf)


class CramerVonMisesStudentGofStatistic(AbstractStudentGofStatistic, CrammerVonMisesStatistic):
    """
    Cramer-von Mises test statistic for the Student's t-distribution.

    The Cramer-von Mises statistic is based on the integrated squared difference
    between the empirical distribution function and the theoretical CDF.

    Parameters
    ----------
    df : float
        Degrees of freedom for the Student's t-distribution. Must be positive.
    loc : float, optional
        Location parameter of the distribution. Default is 0.
    scale : float, optional
        Scale parameter of the distribution. Default is 1.

    Notes
    -----
    The Cramer-von Mises statistic is defined as:

    .. math::

        W^2 = \\frac{1}{12n} + \\sum_{i=1}^{n}\\left[\\frac{2i-1}{2n} - F(X_{(i)})\\right]^2

    where :math:`X_{(i)}` are the order statistics and :math:`F` is the CDF.

    References
    ----------
    .. [1] Cramér, H. (1928). On the composition of elementary errors. Skandinavisk
           Aktuarietidskrift, 11, 13-74.
    .. [2] von Mises, R. (1931). Wahrscheinlichkeitsrechnung und ihre Anwendung in
           der Statistik und theoretischen Physik. Deuticke.
    """

    @staticmethod
    @override
    def code():
        """
        Return the unique identifier code for this statistic.

        Returns
        -------
        str
            The code string "CVM_STUDENT_GOODNESS_OF_FIT".
        """
        return f"CVM_{AbstractStudentGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Calculate the Cramer-von Mises statistic for testing fit to Student's t-distribution.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.
        **kwargs : dict
            Additional keyword arguments (not used).

        Returns
        -------
        float
            The Cramer-von Mises test statistic value.

        Examples
        --------
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> data = np.random.standard_t(df=5, size=100)
        >>> stat = CramerVonMisesStudentGofStatistic(df=5)
        >>> cvm_value = stat.execute_statistic(data)
        """
        rvs = np.sort(rvs)
        # Standardize the data
        standardized = (rvs - self.loc) / self.scale
        cdf_vals = scipy_stats.t.cdf(standardized, self.df)
        return CrammerVonMisesStatistic.execute_statistic(self, rvs, cdf_vals)


class KuiperStudentGofStatistic(AbstractStudentGofStatistic):
    """
    Kuiper's test statistic for the Student's t-distribution.

    Kuiper's test is a variant of the Kolmogorov-Smirnov test that is equally
    sensitive at all points of the distribution. It is defined as the sum of
    the maximum positive and negative deviations.

    Parameters
    ----------
    df : float
        Degrees of freedom for the Student's t-distribution. Must be positive.
    loc : float, optional
        Location parameter of the distribution. Default is 0.
    scale : float, optional
        Scale parameter of the distribution. Default is 1.

    Notes
    -----
    Kuiper's statistic is defined as:

    .. math::

        V = D^+ + D^-

    where :math:`D^+ = \\max_i\\{i/n - F(X_{(i)})\\}` and
    :math:`D^- = \\max_i\\{F(X_{(i)}) - (i-1)/n\\}`.

    References
    ----------
    .. [1] Kuiper, N. H. (1960). Tests concerning random points on a circle.
           Proceedings of the Koninklijke Nederlandse Akademie van Wetenschappen,
           Series A, 63, 38-47.
    """

    @staticmethod
    @override
    def code():
        """
        Return the unique identifier code for this statistic.

        Returns
        -------
        str
            The code string "KUIPER_STUDENT_GOODNESS_OF_FIT".
        """
        return f"KUIPER_{AbstractStudentGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Calculate Kuiper's statistic for testing fit to Student's t-distribution.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.
        **kwargs : dict
            Additional keyword arguments (not used).

        Returns
        -------
        float
            Kuiper's test statistic value.

        Examples
        --------
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> data = np.random.standard_t(df=5, size=100)
        >>> stat = KuiperStudentGofStatistic(df=5)
        >>> v_value = stat.execute_statistic(data)
        """
        n = len(rvs)
        rvs = np.sort(rvs)
        # Standardize the data
        standardized = (rvs - self.loc) / self.scale
        cdf_vals = scipy_stats.t.cdf(standardized, self.df)

        # D+ and D-
        d_plus = np.max(np.arange(1.0, n + 1) / n - cdf_vals)
        d_minus = np.max(cdf_vals - np.arange(0.0, n) / n)

        return d_plus + d_minus


class WatsonStudentGofStatistic(AbstractStudentGofStatistic):
    """
    Watson's U^2 test statistic for the Student's t-distribution.

    Watson's U^2 is a modification of the Cramer-von Mises statistic that is
    invariant to cyclic transformations. It is particularly useful for testing
    distributions on the circle but can also be applied to linear distributions.

    Parameters
    ----------
    df : float
        Degrees of freedom for the Student's t-distribution. Must be positive.
    loc : float, optional
        Location parameter of the distribution. Default is 0.
    scale : float, optional
        Scale parameter of the distribution. Default is 1.

    Notes
    -----
    Watson's U^2 statistic is defined as:

    .. math::

        U^2 = W^2 - n(\\bar{F} - 0.5)^2

    where :math:`W^2` is the Cramer-von Mises statistic and
    :math:`\\bar{F} = \\frac{1}{n}\\sum_{i=1}^{n}F(X_{(i)})`.

    References
    ----------
    .. [1] Watson, G. S. (1961). Goodness-of-fit tests on a circle.
           Biometrika, 48(1/2), 109-114.
    """

    @staticmethod
    @override
    def code():
        """
        Return the unique identifier code for this statistic.

        Returns
        -------
        str
            The code string "WATSON_STUDENT_GOODNESS_OF_FIT".
        """
        return f"WATSON_{AbstractStudentGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Calculate Watson's U^2 statistic for testing fit to Student's t-distribution.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.
        **kwargs : dict
            Additional keyword arguments (not used).

        Returns
        -------
        float
            Watson's U^2 test statistic value.

        Examples
        --------
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> data = np.random.standard_t(df=5, size=100)
        >>> stat = WatsonStudentGofStatistic(df=5)
        >>> u2_value = stat.execute_statistic(data)
        """
        n = len(rvs)
        rvs = np.sort(rvs)
        # Standardize the data
        standardized = (rvs - self.loc) / self.scale
        cdf_vals = scipy_stats.t.cdf(standardized, self.df)

        # Cramer-von Mises statistic
        u = (2 * np.arange(1, n + 1) - 1) / (2 * n)
        w2 = 1 / (12 * n) + np.sum((u - cdf_vals) ** 2)

        # Mean of CDF values
        f_bar = np.mean(cdf_vals)

        # Watson's U^2
        u2 = w2 - n * (f_bar - 0.5) ** 2

        return u2


class ZhangZcStudentGofStatistic(AbstractStudentGofStatistic):
    """
    Zhang's Zc test statistic for the Student's t-distribution.

    Zhang's Zc statistic is designed to have good power against a wide range
    of alternatives. It combines information from both tails of the distribution.

    Parameters
    ----------
    df : float
        Degrees of freedom for the Student's t-distribution. Must be positive.
    loc : float, optional
        Location parameter of the distribution. Default is 0.
    scale : float, optional
        Scale parameter of the distribution. Default is 1.

    Notes
    -----
    Zhang's Zc statistic is defined as:

    .. math::

        Z_c = \\sum_{i=1}^{n}\\left[\\ln\\frac{F(X_{(i)})^{-1} - 1}{(n-0.5)/i - 1}\\right]^2

    References
    ----------
    .. [1] Zhang, J. (2002). Powerful goodness-of-fit tests based on the likelihood ratio.
           Journal of the Royal Statistical Society: Series B, 64(2), 281-294.
    """

    @staticmethod
    @override
    def code():
        """
        Return the unique identifier code for this statistic.

        Returns
        -------
        str
            The code string "ZHANG_ZC_STUDENT_GOODNESS_OF_FIT".
        """
        return f"ZHANG_ZC_{AbstractStudentGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Calculate Zhang's Zc statistic for testing fit to Student's t-distribution.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.
        **kwargs : dict
            Additional keyword arguments (not used).

        Returns
        -------
        float
            Zhang's Zc test statistic value.

        Examples
        --------
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> data = np.random.standard_t(df=5, size=100)
        >>> stat = ZhangZcStudentGofStatistic(df=5)
        >>> zc_value = stat.execute_statistic(data)
        """
        n = len(rvs)
        rvs = np.sort(rvs)
        # Standardize the data
        standardized = (rvs - self.loc) / self.scale
        cdf_vals = scipy_stats.t.cdf(standardized, self.df)

        # Avoid log(0) issues by clipping
        cdf_vals = np.clip(cdf_vals, 1e-10, 1 - 1e-10)

        i = np.arange(1, n + 1)
        # Zhang's Zc statistic
        term1 = 1 / cdf_vals - 1
        term2 = (n - 0.5) / i - 1
        # Avoid division by zero and log of negative numbers
        term2 = np.where(term2 <= 0, 1e-10, term2)
        zc = np.sum(np.log(term1 / term2) ** 2)

        return zc


class ZhangZaStudentGofStatistic(AbstractStudentGofStatistic):
    """
    Zhang's Za test statistic for the Student's t-distribution.

    Zhang's Za statistic is an alternative version of Zhang's test that has
    good power properties for detecting departures from the null distribution.

    Parameters
    ----------
    df : float
        Degrees of freedom for the Student's t-distribution. Must be positive.
    loc : float, optional
        Location parameter of the distribution. Default is 0.
    scale : float, optional
        Scale parameter of the distribution. Default is 1.

    Notes
    -----
    Zhang's Za statistic is defined as:

    .. math::

        Z_a = -\\sum_{i=1}^{n}\\frac{\\ln F(X_{(i)})}{n - i + 0.5}
              -\\sum_{i=1}^{n}\\frac{\\ln(1 - F(X_{(i)}))}{i - 0.5}

    References
    ----------
    .. [1] Zhang, J. (2002). Powerful goodness-of-fit tests based on the likelihood ratio.
           Journal of the Royal Statistical Society: Series B, 64(2), 281-294.
    """

    @staticmethod
    @override
    def code():
        """
        Return the unique identifier code for this statistic.

        Returns
        -------
        str
            The code string "ZHANG_ZA_STUDENT_GOODNESS_OF_FIT".
        """
        return f"ZHANG_ZA_{AbstractStudentGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Calculate Zhang's Za statistic for testing fit to Student's t-distribution.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.
        **kwargs : dict
            Additional keyword arguments (not used).

        Returns
        -------
        float
            Zhang's Za test statistic value.

        Examples
        --------
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> data = np.random.standard_t(df=5, size=100)
        >>> stat = ZhangZaStudentGofStatistic(df=5)
        >>> za_value = stat.execute_statistic(data)
        """
        n = len(rvs)
        rvs = np.sort(rvs)
        # Standardize the data
        standardized = (rvs - self.loc) / self.scale
        cdf_vals = scipy_stats.t.cdf(standardized, self.df)

        # Avoid log(0) issues by clipping
        cdf_vals = np.clip(cdf_vals, 1e-10, 1 - 1e-10)

        i = np.arange(1, n + 1)
        # Zhang's Za statistic
        za = -np.sum(np.log(cdf_vals) / (n - i + 0.5)) - np.sum(
            np.log(1 - cdf_vals) / (i - 0.5)
        )

        return za


class LillieforsStudentGofStatistic(AbstractStudentGofStatistic, KSStatistic):
    """
    Lilliefors-type test statistic for the Student's t-distribution.

    This is a Kolmogorov-Smirnov type test where the parameters are estimated
    from the data rather than specified. The test uses the empirical distribution
    function compared to the fitted Student's t-distribution.

    Parameters
    ----------
    df : float
        Degrees of freedom for the Student's t-distribution. Must be positive.

    Notes
    -----
    Unlike the standard KS test, the Lilliefors test estimates the location
    and scale parameters from the sample, which affects the null distribution
    of the test statistic.

    References
    ----------
    .. [1] Lilliefors, H. W. (1967). On the Kolmogorov-Smirnov test for normality
           with mean and variance unknown. Journal of the American Statistical
           Association, 62(318), 399-402.
    """

    def __init__(self, df: float = 1):
        """
        Initialize the Lilliefors-type test for Student's t-distribution.

        Parameters
        ----------
        df : float, optional
            Degrees of freedom for the Student's t-distribution.
            Must be positive. Default is 1.
        """
        AbstractStudentGofStatistic.__init__(self, df, 0, 1)
        KSStatistic.__init__(self, "two-sided")

    @staticmethod
    @override
    def code():
        """
        Return the unique identifier code for this statistic.

        Returns
        -------
        str
            The code string "LILLIE_STUDENT_GOODNESS_OF_FIT".
        """
        return f"LILLIE_{AbstractStudentGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Calculate the Lilliefors statistic for testing fit to Student's t-distribution.

        The location and scale parameters are estimated from the sample data.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.
        **kwargs : dict
            Additional keyword arguments (not used).

        Returns
        -------
        float
            The Lilliefors test statistic value.

        Raises
        ------
        ValueError
            If the sample standard deviation is zero.

        Examples
        --------
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> data = np.random.standard_t(df=5, size=100)
        >>> stat = LillieforsStudentGofStatistic(df=5)
        >>> lillie_value = stat.execute_statistic(data)
        """
        x = np.asarray(rvs)
        # Estimate location and scale from data
        loc = np.mean(x)
        scale = np.std(x, ddof=1)
        if scale == 0:
            raise ValueError("Sample standard deviation is zero; Lilliefors undefined")
        # Standardize
        z = (x - loc) / scale
        cdf_vals = scipy_stats.t.cdf(z, self.df)
        return KSStatistic.execute_statistic(self, z, cdf_vals)

class ChiSquareStudentGofStatistic(AbstractStudentGofStatistic):
    """
    Chi-square goodness-of-fit test statistic for the Student's t-distribution.

    The chi-square test compares observed frequencies in bins to expected
    frequencies under the null hypothesis.

    Parameters
    ----------
    df : float
        Degrees of freedom for the Student's t-distribution. Must be positive.
    loc : float, optional
        Location parameter of the distribution. Default is 0.
    scale : float, optional
        Scale parameter of the distribution. Default is 1.
    n_bins : int, optional
        Number of bins for the chi-square test. Default is 10.

    Notes
    -----
    The chi-square statistic is defined as:

    .. math::

        \\chi^2 = \\sum_{i=1}^{k}\\frac{(O_i - E_i)^2}{E_i}

    where :math:`O_i` are observed frequencies and :math:`E_i` are expected frequencies.

    References
    ----------
    .. [1] Pearson, K. (1900). On the criterion that a given system of deviations
           from the probable in the case of a correlated system of variables is such
           that it can be reasonably supposed to have arisen from random sampling.
           Philosophical Magazine, 50(302), 157-175.
    """

    def __init__(
        self,
        df: float = 1,
        loc: float = 0,
        scale: float = 1,
        n_bins: int = 10,
    ):
        """
        Initialize the Chi-square test for Student's t-distribution.

        Parameters
        ----------
        df : float, optional
            Degrees of freedom for the Student's t-distribution.
            Must be positive. Default is 1.
        loc : float, optional
            Location parameter of the distribution. Default is 0.
        scale : float, optional
            Scale parameter of the distribution. Default is 1.
        n_bins : int, optional
            Number of bins for the chi-square test. Default is 10.
        """
        super().__init__(df, loc, scale)
        self.n_bins = n_bins

    @staticmethod
    @override
    def code():
        """
        Return the unique identifier code for this statistic.

        Returns
        -------
        str
            The code string "CHI2_STUDENT_GOODNESS_OF_FIT".
        """
        return f"CHI2_{AbstractStudentGofStatistic.code()}"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Calculate the Chi-square statistic for testing fit to Student's t-distribution.

        Parameters
        ----------
        rvs : array_like
            Array of sample data.
        **kwargs : dict
            Additional keyword arguments (not used).

        Returns
        -------
        float
            The Chi-square test statistic value.

        Examples
        --------
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> data = np.random.standard_t(df=5, size=100)
        >>> stat = ChiSquareStudentGofStatistic(df=5, n_bins=10)
        >>> chi2_value = stat.execute_statistic(data)
        """
        n = len(rvs)
        # Standardize the data
        standardized = (np.array(rvs) - self.loc) / self.scale

        # Create bin edges based on quantiles of the t-distribution
        bin_edges = scipy_stats.t.ppf(
            np.linspace(0, 1, self.n_bins + 1), self.df
        )
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        # Observed frequencies
        observed, _ = np.histogram(standardized, bins=bin_edges)

        # Expected frequencies (uniform for equiprobable bins)
        expected = np.ones(self.n_bins) * n / self.n_bins

        # Chi-square statistic
        chi2 = np.sum((observed - expected) ** 2 / expected)

        return chi2
