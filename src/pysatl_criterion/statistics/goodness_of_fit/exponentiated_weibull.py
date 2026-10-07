"""Specified exponentiated Weibull criteria with fixed exponent, shape and scale.

F(x) = (1 - exp(-(x / scale)**shape))**exponent, x >= 0; location is zero.
"""

from abc import ABC

import numpy as np
from typing_extensions import override

from pysatl_criterion.core.distributions.continues.exponentiated_weibull import (
    generate_exponentiated_weibull_cdf,
)
from pysatl_criterion.distribution.distributions import (
    ExponentiatedWeibullDistributionDescriptor as Distribution,
)
from pysatl_criterion.distribution.parameters import HypothesisSupport, ParameterValues
from pysatl_criterion.statistics import AbstractGoodnessOfFitStatistic
from pysatl_criterion.statistics.alternative import AlternativeType, RightAlternative
from pysatl_criterion.statistics.goodness_of_fit._weibull_common import (
    _log_probabilities,
    _sample,
    _weighted_edf,
)
from pysatl_criterion.statistics.goodness_of_fit.common import (
    Chi2Statistic,
    CrammerVonMisesStatistic,
    KSStatistic,
    MinToshiyukiStatistic,
)


class AbstractExponentiatedWeibullGofStatistic(AbstractGoodnessOfFitStatistic, ABC):
    """Fully specified exponentiated Weibull on the nonnegative half-line."""

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        )

    @staticmethod
    def distribution() -> type[Distribution]:
        return Distribution

    @classmethod
    def code(cls) -> str:
        family_code = f"EXPONENTIATED_WEIBULL_{AbstractGoodnessOfFitStatistic.code()}"
        if "short_code" in cls.__abstractmethods__:
            return family_code
        return f"{cls.short_code()}_{family_code}"

    def _validate_storage_calibration(self):
        raise ValueError(
            "Exponentiated Weibull uses new distribution and criterion identifiers: "
            "regenerate calibration; unversioned stored critical values are not supported"
        )

    @property
    def exponent(self) -> float:
        return self._parameters[Distribution.EXPONENT]

    @property
    def shape(self) -> float:
        return self._parameters[Distribution.SHAPE]

    @property
    def scale(self) -> float:
        return self._parameters[Distribution.SCALE]


class MinToshiyukiExponentiatedWeibullGofStatistic(
    AbstractExponentiatedWeibullGofStatistic, MinToshiyukiStatistic
):
    """Weighted EDF distance for a specified exponentiated Weibull.

    Parameters
    ----------
    parameters : ParameterValues
        Distribution.DEFAULT with exponent, shape and scale fixed.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar statistic, fitting parameters internally when needed.
    hypothesis()
        Return the fixed parameters; omitted parameters are unknown.
    alternative()
        Return the critical-region tail.

    Notes
    -----
    F(x)=(1-exp(-(x/scale)**k))**a on x >= 0, with fixed a, k, positive scale
    and location 0. Put u_i=F(x_(i)), i=1,...,n.

    L = sum(max(i/n-u_i, u_i-(i-1)/n)/sqrt(u_i*(1-u_i)))/sqrt(n).

    Reject for right tail values.
    Calibrate using the specified null and the same sample size.
    Previously stored unversioned Weibull calibrations must be regenerated.
    The historical name uses the given names of Liao and Shimokawa.
    This fixed-CDF variant is not their fitted-parameter procedure.
    A zero observation gives positive infinity. Log probabilities preserve
    finite penalties when a CDF rounds to one; overflow may still give infinity.
    A primary source establishing this exact implemented formula was not
    verified in the literature search. Treat it as the specified local
    discrepancy; no named-test tables or chi-square limit are asserted.

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({"exponent": 1, "shape": 2, "scale": 1})
    >>> test = MinToshiyukiExponentiatedWeibullGofStatistic(parameters)
    >>> value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "MT".
        """
        return "MT"

    def alternative(self):
        return RightAlternative()

    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like
            One-dimensional finite nonnegative observations, n >= 1.
            Ties and constant samples are allowed.
        **kwargs : dict
            Unused common-interface keywords.

        Returns
        -------
        statistic : float
            Scalar discrepancy. Infinite boundary penalties are preserved.

        Raises
        ------
        ValueError
            Invalid sample, unsupported settings, or unrepresentable fit.

        Notes
        -----
        The input is never changed and estimated parameters are not retained.
        """
        x = np.sort(_sample(rvs))
        with np.errstate(divide="ignore", over="ignore"):
            z = self.shape * (np.log(x) - np.log(self.scale))
        return _weighted_edf(*_log_probabilities(z, self.exponent))


class Chi2PearsonExponentiatedWeibullGofStatistic(
    AbstractExponentiatedWeibullGofStatistic, Chi2Statistic
):
    """Pearson counts statistic in equal-probability bins.

    Parameters
    ----------
    parameters : ParameterValues
        Distribution.DEFAULT with exponent, shape and scale fixed.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar statistic, fitting parameters internally when needed.
    hypothesis()
        Return the fixed parameters; omitted parameters are unknown.
    alternative()
        Return the critical-region tail.

    Notes
    -----
    F(x)=(1-exp(-(x/scale)**k))**a on x >= 0, with fixed a, k, positive scale
    and location 0. Put u_i=F(x_(i)), i=1,...,n.

    B=ceil(sqrt(n)); T=sum((O_j-n/B)**2/(n/B)), j=1,...,B.
    Bins in CDF space cover [0,1], including both tails.

    Reject for right tail values.
    Calibrate using the specified null and the same sample size.
    Previously stored unversioned Weibull calibrations must be regenerated.
    The source gives the general count statistic, applied here to known
    cell probabilities. For small expected counts use simulation, not
    chi-square tables; the number of bins depends on sample size.
    For n=1 there is one bin and the statistic is identically zero.

    References
    ----------
    .. [1] K. Pearson (1900), On the criterion that a given system of deviations
       from the probable in the case of a correlated system of variables is
       such that it can be reasonably supposed to have arisen from random
       sampling. https://doi.org/10.1080/14786440009463897

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({"exponent": 1, "shape": 2, "scale": 1})
    >>> test = Chi2PearsonExponentiatedWeibullGofStatistic(parameters)
    >>> value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "CHI2_PEARSON".
        """
        return "CHI2_PEARSON"

    def alternative(self):
        return RightAlternative()

    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like
            One-dimensional finite nonnegative observations, n >= 1.
            Ties and constant samples are allowed.
        **kwargs : dict
            Unused common-interface keywords.

        Returns
        -------
        statistic : float
            Scalar discrepancy. Infinite boundary penalties are preserved.

        Raises
        ------
        ValueError
            Invalid sample, unsupported settings, or unrepresentable fit.

        Notes
        -----
        The input is never changed and estimated parameters are not retained.
        """
        x = _sample(rvs)
        n = len(x)
        bins = int(np.ceil(np.sqrt(n)))
        u = generate_exponentiated_weibull_cdf(
            x, exponent=self.exponent, shape=self.shape, scale=self.scale
        )
        observed, _ = np.histogram(u, bins=np.linspace(0, 1, bins + 1))
        return float(self.do_execute_statistic(observed, np.full(bins, n / bins), 1))


class CrammerVonMisesExponentiatedWeibullGofStatistic(
    AbstractExponentiatedWeibullGofStatistic, CrammerVonMisesStatistic
):
    """Cramer-von Mises distance for a specified exponentiated Weibull.

    Parameters
    ----------
    parameters : ParameterValues
        Distribution.DEFAULT with exponent, shape and scale fixed.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar statistic, fitting parameters internally when needed.
    hypothesis()
        Return the fixed parameters; omitted parameters are unknown.
    alternative()
        Return the critical-region tail.

    Notes
    -----
    F(x)=(1-exp(-(x/scale)**k))**a on x >= 0, with fixed a, k, positive scale
    and location 0. Put u_i=F(x_(i)), i=1,...,n.

    W2=1/(12*n)+sum((u_i-(2*i-1)/(2*n))**2).

    Reject for right tail values.
    Calibrate using the specified null and the same sample size.
    Previously stored unversioned Weibull calibrations must be regenerated.
    The reference treats the general EDF functional; here the known CDF
    transforms the null to uniformity.

    References
    ----------
    .. [1] T. W. Anderson and D. A. Darling (1952), Asymptotic theory of certain
       "goodness of fit" criteria based on stochastic processes.
       https://doi.org/10.1214/aoms/1177729437

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({"exponent": 1, "shape": 2, "scale": 1})
    >>> test = CrammerVonMisesExponentiatedWeibullGofStatistic(parameters)
    >>> value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
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

    def alternative(self):
        return RightAlternative()

    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like
            One-dimensional finite nonnegative observations, n >= 1.
            Ties and constant samples are allowed.
        **kwargs : dict
            Unused common-interface keywords.

        Returns
        -------
        statistic : float
            Scalar discrepancy. Infinite boundary penalties are preserved.

        Raises
        ------
        ValueError
            Invalid sample, unsupported settings, or unrepresentable fit.

        Notes
        -----
        The input is never changed and estimated parameters are not retained.
        """
        x = np.sort(_sample(rvs))
        return float(
            self.do_execute_statistic(
                x,
                generate_exponentiated_weibull_cdf(
                    x, exponent=self.exponent, shape=self.shape, scale=self.scale
                ),
            )
        )


class KolmogorovSmirnovExponentiatedWeibullGofStatistic(
    AbstractExponentiatedWeibullGofStatistic, KSStatistic
):
    """Kolmogorov-Smirnov distance for a specified exponentiated Weibull.

    Parameters
    ----------
    parameters : ParameterValues
        Distribution.DEFAULT with exponent, shape and scale fixed.
    alternative_type : AlternativeType, optional
        TWO_TAILED gives D; RIGHT gives D+; LEFT gives D-.
    mode : str, optional
        Compatibility setting, default "auto"; no effect on the statistic.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar statistic, fitting parameters internally when needed.
    hypothesis()
        Return the fixed parameters; omitted parameters are unknown.
    alternative()
        Return the critical-region tail.

    Notes
    -----
    F(x)=(1-exp(-(x/scale)**k))**a on x >= 0, with fixed a, k, positive scale
    and location 0. Put u_i=F(x_(i)), i=1,...,n.

    D+=max(i/n-u_i); D-=max(u_i-(i-1)/n); D=max(D+,D-).
    Every CDF-deviation direction has a right-tail rejection region.

    Reject for right tail values.
    Calibrate using the specified null and the same sample size.
    Previously stored unversioned Weibull calibrations must be regenerated.
    The reference concerns general continuous CDFs, applied here via F.

    References
    ----------
    .. [1] N. Smirnov (1948), Table for estimating the goodness of fit of empirical
       distributions. https://doi.org/10.1214/aoms/1177730256

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({"exponent": 1, "shape": 2, "scale": 1})
    >>> test = KolmogorovSmirnovExponentiatedWeibullGofStatistic(parameters)
    >>> value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
    >>> bool(np.isfinite(value))
    True
    """

    def __init__(
        self,
        parameters: ParameterValues,
        *,
        alternative_type=AlternativeType.TWO_TAILED,
        mode="auto",
    ):
        AbstractExponentiatedWeibullGofStatistic.__init__(self, parameters)
        if alternative_type not in (
            AlternativeType.TWO_TAILED,
            AlternativeType.RIGHT,
            AlternativeType.LEFT,
        ):
            raise ValueError("Invalid alternative_type")
        KSStatistic.__init__(self, alternative_type, mode)

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "KS".
        """
        return "KS"

    def alternative(self):
        return RightAlternative()

    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like
            One-dimensional finite nonnegative observations, n >= 1.
            Ties and constant samples are allowed.
        **kwargs : dict
            Unused common-interface keywords.

        Returns
        -------
        statistic : float
            Scalar discrepancy. Infinite boundary penalties are preserved.

        Raises
        ------
        ValueError
            Invalid sample, unsupported settings, or unrepresentable fit.

        Notes
        -----
        The input is never changed and estimated parameters are not retained.
        """
        x = np.sort(_sample(rvs))
        return float(
            self.do_execute_statistic(
                x,
                generate_exponentiated_weibull_cdf(
                    x, exponent=self.exponent, shape=self.shape, scale=self.scale
                ),
            )
        )


class WatsonExponentiatedWeibullGofStatistic(CrammerVonMisesExponentiatedWeibullGofStatistic):
    """Watson centered EDF statistic for a specified exponentiated Weibull.

    Parameters
    ----------
    parameters : ParameterValues
        Distribution.DEFAULT with exponent, shape and scale fixed.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return a scalar statistic, fitting parameters internally when needed.
    hypothesis()
        Return the fixed parameters; omitted parameters are unknown.
    alternative()
        Return the critical-region tail.

    Notes
    -----
    F(x)=(1-exp(-(x/scale)**k))**a on x >= 0, with fixed a, k, positive scale
    and location 0. Put u_i=F(x_(i)), i=1,...,n.

    U2=W2-n*(mean(u_i)-1/2)**2; W2 is the Cramer-von Mises statistic.

    Reject for right tail values.
    Calibrate using the specified null and the same sample size.
    Previously stored unversioned Weibull calibrations must be regenerated.
    The general centered EDF functional is applied to the known CDF.
    Centering residuals before squaring avoids subtracting two large sums.

    References
    ----------
    .. [1] G. S. Watson (1961), Goodness-of-fit tests on a circle.
       https://doi.org/10.1093/biomet/48.1-2.109

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({"exponent": 1, "shape": 2, "scale": 1})
    >>> test = WatsonExponentiatedWeibullGofStatistic(parameters)
    >>> value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
    >>> bool(np.isfinite(value))
    True
    """

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "W".
        """
        return "W"

    def alternative(self):
        return RightAlternative()

    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic described in the class Notes.

        Parameters
        ----------
        rvs : array_like
            One-dimensional finite nonnegative observations, n >= 1.
            Ties and constant samples are allowed.
        **kwargs : dict
            Unused common-interface keywords.

        Returns
        -------
        statistic : float
            Scalar discrepancy. Infinite boundary penalties are preserved.

        Raises
        ------
        ValueError
            Invalid sample, unsupported settings, or unrepresentable fit.

        Notes
        -----
        The input is never changed and estimated parameters are not retained.
        """
        x = np.sort(_sample(rvs))
        u = generate_exponentiated_weibull_cdf(
            x, exponent=self.exponent, shape=self.shape, scale=self.scale
        )
        target = (np.arange(1, len(x) + 1) - 0.5) / len(x)
        delta = u - target
        return float(1 / (12 * len(x)) + np.sum((delta - np.mean(delta)) ** 2))
