"""Goodness-of-fit statistics for fixed and fitted Laplace distributions.

EDF definitions use fixed parameters; the published specialist tests fit parameters.
"""

from __future__ import annotations

from abc import ABC

import numpy as np
from numba import jit
from typing_extensions import override

from pysatl_criterion.distribution import LaplaceDistributionDescriptor
from pysatl_criterion.distribution import LaplaceDistributionDescriptor as Distribution
from pysatl_criterion.distribution.parameters import HypothesisSupport
from pysatl_criterion.statistics.alternative import (
    Alternative,
    LeftAlternative,
    RightAlternative,
    TwoSidedAlternative,
)
from pysatl_criterion.statistics.goodness_of_fit.continuous import edf, frequency
from pysatl_criterion.statistics.goodness_of_fit.continuous._edf_factory import (
    EDFSpec,
    export_edf_statistics,
)
from pysatl_criterion.statistics.goodness_of_fit.continuous._frequency_factory import (
    FrequencySpec,
    export_frequency_statistics,
)
from pysatl_criterion.statistics.goodness_of_fit.goodness_of_fit_statistic import (
    AbstractGoodnessOfFitStatistic,
)


class AbstractLaplaceGofStatistic(AbstractGoodnessOfFitStatistic, ABC):
    """Abstract base class for Laplace distribution goodness-of-fit statistics."""

    def __getstate__(self):
        """Serialize the statistic without read-only parameter mappings.

        Returns
        -------
        dict
            Instance state with parameter values converted to dictionaries.
        """
        state = self.__dict__.copy()
        for key in ("_parameters", "_hypothesis_parameters"):
            state[key] = state[key].as_dict()
        return state

    def __setstate__(self, state):
        """Restore parameters and algorithm options from serialized state.

        Parameters
        ----------
        state : dict
            Instance state produced by ``__getstate__``.
        """
        state = state.copy()
        for key in ("_parameters", "_hypothesis_parameters"):
            if isinstance(state[key], dict):
                state[key] = Distribution.DEFAULT.parse(state[key])
        self.__dict__.update(state)

    @property
    def t(self) -> float:
        """Return the fixed location parameter.

        Returns
        -------
        float
            Location t; available when the hypothesis fixes location.
        """
        return self._parameters[Distribution.LOCATION]

    @property
    def s(self) -> float:
        """Return the fixed scale parameter.

        Returns
        -------
        float
            Positive scale s; available when the hypothesis fixes scale.
        """
        return self._parameters[Distribution.SCALE]

    @staticmethod
    def _validate_sample(rvs):
        """Validate and sort observations without changing the input.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Nonempty real sample without non-finite or masked observations.

        Returns
        -------
        ndarray of float
            Sorted copy with dtype float64. Ties and constant samples are allowed.

        Raises
        ------
        ValueError
            If the supplied sample or setting violates the stated constraints.
        """
        if np.ma.isMaskedArray(rvs) and np.any(rvs.mask):
            raise ValueError("Sample must contain no masked observations.")
        try:
            sample = np.asarray(rvs)
            if sample.dtype.kind not in "iuf":
                raise ValueError("Observations must be real numbers.")
            sample = np.asarray(sample, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Sample must contain finite real numbers.") from exc
        if sample.ndim != 1:
            raise ValueError("Sample must be one-dimensional.")
        if sample.size == 0:
            raise ValueError("At least one observation is required.")
        if not np.all(np.isfinite(sample)):
            raise ValueError("Sample must contain finite real numbers.")
        return np.sort(sample)

    @staticmethod
    @override
    def distribution() -> type[LaplaceDistributionDescriptor]:
        """Return the Laplace distribution descriptor.

        Returns
        -------
        type of LaplaceDistributionDescriptor
            Descriptor defining the location-scale parameterization.
        """
        return LaplaceDistributionDescriptor

    @classmethod
    @override
    def code(cls) -> str:
        """Return the family code or a concrete criterion identifier.

        Returns
        -------
        str
            Family identifier for an abstract criterion, otherwise the short
            code followed by the Laplace goodness-of-fit identifier.
        """
        family_code = f"LAPLACE_{AbstractGoodnessOfFitStatistic.code()}"
        if "short_code" in cls.__abstractmethods__:
            return family_code
        return f"{cls.short_code()}_{family_code}"

    @staticmethod
    @jit(nopython=True, cache=True)
    def _cdf_values(sample, t, s):
        """Evaluate the Laplace CDF at each observation.

        Parameters
        ----------
        sample : ndarray of float, shape (n,)
            Validated finite observations.
        t : float
            Fixed location.
        s : float
            Positive fixed scale.

        Returns
        -------
        ndarray of float, shape (n,)
            CDF values in sample order; extreme tails may round to zero or one.
        """
        values = np.empty(len(sample))
        for i in range(len(sample)):
            difference = sample[i] - t
            z = sample[i] / s - t / s if np.isinf(difference) else difference / s
            values[i] = 0.5 * np.exp(z) if z < 0 else 1.0 - 0.5 * np.exp(-z)
        return values

    @staticmethod
    @jit(nopython=True, cache=True)
    def _log_probabilities(sample, t, s):
        """Evaluate the log-CDF and log-survival function analytically.

        Parameters
        ----------
        sample : ndarray of float, shape (n,)
            Validated finite observations.
        t : float
            Fixed location.
        s : float
            Positive fixed scale.

        Returns
        -------
        log_cdf : ndarray of float, shape (n,)
            Logarithms of CDF values.
        log_sf : ndarray of float, shape (n,)
            Logarithms of survival probabilities, avoiding subtraction from one.
        """
        log_cdf = np.empty(len(sample))
        log_sf = np.empty(len(sample))
        for i in range(len(sample)):
            difference = sample[i] - t
            z = sample[i] / s - t / s if np.isinf(difference) else difference / s
            small = -abs(z) - np.log(2.0)
            large = np.log1p(-0.5 * np.exp(-abs(z)))
            log_cdf[i] = small if z < 0 else large
            log_sf[i] = large if z < 0 else small
        return log_cdf, log_sf

    def _cdf(self, sample):
        """Evaluate the CDF using the stored fixed parameters.

        Parameters
        ----------
        sample : ndarray of float, shape (n,)
            Validated finite observations.

        Returns
        -------
        ndarray of float, shape (n,)
            Laplace CDF values in sample order.
        """
        return self._cdf_values(sample, self.t, self.s)

    def _logs(self, sample):
        """Evaluate log-probabilities using the stored fixed parameters.

        Parameters
        ----------
        sample : ndarray of float, shape (n,)
            Validated finite observations.

        Returns
        -------
        tuple of ndarray
            Log-CDF and log-survival arrays in sample order.
        """
        return self._log_probabilities(sample, self.t, self.s)

    def do_execute_statistic(self, sorted_rvs):
        """Delegate to the public execution method.

        Parameters
        ----------
        sorted_rvs : array_like, shape (n,)
            Observations; unsorted input is also accepted and validated.

        Returns
        -------
        float
            Statistic calculated according to the concrete criterion.
        """
        return self.execute_statistic(sorted_rvs)


class GreenwoodLaplaceGofStatistic(AbstractLaplaceGofStatistic):
    """Greenwood statistic G for a fully specified Laplace CDF.

    Parameters
    ----------
    parameters : ParameterValues
        Values with t, s fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar statistic without changing the sample or model.
    hypothesis()
        Return the fixed parameters ``{"t": t, "s": s}``.
    alternative()
        Return the right-tailed rejection rule.

    Notes
    -----
    H0: observations are independent Laplace(t, s) on the real line.
    Let u_i = F_(t,s)(x_(i)) for sorted observations and sample size n.
    G = sum((u_i-u_(i-1))**2), i = 1, ..., n+1,
    with u_0=0 and u_(n+1)=1. Both endpoint spacings are included;
    repeated observations yield zero spacings. The result is unscaled.
    This right-tail spacing test detects clustering; it is not asserted
    to be consistent against every nonuniform alternative.

    Large values reject H0. Under H0 the probability integral transform
    gives independent uniforms: the null law depends on n, not t or s.
    Calibration must use this unmodified statistic and fixed parameters.
    Parameters fitted to the tested sample require a different calibration.
    Reference [1]_ supplies the general statistic; applying it through
    the known Laplace CDF does not define a fitted Laplace-family test.
    CDF values may round to 0 or 1 in extreme tails; small spacings can
    lose precision. Observations and probabilities are never clipped.

    References
    ----------
    .. [1] R. J. M. M. Does, R. Helmers and C. A. J. Klaassen (1988),
        "Approximating the distribution of Greenwood's statistic",
        Statistica Neerlandica 42, 153-162, Section 1, equations (1)-(2).
        https://ir.cwi.nl/pub/1694/1694D.pdf

    Examples
    --------
    >>> import numpy as np
    >>> from pysatl_criterion.distribution import LaplaceDistributionDescriptor as Distribution
    >>> parameters = Distribution.DEFAULT.parse({'t': 0.0, 's': 1.0})
    >>> statistic = GreenwoodLaplaceGofStatistic(parameters)
    >>> value = statistic.execute_statistic([-1.0, 0.0, 0.5, 2.0])
    >>> bool(np.isfinite(value))
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        """Return supported fixed-parameter Laplace hypotheses.

        Returns
        -------
        tuple of HypothesisSupport
            The default parameterization with both location and scale fixed.
        """
        return (
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.LOCATION, Distribution.SCALE})
            ),
        )

    @override
    def alternative(self) -> Alternative:
        """Return the right-tailed rejection rule.

        Returns
        -------
        RightAlternative
            Reject the null for large Greenwood statistics.
        """
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """Return the Greenwood short identifier.

        Returns
        -------
        str
            The code ``GRW``.
        """
        return "GRW"

    @staticmethod
    @jit(nopython=True, cache=True)
    def _spacing_statistic(cdf_values):
        """Sum squared uniform spacings, including both endpoints.

        Parameters
        ----------
        cdf_values : ndarray of float, shape (n,)
            Nondecreasing CDF values in [0, 1].

        Returns
        -------
        float
            Unscaled Greenwood statistic, including spacings from zero and to one.

        References
        ----------
        .. [1] Does, Helmers and Klaassen (1988). Approximating the distribution
            of Greenwood's statistic. https://ir.cwi.nl/pub/1694/1694D.pdf
        """
        total = 0.0
        previous = 0.0
        for value in cdf_values:
            total += (value - previous) ** 2
            previous = value
        return total + (1.0 - previous) ** 2

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the unscaled Greenwood statistic G.

        Parameters
        ----------
        rvs : array_like of real numbers, shape (n,)
            Finite one-dimensional sample with n >= 1. Ties and constant
            samples are allowed. Input is copied for sorting.
        **kwargs : dict
            Accepted for interface compatibility; ignored. There are no
            per-call parameter overrides or fitting options.

        Returns
        -------
        float
            G; larger values indicate departure from the fixed null.

        Raises
        ------
        ValueError
            If rvs is empty, non-real, non-finite, or not one-dimensional.

        Notes
        -----
        The mathematical result is finite for every valid finite sample.
        """

        sample = self._validate_sample(rvs)
        return float(self._spacing_statistic(self._cdf(sample)))


_FIXED = (
    HypothesisSupport(Distribution.DEFAULT, frozenset({Distribution.LOCATION, Distribution.SCALE})),
)

_EDF_STATISTICS = {
    "KolmogorovSmirnovLaplaceGofStatistic": EDFSpec(
        edf.AbstractKSStatistic, supported_hypotheses=_FIXED, legacy_ks_options=True
    ),
    "KolmogorovSmirnovDPlusLaplaceGofStatistic": EDFSpec(
        edf.AbstractKSDPlusStatistic, supported_hypotheses=_FIXED
    ),
    "KolmogorovSmirnovDMinusLaplaceGofStatistic": EDFSpec(
        edf.AbstractKSDMinusStatistic, supported_hypotheses=_FIXED
    ),
    "LiaoShimokawaLaplaceGofStatistic": EDFSpec(
        edf.AbstractLiaoShimokawaStatistic, supported_hypotheses=_FIXED
    ),
    "KuiperScaledLaplaceGofStatistic": EDFSpec(
        edf.AbstractKuiperScaledStatistic, supported_hypotheses=_FIXED
    ),
    "KuiperCorrectedLaplaceGofStatistic": EDFSpec(
        edf.AbstractKuiperCorrectedStatistic, supported_hypotheses=_FIXED
    ),
    "ZhangKLaplaceGofStatistic": EDFSpec(edf.AbstractZhangKStatistic, supported_hypotheses=_FIXED),
    "ZhangALaplaceGofStatistic": EDFSpec(edf.AbstractZhangAStatistic, supported_hypotheses=_FIXED),
    "ZhangCLaplaceGofStatistic": EDFSpec(edf.AbstractZhangCStatistic, supported_hypotheses=_FIXED),
    "CramerVonMisesLaplaceGofStatistic": EDFSpec(
        edf.AbstractCramerVonMisesStatistic, supported_hypotheses=_FIXED
    ),
    "AndersonDarlingLaplaceGofStatistic": EDFSpec(
        edf.AbstractAndersonDarlingStatistic, supported_hypotheses=_FIXED
    ),
    "KuiperLaplaceGofStatistic": EDFSpec(
        edf.AbstractKuiperStatistic, short_code="KUI", supported_hypotheses=_FIXED
    ),
    "WatsonLaplaceGofStatistic": EDFSpec(
        edf.AbstractWatsonStatistic, short_code="WAT", supported_hypotheses=_FIXED
    ),
}

_FREQUENCY_STATISTICS = {
    "Chi2PearsonLaplaceGofStatistic": FrequencySpec(
        frequency.AbstractPearsonChi2Statistic, supported_hypotheses=_FIXED
    ),
    "FreemanTukeyLaplaceGofStatistic": FrequencySpec(
        frequency.AbstractFreemanTukeyStatistic, supported_hypotheses=_FIXED
    ),
    "ModifiedLogLikelihoodLaplaceGofStatistic": FrequencySpec(
        frequency.AbstractModifiedLogLikelihoodStatistic, supported_hypotheses=_FIXED
    ),
    "NeymanChi2LaplaceGofStatistic": FrequencySpec(
        frequency.AbstractNeymanChi2Statistic, supported_hypotheses=_FIXED
    ),
    "CressieReadLaplaceGofStatistic": FrequencySpec(
        frequency.AbstractCressieReadStatistic, supported_hypotheses=_FIXED
    ),
}

__all__ = [
    "AbstractLaplaceGofStatistic",
    "GreenwoodLaplaceGofStatistic",
]
__all__.extend(
    export_edf_statistics(
        globals(),
        AbstractLaplaceGofStatistic,
        _EDF_STATISTICS,
        cdf=AbstractLaplaceGofStatistic._cdf,
        log_probabilities=AbstractLaplaceGofStatistic._logs,
        sample_validator=AbstractLaplaceGofStatistic._validate_sample,
    )
)

__all__.extend(
    export_frequency_statistics(globals(), AbstractLaplaceGofStatistic, _FREQUENCY_STATISTICS)
)

globals()["KolmogorovSmirnovLaplaceGofStatistic"].__doc__ = """
Kolmogorov--Smirnov statistic D for a fully specified Laplace CDF.

Parameters
----------
parameters : ParameterValues
    Values with t, s fixed; omitted parameters are unknown.
alternative_type : AlternativeType, default TWO_TAILED
    CDF deviation direction: TWO_TAILED gives D, RIGHT D+, LEFT D-.
    This setting does not change the right tail used for rejection.
mode : {'auto', 'exact', 'asymp', 'approx'}, default 'auto'
    Compatibility setting; has no effect on the statistic. No p-value
    is computed. Internally 'auto' is stored as 'exact'.

Methods
-------
execute_statistic(rvs, **kwargs)
    Return one scalar statistic without changing the sample or model.
hypothesis()
    Return the fixed parameters ``{"t": t, "s": s}``.
alternative()
    Return the right-tailed rejection rule.

Notes
-----
H0: observations are independent Laplace(t, s) on the real line.
Let u_i = F_(t,s)(x_(i)) for sorted observations and sample size n.
D+ = max(i/n - u_i), D- = max(u_i - (i-1)/n), and
D = max(D+, D-), for i = 1, ..., n. RIGHT selects D+; LEFT
selects D-; TWO_TAILED selects D. All reject for large values.

Large values reject H0. Under H0 the probability integral transform
gives independent uniforms: the null law depends on n, not t or s.
Calibration must use this unmodified statistic and fixed parameters.
Parameters fitted to the tested sample require a different calibration.
Reference [1]_ supplies the general statistic; applying it through
the known Laplace CDF does not define a fitted Laplace-family test.
CDF values may round to 0 or 1 in extreme tails; small spacings can
lose precision. Observations and probabilities are never clipped.

Stored calibration cannot distinguish KS directions and is rejected
for LEFT/RIGHT. Use Monte Carlo calibration for those directions.

References
----------
.. [1] M. A. Stephens (1974), "EDF Statistics for Goodness of Fit and
    Some Comparisons", JASA 69, 730-737, Section 2 (definitions and
    computation), Case 0 (fully specified continuous distribution).
    https://doi.org/10.1080/01621459.1974.10480196
    Full text: https://www.math.utah.edu/~morris/Courses/6010/p1/writeup/ks.pdf

Examples
--------
>>> import numpy as np
>>> from pysatl_criterion.distribution import LaplaceDistributionDescriptor as Distribution
>>> parameters = Distribution.DEFAULT.parse({'t': 0.0, 's': 1.0})
>>> statistic = KolmogorovSmirnovLaplaceGofStatistic(parameters)
>>> value = statistic.execute_statistic([-1.0, 0.0, 0.5, 2.0])
>>> bool(np.isfinite(value))
True
"""

globals()["CramerVonMisesLaplaceGofStatistic"].__doc__ = """
Cramer--von Mises statistic W² for a fully specified Laplace CDF.

Parameters
----------
parameters : ParameterValues
    Values with t, s fixed; omitted parameters are unknown.

Methods
-------
execute_statistic(rvs, **kwargs)
    Return one scalar statistic without changing the sample or model.
hypothesis()
    Return the fixed parameters ``{"t": t, "s": s}``.
alternative()
    Return the right-tailed rejection rule.

Notes
-----
H0: observations are independent Laplace(t, s) on the real line.
Let u_i = F_(t,s)(x_(i)) for sorted observations and sample size n.
W² = 1/(12*n) + sum((u_i - (2*i-1)/(2*n))**2), i = 1, ..., n.
No finite-sample or fitted-parameter correction is applied.

Large values reject H0. Under H0 the probability integral transform
gives independent uniforms: the null law depends on n, not t or s.
Calibration must use this unmodified statistic and fixed parameters.
Parameters fitted to the tested sample require a different calibration.
Reference [1]_ supplies the general statistic; applying it through
the known Laplace CDF does not define a fitted Laplace-family test.
CDF values may round to 0 or 1 in extreme tails; small spacings can
lose precision. Observations and probabilities are never clipped.

References
----------
.. [1] M. A. Stephens (1974), "EDF Statistics for Goodness of Fit and
    Some Comparisons", JASA 69, 730-737, Section 2 (definitions and
    computation), Case 0 (fully specified continuous distribution).
    https://doi.org/10.1080/01621459.1974.10480196
    Full text: https://www.math.utah.edu/~morris/Courses/6010/p1/writeup/ks.pdf

Examples
--------
>>> import numpy as np
>>> from pysatl_criterion.distribution import LaplaceDistributionDescriptor as Distribution
>>> parameters = Distribution.DEFAULT.parse({'t': 0.0, 's': 1.0})
>>> statistic = CramerVonMisesLaplaceGofStatistic(parameters)
>>> value = statistic.execute_statistic([-1.0, 0.0, 0.5, 2.0])
>>> bool(np.isfinite(value))
True"""

globals()["AndersonDarlingLaplaceGofStatistic"].__doc__ = """
Anderson--Darling statistic A² for a fully specified Laplace CDF.

Parameters
----------
parameters : ParameterValues
    Values with t, s fixed; omitted parameters are unknown.

Methods
-------
execute_statistic(rvs, **kwargs)
    Return one scalar statistic without changing the sample or model.
hypothesis()
    Return the fixed parameters ``{"t": t, "s": s}``.
alternative()
    Return the right-tailed rejection rule.

Notes
-----
H0: observations are independent Laplace(t, s) on the real line.
Let u_i = F_(t,s)(x_(i)) for sorted observations and sample size n.
A² = -n - sum((2*i-1)/n * (log(u_i) + log(1-u_(n+1-i)))),
i = 1, ..., n. Log probabilities are evaluated analytically:
log(F(x)) = z-log(2) for z < 0 and log1p(-exp(-z)/2) otherwise,
where z = (x-t)/s; log(S(x)) follows by reflection.
This avoids exponential underflow in the logarithmic tails.
No finite-sample or fitted-parameter correction is applied.

Large values reject H0. Under H0 the probability integral transform
gives independent uniforms: the null law depends on n, not t or s.
Calibration must use this unmodified statistic and fixed parameters.
Parameters fitted to the tested sample require a different calibration.
Reference [1]_ supplies the general statistic; applying it through
the known Laplace CDF does not define a fitted Laplace-family test.
CDF values may round to 0 or 1 in extreme tails; small spacings can
lose precision. AD uses log probabilities directly. Values beyond
float64 range can still overflow; observations are never clipped.

References
----------
.. [1] M. A. Stephens (1974), "EDF Statistics for Goodness of Fit and
    Some Comparisons", JASA 69, 730-737, Section 2 (definitions and
    computation), Case 0 (fully specified continuous distribution).
    https://doi.org/10.1080/01621459.1974.10480196
    Full text: https://www.math.utah.edu/~morris/Courses/6010/p1/writeup/ks.pdf

Examples
--------
>>> import numpy as np
>>> from pysatl_criterion.distribution import LaplaceDistributionDescriptor as Distribution
>>> parameters = Distribution.DEFAULT.parse({'t': 0.0, 's': 1.0})
>>> statistic = AndersonDarlingLaplaceGofStatistic(parameters)
>>> value = statistic.execute_statistic([-1.0, 0.0, 0.5, 2.0])
>>> bool(np.isfinite(value))
True"""

globals()["KuiperLaplaceGofStatistic"].__doc__ = """
Kuiper statistic V for a fully specified Laplace CDF.

Parameters
----------
parameters : ParameterValues
    Values with t, s fixed; omitted parameters are unknown.

Methods
-------
execute_statistic(rvs, **kwargs)
    Return one scalar statistic without changing the sample or model.
hypothesis()
    Return the fixed parameters ``{"t": t, "s": s}``.
alternative()
    Return the right-tailed rejection rule.

Notes
-----
H0: observations are independent Laplace(t, s) on the real line.
Let u_i = F_(t,s)(x_(i)) for sorted observations and sample size n.
V = max(i/n - u_i) + max(u_i - (i-1)/n), i = 1, ..., n.
The statistic is unscaled: no sqrt(n) or finite-sample correction.

Large values reject H0. Under H0 the probability integral transform
gives independent uniforms: the null law depends on n, not t or s.
Calibration must use this unmodified statistic and fixed parameters.
Parameters fitted to the tested sample require a different calibration.
Reference [1]_ supplies the general statistic; applying it through
the known Laplace CDF does not define a fitted Laplace-family test.
CDF values may round to 0 or 1 in extreme tails; small spacings can
lose precision. Observations and probabilities are never clipped.

References
----------
.. [1] M. A. Stephens (1974), "EDF Statistics for Goodness of Fit and
    Some Comparisons", JASA 69, 730-737, Section 2 (definitions and
    computation), Case 0 (fully specified continuous distribution).
    https://doi.org/10.1080/01621459.1974.10480196
    Full text: https://www.math.utah.edu/~morris/Courses/6010/p1/writeup/ks.pdf

Examples
--------
>>> import numpy as np
>>> from pysatl_criterion.distribution import LaplaceDistributionDescriptor as Distribution
>>> parameters = Distribution.DEFAULT.parse({'t': 0.0, 's': 1.0})
>>> statistic = KuiperLaplaceGofStatistic(parameters)
>>> value = statistic.execute_statistic([-1.0, 0.0, 0.5, 2.0])
>>> bool(np.isfinite(value))
True"""

globals()["WatsonLaplaceGofStatistic"].__doc__ = """
Watson statistic U² for a fully specified Laplace CDF.

Parameters
----------
parameters : ParameterValues
    Values with t, s fixed; omitted parameters are unknown.

Methods
-------
execute_statistic(rvs, **kwargs)
    Return one scalar statistic without changing the sample or model.
hypothesis()
    Return the fixed parameters ``{"t": t, "s": s}``.
alternative()
    Return the right-tailed rejection rule.

Notes
-----
H0: observations are independent Laplace(t, s) on the real line.
Let u_i = F_(t,s)(x_(i)) for sorted observations and sample size n.
U² = W² - n*(mean(u)-1/2)**2, where
W² = 1/(12*n) + sum((u_i - (2*i-1)/(2*n))**2).
Computation uses the equivalent centered sum of squares of
u_i - (2*i-1)/(2*n) to avoid subtracting two large totals.
No finite-sample correction is applied. For n=1, U²=1/12
for every observation, so the statistic cannot discriminate.

Large values reject H0. Under H0 the probability integral transform
gives independent uniforms: the null law depends on n, not t or s.
Calibration must use this unmodified statistic and fixed parameters.
Parameters fitted to the tested sample require a different calibration.
Reference [1]_ supplies the general statistic; applying it through
the known Laplace CDF does not define a fitted Laplace-family test.
CDF values may round to 0 or 1 in extreme tails; small spacings can
lose precision. Observations and probabilities are never clipped.

References
----------
.. [1] M. A. Stephens (1974), "EDF Statistics for Goodness of Fit and
    Some Comparisons", JASA 69, 730-737, Section 2 (definitions and
    computation), Case 0 (fully specified continuous distribution).
    https://doi.org/10.1080/01621459.1974.10480196
    Full text: https://www.math.utah.edu/~morris/Courses/6010/p1/writeup/ks.pdf

Examples
--------
>>> import numpy as np
>>> from pysatl_criterion.distribution import LaplaceDistributionDescriptor as Distribution
>>> parameters = Distribution.DEFAULT.parse({'t': 0.0, 's': 1.0})
>>> statistic = WatsonLaplaceGofStatistic(parameters)
>>> value = statistic.execute_statistic([-1.0, 0.0, 0.5, 2.0])
>>> bool(np.isfinite(value))
True"""


class AbstractFittedLaplaceGofStatistic(AbstractLaplaceGofStatistic, ABC):
    """Shared validation for the composite Laplace location-scale family.

    Parameters
    ----------
    parameters : ParameterValues
        Unknown location and scale, represented by
        ``LaplaceDistributionDescriptor.DEFAULT.parse({})``.
        Fixed location or scale parameters are not supported.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute a scalar statistic from a finite, nonconstant sample.
    hypothesis()
        Return the composite null with both parameters unknown.
    alternative()
        Return the rejection-tail specification.
    configuration()
        Return versioned algorithm settings for reproducible calibration.

    Raises
    ------
    ValueError
        If the hypothesis, algorithm settings or sample are unsupported.

    Notes
    -----
    Pass ``Distribution.DEFAULT.parse({})``. Each replicate is fitted afresh.
    Constant samples have no positive fitted scale and are rejected.

    Samples must be one-dimensional, finite, real and nonconstant, with
    n >= 3. Each execution refits the sample without changing the input
    or stored hypothesis. Simulation calibration must repeat fitting with
    the same sample size and algorithm settings. No p-value is computed.

    References
    ----------
    .. [1] Desgagne, A., Lafaye de Micheaux, P. and Ouimet, F. (2022).
        A comprehensive empirical power comparison of univariate
        goodness-of-fit tests for the Laplace distribution, Section 3.
        https://arxiv.org/abs/2007.06154
    """

    @staticmethod
    def _standard_calibration_parameters():
        """Return fixed parameters for simulation under the composite null.

        Returns
        -------
        ParameterValues
            Standard Laplace location zero and scale one. Each simulated sample
            must still be fitted by the criterion being calibrated.
        """
        return Distribution.DEFAULT.parse({"t": 0.0, "s": 1.0})

    @classmethod
    def _fit(cls, rvs, minimum=3):
        """Validate a sample and standardize it by the Laplace MLE.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite, real, nonconstant observations.
        minimum : int, default 3
            Minimum accepted number of observations.

        Returns
        -------
        ndarray of float, shape (n,)
            Sorted observations centered at the median and divided by the mean
            absolute deviation. Initial scaling protects against overflow.

        Raises
        ------
        ValueError
            If the supplied sample or setting violates the stated constraints.
        """
        sample = cls._validate_sample(rvs)
        if len(sample) < minimum:
            raise ValueError(f"At least {minimum} observations are required.")
        if sample[0] == sample[-1]:
            raise ValueError("A constant sample has zero fitted scale.")
        return cls._fit_sorted(sample)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _fit_sorted(sample):
        """Standardize a validated sorted nonconstant sample in linear time.

        Parameters
        ----------
        sample : ndarray of float, shape (n,)
            Sorted finite observations with n >= 3 and positive range.

        Returns
        -------
        ndarray of float
            Median-centered observations with unit mean absolute deviation.
            The input is not modified; scaling before centering avoids overflow.

        Raises
        ------
        ValueError
            If the fitted scale is not representable.
        """
        n = len(sample)
        magnitude = max(abs(sample[0]), abs(sample[-1]))
        result = sample / magnitude
        middle = n // 2
        median = result[middle]
        if n % 2 == 0:
            median = (result[middle - 1] + median) / 2
        scale = 0.0
        for i in range(n):
            result[i] -= median
            scale += abs(result[i])
        scale /= n
        if scale <= 0:
            raise ValueError("The fitted scale is not representable.")
        for i in range(n):
            result[i] /= scale
        return result

    def alternative(self):
        """Return the rejection-tail specification.

        Returns
        -------
        Alternative
            Left-tailed for CK, two-sided for signed moment components,
            or right-tailed for omnibus statistics, as defined by this class.
        """
        return RightAlternative()

    @staticmethod
    def _positive(value, name):
        """Convert a positive finite numeric setting to float.

        Parameters
        ----------
        value : scalar
            Candidate numeric setting.
        name : str
            Parameter name included in validation error messages.

        Returns
        -------
        float
            Positive finite setting.

        Raises
        ------
        ValueError
            If the supplied sample or setting violates the stated constraints.
        """
        if isinstance(value, (bool, str)) or not np.isscalar(value):
            raise ValueError(f"{name} must be a positive finite number.")
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number.")
        return value


class AbstractMeintanisLaplaceGofStatistic(AbstractFittedLaplaceGofStatistic, ABC):
    """ECF integral, with MLE or moment fitting and positive weight a.

    Parameters
    ----------
    parameters : ParameterValues
        Unknown location and scale, represented by
        ``LaplaceDistributionDescriptor.DEFAULT.parse({})``.
        Fixed location or scale parameters are not supported.
    a : float or None, default None
        Positive finite weight parameter. None selects 2.0 for family 1 or 0.5 for family 2.
    estimator : {'mle', 'mom'}, default 'mle'
        Standardization by median and mean absolute deviation (MLE), or
        by sample mean and standard deviation divided by sqrt(2) (MOM).

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute a scalar statistic from a finite, nonconstant sample.
    hypothesis()
        Return the composite null with both parameters unknown.
    alternative()
        Return the rejection-tail specification.
    configuration()
        Return versioned algorithm settings for reproducible calibration.

    Raises
    ------
    ValueError
        If the hypothesis, algorithm settings or sample are unsupported.
    ArithmeticError
        If the fallback ECF quadrature fails to converge.

    Notes
    -----
    The weights are exp(-a*abs(t)) for family 1 and exp(-a*t**2)
    for family 2. Analytic pairwise sums take O(n**2) time; adaptive
    quadrature is used when the closed form is numerically unreliable.

    Samples must be one-dimensional, finite, real and nonconstant, with
    n >= 3. Each execution refits the sample without changing the input
    or stored hypothesis. Simulation calibration must repeat fitting with
    the same sample size and algorithm settings. No p-value is computed.

    References
    ----------
    .. [1] Meintanis, S. G. (2004). A class of omnibus tests for the
        Laplace distribution based on the empirical characteristic function.
        Communications in Statistics - Theory and Methods, 33(4), 925-948.
        https://doi.org/10.1081/STA-120028735
    .. [2] Desgagne, A., Lafaye de Micheaux, P. and Ouimet, F. (2022).
        A comprehensive empirical power comparison of univariate
        goodness-of-fit tests for the Laplace distribution, Section 3.
        https://arxiv.org/abs/2007.06154
    """

    def __init__(self, parameters, *, a=None, estimator="mle"):
        """Initialize the criterion and validate its algorithm settings.

        Parameters
        ----------
        parameters : ParameterValues
            Unknown location and scale, represented by
            ``LaplaceDistributionDescriptor.DEFAULT.parse({})``.
            Fixed location or scale parameters are not supported.
        a : float or None, default None
            Positive finite weight parameter. None selects 2.0 for family 1 or 0.5 for family 2.
        estimator : {'mle', 'mom'}, default 'mle'
            Standardization by median and mean absolute deviation (MLE), or
            by sample mean and standard deviation divided by sqrt(2) (MOM).

        Raises
        ------
        TypeError
            If parameters is not a ParameterValues instance.
        ValueError
            If the hypothesis or algorithm settings are unsupported.

        Notes
        -----
        Sample-dependent conditions are checked during execution. See the
        class documentation for the statistic definition and references.
        """
        super().__init__(parameters)
        self.a = self._positive((2.0 if self.family == 1 else 0.5) if a is None else a, "a")
        if estimator not in {"mle", "mom"}:
            raise ValueError("estimator must be 'mle' or 'mom'.")
        self.estimator = estimator

    def configuration(self):
        """Return the settings used to identify a calibration.

        Returns
        -------
        dict
            Schema and statistic versions and all algorithm-specific options.
        """
        return {**super().configuration(), "a": self.a, "estimator": self.estimator}

    @staticmethod
    @jit(nopython=True, cache=True)
    def _ecf_kernel(z, a, family):
        """Compute the analytic Meintanis pairwise sum.

        Parameters
        ----------
        z : ndarray of float, shape (n,)
            Sorted observations standardized by the sample median and mean
            absolute deviation about that median.
        a : float
            Positive weight parameter.
        family : {1, 2}
            Exponential absolute-value weight (1) or Gaussian weight (2).

        Returns
        -------
        float
            ECF discrepancy integral multiplied by n. Cancellation can reduce
            accuracy; the public execution method handles numerical fallback.

        References
        ----------
        .. [1] Meintanis (2004). https://doi.org/10.1081/STA-120028735
        """
        n = len(z)
        total = 0.0
        if family == 1:
            aa = a * a
            total = 2 * n / a
            for x in z:
                u = x * x
                q = aa + u
                total -= 4 * a * (1 / q + 2 * (aa - 3 * u) / q**3)
            for x in z:
                for y in z:
                    u = (x - y) ** 2
                    q = aa + u
                    total += (
                        2
                        * a
                        / n
                        * (
                            1 / q
                            + 4 * (aa - 3 * u) / q**3
                            + 24 * (aa * aa + 5 * u * u - 10 * aa * u) / q**5
                        )
                    )
        else:
            factor = np.sqrt(np.pi / a)
            total = n * factor
            for x in z:
                u = x * x
                total -= 2 * factor * (1 - (u - 2 * a) / (4 * a * a)) * np.exp(-u / (4 * a))
            for x in z:
                for y in z:
                    u = (x - y) ** 2
                    total += (
                        2
                        * factor
                        / n
                        * (
                            0.5
                            - (u - 2 * a) / (4 * a * a)
                            + (u * u + 12 * a * a - 12 * a * u) / (32 * a**4)
                        )
                        * np.exp(-u / (4 * a))
                    )
        return total

    @staticmethod
    @jit(nopython=True, cache=True)
    def _moment_standardize(z):
        """Convert MLE-standardized observations to moment standardization.

        Parameters
        ----------
        z : ndarray of float, shape (n,)
            Nonconstant finite observations.

        Returns
        -------
        ndarray of float
            Mean-centered values divided by sqrt(variance/2), with divisor n.
        """
        mean = np.mean(z)
        result = z - mean
        variance = 0.0
        for value in result:
            variance += value * value
        scale = np.sqrt(variance / (2 * len(z)))
        for i in range(len(z)):
            result[i] /= scale
        return result

    @staticmethod
    @jit(nopython=True, cache=True)
    def _ecf_integrand(u, z, scale, family):
        """Evaluate the rescaled ECF integrand without temporary arrays.

        Parameters
        ----------
        u : float
            Nonnegative integration coordinate.
        z : ndarray of float, shape (n,)
            Standardized observations.
        scale : float
            a for family 1, sqrt(a) for family 2.
        family : {1, 2}
            Exponential or Gaussian weight.

        Returns
        -------
        float
            Squared ECF discrepancy times its weight, before the 2*n/scale
            factor. The cosine difference uses -2*sin(t*z/2)**2 near zero.
        """
        t = u / scale
        real, imag = 0.0, 0.0
        for value in z:
            sine = np.sin(t * value / 2)
            real -= 2 * sine * sine
            imag += np.sin(t * value)
        real = (1 + t * t) * (real / len(z)) + t * t
        imag *= (1 + t * t) / len(z)
        weight = np.exp(-u if family == 1 else -u * u)
        return (real * real + imag * imag) * weight

    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic for a newly fitted Laplace sample.

        Parameters
        ----------
        rvs : array_like of float, shape (n,)
            Finite, real, nonconstant observations. At least three are needed;
            DLO requires four. Entropy windows must satisfy ``2*m < n``.
        **kwargs : dict
            Accepted for interface compatibility; ignored.

        Returns
        -------
        float
            Statistic in the normalization documented by the concrete class.
            Degenerate spacings may produce zero or infinity as documented.

        Raises
        ------
        ValueError
            If the sample or window does not meet the criterion requirements.

        Notes
        -----
        The input and stored hypothesis are not modified. Refer to the
        concrete class for rejection direction, formulas and references.
        """
        z = self._fit(rvs)
        if self.estimator == "mom":
            z = self._moment_standardize(z)
        # Closed forms lose significant digits when all three terms nearly cancel.
        if 0.01 <= self.a <= 10:
            result = self._ecf_kernel(z, self.a, self.family)
            if result > 1e-9:
                return float(result)
        from scipy.integrate import quad

        scale = self.a if self.family == 1 else np.sqrt(self.a)

        result, error = quad(
            self._ecf_integrand,
            0,
            np.inf,
            args=(z, scale, self.family),
            epsabs=1e-12,
            epsrel=1e-9,
            limit=1000,
        )
        if not np.isfinite(result) or error > max(1e-10, 1e-6 * result):
            raise ArithmeticError("ECF quadrature did not converge for this a and sample.")
        return float(2 * len(z) * result / scale)


class AbstractEntropyLaplaceGofStatistic(AbstractFittedLaplaceGofStatistic, ABC):
    """Order-statistic entropy estimates with an explicit integer window.

    Parameters
    ----------
    parameters : ParameterValues
        Unknown location and scale, represented by
        ``LaplaceDistributionDescriptor.DEFAULT.parse({})``.
        Fixed location or scale parameters are not supported.
    m : int, default 1
        Order-statistic window, with ``1 <= m < n/2`` at execution.
        The default is a reproducible setting, not an optimal-window rule.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute a scalar statistic from a finite, nonconstant sample.
    hypothesis()
        Return the composite null with both parameters unknown.
    alternative()
        Return the rejection-tail specification.
    configuration()
        Return versioned algorithm settings for reproducible calibration.

    Raises
    ------
    ValueError
        If the hypothesis, algorithm settings or sample are unsupported.

    Notes
    -----
    Default m=1 is a reproducible algorithm setting, not a power-optimal
    window recommendation. Select m before calibration; 1 <= m < n/2.

    Samples must be one-dimensional, finite, real and nonconstant, with
    n >= 3. Each execution refits the sample without changing the input
    or stored hypothesis. Simulation calibration must repeat fitting with
    the same sample size and algorithm settings. No p-value is computed.

    References
    ----------
    .. [1] Desgagne, A., Lafaye de Micheaux, P. and Ouimet, F. (2022).
        A comprehensive empirical power comparison of univariate
        goodness-of-fit tests for the Laplace distribution, Section 3.
        https://arxiv.org/abs/2007.06154
    """

    def __init__(self, parameters, *, m=1):
        """Initialize the criterion and validate its algorithm settings.

        Parameters
        ----------
        parameters : ParameterValues
            Unknown location and scale, represented by
            ``LaplaceDistributionDescriptor.DEFAULT.parse({})``.
            Fixed location or scale parameters are not supported.
        m : int, default 1
            Order-statistic window, with ``1 <= m < n/2`` at execution.
            The default is a reproducible setting, not an optimal-window rule.

        Raises
        ------
        TypeError
            If parameters is not a ParameterValues instance.
        ValueError
            If the hypothesis or algorithm settings are unsupported.

        Notes
        -----
        Sample-dependent conditions are checked during execution. See the
        class documentation for the statistic definition and references.
        """
        super().__init__(parameters)
        if isinstance(m, bool) or not isinstance(m, (int, np.integer)) or m < 1:
            raise ValueError("m must be a positive integer.")
        self.m = int(m)

    def configuration(self):
        """Return the settings used to identify a calibration.

        Returns
        -------
        dict
            Schema and statistic versions and all algorithm-specific options.
        """
        return {**super().configuration(), "m": self.m}

    def _window_sample(self, rvs):
        """Validate the entropy window and standardize the sample.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite, real, nonconstant observations with ``n > 2*m``.

        Returns
        -------
        ndarray of float, shape (n,)
            Sorted observations standardized by the Laplace MLE.

        Raises
        ------
        ValueError
            If the supplied sample or setting violates the stated constraints.
        """
        z = self._fit(rvs)
        if 2 * self.m >= len(z):
            raise ValueError("The window must satisfy 1 <= m < n/2.")
        return z

    @staticmethod
    @jit(nopython=True, cache=True)
    def _entropy(z, m, kind):  # noqa: C901 - estimator branches share the spacing traversal
        """Compute an order-statistic entropy estimate.

        Parameters
        ----------
        z : ndarray of float, shape (n,)
            Sorted observations standardized by the sample median and mean
            absolute deviation about that median.
        m : int
            Positive spacing window satisfying ``2*m < n``.
        kind : {'v', 'e', 'a', 'z', 'correa', 'van_es'}
            Vasicek, Ebrahimi, Alizadeh Noughabi-Arghami, Zamanzade-Arghami,
            Correa or van Es estimator.

        Returns
        -------
        float
            Entropy of the standardized sample; negative infinity if any
            required observation spacing is zero.

        References
        ----------
        .. [1] Choi and Kim (2006). https://doi.org/10.1080/02331880600822473
            Alizadeh Noughabi and Park (2016).
            https://doi.org/10.1080/00949655.2015.1104685
        """
        n = len(z)
        if kind == "van_es":
            total = 0.0
            for i in range(n - m):
                span = z[i + m] - z[i]
                if span <= 0:
                    return -np.inf
                total += np.log(span) / (n - m)
            for k in range(m, n + 1):
                total += 1.0 / k
            return total
        total = 0.0
        for i in range(n):
            span = z[min(i + m, n - 1)] - z[max(i - m, 0)]
            if span <= 0:
                return -np.inf
            if kind == "correa":
                local = 0.0
                for j in range(i - m, i + m + 1):
                    local += z[min(max(j, 0), n - 1)] / (2 * m + 1)
                numerator, denominator = 0.0, 0.0
                for j in range(i - m, i + m + 1):
                    delta = z[min(max(j, 0), n - 1)] - local
                    numerator += (j - i) * delta / n
                    denominator += delta * delta
                total -= np.log(numerator / denominator) / n
            else:
                c = 2.0
                if kind == "e":
                    c = 1.0 + min(i, m, n - i - 1) / m
                elif kind == "a" and (i < m or i >= n - m):
                    c = 1.0
                elif kind == "z":
                    c = (i + 1) / m if i < m else ((n - i) / m if i >= n - m else 2.0)
                total += np.log(n * span / (c * m)) / n
        return total


class AbstractChoiKimLaplaceGofStatistic(AbstractEntropyLaplaceGofStatistic, ABC):
    """CK=exp(H)/MAD; small values reject the fitted Laplace family.

    Parameters
    ----------
    parameters : ParameterValues
        Unknown location and scale, represented by
        ``LaplaceDistributionDescriptor.DEFAULT.parse({})``.
        Fixed location or scale parameters are not supported.
    m : int, default 1
        Order-statistic window, with ``1 <= m < n/2`` at execution.
        The default is a reproducible setting, not an optimal-window rule.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute a scalar statistic from a finite, nonconstant sample.
    hypothesis()
        Return the composite null with both parameters unknown.
    alternative()
        Return the rejection-tail specification.
    configuration()
        Return versioned algorithm settings for reproducible calibration.

    Raises
    ------
    ValueError
        If the hypothesis, algorithm settings or sample are unsupported.

    Notes
    -----
    The entropy is Vasicek (V), Correa (C), or van Es (E). CK_e does
    not use Ebrahimi entropy. A zero required spacing produces a zero
    statistic. Small values reject the null hypothesis.

    Samples must be one-dimensional, finite, real and nonconstant, with
    n >= 3. Each execution refits the sample without changing the input
    or stored hypothesis. Simulation calibration must repeat fitting with
    the same sample size and algorithm settings. No p-value is computed.

    References
    ----------
    .. [1] Choi, B. and Kim, K. (2006). Testing goodness-of-fit for
        Laplace distribution based on maximum entropy. Statistics, 40(6),
        517-531. https://doi.org/10.1080/02331880600822473
    .. [2] Desgagne, A., Lafaye de Micheaux, P. and Ouimet, F. (2022).
        A comprehensive empirical power comparison of univariate
        goodness-of-fit tests for the Laplace distribution, Section 3.
        https://arxiv.org/abs/2007.06154
    """

    def alternative(self):
        """Return the rejection-tail specification.

        Returns
        -------
        Alternative
            Left-tailed for CK, two-sided for signed moment components,
            or right-tailed for omnibus statistics, as defined by this class.
        """
        return LeftAlternative()

    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic for a newly fitted Laplace sample.

        Parameters
        ----------
        rvs : array_like of float, shape (n,)
            Finite, real, nonconstant observations. At least three are needed;
            DLO requires four. Entropy windows must satisfy ``2*m < n``.
        **kwargs : dict
            Accepted for interface compatibility; ignored.

        Returns
        -------
        float
            Statistic in the normalization documented by the concrete class.
            Degenerate spacings may produce zero or infinity as documented.

        Raises
        ------
        ValueError
            If the sample or window does not meet the criterion requirements.

        Notes
        -----
        The input and stored hypothesis are not modified. Refer to the
        concrete class for rejection direction, formulas and references.
        """
        z = self._window_sample(rvs)
        return float(np.exp(self._entropy(z, self.m, self.kind)))


class AbstractAlizadehNoughabiParkLaplaceGofStatistic(AbstractEntropyLaplaceGofStatistic, ABC):
    """AP=log(2*theta)+1-H with corrected or maximum-likelihood scale.

    Parameters
    ----------
    parameters : ParameterValues
        Unknown location and scale, represented by
        ``LaplaceDistributionDescriptor.DEFAULT.parse({})``.
        Fixed location or scale parameters are not supported.
    m : int, default 1
        Order-statistic window, with ``1 <= m < n/2`` at execution.
        The default is a reproducible setting, not an optimal-window rule.
    scale_estimator : {'corrected', 'mle'}, default 'corrected'
        Use the entropy-specific moment-corrected scale or the Laplace MLE.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute a scalar statistic from a finite, nonconstant sample.
    hypothesis()
        Return the composite null with both parameters unknown.
    alternative()
        Return the rejection-tail specification.
    configuration()
        Return versioned algorithm settings for reproducible calibration.

    Raises
    ------
    ValueError
        If the hypothesis, algorithm settings or sample are unsupported.

    Notes
    -----
    AP_y requires distinct observations. Other AP variants return positive
    infinity for zero required spacings, provided the corrected scale is
    positive. AP_e uses Ebrahimi entropy.

    Samples must be one-dimensional, finite, real and nonconstant, with
    n >= 3. Each execution refits the sample without changing the input
    or stored hypothesis. Simulation calibration must repeat fitting with
    the same sample size and algorithm settings. No p-value is computed.

    References
    ----------
    .. [1] Alizadeh Noughabi, H. and Park, S. (2016). Tests of fit for
        the Laplace distribution based on correcting moments of entropy
        estimators. Journal of Statistical Computation and Simulation,
        86(11), 2165-2181. https://doi.org/10.1080/00949655.2015.1104685
    .. [2] Desgagne, A., Lafaye de Micheaux, P. and Ouimet, F. (2022).
        A comprehensive empirical power comparison of univariate
        goodness-of-fit tests for the Laplace distribution, Section 3.
        https://arxiv.org/abs/2007.06154
    """

    def __init__(self, parameters, *, m=1, scale_estimator="corrected"):
        """Initialize the criterion and validate its algorithm settings.

        Parameters
        ----------
        parameters : ParameterValues
            Unknown location and scale, represented by
            ``LaplaceDistributionDescriptor.DEFAULT.parse({})``.
            Fixed location or scale parameters are not supported.
        m : int, default 1
            Order-statistic window, with ``1 <= m < n/2`` at execution.
            The default is a reproducible setting, not an optimal-window rule.
        scale_estimator : {'corrected', 'mle'}, default 'corrected'
            Use the entropy-specific moment-corrected scale or the Laplace MLE.

        Raises
        ------
        TypeError
            If parameters is not a ParameterValues instance.
        ValueError
            If the hypothesis or algorithm settings are unsupported.

        Notes
        -----
        Sample-dependent conditions are checked during execution. See the
        class documentation for the statistic definition and references.
        """
        super().__init__(parameters, m=m)
        if scale_estimator not in {"corrected", "mle"}:
            raise ValueError("scale_estimator must be 'corrected' or 'mle'.")
        self.scale_estimator = scale_estimator

    def configuration(self):
        """Return the settings used to identify a calibration.

        Returns
        -------
        dict
            Schema and statistic versions and all algorithm-specific options.
        """
        return {**super().configuration(), "scale_estimator": self.scale_estimator}

    @staticmethod
    @jit(nopython=True, cache=True)
    def _y_entropy(z, m):
        """Compute Yousefzadeh-Arghami entropy and total CDF spacing.

        Parameters
        ----------
        z : ndarray of float, shape (n,)
            Sorted observations standardized by the sample median and mean
            absolute deviation about that median.
        m : int
            Positive spacing window satisfying ``2*m < n``.

        Returns
        -------
        h : float
            Weighted entropy estimate. Observations must be distinct.
        c : float
            Sum of estimated CDF spacings used to construct corrected knots.

        References
        ----------
        .. [1] Alizadeh Noughabi and Park (2016).
            https://doi.org/10.1080/00949655.2015.1104685
        """
        n = len(z)
        f = np.empty(n)
        b = (n - 1) / (n * (n + 1))
        f[0] = b * (n / (n - 1) + n / (2 * n - 1))
        f[-1] = b * (((n - 1) ** 2 + 1) / (n - 1) + 1 + (n - 1) / (2 * n - 1))
        for i in range(1, n - 1):
            f[i] = b * (
                ((i + 1) * (n - 1) + 1) / (n - 1) + (z[i] - z[i - 1]) / (z[i + 1] - z[i - 1])
            )
        mass = np.empty(n)
        for i in range(n):
            mass[i] = f[min(i + m, n - 1)] - f[max(i - m, 0)]
        c = np.sum(mass)
        h = 0.0
        for i in range(n):
            span = z[min(i + m, n - 1)] - z[max(i - m, 0)]
            h += mass[i] / c * np.log(span / mass[i])
        return h, c

    @staticmethod
    @jit(nopython=True, cache=True)
    def _corrected_scale(z, m, kind, c):
        """Integrate absolute deviation for entropy-specific quantile knots.

        Parameters
        ----------
        z : ndarray of float, shape (n,)
            Sorted observations standardized by the sample median and mean
            absolute deviation about that median.
        m : int
            Positive spacing window satisfying ``2*m < n``.
        kind : {'v', 'e', 'y', 'a', 'z'}
            Entropy-specific knot construction.
        c : float
            Total estimated CDF spacing for kind 'y'; unused otherwise.

        Returns
        -------
        float
            Mean absolute deviation of the piecewise linear quantile function
            about its median, including the central interval for odd n.

        References
        ----------
        .. [1] Alizadeh Noughabi and Park (2016).
            https://doi.org/10.1080/00949655.2015.1104685
        """
        n = len(z)
        knots = np.empty(n + 1)
        # Start at i=0, including m replicated left endpoints, then slide.
        total = 0.0
        for k in range(-m, m):
            total += z[max(k, 0)] / (2 * m)
        knots[0] = total
        for i in range(1, n + 1):
            total += (z[min(i + m - 1, n - 1)] - z[max(i - m - 1, 0)]) / (2 * m)
            knots[i] = total
        if kind == "y":
            knots *= 2 * m / c
        elif kind != "v":
            left, right = knots[m], knots[n - m]
            for i in range(m - 1, -1, -1):
                k = i + 1
                divisor = m + k - 1 if kind == "e" else (m if kind == "a" else k)
                left -= (z[m + k - 1] - z[0]) / divisor
                knots[i] = left
            for i in range(n - m + 1, n + 1):
                k = i + 1
                divisor = n + m - k + 1 if kind == "e" else (m if kind == "a" else n - k + 2)
                right += (z[-1] - z[k - m - 2]) / divisor
                knots[i] = right
        # Integral of |Q(u)-Q(1/2)| for the piecewise linear quantile.
        total = 0.0
        for i in range(n):
            if 2 * i + 1 < n:
                total -= (knots[i] + knots[i + 1]) / 2
            elif 2 * i + 1 > n:
                total += (knots[i] + knots[i + 1]) / 2
            else:
                total += (knots[i + 1] - knots[i]) / 4
        return total / n

    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic for a newly fitted Laplace sample.

        Parameters
        ----------
        rvs : array_like of float, shape (n,)
            Finite, real, nonconstant observations. At least three are needed;
            DLO requires four. Entropy windows must satisfy ``2*m < n``.
        **kwargs : dict
            Accepted for interface compatibility; ignored.

        Returns
        -------
        float
            Statistic in the normalization documented by the concrete class.
            Degenerate spacings may produce zero or infinity as documented.

        Raises
        ------
        ValueError
            If the sample or window does not meet the criterion requirements.

        Notes
        -----
        The input and stored hypothesis are not modified. Refer to the
        concrete class for rejection direction, formulas and references.
        """
        z = self._window_sample(rvs)
        c = 1.0
        if self.kind == "y":
            if np.any(np.diff(z) <= 0):
                raise ValueError("AP_y requires distinct observations.")
            h, c = self._y_entropy(z, self.m)
        else:
            h = self._entropy(z, self.m, self.kind)
        theta = (
            1.0 if self.scale_estimator == "mle" else self._corrected_scale(z, self.m, self.kind, c)
        )
        if theta <= 0:
            raise ValueError("The corrected scale must be positive.")
        return float(np.log(2 * theta) + 1 - h)


class AbstractMomentLaplaceGofStatistic(AbstractFittedLaplaceGofStatistic, ABC):
    """Moment and finite-sample corrected score kernels.

    Parameters
    ----------
    parameters : ParameterValues
        Unknown location and scale, represented by
        ``LaplaceDistributionDescriptor.DEFAULT.parse({})``.
        Fixed location or scale parameters are not supported.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute a scalar statistic from a finite, nonconstant sample.
    hypothesis()
        Return the composite null with both parameters unknown.
    alternative()
        Return the rejection-tail specification.
    configuration()
        Return versioned algorithm settings for reproducible calibration.

    Raises
    ------
    ValueError
        If the hypothesis, algorithm settings or sample are unsupported.

    Notes
    -----
    Samples must be one-dimensional, finite, real and nonconstant, with
    n >= 3. Each execution refits the sample without changing the input
    or stored hypothesis. Simulation calibration must repeat fitting with
    the same sample size and algorithm settings. No p-value is computed.

    References
    ----------
    .. [1] Desgagne, A., Lafaye de Micheaux, P. and Ouimet, F. (2022).
        A comprehensive empirical power comparison of univariate
        goodness-of-fit tests for the Laplace distribution, Section 3.
        https://arxiv.org/abs/2007.06154
    """

    @staticmethod
    @jit(nopython=True, cache=True)
    def _moments(z, kind):
        """Compute a moment-based Laplace statistic.

        Parameters
        ----------
        z : ndarray of float, shape (n,)
            Sorted observations standardized by the sample median and mean
            absolute deviation about that median.
        kind : {'gel', 'gv', 'v3', 'v4'}
            Gel, Gonzalez-Estrada-Villasenor, smooth skewness or smooth kurtosis.

        Returns
        -------
        float
            Statistic using divisor-n central moments. Smooth components are
            signed and scaled to unit asymptotic variance.
        """
        n = len(z)
        mean = np.mean(z)
        second, third, fourth, absolute = 0.0, 0.0, 0.0, 0.0
        for value in z:
            y = value - mean
            square = y * y
            second += square
            third += square * y
            fourth += square * square
            absolute += abs(y)
        variance = second / n
        if kind == "gel":
            q3 = third / n / (2 * np.sqrt(2.0))
            q4 = fourth / n / 4
            return n * (q3 * q3 / 60 + (q4 - 6) ** 2 / 1200)
        if kind == "gv":
            return 2 * np.sqrt(n) * (np.sqrt(variance) / (np.sqrt(2.0) * absolute / n) - 1)
        if kind == "v3":
            return np.sqrt(n / 63.0) * (third / n) / variance**1.5
        return np.sqrt(n / 1188.0) * ((fourth / n) / variance**2 - 6)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _dlo(z):
        """Compute finite-sample corrected DLO skewness and net kurtosis.

        Parameters
        ----------
        z : ndarray of float, shape (n,)
            Sorted observations standardized by the sample median and mean
            absolute deviation about that median.

        Returns
        -------
        zs : float
            Corrected first-power skewness score.
        zk : float
            Corrected net-kurtosis score, using the appropriate odd/even-n
            corrections. Requires n >= 4; zero log-moments contribute zero.

        References
        ----------
        .. [1] Desgagne, Lafaye de Micheaux and Ouimet (2023).
            https://doi.org/10.1080/02331888.2022.2144859
        """
        n = len(z)
        skew = np.mean(z)
        kurt = 0.0
        for x in z:
            if x != 0:
                kurt += abs(x) * np.log(abs(x)) / n
        net = max(0.0, kurt - skew * skew / 2)
        c = 1.0 - 0.5772156649015328606
        if n % 2 == 0:
            zs = np.sqrt(n) * skew / np.sqrt(1 - 1.856 / n**1.06)
            bias = 1 - 0.422 / n**1.01
            correction = 1 - 1.950 / n**0.92 + 39.349 / n**2.3
        else:
            zs = np.sqrt(n) * skew / np.sqrt(1 - 0.281 / n**1.03)
            bias = 1 - 0.198 / n**0.86
            correction = 1 - 3.827 / n**1.04
        zk = (
            np.sqrt(n)
            * (net**0.25 - c**0.25 * bias)
            / np.sqrt(c ** (-1.5) * (np.pi**2 / 3 - 3) * correction / 16)
        )
        return zs, zk

    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic for a newly fitted Laplace sample.

        Parameters
        ----------
        rvs : array_like of float, shape (n,)
            Finite, real, nonconstant observations. At least three are needed;
            DLO requires four. Entropy windows must satisfy ``2*m < n``.
        **kwargs : dict
            Accepted for interface compatibility; ignored.

        Returns
        -------
        float
            Statistic in the normalization documented by the concrete class.
            Degenerate spacings may produce zero or infinity as documented.

        Raises
        ------
        ValueError
            If the sample or window does not meet the criterion requirements.

        Notes
        -----
        The input and stored hypothesis are not modified. Refer to the
        concrete class for rejection direction, formulas and references.
        """
        z = self._fit(rvs, minimum=4 if self.kind.startswith("dlo") else 3)
        if self.kind.startswith("dlo"):
            zs, zk = self._dlo(z)
            return float(zs * zs + zk * zk if self.kind == "dlo_x" else zk)
        return float(self._moments(z, self.kind))


class AlizadehNoughabiLikelihoodLaplaceGofStatistic(AbstractFittedLaplaceGofStatistic):
    """Density-based empirical likelihood ratio A_rat (Noughabi, 2016).

    Parameters
    ----------
    parameters : ParameterValues
        Unknown location and scale, represented by
        ``LaplaceDistributionDescriptor.DEFAULT.parse({})``.
        Fixed location or scale parameters are not supported.
    delta : float, default 0.5
        Exponent in the strict window bound ``m < min(n**delta, n/2)``;
        must satisfy ``0 < delta < 1``.
    log_scale : bool, default False
        Return log(A_rat) instead of A_rat. Calibrate on the same scale.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute a scalar statistic from a finite, nonconstant sample.
    hypothesis()
        Return the composite null with both parameters unknown.
    alternative()
        Return the rejection-tail specification.
    configuration()
        Return versioned algorithm settings for reproducible calibration.

    Raises
    ------
    ValueError
        If the hypothesis, algorithm settings or sample are unsupported.

    Notes
    -----
    Minimum over integer 1 <= m < min(n**delta, n/2) of the product
    2m/[n*(x_(i+m)-x_(i-m))*f_hat(x_i)]. Endpoint ranks are clamped.
    Products are evaluated in log space. ``log_scale=True`` returns log(A_rat)
    and requires its own calibration. Zero spacings give +infinity.

    Samples must be one-dimensional, finite, real and nonconstant, with
    n >= 3. Each execution refits the sample without changing the input
    or stored hypothesis. Simulation calibration must repeat fitting with
    the same sample size and algorithm settings. No p-value is computed.

    References
    ----------
    .. [1] Alizadeh Noughabi, H. (2016). Empirical Likelihood
        Ratio-Based Goodness-of-Fit Test for the Laplace Distribution.
        Communications in Mathematics and Statistics, 4, 459-471.
        https://doi.org/10.1007/s40304-016-0095-0
    .. [2] Desgagne, A., Lafaye de Micheaux, P. and Ouimet, F. (2022).
        A comprehensive empirical power comparison of univariate
        goodness-of-fit tests for the Laplace distribution, Section 3.
        https://arxiv.org/abs/2007.06154
    """

    @classmethod
    def supported_hypotheses(cls):
        """Return supported composite Laplace hypotheses.

        Returns
        -------
        tuple of HypothesisSupport
            The default parameterization with no fixed parameters.
        """
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @staticmethod
    def short_code():
        """Return the criterion identifier within the Laplace family.

        Returns
        -------
        str
            Short code used to build the full goodness-of-fit identifier.
        """
        return "A_RAT"

    def __init__(self, parameters, *, delta=0.5, log_scale=False):
        """Initialize the criterion and validate its algorithm settings.

        Parameters
        ----------
        parameters : ParameterValues
            Unknown location and scale, represented by
            ``LaplaceDistributionDescriptor.DEFAULT.parse({})``.
            Fixed location or scale parameters are not supported.
        delta : float, default 0.5
            Exponent in the strict window bound ``m < min(n**delta, n/2)``;
            must satisfy ``0 < delta < 1``.
        log_scale : bool, default False
            Return log(A_rat) instead of A_rat. Calibrate on the same scale.

        Raises
        ------
        TypeError
            If parameters is not a ParameterValues instance.
        ValueError
            If the hypothesis or algorithm settings are unsupported.

        Notes
        -----
        Sample-dependent conditions are checked during execution. See the
        class documentation for the statistic definition and references.
        """
        super().__init__(parameters)
        self.delta = self._positive(delta, "delta")
        if self.delta >= 1:
            raise ValueError("delta must satisfy 0 < delta < 1.")
        if not isinstance(log_scale, bool):
            raise TypeError("log_scale must be boolean.")
        self.log_scale = log_scale

    def configuration(self):
        """Return the settings used to identify a calibration.

        Returns
        -------
        dict
            Schema and statistic versions and all algorithm-specific options.
        """
        return {**super().configuration(), "delta": self.delta, "log_scale": self.log_scale}

    @staticmethod
    @jit(nopython=True, cache=True)
    def _log_ratio(z, delta):
        """Minimize the empirical log-likelihood ratio over window sizes.

        Parameters
        ----------
        z : ndarray of float, shape (n,)
            Sorted observations standardized by the sample median and mean
            absolute deviation about that median.
        delta : float
            Exponent in (0, 1) defining ``m < min(n**delta, n/2)``.

        Returns
        -------
        float
            Minimum log-ratio. A candidate with a zero spacing is infinite;
            the result is infinite if every admissible candidate is infinite.

        References
        ----------
        .. [1] Alizadeh Noughabi (2016). https://doi.org/10.1007/s40304-016-0095-0
        """
        n = len(z)
        bound = min(n**delta, n / 2)
        best = np.inf
        for m in range(1, int(np.ceil(bound))):
            total = 0.0
            for i in range(n):
                span = z[min(i + m, n - 1)] - z[max(i - m, 0)]
                if span <= 0:
                    total = np.inf
                    break
                total += np.log(4 * m / n) - np.log(span) + abs(z[i])
            best = min(best, total)
        return best

    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic for a newly fitted Laplace sample.

        Parameters
        ----------
        rvs : array_like of float, shape (n,)
            Finite, real, nonconstant observations. At least three are needed;
            DLO requires four. Entropy windows must satisfy ``2*m < n``.
        **kwargs : dict
            Accepted for interface compatibility; ignored.

        Returns
        -------
        float
            Statistic in the normalization documented by the concrete class.
            Degenerate spacings may produce zero or infinity as documented.

        Raises
        ------
        ValueError
            If the sample or window does not meet the criterion requirements.

        Notes
        -----
        The input and stored hypothesis are not modified. Refer to the
        concrete class for rejection direction, formulas and references.
        """
        value = self._log_ratio(self._fit(rvs), self.delta)
        if self.log_scale:
            return float(value)
        with np.errstate(over="ignore", under="ignore"):
            return float(np.exp(value))


class AlizadehNoughabiJarrahiferizLaplaceGofStatistic(AbstractEntropyLaplaceGofStatistic):
    """Informational energy AJ: mean 2m/[n*(F_hat(x_hi)-F_hat(x_lo))].

    Parameters
    ----------
    parameters : ParameterValues
        Unknown location and scale, represented by
        ``LaplaceDistributionDescriptor.DEFAULT.parse({})``.
        Fixed location or scale parameters are not supported.
    m : int, default 1
        Order-statistic window, with ``1 <= m < n/2`` at execution.
        The default is a reproducible setting, not an optimal-window rule.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute a scalar statistic from a finite, nonconstant sample.
    hypothesis()
        Return the composite null with both parameters unknown.
    alternative()
        Return the rejection-tail specification.
    configuration()
        Return versioned algorithm settings for reproducible calibration.

    Raises
    ------
    ValueError
        If the hypothesis, algorithm settings or sample are unsupported.

    Notes
    -----
    Large values reject. Window m=1 by default; 1 <= m < n/2.
    Zero CDF spacings yield +infinity. Same-tail differences are evaluated
    with expm1 rather than subtracting rounded CDF values.

    Samples must be one-dimensional, finite, real and nonconstant, with
    n >= 3. Each execution refits the sample without changing the input
    or stored hypothesis. Simulation calibration must repeat fitting with
    the same sample size and algorithm settings. No p-value is computed.

    References
    ----------
    .. [1] Alizadeh Noughabi and Jarrahiferiz (2019).
        Informational energy-based goodness-of-fit test for Laplace
        distribution. International Journal of Information and Decision
        Sciences, 11(3), 256-267.
        https://www.inderscience.com/info/inarticle.php?artid=102022
    .. [2] Desgagne, A., Lafaye de Micheaux, P. and Ouimet, F. (2022).
        A comprehensive empirical power comparison of univariate
        goodness-of-fit tests for the Laplace distribution, Section 3.
        https://arxiv.org/abs/2007.06154
    """

    @classmethod
    def supported_hypotheses(cls):
        """Return supported composite Laplace hypotheses.

        Returns
        -------
        tuple of HypothesisSupport
            The default parameterization with no fixed parameters.
        """
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @staticmethod
    def short_code():
        """Return the criterion identifier within the Laplace family.

        Returns
        -------
        str
            Short code used to build the full goodness-of-fit identifier.
        """
        return "AJ"

    @staticmethod
    @jit(nopython=True, cache=True)
    def _energy(z, m):
        """Compute informational energy from fitted CDF spacings.

        Parameters
        ----------
        z : ndarray of float, shape (n,)
            Sorted observations standardized by the sample median and mean
            absolute deviation about that median.
        m : int
            Positive spacing window satisfying ``2*m < n``.

        Returns
        -------
        float
            Mean inverse CDF spacing multiplied by 2*m/n. A zero spacing
            or a spacing that underflows to zero produces positive infinity.

        References
        ----------
        .. [1] Alizadeh Noughabi and Jarrahiferiz (2019).
            https://www.inderscience.com/info/inarticle.php?artid=102022
        """
        n = len(z)
        result = 0.0
        for i in range(n):
            lo, hi = z[max(i - m, 0)], z[min(i + m, n - 1)]
            if hi == lo:
                return np.inf
            if hi <= 0:
                mass = 0.5 * np.exp(hi) * (-np.expm1(lo - hi))
            elif lo >= 0:
                mass = 0.5 * np.exp(-lo) * (-np.expm1(lo - hi))
            else:
                mass = -0.5 * (np.expm1(lo) + np.expm1(-hi))
            if mass == 0:
                return np.inf
            result += (2 * m / n) / mass / n
        return result

    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic for a newly fitted Laplace sample.

        Parameters
        ----------
        rvs : array_like of float, shape (n,)
            Finite, real, nonconstant observations. At least three are needed;
            DLO requires four. Entropy windows must satisfy ``2*m < n``.
        **kwargs : dict
            Accepted for interface compatibility; ignored.

        Returns
        -------
        float
            Statistic in the normalization documented by the concrete class.
            Degenerate spacings may produce zero or infinity as documented.

        Raises
        ------
        ValueError
            If the sample or window does not meet the criterion requirements.

        Notes
        -----
        The input and stored hypothesis are not modified. Refer to the
        concrete class for rejection direction, formulas and references.
        """
        return float(self._energy(self._window_sample(rvs), self.m))


class Meintanis1LaplaceGofStatistic(AbstractMeintanisLaplaceGofStatistic):
    """T_(n,a)^(1): n integral |(1+t^2) mean exp(it*z)-1|^2 w_a(t) dt.

    Parameters
    ----------
    parameters : ParameterValues
        Unknown location and scale, represented by
        ``LaplaceDistributionDescriptor.DEFAULT.parse({})``.
        Fixed location or scale parameters are not supported.
    a : float or None, default None
        Positive finite weight parameter. None selects 2.0.
    estimator : {'mle', 'mom'}, default 'mle'
        Standardization by median and mean absolute deviation (MLE), or
        by sample mean and standard deviation divided by sqrt(2) (MOM).

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute a scalar statistic from a finite, nonconstant sample.
    hypothesis()
        Return the composite null with both parameters unknown.
    alternative()
        Return the rejection-tail specification.
    configuration()
        Return versioned algorithm settings for reproducible calibration.

    Raises
    ------
    ValueError
        If the hypothesis, algorithm settings or sample are unsupported.
    ArithmeticError
        If the fallback ECF quadrature fails to converge.

    Notes
    -----
    Composite Laplace null: both location and scale are unknown.
    Use Distribution.DEFAULT.parse({}); refit on every simulated sample.
    Rejection direction: right. Constant samples are invalid.

    The weights are exp(-a*abs(t)) for family 1 and exp(-a*t**2)
    for family 2. Analytic pairwise sums take O(n**2) time; adaptive
    quadrature is used when the closed form is numerically unreliable.

    Samples must be one-dimensional, finite, real and nonconstant, with
    n >= 3. Each execution refits the sample without changing the input
    or stored hypothesis. Simulation calibration must repeat fitting with
    the same sample size and algorithm settings. No p-value is computed.

    References
    ----------
    .. [1] Meintanis, S. G. (2004). A class of omnibus tests for the
        Laplace distribution based on the empirical characteristic function.
        Communications in Statistics - Theory and Methods, 33(4), 925-948.
        https://doi.org/10.1081/STA-120028735
    .. [2] Desgagne, A., Lafaye de Micheaux, P. and Ouimet, F. (2022).
        A comprehensive empirical power comparison of univariate
        goodness-of-fit tests for the Laplace distribution, Section 3.
        https://arxiv.org/abs/2007.06154
    """

    family = 1

    @classmethod
    def supported_hypotheses(cls):
        """Return supported composite Laplace hypotheses.

        Returns
        -------
        tuple of HypothesisSupport
            The default parameterization with no fixed parameters.
        """
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @staticmethod
    def short_code():
        """Return the criterion identifier within the Laplace family.

        Returns
        -------
        str
            Short code used to build the full goodness-of-fit identifier.
        """
        return "ME1"


class Meintanis2LaplaceGofStatistic(AbstractMeintanisLaplaceGofStatistic):
    """T_(n,a)^(2): n integral |(1+t^2) mean exp(it*z)-1|^2 w_a(t) dt.

    Parameters
    ----------
    parameters : ParameterValues
        Unknown location and scale, represented by
        ``LaplaceDistributionDescriptor.DEFAULT.parse({})``.
        Fixed location or scale parameters are not supported.
    a : float or None, default None
        Positive finite weight parameter. None selects 0.5.
    estimator : {'mle', 'mom'}, default 'mle'
        Standardization by median and mean absolute deviation (MLE), or
        by sample mean and standard deviation divided by sqrt(2) (MOM).

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute a scalar statistic from a finite, nonconstant sample.
    hypothesis()
        Return the composite null with both parameters unknown.
    alternative()
        Return the rejection-tail specification.
    configuration()
        Return versioned algorithm settings for reproducible calibration.

    Raises
    ------
    ValueError
        If the hypothesis, algorithm settings or sample are unsupported.
    ArithmeticError
        If the fallback ECF quadrature fails to converge.

    Notes
    -----
    Composite Laplace null: both location and scale are unknown.
    Use Distribution.DEFAULT.parse({}); refit on every simulated sample.
    Rejection direction: right. Constant samples are invalid.

    The weights are exp(-a*abs(t)) for family 1 and exp(-a*t**2)
    for family 2. Analytic pairwise sums take O(n**2) time; adaptive
    quadrature is used when the closed form is numerically unreliable.

    Samples must be one-dimensional, finite, real and nonconstant, with
    n >= 3. Each execution refits the sample without changing the input
    or stored hypothesis. Simulation calibration must repeat fitting with
    the same sample size and algorithm settings. No p-value is computed.

    References
    ----------
    .. [1] Meintanis, S. G. (2004). A class of omnibus tests for the
        Laplace distribution based on the empirical characteristic function.
        Communications in Statistics - Theory and Methods, 33(4), 925-948.
        https://doi.org/10.1081/STA-120028735
    .. [2] Desgagne, A., Lafaye de Micheaux, P. and Ouimet, F. (2022).
        A comprehensive empirical power comparison of univariate
        goodness-of-fit tests for the Laplace distribution, Section 3.
        https://arxiv.org/abs/2007.06154
    """

    family = 2

    @classmethod
    def supported_hypotheses(cls):
        """Return supported composite Laplace hypotheses.

        Returns
        -------
        tuple of HypothesisSupport
            The default parameterization with no fixed parameters.
        """
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @staticmethod
    def short_code():
        """Return the criterion identifier within the Laplace family.

        Returns
        -------
        str
            Short code used to build the full goodness-of-fit identifier.
        """
        return "ME2"


class ChoiKimVLaplaceGofStatistic(AbstractChoiKimLaplaceGofStatistic):
    """CK_v = exp(H_v) / MAD; m=1 by default, 1 <= m < n/2.

    Parameters
    ----------
    parameters : ParameterValues
        Unknown location and scale, represented by
        ``LaplaceDistributionDescriptor.DEFAULT.parse({})``.
        Fixed location or scale parameters are not supported.
    m : int, default 1
        Order-statistic window, with ``1 <= m < n/2`` at execution.
        The default is a reproducible setting, not an optimal-window rule.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute a scalar statistic from a finite, nonconstant sample.
    hypothesis()
        Return the composite null with both parameters unknown.
    alternative()
        Return the rejection-tail specification.
    configuration()
        Return versioned algorithm settings for reproducible calibration.

    Raises
    ------
    ValueError
        If the hypothesis, algorithm settings or sample are unsupported.

    Notes
    -----
    Composite Laplace null: both location and scale are unknown.
    Use Distribution.DEFAULT.parse({}); refit on every simulated sample.
    Rejection direction: left. Constant samples are invalid.

    The entropy is Vasicek (V), Correa (C), or van Es (E). CK_e does
    not use Ebrahimi entropy. A zero required spacing produces a zero
    statistic. Small values reject the null hypothesis.

    Samples must be one-dimensional, finite, real and nonconstant, with
    n >= 3. Each execution refits the sample without changing the input
    or stored hypothesis. Simulation calibration must repeat fitting with
    the same sample size and algorithm settings. No p-value is computed.

    References
    ----------
    .. [1] Choi, B. and Kim, K. (2006). Testing goodness-of-fit for
        Laplace distribution based on maximum entropy. Statistics, 40(6),
        517-531. https://doi.org/10.1080/02331880600822473
    .. [2] Desgagne, A., Lafaye de Micheaux, P. and Ouimet, F. (2022).
        A comprehensive empirical power comparison of univariate
        goodness-of-fit tests for the Laplace distribution, Section 3.
        https://arxiv.org/abs/2007.06154
    """

    kind = "v"

    @classmethod
    def supported_hypotheses(cls):
        """Return supported composite Laplace hypotheses.

        Returns
        -------
        tuple of HypothesisSupport
            The default parameterization with no fixed parameters.
        """
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @staticmethod
    def short_code():
        """Return the criterion identifier within the Laplace family.

        Returns
        -------
        str
            Short code used to build the full goodness-of-fit identifier.
        """
        return "CK_V"


class ChoiKimCLaplaceGofStatistic(AbstractChoiKimLaplaceGofStatistic):
    """CK_c = exp(H_correa) / MAD; m=1 by default, 1 <= m < n/2.

    Parameters
    ----------
    parameters : ParameterValues
        Unknown location and scale, represented by
        ``LaplaceDistributionDescriptor.DEFAULT.parse({})``.
        Fixed location or scale parameters are not supported.
    m : int, default 1
        Order-statistic window, with ``1 <= m < n/2`` at execution.
        The default is a reproducible setting, not an optimal-window rule.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute a scalar statistic from a finite, nonconstant sample.
    hypothesis()
        Return the composite null with both parameters unknown.
    alternative()
        Return the rejection-tail specification.
    configuration()
        Return versioned algorithm settings for reproducible calibration.

    Raises
    ------
    ValueError
        If the hypothesis, algorithm settings or sample are unsupported.

    Notes
    -----
    Composite Laplace null: both location and scale are unknown.
    Use Distribution.DEFAULT.parse({}); refit on every simulated sample.
    Rejection direction: left. Constant samples are invalid.

    The entropy is Vasicek (V), Correa (C), or van Es (E). CK_e does
    not use Ebrahimi entropy. A zero required spacing produces a zero
    statistic. Small values reject the null hypothesis.

    Samples must be one-dimensional, finite, real and nonconstant, with
    n >= 3. Each execution refits the sample without changing the input
    or stored hypothesis. Simulation calibration must repeat fitting with
    the same sample size and algorithm settings. No p-value is computed.

    References
    ----------
    .. [1] Choi, B. and Kim, K. (2006). Testing goodness-of-fit for
        Laplace distribution based on maximum entropy. Statistics, 40(6),
        517-531. https://doi.org/10.1080/02331880600822473
    .. [2] Desgagne, A., Lafaye de Micheaux, P. and Ouimet, F. (2022).
        A comprehensive empirical power comparison of univariate
        goodness-of-fit tests for the Laplace distribution, Section 3.
        https://arxiv.org/abs/2007.06154
    """

    kind = "correa"

    @classmethod
    def supported_hypotheses(cls):
        """Return supported composite Laplace hypotheses.

        Returns
        -------
        tuple of HypothesisSupport
            The default parameterization with no fixed parameters.
        """
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @staticmethod
    def short_code():
        """Return the criterion identifier within the Laplace family.

        Returns
        -------
        str
            Short code used to build the full goodness-of-fit identifier.
        """
        return "CK_C"


class ChoiKimELaplaceGofStatistic(AbstractChoiKimLaplaceGofStatistic):
    """CK_e = exp(H_van_es) / MAD; m=1 by default, 1 <= m < n/2.

    Parameters
    ----------
    parameters : ParameterValues
        Unknown location and scale, represented by
        ``LaplaceDistributionDescriptor.DEFAULT.parse({})``.
        Fixed location or scale parameters are not supported.
    m : int, default 1
        Order-statistic window, with ``1 <= m < n/2`` at execution.
        The default is a reproducible setting, not an optimal-window rule.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute a scalar statistic from a finite, nonconstant sample.
    hypothesis()
        Return the composite null with both parameters unknown.
    alternative()
        Return the rejection-tail specification.
    configuration()
        Return versioned algorithm settings for reproducible calibration.

    Raises
    ------
    ValueError
        If the hypothesis, algorithm settings or sample are unsupported.

    Notes
    -----
    Composite Laplace null: both location and scale are unknown.
    Use Distribution.DEFAULT.parse({}); refit on every simulated sample.
    Rejection direction: left. Constant samples are invalid.

    The entropy is Vasicek (V), Correa (C), or van Es (E). CK_e does
    not use Ebrahimi entropy. A zero required spacing produces a zero
    statistic. Small values reject the null hypothesis.

    Samples must be one-dimensional, finite, real and nonconstant, with
    n >= 3. Each execution refits the sample without changing the input
    or stored hypothesis. Simulation calibration must repeat fitting with
    the same sample size and algorithm settings. No p-value is computed.

    References
    ----------
    .. [1] Choi, B. and Kim, K. (2006). Testing goodness-of-fit for
        Laplace distribution based on maximum entropy. Statistics, 40(6),
        517-531. https://doi.org/10.1080/02331880600822473
    .. [2] Desgagne, A., Lafaye de Micheaux, P. and Ouimet, F. (2022).
        A comprehensive empirical power comparison of univariate
        goodness-of-fit tests for the Laplace distribution, Section 3.
        https://arxiv.org/abs/2007.06154
    """

    kind = "van_es"

    @classmethod
    def supported_hypotheses(cls):
        """Return supported composite Laplace hypotheses.

        Returns
        -------
        tuple of HypothesisSupport
            The default parameterization with no fixed parameters.
        """
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @staticmethod
    def short_code():
        """Return the criterion identifier within the Laplace family.

        Returns
        -------
        str
            Short code used to build the full goodness-of-fit identifier.
        """
        return "CK_E"


class AlizadehNoughabiParkVLaplaceGofStatistic(AbstractAlizadehNoughabiParkLaplaceGofStatistic):
    """AP_v = log(2*theta_v) + 1 - H_v; m=1 by default.

    Parameters
    ----------
    parameters : ParameterValues
        Unknown location and scale, represented by
        ``LaplaceDistributionDescriptor.DEFAULT.parse({})``.
        Fixed location or scale parameters are not supported.
    m : int, default 1
        Order-statistic window, with ``1 <= m < n/2`` at execution.
        The default is a reproducible setting, not an optimal-window rule.
    scale_estimator : {'corrected', 'mle'}, default 'corrected'
        Use the entropy-specific moment-corrected scale or the Laplace MLE.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute a scalar statistic from a finite, nonconstant sample.
    hypothesis()
        Return the composite null with both parameters unknown.
    alternative()
        Return the rejection-tail specification.
    configuration()
        Return versioned algorithm settings for reproducible calibration.

    Raises
    ------
    ValueError
        If the hypothesis, algorithm settings or sample are unsupported.

    Notes
    -----
    Composite Laplace null: both location and scale are unknown.
    Use Distribution.DEFAULT.parse({}); refit on every simulated sample.
    Rejection direction: right. Constant samples are invalid.

    AP_y requires distinct observations. Other AP variants return positive
    infinity for zero required spacings, provided the corrected scale is
    positive. AP_e uses Ebrahimi entropy.

    Samples must be one-dimensional, finite, real and nonconstant, with
    n >= 3. Each execution refits the sample without changing the input
    or stored hypothesis. Simulation calibration must repeat fitting with
    the same sample size and algorithm settings. No p-value is computed.

    References
    ----------
    .. [1] Alizadeh Noughabi, H. and Park, S. (2016). Tests of fit for
        the Laplace distribution based on correcting moments of entropy
        estimators. Journal of Statistical Computation and Simulation,
        86(11), 2165-2181. https://doi.org/10.1080/00949655.2015.1104685
    .. [2] Desgagne, A., Lafaye de Micheaux, P. and Ouimet, F. (2022).
        A comprehensive empirical power comparison of univariate
        goodness-of-fit tests for the Laplace distribution, Section 3.
        https://arxiv.org/abs/2007.06154
    """

    kind = "v"

    @classmethod
    def supported_hypotheses(cls):
        """Return supported composite Laplace hypotheses.

        Returns
        -------
        tuple of HypothesisSupport
            The default parameterization with no fixed parameters.
        """
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @staticmethod
    def short_code():
        """Return the criterion identifier within the Laplace family.

        Returns
        -------
        str
            Short code used to build the full goodness-of-fit identifier.
        """
        return "AP_V"


class AlizadehNoughabiParkELaplaceGofStatistic(AbstractAlizadehNoughabiParkLaplaceGofStatistic):
    """AP_e = log(2*theta_e) + 1 - H_e; m=1 by default.

    Parameters
    ----------
    parameters : ParameterValues
        Unknown location and scale, represented by
        ``LaplaceDistributionDescriptor.DEFAULT.parse({})``.
        Fixed location or scale parameters are not supported.
    m : int, default 1
        Order-statistic window, with ``1 <= m < n/2`` at execution.
        The default is a reproducible setting, not an optimal-window rule.
    scale_estimator : {'corrected', 'mle'}, default 'corrected'
        Use the entropy-specific moment-corrected scale or the Laplace MLE.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute a scalar statistic from a finite, nonconstant sample.
    hypothesis()
        Return the composite null with both parameters unknown.
    alternative()
        Return the rejection-tail specification.
    configuration()
        Return versioned algorithm settings for reproducible calibration.

    Raises
    ------
    ValueError
        If the hypothesis, algorithm settings or sample are unsupported.

    Notes
    -----
    Composite Laplace null: both location and scale are unknown.
    Use Distribution.DEFAULT.parse({}); refit on every simulated sample.
    Rejection direction: right. Constant samples are invalid.

    AP_y requires distinct observations. Other AP variants return positive
    infinity for zero required spacings, provided the corrected scale is
    positive. AP_e uses Ebrahimi entropy.

    Samples must be one-dimensional, finite, real and nonconstant, with
    n >= 3. Each execution refits the sample without changing the input
    or stored hypothesis. Simulation calibration must repeat fitting with
    the same sample size and algorithm settings. No p-value is computed.

    References
    ----------
    .. [1] Alizadeh Noughabi, H. and Park, S. (2016). Tests of fit for
        the Laplace distribution based on correcting moments of entropy
        estimators. Journal of Statistical Computation and Simulation,
        86(11), 2165-2181. https://doi.org/10.1080/00949655.2015.1104685
    .. [2] Desgagne, A., Lafaye de Micheaux, P. and Ouimet, F. (2022).
        A comprehensive empirical power comparison of univariate
        goodness-of-fit tests for the Laplace distribution, Section 3.
        https://arxiv.org/abs/2007.06154
    """

    kind = "e"

    @classmethod
    def supported_hypotheses(cls):
        """Return supported composite Laplace hypotheses.

        Returns
        -------
        tuple of HypothesisSupport
            The default parameterization with no fixed parameters.
        """
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @staticmethod
    def short_code():
        """Return the criterion identifier within the Laplace family.

        Returns
        -------
        str
            Short code used to build the full goodness-of-fit identifier.
        """
        return "AP_E"


class AlizadehNoughabiParkYLaplaceGofStatistic(AbstractAlizadehNoughabiParkLaplaceGofStatistic):
    """AP_y = log(2*theta_y) + 1 - H_y; m=1 by default.

    Parameters
    ----------
    parameters : ParameterValues
        Unknown location and scale, represented by
        ``LaplaceDistributionDescriptor.DEFAULT.parse({})``.
        Fixed location or scale parameters are not supported.
    m : int, default 1
        Order-statistic window, with ``1 <= m < n/2`` at execution.
        The default is a reproducible setting, not an optimal-window rule.
    scale_estimator : {'corrected', 'mle'}, default 'corrected'
        Use the entropy-specific moment-corrected scale or the Laplace MLE.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute a scalar statistic from a finite, nonconstant sample.
    hypothesis()
        Return the composite null with both parameters unknown.
    alternative()
        Return the rejection-tail specification.
    configuration()
        Return versioned algorithm settings for reproducible calibration.

    Raises
    ------
    ValueError
        If the hypothesis, algorithm settings or sample are unsupported.

    Notes
    -----
    Composite Laplace null: both location and scale are unknown.
    Use Distribution.DEFAULT.parse({}); refit on every simulated sample.
    Rejection direction: right. Constant samples are invalid.

    AP_y requires distinct observations. Other AP variants return positive
    infinity for zero required spacings, provided the corrected scale is
    positive. AP_e uses Ebrahimi entropy.

    Samples must be one-dimensional, finite, real and nonconstant, with
    n >= 3. Each execution refits the sample without changing the input
    or stored hypothesis. Simulation calibration must repeat fitting with
    the same sample size and algorithm settings. No p-value is computed.

    References
    ----------
    .. [1] Alizadeh Noughabi, H. and Park, S. (2016). Tests of fit for
        the Laplace distribution based on correcting moments of entropy
        estimators. Journal of Statistical Computation and Simulation,
        86(11), 2165-2181. https://doi.org/10.1080/00949655.2015.1104685
    .. [2] Desgagne, A., Lafaye de Micheaux, P. and Ouimet, F. (2022).
        A comprehensive empirical power comparison of univariate
        goodness-of-fit tests for the Laplace distribution, Section 3.
        https://arxiv.org/abs/2007.06154
    """

    kind = "y"

    @classmethod
    def supported_hypotheses(cls):
        """Return supported composite Laplace hypotheses.

        Returns
        -------
        tuple of HypothesisSupport
            The default parameterization with no fixed parameters.
        """
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @staticmethod
    def short_code():
        """Return the criterion identifier within the Laplace family.

        Returns
        -------
        str
            Short code used to build the full goodness-of-fit identifier.
        """
        return "AP_Y"


class AlizadehNoughabiParkALaplaceGofStatistic(AbstractAlizadehNoughabiParkLaplaceGofStatistic):
    """AP_a = log(2*theta_a) + 1 - H_a; m=1 by default.

    Parameters
    ----------
    parameters : ParameterValues
        Unknown location and scale, represented by
        ``LaplaceDistributionDescriptor.DEFAULT.parse({})``.
        Fixed location or scale parameters are not supported.
    m : int, default 1
        Order-statistic window, with ``1 <= m < n/2`` at execution.
        The default is a reproducible setting, not an optimal-window rule.
    scale_estimator : {'corrected', 'mle'}, default 'corrected'
        Use the entropy-specific moment-corrected scale or the Laplace MLE.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute a scalar statistic from a finite, nonconstant sample.
    hypothesis()
        Return the composite null with both parameters unknown.
    alternative()
        Return the rejection-tail specification.
    configuration()
        Return versioned algorithm settings for reproducible calibration.

    Raises
    ------
    ValueError
        If the hypothesis, algorithm settings or sample are unsupported.

    Notes
    -----
    Composite Laplace null: both location and scale are unknown.
    Use Distribution.DEFAULT.parse({}); refit on every simulated sample.
    Rejection direction: right. Constant samples are invalid.

    AP_y requires distinct observations. Other AP variants return positive
    infinity for zero required spacings, provided the corrected scale is
    positive. AP_e uses Ebrahimi entropy.

    Samples must be one-dimensional, finite, real and nonconstant, with
    n >= 3. Each execution refits the sample without changing the input
    or stored hypothesis. Simulation calibration must repeat fitting with
    the same sample size and algorithm settings. No p-value is computed.

    References
    ----------
    .. [1] Alizadeh Noughabi, H. and Park, S. (2016). Tests of fit for
        the Laplace distribution based on correcting moments of entropy
        estimators. Journal of Statistical Computation and Simulation,
        86(11), 2165-2181. https://doi.org/10.1080/00949655.2015.1104685
    .. [2] Desgagne, A., Lafaye de Micheaux, P. and Ouimet, F. (2022).
        A comprehensive empirical power comparison of univariate
        goodness-of-fit tests for the Laplace distribution, Section 3.
        https://arxiv.org/abs/2007.06154
    """

    kind = "a"

    @classmethod
    def supported_hypotheses(cls):
        """Return supported composite Laplace hypotheses.

        Returns
        -------
        tuple of HypothesisSupport
            The default parameterization with no fixed parameters.
        """
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @staticmethod
    def short_code():
        """Return the criterion identifier within the Laplace family.

        Returns
        -------
        str
            Short code used to build the full goodness-of-fit identifier.
        """
        return "AP_A"


class AlizadehNoughabiParkZLaplaceGofStatistic(AbstractAlizadehNoughabiParkLaplaceGofStatistic):
    """AP_z = log(2*theta_z) + 1 - H_z; m=1 by default.

    Parameters
    ----------
    parameters : ParameterValues
        Unknown location and scale, represented by
        ``LaplaceDistributionDescriptor.DEFAULT.parse({})``.
        Fixed location or scale parameters are not supported.
    m : int, default 1
        Order-statistic window, with ``1 <= m < n/2`` at execution.
        The default is a reproducible setting, not an optimal-window rule.
    scale_estimator : {'corrected', 'mle'}, default 'corrected'
        Use the entropy-specific moment-corrected scale or the Laplace MLE.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute a scalar statistic from a finite, nonconstant sample.
    hypothesis()
        Return the composite null with both parameters unknown.
    alternative()
        Return the rejection-tail specification.
    configuration()
        Return versioned algorithm settings for reproducible calibration.

    Raises
    ------
    ValueError
        If the hypothesis, algorithm settings or sample are unsupported.

    Notes
    -----
    Composite Laplace null: both location and scale are unknown.
    Use Distribution.DEFAULT.parse({}); refit on every simulated sample.
    Rejection direction: right. Constant samples are invalid.

    AP_y requires distinct observations. Other AP variants return positive
    infinity for zero required spacings, provided the corrected scale is
    positive. AP_e uses Ebrahimi entropy.

    Samples must be one-dimensional, finite, real and nonconstant, with
    n >= 3. Each execution refits the sample without changing the input
    or stored hypothesis. Simulation calibration must repeat fitting with
    the same sample size and algorithm settings. No p-value is computed.

    References
    ----------
    .. [1] Alizadeh Noughabi, H. and Park, S. (2016). Tests of fit for
        the Laplace distribution based on correcting moments of entropy
        estimators. Journal of Statistical Computation and Simulation,
        86(11), 2165-2181. https://doi.org/10.1080/00949655.2015.1104685
    .. [2] Desgagne, A., Lafaye de Micheaux, P. and Ouimet, F. (2022).
        A comprehensive empirical power comparison of univariate
        goodness-of-fit tests for the Laplace distribution, Section 3.
        https://arxiv.org/abs/2007.06154
    """

    kind = "z"

    @classmethod
    def supported_hypotheses(cls):
        """Return supported composite Laplace hypotheses.

        Returns
        -------
        tuple of HypothesisSupport
            The default parameterization with no fixed parameters.
        """
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @staticmethod
    def short_code():
        """Return the criterion identifier within the Laplace family.

        Returns
        -------
        str
            Short code used to build the full goodness-of-fit identifier.
        """
        return "AP_Z"


class BestRaynerThasV3LaplaceGofStatistic(AbstractMomentLaplaceGofStatistic):
    """Rescaled smooth component V3: sqrt(n/63) times sample skewness.

    Parameters
    ----------
    parameters : ParameterValues
        Unknown location and scale, represented by
        ``LaplaceDistributionDescriptor.DEFAULT.parse({})``.
        Fixed location or scale parameters are not supported.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute a scalar statistic from a finite, nonconstant sample.
    hypothesis()
        Return the composite null with both parameters unknown.
    alternative()
        Return the rejection-tail specification.
    configuration()
        Return versioned algorithm settings for reproducible calibration.

    Raises
    ------
    ValueError
        If the hypothesis, algorithm settings or sample are unsupported.

    Notes
    -----
    Composite Laplace null: both location and scale are unknown.
    Use Distribution.DEFAULT.parse({}); refit on every simulated sample.
    Rejection direction: two-sided. Constant samples are invalid.

    Central moments use divisor n. The signed component is rescaled
    to unit asymptotic variance; it is not squared. Calibrate this
    normalization, rather than the original unscaled component.

    Samples must be one-dimensional, finite, real and nonconstant, with
    n >= 3. Each execution refits the sample without changing the input
    or stored hypothesis. Simulation calibration must repeat fitting with
    the same sample size and algorithm settings. No p-value is computed.

    References
    ----------
    .. [1] Best, D. J., Rayner, J. C. W. and Thas, O. (2008).
        Comparison of some tests of fit for the Laplace distribution.
        Computational Statistics & Data Analysis, 52(12), 5338-5343.
        https://doi.org/10.1016/j.csda.2008.05.023
    """

    kind = "v3"

    @classmethod
    def supported_hypotheses(cls):
        """Return supported composite Laplace hypotheses.

        Returns
        -------
        tuple of HypothesisSupport
            The default parameterization with no fixed parameters.
        """
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @staticmethod
    def short_code():
        """Return the criterion identifier within the Laplace family.

        Returns
        -------
        str
            Short code used to build the full goodness-of-fit identifier.
        """
        return "BRT_V3"

    def alternative(self):
        """Return the rejection-tail specification.

        Returns
        -------
        Alternative
            Left-tailed for CK, two-sided for signed moment components,
            or right-tailed for omnibus statistics, as defined by this class.
        """
        return TwoSidedAlternative()


class BestRaynerThasV4LaplaceGofStatistic(AbstractMomentLaplaceGofStatistic):
    """Rescaled smooth component V4: sqrt(n/1188) times (sample kurtosis - 6).

    Parameters
    ----------
    parameters : ParameterValues
        Unknown location and scale, represented by
        ``LaplaceDistributionDescriptor.DEFAULT.parse({})``.
        Fixed location or scale parameters are not supported.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute a scalar statistic from a finite, nonconstant sample.
    hypothesis()
        Return the composite null with both parameters unknown.
    alternative()
        Return the rejection-tail specification.
    configuration()
        Return versioned algorithm settings for reproducible calibration.

    Raises
    ------
    ValueError
        If the hypothesis, algorithm settings or sample are unsupported.

    Notes
    -----
    Composite Laplace null: both location and scale are unknown.
    Use Distribution.DEFAULT.parse({}); refit on every simulated sample.
    Rejection direction: two-sided. Constant samples are invalid.

    Central moments use divisor n. The signed component is rescaled
    to unit asymptotic variance; it is not squared. Calibrate this
    normalization, rather than the original unscaled component.

    Samples must be one-dimensional, finite, real and nonconstant, with
    n >= 3. Each execution refits the sample without changing the input
    or stored hypothesis. Simulation calibration must repeat fitting with
    the same sample size and algorithm settings. No p-value is computed.

    References
    ----------
    .. [1] Best, D. J., Rayner, J. C. W. and Thas, O. (2008).
        Comparison of some tests of fit for the Laplace distribution.
        Computational Statistics & Data Analysis, 52(12), 5338-5343.
        https://doi.org/10.1016/j.csda.2008.05.023
    """

    kind = "v4"

    @classmethod
    def supported_hypotheses(cls):
        """Return supported composite Laplace hypotheses.

        Returns
        -------
        tuple of HypothesisSupport
            The default parameterization with no fixed parameters.
        """
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @staticmethod
    def short_code():
        """Return the criterion identifier within the Laplace family.

        Returns
        -------
        str
            Short code used to build the full goodness-of-fit identifier.
        """
        return "BRT_V4"

    def alternative(self):
        """Return the rejection-tail specification.

        Returns
        -------
        Alternative
            Left-tailed for CK, two-sided for signed moment components,
            or right-tailed for omnibus statistics, as defined by this class.
        """
        return TwoSidedAlternative()


class GelLaplaceGofStatistic(AbstractMomentLaplaceGofStatistic):
    """Ge = n*q3^2/60 + n*(q4-6)^2/1200; qj use sqrt(2)*MAD scale.

    Parameters
    ----------
    parameters : ParameterValues
        Unknown location and scale, represented by
        ``LaplaceDistributionDescriptor.DEFAULT.parse({})``.
        Fixed location or scale parameters are not supported.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute a scalar statistic from a finite, nonconstant sample.
    hypothesis()
        Return the composite null with both parameters unknown.
    alternative()
        Return the rejection-tail specification.
    configuration()
        Return versioned algorithm settings for reproducible calibration.

    Raises
    ------
    ValueError
        If the hypothesis, algorithm settings or sample are unsupported.

    Notes
    -----
    Composite Laplace null: both location and scale are unknown.
    Use Distribution.DEFAULT.parse({}); refit on every simulated sample.
    Rejection direction: right. Constant samples are invalid.

    Here q_j=mean(((x-mean(x))/(sqrt(2)*b))**j), where b is the
    mean absolute deviation about the median. The constants 60 and 1200
    are the recommended small-to-moderate-sample choices.

    Samples must be one-dimensional, finite, real and nonconstant, with
    n >= 3. Each execution refits the sample without changing the input
    or stored hypothesis. Simulation calibration must repeat fitting with
    the same sample size and algorithm settings. No p-value is computed.

    References
    ----------
    .. [1] Gel, Y. R. (2010). Test of fit for a Laplace distribution
        against heavier tailed alternatives. Computational Statistics & Data
        Analysis, 54(4), 958-965. https://doi.org/10.1016/j.csda.2009.10.008
    .. [2] Desgagne, A., Lafaye de Micheaux, P. and Ouimet, F. (2022).
        A comprehensive empirical power comparison of univariate
        goodness-of-fit tests for the Laplace distribution, Section 3.
        https://arxiv.org/abs/2007.06154
    """

    kind = "gel"

    @classmethod
    def supported_hypotheses(cls):
        """Return supported composite Laplace hypotheses.

        Returns
        -------
        tuple of HypothesisSupport
            The default parameterization with no fixed parameters.
        """
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @staticmethod
    def short_code():
        """Return the criterion identifier within the Laplace family.

        Returns
        -------
        str
            Short code used to build the full goodness-of-fit identifier.
        """
        return "GE"


class GonzalezEstradaVillasenorLaplaceGofStatistic(AbstractMomentLaplaceGofStatistic):
    """GV = 2*sqrt(n)*(s_n/(sqrt(2)*mean|x-mean(x)|)-1).

    Parameters
    ----------
    parameters : ParameterValues
        Unknown location and scale, represented by
        ``LaplaceDistributionDescriptor.DEFAULT.parse({})``.
        Fixed location or scale parameters are not supported.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute a scalar statistic from a finite, nonconstant sample.
    hypothesis()
        Return the composite null with both parameters unknown.
    alternative()
        Return the rejection-tail specification.
    configuration()
        Return versioned algorithm settings for reproducible calibration.

    Raises
    ------
    ValueError
        If the hypothesis, algorithm settings or sample are unsupported.

    Notes
    -----
    Composite Laplace null: both location and scale are unknown.
    Use Distribution.DEFAULT.parse({}); refit on every simulated sample.
    Rejection direction: two-sided. Constant samples are invalid.

    The standard deviation uses divisor n. The denominator is the mean
    absolute deviation about the sample mean, not about the median.

    Samples must be one-dimensional, finite, real and nonconstant, with
    n >= 3. Each execution refits the sample without changing the input
    or stored hypothesis. Simulation calibration must repeat fitting with
    the same sample size and algorithm settings. No p-value is computed.

    References
    ----------
    .. [1] Gonzalez-Estrada, E. and Villasenor, J. A. (2016).
        A ratio goodness-of-fit test for the Laplace distribution.
        Statistics & Probability Letters, 119, 30-35.
        https://doi.org/10.1016/j.spl.2016.07.003
    .. [2] Desgagne, A., Lafaye de Micheaux, P. and Ouimet, F. (2022).
        A comprehensive empirical power comparison of univariate
        goodness-of-fit tests for the Laplace distribution, Section 3.
        https://arxiv.org/abs/2007.06154
    """

    kind = "gv"

    @classmethod
    def supported_hypotheses(cls):
        """Return supported composite Laplace hypotheses.

        Returns
        -------
        tuple of HypothesisSupport
            The default parameterization with no fixed parameters.
        """
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @staticmethod
    def short_code():
        """Return the criterion identifier within the Laplace family.

        Returns
        -------
        str
            Short code used to build the full goodness-of-fit identifier.
        """
        return "GV"

    def alternative(self):
        """Return the rejection-tail specification.

        Returns
        -------
        Alternative
            Left-tailed for CK, two-sided for signed moment components,
            or right-tailed for omnibus statistics, as defined by this class.
        """
        return TwoSidedAlternative()


class DesgagneLafayeDeMicheauxOuimetXLaplaceGofStatistic(AbstractMomentLaplaceGofStatistic):
    """DLO_X = Z(S)^2 + Z(Knet)^2. Requires n >= 4.

    Parameters
    ----------
    parameters : ParameterValues
        Unknown location and scale, represented by
        ``LaplaceDistributionDescriptor.DEFAULT.parse({})``.
        Fixed location or scale parameters are not supported.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute a scalar statistic from a finite, nonconstant sample.
    hypothesis()
        Return the composite null with both parameters unknown.
    alternative()
        Return the rejection-tail specification.
    configuration()
        Return versioned algorithm settings for reproducible calibration.

    Raises
    ------
    ValueError
        If the hypothesis, algorithm settings or sample are unsupported.

    Notes
    -----
    Composite Laplace null: both location and scale are unknown.
    Use Distribution.DEFAULT.parse({}); refit on every simulated sample.
    Rejection direction: right. Constant samples are invalid.

    The implementation includes distinct finite-sample corrections for
    odd and even n and uses 0*log(0)=0. Normal (DLO_Z) and chi-square
    with two degrees of freedom (DLO_X) approximations are recommended
    for n >= 20; smaller samples require simulation calibration.

    Samples must be one-dimensional, finite, real and nonconstant, with
    n >= 4. Each execution refits the sample without changing the input
    or stored hypothesis. Simulation calibration must repeat fitting with
    the same sample size and algorithm settings. No p-value is computed.

    References
    ----------
    .. [1] Desgagne, A., Lafaye de Micheaux, P. and Ouimet, F. (2023).
        Goodness-of-fit tests for Laplace, Gaussian and exponential power
        distributions based on lambda-th power skewness and kurtosis.
        Statistics, 57(1), 94-122.
        https://doi.org/10.1080/02331888.2022.2144859
        Author manuscript: https://arxiv.org/abs/1809.02852
    .. [2] Desgagne, A., Lafaye de Micheaux, P. and Ouimet, F. (2022).
        A comprehensive empirical power comparison of univariate
        goodness-of-fit tests for the Laplace distribution, Section 3.
        https://arxiv.org/abs/2007.06154
    """

    kind = "dlo_x"

    @classmethod
    def supported_hypotheses(cls):
        """Return supported composite Laplace hypotheses.

        Returns
        -------
        tuple of HypothesisSupport
            The default parameterization with no fixed parameters.
        """
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @staticmethod
    def short_code():
        """Return the criterion identifier within the Laplace family.

        Returns
        -------
        str
            Short code used to build the full goodness-of-fit identifier.
        """
        return "DLO_X"


class DesgagneLafayeDeMicheauxOuimetZLaplaceGofStatistic(AbstractMomentLaplaceGofStatistic):
    """DLO_Z = Z(Knet), signed kurtosis component. Requires n >= 4.

    Parameters
    ----------
    parameters : ParameterValues
        Unknown location and scale, represented by
        ``LaplaceDistributionDescriptor.DEFAULT.parse({})``.
        Fixed location or scale parameters are not supported.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute a scalar statistic from a finite, nonconstant sample.
    hypothesis()
        Return the composite null with both parameters unknown.
    alternative()
        Return the rejection-tail specification.
    configuration()
        Return versioned algorithm settings for reproducible calibration.

    Raises
    ------
    ValueError
        If the hypothesis, algorithm settings or sample are unsupported.

    Notes
    -----
    Composite Laplace null: both location and scale are unknown.
    Use Distribution.DEFAULT.parse({}); refit on every simulated sample.
    Rejection direction: two-sided. Constant samples are invalid.

    The implementation includes distinct finite-sample corrections for
    odd and even n and uses 0*log(0)=0. Normal (DLO_Z) and chi-square
    with two degrees of freedom (DLO_X) approximations are recommended
    for n >= 20; smaller samples require simulation calibration.

    Samples must be one-dimensional, finite, real and nonconstant, with
    n >= 4. Each execution refits the sample without changing the input
    or stored hypothesis. Simulation calibration must repeat fitting with
    the same sample size and algorithm settings. No p-value is computed.

    References
    ----------
    .. [1] Desgagne, A., Lafaye de Micheaux, P. and Ouimet, F. (2023).
        Goodness-of-fit tests for Laplace, Gaussian and exponential power
        distributions based on lambda-th power skewness and kurtosis.
        Statistics, 57(1), 94-122.
        https://doi.org/10.1080/02331888.2022.2144859
        Author manuscript: https://arxiv.org/abs/1809.02852
    .. [2] Desgagne, A., Lafaye de Micheaux, P. and Ouimet, F. (2022).
        A comprehensive empirical power comparison of univariate
        goodness-of-fit tests for the Laplace distribution, Section 3.
        https://arxiv.org/abs/2007.06154
    """

    kind = "dlo_z"

    @classmethod
    def supported_hypotheses(cls):
        """Return supported composite Laplace hypotheses.

        Returns
        -------
        tuple of HypothesisSupport
            The default parameterization with no fixed parameters.
        """
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @staticmethod
    def short_code():
        """Return the criterion identifier within the Laplace family.

        Returns
        -------
        str
            Short code used to build the full goodness-of-fit identifier.
        """
        return "DLO_Z"

    def alternative(self):
        """Return the rejection-tail specification.

        Returns
        -------
        Alternative
            Left-tailed for CK, two-sided for signed moment components,
            or right-tailed for omnibus statistics, as defined by this class.
        """
        return TwoSidedAlternative()


class DataDrivenSmoothLaplaceGofStatistic(AbstractFittedLaplaceGofStatistic):
    """Data-driven efficient Legendre score test for fitted Laplace data.

    Parameters
    ----------
    parameters : ParameterValues
        Unknown location and scale, represented by
        ``LaplaceDistributionDescriptor.DEFAULT.parse({})``.
        Fixed location or scale parameters are not supported.
    max_components : int, default 4
        Largest candidate number of components, from 1 through 10.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Compute a scalar statistic from a finite, nonconstant sample.
    hypothesis()
        Return the composite null with both parameters unknown.
    alternative()
        Return the rejection-tail specification.
    configuration()
        Return versioned algorithm settings for reproducible calibration.

    Raises
    ------
    ValueError
        If the hypothesis, algorithm settings or sample are unsupported.

    Notes
    -----
    W_k=n*v_k'*(I-A_k*A_k')^(-1)*v_k, v=mean(phi)-A*mean(score).
    Select the smallest k maximizing W_k-k*log(n), 1 <= k <= max_components.
    This is an explicit Laplace specialization of the general score selection
    construction, not a reproduction of the DD implementation compared in
    Best et al. (2008). Calibrate by refitting simulated Laplace(0,1) samples.

    Samples must be one-dimensional, finite, real and nonconstant, with
    n >= 3. Each execution refits the sample without changing the input
    or stored hypothesis. Simulation calibration must repeat fitting with
    the same sample size and algorithm settings. No p-value is computed.

    References
    ----------
    .. [1] Janic, A. and Ledwina, T. Data-Driven Smooth Tests for a
        Location-Scale Family Revisited, Sections 2 and 2.1.
        https://www.impan.pl/~ledwina/Janic-Ledwina-24-09-08.pdf
    .. [2] Best, D. J., Rayner, J. C. W. and Thas, O. (2008).
        Comparison of some tests of fit for the Laplace distribution.
        Computational Statistics & Data Analysis, 52(12), 5338-5343.
        https://doi.org/10.1016/j.csda.2008.05.023
    """

    @classmethod
    def supported_hypotheses(cls):
        """Return supported composite Laplace hypotheses.

        Returns
        -------
        tuple of HypothesisSupport
            The default parameterization with no fixed parameters.
        """
        return (HypothesisSupport(Distribution.DEFAULT, frozenset()),)

    @staticmethod
    def short_code():
        """Return the criterion identifier within the Laplace family.

        Returns
        -------
        str
            Short code used to build the full goodness-of-fit identifier.
        """
        return "DD_SMOOTH"

    def __init__(self, parameters, *, max_components=4):
        """Initialize the criterion and validate its algorithm settings.

        Parameters
        ----------
        parameters : ParameterValues
            Unknown location and scale, represented by
            ``LaplaceDistributionDescriptor.DEFAULT.parse({})``.
            Fixed location or scale parameters are not supported.
        max_components : int, default 4
            Largest candidate number of components, from 1 through 10.

        Raises
        ------
        TypeError
            If parameters is not a ParameterValues instance.
        ValueError
            If the hypothesis or algorithm settings are unsupported.

        Notes
        -----
        Sample-dependent conditions are checked during execution. See the
        class documentation for the statistic definition and references.
        """
        super().__init__(parameters)
        if (
            isinstance(max_components, bool)
            or not isinstance(max_components, (int, np.integer))
            or not 1 <= max_components <= 10
        ):
            raise ValueError("max_components must be an integer from 1 to 10.")
        self.max_components = int(max_components)
        self._projection, self._inverses = self._score_matrices(self.max_components)

    def configuration(self):
        """Return the settings used to identify a calibration.

        Returns
        -------
        dict
            Schema and statistic versions and all algorithm-specific options.
        """
        return {**super().configuration(), "max_components": self.max_components}

    @staticmethod
    def _score_matrices(degree):
        # Deterministic population integrals; computed once, outside the sample loop.
        """Construct population score projections and inverse covariances.

        Parameters
        ----------
        degree : int
            Maximum number of Legendre components, between 1 and 10.

        Returns
        -------
        projection : ndarray of float, shape (degree, 2)
            Projections onto location and scale scores under standard Laplace.
        inverses : ndarray of float, shape (degree, degree, degree)
            Leading k-by-k blocks contain inverse efficient-score covariance
            matrices at index k-1; remaining entries are zero.

        References
        ----------
        .. [1] Janic and Ledwina, Data-Driven Smooth Tests for a Location-Scale
            Family Revisited. https://www.impan.pl/~ledwina/Janic-Ledwina-24-09-08.pdf
        """
        from scipy.integrate import quad
        from scipy.special import eval_legendre

        projection = np.zeros((degree, 2))
        for j in range(1, degree + 1):

            def phi(u, order=j):
                return np.sqrt(2 * order + 1) * eval_legendre(order, 2 * u - 1)

            if j % 2:
                projection[j - 1, 0] = -2 * quad(phi, 0, 0.5, epsabs=1e-12)[0]
            else:
                projection[j - 1, 1] = (
                    2 * quad(lambda u: phi(u) * (-np.log(2 * u) - 1), 0, 0.5, epsabs=1e-12)[0]
                )
        inverses = np.zeros((degree, degree, degree))
        for k in range(1, degree + 1):
            covariance = np.eye(k) - projection[:k] @ projection[:k].T
            inverses[k - 1, :k, :k] = np.linalg.inv(covariance)
        return projection, inverses

    @staticmethod
    @jit(nopython=True, cache=True)
    def _selected_score(z, projection, inverses):
        """Select an efficient-score statistic using the Schwarz penalty.

        Parameters
        ----------
        z : ndarray of float, shape (n,)
            Sorted observations standardized by the sample median and mean
            absolute deviation about that median.
        projection : ndarray of float, shape (d, 2)
            Population projections returned by ``_score_matrices``.
        inverses : ndarray of float, shape (d, d, d)
            Inverse covariance blocks returned by ``_score_matrices``.

        Returns
        -------
        float
            Unpenalized W_k at the smallest k maximizing W_k-k*log(n).

        References
        ----------
        .. [1] Janic and Ledwina, Data-Driven Smooth Tests for a Location-Scale
            Family Revisited. https://www.impan.pl/~ledwina/Janic-Ledwina-24-09-08.pdf
        """
        n, degree = len(z), len(projection)
        values = np.zeros(degree)
        scores = np.zeros(2)
        for x in z:
            scores[0] += np.sign(x) / n
            scores[1] += (abs(x) - 1) / n
            u = 0.5 * np.exp(x) if x < 0 else 1 - 0.5 * np.exp(-x)
            q = 2 * u - 1
            prev, current = 1.0, q
            for j in range(1, degree + 1):
                values[j - 1] += np.sqrt(2 * j + 1) * current / n
                prev, current = current, ((2 * j + 1) * q * current - j * prev) / (j + 1)
        values -= projection @ scores
        best, selected = -np.inf, 0.0
        for k in range(1, degree + 1):
            w = 0.0
            for i in range(k):
                for j in range(k):
                    w += n * values[i] * inverses[k - 1, i, j] * values[j]
            penalized = w - k * np.log(n)
            if penalized > best:
                best, selected = penalized, w
        return selected

    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic for a newly fitted Laplace sample.

        Parameters
        ----------
        rvs : array_like of float, shape (n,)
            Finite, real, nonconstant observations. At least three are needed;
            DLO requires four. Entropy windows must satisfy ``2*m < n``.
        **kwargs : dict
            Accepted for interface compatibility; ignored.

        Returns
        -------
        float
            Statistic in the normalization documented by the concrete class.
            Degenerate spacings may produce zero or infinity as documented.

        Raises
        ------
        ValueError
            If the sample or window does not meet the criterion requirements.

        Notes
        -----
        The input and stored hypothesis are not modified. Refer to the
        concrete class for rejection direction, formulas and references.
        """
        return float(self._selected_score(self._fit(rvs), self._projection, self._inverses))


__all__.extend(
    [
        "AbstractAlizadehNoughabiParkLaplaceGofStatistic",
        "AbstractChoiKimLaplaceGofStatistic",
        "AbstractEntropyLaplaceGofStatistic",
        "AbstractFittedLaplaceGofStatistic",
        "AbstractMeintanisLaplaceGofStatistic",
        "AbstractMomentLaplaceGofStatistic",
        "AlizadehNoughabiJarrahiferizLaplaceGofStatistic",
        "AlizadehNoughabiLikelihoodLaplaceGofStatistic",
        "AlizadehNoughabiParkALaplaceGofStatistic",
        "AlizadehNoughabiParkELaplaceGofStatistic",
        "AlizadehNoughabiParkVLaplaceGofStatistic",
        "AlizadehNoughabiParkYLaplaceGofStatistic",
        "AlizadehNoughabiParkZLaplaceGofStatistic",
        "BestRaynerThasV3LaplaceGofStatistic",
        "BestRaynerThasV4LaplaceGofStatistic",
        "ChoiKimCLaplaceGofStatistic",
        "ChoiKimELaplaceGofStatistic",
        "ChoiKimVLaplaceGofStatistic",
        "DataDrivenSmoothLaplaceGofStatistic",
        "DesgagneLafayeDeMicheauxOuimetXLaplaceGofStatistic",
        "DesgagneLafayeDeMicheauxOuimetZLaplaceGofStatistic",
        "GelLaplaceGofStatistic",
        "GonzalezEstradaVillasenorLaplaceGofStatistic",
        "Meintanis1LaplaceGofStatistic",
        "Meintanis2LaplaceGofStatistic",
    ]
)
