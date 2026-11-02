import inspect
import sys
from abc import ABC

import numpy as np
import scipy.stats as scipy_stats
from numba import jit
from scipy import integrate
from typing_extensions import override

from pysatl_criterion.distribution import (
    LogNormalDistributionDescriptor as Distribution,
)
from pysatl_criterion.distribution import NormalDistributionDescriptor as Normal
from pysatl_criterion.distribution.parameters import (
    HypothesisSupport,
    ParameterizationDescriptor,
)
from pysatl_criterion.statistics.alternative import Alternative, RightAlternative
from pysatl_criterion.statistics.goodness_of_fit import edf, frequency
from pysatl_criterion.statistics.goodness_of_fit.continuous import normal
from pysatl_criterion.statistics.goodness_of_fit.continuous._edf_factory import (
    EDFSpec,
    export_edf_statistics,
)
from pysatl_criterion.statistics.goodness_of_fit.continuous._frequency_factory import (
    FrequencySpec,
    export_frequency_statistics,
)
from pysatl_criterion.statistics.goodness_of_fit.continuous.normal import (
    AbstractNormalityGofStatistic,
)
from pysatl_criterion.statistics.goodness_of_fit.goodness_of_fit_statistic import (
    AbstractGoodnessOfFitStatistic,
)


class AbstractLogNormalGofStatistic(AbstractGoodnessOfFitStatistic, ABC):
    """
    Base class for Log-Normal statistics using explicit shape/median coordinates.

    Concrete criteria declare fixed or composite hypotheses in both logarithmic
    and shape/median coordinates. Calculations use SHAPE_SCALE; hypothesis()
    preserves the input coordinates. Missing parameter values are never filled.
    """

    @staticmethod
    @jit(nopython=True, cache=True)
    def _kl_coefficients(z):
        """Cumulant coefficients of D(r)/r**3 through order seven."""
        moments = np.empty(11)
        moments[0], moments[1] = 1.0, 0.0
        for k in range(2, 11):
            moments[k] = np.mean(z**k)
        cumulants = np.zeros(11)
        coefficients = np.empty(8)
        factorial = 1.0
        for k in range(2, 11):
            factorial *= k
            cumulants[k] = moments[k]
            combination = 1.0
            for j in range(1, k):
                cumulants[k] -= combination * cumulants[j] * moments[k - j]
                combination *= (k - j) / j
            if k >= 3:
                coefficients[k - 3] = (2 - k) * cumulants[k] / factorial
        return coefficients

    @staticmethod
    @jit(nopython=True, cache=True)
    def _kl_ratio(r, z, coefficients, bound):
        """Stable D(r)/sigma(r), including its continuous absolute limit at zero.

        Near zero use cumulant and variance series; elsewhere evaluate the
        exponential weights after subtracting their maximum to avoid overflow.
        """
        t = r * r
        if abs(r) * bound <= 0.05:
            numerator = 0.0
            for k in range(len(coefficients) - 1, -1, -1):
                numerator = numerator * r + coefficients[k]
            term = 1.0 / 6.0
            variance = 0.0
            for j in range(12):
                variance += (j + 1) ** 2 * term
                term *= t / (j + 4)
            return numerator / np.sqrt(variance) * (-1 if r < 0 else 1)
        a = r * z
        maximum = np.max(a)
        weights = np.exp(a - maximum)
        total = np.sum(weights)
        log_total = maximum + np.log(total)
        derivative = np.sum(weights * z) / total
        d = 2 * (log_total - np.log(len(z))) - r * derivative
        if t < 0.5:
            term = 1.0 / 6.0
            variance = 0.0
            for j in range(16):
                variance += (j + 1) ** 2 * term
                term *= t / (j + 4)
            variance *= r**6
        else:
            variance = np.exp(t) * (t * t - 3 * t + 4) - 4 - t
        return d / np.sqrt(variance)

    @classmethod
    def _kl_ratio_function(cls, z):
        coefficients = cls._kl_coefficients(z)
        bound = np.max(np.abs(z))
        return lambda r: cls._kl_ratio(r, z, coefficients, bound)

    @staticmethod
    def _legacy_critical_value(value):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Legacy critical-value approximation is invalid here; use Monte Carlo")
        return float(value)

    @staticmethod
    def _validated_sample(rvs):
        """Validate before conversion so masks and imaginary parts cannot disappear."""
        if np.iscomplexobj(rvs) or (np.ma.isMaskedArray(rvs) and np.any(rvs.mask)):
            raise ValueError("Sample must be real and contain no masked observations")
        try:
            x = np.asarray(rvs, dtype=float)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("Sample must contain real finite observations") from error
        if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
            raise ValueError("Sample must be nonempty, one-dimensional and finite")
        return x

    @staticmethod
    def _positive_logs(rvs):
        x = AbstractLogNormalGofStatistic._validated_sample(rvs)
        if np.any(x <= 0):
            raise ValueError("Logarithmic statistics require strictly positive observations")
        return np.log(x)

    @staticmethod
    def _standardized_logs(rvs):
        y = AbstractLogNormalGofStatistic._positive_logs(rvs)
        if y.size < 2 or np.ptp(y) == 0:
            raise ValueError("KL statistics require at least two nonconstant observations")
        y = y - np.mean(y)
        return y / np.std(y, ddof=0)

    @staticmethod
    def _create_lognormal_class(normal_cls):
        new_class_name = normal_cls.__name__.replace("Normality", "LogNormal")
        class_short_code = normal_cls.short_code()

        class LogNormalClass(AbstractLogNormalGofStatistic, ABC):
            @classmethod
            def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
                return (
                    HypothesisSupport(Distribution.LOG_LOCATION_SCALE, frozenset()),
                    HypothesisSupport(Distribution.SHAPE_SCALE, frozenset()),
                    HypothesisSupport(
                        Distribution.LOG_LOCATION_SCALE,
                        frozenset({Distribution.LOG_LOCATION, Distribution.LOG_SCALE}),
                    ),
                    HypothesisSupport(
                        Distribution.SHAPE_SCALE,
                        frozenset({Distribution.LOG_SCALE, Distribution.MEDIAN}),
                    ),
                )

            @classmethod
            def _standard_calibration_parameters(cls):
                return Distribution.LOG_LOCATION_SCALE.parse({"mu": 0, "s": 1})

            @override
            def alternative(self) -> Alternative:
                return normal_cls(Normal.DEFAULT.parse({})).alternative()

            @override
            def execute_statistic(self, rvs):
                """
                Execute the wrapped normality test on log-transformed data.

                :param rvs: array of observed data samples.
                :return: test statistic value.
                """
                log_rvs = AbstractLogNormalGofStatistic._positive_logs(rvs)
                return normal_cls(Normal.DEFAULT.parse({})).execute_statistic(log_rvs)

            @staticmethod
            @override
            def short_code():
                """
                Get short code identifier for this dynamically generated test.

                :return: short code inherited from the parent Normality test.
                """
                return class_short_code

        parent_doc = inspect.getdoc(normal_cls) or ""
        notes = parent_doc.split("Notes\n-----", 1)[-1].split("Examples\n--------", 1)[0]
        if "References" not in notes and "References\n----------" in parent_doc:
            notes += "\n\nReferences\n----------" + parent_doc.split("References\n----------", 1)[1]
        LogNormalClass.__doc__ = (
            "Log-normal family statistic evaluated on log(rvs).\n\n"
            "Use Distribution.DEFAULT.parse({}). Fixed schemas are accepted as metadata\n"
            "but cannot make this statistic sensitive to log-location or log-scale.\n"
            "Strictly positive finite data are required. The following normality\n"
            "definition uses x to denote log(rvs), at the parent's default settings.\n\n" + notes
        )
        LogNormalClass.__name__ = new_class_name
        LogNormalClass.__qualname__ = new_class_name
        return LogNormalClass

    @staticmethod
    def _lognormal_log_probabilities(criterion, sample):
        if criterion.short_code() in {"ZWA", "ZWC"} and len(sample) < 4:
            raise ValueError("Zhang-Wu statistics require at least four observations")
        bound = Distribution.bind(criterion._parameters)
        return bound.logcdf(sample), bound.logsf(sample)

    @staticmethod
    def _lognormal_cdf(criterion, sample):
        return Distribution.cdf(sample, criterion._parameters)

    def _zhang_wu_a_statistic(self, rvs, **kwargs):
        value = globals()["ZhangALogNormalGofStatistic"].execute_statistic(self, rvs, **kwargs)
        return 10 * value - 32

    @property
    def s(self) -> float:
        """Read s by its stable parameter identity."""
        return self._parameters[Distribution.LOG_SCALE]

    @property
    def scale(self) -> float:
        """Read scale by its stable parameter identity."""
        return self._parameters[Distribution.MEDIAN]

    @classmethod
    def calculation_parameterization(cls) -> ParameterizationDescriptor:
        return Distribution.SHAPE_SCALE

    @staticmethod
    @override
    def distribution() -> type[Distribution]:
        """Return the distribution descriptor class."""
        return Distribution

    @classmethod
    @override
    def code(cls) -> str:
        """Return the family identifier or the concrete statistic's full identifier."""
        family_code = f"LOGNORMAL_{AbstractGoodnessOfFitStatistic.code()}"
        if "short_code" in cls.__abstractmethods__:
            return family_code
        return f"{cls.short_code()}_{family_code}"


# =================================================================================================
# EXPLICIT IMPLEMENTATIONS
# =================================================================================================


class QuesenberryMillerLogNormalGofStatistic(AbstractLogNormalGofStatistic):
    """Parameters
    ----------
    parameters : ParameterValues
        Values with mu, s fixed; omitted parameters are unknown.

    Quesenberry and Miller's Q-test for Log-Normal distribution.
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (
            HypothesisSupport(
                Distribution.LOG_LOCATION_SCALE,
                frozenset({Distribution.LOG_LOCATION, Distribution.LOG_SCALE}),
            ),
            HypothesisSupport(
                Distribution.SHAPE_SCALE, frozenset({Distribution.LOG_SCALE, Distribution.MEDIAN})
            ),
        )

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "QUESENBERRY_MILLER".
        """
        return "QUESENBERRY_MILLER"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """
        Execute the Quesenberry-Miller Q-test statistic for Log-Normal distribution.

        :param rvs: array of observed data samples.
        :return: Quesenberry-Miller test statistic value.
        """
        rvs = AbstractLogNormalGofStatistic._validated_sample(rvs)

        # Transform data to [0, 1] using the CDF of the Log-Normal distribution
        cdf_vals = scipy_stats.lognorm.cdf(rvs, s=self.s, scale=self.scale)

        return self.do_execute_statistic(cdf_vals)

    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(cdf_vals):
        """Sum squared and adjacent uniform spacings, including both boundaries."""
        ordered = np.sort(cdf_vals)
        total = 0.0
        previous = 0.0
        previous_spacing = 0.0
        for value in ordered:
            spacing = value - previous
            total += spacing * spacing + previous_spacing * spacing
            previous = value
            previous_spacing = spacing
        spacing = 1.0 - previous
        return total + spacing * spacing + previous_spacing * spacing


class KLSupremumLogNormalGoFStatistic(AbstractLogNormalGofStatistic):
    """Parameters
    ----------
    parameters : ParameterValues
        Use an empty schema for the log-normal family. Fully fixed schemas
        remain accepted as metadata but do not affect this shape statistic.

    Supremum test for lognormality based on Kullback-Leibler divergences.
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (
            HypothesisSupport(Distribution.LOG_LOCATION_SCALE, frozenset()),
            HypothesisSupport(Distribution.SHAPE_SCALE, frozenset()),
            HypothesisSupport(
                Distribution.LOG_LOCATION_SCALE,
                frozenset({Distribution.LOG_LOCATION, Distribution.LOG_SCALE}),
            ),
            HypothesisSupport(
                Distribution.SHAPE_SCALE, frozenset({Distribution.LOG_SCALE, Distribution.MEDIAN})
            ),
        )

    @classmethod
    def _standard_calibration_parameters(cls):
        return Distribution.LOG_LOCATION_SCALE.parse({"mu": 0, "s": 1})

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "KL_SUP".
        """
        return "KL_SUP"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Evaluate the documented truncated KL functional on fitted log data."""
        z = AbstractLogNormalGofStatistic._standardized_logs(rvs)
        return self.do_execute_statistic(z, self._kl_coefficients(z), self._kl_ratio)

    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(z, coefficients, ratio):
        """Evaluate the entire fixed grid in compiled code."""
        bound = np.max(np.abs(z))
        maximum = 0.0
        for r in np.linspace(-5, 5, 1000):
            maximum = max(maximum, abs(ratio(r, z, coefficients, bound)))
        return np.sqrt(len(z)) * maximum

    def calculate_critical_value(self, n, alpha=0.05):
        """
        Calculate critical value for the KL Supremum test.

        :param n: sample size.
        :param alpha: significance level: only 0.01 and 0.05 are implemented.
        :return: legacy regression value, not a verified finite-sample quantile.
        :raises ValueError: invalid n or alpha.
        :raises NotImplementedError: other levels require Monte Carlo calibration.

        The original regression domain and compatibility with the truncated
        statistic have not been verified. Prefer simulation at the actual n.
        """
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 2:
            raise ValueError("n must be an integer of at least two")
        if not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("alpha must lie strictly between zero and one")
        if alpha == 0.05:
            # Formula (27)
            return AbstractLogNormalGofStatistic._legacy_critical_value(
                2.67076 - 3.90704 / np.sqrt(n) + 1.07162 / n
            )
        elif alpha == 0.01:
            # Formula (29)
            return AbstractLogNormalGofStatistic._legacy_critical_value(
                4.10183 - 7.48001 / np.sqrt(n) + 0.638394 * np.log(n) / np.sqrt(n)
            )
        else:
            raise NotImplementedError(
                "Only legacy alpha=0.01/0.05 approximations are available; "
                "calibrate other levels by Monte Carlo"
            )


class KLIntegralLogNormalGoFStatistic(AbstractLogNormalGofStatistic):
    """Parameters
    ----------
    parameters : ParameterValues
        Use an empty schema for the log-normal family. Fully fixed schemas
        remain accepted as metadata but do not affect this shape statistic.

    Integral test for lognormality based on Kullback-Leibler divergences.
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (
            HypothesisSupport(Distribution.LOG_LOCATION_SCALE, frozenset()),
            HypothesisSupport(Distribution.SHAPE_SCALE, frozenset()),
            HypothesisSupport(
                Distribution.LOG_LOCATION_SCALE,
                frozenset({Distribution.LOG_LOCATION, Distribution.LOG_SCALE}),
            ),
            HypothesisSupport(
                Distribution.SHAPE_SCALE, frozenset({Distribution.LOG_SCALE, Distribution.MEDIAN})
            ),
        )

    @classmethod
    def _standard_calibration_parameters(cls):
        return Distribution.LOG_LOCATION_SCALE.parse({"mu": 0, "s": 1})

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        """
        Get short code identifier for this test.

        :return: short code string "KL_INT".
        """
        return "KL_INT"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Evaluate the documented truncated KL functional on fitted log data."""
        z = AbstractLogNormalGofStatistic._standardized_logs(rvs)
        coefficients = self._kl_coefficients(z)
        bound = np.max(np.abs(z))
        integral, _ = integrate.quad(
            self.do_execute_statistic,
            -5,
            5,
            args=(z, coefficients, bound, self._kl_ratio),
            points=[0],
            epsabs=1e-9,
            epsrel=1e-9,
            limit=200,
        )
        return float(len(z) * integral)

    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(r, z, coefficients, bound, ratio):
        """Compiled squared ratio; adaptive quadrature remains in SciPy."""
        value = ratio(r, z, coefficients, bound)
        return value * value

    def calculate_critical_value(self, n, alpha=0.05):
        """
        Calculate critical value for the KL Integral test.

        :param n: sample size.
        :param alpha: significance level: only 0.01 and 0.05 are implemented.
        :return: legacy regression value, not a verified finite-sample quantile.
        :raises ValueError: invalid n or alpha.
        :raises NotImplementedError: other levels require Monte Carlo calibration.

        The original regression domain and compatibility with the truncated
        statistic have not been verified. Prefer simulation at the actual n.
        """
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 2:
            raise ValueError("n must be an integer of at least two")
        if not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("alpha must lie strictly between zero and one")
        if alpha == 0.05:
            # Formula (28)
            return AbstractLogNormalGofStatistic._legacy_critical_value(
                -2.88971 - 0.130082 * (np.log(n) ** 2) + 2.71403 * np.log(n)
            )
        elif alpha == 0.01:
            # Formula (30)
            return AbstractLogNormalGofStatistic._legacy_critical_value(
                24.9099 - 31.9302 / np.sqrt(n) - 13.2562 * np.log(n) / np.sqrt(n)
            )
        else:
            raise NotImplementedError(
                "Only legacy alpha=0.01/0.05 approximations are available; "
                "calibrate other levels by Monte Carlo"
            )


# =================================================================================================
# DYNAMIC GENERATION
# =================================================================================================


_FULLY_SPECIFIED_HYPOTHESES = (
    HypothesisSupport(
        Distribution.LOG_LOCATION_SCALE,
        frozenset({Distribution.LOG_LOCATION, Distribution.LOG_SCALE}),
    ),
    HypothesisSupport(
        Distribution.SHAPE_SCALE,
        frozenset({Distribution.LOG_SCALE, Distribution.MEDIAN}),
    ),
)

_EDF_STATISTICS = {
    "ZhangWuALogNormalGofStatistic": EDFSpec(
        edf.AbstractZhangAStatistic,
        short_code="ZWA",
        supported_hypotheses=_FULLY_SPECIFIED_HYPOTHESES,
    ),
    "ZhangWuCLogNormalGofStatistic": EDFSpec(
        edf.AbstractZhangCStatistic,
        short_code="ZWC",
        supported_hypotheses=_FULLY_SPECIFIED_HYPOTHESES,
    ),
    "KolmogorovSmirnovDPlusLogNormalGofStatistic": EDFSpec(
        edf.AbstractKSDPlusStatistic, supported_hypotheses=_FULLY_SPECIFIED_HYPOTHESES
    ),
    "KolmogorovSmirnovDMinusLogNormalGofStatistic": EDFSpec(
        edf.AbstractKSDMinusStatistic, supported_hypotheses=_FULLY_SPECIFIED_HYPOTHESES
    ),
    "KolmogorovSmirnovLogNormalGofStatistic": EDFSpec(
        edf.AbstractKSStatistic,
        supported_hypotheses=_FULLY_SPECIFIED_HYPOTHESES,
        legacy_ks_options=True,
    ),
    "AndersonDarlingLogNormalGofStatistic": EDFSpec(
        edf.AbstractAndersonDarlingStatistic, supported_hypotheses=_FULLY_SPECIFIED_HYPOTHESES
    ),
    "CramerVonMiseLogNormalGofStatistic": EDFSpec(
        edf.AbstractCramerVonMisesStatistic, supported_hypotheses=_FULLY_SPECIFIED_HYPOTHESES
    ),
    "LiaoShimokawaLogNormalGofStatistic": EDFSpec(
        edf.AbstractLiaoShimokawaStatistic, supported_hypotheses=_FULLY_SPECIFIED_HYPOTHESES
    ),
    "WatsonLogNormalGofStatistic": EDFSpec(
        edf.AbstractWatsonStatistic, supported_hypotheses=_FULLY_SPECIFIED_HYPOTHESES
    ),
    "KuiperLogNormalGofStatistic": EDFSpec(
        edf.AbstractKuiperStatistic, supported_hypotheses=_FULLY_SPECIFIED_HYPOTHESES
    ),
    "KuiperScaledLogNormalGofStatistic": EDFSpec(
        edf.AbstractKuiperScaledStatistic, supported_hypotheses=_FULLY_SPECIFIED_HYPOTHESES
    ),
    "KuiperCorrectedLogNormalGofStatistic": EDFSpec(
        edf.AbstractKuiperCorrectedStatistic, supported_hypotheses=_FULLY_SPECIFIED_HYPOTHESES
    ),
    "ZhangKLogNormalGofStatistic": EDFSpec(
        edf.AbstractZhangKStatistic, supported_hypotheses=_FULLY_SPECIFIED_HYPOTHESES
    ),
    "ZhangALogNormalGofStatistic": EDFSpec(
        edf.AbstractZhangAStatistic, supported_hypotheses=_FULLY_SPECIFIED_HYPOTHESES
    ),
    "ZhangCLogNormalGofStatistic": EDFSpec(
        edf.AbstractZhangCStatistic, supported_hypotheses=_FULLY_SPECIFIED_HYPOTHESES
    ),
}

_FREQUENCY_STATISTICS = {
    "Chi2PearsonLogNormalGofStatistic": FrequencySpec(
        frequency.AbstractPearsonChi2Statistic, supported_hypotheses=_FULLY_SPECIFIED_HYPOTHESES
    ),
    "FreemanTukeyLogNormalGofStatistic": FrequencySpec(
        frequency.AbstractFreemanTukeyStatistic, supported_hypotheses=_FULLY_SPECIFIED_HYPOTHESES
    ),
    "ModifiedLogLikelihoodLogNormalGofStatistic": FrequencySpec(
        frequency.AbstractModifiedLogLikelihoodStatistic,
        supported_hypotheses=_FULLY_SPECIFIED_HYPOTHESES,
    ),
    "NeymanChi2LogNormalGofStatistic": FrequencySpec(
        frequency.AbstractNeymanChi2Statistic, supported_hypotheses=_FULLY_SPECIFIED_HYPOTHESES
    ),
    "CressieReadLogNormalGofStatistic": FrequencySpec(
        frequency.AbstractCressieReadStatistic, supported_hypotheses=_FULLY_SPECIFIED_HYPOTHESES
    ),
}


# Normality classes excluded from logarithmic wrappers.
# EDF and frequency subclasses are excluded separately and exported by the shared factories.
EXPLICITLY_IMPLEMENTED_NORMAL_STATS = [
    "KolmogorovSmirnovNormalityGofStatistic",
    "CramerVonMiseNormalityGofStatistic",
    "QuesenberryMillerNormalityGofStatistic",
    "KLSupremumLogNormalGoFStatistic",
    "KLIntegralLogNormalGoFStatistic",
    "BHSNormalityGofStatistic",
]

current_module = sys.modules[__name__]
__all__ = [
    "AbstractLogNormalGofStatistic",
    "KLIntegralLogNormalGoFStatistic",
    "KLSupremumLogNormalGoFStatistic",
    "QuesenberryMillerLogNormalGofStatistic",
]

for name, obj in inspect.getmembers(normal):
    if (
        inspect.isclass(obj)
        and issubclass(obj, AbstractNormalityGofStatistic)
        and obj is not AbstractNormalityGofStatistic
        and name not in EXPLICITLY_IMPLEMENTED_NORMAL_STATS
        and not issubclass(obj, (edf.AbstractEDFStatistic, frequency.AbstractFrequencyStatistic))
        and not name.startswith("Abstract")
        and not name.startswith("Graph")
        and name != "LillieforsNormalityGofStatistic"
    ):
        ln_class = AbstractLogNormalGofStatistic._create_lognormal_class(obj)
        setattr(current_module, ln_class.__name__, ln_class)
        __all__.append(ln_class.__name__)


__all__.extend(
    export_edf_statistics(
        globals(),
        AbstractLogNormalGofStatistic,
        _EDF_STATISTICS,
        log_probabilities=AbstractLogNormalGofStatistic._lognormal_log_probabilities,
        cdf=AbstractLogNormalGofStatistic._lognormal_cdf,
    )
)
__all__.extend(
    export_frequency_statistics(globals(), AbstractLogNormalGofStatistic, _FREQUENCY_STATISTICS)
)


globals()[
    "ZhangWuALogNormalGofStatistic"
].execute_statistic = AbstractLogNormalGofStatistic._zhang_wu_a_statistic

globals()["ZhangWuALogNormalGofStatistic"].__doc__ += """
    This historical variant returns 10 * Z_A - 32 and requires at least
    four observations. Calibration must use the same normalization.
    """

for _name in (*_EDF_STATISTICS, *_FREQUENCY_STATISTICS):
    globals()[_name].__doc__ += """

    Log-normal hypothesis
    ---------------------
    Both parameters must be fixed: LOG_LOCATION_SCALE accepts mu and s;
    SHAPE_SCALE accepts s and scale (the median). The supplied hypothesis
    is preserved; calculations use shape/median coordinates. Parameters
    are never fitted to the sample. Calibrate under this fully specified
    null, rather than using fitted-normality critical values.
    """
    if _name in _FREQUENCY_STATISTICS:
        globals()[_name].__doc__ += """
    The binning keyword accepts a BinningStrategy; the default is
    EqualProbabilityBins(). Bins are built in the original sample units.
    """
del _name


for _name in __all__:
    if _name.startswith("Abstract"):
        continue
    _cls = globals()[_name]
    if _name.startswith("KL"):
        _cls.__doc__ += """

    References
    ----------
    Batsidis, Economou and Tzavelas (2016), Tests of fit for a lognormal distribution.
    https://doi.org/10.1080/00949655.2014.1003138
    The full text was inaccessible: the documented finite-window functional is
    verified numerically, but exact original-paper equivalence is not certified.
    See docs/criteria/goodness-of-fit/log-normal/mathematical-validation.md.
    """
    elif _name == "QuesenberryMillerLogNormalGofStatistic":
        _cls.__doc__ += """

    References
    ----------
    Quesenberry and Miller (1977), Power studies of some tests for uniformity.
    https://doi.org/10.1080/00949657708810150
    Applied to a fixed log-normal CDF; all n+1 spacings and adjacent products,
    without a cyclic end-to-start product. Reject large values.
    """
    if _name == "KuiperLogNormalGofStatistic":
        _cls.__doc__ += """

    References
    ----------
    Kuiper (1960), Tests concerning random points on a circle.
    https://doi.org/10.1016/S1385-7258(60)50006-0
    """
    _fixed = not hasattr(_cls, "_standard_calibration_parameters")
    _parameters = '{"mu": 0, "s": 1}' if _fixed else "{}"
    _cls.__doc__ += f"""

    Examples
    --------
    >>> import numpy as np
    >>> from pysatl_criterion.distribution import LogNormalDistributionDescriptor as D
    >>> from pysatl_criterion.statistics.goodness_of_fit.continuous.log_normal import {_name}
    >>> statistic = {_name}(D.DEFAULT.parse({_parameters}))
    >>> sample = np.exp([-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1, 1.4, 2.1])
    >>> bool(np.isfinite(statistic.execute_statistic(sample)))
    True
    """
del _name, _cls, _fixed, _parameters
