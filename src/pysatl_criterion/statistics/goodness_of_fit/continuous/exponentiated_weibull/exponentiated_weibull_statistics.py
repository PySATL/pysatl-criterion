"""Specified exponentiated Weibull criteria with fixed exponent, shape and scale.

F(x) = (1 - exp(-(x / scale)**shape))**exponent, x >= 0; location is zero.
"""

from abc import ABC

import numpy as np
from numba import jit

from pysatl_criterion.distribution import (
    ExponentiatedWeibullDistributionDescriptor as Distribution,
)
from pysatl_criterion.distribution.parameters import HypothesisSupport
from pysatl_criterion.statistics.goodness_of_fit.continuous import edf, frequency
from pysatl_criterion.statistics.goodness_of_fit.continuous._edf_factory import (
    EDFSpec,
    export_edf_statistics,
)
from pysatl_criterion.statistics.goodness_of_fit.continuous._frequency_factory import (
    FrequencySpec,
    export_frequency_statistics,
)
from pysatl_criterion.statistics.goodness_of_fit.continuous._weibull_common import (
    _sample,
)
from pysatl_criterion.statistics.goodness_of_fit.goodness_of_fit_statistic import (
    AbstractGoodnessOfFitStatistic,
)


class AbstractExponentiatedWeibullGofStatistic(AbstractGoodnessOfFitStatistic, ABC):
    """Fully specified exponentiated Weibull on the nonnegative half-line."""

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

    @staticmethod
    def _validate_input(rvs):
        if np.ma.isMaskedArray(rvs) and np.any(rvs.mask):
            raise ValueError("Sample must contain no masked observations")
        return _sample(rvs)

    @staticmethod
    def _sorted_sample(rvs):
        return np.sort(AbstractExponentiatedWeibullGofStatistic._validate_input(rvs))

    def _log_probabilities(self, sample):
        return self.do_log_probabilities(sample, self.exponent, self.shape, self.scale)

    def _cdf(self, sample):
        log_cdf, _ = self.do_log_probabilities(
            sample, self.exponent, self.shape, self.scale, compute_sf=False
        )
        return np.exp(log_cdf)

    def _bind_distribution(self, sample):
        return _ExponentiatedWeibullBoundDistribution(self)

    @staticmethod
    @jit(nopython=True, cache=True)
    def do_log_probabilities(sample, exponent, shape, scale, compute_sf=True):
        """Evaluate both logarithmic tails in one pass without temporary masks.

        Input is a flat float64 array on [0, infinity]; NaNs propagate.
        With compute_sf=False, skip survival work and return an empty second array.
        Preserve finite logarithmic tails even when F rounds to one or the
        intermediate shape * log(x / scale) overflows in the lower tail.
        """
        log_cdf = np.empty(len(sample), dtype=np.float64)
        log_sf = np.empty(len(sample) if compute_sf else 0, dtype=np.float64)
        log_scale = np.log(scale)
        log_exponent = np.log(exponent)
        tiny_log_cdf = -np.exp(-36.0)
        for i in range(len(sample)):
            x = sample[i]
            if x == 0.0:
                log_cdf[i] = -np.inf
                if compute_sf:
                    log_sf[i] = 0.0
                continue
            log_ratio = np.log(x) - log_scale
            z = shape * log_ratio
            if z < -36.0:
                if np.isneginf(z):
                    lc = (exponent * shape) * log_ratio
                    if compute_sf and lc > tiny_log_cdf:
                        log_q = np.log(-lc)
                else:
                    lc = exponent * z
                    if compute_sf and lc > tiny_log_cdf:
                        log_q = log_exponent + np.log(-z)
            else:
                t = np.exp(z)
                if t > 36.0:
                    log_q = log_exponent - t
                    lc = -np.exp(log_q)
                else:
                    base_log_cdf = np.log(-np.expm1(-t))
                    lc = exponent * base_log_cdf
                    if compute_sf and lc > tiny_log_cdf:
                        log_q = log_exponent + np.log(-base_log_cdf)
            log_cdf[i] = lc
            if compute_sf:
                if lc > tiny_log_cdf:
                    log_sf[i] = log_q
                else:
                    log_sf[i] = np.log(-np.expm1(lc))
        return log_cdf, log_sf


class _ExponentiatedWeibullBoundDistribution:
    """Use the same stable tails for EDF statistics and frequency binning."""

    def __init__(self, criterion):
        self.criterion = criterion

    def _logs(self, x):
        sample = np.maximum(np.asarray(x, dtype=float), 0.0)
        logs = self.criterion._log_probabilities(sample.reshape(-1))
        return tuple(value.reshape(sample.shape) for value in logs)

    def logcdf(self, x):
        result = self._logs(x)[0]
        return float(result) if result.ndim == 0 else result

    def logsf(self, x):
        result = self._logs(x)[1]
        return float(result) if result.ndim == 0 else result

    def cdf(self, x):
        sample = np.maximum(np.asarray(x, dtype=float), 0.0)
        result = self.criterion._cdf(sample.reshape(-1)).reshape(sample.shape)
        return float(result) if result.ndim == 0 else result

    def sf(self, x):
        return np.exp(self.logsf(x))


_EDF_STATISTICS = {
    "KolmogorovSmirnovDPlusExponentiatedWeibullGofStatistic": EDFSpec(
        edf.AbstractKSDPlusStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        ),
    ),
    "KolmogorovSmirnovDMinusExponentiatedWeibullGofStatistic": EDFSpec(
        edf.AbstractKSDMinusStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        ),
    ),
    "KolmogorovSmirnovExponentiatedWeibullGofStatistic": EDFSpec(
        edf.AbstractKSStatistic,
        legacy_ks_options=True,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        ),
    ),
    "AndersonDarlingExponentiatedWeibullGofStatistic": EDFSpec(
        edf.AbstractAndersonDarlingStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        ),
    ),
    "CrammerVonMisesExponentiatedWeibullGofStatistic": EDFSpec(
        edf.AbstractCramerVonMisesStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        ),
    ),
    "MinToshiyukiExponentiatedWeibullGofStatistic": EDFSpec(
        edf.AbstractLiaoShimokawaStatistic,
        short_code="MT",
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        ),
    ),
    "WatsonExponentiatedWeibullGofStatistic": EDFSpec(
        edf.AbstractWatsonStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        ),
    ),
    "KuiperExponentiatedWeibullGofStatistic": EDFSpec(
        edf.AbstractKuiperStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        ),
    ),
    "KuiperScaledExponentiatedWeibullGofStatistic": EDFSpec(
        edf.AbstractKuiperScaledStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        ),
    ),
    "KuiperCorrectedExponentiatedWeibullGofStatistic": EDFSpec(
        edf.AbstractKuiperCorrectedStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        ),
    ),
    "ZhangKExponentiatedWeibullGofStatistic": EDFSpec(
        edf.AbstractZhangKStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        ),
    ),
    "ZhangAExponentiatedWeibullGofStatistic": EDFSpec(
        edf.AbstractZhangAStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        ),
    ),
    "ZhangCExponentiatedWeibullGofStatistic": EDFSpec(
        edf.AbstractZhangCStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        ),
    ),
}

_FREQUENCY_STATISTICS = {
    "Chi2PearsonExponentiatedWeibullGofStatistic": FrequencySpec(
        frequency.AbstractPearsonChi2Statistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        ),
    ),
    "FreemanTukeyExponentiatedWeibullGofStatistic": FrequencySpec(
        frequency.AbstractFreemanTukeyStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        ),
    ),
    "ModifiedLogLikelihoodExponentiatedWeibullGofStatistic": FrequencySpec(
        frequency.AbstractModifiedLogLikelihoodStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        ),
    ),
    "NeymanChi2ExponentiatedWeibullGofStatistic": FrequencySpec(
        frequency.AbstractNeymanChi2Statistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        ),
    ),
    "CressieReadExponentiatedWeibullGofStatistic": FrequencySpec(
        frequency.AbstractCressieReadStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        ),
    ),
}

__all__ = ["AbstractExponentiatedWeibullGofStatistic"]
__all__.extend(
    export_edf_statistics(
        globals(),
        AbstractExponentiatedWeibullGofStatistic,
        _EDF_STATISTICS,
        log_probabilities=AbstractExponentiatedWeibullGofStatistic._log_probabilities,
        cdf=AbstractExponentiatedWeibullGofStatistic._cdf,
        sample_validator=AbstractExponentiatedWeibullGofStatistic._sorted_sample,
    )
)
__all__.extend(
    export_frequency_statistics(
        globals(),
        AbstractExponentiatedWeibullGofStatistic,
        _FREQUENCY_STATISTICS,
        bind_distribution=AbstractExponentiatedWeibullGofStatistic._bind_distribution,
    )
)

# Preserve the family's calibration guard with the legacy KS constructor.
globals()[
    "KolmogorovSmirnovExponentiatedWeibullGofStatistic"
]._validate_storage_calibration = (
    AbstractExponentiatedWeibullGofStatistic._validate_storage_calibration
)

for _name in (*_EDF_STATISTICS, *_FREQUENCY_STATISTICS):
    globals()[_name].__doc__ += """

    Exponentiated Weibull hypothesis
    -------------------------------
    Use Distribution.DEFAULT.parse({"exponent": a, "shape": k, "scale": s})
    with all three positive parameters fixed and location zero. Samples must
    be finite, nonnegative and nonempty. Parameters are never fitted.
    Calibrate using the specified null and the same sample size.
    Unversioned stored calibrations must be regenerated.
    """
    globals()[_name].__doc__ += f"""

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({{"exponent": 1, "shape": 2, "scale": 1}})
    >>> test = {_name}(parameters)
    >>> value = test.execute_statistic([0.1, 0.25, 0.4, 0.55, 0.7, 0.9, 1.2, 1.5, 2.0])
    >>> bool(np.isfinite(value))
    True
    """
    if _name in _FREQUENCY_STATISTICS:
        globals()[_name].__doc__ += """

    Binning
    -------
    The keyword-only binning argument accepts a BinningStrategy. The default
    uses equal-probability cells with min(1000, max(2, ceil(2*n**0.4))) bins.
    Even n=1 uses two cells. Repeat the binning procedure during calibration;
    small expected counts do not justify a chi-square approximation.
    """
del _name
