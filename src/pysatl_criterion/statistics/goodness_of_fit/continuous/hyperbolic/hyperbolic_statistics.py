from __future__ import annotations

from abc import ABC

import numpy as np
import scipy.stats as scipy_stats
from typing_extensions import override

from pysatl_criterion.distribution import HyperbolicDistributionDescriptor
from pysatl_criterion.distribution import (
    HyperbolicDistributionDescriptor as Distribution,
)
from pysatl_criterion.distribution.parameters import HypothesisSupport, ParameterValues
from pysatl_criterion.statistics import AbstractGoodnessOfFitStatistic
from pysatl_criterion.statistics.goodness_of_fit.continuous import edf, frequency
from pysatl_criterion.statistics.goodness_of_fit.continuous._edf_factory import (
    EDFSpec,
    export_edf_statistics,
)
from pysatl_criterion.statistics.goodness_of_fit.continuous._frequency_factory import (
    FrequencySpec,
    export_frequency_statistics,
)


class AbstractHyperbolicGofStatistic(AbstractGoodnessOfFitStatistic, ABC):
    """Base class for hyperbolic distribution goodness-of-fit statistics.

    The classical hyperbolic distribution is the generalized hyperbolic
    distribution with tail parameter ``p = 1``. The parameterization follows
    Barndorff-Nielsen (1978), *Hyperbolic Distributions and Distributions on
    Hyperbolae*, Scandinavian Journal of Statistics, 5(3), 151--157.
    """

    @property
    def alpha(self) -> float:
        """Read alpha by its stable parameter identity."""
        return self._parameters[Distribution.SHAPE]

    @property
    def beta(self) -> float:
        """Read beta by its stable parameter identity."""
        return self._parameters[Distribution.SKEWNESS]

    @property
    def delta(self) -> float:
        """Read delta by its stable parameter identity."""
        return self._parameters[Distribution.SCALE]

    @property
    def mu(self) -> float:
        """Read mu by its stable parameter identity."""
        return self._parameters[Distribution.LOCATION]

    def __init__(self, parameters: ParameterValues) -> None:
        AbstractGoodnessOfFitStatistic.__init__(self, parameters)
        if abs(self.beta) >= self.alpha:
            raise ValueError("Beta must satisfy abs(beta) < alpha.")
        a, b = (self.alpha * self.delta, self.beta * self.delta)
        if not np.isfinite(a) or not np.isfinite(b) or a <= abs(b):
            raise ValueError(
                "Scaled parameters must be finite and satisfy alpha*delta > |beta*delta|."
            )

    def __getstate__(self):
        """Serialize fixed parameters without ParameterValues' mapping proxies."""
        state = self.__dict__.copy()
        for key in ("_parameters", "_hypothesis_parameters"):
            state[key] = state[key].as_dict()
        return state

    def __setstate__(self, state):
        """Restore parameters in the sole supported classical parameterization."""
        state = state.copy()
        for key in ("_parameters", "_hypothesis_parameters"):
            if isinstance(state[key], dict):
                state[key] = Distribution.DEFAULT.parse(state[key])
        self.__dict__.update(state)

    @staticmethod
    @override
    def distribution() -> type[HyperbolicDistributionDescriptor]:
        """Return the distribution descriptor class."""
        return HyperbolicDistributionDescriptor

    @classmethod
    @override
    def code(cls) -> str:
        """Return the family identifier or the concrete statistic's full identifier."""
        family_code = f"HYPERBOLIC_{AbstractGoodnessOfFitStatistic.code()}"
        if "short_code" in cls.__abstractmethods__:
            return family_code
        return f"{cls.short_code()}_{family_code}"

    def _cdf(self, sorted_rvs: np.ndarray) -> np.ndarray:
        """Evaluate the hyperbolic CDF for a sorted sample.

        :param sorted_rvs: sample sorted in ascending order.
        :return: CDF values in sample order.
        """
        values = np.asarray(
            scipy_stats.genhyperbolic.cdf(
                sorted_rvs,
                p=1.0,
                a=self.alpha * self.delta,
                b=self.beta * self.delta,
                loc=self.mu,
                scale=self.delta,
            ),
            dtype=np.float64,
        )
        if (
            not np.all(np.isfinite(values))
            or np.any(values < 0)
            or np.any(values > 1)
            or np.any(np.diff(values) < 0)
        ):
            raise FloatingPointError("Hyperbolic CDF evaluation failed.")
        return values

    def _log_probabilities(self, sorted_rvs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate stable log-CDF and log-survival probabilities.

        :param sorted_rvs: sample sorted in ascending order.
        :return: pair containing log-CDF and log-survival arrays.
        """
        parameters = {
            "p": 1.0,
            "a": self.alpha * self.delta,
            "b": self.beta * self.delta,
            "loc": self.mu,
            "scale": self.delta,
        }
        log_cdf = scipy_stats.genhyperbolic.logcdf(sorted_rvs, **parameters)
        log_sf = scipy_stats.genhyperbolic.logsf(sorted_rvs, **parameters)
        if (
            not np.all(np.isfinite(log_cdf))
            or not np.all(np.isfinite(log_sf))
            or np.any(log_cdf > 0)
            or np.any(log_sf > 0)
        ):
            raise FloatingPointError("Hyperbolic log-tail evaluation failed or underflowed.")
        return (np.asarray(log_cdf, dtype=np.float64), np.asarray(log_sf, dtype=np.float64))


_EDF_STATISTICS = {
    "KolmogorovSmirnovHyperbolicGofStatistic": EDFSpec(
        edf.AbstractKSStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        ),
    ),
    "CramerVonMisesHyperbolicGofStatistic": EDFSpec(
        edf.AbstractCramerVonMisesStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        ),
    ),
    "AndersonDarlingHyperbolicGofStatistic": EDFSpec(
        edf.AbstractAndersonDarlingStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        ),
    ),
    "KuiperHyperbolicGofStatistic": EDFSpec(
        edf.AbstractKuiperStatistic,
        short_code="KUI",
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        ),
    ),
    "WatsonHyperbolicGofStatistic": EDFSpec(
        edf.AbstractWatsonStatistic,
        short_code="WAT",
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        ),
    ),
}

_FREQUENCY_STATISTICS = {
    "Chi2PearsonHyperbolicGofStatistic": FrequencySpec(
        frequency.AbstractPearsonChi2Statistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        ),
    ),
    "FreemanTukeyHyperbolicGofStatistic": FrequencySpec(
        frequency.AbstractFreemanTukeyStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        ),
    ),
    "ModifiedLogLikelihoodHyperbolicGofStatistic": FrequencySpec(
        frequency.AbstractModifiedLogLikelihoodStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        ),
    ),
    "NeymanChi2HyperbolicGofStatistic": FrequencySpec(
        frequency.AbstractNeymanChi2Statistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        ),
    ),
    "CressieReadHyperbolicGofStatistic": FrequencySpec(
        frequency.AbstractCressieReadStatistic,
        supported_hypotheses=(
            HypothesisSupport(Distribution.DEFAULT, frozenset(Distribution.DEFAULT.parameters)),
        ),
    ),
}

__all__ = ["AbstractHyperbolicGofStatistic"]
__all__.extend(
    export_edf_statistics(
        globals(),
        AbstractHyperbolicGofStatistic,
        _EDF_STATISTICS,
        cdf=AbstractHyperbolicGofStatistic._cdf,
        log_probabilities=AbstractHyperbolicGofStatistic._log_probabilities,
    )
)

__all__.extend(
    export_frequency_statistics(globals(), AbstractHyperbolicGofStatistic, _FREQUENCY_STATISTICS)
)

for _name in (*_EDF_STATISTICS, *_FREQUENCY_STATISTICS):
    globals()[_name].__doc__ += """

    Hyperbolic hypothesis
    --------------------
    All four classical parameters alpha, beta, delta, mu must be fixed.
    Require alpha > abs(beta), delta > 0 and finite scaled shapes.
    Samples must be nonempty, one-dimensional, real and finite.
    No parameters are fitted. Ties and constant samples are accepted.
    Calibrate for the sample size under a fully specified continuous iid null.
    Fitted or rounded models require their own calibration.
    CDF and log-tail integration uses SciPy; numerical failures are rejected.
    AD uses direct log-CDF/log-SF without clipping.
    """
    if _name in _FREQUENCY_STATISTICS:
        globals()[_name].__doc__ += """
    Binning defaults to EqualProbabilityBins(); pass binning= to select a strategy.
    Expected counts are n times the fixed null probabilities. Empty observed
    cells yield infinity for modified log likelihood and Neyman chi-square.
    Calibration must reproduce binning, count selection and merging. Fixed cells
    with positive probabilities admit the usual asymptotic chi-square(k-1) law;
    small expected counts and adaptive bins require simulation. The built-in
    resolver has no hyperbolic generator; use external calibration.
    """
del _name
