from __future__ import annotations

from abc import ABC

import numpy as np
import scipy.stats as scipy_stats
from numba import jit
from typing_extensions import override

from pysatl_criterion.distribution import InverseGammaDistributionDescriptor
from pysatl_criterion.distribution import (
    InverseGammaDistributionDescriptor as Distribution,
)
from pysatl_criterion.distribution.parameters import HypothesisSupport
from pysatl_criterion.statistics.alternative import (
    Alternative,
    RightAlternative,
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


class AbstractInverseGammaGofStatistic(AbstractGoodnessOfFitStatistic, ABC):
    """Shared parameter access and validation for Inverse Gamma statistics."""

    @property
    def alpha(self) -> float:
        """Read alpha by its stable parameter identity."""
        return self._parameters[Distribution.SHAPE]

    @property
    def beta(self) -> float:
        """Read beta by its stable parameter identity."""
        return self._parameters[Distribution.SCALE]

    @staticmethod
    def _prepare_sample(rvs, minimum=1):
        sample = np.asarray(rvs)
        if sample.ndim != 1 or sample.dtype.kind not in "iuf":
            raise ValueError("Sample must be a one-dimensional real numeric array")
        if sample.size < minimum:
            raise ValueError(f"At least {minimum} observations are required")
        sample = np.asarray(sample, dtype=float)
        if not np.all(np.isfinite(sample)) or np.any(sample <= 0):
            raise ValueError("Observations must be finite and strictly positive for Inverse Gamma")
        return np.sort(sample)

    _validate_input = _prepare_sample

    def _cdf(self, sample):
        values = scipy_stats.invgamma.cdf(sample, a=self.alpha, scale=self.beta)
        if (
            not np.all(np.isfinite(values))
            or np.any(values < 0)
            or np.any(values > 1)
            or np.any(np.diff(values) < 0)
        ):
            raise FloatingPointError("Invalid Inverse Gamma CDF values")
        return values

    def _log_probabilities(self, sample):
        log_cdf = scipy_stats.invgamma.logcdf(sample, a=self.alpha, scale=self.beta)
        log_sf = scipy_stats.invgamma.logsf(sample, a=self.alpha, scale=self.beta)
        if (
            not np.all(np.isfinite(log_cdf))
            or not np.all(np.isfinite(log_sf))
            or np.any(log_cdf > 0)
            or np.any(log_sf > 0)
        ):
            raise FloatingPointError("Inverse Gamma log tails exceed numerical precision")
        return log_cdf, log_sf

    @staticmethod
    @override
    def distribution() -> type[InverseGammaDistributionDescriptor]:
        """Return the distribution descriptor class."""
        return InverseGammaDistributionDescriptor

    @classmethod
    @override
    def code(cls) -> str:
        """Return the family identifier or the concrete statistic's full identifier."""
        family_code = f"INV_GAMMA_{AbstractGoodnessOfFitStatistic.code()}"
        if "short_code" in cls.__abstractmethods__:
            return family_code
        return f"{cls.short_code()}_{family_code}"


class GreenwoodInverseGammaGofStatistic(AbstractInverseGammaGofStatistic):
    """Greenwood sum of squared spacings after the fixed CDF.

    Parameters
    ----------
    parameters : ParameterValues
        Values with alpha, beta fixed; omitted parameters are unknown.

    Methods
    -------
    execute_statistic(rvs, **kwargs)
        Return one scalar without modifying the input or retaining estimates.
    hypothesis()
        Return fixed parameters; omitted parameters are unknown.
    alternative()
        Return the right critical tail.

    Notes
    -----
    The iid null model has density beta**alpha/Gamma(alpha) *
    x**(-alpha-1)*exp(-beta/x), x > 0, with location fixed at zero.
    Write u_i=F(x_(i)), i=1,...,n, for sorted observations.
    G = sum((u_(i+1)-u_i)**2), i=0,...,n, with u_0=0 and u_(n+1)=1.
    Include both endpoint gaps. This is the unscaled sum, with null mean
    2/(n+2); large values detect clustering. No normal approximation is used.

    Both parameters are fixed; no fitting is performed. Under the continuous
    null the probability transform removes alpha and beta from the null law,
    which still depends on n. Large values reject. Ties and constants are
    accepted; rounded data need separate calibration. The built-in Monte
    Carlo resolver has no Inverse Gamma generator; simulate externally.
    Finite positive observations give mathematically finite statistics.
    Numerical log-tail underflow or overflow raises FloatingPointError,
    rather than replacing probabilities with epsilon. Ordinary CDF values
    and spacings may still round in extreme tails.
    The reference concerns the general statistic, applied here through the
    fixed CDF or cell probabilities, not an Inverse Gamma fitted test.

    References
    ----------
    .. [1] Greenwood, M. (1946). The statistical study of infectious diseases.
       J. R. Stat. Soc. 109, 85-110, discussion of random intervals, pp. 99-101.
       https://www.ime.usp.br/~abe/lista/pdfr62zHOGxek.pdf

    Examples
    --------
    >>> parameters = Distribution.DEFAULT.parse({'alpha': 3.0, 'beta': 2.0})
    >>> statistic = GreenwoodInverseGammaGofStatistic(parameters)
    >>> value = statistic.execute_statistic([0.4, 0.7, 1.2, 2.0])
    >>> bool(np.isfinite(value))
    True
    """

    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]:
        return (
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.SCALE})
            ),
        )

    @override
    def alternative(self) -> Alternative:
        return RightAlternative()

    @staticmethod
    @override
    def short_code():
        return "GRW"

    @override
    def execute_statistic(self, rvs, **kwargs):
        """Compute the statistic defined in the class Notes.

        Parameters
        ----------
        rvs : array_like, shape (n,)
            Finite real strictly positive observations. At least one value;
            ties and constant samples are allowed.
        **kwargs : dict
            Unused compatibility arguments.

        Returns
        -------
        float or numpy.float64
            Scalar statistic; large values reject the null.

        Raises
        ------
        ValueError
            If sample dimension, size, finiteness or support is invalid.
        FloatingPointError
            If CDF evaluation, required log tails or quantiles are numerically
            invalid, or the statistic exceeds float64 range.

        Notes
        -----
        No mathematically infinite result occurs for admissible finite data.
        No p-value or finite-sample correction is returned.
        """

        sorted_rvs = self._prepare_sample(rvs)
        cdf_vals = self._cdf(sorted_rvs)
        return self.do_execute_statistic(cdf_vals)

    @staticmethod
    @jit(nopython=True, cache=True)
    def do_execute_statistic(cdf_vals) -> float:
        """Sum squared spacings of validated, sorted CDF values, including endpoints."""
        total = 0.0
        previous = 0.0
        for value in cdf_vals:
            spacing = value - previous
            total += spacing * spacing
            previous = value
        final_spacing = 1.0 - previous
        return total + final_spacing * final_spacing


_EDF_STATISTICS = {
    "KolmogorovSmirnovInverseGammaGofStatistic": EDFSpec(
        edf.AbstractKSStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.SCALE})
            ),
        ),
        legacy_ks_options=True,
    ),
    "KolmogorovSmirnovDPlusInverseGammaGofStatistic": EDFSpec(
        edf.AbstractKSDPlusStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.SCALE})
            ),
        ),
    ),
    "KolmogorovSmirnovDMinusInverseGammaGofStatistic": EDFSpec(
        edf.AbstractKSDMinusStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.SCALE})
            ),
        ),
    ),
    "AndersonDarlingInverseGammaGofStatistic": EDFSpec(
        edf.AbstractAndersonDarlingStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.SCALE})
            ),
        ),
    ),
    "CramerVonMisesInverseGammaGofStatistic": EDFSpec(
        edf.AbstractCramerVonMisesStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.SCALE})
            ),
        ),
    ),
    "WatsonInverseGammaGofStatistic": EDFSpec(
        edf.AbstractWatsonStatistic,
        short_code="WAT",
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.SCALE})
            ),
        ),
    ),
    "KuiperInverseGammaGofStatistic": EDFSpec(
        edf.AbstractKuiperStatistic,
        short_code="KUI_INVGAMMA",
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.SCALE})
            ),
        ),
    ),
    "KuiperScaledInverseGammaGofStatistic": EDFSpec(
        edf.AbstractKuiperScaledStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.SCALE})
            ),
        ),
    ),
    "KuiperCorrectedInverseGammaGofStatistic": EDFSpec(
        edf.AbstractKuiperCorrectedStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.SCALE})
            ),
        ),
    ),
    "LiaoShimokawaInverseGammaGofStatistic": EDFSpec(
        edf.AbstractLiaoShimokawaStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.SCALE})
            ),
        ),
    ),
    "ZhangAInverseGammaGofStatistic": EDFSpec(
        edf.AbstractZhangAStatistic,
        short_code="ZAA",
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.SCALE})
            ),
        ),
    ),
    "ZhangCInverseGammaGofStatistic": EDFSpec(
        edf.AbstractZhangCStatistic,
        short_code="ZAC",
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.SCALE})
            ),
        ),
    ),
    "ZhangKInverseGammaGofStatistic": EDFSpec(
        edf.AbstractZhangKStatistic,
        short_code="ZAK",
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.SCALE})
            ),
        ),
    ),
}

_FREQUENCY_STATISTICS = {
    "Chi2PearsonInverseGammaGofStatistic": FrequencySpec(
        frequency.AbstractPearsonChi2Statistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.SCALE})
            ),
        ),
    ),
    "FreemanTukeyInverseGammaGofStatistic": FrequencySpec(
        frequency.AbstractFreemanTukeyStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.SCALE})
            ),
        ),
    ),
    "ModifiedLogLikelihoodInverseGammaGofStatistic": FrequencySpec(
        frequency.AbstractModifiedLogLikelihoodStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.SCALE})
            ),
        ),
    ),
    "NeymanChi2InverseGammaGofStatistic": FrequencySpec(
        frequency.AbstractNeymanChi2Statistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.SCALE})
            ),
        ),
    ),
    "CressieReadInverseGammaGofStatistic": FrequencySpec(
        frequency.AbstractCressieReadStatistic,
        supported_hypotheses=(
            HypothesisSupport(
                Distribution.DEFAULT, frozenset({Distribution.SHAPE, Distribution.SCALE})
            ),
        ),
    ),
}

__all__ = [
    "AbstractInverseGammaGofStatistic",
    "GreenwoodInverseGammaGofStatistic",
]
__all__.extend(
    export_edf_statistics(
        globals(),
        AbstractInverseGammaGofStatistic,
        _EDF_STATISTICS,
        log_probabilities=AbstractInverseGammaGofStatistic._log_probabilities,
        cdf=AbstractInverseGammaGofStatistic._cdf,
        sample_validator=AbstractInverseGammaGofStatistic._prepare_sample,
    )
)
__all__.extend(
    export_frequency_statistics(
        globals(),
        AbstractInverseGammaGofStatistic,
        _FREQUENCY_STATISTICS,
    )
)
