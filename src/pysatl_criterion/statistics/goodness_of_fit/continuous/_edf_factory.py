"""Publish ordinary module-level classes backed by explicit EDF kernels."""

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from pysatl_criterion.distribution.distribution_kind import DistributionKind
from pysatl_criterion.distribution.parameters import HypothesisSupport
from pysatl_criterion.statistics.alternative import AlternativeType
from pysatl_criterion.statistics.goodness_of_fit.continuous import edf
from pysatl_criterion.statistics.goodness_of_fit.continuous.edf import AbstractEDFStatistic
from pysatl_criterion.statistics.goodness_of_fit.goodness_of_fit_statistic import (
    AbstractGoodnessOfFitStatistic,
)


@dataclass(frozen=True)
class EDFSpec:
    algorithm: type[AbstractEDFStatistic]
    short_code: str | None = None
    supported_hypotheses: tuple[HypothesisSupport, ...] | None = None
    legacy_ks_options: bool = False


class _EDFSample:
    @staticmethod
    def validate(rvs):
        if np.iscomplexobj(rvs) or (np.ma.isMaskedArray(rvs) and np.any(rvs.mask)):
            raise ValueError("Sample must be real and contain no masked observations")
        try:
            sample = np.asarray(rvs, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("Sample must contain real finite observations") from error
        if sample.ndim != 1 or sample.size == 0 or not np.all(np.isfinite(sample)):
            raise ValueError("Sample must be nonempty, one-dimensional and finite")
        return np.sort(sample)


class _LegacyKSCompatibility:
    """Preserve historical KS constructors when a family moves to generation."""

    @staticmethod
    def attributes(family_base):
        def __init__(self, parameters, *, alternative_type=AlternativeType.TWO_TAILED, mode="auto"):
            family_base.__init__(self, parameters)
            if alternative_type not in tuple(AlternativeType):
                raise ValueError("Invalid CDF deviation direction.")
            if not isinstance(mode, str) or mode not in ("auto", "exact", "asymp", "approx"):
                raise ValueError("Invalid KS mode.")
            self.alternative_type = alternative_type
            self.mode = "exact" if mode == "auto" else mode

        def _validate_storage_calibration(self):
            if self.alternative_type != AlternativeType.TWO_TAILED:
                raise ValueError("Stored KS calibration does not encode the CDF direction")

        return {
            "__init__": __init__,
            "_validate_storage_calibration": _validate_storage_calibration,
        }

    @staticmethod
    def algorithm(criterion):
        return {
            AlternativeType.RIGHT: edf.AbstractKSDPlusStatistic,
            AlternativeType.LEFT: edf.AbstractKSDMinusStatistic,
        }.get(criterion.alternative_type, edf.AbstractKSStatistic)


def _create_edf_class(name, module, family_base, spec, log_probabilities, cdf, sample_validator):
    def execute_statistic(self, rvs: ArrayLike, **kwargs) -> float:
        if sample_validator is not None:
            sample = sample_validator(rvs)
        else:
            sample = _EDFSample.validate(rvs)
        log_kernel = getattr(spec.algorithm, "do_execute_log_statistic", None)
        if cdf is not None and (log_probabilities is None or log_kernel is None):
            cdf_values = cdf(self, sample)
        elif log_probabilities is not None:
            log_cdf, log_sf = log_probabilities(self, sample)
            if log_kernel is not None:
                return float(log_kernel(log_cdf, log_sf))
            cdf_values = np.exp(log_cdf)
        else:
            cdf_values = self.distribution().cdf(sample, self._parameters)
        algorithm = spec.algorithm
        if spec.legacy_ks_options:
            algorithm = _LegacyKSCompatibility.algorithm(self)
        return float(algorithm.do_execute_statistic(cdf_values))

    attributes = {
        "__module__": module,
        "__qualname__": name,
        "__doc__": spec.algorithm.__doc__,
        "execute_statistic": execute_statistic,
    }
    if spec.supported_hypotheses is not None:
        attributes["supported_hypotheses"] = classmethod(lambda cls: spec.supported_hypotheses)
    if spec.short_code is not None:
        attributes["short_code"] = staticmethod(lambda: spec.short_code)
    if spec.legacy_ks_options:
        attributes.update(_LegacyKSCompatibility.attributes(family_base))
    cls = type(
        name,
        (family_base, spec.algorithm),
        attributes,
    )
    if inspect.isabstract(cls):
        raise TypeError(f"Unresolved abstract methods in {name}: {cls.__abstractmethods__}")
    return cls


def export_edf_statistics(
    namespace: dict[str, Any],
    family_base: type[AbstractGoodnessOfFitStatistic],
    statistics: Mapping[str, EDFSpec],
    *,
    log_probabilities: Callable | None = None,
    cdf: Callable | None = None,
    sample_validator: Callable | None = None,
) -> list[str]:
    """Export continuous EDF statistics using a CDF or explicit log probabilities.

    The family supplies its constructor and code conventions. Specs declare
    supported hypotheses; omitted declarations inherit the family contract. No
    parameters are estimated by default. Composite hypotheses require a callback
    ``log_probabilities(criterion, sorted_sample) -> (logcdf, logsf)`` which fits
    unknown parameters on each call. The family is responsible for calibration.
    Existing names are rejected; call once during module initialization.
    EDFSpec.legacy_ks_options preserves historical KS direction/mode arguments
    and the storage-calibration guard; it is opt-in for the two-sided KS kernel.
    An optional validated ``cdf(criterion, sorted_sample)`` callback avoids
    computing logarithmic tails for algorithms that only need probabilities.
    An optional sample_validator returns a validated, sorted float64 sample and
    preserves a family's stricter input contract without repeating sorting.
    """
    distribution = family_base.distribution()
    if distribution.kind() != DistributionKind.CONTINUOUS:
        raise ValueError("CDF-only EDF kernels require a continuous distribution")
    codes = set()
    for name, spec in statistics.items():
        supports = (
            family_base.supported_hypotheses()
            if spec.supported_hypotheses is None
            else spec.supported_hypotheses
        )
        if not supports:
            raise ValueError("The criterion must declare supported hypotheses")
        if log_probabilities is None and any(
            s.fixed_parameters != frozenset(s.parameterization.parameters) for s in supports
        ):
            raise ValueError("EDF generation requires fully specified hypotheses")
        if not name.isidentifier() or name in namespace:
            raise ValueError(f"Invalid or already exported name: {name}")
        if not issubclass(spec.algorithm, AbstractEDFStatistic):
            raise TypeError("An EDF algorithm is required")
        if spec.legacy_ks_options and spec.algorithm is not edf.AbstractKSStatistic:
            raise ValueError("Legacy KS options require the two-sided KS algorithm")
        short_code = spec.short_code or spec.algorithm.short_code()
        if short_code in codes:
            raise ValueError(f"Duplicate EDF code: {short_code}")
        codes.add(short_code)
    classes = {
        name: _create_edf_class(
            name, namespace["__name__"], family_base, spec, log_probabilities, cdf, sample_validator
        )
        for name, spec in statistics.items()
    }
    namespace.update(classes)
    return list(classes)
