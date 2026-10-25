"""Generate frequency criteria with a shared, configurable binning pipeline."""

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from pysatl_criterion.distribution.parameters import HypothesisSupport, ParameterValues
from pysatl_criterion.statistics.goodness_of_fit.binning import (
    BinningStrategy,
    EqualProbabilityBins,
)
from pysatl_criterion.statistics.goodness_of_fit.binning.base import FrequencyTable, validate_sample
from pysatl_criterion.statistics.goodness_of_fit.continuous.frequency import (
    AbstractFrequencyStatistic,
)
from pysatl_criterion.statistics.goodness_of_fit.goodness_of_fit_statistic import (
    AbstractGoodnessOfFitStatistic,
)


@dataclass(frozen=True)
class FrequencySpec:
    algorithm: type[AbstractFrequencyStatistic]
    supported_hypotheses: tuple[HypothesisSupport, ...] | None = None


def _frequency_configuration(self) -> dict:
    from pysatl_criterion.statistics.configuration import normalize_configuration

    serialize = getattr(self.binning, "to_config", None)
    if not callable(serialize):
        raise TypeError("Binning must provide stable to_config() for storage calibration")
    config = serialize()
    if not isinstance(config, dict):
        raise TypeError("Binning to_config() must return a dictionary")
    return normalize_configuration(
        {
            "schema_version": 1,
            "statistic_version": 1,
            "binning": config,
        }
    )


def _create_frequency_class(name, module, family_base, spec, bind_distribution):
    def __init__(self, parameters: ParameterValues, *, binning: BinningStrategy | None = None):
        family_base.__init__(self, parameters)
        self.binning = EqualProbabilityBins() if binning is None else binning
        if not callable(getattr(self.binning, "build", None)) or not callable(
            getattr(self.binning, "supports", None)
        ):
            raise TypeError("binning must implement build() and supports()")
        if not self.binning.supports(self.distribution().kind()):
            raise ValueError("Binning strategy does not support this distribution kind")

    def execute_statistic(self, rvs, **kwargs) -> float:
        sample = validate_sample(rvs)
        # Distribution-family validation includes support membership, when available.
        validate = getattr(self, "_validate_input", None)
        if validate is not None:
            sample = validate(sample)
        bound = (
            self.distribution().bind(self._parameters)
            if bind_distribution is None
            else bind_distribution(self, sample)
        )
        table = self.binning.build(sample, bound)
        if not isinstance(table, FrequencyTable):
            raise TypeError("Binning strategy must return FrequencyTable")
        if abs(table.observed.sum() - len(sample)) > 1e-8:
            raise ValueError("Binning strategy lost observations")
        return float(spec.algorithm.do_execute_statistic(table.observed, table.expected))

    def _validate_storage_calibration(self):
        if bind_distribution is not None:
            family_base._validate_storage_calibration(self)
        self.configuration()

    def configuration(self):
        config = family_base.configuration(self)
        return {**config, "binning": _frequency_configuration(self)["binning"]}

    cls = type(
        name,
        (family_base, spec.algorithm),
        {
            "__module__": module,
            "__qualname__": name,
            "__doc__": spec.algorithm.__doc__,
            "__init__": __init__,
            "execute_statistic": execute_statistic,
            "_validate_storage_calibration": _validate_storage_calibration,
            "configuration": configuration,
            **(
                {"supported_hypotheses": classmethod(lambda cls: spec.supported_hypotheses)}
                if spec.supported_hypotheses is not None
                else {}
            ),
        },
    )
    if inspect.isabstract(cls):
        raise TypeError(f"Unresolved abstract methods in {name}: {cls.__abstractmethods__}")
    return cls


def export_frequency_statistics(
    namespace: dict[str, Any],
    family_base: type[AbstractGoodnessOfFitStatistic],
    statistics: Mapping[str, FrequencySpec],
    *,
    bind_distribution: Callable | None = None,
) -> list[str]:
    """Export frequency kernels, optionally binding a fitted distribution.

    Composite hypotheses require ``bind_distribution(criterion, sample)`` to fit
    on every call and return a BoundDistribution in the original sample units.
    Their calibration constraints remain the family base's responsibility.
    """
    codes = set()
    for name, spec in statistics.items():
        supports = (
            family_base.supported_hypotheses()
            if spec.supported_hypotheses is None
            else spec.supported_hypotheses
        )
        if not supports or (
            bind_distribution is None
            and any(
                s.fixed_parameters != frozenset(s.parameterization.parameters) for s in supports
            )
        ):
            raise ValueError("Frequency generation requires fully specified hypotheses")
        if not name.isidentifier() or name in namespace:
            raise ValueError(f"Invalid or already exported name: {name}")
        if not issubclass(spec.algorithm, AbstractFrequencyStatistic):
            raise TypeError("A frequency algorithm is required")
        code = spec.algorithm.short_code()
        if code in codes:
            raise ValueError(f"Duplicate frequency code: {code}")
        codes.add(code)
    classes = {
        name: _create_frequency_class(
            name, namespace["__name__"], family_base, spec, bind_distribution
        )
        for name, spec in statistics.items()
    }
    namespace.update(classes)
    return list(classes)
