"""Stable parameter identities and explicit distribution parameterizations.

Names are an input/output concern. Values and capability declarations use stable
identities. No implicit conversion or default filling is performed for hypotheses.
"""

import math
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TypeVar

from pysatl_criterion.distribution.distribution_type import DistributionType


@dataclass(frozen=True)
class ParameterSpec:
    """One parameter; equality and hashing depend only on its permanent ID."""

    id: str
    name: str = field(compare=False)
    display_name: str = field(compare=False)
    description: str = field(default="", compare=False)
    default: float | None = field(default=None, compare=False)
    validator: Callable[[float], bool] | None = field(default=None, compare=False)
    aliases: tuple[str, ...] = field(default=(), compare=False)

    @property
    def default_value(self) -> float:
        """Return a required default, for legacy constructor adapters."""
        if self.default is None:
            raise ValueError(f"No default for {self.name}")
        return self.default


ParameterKey = TypeVar("ParameterKey", bound=str | ParameterSpec)


@dataclass(frozen=True)
class ParameterizationDescriptor:
    """A coordinate system for a distribution, with explicitly ordered parameters."""

    id: str
    distribution: DistributionType
    parameters: tuple[ParameterSpec, ...]

    def __post_init__(self):
        ids = [parameter.id for parameter in self.parameters]
        names = [
            name for parameter in self.parameters for name in (parameter.name, *parameter.aliases)
        ]
        if len(set(ids)) != len(ids) or len(set(names)) != len(names):
            raise ValueError("Parameter IDs, names and aliases must be unique within a schema")

    def parse(
        self,
        values: Mapping[ParameterKey, float],
        *,
        fill_defaults: bool = False,
    ) -> "ParameterValues":
        """Resolve names and aliases; only fill defaults when explicitly requested."""
        if isinstance(values, ParameterValues) and values.parameterization != self:
            raise ValueError("Parameterizations require an explicit conversion")
        return ParameterValues(self, values, fill_defaults=fill_defaults)


class ParameterValues(Mapping[ParameterSpec, float]):
    """Validated, immutable values tied to one explicit parameterization."""

    def __init__(
        self,
        parameterization: ParameterizationDescriptor,
        values: Mapping[ParameterKey, float],
        *,
        fill_defaults: bool = False,
    ):
        if isinstance(values, ParameterValues) and values.parameterization != parameterization:
            raise ValueError("Parameterizations require an explicit conversion")
        by_name = {
            name: parameter
            for parameter in parameterization.parameters
            for name in (parameter.name, *parameter.aliases)
        }
        by_id = {parameter.id: parameter for parameter in parameterization.parameters}
        resolved = {}
        for key, value in values.items():
            parameter = by_id.get(key.id) if isinstance(key, ParameterSpec) else by_name.get(key)
            if parameter is None:
                raise ValueError(f"Unknown parameter {key!r} for {parameterization.id}")
            if parameter in resolved:
                raise ValueError(f"Duplicate parameter: {parameter.name}")
            resolved[parameter] = value
        if fill_defaults:
            for parameter in parameterization.parameters:
                if parameter not in resolved and parameter.default is not None:
                    resolved[parameter] = parameter.default
        for parameter, value in resolved.items():
            if not math.isfinite(value) or (
                parameter.validator is not None and not parameter.validator(value)
            ):
                raise ValueError(f"Invalid value for {parameter.name}: {value!r}")
        self._parameterization = parameterization
        self._values = MappingProxyType(resolved)

    @property
    def parameterization(self) -> ParameterizationDescriptor:
        return self._parameterization

    def __getitem__(self, parameter: ParameterSpec) -> float:
        return self._values[parameter]

    def __iter__(self) -> Iterator[ParameterSpec]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def as_dict(self) -> dict[str, float]:
        """Export current public names, never IDs or aliases."""
        return {parameter.name: self[parameter] for parameter in self}

    def storage_identity(self) -> tuple[str, tuple[tuple[str, float], ...]]:
        """Rename-independent identity including the parameterization."""
        return self.parameterization.id, tuple(sorted((p.id, self[p]) for p in self))


@dataclass(frozen=True)
class HypothesisSupport:
    """An exact supported set of fixed parameters in one parameterization."""

    parameterization: ParameterizationDescriptor
    fixed_parameters: frozenset[ParameterSpec]

    def __post_init__(self):
        if not self.fixed_parameters <= frozenset(self.parameterization.parameters):
            raise ValueError("Supported parameters must belong to the parameterization")

    def supports(self, values: ParameterValues) -> bool:
        return (
            values.parameterization == self.parameterization
            and frozenset(values) == self.fixed_parameters
        )
