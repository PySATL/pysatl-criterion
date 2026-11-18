from dataclasses import dataclass
from decimal import Decimal
from math import isfinite

from pysatl_criterion.persistence.models.distribution_key import DistributionKey
from pysatl_criterion.statistics.alternative import AlternativeType


def normalize_significance_level(value: Decimal) -> Decimal:
    """Validate the exact precision supported by Numeric(12, 10)."""
    value = Decimal(str(value))
    if not value.is_finite() or not 0 < value < 1:
        raise ValueError("significance_level must be between 0 and 1")
    normalized = value.quantize(Decimal("0.0000000001"))
    if normalized != value:
        raise ValueError("significance_level supports at most 10 decimal places")
    return normalized


@dataclass(frozen=True)
class CriticalValueKey:
    """Identity of cached bounds, independent of the source simulation count."""

    distribution_key: DistributionKey
    significance_level: Decimal
    alternative_type: AlternativeType

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "significance_level", normalize_significance_level(self.significance_level)
        )


@dataclass(frozen=True)
class CriticalValueModel:
    distribution_key: DistributionKey
    monte_carlo_count: int
    significance_level: Decimal
    alternative_type: AlternativeType
    lower_value: float | None = None
    upper_value: float | None = None

    @property
    def key(self) -> CriticalValueKey:
        return CriticalValueKey(
            self.distribution_key, self.significance_level, self.alternative_type
        )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "significance_level", normalize_significance_level(self.significance_level)
        )
        if self.monte_carlo_count <= 0:
            raise ValueError("monte_carlo_count must be positive")
        required = {
            AlternativeType.LEFT: (True, False),
            AlternativeType.RIGHT: (False, True),
            AlternativeType.TWO_TAILED: (True, True),
        }
        if (self.lower_value is not None, self.upper_value is not None) != required.get(
            self.alternative_type
        ):
            raise ValueError("Critical value bounds must match alternative_type")
        if any(v is not None and not isfinite(v) for v in (self.lower_value, self.upper_value)):
            raise ValueError("Critical value bounds must be finite")
        if (
            self.lower_value is not None
            and self.upper_value is not None
            and self.lower_value > self.upper_value
        ):
            raise ValueError("lower_value must not exceed upper_value")
