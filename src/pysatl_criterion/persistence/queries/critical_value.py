from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from pysatl_criterion.persistence.models.critical_value import normalize_significance_level
from pysatl_criterion.persistence.models.distribution_key import DistributionKey
from pysatl_criterion.persistence.queries.limit_distribution import LimitDistributionFilter
from pysatl_criterion.statistics.alternative import AlternativeType


@dataclass(frozen=True)
class CriticalValueLookupQuery:
    distribution_key: DistributionKey
    monte_carlo_count: int
    significance_level: Decimal
    alternative_type: AlternativeType

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "significance_level", normalize_significance_level(self.significance_level)
        )


@dataclass(frozen=True)
class CriticalValueFilter(LimitDistributionFilter):
    """Select current cached bounds with optional level and alternative filters."""

    significance_levels: Sequence[Decimal] | None = None
    alternative_types: Sequence[AlternativeType] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.significance_levels is not None:
            object.__setattr__(
                self,
                "significance_levels",
                tuple(normalize_significance_level(level) for level in self.significance_levels),
            )
        if self.alternative_types is not None and any(
            not isinstance(alternative, AlternativeType) for alternative in self.alternative_types
        ):
            raise ValueError("alternative_types must contain AlternativeType values")
