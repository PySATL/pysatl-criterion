from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class CriterionFilter:
    """Match a code and its complete parameter set.

    None matches all parameter variants of this code; {} matches only parameterless criteria.
    """

    criterion_code: str
    criterion_parameters: dict[str, float] | None = None


@dataclass(frozen=True)
class LimitDistributionFilter:
    """Select all matching distributions, with inclusive size/count bounds.

    None leaves a field unrestricted. An empty criteria list matches nothing. Each criterion
    binds a code to its parameters; a distribution must match at least one listed criterion.
    """

    criteria: Sequence[CriterionFilter] | None = None
    min_sample_size: int | None = None
    max_sample_size: int | None = None
    min_monte_carlo_count: int | None = None

    def __post_init__(self) -> None:
        for name in ("min_sample_size", "max_sample_size", "min_monte_carlo_count"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")
        if (
            self.min_sample_size is not None
            and self.max_sample_size is not None
            and self.min_sample_size > self.max_sample_size
        ):
            raise ValueError("min_sample_size must not exceed max_sample_size")
