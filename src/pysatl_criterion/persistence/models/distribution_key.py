from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pysatl_criterion.statistics.configuration import normalize_configuration


if TYPE_CHECKING:
    from pysatl_criterion.statistics.goodness_of_fit import AbstractGoodnessOfFitStatistic


@dataclass(frozen=True)
class DistributionKey:
    """Identity of a distribution, independent of simulation count."""

    criterion_code: str
    criterion_parameters: dict[str, float]
    sample_size: int
    criterion_configuration: dict[str, Any] | None = None

    @classmethod
    def from_statistic(
        cls, statistic: "AbstractGoodnessOfFitStatistic", sample_size: int
    ) -> "DistributionKey":
        """Build the same calibration identity for storage reads and writes."""
        validate = getattr(statistic, "_validate_storage_calibration", None)
        if validate is not None:
            validate()
        configuration = statistic.configuration()
        if not isinstance(configuration, dict):
            raise TypeError(
                "configuration() must return a dictionary; None is reserved for legacy data"
            )
        return cls(
            statistic.code(),
            dict(statistic.hypothesis().parameters()),
            sample_size,
            normalize_configuration(configuration),
        )
