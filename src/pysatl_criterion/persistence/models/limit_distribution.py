from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from pysatl_criterion.statistics.goodness_of_fit import AbstractGoodnessOfFitStatistic

from pysatl_criterion.persistence.models.distribution_key import DistributionKey


@dataclass
class LimitDistributionModel:
    """
    Model for storing limit distribution data from Monte Carlo simulations.
    """

    criterion_code: str
    criterion_parameters: dict[str, float]
    sample_size: int
    monte_carlo_count: int
    results_statistics: list[float]
    criterion_configuration: dict[str, Any] | None = None

    @property
    def key(self) -> DistributionKey:
        return DistributionKey(
            self.criterion_code,
            dict(self.criterion_parameters),
            self.sample_size,
            self.criterion_configuration,
        )

    @classmethod
    def from_statistic(
        cls,
        statistic: "AbstractGoodnessOfFitStatistic",
        sample_size: int,
        results_statistics: list[float],
    ) -> "LimitDistributionModel":
        """Attach the statistic configuration to freshly simulated values."""
        key = DistributionKey.from_statistic(statistic, sample_size)
        return cls(
            key.criterion_code,
            key.criterion_parameters,
            sample_size,
            len(results_statistics),
            list(results_statistics),
            key.criterion_configuration,
        )
