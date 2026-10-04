from dataclasses import dataclass

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

    @property
    def key(self) -> DistributionKey:
        return DistributionKey(
            self.criterion_code, dict(self.criterion_parameters), self.sample_size
        )
