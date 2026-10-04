from dataclasses import dataclass


@dataclass(frozen=True)
class DistributionKey:
    """Identity of a distribution, independent of simulation count."""

    criterion_code: str
    criterion_parameters: dict[str, float]
    sample_size: int
