from enum import Enum


class DistributionKind(Enum):
    """Enumerate the kinds of probability distributions."""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    SINGULAR = "singular"
    MIXED = "mixed"

    @classmethod
    def list(cls) -> list[str]:
        """Return the string values for all distribution kinds."""
        return [member.value for member in cls]
