"""Rules for the number of categories, independent of their boundaries."""

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Protocol


def positive_integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


class CountRule(Protocol):
    def resolve(self, sample_size: int) -> int: ...


class _ConfiguredCount:
    def to_config(self) -> dict:
        from pysatl_criterion.statistics.goodness_of_fit.binning.configuration import (
            count_configuration,
        )

        return count_configuration(self)


@dataclass(frozen=True)
class FixedCount(_ConfiguredCount):
    """User-selected number of bins; no data-driven selection rule is applied."""

    count: int

    def __post_init__(self):
        positive_integer(self.count, "count", 2)

    def resolve(self, sample_size: int) -> int:
        positive_integer(sample_size, "sample_size")
        return self.count


@dataclass(frozen=True)
class SturgesCount(_ConfiguredCount):
    """Logarithmic bin count; rounding and minimum count are library policies.

    References
    ----------
    Sturges, H. A. (1926). The Choice of a Class Interval.
    Journal of the American Statistical Association, 21(153), 65–66.
    https://doi.org/10.1080/01621459.1926.10502161
    """

    def resolve(self, sample_size: int) -> int:
        n = positive_integer(sample_size, "sample_size")
        return max(2, math.ceil(1 + math.log2(n)))


@dataclass(frozen=True)
class SqrtCount(_ConfiguredCount):
    """Square-root convenience heuristic, without a GoF optimality claim.

    This implementation is not attributed to a specific research article.
    """

    def resolve(self, sample_size: int) -> int:
        n = positive_integer(sample_size, "sample_size")
        return max(2, math.ceil(math.sqrt(n)))


@dataclass(frozen=True)
class PowerCount(_ConfiguredCount):
    """Explicit power heuristic; defaults to ceil(2 * n**0.4).

    The default coefficient and exponent follow the NIST handbook heuristic.
    Rounding and the minimum of two bins are implementation choices. This is
    not an implementation of a specific Mann–Wald optimality formula.

    References
    ----------
    NIST/SEMATECH e-Handbook, section 7.2.1.1 (practical guidance, not a paper).
    https://www.itl.nist.gov/div898/handbook/prc/section2/prc211.htm

    Historical background only: Mann, H. B., and Wald, A. (1942).
    On the Choice of the Number of Class Intervals in the Application of the
    Chi Square Test. Annals of Mathematical Statistics, 13(3), 306–317.
    https://doi.org/10.1214/aoms/1177731569
    """

    coefficient: float = 2.0
    exponent: float = 0.4

    def __post_init__(self):
        for name in ("coefficient", "exponent"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    def resolve(self, sample_size: int) -> int:
        n = positive_integer(sample_size, "sample_size")
        return max(2, math.ceil(self.coefficient * n**self.exponent))
