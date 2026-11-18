"""Select a binning strategy before testing, using a specified alternative."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pysatl_criterion.distribution.evaluation import BoundDistribution
from pysatl_criterion.statistics.goodness_of_fit.binning.base import (
    BinningStrategy,
    validate_sample,
)
from pysatl_criterion.statistics.goodness_of_fit.binning.counts import positive_integer


Sampler = Callable[[int, np.random.Generator], NDArray[np.float64]]
Kernel = Callable[[NDArray[np.float64], NDArray[np.float64]], float]


@dataclass(frozen=True)
class AlternativeOptimizedBins:
    """Frozen result of a simulation-based search over explicit candidates.

    Selection uses independent synthetic null/alternative samples, never the
    observations to be tested. Power estimates are selection scores, not
    unbiased performance estimates. Calibrate the selected test separately.
    This is a Monte Carlo candidate search, not the exact Rolke procedure.

    References
    ----------
    Rolke, W., and Gutierrez Gongora, C. (2021). A chi-square goodness-of-fit
    test for continuous distributions against a known alternative.
    Computational Statistics, 36, 1885–1900.
    https://doi.org/10.1007/s00180-020-00997-x
    Background on alternative-specific selection: 2020 preprint, section 5.3.
    https://arxiv.org/html/2005.02793v1#S5.SS3
    """

    selected: BinningStrategy
    sample_size: int
    estimated_powers: tuple[float, ...]

    def __post_init__(self):
        positive_integer(self.sample_size, "sample_size")

    def to_config(self) -> dict:
        from pysatl_criterion.statistics.goodness_of_fit.binning.configuration import (
            binning_configuration,
        )

        return binning_configuration(self)

    def supports(self, kind):
        return self.selected.supports(kind)

    def build(self, sample, distribution):
        sample = validate_sample(sample)
        if len(sample) != self.sample_size:
            raise ValueError("Optimized binning requires the sample size used during selection")
        return self.selected.build(sample, distribution)

    @classmethod
    def select(
        cls,
        candidates: Sequence[BinningStrategy],
        *,
        distribution: BoundDistribution,
        null_sampler: Sampler,
        alternative_sampler: Sampler,
        kernel: Kernel,
        sample_size: int,
        repetitions: int = 1000,
        alpha: float = 0.05,
        seed: int = 0,
    ) -> "AlternativeOptimizedBins":
        """Choose greatest estimated power; ties keep the first candidate.

        Samplers accept (sample_size, rng). The null sampler must match the
        bound distribution, and kernel must be the statistic used downstream.
        Candidate failures are reported rather than silently excluded.
        """
        positive_integer(sample_size, "sample_size")
        positive_integer(repetitions, "repetitions", 20)
        if not candidates or not 0 < alpha < 1:
            raise ValueError("Candidates and alpha in (0, 1) are required")
        rng = np.random.default_rng(seed)
        null_values = np.empty((len(candidates), repetitions))
        alt_values = np.empty_like(null_values)
        for j in range(repetitions):
            null = validate_sample(null_sampler(sample_size, rng))
            alternative = validate_sample(alternative_sampler(sample_size, rng))
            if len(null) != sample_size or len(alternative) != sample_size:
                raise ValueError("Sampler returned an unexpected sample size")
            for i, candidate in enumerate(candidates):
                null_table = candidate.build(null, distribution)
                alt_table = candidate.build(alternative, distribution)
                null_values[i, j] = kernel(null_table.observed, null_table.expected)
                alt_values[i, j] = kernel(alt_table.observed, alt_table.expected)
        if np.any(np.isnan(null_values)) or np.any(np.isnan(alt_values)):
            raise ValueError("Kernel produced NaN during selection")
        # An observed order statistic also works when a kernel returns infinity.
        rank = min(repetitions - 1, int(np.ceil((1 - alpha) * (repetitions - 1))))
        critical = np.sort(null_values, axis=1)[:, rank]
        powers = tuple(float(p) for p in np.mean(alt_values > critical[:, None], axis=1))
        return cls(candidates[int(np.argmax(powers))], sample_size, powers)
