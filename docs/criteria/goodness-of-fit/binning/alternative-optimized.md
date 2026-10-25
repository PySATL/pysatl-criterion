# Binning selected for a specified alternative

`AlternativeOptimizedBins.select(...)` chooses from an explicit list of candidate
strategies using synthetic null and alternative samples. Selection happens before
the tested observations are processed.

## Algorithm

1. Supply a bound null distribution, a matching null sampler, an alternative
   sampler, the frequency kernel, sample size and candidate strategies.
2. Generate `repetitions` null and alternative samples with a seeded NumPy RNG.
   Each sampler receives `(sample_size, rng)`. Candidates share these samples.
3. Evaluate every candidate/kernel combination on the generated samples.
4. For each candidate, sort its null values and select zero-based rank
   `ceil((1-alpha) * (repetitions-1))` as a simulated critical value.
5. Score the candidate by the fraction of alternative values strictly above its
   critical value. Choose the largest score; ties retain the first candidate.
6. Freeze the winning strategy, sample size and all `estimated_powers`.

Defaults are `repetitions=1000`, `alpha=0.05`, and `seed=0`; at least 20
repetitions and `0 < alpha < 1` are required. Candidate errors are propagated.
NaN statistics are rejected; infinite statistics can occur for some kernels.

## Example

```python
import numpy as np
from scipy import stats
from pysatl_criterion.statistics.goodness_of_fit.binning import (
    AlternativeOptimizedBins, EqualProbabilityBins,
)


def pearson(observed, expected):
    return float(np.sum((observed - expected) ** 2 / expected))


selected = AlternativeOptimizedBins.select(
    [EqualProbabilityBins(2), EqualProbabilityBins(4)],
    distribution=stats.uniform(),
    null_sampler=lambda n, rng: rng.uniform(size=n),
    alternative_sampler=lambda n, rng: rng.beta(1, 4, size=n),
    kernel=pearson,
    sample_size=40,
    repetitions=100,  # Small demonstration budget, not a precision recommendation.
    seed=19,
)

# New observations, not the synthetic selection samples.
sample = np.random.default_rng(100).uniform(size=40)
table = selected.build(sample, stats.uniform())
assert table.observed.sum() == 40
assert len(selected.estimated_powers) == 2
```

Pass the resulting strategy as `binning=selected` to the criterion using that
same kernel and null model. A different sample size is rejected. The caller must
also repeat selection when changing the null model or kernel; only sample size
is checked automatically.

The returned power scores are selection estimates, not unbiased performance
estimates. Calibrate the selected test on fresh null simulations. Do not choose
candidates by maximizing the statistic on the actual tested observations.

## Scientific context and differences

[Rolke and Gutierrez Gongora (2021), *A chi-square goodness-of-fit test for continuous distributions against a known alternative*](https://doi.org/10.1007/s00180-020-00997-x) studies selection against a known alternative; see the
[open preprint, §5.3](https://arxiv.org/html/2005.02793v1#S5.SS3).
Our implementation is a Monte Carlo candidate search, **not a reproduction of
their exact selection algorithm**, and it provides no universal optimality claim.

[All binning methods](index.md)
