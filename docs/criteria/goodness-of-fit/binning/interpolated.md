# Interpolated bins

`InterpolatedBins(lower, upper, count=PowerCount(), min_expected=None,
max_bins=1000, kappa=0.5)` blends equal-probability and equal-width boundaries.

## Algorithm

For `0 < kappa < 1`:

1. Construct equal-width central edges and any required tail cells.
2. Let $K$ be the total number of cells, including tails.
3. Find the null quantiles $q_i=F_0^{-1}(i/K)$ for interior indices.
4. Replace each interior equal-width edge $w_i$ by
   $b_i=(1-\kappa)q_i+\kappa w_i$; keep the outer coverage edges.
5. Count observations and compute actual expected probabilities, then apply
   optional merging.

Null quantiles are found by numerical CDF inversion. The supplied finite bounds
seed a bracket that is expanded if needed. Failure to bracket a finite quantile
or obtain valid ordered edges raises an error.

At `kappa=0` the implementation delegates to `EqualProbabilityBins`; at `kappa=1`
it delegates to `EqualWidthBins`. The number of cells can therefore change at
the zero endpoint when width-based grouping has extra tails. This is a library
policy, not a continuous fixed-cell-count interpolation across that endpoint.

## Example

```python
import numpy as np
from scipy import stats
from pysatl_criterion.statistics.goodness_of_fit.binning import InterpolatedBins

strategy = InterpolatedBins(0, 4, count=2, kappa=0.5)
table = strategy.build([0.1, 0.5, 1.5, 3, 10], stats.expon())
assert len(table.expected) == 3  # Two central bins and an upper tail.
assert table.observed.sum() == 5
np.testing.assert_allclose(table.expected.sum(), 5)
```

## Scientific source

[Rolke and Gutierrez Gongora (2021), *A chi-square goodness-of-fit test for continuous distributions against a known alternative*](https://doi.org/10.1007/s00180-020-00997-x), [open preprint, §5.1](https://arxiv.org/html/2005.02793v1#S5.SS1), equation (3), gives the boundary interpolation formula.
Numerical inversion, tail handling and endpoint delegation described above are
specific to this implementation.

[All binning methods](index.md)
