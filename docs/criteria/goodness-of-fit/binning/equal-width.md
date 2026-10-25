# Equal-width bins with tails

`EqualWidthBins(lower, upper, count=PowerCount(), min_expected=None, max_bins=1000)`
creates equal-width central intervals and preserves nonempty null tails.

## Algorithm

1. Require fixed finite bounds `lower < upper` with finite width.
2. Resolve and cap the number $k$ of **central** bins.
3. Place $k+1$ equally spaced edges between the bounds.
4. Add `-inf` if the null assigns probability below `lower`; add `inf` if it
   assigns probability above `upper`.
5. Build an [explicit frequency table](explicit.md), including optional merging.

The final count can be $k$, $k+1$, or $k+2$ before merging. Expected counts are
computed from the null CDF and generally differ across bins. `max_bins` limits
central bins, not the additional tails.

## Example

```python
import numpy as np
from scipy import stats
from pysatl_criterion.statistics.goodness_of_fit.binning import EqualWidthBins

null = stats.norm()
table = EqualWidthBins(-1, 1, count=2).build([-3, -0.5, 0.5, 3], null)
np.testing.assert_array_equal(table.observed, [1, 1, 1, 1])
np.testing.assert_allclose(
    table.expected, 4 * np.diff(null.cdf([-np.inf, -1, 0, 1, np.inf])),
)
```

Set central bounds before testing. Selecting them from sample extrema introduces
another data-dependent step which this class does not implement.

## Scientific source

[Rolke and Gutierrez Gongora (2021), *A chi-square goodness-of-fit test for continuous distributions against a known alternative*](https://doi.org/10.1007/s00180-020-00997-x), [open preprint, §5.1](https://arxiv.org/html/2005.02793v1#S5.SS1), discusses equal-width grouping.
The separate tail categories and finite central-bound API are our implementation
policies, not a claim to reproduce every boundary convention in the paper.

[All binning methods](index.md)
