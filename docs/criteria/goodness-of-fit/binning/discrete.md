# Discrete category bins

`DiscreteCategoryBins(cut_points, min_expected=None)` groups a discrete null
using ordered cut points. It uses actual category masses, not equal probabilities.

## Algorithm

For finite increasing cuts $c_1,\ldots,c_m$, form

$$
(-\infty,c_1],\ (c_1,c_2],\ \ldots,\ (c_m,\infty).
$$

1. Assign each observation to its right-closed category. An atom exactly at a cut
   belongs to the category on its left.
2. Calculate null masses using CDF differences, with stable SF differences in
   the upper tail; multiply by $n$ for expected counts.
3. Remove zero-probability empty categories and optionally merge adjacent small
   expected categories using the [shared policy](index.md#defaults-and-small-expected-counts).

Cuts need not be integers. The distribution defines the masses; the caller is
responsible for valid sample support when using this strategy directly.
At least two positive-probability categories must remain. This strategy declares
compatibility with discrete distributions and is rejected by Uniform
criteria.

## Example

```python
import numpy as np
from scipy import stats
from pysatl_criterion.statistics.goodness_of_fit.binning import DiscreteCategoryBins

# Geometric here counts trials including the first success: 1, 2, ...
table = DiscreteCategoryBins((1, 3)).build([1, 1, 2, 3, 4, 5], stats.geom(0.5))
np.testing.assert_array_equal(table.observed, [2, 2, 2])
np.testing.assert_allclose(table.expected, 6 * np.array([0.5, 0.375, 0.125]))
```

## Scientific context

[Cressie and Read (1984), *Multinomial Goodness-of-Fit Tests*](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x), section 1, treats exhaustive categories through multinomial
probabilities. Right-closed cut points and this library's deterministic merging
policy are implementation choices rather than an algorithm claimed from the paper.

[All binning methods](index.md)
