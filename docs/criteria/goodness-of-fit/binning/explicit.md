# Explicit bin edges

`ExplicitBins(edges, min_expected=None)` uses supplied continuous bin boundaries.
They must cover the full null probability and all observations.

## Algorithm

1. Require at least three strictly increasing edges, without NaNs.
2. Verify that the null probability below the first edge and above the last is zero.
3. Count observations using `[left, right)` cells; include the final right endpoint.
4. Calculate $E_i=n(F_0(b_{i+1})-F_0(b_i))$.
5. Remove empty zero-probability categories and apply optional expected-count merging.

Upper-tail probabilities use survival-function differences to avoid subtracting
CDF values near one. A positive observation count in a zero-probability category
is an error. At least two categories must remain.

## Example

```python
import numpy as np
from scipy import stats
from pysatl_criterion.statistics.goodness_of_fit.binning import ExplicitBins

strategy = ExplicitBins((-np.inf, 0, np.inf))
table = strategy.build([-2, -0.5, 0.5, 2], stats.norm())
np.testing.assert_array_equal(table.observed, [2, 2])
np.testing.assert_allclose(table.expected, [2, 2])
```

Choose boundaries independently of the tested observations, or calibrate any
additional boundary-selection procedure explicitly. Finite edges are appropriate
when they cover a bounded null support.

## Scientific context

[Cressie and Read (1984), *Multinomial Goodness-of-Fit Tests*](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x), section 1, supplies the multinomial-category framework.
The explicit boundary API and the adjacent-cell merging algorithm are library
policies, not a bin-selection algorithm attributed to this paper.

[All binning methods](index.md)
