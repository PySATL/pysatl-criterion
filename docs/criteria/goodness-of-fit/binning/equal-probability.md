# Equal-probability bins

`EqualProbabilityBins(count=PowerCount(), min_expected=None, max_bins=1000)`
constructs cells with equal probability under a continuous null distribution.

## Algorithm

1. Resolve $k$ from the sample size and cap it by `max_bins`.
2. If `min_expected=m` is supplied, also cap it by $\lfloor n/m\rfloor$.
   Reject the request if fewer than two bins remain.
3. Transform each observation to $u_j=F_0(x_j)$.
4. Count these values in $k$ equal intervals over $[0,1]$.
5. Set every expected count to $n/k$.

This avoids evaluating an inverse CDF. The probability-space bins use NumPy's
left-closed convention; the final bin includes 1. The CDF must return finite
values in $[0,1]$ aligned with the sample.

## Example

```python
import numpy as np
from scipy import stats
from pysatl_criterion.statistics.goodness_of_fit.binning import EqualProbabilityBins

null = stats.norm(loc=2, scale=3)
sample = null.ppf([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
table = EqualProbabilityBins(count=4).build(sample, null)
np.testing.assert_array_equal(table.observed, [2, 1, 1, 2])
np.testing.assert_allclose(table.expected, [1.5] * 4)
```

A descriptor-bound distribution can replace the SciPy frozen distribution.
The null parameters must already be specified; they are not fitted here.
Uniform criteria additionally reject observations outside their fixed support.

Do not apply this strategy to a discrete CDF: jumps prevent the equal-probability
assumption. Use [DiscreteCategoryBins](discrete.md) instead.

## Scientific source

[Rolke and Gutierrez Gongora (2021), *A chi-square goodness-of-fit test for continuous distributions against a known alternative*](https://doi.org/10.1007/s00180-020-00997-x), [open preprint, §5.1](https://arxiv.org/html/2005.02793v1#S5.SS1), discusses equal-probability grouping. Our count resolution,
probability-space boundary convention and limits are implementation policies.

[All binning methods](index.md)
