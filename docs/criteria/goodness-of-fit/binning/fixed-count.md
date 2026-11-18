# Fixed number of bins

`FixedCount(count)` returns a user-selected integer independently of sample size.
It is a count rule, not a partition of the sample.

## Algorithm

1. Require an integer `count >= 2`; booleans are rejected.
2. Require a positive integer sample size.
3. Return the configured count.

A consuming strategy can reduce it because of `max_bins` or `min_expected`.
Passing an integer to a strategy is shorthand for `FixedCount`.

## Example

```python
from scipy import stats
from pysatl_criterion.statistics.goodness_of_fit.binning import FixedCount, EqualProbabilityBins

rule = FixedCount(4)
assert rule.resolve(100) == 4
strategy = EqualProbabilityBins(count=rule)
table = strategy.build([0.1, 0.2, 0.4, 0.6, 0.8, 0.9], stats.uniform())
assert len(table.expected) == 4
```

Choose this when a study specifies the grouping in advance. Increasing the
sample size does not increase the number of bins automatically.

## Scientific context

[Cressie and Read (1984), *Multinomial Goodness-of-Fit Tests*](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x) treats observations as counts in specified categories. This provides
the multinomial context, not a formula for choosing this class's `count`.
The fixed-count setting is a user decision and is not attributed to an optimal
selection algorithm in that paper.

[All binning methods](index.md)
