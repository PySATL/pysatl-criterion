# Sturges count rule

`SturgesCount()` chooses a logarithmically growing number of bins.

## Algorithm

For a positive integer sample size $n$, return

$$
k=\max(2,\lceil1+\log_2 n\rceil).
$$

Ceiling and the minimum of two are library policies. A binning strategy applies
its own count cap and expected-count policy afterward. No sample values enter
the formula.

## Example

```python
from scipy import stats
from pysatl_criterion.statistics.goodness_of_fit.binning import SturgesCount, EqualProbabilityBins

rule = SturgesCount()
assert rule.resolve(100) == 8
strategy = EqualProbabilityBins(count=rule, min_expected=5)
table = strategy.build([0.25, 0.75] * 50, stats.uniform())
assert len(table.expected) == 8
```

The rule offers a simple grouping convention; it does not guarantee maximum
power against a particular alternative.

## Scientific source

[Sturges (1926), *The Choice of a Class Interval*](https://doi.org/10.1080/01621459.1926.10502161). Journal of the American Statistical Association, 21(153), 65–66.
[Open article scan](https://www.stat.rice.edu/~scottdw/stat550/DEMOS/09-14-hist/Sturges1926.pdf).
The original work motivates the logarithmic class-count relationship.

[All binning methods](index.md)
