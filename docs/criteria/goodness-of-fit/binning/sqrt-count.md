# Square-root count rule

`SqrtCount()` is a convenience heuristic based only on sample size.

## Algorithm

For a positive integer $n$, return

$$
k=\max(2,\lceil\sqrt n\rceil).
$$

The strategy then applies `max_bins` and, where applicable, `min_expected`.
The rule neither inspects the sample's dispersion nor constructs boundaries.

## Example

```python
from scipy import stats
from pysatl_criterion.statistics.goodness_of_fit.binning import SqrtCount, EqualProbabilityBins

rule = SqrtCount()
assert rule.resolve(100) == 10
table = EqualProbabilityBins(count=rule).build([0.2, 0.8] * 50, stats.uniform())
assert len(table.expected) == 10
```

## Scientific context

This heuristic is not attributed to a specific research article and does not
carry a GoF optimality claim. For research on the separate problem of choosing
chi-square class intervals, see [Mann and Wald (1942), *On the Choice of the Number of Class Intervals in the Application of the Chi Square Test*](https://doi.org/10.1214/aoms/1177731569). That reference is background, not the
source of the square-root formula.

[All binning methods](index.md)
