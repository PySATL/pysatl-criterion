# Power count rule

`PowerCount(coefficient=2.0, exponent=0.4)` provides an explicit sample-size
heuristic whose parameters can be changed.

## Algorithm

Require positive finite $c$ and $p$ and a positive integer sample size $n$:

$$
k=\max(2,\lceil c n^p\rceil).
$$

The default is $\lceil2n^{2/5}\rceil$ with a minimum of two.
The consuming strategy applies its caps. For example, `min_expected=5` in
equal-probability binning restricts $k\le\lfloor n/5\rfloor$.

## Example

```python
from scipy import stats
from pysatl_criterion.statistics.goodness_of_fit.binning import PowerCount, EqualProbabilityBins

rule = PowerCount()
assert rule.resolve(100) == 13
table = EqualProbabilityBins(count=rule, min_expected=5).build(
    [0.2, 0.8] * 10, stats.uniform(),
)
assert len(table.expected) == 4  # Expected-count policy reduces the requested 7.
```

## Sources and scope

[NIST/SEMATECH handbook, §7.2.1.1](https://www.itl.nist.gov/div898/handbook/prc/section2/prc211.htm) gives $2n^{2/5}$ as a practical starting heuristic. It is a handbook,
not a research paper. Rounding and caps are implementation choices.

[Mann and Wald (1942), *On the Choice of the Number of Class Intervals in the Application of the Chi Square Test*](https://doi.org/10.1214/aoms/1177731569) is historical scientific background on choosing class intervals. This
class does not implement a particular Mann–Wald optimality formula. Its general
coefficient/exponent API does not establish optimality for arbitrary settings.

[All binning methods](index.md)
