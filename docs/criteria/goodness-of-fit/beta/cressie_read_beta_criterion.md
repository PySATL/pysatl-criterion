# Cressie–Read for a specified Beta distribution

[All Beta criteria](index.md) · [Mathematical verification](mathematical-validation.md)

Class: `CressieReadBetaGofStatistic`. Code: `CRESSIE_READ_BETA_GOODNESS_OF_FIT`.

## Hypothesis and formula

Both shape parameters `a > 0` and `b > 0` are finite and fixed independently of
the sample. Pass `Beta.DEFAULT.parse({"a": 2, "b": 5})`; omitted shapes are
unknown and are rejected by this class. The support is fixed at $[0,1]$.

For bins $j=1,\ldots,J$, write $O_j$ for the observed count and $E_j=np_j>0$
for the expected count under the specified Beta law.

$$
CR=\frac95\sum_j O_j\left[(O_j/E_j)^{2/3}-1\right]
$$

With an empty observed cell and positive expected count, Pearson contributes $E_j$, Freeman–Tukey $4E_j$, modified log-likelihood and Neyman infinity, and Cressie–Read zero.

## Input, result and calibration

`execute_statistic(rvs, **kwargs)` accepts a nonempty, finite, one-dimensional
real sample in $[0,1]$. Ties and constants are allowed; masked observations and
complex values are rejected. It returns a scalar and leaves the sample unchanged.
Large values reject (`RightAlternative`). Extra execution keywords are ignored.

The constructor accepts `binning: BinningStrategy`; the default is
`EqualProbabilityBins()` with `PowerCount(coefficient=2, exponent=0.4)`,
a maximum of 1000 bins and no minimum expected-count pooling.
CDF-space boundaries belong to the cell on their right; 1 belongs to the last cell.
See [binning strategies](../binning/index.md). Each of the five divergences has
its own class; `lambda_` is not a supported constructor argument.

For fixed deterministic bins with positive probabilities the usual asymptotic
null is chi-square with $J-1$ degrees of freedom, with no fitted-shape deduction.
That statement does not justify a fixed-degree limit for a growing number of bins
or data-dependent boundaries. Calibrate using the same binning rule and sample size.
Storage identity includes the serialized binning configuration.

## Example

```python
from pysatl_criterion.distribution import BetaDistributionDescriptor as Beta
from pysatl_criterion.statistics.goodness_of_fit.continuous.beta import CressieReadBetaGofStatistic
from pysatl_criterion.statistics.goodness_of_fit.binning import EqualProbabilityBins, FixedCount

statistic = CressieReadBetaGofStatistic(Beta.DEFAULT.parse({"a": 2, "b": 5}), binning=EqualProbabilityBins(count=FixedCount(3)))
value = statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
print(value)
```

## Scientific source

[Cressie and Read (1984), Multinomial Goodness-of-Fit Tests](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x), $\lambda=2/3$.

The source identifies the shared statistical functional; applying it through
the specified Beta distribution does not constitute a new Beta-specific test.
