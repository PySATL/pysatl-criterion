# Modified log-likelihood for exponentiated Weibull

`ModifiedLogLikelihoodExponentiatedWeibullGofStatistic`

Short code: `MODIFIED_LOG_LIKELIHOOD`. Full code: `MODIFIED_LOG_LIKELIHOOD_EXPONENTIATED_WEIBULL_GOODNESS_OF_FIT`.

## Hypothesis and input

The null is fully specified:

$$
F_{a,k,s}(x)=\left[1-\exp\left(-(x/s)^k\right)\right]^a,\quad x\ge0,
\qquad a,k,s>0.
$$

Fix `exponent=a`, `shape=k`, `scale=s` in
`ExponentiatedWeibullDistributionDescriptor.DEFAULT.parse(...)`; location is zero.
All three parameters are required and are never estimated from the observations.
`execute_statistic` accepts a nonempty one-dimensional finite nonnegative sample;
ties and constant samples are allowed. It returns a scalar statistic, not a p-value,
and leaves the sample and parameters unchanged.

## Formula

Here $O_j$ and $E_j=np_j>0$ are observed and expected counts in exhaustive,
mutually exclusive cells, with $\sum_jO_j=\sum_jE_j=n$.
The constructor accepts keyword-only `binning`. The default is
`EqualProbabilityBins(PowerCount(2.0, 0.4), min_expected=None, max_bins=1000)`:
$B=\min(1000,\max(2,\lceil2n^{0.4}\rceil))$, and $p_j=1/B$.
Thus the default has two cells even for a singleton sample. The old
$\lceil\sqrt n\rceil$ rule and its one-cell singleton result no longer apply.

The example chooses three equal-probability cells explicitly. You can instead use
`ExplicitBins((0, b1, b2, float("inf")))` for fixed full-support edges; then
$p_j=F(b_{j+1})-F(b_j)$. See [binning strategies](../binning/index.md).

$$
T_{ML}=2\sum_{j=1}^B E_j\log\frac{E_j}{O_j}.
$$

Power divergence at lambda=−1. This reverses the usual likelihood-ratio divergence. Any zero O_j with E_j>0 gives positive infinity.

## Calibration

Reject for large values (right tail). Use the same sample size and statistic normalization when calibrating the fixed null. For fixed B and sufficiently large expected counts the usual reference is chi-squared with B−1 degrees of freedom; no parameters are fitted. Small or empty cells and a sample-size-dependent B require appropriate calibration, preferably Monte Carlo. Repeat the chosen binning procedure in every replicate. The family currently rejects stored unversioned calibrations; regenerate them and do not assume storage loading is enabled.

## Example

```python
from pysatl_criterion.distribution import ExponentiatedWeibullDistributionDescriptor as Distribution
from pysatl_criterion.statistics.goodness_of_fit.continuous.exponentiated_weibull import (
    ModifiedLogLikelihoodExponentiatedWeibullGofStatistic,
)
from pysatl_criterion.statistics.goodness_of_fit.binning import EqualProbabilityBins

parameters = Distribution.DEFAULT.parse({"exponent": 1, "shape": 2, "scale": 1})
test = ModifiedLogLikelihoodExponentiatedWeibullGofStatistic(parameters, binning=EqualProbabilityBins(count=3))
value = test.execute_statistic([0.1, 0.25, 0.4, 0.55, 0.7, 0.9, 1.2, 1.5, 2.0])
print(value)
```

## References

[Cressie and Read (1984), Multinomial Goodness-of-Fit Tests](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x); [full text](https://pages.stern.nyu.edu/~dbackus/Exotic/1Robustness/CressieRead%20JRSSB%2084.pdf).

See the [complete catalog](index.md) and [mathematical verification](mathematical-validation.md).
