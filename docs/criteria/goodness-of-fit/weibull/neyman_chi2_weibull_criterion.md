# Neyman chi-squared test for the Weibull family

## Description

`NeymanChi2WeibullGofStatistic` has algorithm code `CHI2_NEYMAN_WEIBULL_GOODNESS_OF_FIT`.
It is generated from the shared frequency kernel.

The null is the ordinary Weibull family with zero location:

$$
F(x;k,\eta)=1-\exp\left[-(x/\eta)^k\right],\qquad x>0,\quad k,\eta>0.
$$

Both parameters are fitted by maximum likelihood for every sample.

## Usage

```python
from pysatl_criterion.distribution import WeibullDistributionDescriptor as Weibull
from pysatl_criterion.statistics.goodness_of_fit.binning import EqualProbabilityBins
from pysatl_criterion.statistics.goodness_of_fit.continuous.weibull import (
    NeymanChi2WeibullGofStatistic,
)

criterion = NeymanChi2WeibullGofStatistic(Weibull.DEFAULT.parse({}), binning=EqualProbabilityBins(count=3))
statistic = criterion.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
print(statistic)
```

## Arguments

`parameters` — `Weibull.DEFAULT.parse({})`, declaring both shape and scale
unknown. Fixed or partially fixed parameter sets are unsupported.

`rvs` — a one-dimensional, finite, strictly positive, nonconstant sample with
at least two observations. Ties are allowed. Invalid samples raise `ValueError`.
The input is not modified and fitted parameters are not retained on the instance.

`binning` — keyword-only `BinningStrategy`, default `EqualProbabilityBins()`.
See [binning strategies](../binning/index.md). The chosen rule is included in
`configuration()`; changing it changes the statistic's calibration.

## Details

After fitting, the selected binning rule constructs observed counts $O_j$
and strictly positive expected counts $E_j=n p_j$. For interval bins with
boundaries $b_0,\ldots,b_m$,

$$
p_j=F(b_j;\widehat k,\widehat\eta)-F(b_{j-1};\widehat k,\widehat\eta).
$$

The rule must account for all observations. Equal-probability bins use quantiles
of the fitted distribution, giving $E_j=n/m$ before any rule-specific merging.

$$
T_N=\sum_{j=1}^{m}\frac{(O_j-E_j)^2}{O_j}.
$$

Power divergence with $\lambda=-2$. Observed counts appear in the denominator; an empty observed bin gives positive infinity.

## Calibration

Large values form the right-tail rejection region. `execute_statistic` returns
a statistic, not a p-value. For `EqualProbabilityBins`, simulate ordinary Weibull
samples at shape = scale = 1, refit both parameters, and reconstruct the bins in
every replicate. Keep the sample size and binning settings identical.

The built-in Monte Carlo resolver supports this invariant binning rule. Other
rules, including fixed boundaries in observation units, require external
calibration with explicit simulation parameters. A chi-squared reference with
an automatically chosen number of degrees of freedom is not supplied or justified
by this implementation. Stored unversioned Weibull critical values are rejected.

## References

Cressie, N., and Read, T. R. C. (1984). [Multinomial Goodness-of-Fit Tests](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x). *Journal of the Royal Statistical Society: Series B*, 46(3), 440–464.

The reference describes the frequency discrepancy; the fitted-Weibull binning and calibration procedure above specifies its use in this library.

See the [Weibull criterion catalog](index.md) for all implementations and shared
calibration requirements.
