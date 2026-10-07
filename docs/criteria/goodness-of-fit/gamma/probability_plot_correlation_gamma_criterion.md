# Gamma quantile correlation discrepancy with Blom positions.

Return 1-r, where r is the Pearson correlation of ordered x with
Gamma.ppf((i-0.375)/(n+0.25), alpha, scale=1). The rate cancels
from correlation; beta remains fixed in the declared null hypothesis.
The statistic cannot detect positive affine changes of the sample.
The null law depends on alpha. A nonconstant sample with n>=2 is
required; for n=2 the statistic is degenerate and has no useful power.
Independent rescaling of data and quantiles avoids moment overflow.
No primary source validating this exact Gamma/Blom test was found;
Filliben normality tables do not calibrate it.

For ordered observations x_(i), let u_i=F(x_(i); alpha, beta),
where Gamma has shape alpha, rate beta and known origin zero.
Parameters are fixed, not fitted. Samples must be finite, real,
one-dimensional and nonempty, in the closed support [0, infinity).
Zero is accepted as a boundary value. Ties and constant samples are
allowed except where stated. Inputs and instance state are unchanged.
Continuous-null calibration assumes iid observations; rounded data
require calibration of the observation process.

## Parameters

```text
parameters : ParameterValues
    Values with alfa, beta fixed; omitted parameters are unknown.
```

## Usage

```python
from pysatl_criterion.distribution.distributions import GammaDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit.gamma import ProbabilityPlotCorrelationGammaGofStatistic

statistic = ProbabilityPlotCorrelationGammaGofStatistic(GammaDistributionDescriptor.DEFAULT.parse({'alfa': 1, 'beta': 1}))
value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
```

Returns a scalar; large values oppose the null. Invalid samples raise `ValueError`.
See the [audit](../../../gamma-statistics-audit.md) for calibration changes.
