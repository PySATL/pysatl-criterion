# Size of the largest independent set (n for an edgeless graph).

Size of the largest independent set (n for an edgeless graph).
Vertices are observations transformed by the fixed Gamma CDF.
An edge joins i and j exactly when abs(u_i-u_j) < h, with
h=(max(u)-min(u))/10. This uses CDF positions, not their spacings.
Ties are separate vertices; h=0 yields an edgeless graph.
Both tails are selected as a local convention; calibrate this exact
graph construction. No primary source establishing this Gamma-specific
procedure or its power properties was found. Floating-point saturation
of the CDF can merge distinct tail values.

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
from pysatl_criterion.statistics.goodness_of_fit.gamma import GraphIndependenceNumberGammaGofStatistic

statistic = GraphIndependenceNumberGammaGofStatistic(GammaDistributionDescriptor.DEFAULT.parse({'alfa': 1, 'beta': 1}))
value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
```

Returns a scalar; both tails are used. Invalid samples raise `ValueError`.
See the [audit](../../../gamma-statistics-audit.md) for calibration changes.
