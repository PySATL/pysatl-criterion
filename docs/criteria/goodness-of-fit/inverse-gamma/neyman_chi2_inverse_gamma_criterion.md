# Neyman modified chi-squared for inverse gamma

## API and hypothesis

`NeymanChi2InverseGammaGofStatistic` has code `CHI2_NEYMAN_INV_GAMMA_GOODNESS_OF_FIT`.

Both positive parameters must be fixed before observing the sample; location is zero. No parameter fitting is performed. The input must be a nonempty one-dimensional sample of finite, strictly positive observations. Ties and constant samples are accepted; rounded observations require appropriate calibration. The result is one scalar statistic, not a p-value. Large values reject the null.

Set `binning=EqualProbabilityBins(k)` to use a fixed number of cells. The default `EqualProbabilityBins()` uses the shared sample-size count rule, not a fixed eight cells. Here $O_j$ are observed counts and $E_j=n p_j>0$; for equal-probability cells $p_j=1/k$. The binning configuration is part of the storage identity. See [binning](../binning/index.md).

An empty observed cell returns **positive infinity**. This is the mathematical zero-count limit, not numerical failure.

For fixed cells and sufficiently large expected counts, the usual asymptotic reference has $k-1$ degrees of freedom. Sparse counts, adaptive edges or a growing cell count need calibration for the actual binning procedure.

## Formula

$$
T=\sum_{j=1}^k\frac{(O_j-E_j)^2}{O_j}.
$$

## Example

```python
import numpy as np
from pysatl_criterion.distribution import InverseGammaDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit.binning import EqualProbabilityBins
from pysatl_criterion.statistics.goodness_of_fit.continuous.inverse_gamma import (
    NeymanChi2InverseGammaGofStatistic,
)

parameters = InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": 3.0, "beta": 2.0})
statistic = NeymanChi2InverseGammaGofStatistic(parameters, binning=EqualProbabilityBins(4))
value = statistic.execute_statistic([0.35, 0.5, 0.7, 1.0, 1.5, 3.0])
assert not np.isnan(value)
```

## Scientific references

[Cressie and Read (1984), Multinomial Goodness-of-Fit Tests](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x), power-divergence family.

These references support the statistic or its general construction, not a test
with estimated inverse-gamma parameters. See the [family overview](index.md)
and [mathematical verification](mathematical-validation.md) for calibration
and numerical qualifications.
