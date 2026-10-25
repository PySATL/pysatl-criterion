# Modified log-likelihood for Pareto I

Public class: `ModifiedLogLikelihoodParetoGofStatistic`. Full code: `MODIFIED_LOG_LIKELIHOOD_PARETO_GOODNESS_OF_FIT`.

## Hypothesis and statistic

Both $a=\mathrm{shape}>0$ and $s=\mathrm{scale}>0$ are fixed before observing the sample.
Location is zero; $F(x)=1-(s/x)^a$ for $x\ge s$. No parameters are estimated.

For full-support edges $b_0,\ldots,b_k$, let $O_j$ be the observed counts and
$E_j=n[F(b_j)-F(b_{j-1})]$. Expected counts must be positive and totals must agree.

$$
T=2\sum_j E_j\log(E_j/O_j).
$$

Power-divergence parameter $\lambda=-1$. Any empty observed cell with positive expected count gives positive infinity. This is the reverse likelihood discrepancy, not the usual $2\sum O_j\log(O_j/E_j)$.

Input is a finite real one-dimensional sample with $n\ge1$ and $x_i\ge s$.
Masked observations and data below scale are rejected. The method returns one
scalar statistic; large values reject. Ties are accepted computationally, but
continuous-null calibration does not account for rounding or discrete data.

## Binning and calibration

The constructor accepts keyword-only `binning`. The default is
`EqualProbabilityBins(PowerCount(2.0, 0.4), min_expected=None, max_bins=1000)`.
It counts transformed values in equal-probability cells and sets $E_j=n/k$.
There must be at least two retained cells. See [binning](../binning/index.md).
For fixed cells and large expected counts, the usual asymptotic reference is
$\chi^2_{k-1}$; this is not an exact small-sample calibration. Sparse cells,
merging and adaptive rules require calibration reproducing the same policy.
Binning settings are included in `configuration()`.

Built-in Pareto Monte Carlo generation is unavailable; calibration is external.

## Example

```python
from pysatl_criterion.distribution import ParetoDistributionDescriptor as Pareto
from pysatl_criterion.statistics.goodness_of_fit.continuous.pareto import ModifiedLogLikelihoodParetoGofStatistic
from pysatl_criterion.statistics.goodness_of_fit.binning import ExplicitBins

statistic = ModifiedLogLikelihoodParetoGofStatistic(
    Pareto.DEFAULT.parse({"shape": 2, "scale": 1}),
    binning=ExplicitBins((1, 1.5, 3, float("inf"))),
)
value = statistic.execute_statistic([1.1, 1.4, 1.9, 2.7, 4.2, 8.0])
print(value)
```

## Scientific sources

[Cressie and Read (1984), Multinomial Goodness-of-Fit Tests](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x). The formula is the corresponding member of their power-divergence family.

[All Pareto criteria](index.md) · [Mathematical verification](mathematical-validation.md).
