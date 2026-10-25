# Anderson–Darling for Pareto I

Public class: `AndersonDarlingParetoGofStatistic`. Full code: `AD_PARETO_GOODNESS_OF_FIT`.

## Hypothesis and statistic

Both $a=\mathrm{shape}>0$ and $s=\mathrm{scale}>0$ are fixed before observing the sample.
Location is zero; $F(x)=1-(s/x)^a$ for $x\ge s$. No parameters are estimated.

For sorted data set $u_i=F(x_{(i)})$,
$D^+=\max_i(i/n-u_i)$ and $D^-=\max_i(u_i-(i-1)/n)$.

$$
A^2=-n-\frac1n\sum_{i=1}^n(2i-1)[\log u_i+\log(1-u_{n+1-i})].
$$

No finite-sample correction is applied. An observation at the lower endpoint gives positive infinity. Log probabilities preserve finite upper-tail contributions when the CDF rounds to one.

Input is a finite real one-dimensional sample with $n\ge1$ and $x_i\ge s$.
Masked observations and data below scale are rejected. The method returns one
scalar statistic; large values reject. Ties are accepted computationally, but
continuous-null calibration does not account for rounding or discrete data.

## Calibration

The probability integral transform gives the same finite-sample statistic law
as independent uniform observations when both Pareto parameters are truly fixed.
Do not substitute estimates from this sample and then use the fixed-null tables.
Use critical values for this precise normalization and sample size.

Built-in Pareto Monte Carlo generation is unavailable; calibration is external.

## Example

```python
from pysatl_criterion.distribution import ParetoDistributionDescriptor as Pareto
from pysatl_criterion.statistics.goodness_of_fit.continuous.pareto import AndersonDarlingParetoGofStatistic

statistic = AndersonDarlingParetoGofStatistic(Pareto.DEFAULT.parse({"shape": 2, "scale": 1}))
value = statistic.execute_statistic([1.1, 1.4, 1.9, 2.7, 4.2, 8.0])
print(value)
```

## Scientific sources

[Anderson and Darling (1952), Asymptotic Theory of Certain Goodness of Fit Criteria Based on Stochastic Processes](https://doi.org/10.1214/aoms/1177729437). Computational form: [Zhang (2002), §§2–3, Powerful goodness-of-fit tests based on the likelihood ratio](https://doi.org/10.1111/1467-9868.00337); [full text](https://anakena.dcc.uchile.cl/~mnmonsal/eso.pdf).

[All Pareto criteria](index.md) · [Mathematical verification](mathematical-validation.md).
