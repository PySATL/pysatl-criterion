# Zhang Z_A for Pareto I

Public class: `ZhangAParetoGofStatistic`. Full code: `ZHANG_ZA_PARETO_GOODNESS_OF_FIT`.

## Hypothesis and statistic

Both $a=\mathrm{shape}>0$ and $s=\mathrm{scale}>0$ are fixed before observing the sample.
Location is zero; $F(x)=1-(s/x)^a$ for $x\ge s$. No parameters are estimated.

For sorted data set $u_i=F(x_{(i)})$,
$D^+=\max_i(i/n-u_i)$ and $D^-=\max_i(u_i-(i-1)/n)$.

$$
Z_A=-\sum_i\left[\frac{\log u_i}{n-i+1/2}+\frac{\log(1-u_i)}{i-1/2}\right].
$$

The formula is eq. (3.2), without extra normalization. Log survival probabilities avoid upper-tail rounding. An observation at scale gives positive infinity.

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
from pysatl_criterion.statistics.goodness_of_fit.continuous.pareto import ZhangAParetoGofStatistic

statistic = ZhangAParetoGofStatistic(Pareto.DEFAULT.parse({"shape": 2, "scale": 1}))
value = statistic.execute_statistic([1.1, 1.4, 1.9, 2.7, 4.2, 8.0])
print(value)
```

## Scientific sources

[Zhang (2002), §§2–3, Powerful goodness-of-fit tests based on the likelihood ratio](https://doi.org/10.1111/1467-9868.00337); [full text](https://anakena.dcc.uchile.cl/~mnmonsal/eso.pdf).

[All Pareto criteria](index.md) · [Mathematical verification](mathematical-validation.md).
