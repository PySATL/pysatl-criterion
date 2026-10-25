# Liao–Shimokawa functional (legacy MinToshiyuki name) for Pareto I

Public class: `MinToshiyukiParetoGofStatistic`. Full code: `MT_PARETO_GOODNESS_OF_FIT`.

## Hypothesis and statistic

Both $a=\mathrm{shape}>0$ and $s=\mathrm{scale}>0$ are fixed before observing the sample.
Location is zero; $F(x)=1-(s/x)^a$ for $x\ge s$. No parameters are estimated.

For sorted data set $u_i=F(x_{(i)})$,
$D^+=\max_i(i/n-u_i)$ and $D^-=\max_i(u_i-(i-1)/n)$.

$$
L_n=\frac1{\sqrt n}\sum_{i=1}^n\frac{\max(i/n-u_i,u_i-(i-1)/n)}{\sqrt{u_i(1-u_i)}}.
$$

The public class and code retain their legacy names. The shared kernel is Liao–Shimokawa. Here it is applied to a fully specified Pareto CDF; the fitted Weibull/extreme-value critical values in the original paper do not apply. At the lower endpoint the value is positive infinity. Extreme finite penalties can overflow float64.

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
from pysatl_criterion.statistics.goodness_of_fit.continuous.pareto import MinToshiyukiParetoGofStatistic

statistic = MinToshiyukiParetoGofStatistic(Pareto.DEFAULT.parse({"shape": 2, "scale": 1}))
value = statistic.execute_statistic([1.1, 1.4, 1.9, 2.7, 4.2, 8.0])
print(value)
```

## Scientific sources

[Liao and Shimokawa (1999), A new goodness-of-fit test for Type-I extreme-value and 2-parameter Weibull distributions with estimated parameters](https://doi.org/10.1080/00949659908811965). The expression was also checked in [Al-Zahrani (2012), Goodness-of-Fit for the Topp-Leone Distribution with Unknown Parameters, eq. (13)](https://www.m-hikari.com/ams/ams-2012/ams-125-128-2012/alzahraniAMS125-128-2012.pdf). These sources do not supply Pareto-specific fitted-model calibration.

[All Pareto criteria](index.md) · [Mathematical verification](mathematical-validation.md).
