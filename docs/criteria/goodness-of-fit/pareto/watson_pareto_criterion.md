# Watson for Pareto I

Public class: `WatsonParetoGofStatistic`. Full code: `WATSON_PARETO_GOODNESS_OF_FIT`.

## Hypothesis and statistic

Both $a=\mathrm{shape}>0$ and $s=\mathrm{scale}>0$ are fixed before observing the sample.
Location is zero; $F(x)=1-(s/x)^a$ for $x\ge s$. No parameters are estimated.

For sorted data set $u_i=F(x_{(i)})$,
$D^+=\max_i(i/n-u_i)$ and $D^-=\max_i(u_i-(i-1)/n)$.

$$
U^2=W^2-n(\bar u-1/2)^2.
$$

The implementation evaluates centered squared deviations plus $1/(12n)$ to avoid cancellation. It applies no small-sample correction. At $n=1$ the value is always $1/12$, so the statistic is uninformative.

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
from pysatl_criterion.statistics.goodness_of_fit.continuous.pareto import WatsonParetoGofStatistic

statistic = WatsonParetoGofStatistic(Pareto.DEFAULT.parse({"shape": 2, "scale": 1}))
value = statistic.execute_statistic([1.1, 1.4, 1.9, 2.7, 4.2, 8.0])
print(value)
```

## Scientific sources

[Watson (1961), Goodness-of-fit tests on a circle](https://doi.org/10.1093/biomet/48.1-2.109). Computational form: [Lemeshko et al. (2014), Solving Problems of Using Some Nonparametric Goodness-of-Fit Tests, eqs. (1), (3), (6)](https://doi.org/10.3103/S875669901401004X); [author-hosted full text](https://ami.nstu.ru/~headrd/seminar/publik_html/Awt_1_2014_eng.pdf).

[All Pareto criteria](index.md) · [Mathematical verification](mathematical-validation.md).
