# Greenwood logarithmic residual ratio for Pareto I

Public class: `GreenwoodParetoGofStatistic`. Full code: `Greenwood_PARETO_GOODNESS_OF_FIT`.

## Hypothesis and statistic

Both shape and scale are unknown. For positive observations define

$$
y_i=\log(x_i/x_{(1)}),\qquad
G=\frac{\sum_i y_i^2}{(\sum_i y_i)^2}.
$$

The sample must be finite, real, unmasked, one-dimensional, nonconstant and have
$n\ge3$ observations. At $n=2$ the ratio is identically one, so this case is
rejected. Ties are accepted if the sample remains nonconstant. Small **and** large
values reject; the method returns the statistic, not a p-value.

## Derivation and calibration

Under Pareto I, $\log(X/s)$ is exponential with rate $a$. After subtracting the
minimum, one residual is zero and the other $n-1$ residuals, before ordering,
have the exponential memoryless law. Their normalized proportions follow
$\operatorname{Dirichlet}(1,\ldots,1)$ with $n-1$ components. Thus, by direct derivation,

$$
\frac1{n-1}\le G\le1,\qquad
\mathbb E G=\frac2n,\qquad
\operatorname{Var}(G)=\frac{4(n-2)}{n^2(n+1)(n+2)}.
$$

Scale cancels in the logarithmic ratios and shape cancels when normalizing.
The continuous IID null law is parameter-free. Calibrate both tails using this
$n-1$-component law or externally simulated Pareto samples. Ordinary Greenwood
CDF spacings from $n$ observations have $n+1$ components and different calibration.
Rounding and ties change the continuous-null law. Built-in Pareto simulation is unavailable.

## Example

```python
from pysatl_criterion.distribution import ParetoDistributionDescriptor as Pareto
from pysatl_criterion.statistics.goodness_of_fit.continuous.pareto import GreenwoodParetoGofStatistic

statistic = GreenwoodParetoGofStatistic(Pareto.DEFAULT.parse({}))
value = statistic.execute_statistic([1, 2, 4, 8])
assert abs(value - 14 / 36) < 1e-12
```

## Scientific source and attribution

[Does, Helmers and Klaassen (1988), Approximating the distribution of Greenwood's
statistic, eqs. (1)–(2)](https://ir.cwi.nl/pub/1694/1694D.pdf) describes the squared
spacing concentration functional. The logarithmic residual construction above
is a library adaptation with the direct derivation given here. An original paper
establishing this exact fitted-Pareto implementation was not located; the source
is not presented as such a claim.

[All Pareto criteria](index.md) · [Mathematical verification](mathematical-validation.md).
