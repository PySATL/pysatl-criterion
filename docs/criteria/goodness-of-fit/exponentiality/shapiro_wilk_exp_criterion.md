# Shapiro–Wilk exponential W exponentiality criterion

`ShapiroWilkExponentialityGofStatistic` — short code `SW`; both tails.

[All exponential criteria](index.md) · [Mathematical verification](mathematical-validation.md)

## Hypothesis and formula

The model is $F(x)=1-\exp(-\lambda x)$ for $x\ge0$, with known origin zero and $\lambda>0$.
Unknown rate only: pass `E.DEFAULT.parse({})`.
Omitted parameters remain unknown. Where fitting is needed, $\widehat\lambda=1/\bar x$
is recomputed for each sample; no fitted state is retained.

Write $y_i=x_i/\bar x$, $u_i=1-\exp(-\lambda x_{(i)})$ with the specified or
fitted rate, $D^+=\max_i(i/n-u_i)$ and $D^-=\max_i(u_i-(i-1)/n)$.
For spacing statistics, $d_i=(n-i+1)(x_{(i)}-x_{(i-1)})$ and $x_{(0)}=0$.
For frequency statistics, $O_j$ and $E_j=np_j>0$ are aligned counts.

$$
\frac{n(\bar x-x_{(1)})^2}{(n-1)\sum_i(x_i-\bar x)^2}
$$

n>=3 and nonconstant observations; both tails. Translation invariance means that this statistic cannot detect a positive shift of an exponential distribution, despite the zero-origin input model. This is not the normality Shapiro–Wilk statistic.

## Input and calibration

Observations must be finite, real, nonnegative and one-dimensional. Masked
observations are rejected. Fixed-rate generated criteria accept $n\ge1$;
unknown-rate EDF/frequency criteria require $n\ge2$ and positive total.
Specialized criteria may require more observations, distinct spacings or a
nonconstant sample, as stated above and in the class contract. Ties are not
silently jittered. Infinite mathematical boundary penalties are retained.

`execute_statistic` returns a statistic, not a p-value. Use
`monte_carlo_exponential` with the same sample size and constructor settings.
It refits each simulated sample for a composite hypothesis. Rescaling to
Exp(1) is justified only for a scale-invariant statistic. A continuous-null
calibration does not model rounding, censoring or dependence.

## Runnable example

```python
from pysatl_criterion.distribution import ExponentialDistributionDescriptor as E
from pysatl_criterion.statistics.goodness_of_fit.continuous.exponent import (
    ShapiroWilkExponentialityGofStatistic,
)

sample = [0.12, 0.30, 0.55, 0.90, 1.40, 2.10, 3.20, 4.80]
criterion = ShapiroWilkExponentialityGofStatistic(E.DEFAULT.parse({}))
value = criterion.execute_statistic(sample)
print(value)
```

## Scientific references and verification status

[Shapiro and Wilk (1972), An analysis of variance test for the exponential distribution (complete samples)](https://doi.org/10.1080/00401706.1972.10488921).

[Spinelli and Stephens (1987), Tests for exponentiality when origin and scale parameters are unknown, §3](https://www.stat.cmu.edu/technometrics/80-89/VOL-29-04/v2904471.pdf).

The formula and input behavior are tested independently; that does not by
itself certify a historical name, an asymptotic approximation or published
critical values. Source qualifications above are part of this criterion's
contract. See the [audit](mathematical-validation.md) for evidence and open items.
