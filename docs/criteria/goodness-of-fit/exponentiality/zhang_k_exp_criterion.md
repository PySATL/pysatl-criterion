# Zhang Z_K exponentiality criterion

`ZhangKExponentialityGofStatistic` — short code `ZHANG_ZK`; right tail.

[All exponential criteria](index.md) · [Mathematical verification](mathematical-validation.md)

## Hypothesis and formula

The model is $F(x)=1-\exp(-\lambda x)$ for $x\ge0$, with known origin zero and $\lambda>0$.
Fixed or unknown rate: pass `E.DEFAULT.parse({"lam": rate})` or `E.DEFAULT.parse({})`, respectively.
Omitted parameters remain unknown. Where fitting is needed, $\widehat\lambda=1/\bar x$
is recomputed for each sample; no fitted state is retained.

Write $y_i=x_i/\bar x$, $u_i=1-\exp(-\lambda x_{(i)})$ with the specified or
fitted rate, $D^+=\max_i(i/n-u_i)$ and $D^-=\max_i(u_i-(i-1)/n)$.
For spacing statistics, $d_i=(n-i+1)(x_{(i)}-x_{(i-1)})$ and $x_{(0)}=0$.
For frequency statistics, $O_j$ and $E_j=np_j>0$ are aligned counts.

$$
\max_i\left[r_i\log\frac{r_i}{nu_i}+(n-r_i)\log\frac{n-r_i}{n(1-u_i)}\right],\quad r_i=i-1/2
$$

Equation (3.1). Endpoints give positive infinity. Fitted exponential calibration must repeat fitting.

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
    ZhangKExponentialityGofStatistic,
)

sample = [0.12, 0.30, 0.55, 0.90, 1.40, 2.10, 3.20, 4.80]
criterion = ZhangKExponentialityGofStatistic(E.DEFAULT.parse({}))
value = criterion.execute_statistic(sample)
print(value)
```

## Scientific references and verification status

[Zhang (2002), Powerful goodness-of-fit tests based on the likelihood ratio](https://doi.org/10.1111/1467-9868.00337).

The formula and input behavior are tested independently; that does not by
itself certify a historical name, an asymptotic approximation or published
critical values. Source qualifications above are part of this criterion's
contract. See the [audit](mathematical-validation.md) for evidence and open items.
