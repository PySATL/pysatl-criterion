# Vasicek–Song fixed-window KL statistic for Pareto I

Public class: `LequesneKlParetoGofStatistic`. Full code: `Lequesne_KL_PARETO_GOODNESS_OF_FIT`.

## Hypothesis and statistic

Both shape and scale are unknown. Each call fits

$$
\widehat s=x_{(1)},\qquad
\widehat a=\left[\frac1n\sum_i\log(x_i/\widehat s)\right]^{-1}.
$$

For ordered data, extend indices outside $1,\ldots,n$ by the corresponding endpoint.
With integer window $1\le m<n/2$, define

$$
V_{mn}=\frac1n\sum_i\log\left[\frac n{2m}(x_{(i+m)}-x_{(i-m)})\right],
$$

$$
I_{mn}=-V_{mn}-\log\widehat a+\log\widehat s
 +(\widehat a+1)\frac1n\sum_i\log(x_i/\widehat s).
$$

This equals the negative Vasicek entropy estimate minus the mean fitted log density.
It is the **raw fixed-window** statistic. It does not implement adaptive window
selection, the entropy constraint or the bias-corrected p-value procedure in
`vsgoftest`. Negative estimates are retained; reject for large values only.

## Input and calibration

Require finite real unmasked one-dimensional positive nonconstant data with $n\ge4$.
A constant sample has no finite shape MLE and raises `ValueError`. A zero window
spacing returns positive infinity. The default window is
`min(int(sqrt(n)), (n - 1) // 2)`. Pass a different integer `m` to `execute_statistic`;
booleans, nonintegers and values outside the stated range are rejected.
Fits are local to each execution and do not change the declared hypothesis.

The statistic is scale invariant but its finite-sample law depends on shape.
External simulation must refit both parameters in every replicate using the
same window rule. A plug-in shape bootstrap is approximate. Built-in Pareto
simulation is unavailable and generic storage calibration is blocked because
its identity does not encode the required shape and execution-time window.
No claim of an exact finite-sample test of the whole family is made.

## Example

```python
from pysatl_criterion.distribution import ParetoDistributionDescriptor as Pareto
from pysatl_criterion.statistics.goodness_of_fit.continuous.pareto import LequesneKlParetoGofStatistic

statistic = LequesneKlParetoGofStatistic(Pareto.DEFAULT.parse({}))
value = statistic.execute_statistic([1.1, 1.4, 1.9, 2.7, 4.2, 8.0], m=2)
print(value)
```

## Scientific source

[Lequesne and Regnault (2020), vsgoftest: An R Package for Goodness-of-Fit Testing
Based on Kullback–Leibler Divergence](https://doi.org/10.18637/jss.v096.c01).
The [author preprint, §2, eqs. (5) and (8)](https://arxiv.org/html/1806.07244)
gives the entropy estimator and raw KL estimate. The Pareto specialization here
follows by inserting its density and maximum-likelihood estimates.

[All Pareto criteria](index.md) · [Mathematical verification](mathematical-validation.md).
