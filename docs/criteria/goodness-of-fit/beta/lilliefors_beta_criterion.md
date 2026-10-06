# Fitted-Beta KS (Lilliefors-type) test

## Description

Both shape parameters are unknown; location 0 and scale 1 are fixed.
The implementation fits both shapes by maximum likelihood on every call and
computes the two-sided KS distance from the fitted CDF. Construction takes no
fixed `alpha` or `beta` arguments. `hypothesis().parameters()` is `{}`.
At least two distinct finite observations strictly inside `(0, 1)` are required.
Boundary observations are rejected rather than silently clipped.

## Usage

```python
from pysatl_criterion.statistics.goodness_of_fit.beta import LillieforsTestBetaGofStatistic

sample = [0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57]
statistic = LillieforsTestBetaGofStatistic()
distance = statistic.execute_statistic(sample)
print(distance)
```

## Calibration

The class implements the same `execute_statistic(rvs)` contract as every other
Beta statistic: it returns a scalar distance. Parameter estimation takes place
inside each call; the instance stores no fitted state. It does not calculate
p-values or simulate calibration samples.

Calibration belongs to the hypothesis-testing layer. For a parametric bootstrap,
that layer must choose the fitted generating shapes and call `execute_statistic`
on every generated sample, thereby refitting both shapes on every replicate.
Ordinary KS tables and simulation under an arbitrary Beta(1,1) distribution do
not provide the correct fitted calibration. The generic hypothesis sampler
cannot choose simulation shapes from an empty composite hypothesis without
observations and therefore raises an explicit error.

## Source and terminology

Ebner and Liebenberg study this procedure as the **KS competitor** in Section 4;
it is not the new conditional-moment statistic proposed in that article.
The historical class name uses “Lilliefors-type” for the fitted-CDF principle.
The source discusses Beta with estimated shapes, rather than a normality test.
The article discusses bootstrap critical quantiles; this statistic class
implements the fitted KS distance used in that procedure.

## References

B. Ebner and S. C. Liebenberg (2021),
[On a new test of fit to the beta distribution](https://doi.org/10.1002/sta4.341),
Stat 10, e341, Sections 2 and 4.
[Open preprint](https://arxiv.org/pdf/2009.13995).
