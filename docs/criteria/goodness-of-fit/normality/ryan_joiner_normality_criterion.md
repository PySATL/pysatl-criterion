# Ryan-Joiner test for normality

## Description

Performs the Ryan-Joiner goodness-of-fit test for the hypothesis of normality.
The null hypothesis is the normal family with unknown mean and variance.

Hypothesis of Normality
The population mean and variance are free parameters of the normal family.
The statistic estimates them from the sample or eliminates them through
location- and scale-invariant calculations. No mean or variance is fixed in
advance, so `hypothesis().parameters()` returns `{}`. Estimates computed from
observations are not added to that dictionary.

Test Statistic
The statistic is based on the correlation between ordered observations and expected normal scores.

## Usage

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    RyanJoinerNormalityGofStatistic,
)


test_statistic = RyanJoinerNormalityGofStatistic(weighted=False, cte_alpha="3/8")
assert test_statistic.hypothesis().parameters() == {}
statistic_result = test_statistic.execute_statistic([-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43])
print(statistic_result)
```

## Arguments

`weighted` and `cte_alpha` configure the statistic, not the normal distribution.

`weighted` - whether tied observations use grouped plotting positions. Default value is `False`.

`cte_alpha` - plotting-position constant. Supported values include `3/8`, `1/2`, and `0`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details

The null allows arbitrary normal mean and positive variance;
`hypothesis().parameters()` is empty. The two constructor arguments
configure the statistic, not the distribution under the null.

The statistic is the Pearson correlation of sorted observations with
their normal scores. Smaller values indicate poorer fit, and
`alternative()` returns `LeftAlternative`. Supply finite,
nonconstant data with at least three observations for a nontrivial
result. Calibration must use the same plotting-position and tie options.

## References

1. Ryan, T. A., Jr. and Joiner, B. L. (1976). Normal Probability Plots and Tests for Normality. Technical report, Statistics Department, The Pennsylvania State University. Original report: [Source](https://www.additive-net.de/de/component/jdownloads/send/70-support/236-normal-probability-plots-and-tests-for-normality-thomas-a-ryan-jr-bryan-l-joiner)

## Author(s)

Alexey Mironov

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    RyanJoinerNormalityGofStatistic,
)


test_statistic = RyanJoinerNormalityGofStatistic(weighted=False, cte_alpha="3/8")
assert test_statistic.hypothesis().parameters() == {}
statistic_result = test_statistic.execute_statistic([-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43])
print(statistic_result)
```
