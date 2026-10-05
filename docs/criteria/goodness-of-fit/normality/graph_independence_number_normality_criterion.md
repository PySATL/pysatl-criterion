# Graph independence number test for normality

## Description

Performs the Graph independence number goodness-of-fit test for the hypothesis of normality.
The null hypothesis is that the sample comes from a normal distribution with
unknown mean and the specified variance.

Hypothesis of Normality
The population variance is fixed in advance by the keyword-only `var` argument;
the population mean remains unrestricted. `hypothesis().parameters()` returns
`{"var": var}`. The empirical variance used to construct the graph is calculated
from the observations and is distinct from this fixed null parameter.

Test Statistic
The statistic is based on the independence number of a proximity graph built from the sample.

## Usage

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    GraphIndependenceNumberNormalityGofStatistic,
)


test_statistic = GraphIndependenceNumberNormalityGofStatistic(var=1)
assert test_statistic.hypothesis().parameters() == {"var": 1}
statistic_result = test_statistic.execute_statistic(
    [-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43]
)
print(statistic_result)
```

## Arguments

`var` - positive, finite normal variance used for calibration (not standard
deviation). This optional constructor argument must be supplied by name, for
example `var=4`; its default value is `1`. Invalid values raise `ValueError`
during construction. The constructor does not accept `mean`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details

The graph connects observations whose absolute difference is strictly less than
`(max(rvs) - min(rvs)) / (10 * np.var(rvs, ddof=0))`. If the sample variance is
zero, the threshold is the sample range divided by ten.

This construction is invariant to a shift of the sample, but changes under
rescaling. `var` therefore determines the null distribution for Monte Carlo
calibration; it does not enter the statistic calculation directly. For a given
sample, changing `var` leaves the statistic value unchanged but requires a
matching calibration distribution. The alternative is two-sided.

The threshold rule is documented from the PySATL implementation. An originating
research article has not been identified.

## References

1. PySATL. "pysatl-criterion", normal proximity-graph implementation. [Source](https://github.com/PySATL/pysatl-criterion)

## Author(s)

Alexey Mironov

## Examples

Use a specified variance of `4` (standard deviation `2`) and an unknown mean:

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    GraphIndependenceNumberNormalityGofStatistic,
)


test_statistic = GraphIndependenceNumberNormalityGofStatistic(var=4)
assert test_statistic.hypothesis().parameters() == {"var": 4}
statistic_result = test_statistic.execute_statistic(
    [-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43]
)
print(statistic_result)
```
