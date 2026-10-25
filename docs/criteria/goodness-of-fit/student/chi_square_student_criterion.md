# Pearson chi-squared for a specified Student distribution

[All Student criteria](index.md) · [Mathematical verification](mathematical-validation.md)

Class: `ChiSquareStudentGofStatistic`. Code: `CHI2_PEARSON_STUDENT_GOODNESS_OF_FIT`.

## Hypothesis and formula

Fix all three parameters: `df > 0`, finite `loc`, and `scale > 0`.
Use `StudentDistributionDescriptor.DEFAULT.parse({'df': 5, 'loc': 0, 'scale': 1})`.
Omitted parameters remain unknown and are not supported by this criterion.
Write $O_j$ for observed bin counts and $E_j=np_j>0$ for expected counts under the specified Student model.

$$
X^2=\sum_j\frac{(O_j-E_j)^2}{E_j}
$$

An empty observed bin contributes $E_j$.

The keyword-only `binning` accepts a [BinningStrategy](../binning/index.md).
The default `EqualProbabilityBins()` selects its count from the sample size.
Use `EqualProbabilityBins(count=4)` for four cells of probability 1/4.
Its CDF-space intervals are left-closed/right-open, except that the last cell
also includes 1. `n_bins` is not a constructor argument.

## Input, result and calibration

`execute_statistic(rvs, **kwargs)` accepts a nonempty, finite, one-dimensional
real sample. Ties and constants are allowed. It returns one scalar without
changing the input or retaining fitted state. Large values reject
(`RightAlternative`). Extra execution keywords are ignored.

The shared binning pipeline constructs observed and expected counts on every
call. Infinite empty-cell penalties are valid statistic values, not invalid
samples. Additional binning constraints, such as `min_expected`, can reject
small samples. See the formula's zero-count convention above.

Preserve both sample size and binning strategy in Monte Carlo calibration.
For fixed equiprobable cells the null counts are multinomial; the usual
chi-square limit with number-of-cells minus one degrees of freedom is only
asymptotic. No automatic p-value or exact chi-square calibration is provided.
Other binning rules need their matching calibration. The source describes a
general count discrepancy applied here to Student probabilities.

## Example

```python
from pysatl_criterion.distribution import StudentDistributionDescriptor as Student
from pysatl_criterion.statistics.goodness_of_fit.binning import EqualProbabilityBins
from pysatl_criterion.statistics.goodness_of_fit.continuous.student import (
    ChiSquareStudentGofStatistic,
)

parameters = Student.DEFAULT.parse({'df': 5, 'loc': 0, 'scale': 1})
statistic = ChiSquareStudentGofStatistic(parameters, binning=EqualProbabilityBins(count=4))
value = statistic.execute_statistic([-3.2, -1.1, -0.3, 0.2, 0.7, 1.8, 4.1])
print(value)
```

## Scientific source

[Pearson (1900), On the criterion that a given system of deviations from the probable …](https://doi.org/10.1080/14786440009463897); [Cressie and Read (1984), Multinomial Goodness-of-Fit Tests](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x).

## Author(s)

Dmitriy Rusanov, Alexey Mironov
