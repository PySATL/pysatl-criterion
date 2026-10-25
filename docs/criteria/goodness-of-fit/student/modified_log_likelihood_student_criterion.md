# Modified log-likelihood for a specified Student distribution

[All Student criteria](index.md) · [Mathematical verification](mathematical-validation.md)

Class: `ModifiedLogLikelihoodStudentGofStatistic`. Code: `MODIFIED_LOG_LIKELIHOOD_STUDENT_GOODNESS_OF_FIT`.

## Hypothesis and formula

Fix all three parameters: `df > 0`, finite `loc`, and `scale > 0`.
Use `StudentDistributionDescriptor.DEFAULT.parse({'df': 5, 'loc': 0, 'scale': 1})`.
Omitted parameters remain unknown and are not supported by this criterion.
Write $O_j$ for observed bin counts and $E_j=np_j>0$ for expected counts under the specified Student model.

$$
MLL=2\sum_j E_j\log(E_j/O_j)
$$

Any $O_j=0$ with $E_j>0$ gives positive infinity. This is the reverse log-likelihood statistic, with expected counts as weights.

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
    ModifiedLogLikelihoodStudentGofStatistic,
)

parameters = Student.DEFAULT.parse({'df': 5, 'loc': 0, 'scale': 1})
statistic = ModifiedLogLikelihoodStudentGofStatistic(parameters, binning=EqualProbabilityBins(count=4))
value = statistic.execute_statistic([-3.2, -1.1, -0.3, 0.2, 0.7, 1.8, 4.1])
print(value)
```

## Scientific source

[Cressie and Read (1984), Multinomial Goodness-of-Fit Tests](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x), $\lambda=-1$.
