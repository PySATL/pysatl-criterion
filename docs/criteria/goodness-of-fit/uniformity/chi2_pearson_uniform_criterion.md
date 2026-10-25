# Pearson chi-squared test for uniformity

## Description

`Chi2PearsonUniformGofStatistic` has algorithm code
`CHI2_PEARSON_UNIFORM_GOODNESS_OF_FIT`. See the [uniformity catalog](index.md).

Performs Pearson's chi-squared goodness-of-fit test for the hypothesis of uniformity on the interval $[a, b]$.
The implementation bins the sample and compares observed bin counts with equal expected counts.

Hypothesis of Uniformity
The null hypothesis is that the sample comes from a uniform distribution on the interval $[a, b]$.

## Usage
```python
from pysatl_criterion.distribution import UniformDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit.binning import EqualProbabilityBins, SturgesCount
from pysatl_criterion.statistics.goodness_of_fit.continuous import (
    Chi2PearsonUniformGofStatistic,
)


test_statistic = Chi2PearsonUniformGofStatistic(
    UniformDistributionDescriptor.DEFAULT.parse({"a": 0, "b": 1}),
    binning=EqualProbabilityBins(count=SturgesCount()),
)
statistic_result = test_statistic.execute_statistic([0.12, 0.25, 0.41, 0.53, 0.77, 0.91])
print(statistic_result)
```

## Arguments
`a` - explicitly fixed left boundary of the uniform distribution.

`b` - explicitly fixed right boundary of the uniform distribution.

`binning` - a binning strategy. The default is `EqualProbabilityBins()` with
`PowerCount(2, 0.4)`, `max_bins=1000` and no minimum expected-count constraint.

The old `bins` and `lambda_` options have been removed. Other divergences use
separate classes, such as `CressieReadUniformGofStatistic`.
See [frequency binning](../binning/index.md) for all strategies and
Monte Carlo calibration requirements.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The sample is divided into bins over $[a, b]$.
For $m$ equal-probability bins and sample size $n$, each bin has expected count $n / m$.
Other strategies calculate each expected count from its null probability.
The statistic compares observed and expected counts:

$$ X^2 = \sum_{j=1}^{m} \frac{(O_j-E_j)^2}{E_j}. $$

Zero observed counts contribute $E_j$. Large values indicate disagreement with the null.
Calibrations are distinguished by binning configuration; see
[calibration storage](../binning/calibration.md).

## Author(s)
Aleksandr Podmarev, Alexey Mironov

## References

Pearson, K. (1900). [On the criterion that a given system of deviations from the probable in the case of a correlated system of variables is such that it can be reasonably supposed to have arisen from random sampling](https://doi.org/10.1080/14786440009463897). *Philosophical Magazine*, 50, 157–175.

## Examples

### Constructing with `ParameterValues`

Create `ParameterValues` with both fixed boundaries, then pass it to
the constructor. It validates the supported parameterization and exact fixed
parameter set. Omitted bounds are not filled in:
a partial mapping such as `{"a": 0}` raises `ValueError` in the constructor.

Algorithm options (`binning` here) are passed separately, outside `ParameterValues`.

```python
from pysatl_criterion.distribution import UniformDistributionDescriptor as Uniform
from pysatl_criterion.statistics.goodness_of_fit.binning import EqualProbabilityBins, SturgesCount
from pysatl_criterion.statistics.goodness_of_fit.continuous import (
    Chi2PearsonUniformGofStatistic,
)


parameters = Uniform.DEFAULT.parse({"a": 0, "b": 1})
test_statistic = Chi2PearsonUniformGofStatistic(
    parameters, binning=EqualProbabilityBins(count=SturgesCount())
)
assert test_statistic.hypothesis().parameters() == {"a": 0, "b": 1}
statistic_result = test_statistic.execute_statistic([0.12, 0.25, 0.41, 0.53, 0.77, 0.91])
print(statistic_result)
```
