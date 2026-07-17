# Kimber-Michael test for exponentiality

## Description
Performs Kimber-Michael test for the composite hypothesis of exponentiality, see e.g. Michael (1983) and Kimber (1985).
The Kimber-Michael test is a statistical hypothesis test used to assess the composite hypothesis of exponentiality. This test is based on the stabilized probability plot and is designed to determine whether a given sample of data is consistent with an exponential distribution.

Composite Hypothesis of Exponentiality
The composite hypothesis of exponentiality refers to the null hypothesis that the data comes from an exponential distribution with an unspecified rate parameter
lambda. The Kimber-Michael statistic removes the unknown exponential scale by evaluating the exponential distribution at observations normalized by the sample mean.

Test Statistic
The Kimber-Michael test statistic is based on an arcsine square-root transformation of probabilities. This transformation stabilizes the variance of probability plot points, and the statistic measures the maximum absolute distance between transformed empirical plotting positions and transformed fitted exponential probabilities.

Calculate the Test Statistic: The observations are sorted, divided by the sample mean, transformed through the fitted exponential distribution function, mapped to the stabilized probability scale, and compared with the stabilized plotting positions.

Limitations
The test assumes nonnegative lifetime-type observations.

The statistic is a supremum-type distance, so it is most sensitive to the largest stabilized probability plot discrepancy.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    KimberMichaelExponentialityGofStatistic,
)


test_statistic = KimberMichaelExponentialityGofStatistic()
statistic_result = test_statistic.execute_statistic([1, 2, 3, 4, 5, 6, 7])
print(statistic_result)
```

## Arguments
`rvs` - array-like sample data passed to `execute_statistic`.

## Details

The Kimber-Michael test is a test for the composite hypothesis of exponentiality. Let
$X_{(1)} \leq \cdots \leq X_{(n)}$ denote the ordered sample. The test statistic implemented here is

$$ KM_n = \max_{1 \leq i \leq n} \left| \frac{2}{\pi}\arcsin\sqrt{\frac{i-1/2}{n}} - \frac{2}{\pi}\arcsin\sqrt{1-exp(-X_{(i)}/\overline{X})} \right|. $$

Here $n$ is the sample size and $\overline{X}$ is the sample mean.

The function

$$ p \mapsto \frac{2}{\pi}\arcsin\sqrt{p} $$

is the stabilized probability scale introduced by Michael (1983). Kimber (1985) applied this idea to tests for the exponential, Weibull, and Gumbel distributions.

The implementation uses a left-sided alternative, so unusually small values of $KM_n$ are treated as evidence in the direction used by this statistic.

## Author(s)
Lev Golofastov

## References
Michael, J.R. (1983): The stabilized probability plot. - Biometrika, vol. 70, no. 1, pp. 11-17. https://doi.org/10.1093/biomet/70.1.11

Kimber, A.C. (1985): Tests for the exponential, Weibull and Gumbel distributions based on the stabilized probability plot. - Biometrika, vol. 72, no. 3, pp. 661-663. https://doi.org/10.1093/biomet/72.3.661

Henze, N. and Meintanis, S.G. (2005): Recent and classical tests for exponentiality: a partial review with comparisons. - Metrika, vol. 61, pp. 29-45.

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    KimberMichaelExponentialityGofStatistic,
)


test_statistic = KimberMichaelExponentialityGofStatistic()
statistic_result = test_statistic.execute_statistic([1, 2, 3, 4, 5, 6, 7])
print(statistic_result)
```
