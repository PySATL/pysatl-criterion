# Riad–Abd El Mabood Beta criterion

`RiadAbdElMaboodBetaGofStatistic` (`RF_BETA_GOODNESS_OF_FIT`) tests the Beta family on the
fixed support $[0,1]$. Both shapes are unknown and fitted by maximum likelihood.
Reject large statistic values. See the [catalog](index.md) and
[verification report](mathematical-validation.md).

## Definition

For fitted shapes $\hat\alpha,\hat\beta$, define

$$
L_n(t)=\frac1n\sum_i e^{-tx_i},\qquad
D_n(t)=\frac{t}{\hat\alpha}L_n''(t)
 +\left(1+\frac{\hat\beta}{\hat\alpha}+\frac{t}{\hat\alpha}\right)L_n'(t)+L_n(t).
$$

The statistic, with `weight` $\rho>0$, is

$$
T_{n,\rho}=n\int_0^\infty D_n(t)^2e^{-\rho t}\,dt.
$$

Put $A_i=x_i(x_i-1)/\hat\alpha$, $B_i=1-(1+\hat\beta/\hat\alpha)x_i$
and $s_{ij}=x_i+x_j+\rho$. Exact integration gives

$$
T_{n,\rho}=\frac1n\sum_{i,j}\left[
\frac{2A_iA_j}{s_{ij}^3}+\frac{A_iB_j+A_jB_i}{s_{ij}^2}
+\frac{B_iB_j}{s_{ij}}\right].
$$

`weight=2.0` is the library default, one of the values studied by Ebner and
Liebenberg. Fix it before examining the sample. The fitted mean is
$\hat\alpha/(\hat\alpha+\hat\beta)$, not the arithmetic sample mean.
The printed expanded RF formula in their Section 4 has an apparent extra
factor $\hat\alpha$ in the cross-term denominator. This implementation follows
the defining differential equation above; it does not reproduce that factor.

Samples require at least two distinct finite observations strictly inside
$(0,1)$. Invalid weights and samples raise `ValueError`. Analytic evaluation
uses compensated summation in $O(n^2)$ time and $O(n)$ memory; severe cancellation
triggers scaled quadrature of the nonnegative integrand. Numerical failures
raise errors rather than yielding a negative squared norm.

## Scientific sources

- Riad, M.; Abd El Mabood, O. F. (2018). *A new goodness of fit test for the beta distribution based on the empirical Laplace transform*. Advances and Applications in Statistics 53(2), 165–177. [DOI](https://doi.org/10.17654/AS053020165).
- Ebner, B.; Liebenberg, S. C. (2021). *On a new test of fit to the beta distribution*, Section 4. [DOI](https://doi.org/10.1002/sta4.341), [open published text](https://publikationen.bibliothek.kit.edu/1000134155/118245046).

The original 2018 full text was not accessible during this audit. The
characterization and MLE convention were checked against the published 2021
comparison; the integral was expanded independently.

## Example and calibration

```python
from pysatl_criterion.distribution import BetaDistributionDescriptor as Beta
from pysatl_criterion.statistics.goodness_of_fit.continuous.beta import (
    RiadAbdElMaboodBetaGofStatistic,
)
from pysatl_criterion.hypothesis_testing.beta_bootstrap import parametric_bootstrap_beta

sample = [0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57]
criterion = RiadAbdElMaboodBetaGofStatistic(Beta.DEFAULT.parse({}), weight=2.0)
value = criterion.execute_statistic(sample)
result = parametric_bootstrap_beta(
    criterion, sample, n_resamples=99, random_state=42,
)
print(value, result.p_value)
```

The small bootstrap count keeps this example fast. Use more replicates for
inference. Calibration is approximate and shape-dependent: every replicate
refits both shapes and retains the same settings. `execute_statistic` returns
only the statistic; stored universal critical values are unsupported.
