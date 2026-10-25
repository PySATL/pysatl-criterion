# Cadirci kNN Rényi-entropy criterion for Student distributions

[All Student criteria](index.md) · [Mathematical verification](mathematical-validation.md)

`CadirciRenyiStudentGofStatistic` implements equation (20) of
[Çadirci, Evans, Leonenko, Makogin and Seleznjev (2025)](https://arxiv.org/abs/2502.08654)
for dimension one.

## Hypothesis and statistic

Fix `df = ν > 2`; location and scale remain unknown. Set `q=1-2/(ν+1)` and use
sample variance with denominator **n−1**:

$$
\widehat\sigma^2=(1-2/\nu)\frac{\sum_j(X_j-\bar X)^2}{n-1}.
$$

Let `ρ_i` be the distance to observation i's kth nearest neighbour, excluding
itself. With `C_k=[Γ(k)/Γ(k+1−q)]^{1/(1−q)}`, estimate entropy by

$$
\widehat H_q=\frac{1}{1-q}\log\left[
\frac1n\sum_i\{2(n-1)C_k\rho_i\}^{1-q}\right].
$$

The statistic is `W=H_q(t_ν(scale=σ̂))−Ĥ_q`. The theoretical entropy is evaluated
analytically. Calculations use unit sample variance to avoid overflow; in one
dimension the two entropy scale terms cancel exactly. This preserves the paper's
unscaled statistic and uses the original Euclidean neighbour distances up to a
common scale factor.

## Use and calibration

```python
from pysatl_criterion.distribution import StudentDistributionDescriptor as Student
from pysatl_criterion.statistics.goodness_of_fit.continuous.student import (
    CadirciRenyiStudentGofStatistic,
)

criterion = CadirciRenyiStudentGofStatistic(Student.DEFAULT.parse({'df': 5}), k=3)
value = criterion.execute_statistic([-3.2, -1.1, -0.3, 0.2, 0.7, 1.8, 4.1])
```

`k` defaults to 3 and must satisfy `1 <= k < n`. Samples must be finite,
one-dimensional and contain distinct observations; ties raise `ValueError`.
Large values reject. **Negative finite-sample values are valid** and remain
unclipped. No parameter estimates or sample data are retained.

Monte Carlo calibration uses the fixed `df`, the same `k`, and recomputes the
sample variance and neighbours on every replicate. The built-in sampler chooses
location zero and scale one. There is no fitted `df` or universal chi-square
reference distribution. The formula is available for `df > 2`; the stronger
moment conditions for mean-square convergence in Theorem 6 must be checked
separately and are not claimed throughout this whole interval.

Code: `CADIRCI_RENYI_STUDENT_GOODNESS_OF_FIT`.
