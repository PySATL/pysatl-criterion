# Lequesne Rényi-entropy criterion for Student distributions

[All Student criteria](index.md) · [Mathematical verification](mathematical-validation.md)

`LequesneRenyiStudentGofStatistic` implements both statistics in
[Lequesne (2015), AIP Conference Proceedings 1641, 487–494](https://doi.org/10.1063/1.4906014),
equations (7)–(10).

## Hypotheses and formulas

With `Student.DEFAULT.parse({'df': nu})`, equation (7) gives

$$
T_1=h_q(\nu,\widehat\sigma^2)-\widehat H_{n,k,q},\qquad
q=\frac{\nu-1}{\nu+1},\qquad
\widehat\sigma^2=\frac{\nu-2}{\nu}\widehat\Sigma^2.
$$

The implementation chooses the unbiased sample variance (denominator `n−1`)
for `Σ̂²`. The article allows a consistent variance estimator without specifying
the denominator. With this choice, `T1` equals the univariate Çadirci statistic
at the same `df` and `k`.

With `Student.DEFAULT.parse({})`, equation (8) gives

$$
T_2=h_{\widehat q}(\widehat\nu,\widehat\sigma^2)
       -\widehat H_{n,k,\widehat q},\qquad
\widehat q=\frac{\widehat\nu-1}{\widehat\nu+1}.
$$

All three parameters are fitted numerically by maximum likelihood on each call.
The theoretical entropy uses the **MLE scale**, and both entropy terms use the
**fitted order**. The implementation checks optimizer convergence. It rescales
the sample before fitting for numerical stability; the common logarithmic scale
term cancels from the entropy difference. Estimates are not stored on the instance.

In both cases the sample entropy is the kth-nearest-neighbour estimate (10):

$$
\widehat H_{n,k,q}=\frac{1}{1-q}\log\left[
\frac1n\sum_i\{2(n-1)G_k\rho_i^{(k)}\}^{1-q}\right],\qquad
G_k=\left[\frac{\Gamma(k)}{\Gamma(k+1-q)}\right]^{1/(1-q)}.
$$

## Usage

```python
from pysatl_criterion.distribution import StudentDistributionDescriptor as Student
from pysatl_criterion.statistics.goodness_of_fit.continuous.student import (
    LequesneRenyiStudentGofStatistic,
)

known_df = LequesneRenyiStudentGofStatistic(Student.DEFAULT.parse({'df': 5}), k=3)
unknown_df = LequesneRenyiStudentGofStatistic(Student.DEFAULT.parse({}), k=3)
value = known_df.execute_statistic([-3.2, -1.1, -0.3, 0.2, 0.7, 1.8, 4.1])
```

`k` defaults to 3 and must satisfy `1 <= k < n`. Observations must be finite,
distinct and one-dimensional. For `T2`, at least four observations are required.
The finite-variance construction requires fixed or fitted `df > 2`. Fits outside
this domain and nonconverged fits raise `ValueError`; they are not clamped to a
boundary. The convergence result in Theorem 2 has the stronger condition `df > 4`.
Negative finite-sample statistics remain unchanged.

## Rejection direction and calibration

Large entropy deficits reject: the criterion returns `RightAlternative()`.
The lower-tail inequalities printed on page 493 conflict with the positive
alternative limit in Theorem 2 and the entropy-deficit construction. This
implementation uses the upper tail and does not embed the paper's critical-value
table or claim its inequalities are correct as printed.

For `T1`, the built-in sampler uses standard Student observations at the fixed
`df`. Each simulation must repeat the variance and neighbour calculations and
preserve `k`.

For `T2`, there is no universal shape-free calibration. External parametric
bootstrap must simulate at the observed fitted parameters and repeat the entire
fit, including the entropy order, on every replicate. Failed fits must not be
silently dropped. The built-in hypothesis sampler and stored-distribution lookup
reject this unspecified-`df` case with an explicit error. The hypothesis remains
empty; fitted parameters are not substituted into it.

Code: `LEQUESNE_RENYI_STUDENT_GOODNESS_OF_FIT`.
