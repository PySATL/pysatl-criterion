# Meintanis ECF criterion for Student distributions

[All Student criteria](index.md) · [Mathematical verification](mathematical-validation.md)

`MeintanisStudentGofStatistic` implements the univariate i.i.d. specialization
of [Meintanis, Milošević, Obradović and Veljović (2024)](https://doi.org/10.1111/jtsa.12713),
equations (2.3)–(2.8), with the moment estimates in (3.3).

## Hypothesis and statistic

Only `df = ν > 2` is fixed. Location and scale are unknown and re-estimated on
**every** call. Write

$$
S_n=\frac1n\sum_j(X_j-\bar X)^2,\quad
\widehat V=\frac{\nu-2}{\nu}S_n,\quad
Y_j=\frac{X_j-\bar X}{\sqrt{\nu\widehat V}}.
$$

For the reduced-scale Student characteristic function

$$
\phi_\nu(t)=\frac{2^{1-\nu/2}}{\Gamma(\nu/2)}
|t|^{\nu/2}K_{\nu/2}(|t|),\qquad \phi_\nu(0)=1,
$$

the returned statistic is

$$
T=\frac{n}{2\pi}\int_{\mathbb R}
\left|\frac1n\sum_j e^{itY_j}-\phi_\nu(t)\right|^2
\phi_{\nu_w}(t)\,dt.
$$

By default `weight_df=None` uses `ν_w=ν`. A positive `weight_df` selects the
alternative Student weight discussed in equation (4.7). The implementation
evaluates the one-dimensional integral directly, including its normalization.
`epsabs=1e-8`, `epsrel=1e-7` and `limit=250` control quadrature; failure to reach
the requested accuracy raises `ValueError`.

## Use and calibration

```python
from pysatl_criterion.distribution import StudentDistributionDescriptor as Student
from pysatl_criterion.statistics.goodness_of_fit.continuous.student import (
    MeintanisStudentGofStatistic,
)

criterion = MeintanisStudentGofStatistic(Student.DEFAULT.parse({'df': 5}))
value = criterion.execute_statistic([-3.2, -1.1, -0.3, 0.2, 0.7, 1.8, 4.1])
```

Large values reject. Samples must be finite, one-dimensional, nonconstant and
contain at least two observations. Ties are allowed. The criterion is invariant
to location, nonzero scale and reflection; it leaves input and hypothesis unchanged.
The built-in hypothesis sampler uses `t(df, loc=0, scale=1)`. Each Monte Carlo
replicate must call `execute_statistic` to re-estimate the moments. Configuration
records the weight and numerical settings, so distinct calibrations remain distinct.

This class uses moment estimation, which requires `df > 2`. It does not fit
unknown degrees of freedom, multivariate scatter matrices or GARCH models.

Code: `MEINTANIS_ECF_STUDENT_GOODNESS_OF_FIT`.
