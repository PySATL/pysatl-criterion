# Obradović–Jovanović–Milošević integral statistic

Public class: `ObradovicParetoGofStatistic`. Full code: `OJM_Integral_PARETO_GOODNESS_OF_FIT`.

## Hypothesis and statistic

Scale $s>0$ is known and shape $a>0$ is unknown. Put $z_i=x_i/s$ and define

$$
F_n(t)=\frac1n\sum_i\mathbf1\{z_i\le t\},\qquad
M_n(t)=\binom n2^{-1}\sum_{j<k}\mathbf1\{\max(z_j/z_k,z_k/z_j)\le t\}.
$$

The returned statistic is

$$
T_n=\frac1n\sum_i[M_n(z_i)-F_n(z_i)].
$$

This is the signed empirical integral, without absolute value or $\sqrt n$
scaling. It is distinct from the paper's supremum statistic $V_n$.
The upper critical tail follows the paper's convention; it is not asserted to
be consistent against every alternative.

## Input and calibration

Require finite real unmasked one-dimensional data, $n\ge2$ and $x_i\ge s$.
Constants and ties are accepted. Indicators use inclusive comparisons.
A cached Numba kernel counts pairs using log ratios to avoid overflow.
Numerically ambiguous boundaries trigger recomputation with exact rational
comparisons in Python. Cost is $O(n^2\log n)$ and working memory $O(n)$.

The paper uses $s=1$; division by the known scale gives that case.
Raising every $z_i$ to the same positive power preserves each comparison and
eliminates the nuisance shape from the continuous IID null law. Estimate neither
parameter in this procedure. A scale estimated from the same sample changes the
null law and is unsupported. Calibration is external; built-in Pareto simulation
is unavailable. Finite-sample calibration must retain the same normalization and
inclusive tie convention, and model rounding if present.

## Example

```python
from pysatl_criterion.distribution import ParetoDistributionDescriptor as Pareto
from pysatl_criterion.statistics.goodness_of_fit.continuous.pareto import ObradovicParetoGofStatistic

statistic = ObradovicParetoGofStatistic(Pareto.DEFAULT.parse({"scale": 2}))
value = statistic.execute_statistic([2.2, 2.8, 3.8, 5.4, 8.4, 16])
print(value)
```

## Scientific source

[Obradović, Jovanović and Milošević (2014), Goodness of Fit Tests for Pareto
Distribution Based on a Characterization and their Asymptotics, §2](https://arxiv.org/html/1310.5510)
defines $M_n$, $T_n$ and the upper-tail convention. The fixed-scale extension
and numerical tie handling are implementation details.

[All Pareto criteria](index.md) · [Mathematical verification](mathematical-validation.md).
