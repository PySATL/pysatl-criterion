# Two-stage Benjamini–Krieger–Yekutieli (BKY)

## Name and purpose

`BenjaminiKriegerYekutieli` is a adaptive procedure for FDR. It returns one decision for each input hypothesis.

## Original authors and publication

Yoav Benjamini, Abba M. Krieger and Daniel Yekutieli (2006). *Adaptive Linear Step-up Procedures That Control the False Discovery Rate*. Biometrika 93(3), 491–507. [Primary publication](https://doi.org/10.1093/biomet/93.3.491); [authors’ manuscript](https://www.math.tau.ac.il/~ybenja/MyPapers/bkymarch9.pdf).

## Implementation authors / maintainers

Implementation authors: not individually documented. Maintained by the PySATL project; see [overview](index.md).

## Error rate controlled

FDR is bounded by alpha under the stated conditions, including mixtures of true and false null hypotheses.

## Assumptions

Super-uniform null p-values and mutual independence of the p-values. No arbitrary-dependence guarantee is asserted.

## Mathematical algorithm

Let m denote the family size and alpha the requested error level.

1. Sort p-values increasingly as $p_{(1)}\le\cdots\le p_{(m)}$, retaining their original indices.
2. Use the two-stage construction below with $f=1+\alpha$, hence $q_1=\alpha/(1+\alpha)$. This is Definition 6 (TST) and Theorem 1 in the authors’ manuscript.
3. Restore original hypothesis order and compare adjusted values to the supplied alpha. Empty input returns empty arrays.

The shared two-stage construction is:

1. Calculate BH adjusted values b and the first-stage rejection count
   $r_1=\#\{i:b_i\le q_1\}$.
2. If $r_1=0$ or $r_1=m$, stop and report $a_i=\min(1,fb_i)$.
3. Otherwise estimate $\widehat m_0=m-r_1$ and run BH at
   $q_2=q_1m/\widehat m_0$. Report
   $a_i=\min(1,fb_i\widehat m_0/m)$.
4. These reported values support $a_i\le\alpha$ only at this run's alpha;
   they are not alpha-independent adjusted p-values or Storey q-values.


## Input and output

`PValues(values)` owns a read-only copy of a finite real NumPy vector of shape
`(m,)`, with values in `[0, 1]`. `test(data, alpha=0.05)` requires `0 < alpha < 1`.
`MultipleTestingResult` contains boolean `rejected`, float `alpha`, and
`adjusted_p_values` in original order, each array of shape `(m,)`.

This procedure has no `adjust(p)` method. `alpha_dependent=True`; `result.decisions_at(other_alpha)` raises `ValueError`. NumPy arrays remain inspectable, so callers must not manually reuse adaptive values at another level.

## Usage example

```python
import numpy as np
from pysatl_criterion.multiple_testing import BenjaminiKriegerYekutieli, PValues

p = np.array([0.04, 0.001, 0.7, 0.02, 0.02])
procedure = BenjaminiKriegerYekutieli()
result = procedure.test(PValues(p), alpha=0.05)
np.testing.assert_allclose(
    result.adjusted_p_values, [0.021, 0.0021, 0.294, 0.014, 0.014], atol=1e-11
)
assert result.rejected.tolist() == [True, True, False, True, True]
# True means reject that null hypothesis, not proof that its alternative is true.
other = procedure.test(PValues(p), alpha=0.10)
assert other.alpha == 0.10
```

## Worked numerical example

For the unsorted input `[0.04, 0.001, 0.7, 0.02, 0.02]` (including ties) at alpha 0.05,
the adjusted values are `[0.021, 0.0021, 0.294, 0.014, 0.014]` and decisions are `[True, True, False, True, True]`.
These numbers were obtained independently with statsmodels 0.14.6,
`multipletests(..., method="fdr_tsbky", maxiter=1)`, not from the PySATL implementation.
The committed reference fixture retains full precision and tests both values and decisions.

## Limitations and pitfalls

Only two stages are implemented. The distinct multistage rule and iterative variants are not aliases. No general PRDS guarantee follows from the theorem for independent statistics. Ties are handled without changing the input hypothesis order.
Invalid p-values, non-finite entries and multidimensional arrays raise `ValueError`.
No constructor tuning parameters are needed for this fixed procedure.

## References

The primary scientific references are listed under Original authors and publication.
[Independent software reference](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html).
See the [overview](index.md) for method selection and [migration guide](index.md)
for differences from the legacy API.

## Implementation link

[Source](https://github.com/PySATL/pysatl-criterion/blob/main/src/pysatl_criterion/multiple_testing/pvalue/bky.py) ·
[Reference tests](https://github.com/PySATL/pysatl-criterion/blob/main/tests/multiple_testing/test_procedures.py) ·
[Fixtures](https://github.com/PySATL/pysatl-criterion/blob/main/tests/multiple_testing/fixtures/statsmodels.json) ·
[Executable documentation checks](https://github.com/PySATL/pysatl-criterion/blob/main/tests/multiple_testing/test_validation.py)
