# Benjamini–Yekutieli (BY)

## Name and purpose

`BenjaminiYekutieli` is a step-up procedure for FDR. It returns one decision for each input hypothesis.

## Original authors and publication

Yoav Benjamini and Daniel Yekutieli (2001). *The Control of the False Discovery Rate in Multiple Testing under Dependency*. Annals of Statistics 29(4), 1165–1188. [Primary publication](https://doi.org/10.1214/aos/1013699998).

## Implementation authors / maintainers

Original PySATL implementation: AxelF23, verified in commit `27953e2`. Refactor authors: not individually documented. See [attribution evidence](index.md).

## Error rate controlled

FDR is bounded by alpha under the stated conditions, including mixtures of true and false null hypotheses.

## Assumptions

Super-uniform null p-values: $P(p_i \le t) \le t$. Arbitrary dependence is allowed.

## Mathematical algorithm

Let m denote the family size and alpha the requested error level.

1. Sort p-values increasingly as $p_{(1)}\le\cdots\le p_{(m)}$, retaining their original indices.
2. Compute $H_m=\sum_{j=1}^m1/j$, then $a_{(i)}=\min(1,\min_{j\ge i}mH_mp_{(j)}/j)$. The step-up critical values are $i\alpha/(mH_m)$.
3. Restore original hypothesis order and compare adjusted values to the supplied alpha. Empty input returns empty arrays.

## Input and output

`PValues(values)` owns a read-only copy of a finite real NumPy vector of shape
`(m,)`, with values in `[0, 1]`. `test(data, alpha=0.05)` requires `0 < alpha < 1`.
`MultipleTestingResult` contains boolean `rejected`, float `alpha`, and
`adjusted_p_values` in original order, each array of shape `(m,)`.

`adjust(p)` accepts a float NumPy vector and returns adjusted p-values independent of alpha. `result.decisions_at(other_alpha)` can reuse them.

## Usage example

```python
import numpy as np
from pysatl_criterion.multiple_testing import BenjaminiYekutieli, PValues

p = np.array([0.04, 0.001, 0.7, 0.02, 0.02])
procedure = BenjaminiYekutieli()
result = procedure.test(PValues(p), alpha=0.05)
np.testing.assert_allclose(
    result.adjusted_p_values,
    [0.114166666667, 0.011416666667, 1.0, 0.076111111111, 0.076111111111],
    atol=1e-11,
)
assert result.rejected.tolist() == [False, True, False, False, False]
# True means reject that null hypothesis, not proof that its alternative is true.
other = procedure.test(PValues(p), alpha=0.10)
assert other.alpha == 0.10
```

## Worked numerical example

For the unsorted input `[0.04, 0.001, 0.7, 0.02, 0.02]` (including ties) at alpha 0.05,
the adjusted values are `[0.114166666667, 0.011416666667, 1.0, 0.076111111111, 0.076111111111]` and decisions are `[False, True, False, False, False]`.
These numbers were obtained independently with statsmodels 0.14.6,
`multipletests(..., method="fdr_by", maxiter=1)`, not from the PySATL implementation.
The committed reference fixture retains full precision and tests both values and decisions.

## Limitations and pitfalls

The harmonic factor can be conservative. The implementation sums the full harmonic series rather than replacing it with a downward approximation at large m. Complexity $O(m\log m)$. Ties are handled without changing the input hypothesis order.
Invalid p-values, non-finite entries and multidimensional arrays raise `ValueError`.
No constructor tuning parameters are needed for this fixed procedure.

## References

The primary scientific references are listed under Original authors and publication.
[Independent software reference](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html).
See the [overview](index.md) for method selection and [migration guide](index.md)
for differences from the legacy API.

## Implementation link

[Source](https://github.com/PySATL/pysatl-criterion/blob/main/src/pysatl_criterion/multiple_testing/pvalue/benjamini_yekutieli.py) ·
[Reference tests](https://github.com/PySATL/pysatl-criterion/blob/main/tests/multiple_testing/test_procedures.py) ·
[Fixtures](https://github.com/PySATL/pysatl-criterion/blob/main/tests/multiple_testing/fixtures/statsmodels.json) ·
[Executable documentation checks](https://github.com/PySATL/pysatl-criterion/blob/main/tests/multiple_testing/test_validation.py)
