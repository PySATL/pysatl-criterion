# Benjamini–Hochberg (BH)

## Name and purpose

`BenjaminiHochberg` is a step-up procedure for FDR. It returns one decision for each input hypothesis.

## Original authors and publication

Yoav Benjamini and Yosef Hochberg (1995). *Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing*. Journal of the Royal Statistical Society, Series B 57(1), 289–300. [Primary publication](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x).

Yoav Benjamini and Daniel Yekutieli (2001). *The Control of the False Discovery Rate in Multiple Testing under Dependency*. Annals of Statistics 29(4), 1165–1188. [Primary publication](https://doi.org/10.1214/aos/1013699998).

## Implementation authors / maintainers

Implementation authors: not individually documented. Maintained by the PySATL project; see [overview](index.md).

## Error rate controlled

FDR is bounded by alpha under the stated conditions, including mixtures of true and false null hypotheses.

## Assumptions

Valid null p-values; independence or PRDS (positive regression dependency on each member of the true null subset). PRDS is a distributional condition, not simply positive pairwise correlation.

## Mathematical algorithm

Let m denote the family size and alpha the requested error level.

1. Sort p-values increasingly as $p_{(1)}\le\cdots\le p_{(m)}$, retaining their original indices.
2. Compute $a_{(i)}=\min(1,\min_{j\ge i}mp_{(j)}/j)$. Find the largest rank k satisfying $p_{(k)}\le k\alpha/m$ and reject ranks 1 through k.
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
from pysatl_criterion.multiple_testing import BenjaminiHochberg, PValues

p = np.array([0.04, 0.001, 0.7, 0.02, 0.02])
procedure = BenjaminiHochberg()
result = procedure.test(PValues(p), alpha=0.05)
np.testing.assert_allclose(
    result.adjusted_p_values, [0.05, 0.005, 0.7, 0.033333333333, 0.033333333333], atol=1e-11
)
assert result.rejected.tolist() == [True, True, False, True, True]
# True means reject that null hypothesis, not proof that its alternative is true.
other = procedure.test(PValues(p), alpha=0.10)
assert other.alpha == 0.10
```

## Worked numerical example

For the unsorted input `[0.04, 0.001, 0.7, 0.02, 0.02]` (including ties) at alpha 0.05,
the adjusted values are `[0.05, 0.005, 0.7, 0.033333333333, 0.033333333333]` and decisions are `[True, True, False, True, True]`.
These numbers were obtained independently with statsmodels 0.14.6,
`multipletests(..., method="fdr_bh", maxiter=1)`, not from the PySATL implementation.
The committed reference fixture retains full precision and tests both values and decisions.

## Limitations and pitfalls

FDR control does not bound the chance of any false discovery when some nulls are false. BH adjusted p-values are not automatically Storey estimates of q-values. Complexity $O(m\log m)$. Ties are handled without changing the input hypothesis order.
Invalid p-values, non-finite entries and multidimensional arrays raise `ValueError`.
No constructor tuning parameters are needed for this fixed procedure.

## References

The primary scientific references are listed under Original authors and publication.
[Independent software reference](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html).
See the [overview](index.md) for method selection and [migration guide](index.md)
for differences from the legacy API.

## Implementation link

[Source](https://github.com/PySATL/pysatl-criterion/blob/main/src/pysatl_criterion/multiple_testing/pvalue/benjamini_hochberg.py) ·
[Reference tests](https://github.com/PySATL/pysatl-criterion/blob/main/tests/multiple_testing/test_procedures.py) ·
[Fixtures](https://github.com/PySATL/pysatl-criterion/blob/main/tests/multiple_testing/fixtures/statsmodels.json) ·
[Executable documentation checks](https://github.com/PySATL/pysatl-criterion/blob/main/tests/multiple_testing/test_validation.py)
