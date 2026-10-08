# Holm–Bonferroni

## Name and purpose

`Holm` is a step-down procedure for FWER. It returns one decision for each input hypothesis.

## Original authors and publication

Sture Holm (1979). *A Simple Sequentially Rejective Multiple Test Procedure*. Scandinavian Journal of Statistics 6(2), 65–70. [Primary publication](https://www.jstor.org/stable/4615733).

## Implementation authors / maintainers

Original PySATL implementation: AxelF23, verified in commit `27953e2`. Refactor authors: not individually documented. See [attribution evidence](index.md).

## Error rate controlled

Strong FWER control under the stated conditions, for every configuration of true and false nulls.

## Assumptions

Super-uniform null p-values: $P(p_i \le t) \le t$. Arbitrary dependence is allowed.

## Mathematical algorithm

Let m denote the family size and alpha the requested error level.

1. Sort p-values increasingly as $p_{(1)}\le\cdots\le p_{(m)}$, retaining their original indices.
2. Compute $a_{(i)}=\min(1,\max_{j\le i}(m-j+1)p_{(j)})$. Equivalently, compare $p_{(i)}$ to $\alpha/(m-i+1)$ in increasing order and stop at the first failure.
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
from pysatl_criterion.multiple_testing import Holm, PValues

p = np.array([0.04, 0.001, 0.7, 0.02, 0.02])
procedure = Holm()
result = procedure.test(PValues(p), alpha=0.05)
np.testing.assert_allclose(result.adjusted_p_values, [0.08, 0.005, 0.7, 0.08, 0.08], atol=1e-11)
assert result.rejected.tolist() == [False, True, False, False, False]
# True means reject that null hypothesis, not proof that its alternative is true.
other = procedure.test(PValues(p), alpha=0.10)
assert other.alpha == 0.10
```

## Worked numerical example

For the unsorted input `[0.04, 0.001, 0.7, 0.02, 0.02]` (including ties) at alpha 0.05,
the adjusted values are `[0.08, 0.005, 0.7, 0.08, 0.08]` and decisions are `[False, True, False, False, False]`.
These numbers were obtained independently with statsmodels 0.14.6,
`multipletests(..., method="holm", maxiter=1)`, not from the PySATL implementation.
The committed reference fixture retains full precision and tests both values and decisions.

## Limitations and pitfalls

Requires a prefix maximum, not a suffix minimum. The latter produces Hochberg and was a bug in the previous implementation. Complexity $O(m\log m)$. Ties are handled without changing the input hypothesis order.
Invalid p-values, non-finite entries and multidimensional arrays raise `ValueError`.
No constructor tuning parameters are needed for this fixed procedure.

## References

The primary scientific references are listed under Original authors and publication.
[Independent software reference](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html).
See the [overview](index.md) for method selection and [migration guide](index.md)
for differences from the legacy API.

## Implementation link

[Source](https://github.com/PySATL/pysatl-criterion/blob/main/src/pysatl_criterion/multiple_testing/pvalue/holm.py) ·
[Reference tests](https://github.com/PySATL/pysatl-criterion/blob/main/tests/multiple_testing/test_procedures.py) ·
[Fixtures](https://github.com/PySATL/pysatl-criterion/blob/main/tests/multiple_testing/fixtures/statsmodels.json) ·
[Executable documentation checks](https://github.com/PySATL/pysatl-criterion/blob/main/tests/multiple_testing/test_validation.py)
