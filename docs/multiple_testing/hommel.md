# Hommel

## Name and purpose

`Hommel` is a closed testing procedure for FWER. It returns one decision for each input hypothesis.

## Original authors and publication

Gerhard Hommel (1988). *A Stagewise Rejective Multiple Test Procedure Based on a Modified Bonferroni Test*. Biometrika 75(2), 383–386. [Primary publication](https://doi.org/10.1093/biomet/75.2.383).

## Implementation authors / maintainers

Implementation authors: not individually documented. Maintained by the PySATL project; see [overview](index.md).

## Error rate controlled

Strong FWER control under the stated conditions, for every configuration of true and false nulls.

## Assumptions

Every intersection of true null hypotheses must admit a valid Simes local test. Independence suffices. Arbitrary dependence does not; a generic statement that correlations are positive is insufficient.

## Mathematical algorithm

Let m denote the family size and alpha the requested error level.

1. Sort p-values increasingly as $p_{(1)}\le\cdots\le p_{(m)}$, retaining their original indices.
2. Initialize $a_{(i)}=p_{(i)}$. For $s=m,m-1,\ldots,2$, compute

   $$c_s=\min_{k=1,\ldots,s}\frac{s\,p_{(m-s+k)}}{k}.$$

   Replace the last s entries by $\max(a_{(i)},c_s)$ and each earlier entry by
   $\max(a_{(i)},\min(sp_{(i)},c_s))$. Clip to one.
   This is the shortcut for closed Simes testing: each individual adjusted value
   is the maximum local Simes p-value over all intersections containing it.
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
from pysatl_criterion.multiple_testing import Hommel, PValues

p = np.array([0.04, 0.001, 0.7, 0.02, 0.02])
procedure = Hommel()
result = procedure.test(PValues(p), alpha=0.05)
np.testing.assert_allclose(result.adjusted_p_values, [0.08, 0.005, 0.7, 0.06, 0.06], atol=1e-11)
assert result.rejected.tolist() == [False, True, False, False, False]
# True means reject that null hypothesis, not proof that its alternative is true.
other = procedure.test(PValues(p), alpha=0.10)
assert other.alpha == 0.10
```

## Worked numerical example

For the unsorted input `[0.04, 0.001, 0.7, 0.02, 0.02]` (including ties) at alpha 0.05,
the adjusted values are `[0.08, 0.005, 0.7, 0.06, 0.06]` and decisions are `[False, True, False, False, False]`.
These numbers were obtained independently with statsmodels 0.14.6,
`multipletests(..., method="hommel", maxiter=1)`, not from the PySATL implementation.
The committed reference fixture retains full precision and tests both values and decisions.

## Limitations and pitfalls

The shortcut costs $O(m^2)$ time and $O(m)$ working storage. Tests compare it against exhaustive closure for small families. Independence is sufficient, but arbitrary dependence is not. Ties are handled without changing the input hypothesis order.
Invalid p-values, non-finite entries and multidimensional arrays raise `ValueError`.
No constructor tuning parameters are needed for this fixed procedure.

## References

The primary scientific references are listed under Original authors and publication.
[Independent software reference](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html).
See the [overview](index.md) for method selection and [migration guide](index.md)
for differences from the legacy API.

## Implementation link

[Source](https://github.com/PySATL/pysatl-criterion/blob/main/src/pysatl_criterion/multiple_testing/pvalue/hommel.py) ·
[Reference tests](https://github.com/PySATL/pysatl-criterion/blob/main/tests/multiple_testing/test_procedures.py) ·
[Fixtures](https://github.com/PySATL/pysatl-criterion/blob/main/tests/multiple_testing/fixtures/statsmodels.json) ·
[Executable documentation checks](https://github.com/PySATL/pysatl-criterion/blob/main/tests/multiple_testing/test_validation.py)
