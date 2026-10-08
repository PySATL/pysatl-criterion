# Bonferroni

## Name and purpose

`Bonferroni` is a single-step procedure for FWER. It returns one decision for each input hypothesis.

## Original authors and publication

Olive Jean Dunn (1961). *Multiple Comparisons among Means*. Journal of the American Statistical Association 56(293), 52–64. [Primary publication](https://doi.org/10.1080/01621459.1961.10482090). The inequality is traditionally attributed to Carlo Emilio Bonferroni; Dunn is the primary multiple-comparisons reference used here.

Historical origin: Carlo Emilio Bonferroni (1936), *Teoria statistica delle classi e calcolo delle probabilità*, Pubblicazioni del R. Istituto Superiore di Scienze Economiche e Commerciali di Firenze, 8, 3–62. No DOI for that work is asserted here. The implemented multiple-testing rule is justified directly by the union bound and the Dunn reference above.

## Implementation authors / maintainers

Original PySATL implementation: Ksenia Nigmatulina, verified in commit `0eb6cfa`. Refactor authors: not individually documented. See [attribution evidence](index.md).

## Error rate controlled

Strong FWER control under the stated conditions, for every configuration of true and false nulls.

## Assumptions

Super-uniform null p-values: $P(p_i \le t) \le t$. Arbitrary dependence is allowed.

## Mathematical algorithm

Let m denote the family size and alpha the requested error level.

1. Sort p-values increasingly as $p_{(1)}\le\cdots\le p_{(m)}$, retaining their original indices.
2. Compute $a_i=\min(1,mp_i)$ for each hypothesis. Reject when $a_i\le\alpha$, equivalently $p_i\le\alpha/m$.
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
from pysatl_criterion.multiple_testing import Bonferroni, PValues

p = np.array([0.04, 0.001, 0.7, 0.02, 0.02])
procedure = Bonferroni()
result = procedure.test(PValues(p), alpha=0.05)
np.testing.assert_allclose(result.adjusted_p_values, [0.2, 0.005, 1.0, 0.1, 0.1], atol=1e-11)
assert result.rejected.tolist() == [False, True, False, False, False]
# True means reject that null hypothesis, not proof that its alternative is true.
other = procedure.test(PValues(p), alpha=0.10)
assert other.alpha == 0.10
```

## Worked numerical example

For the unsorted input `[0.04, 0.001, 0.7, 0.02, 0.02]` (including ties) at alpha 0.05,
the adjusted values are `[0.2, 0.005, 1.0, 0.1, 0.1]` and decisions are `[False, True, False, False, False]`.
These numbers were obtained independently with statsmodels 0.14.6,
`multipletests(..., method="bonferroni", maxiter=1)`, not from the PySATL implementation.
The committed reference fixture retains full precision and tests both values and decisions.

## Limitations and pitfalls

Conservative for large families. Runs in linear time mathematically; the shared implementation sorts, taking $O(m\log m)$. Ties are handled without changing the input hypothesis order.
Invalid p-values, non-finite entries and multidimensional arrays raise `ValueError`.
No constructor tuning parameters are needed for this fixed procedure.

## References

The primary scientific references are listed under Original authors and publication.
[Independent software reference](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html).
See the [overview](index.md) for method selection and [migration guide](index.md)
for differences from the legacy API.

## Implementation link

[Source](https://github.com/PySATL/pysatl-criterion/blob/main/src/pysatl_criterion/multiple_testing/pvalue/bonferroni.py) ·
[Reference tests](https://github.com/PySATL/pysatl-criterion/blob/main/tests/multiple_testing/test_procedures.py) ·
[Fixtures](https://github.com/PySATL/pysatl-criterion/blob/main/tests/multiple_testing/fixtures/statsmodels.json) ·
[Executable documentation checks](https://github.com/PySATL/pysatl-criterion/blob/main/tests/multiple_testing/test_validation.py)
