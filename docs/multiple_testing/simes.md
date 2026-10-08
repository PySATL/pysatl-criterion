# Simes global test

## Name and purpose

`Simes` tests the intersection of all supplied null hypotheses. Rejection means
there is evidence against their joint truth; it does not identify individual false nulls.

## Original authors and publication

R. John Simes (1986), *An improved Bonferroni procedure for multiple tests of
significance*, Biometrika 73(3), 751–754.
[Primary publication](https://doi.org/10.1093/biomet/73.3.751).

## Error rate controlled

Type I error for the global intersection, at the requested alpha.

## Assumptions

The intersection-null distribution must satisfy the Simes inequality. Independent
uniform null p-values suffice. Arbitrary dependence is not supported by this guarantee.

## Mathematical algorithm

Sort increasingly and calculate
$p_{global}=\min(1,\min_{i=1}^m m p_{(i)}/i)$.
Reject the intersection exactly when $p_{global}\le\alpha$.

## Input and output

`test(PValues(p), alpha)` returns `GlobalTestResult(p_value, alpha)` with scalar
boolean property `rejected`. `combine(p)` returns the scalar global p-value.
There is no individual `adjust()` API. Alpha must be strictly between zero and one.

## Usage example

```python
import numpy as np
from pysatl_criterion.multiple_testing import Simes, PValues

result = Simes().test(PValues(np.array([0.04, 0.001, 0.7, 0.02, 0.02])))
assert result.p_value == 0.005
assert result.rejected
```

## Worked numerical example

For this sorted family `[.001, .02, .02, .04, .7]`, the five candidates are
`.005, .05, .033333..., .05, .7`; their minimum is `.005`.
This calculation checks the example without another multiple-testing implementation.

## Limitations and pitfalls

Complexity is $O(m\log m)$. The empty intersection returns p-value one and is never
rejected. Ties are allowed. Non-finite, out-of-range or non-vector input is rejected.
A significant global result must not be turned into five individual rejections.

## References

See Simes (1986) above. [Hommel](hommel.md) uses Simes local tests within closed testing.

## Implementation authors / maintainers

New implementation authors are not individually documented. Scientific attribution
below does not imply authorship of this Python implementation.


## Verification and implementation

[Unit tests](https://github.com/PySATL/pysatl-criterion/blob/main/tests/multiple_testing/test_extended.py)
include independent numerical checks and invalid-input cases. Documentation examples
are executed by the test suite. See the [overview](index.md) for the shared API.

[Source](https://github.com/PySATL/pysatl-criterion/blob/main/src/pysatl_criterion/multiple_testing/pvalue/simes.py).
