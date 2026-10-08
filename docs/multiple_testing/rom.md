# Rom step-up procedure

## Name and purpose

`Rom` is a step-up FWER procedure using improved Bonferroni critical values.

## Original authors and publication

Dror M. Rom (1990), *A sequentially rejective test procedure based on a modified
Bonferroni inequality*, Biometrika 77(3), 663–665.
[Primary publication](https://doi.org/10.1093/biomet/77.3.663).

## Error rate controlled

Strong FWER under independence and valid null p-values.

## Assumptions

Mutual independence is required for the guarantee claimed here. Do not use the
power improvement as a justification for arbitrary dependence.

## Mathematical algorithm

For level $q$, define $c_1(q)=q$, $c_2(q)=q/2$ and

$$c_k(q)=\frac{\sum_{j=1}^{k-1}q^j-
\sum_{j=1}^{k-2}{k\choose j}c_{j+1}(q)^{k-j}}{k}.$$

Sort increasingly. Find the largest rank $i$ with
$p_{(i)}\le c_{m-i+1}(\alpha)$ and reject all ranks through $i$.
For adjusted values, numerically invert each critical function and take a suffix
minimum. If a critical function never reaches the raw p-value below level one,
its inverse is set to one. Restore original order.

## Input and output

`test(PValues(p), alpha)` returns `MultipleTestingResult`.
`adjust(p)` returns alpha-independent adjusted values.
`critical_values(m, alpha)` returns rank-ordered **thresholds**, not p-values.
Alpha is in `(0, 1)` and `m` is a nonnegative Python integer.

## Usage example

```python
import numpy as np
from pysatl_criterion.multiple_testing import Rom, PValues

procedure = Rom()
np.testing.assert_allclose(procedure.critical_values(3), [.016875, .025, .05])
result = procedure.test(PValues(np.array([.8, .01, .03])))
expected = [.8, 6*.01/(np.sqrt(1+3*.01)+1), .06]
np.testing.assert_allclose(result.adjusted_p_values, expected)
assert result.rejected.tolist() == [False, True, False]
```

## Worked numerical example

For three hypotheses, $c_3(q)=q/3+q^2/12$. Its inverse at `.01` is
$6(.01)/(\sqrt{1+3(.01)}+1)$, approximately `.0297783`.
The second sorted p-value `.03` requires level `.06`; the last `.8` requires `.8`.
These analytic values independently check the numerical inversion.

## Limitations and pitfalls

The implementation uses 60 bisection steps per inverse and a quadratic recurrence:
worst-case adjustment cost is $O(60m^3)$, intended for modest family sizes.
Threshold generation alone is $O(m^2)$. Empty, singleton, zero, one and tied values
are supported. Finite-precision arithmetic applies near rejection boundaries;
non-finite or negative recurrence results raise `ArithmeticError`.
Multiplying p-values by `alpha / critical` gives alpha-specific quantities; this
implementation instead inverts the whole critical function for reusable adjustments.

## References

The recurrence is also documented in the
[authors' supplementary material for García et al.](https://sci2s.ugr.es/sicidm).
The tests independently evaluate it with 70-digit Decimal arithmetic, and compare
step-up decisions across several levels and family sizes.

## Implementation authors / maintainers

New implementation authors are not individually documented. Scientific attribution
below does not imply authorship of this Python implementation.


## Verification and implementation

[Unit tests](https://github.com/PySATL/pysatl-criterion/blob/main/tests/multiple_testing/test_extended.py)
include independent numerical checks and invalid-input cases. Documentation examples
are executed by the test suite. See the [overview](index.md) for the shared API.

[Source](https://github.com/PySATL/pysatl-criterion/blob/main/src/pysatl_criterion/multiple_testing/pvalue/rom.py).
