# Holland–Copenhaver

## Name and purpose

`HollandCopenhaver` is a step-down Šidák-type procedure. Without logical bounds
it gives the same results as [Holm–Šidák](holm_sidak.md).

## Original authors and publication

Burt S. Holland and Margaret DiPonzio Copenhaver (1987), *An improved sequentially
rejective Bonferroni test procedure*, Biometrics 43(2), 417–423.
[Primary publication](https://doi.org/10.2307/2531823).

## Error rate controlled

Strong FWER, provided both the dependence condition and any supplied logical
bounds are justified.

## Assumptions

Valid null p-values and independence suffice. The paper also permits a specific
positive orthant dependence condition on the test statistics. Positive pairwise
correlations alone do not establish this condition.
A custom bound at each rank must dominate the possible number of true nulls after
earlier rejections, for every possible ordering of hypotheses.

## Mathematical algorithm

Sort p-values increasingly. With bounds $t_i$, reject sequentially while
$p_{(i)}\le 1-(1-\alpha)^{1/t_i}$, stopping at the first failure.
The adjusted values are $\min(1,\max_{j\le i}[1-(1-p_{(j)})^{t_j}])$.
Restore original order. The default is $t_i=m-i+1$.

## Input and output

`HollandCopenhaver(true_null_bounds=None)` accepts an optional fixed tuple of
positive integers, one per rank, nonincreasing and no larger than `m-i` at zero-based
rank `i`. `test(PValues(p), alpha)` returns `MultipleTestingResult`; `adjust(p)`
returns alpha-independent adjusted values. The tuple is not inferred from data.

## Usage example

```python
import numpy as np
from pysatl_criterion.multiple_testing import HollandCopenhaver, PValues

# Three pairwise equality nulls: if one is false, at most one can remain true.
# Dependence assumptions still need separate justification for the actual tests.
procedure = HollandCopenhaver(true_null_bounds=(3, 1, 1))
result = procedure.test(PValues(np.array([.8, .01, .04])))
np.testing.assert_allclose(result.adjusted_p_values, [.8, .029701, .04])
assert result.rejected.tolist() == [False, True, True]
```

## Worked numerical example

Sorted values are `.01, .04, .8`. Applying exponents `3, 1, 1` produces
`1-.99**3 = .029701, .04, .8`; the prefix maximum leaves them unchanged.

## Limitations and pitfalls

Structural validation cannot prove that custom bounds are scientifically valid.
The API supports fixed rank bounds, not hypothesis-specific dynamic constraints.
Default bounds coincide with Holm–Šidák and are tested against statsmodels.
Complexity is $O(m\log m)$. Empty input is supported; mismatched bound counts,
invalid probabilities or alpha raise errors. Numerical evaluation uses `log1p`
and `expm1` to preserve precision near zero.

## References

See Holland and Copenhaver (1987), especially Theorem 3.1, and the
[author paper scan](https://sci2s.ugr.es/keel/pdf/algorithm/articulo/1987-Holland-BIO.pdf).

## Implementation authors / maintainers

New implementation authors are not individually documented. Scientific attribution
below does not imply authorship of this Python implementation.


## Verification and implementation

[Unit tests](https://github.com/PySATL/pysatl-criterion/blob/main/tests/multiple_testing/test_extended.py)
include independent numerical checks and invalid-input cases. Documentation examples
are executed by the test suite. See the [overview](index.md) for the shared API.

[Source](https://github.com/PySATL/pysatl-criterion/blob/main/src/pysatl_criterion/multiple_testing/pvalue/holland_copenhaver.py).
