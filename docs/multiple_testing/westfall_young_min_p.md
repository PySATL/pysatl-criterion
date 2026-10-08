# Westfall–Young minP

## Name and purpose

`WestfallYoungMinP` adjusts a family using its joint resampling distribution,
with single-step and step-down modes.

## Original authors and publication

Peter H. Westfall and S. Stanley Young (1993), *Resampling-Based Multiple Testing:
Examples and Methods for p-Value Adjustment*, Wiley.
The computational reference is Ge, Dudoit and Speed (2003), *Resampling-based
multiple testing for microarray data analysis*, [technical report](https://statistics.berkeley.edu/sites/default/files/tech-reports/633.pdf),
Box 3.

## Error rate controlled

Strong FWER under a valid exchangeable joint randomization scheme and subset
pivotality. The numeric arrays alone cannot establish either condition.

## Assumptions

Each row must come from one shared null resampling operation across all columns.
Subset pivotality requires that true-null subvectors have the same distribution
under partial and complete nulls. Construct `JointResamplingAssumptions` explicitly
only after justifying these properties for the experimental design.

All entries must be calibrated raw p-values in `[0, 1]`, computed using a
consistent null calibration for observed and resampled data. Marginal p-values
must reflect the intended tail already. The class neither computes marginal
p-values nor accepts a vector of statistics in their place.

## Mathematical algorithm

Sort observed p-values increasingly. For each rank, take the minimum raw
p-value over all remaining columns in each resampled row; count how often it
is at most the observed p-value.
In step-down mode, take the prefix maximum of these tail probabilities and
restore original order. Single-step mode uses the entire family for every rank.

With `sampling="monte_carlo"`, use `(count + 1)/(B + 1)`: the B rows are uniform
random draws with replacement from the valid joint null randomization orbit.
The observed configuration supplies the extra identity contribution. Random draws
may happen to equal it; do not append an extra identity row manually in this mode.
With `sampling="exact"`, enumerate the complete equally weighted orbit, including
the identity, and use `count/B`. The implementation checks that an observed row
is present; it cannot verify completeness or uniform weighting.

## Input and output

`ResamplingPValues(observed, resampled)` holds `(m,)` and `(B, m)` read-only
copies; `B > 0`. The constructor takes `assumptions`, `step_down=True`, and
`sampling="monte_carlo"`. `test(data, alpha)` returns `MultipleTestingResult` with
alpha-independent adjusted p-values and decisions in original column order.
Alpha must lie strictly between zero and one.

## Usage example

```python
import numpy as np
from pysatl_criterion.multiple_testing import (
    WestfallYoungMinP, ResamplingPValues, JointResamplingAssumptions,
)

# Numerical illustration only: a real analysis must justify its resampling design.
observed = np.array([.3, .2, .4])
draws = np.array([[.3, .5, .6], [.6, .1, .4], [.5, .4, .3], [.4, .3, .5]])
assumptions = JointResamplingAssumptions(exchangeable=True, subset_pivotal=True)
procedure = WestfallYoungMinP(assumptions, step_down=True, sampling="monte_carlo")
result = procedure.test(ResamplingPValues(observed, draws), alpha=.5)
np.testing.assert_allclose(result.adjusted_p_values, [.6, .4, .6])
assert result.rejected.tolist() == [False, True, False]
```

## Worked numerical example

The ordered family has extreme-count totals `1, 2, 2` among four joint draws.
Adding the identity yields `2/5, 3/5, 3/5`. The prefix maximum is unchanged;
restoring input order gives `.6, .4, .6`. Single-step adjustment gives `1, .4, 1`.

## Limitations and pitfalls

Independent shuffling of columns destroys dependence information. The assertion
object records caller declarations; it does not certify the design. The smallest
Monte Carlo value is `1/(B+1)`, so too few draws limit resolution. Ties use inclusive
comparisons. Empty families return empty results. Complexity is
$O(m\log m + Bm)$ with $O(Bm)$ intermediate memory.

## References

For random sampling and the identity correction, see Phipson and Smyth (2010),
*Permutation P-values Should Never Be Zero*,
[primary publication](https://doi.org/10.2202/1544-6115.1585).
Tests compare step-down results to exhaustive closed intersection tests and check
FWER over a complete sign-flip orbit under global and partial null configurations.

## Implementation authors / maintainers

New implementation authors are not individually documented. Scientific attribution
below does not imply authorship of this Python implementation.


## Verification and implementation

[Unit tests](https://github.com/PySATL/pysatl-criterion/blob/main/tests/multiple_testing/test_extended.py)
include independent numerical checks and invalid-input cases. Documentation examples
are executed by the test suite. See the [overview](index.md) for the shared API.

[Source](https://github.com/PySATL/pysatl-criterion/blob/main/src/pysatl_criterion/multiple_testing/resampling/westfall_young_min_p.py).
