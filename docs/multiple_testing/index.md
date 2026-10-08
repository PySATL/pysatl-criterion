# Multiple Hypothesis Testing

Testing many hypotheses at an unadjusted 5% level can produce false discoveries
even when every null is true. Choose a family of hypotheses before examining
its results, then apply one procedure to all raw p-values in that family.

## Definitions

- **m**: number of hypotheses; **R**: number rejected; **V**: rejected true nulls.
- **FWER**: $P(V>0)$, the chance of at least one false rejection. Strong control
  applies to every true/false configuration; weak control applies only to the global null.
- **FDR**: $E[V/\max(R,1)]$, the expected false discovery proportion. Under the global
  null FDR equals FWER, but generally an FDR guarantee is less restrictive.
- **Raw p-value**: a single-test measure satisfying $P(p_i\le t)\le t$ under its null.
- **Adjusted p-value**: the smallest error level at which a fixed adjustment rejects
  a hypothesis. Ordinary adjusted values can be compared to different alpha levels.
  The reported **adaptive values** here are an exception: they apply only at the
  supplied alpha and must be recomputed when it changes.
- **q-value**: in Storey's framework, the minimum positive FDR associated with a
  rejection region containing the observation (estimated in applications using pi0).
  It is not a universal synonym for every FDR-adjusted value. Storey estimation is deferred.
  See [Storey (2002), A direct approach to false discovery rates](https://doi.org/10.1111/1467-9868.00346).

## Architecture

```text
Statistic -> PValueCalculator / p-value resolver -> PValues
          -> AbstractMultipleTesting[PValues].test -> MultipleTestingResult
             |- AbstractPValueAdjustment.adjust (alpha-independent)
             |- AbstractAlphaDependentPValueProcedure (alpha required)

Joint resampling preserving dependence
  -> ResamplingStatistics (observed statistics, joint statistics; larger is extreme)
     -> AbstractResamplingMultipleTesting -> MultipleTestingResult
  -> ResamplingPValues (observed raw p-values, jointly resampled raw p-values)
     -> AbstractResamplingPValueTesting -> MultipleTestingResult
```

The last two paths implement Westfall–Young maxT and minP. Simes uses a separate
`GlobalTestResult`, because it tests one intersection rather than individual nulls. Tests of generic annotations ensure mypy rejects the wrong input type.
No GOF-statistic registry or distribution-calibration responsibility is mixed into
multiple testing. The dedicated immutable `P_VALUE_PROCEDURES` mapping exposes
classes by `short_code()`; `code()` prefixes these identifiers with `MULTIPLE_TESTING_`.

## Package layout

The common contract, inputs, result and metadata live in `base.py`, `inputs.py`,
`result.py` and `metadata.py`. Each p-value procedure is implemented in its own
module under `pvalue/`, with family contracts in `pvalue/base.py`. Joint-resampling
contracts and explicit design assumptions live in `resampling/base.py`. Separate
Westfall–Young modules implement maxT and minP with exact and Monte Carlo modes.
The root package and `api.testing` export the same typed classes. Compatibility
modules have been removed.

## Implemented methods

All procedures require valid raw null p-values; none repairs invalid calibration.

| Method | Error criterion | Input | Alpha-dependent | Dependence / guarantee |
|---|---|---|---|---|
| [Bonferroni](bonferroni.md) | FWER | PValues | No | Super-uniform null p-values: $P(p_i \le t) \le t$. Arbitrary dependence is allowed. |
| [Šidák](sidak.md) | FWER | PValues | No | Super-uniform null p-values and mutual independence of the p-values. No arbitrary-dependence guarantee is asserted. |
| [Holm–Bonferroni](holm.md) | FWER | PValues | No | Super-uniform null p-values: $P(p_i \le t) \le t$. Arbitrary dependence is allowed. |
| [Holm–Šidák](holm_sidak.md) | FWER | PValues | No | Super-uniform null p-values and mutual independence of the p-values. No arbitrary-dependence guarantee is asserted. |
| [Hochberg (Simes–Hochberg)](hochberg.md) | FWER | PValues | No | Every intersection of true null hypotheses must admit a valid Simes local test. Independence suffices. Arbitrary dependence does not; a generic statement that correlations are positive is insufficient. |
| [Hommel](hommel.md) | FWER | PValues | No | Every intersection of true null hypotheses must admit a valid Simes local test. Independence suffices. Arbitrary dependence does not; a generic statement that correlations are positive is insufficient. |
| [Benjamini–Hochberg (BH)](benjamini_hochberg.md) | FDR | PValues | No | Valid null p-values; independence or PRDS (positive regression dependency on each member of the true null subset). PRDS is a distributional condition, not simply positive pairwise correlation. |
| [Benjamini–Yekutieli (BY)](benjamini_yekutieli.md) | FDR | PValues | No | Super-uniform null p-values: $P(p_i \le t) \le t$. Arbitrary dependence is allowed. |
| [Two-stage Benjamini–Hochberg](two_stage_bh.md) | FDR target | PValues | Yes | Exploratory uncorrected variant: nominal FDR control is not guaranteed even under independence. Use BKY when relying on the proved independent-case guarantee. |
| [Two-stage Benjamini–Krieger–Yekutieli (BKY)](bky.md) | FDR | PValues | Yes | Super-uniform null p-values and mutual independence of the p-values. No arbitrary-dependence guarantee is asserted. |
| [Simes](simes.md) | Global Type I | PValues | No | Valid Simes inequality; independent uniform nulls suffice. One global result. |
| [Rom](rom.md) | FWER | PValues | No | Independence; step-up critical values, inverted for adjusted p-values. |
| [Holland–Copenhaver](holland_copenhaver.md) | FWER | PValues | No | Independence or the paper's positive orthant dependence condition; optional justified logical bounds. |
| [Westfall–Young maxT](westfall_young_max_t.md) | FWER | ResamplingStatistics | No | Valid joint exchangeable resampling, subset pivotality, comparable statistics. |
| [Westfall–Young minP](westfall_young_min_p.md) | FWER | ResamplingPValues | No | Valid joint exchangeable resampling, subset pivotality, calibrated raw p-values. |

Single-step methods apply a common correction. Step-down methods examine small
p-values first and stop at a failure; step-up methods locate the largest passing
rank. Closed testing requires rejection of every intersection containing a null.
Adaptive methods estimate the number of true nulls from the same family.
Resampling methods use a joint null distribution rather than just raw p-values.

## Quick start

```python
import numpy as np
from pysatl_criterion.multiple_testing import BenjaminiHochberg, Holm, PValues

p = PValues(np.array([0.04, 0.001, 0.7, 0.02, 0.02]))
bh = BenjaminiHochberg().test(p, alpha=0.05)
holm = Holm().test(p, alpha=0.05)
assert bh.rejected.tolist() == [True, True, False, True, True]
assert holm.rejected.tolist() == [False, True, False, False, False]
np.testing.assert_allclose(holm.adjusted_p_values, [0.08, 0.005, 0.7, 0.08, 0.08])
```

```python
import numpy as np
from pysatl_criterion.multiple_testing import PValues, BenjaminiKriegerYekutieli

procedure = BenjaminiKriegerYekutieli()
result = procedure.test(PValues(np.array([0.001, 0.02, 0.2])), alpha=0.10)
assert result.alpha_dependent
# To use another alpha, run test again. decisions_at(0.05) would raise ValueError.
recomputed = procedure.test(PValues(np.array([0.001, 0.02, 0.2])), alpha=0.05)
assert recomputed.alpha == 0.05
```

## Choosing a method

For strong FWER control without dependence restrictions, use Holm or Bonferroni.
For FDR under arbitrary dependence, BY is available, often conservatively.
BH is appropriate when independence or PRDS is justified. Independence permits
Šidák procedures and the two-stage BKY guarantee. Hochberg and Hommel require
valid Simes local tests. The uncorrected two-stage BH option is for reproducibility,
without a claimed nominal FDR guarantee. Positive correlation alone is not PRDS.

## Limitations and common mistakes

Typed APIs require `0 < alpha < 1`. Inputs are finite real vectors in `[0, 1]`;
empty families produce empty results. Stored arrays are defensive read-only copies.
Do not change the family after inspecting significance, pass rejection thresholds
as p-values, or interpret non-rejection as acceptance of the null. Adjustment
cannot fix invalid single-test p-values, optional stopping or unmodeled selection.
A global Simes test is distinct from a procedure making individual decisions.

Adaptive output arrays are inspectable but alpha-specific: the result's
`decisions_at()` guard rejects reuse at a different level. A frozen dataclass
and read-only NumPy flags protect against accidental mutation, not deliberate
circumvention of array flags. Hommel is quadratic in the number of hypotheses.

Legacy classmethod calls are no longer supported. See the migration guide below
for the replacement API and boundary behavior.

## Deferred extensions

Storey/q-values are not implemented. Westfall–Young requires explicit
`JointResamplingAssumptions`; these caller declarations do not establish the
statistical validity of the supplied arrays. Its exact mode requires a complete
equally weighted orbit, and its Monte Carlo mode uses the identity correction.
Iterated and multistage BKY are also deferred; the implemented BKY is exactly two-stage.

## Verification

See the [implementation report](implementation_report.md) for the file manifest,
method inventory, measured simulation rates and actual validation outcomes.

Every method has its own page above, primary references, attribution evidence,
independent worked example and executable Python example. Unit tests execute all
Python fences on these pages and check local links. Fixed fixtures are generated
with statsmodels 0.14.6 (a development dependency only), with seeded live comparisons.
Hommel also has an exhaustive closed-Simes oracle, and BKY a direct two-stage oracle.
Rom has a high-precision polynomial oracle and analytic inversion checks;
Holland–Copenhaver has logical-bound examples and independent software comparisons.
Westfall–Young is checked against exhaustive closed testing and exact sign-flip
FWER calculations. Simes has scalar hand-calculated examples.

`P_VALUE_PROCEDURES` contains twelve per-hypothesis p-value procedures. Simes is
in `GLOBAL_TESTS`; the two Westfall–Young classes have separate typed resampling
registries. All concrete classes are exported from the package root and `api.testing`.

```sh
python -m pytest tests/multiple_testing
PYSATL_MULTIPLE_TESTING_MONTE_CARLO=1 python -m pytest tests/multiple_testing/test_validation.py -k monte_carlo -s
python -m mkdocs build --strict
```

Monte Carlo validation uses seed 62173, 5,000 independent families of 20 hypotheses,
alpha 0.05, both the global null and five false nulls fixed at zero. It measures
FWER or FDR as appropriate, using a six-standard-error upper tolerance. Simulations
are skipped unless explicitly enabled and do not replace mathematical guarantees.

## Migration from the removed API

The `abstract_multiple_testing.py`, `fdr.py` and `fwer.py` adapters have been removed.
This intentionally breaks old classmethod/list calls. All three public entry points
(`pysatl_criterion`, `pysatl_criterion.api.testing`, and
`pysatl_criterion.multiple_testing`) now expose the same typed procedure classes.

| Old name | New name |
|---|---|
| BonferroniMultipleTesting | Bonferroni |
| SidakMultipleTesting | Sidak |
| SidakHolm | HolmSidak |

Instantiate the procedure, wrap a NumPy vector in `PValues`, replace `threshold`
with `alpha`, and read result fields instead of unpacking lists:

```python
import numpy as np
from pysatl_criterion import Holm, PValues

result = Holm().test(PValues(np.array([0.01, 0.03, 0.04])), alpha=0.05)
assert result.rejected.tolist() == [True, False, False]
np.testing.assert_allclose(result.adjusted_p_values, [0.03, 0.06, 0.06])
```

The new API compares with `<= alpha` and requires `0 < alpha < 1`. The former API
used strict `< threshold` and allowed endpoint thresholds. Empty families still
produce empty outputs. External subclasses should use the generic contract in
`base.py` or family contracts in `pvalue/base.py` and `resampling/base.py`.
The corrected Holm and Holm–Šidák algorithms intentionally change old numerical
results. For reproducing the former incorrect recurrences, explicitly select
Hochberg or single-step Sidak, respectively.

## Attribution evidence

Original PySATL Bonferroni and Sidak implementations were introduced by Ksenia
Nigmatulina in commit `0eb6cfa2fd53b7489d3daa0af998b936b552efcd`. Holm, SidakHolm and
BY were introduced by AxelF23 in commit `27953e2f76eca87bcaaa0d9f1d61e1c199ab21ad`.
These historical implementation credits are separate from scientific authors and
do not assign authorship of the refactor. New implementation authors are not
individually documented.
