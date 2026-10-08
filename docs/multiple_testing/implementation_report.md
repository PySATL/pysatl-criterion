# Multiple testing implementation report

## Architectural decisions

The new generic contract is `pysatl_criterion.multiple_testing.AbstractMultipleTesting[InputT]`.
It uses mathematical input objects and a common `MultipleTestingResult`, without
optional context dictionaries or generic keyword payloads. Ten ordinary
procedures share `AbstractPValueAdjustment.adjust`; two adaptive procedures have
only an alpha-explicit `test` and mark their results alpha-dependent.
`decisions_at` prevents accidental reuse at another alpha. Holland–Copenhaver optionally accepts logical true-null bounds. Simes returns a
separate scalar `GlobalTestResult`. Resampling procedures require explicit design
assumptions and expose single-step/step-down and exact/Monte Carlo settings.

The legacy abstract class and five adapters have been removed at the user's
request. Root and `api.testing` exports now refer to the typed classes. This
intentionally breaks the classmethod/list API; see [migration](index.md).
The typed API uses `<= alpha`, requires `0 < alpha < 1`, and returns named fields.

Joint statistic and raw-p-value matrices are separate validated containers because
maxT and minP need different mathematical information. Westfall–Young maxT and minP implement these contracts. Strong FWER is conditional
on valid joint exchangeability and subset pivotality, declared by the caller.
The implementation cannot infer these properties from numeric arrays.

## Method inventory

There are twelve per-hypothesis p-value procedures in `P_VALUE_PROCEDURES`,
one global test in `GLOBAL_TESTS`, and two procedures in separate typed resampling
registries. The original ten procedures have independent statsmodels fixtures and
live comparisons. Rom has a high-precision recurrence oracle and analytic inversion
checks. Holland–Copenhaver has independent Holm–Šidák comparisons and logical-bound
examples. Simes has hand-calculated scalar examples. Westfall–Young is checked
against exhaustive closed testing, hand-counted matrices, and a complete sign-flip
orbit under global and partial null configurations.

All procedures have executable documentation examples and public-export checks.
Tests live in `test_procedures.py`, `test_extended.py`, and `test_validation.py`
under `tests/multiple_testing`. Hommel additionally uses exhaustive Simes closure;
BKY uses an independent critical-value decision oracle.

| Method | Status | Criterion | Input | Documentation | Tests | Primary reference |
|---|---|---|---|---|---|---|
| Bonferroni | Refactored | FWER | `PValues` | [bonferroni.md](bonferroni.md) | Fixtures + live + properties + docs | [Dunn (1961)](https://doi.org/10.1080/01621459.1961.10482090) |
| Šidák | Refactored | FWER | `PValues` | [sidak.md](sidak.md) | Fixtures + live + properties + docs | [Šidák (1967)](https://doi.org/10.1080/01621459.1967.10482935) |
| Holm | Corrected | FWER | `PValues` | [holm.md](holm.md) | Fixtures + live + properties + docs | [Holm (1979)](https://www.jstor.org/stable/4615733) |
| Holm–Šidák | Corrected | FWER | `PValues` | [holm_sidak.md](holm_sidak.md) | Fixtures + live + properties + docs | [Holm (1979) + Šidák (1967)](https://www.jstor.org/stable/4615733) |
| Hochberg | Added | FWER | `PValues` | [hochberg.md](hochberg.md) | Fixtures + live + properties + docs | [Hochberg (1988)](https://doi.org/10.1093/biomet/75.4.800) |
| Hommel | Added | FWER | `PValues` | [hommel.md](hommel.md) | Fixtures + live + properties + docs | [Hommel (1988)](https://doi.org/10.1093/biomet/75.2.383) |
| BH | Added | FDR | `PValues` | [benjamini_hochberg.md](benjamini_hochberg.md) | Fixtures + live + properties + docs | [Benjamini & Hochberg (1995)](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x) |
| BY | Refactored | FDR | `PValues` | [benjamini_yekutieli.md](benjamini_yekutieli.md) | Fixtures + live + properties + docs | [Benjamini & Yekutieli (2001)](https://doi.org/10.1214/aos/1013699998) |
| Two-stage BH | Added (exploratory) | FDR target only | `PValues` | [two_stage_bh.md](two_stage_bh.md) | Fixtures + live + properties + docs | [BKY (2006); uncorrected software variant](https://doi.org/10.1093/biomet/93.3.491) |
| Two-stage BKY | Added | FDR | `PValues` | [bky.md](bky.md) | Fixtures + live + properties + docs | [BKY (2006), TST](https://doi.org/10.1093/biomet/93.3.491) |
| Storey/q-value | Deferred | FDR/pFDR | Estimator contract undecided | [Overview](index.md) | Not implemented | [Storey (2002)](https://doi.org/10.1111/1467-9868.00346) |
| Simes | Added | Global Type I | `PValues` | [simes.md](simes.md) | Hand calculations + validation + docs | [Simes (1986)](https://doi.org/10.1093/biomet/73.3.751) |
| Rom | Added | FWER | `PValues` | [rom.md](rom.md) | Decimal oracle + analytic inversion + decisions + properties | [Rom (1990)](https://doi.org/10.1093/biomet/77.3.663) |
| Holland–Copenhaver | Added | FWER | `PValues` | [holland_copenhaver.md](holland_copenhaver.md) | Independent software + logical bounds + properties | [Holland & Copenhaver (1987)](https://doi.org/10.2307/2531823) |
| Westfall–Young maxT | Added | FWER, conditional on design | `ResamplingStatistics` | [westfall_young_max_t.md](westfall_young_max_t.md) | Counts + closure + sign-flip FWER + tails | Westfall & Young (1993); Ge, Dudoit & Speed (2003) |
| Westfall–Young minP | Added | FWER, conditional on design | `ResamplingPValues` | [westfall_young_min_p.md](westfall_young_min_p.md) | Counts + closure + sign-flip FWER | Westfall & Young (1993); Ge, Dudoit & Speed (2003) |

## Public API

```python
import numpy as np
from pysatl_criterion.multiple_testing import Holm, PValues

result = Holm().test(PValues(np.array([0.01, 0.04, 0.03])), alpha=0.05)
np.testing.assert_allclose(result.adjusted_p_values, [0.03, 0.06, 0.06])
assert result.rejected.tolist() == [True, False, False]
```

See [the overview](index.md) for BH, adaptive usage and discovery, and
[the overview](index.md) for legacy migration and verified attribution.

## Exact file manifest

Paths below are relative to the repository root. Existing unrelated untracked
files, including the supplied task and `poetry.lock`, were preserved.

Modified:

- `mkdocs.yml`
- `pyproject.toml`
- `src/pysatl_criterion/__init__.py`
- `src/pysatl_criterion/api/testing.py`
- `tests/test_public_api_coverage.py`
- `tests/test_protocol_coverage.py`

Removed after consolidation:

- `src/pysatl_criterion/hypothesis_testing/multiple_testing/__init__.py`
- `src/pysatl_criterion/hypothesis_testing/multiple_testing/abstract_multiple_testing.py`
- `src/pysatl_criterion/hypothesis_testing/multiple_testing/fdr.py`
- `src/pysatl_criterion/hypothesis_testing/multiple_testing/fwer.py`
- `tests/hypothesis_testing/multiple_testing/test_fwer.py`
- `tests/hypothesis_testing/multiple_testing/test_fdr.py`

Created (including moved files):

- `docs/multiple_testing/benjamini_hochberg.md`
- `docs/multiple_testing/benjamini_yekutieli.md`
- `docs/multiple_testing/bky.md`
- `docs/multiple_testing/bonferroni.md`
- `docs/multiple_testing/hochberg.md`
- `docs/multiple_testing/holm.md`
- `docs/multiple_testing/holm_sidak.md`
- `docs/multiple_testing/hommel.md`
- `docs/multiple_testing/implementation_report.md`
- `docs/multiple_testing/index.md`
- `docs/multiple_testing/simes.md`
- `docs/multiple_testing/rom.md`
- `docs/multiple_testing/holland_copenhaver.md`
- `docs/multiple_testing/westfall_young_max_t.md`
- `docs/multiple_testing/westfall_young_min_p.md`
- `docs/multiple_testing/sidak.md`
- `docs/multiple_testing/two_stage_bh.md`
- `src/pysatl_criterion/multiple_testing/__init__.py`
- `src/pysatl_criterion/multiple_testing/base.py`
- `src/pysatl_criterion/multiple_testing/inputs.py`
- `src/pysatl_criterion/multiple_testing/metadata.py`
- `src/pysatl_criterion/multiple_testing/pvalue/__init__.py`
- `src/pysatl_criterion/multiple_testing/pvalue/_numerics.py`
- `src/pysatl_criterion/multiple_testing/pvalue/_two_stage.py`
- `src/pysatl_criterion/multiple_testing/pvalue/base.py`
- `src/pysatl_criterion/multiple_testing/pvalue/benjamini_hochberg.py`
- `src/pysatl_criterion/multiple_testing/pvalue/benjamini_yekutieli.py`
- `src/pysatl_criterion/multiple_testing/pvalue/bky.py`
- `src/pysatl_criterion/multiple_testing/pvalue/bonferroni.py`
- `src/pysatl_criterion/multiple_testing/pvalue/hochberg.py`
- `src/pysatl_criterion/multiple_testing/pvalue/holm.py`
- `src/pysatl_criterion/multiple_testing/pvalue/holm_sidak.py`
- `src/pysatl_criterion/multiple_testing/pvalue/hommel.py`
- `src/pysatl_criterion/multiple_testing/pvalue/simes.py`
- `src/pysatl_criterion/multiple_testing/pvalue/rom.py`
- `src/pysatl_criterion/multiple_testing/pvalue/holland_copenhaver.py`
- `src/pysatl_criterion/multiple_testing/pvalue/sidak.py`
- `src/pysatl_criterion/multiple_testing/pvalue/two_stage_bh.py`
- `src/pysatl_criterion/multiple_testing/resampling/__init__.py`
- `src/pysatl_criterion/multiple_testing/resampling/base.py`
- `src/pysatl_criterion/multiple_testing/resampling/westfall_young_max_t.py`
- `src/pysatl_criterion/multiple_testing/resampling/westfall_young_min_p.py`
- `src/pysatl_criterion/multiple_testing/result.py`
- `src/pysatl_criterion/py.typed`
- `tests/multiple_testing/fixtures/statsmodels.json`
- `tests/multiple_testing/generate_reference.py`
- `tests/multiple_testing/test_procedures.py`
- `tests/multiple_testing/test_validation.py`
- `tests/multiple_testing/test_extended.py`

## Earlier refactor validation

The following records describe earlier refactor runs, before adding Simes, Rom,
Holland–Copenhaver and concrete Westfall–Young procedures. Current checks are
recorded separately below.

Executed locally with Python 3.11.11, NumPy 2.4.6, SciPy 1.17.1 and statsmodels
0.14.6. Commands used `.venv/bin/` executables. statsmodels is a development-only
dependency; the runtime algorithms use NumPy. Reference fixtures contain 240
method/input/alpha combinations and record the generating implementation version.

| Command | Outcome |
|---|---|
| `python tests/multiple_testing/generate_reference.py` | Generated independent statsmodels fixtures; benign log1p endpoint warnings in reference library |
| `pytest tests/multiple_testing -q` | 463 passed, 5 opt-in simulation tests skipped |
| `PYSATL_MULTIPLE_TESTING_MONTE_CARLO=1 pytest tests/multiple_testing/test_validation.py -k monte_carlo -s -q` | 5 passed; both global-null and mixture scenarios |
| `mypy src/pysatl_criterion tests` | Passed, 243 source files |
| `mypy --strict --follow-imports=silent src/pysatl_criterion/multiple_testing` | Passed, all 26 source modules; negative call-site test separately checks rejection of wrong inputs |
| `ruff check` on new/changed Python paths | Passed |
| `ruff format --check` on new/changed Python paths and `docs/multiple_testing` | Passed |
| `mkdocs build --strict -d /tmp/pysatl-multiple-site` | Passed; existing pages outside navigation produce informational notices |
| `git diff --check` | Passed |
| `python -m pip wheel --no-deps --no-build-isolation . -w /tmp/pysatl-multiple-wheel` | Passed after installing the missing local poetry-core backend; wheel inspected for new modules and `py.typed` |
| `pytest -q` | Collection blocked by two existing missing-module imports, detailed below |
| `pytest -q --ignore=tests/loader/test_remote_loader.py --ignore=tests/test_protocol_coverage.py` | 6439 passed, 71 skipped, 1 unrelated Weibull metadata failure (run before adding this report's example) |
| Repository-wide `ruff check` / `ruff format --check` | Failed on existing issues outside this change; scoped checks passed |
| `pre-commit run --all-files` | Failed on old mypy, import sorting and codespell findings; hook changes outside the task were reverted |
| `pre-commit run --files` with the manifest's relevant Python/docs/config paths | All hooks passed except existing mypy 1.13 errors in `distribution/parameters.py:63,78` (`Never.parameterization`) |

The full-suite missing modules are `pysatl_criterion.loader.remote_loader` and
`pysatl_criterion.persistence.sqlalchemy.alchemy_decorator`. The remaining failure
is `test_all_weibull_statistics_expose_metadata`, which expects 19 classes but
finds 18. These files and APIs are outside the multiple-testing change.
The regular installed mypy succeeds; pre-commit uses an older isolated version.
No check failure was suppressed or converted into a passing result.

Python 3.10 and 3.12 execution and remote CI were not run locally. Source syntax
uses Python 3.10-compatible generics. Documentation examples run in the normal
pytest suite used by CI. Publishing, pushing and releasing were not performed.

### Monte Carlo observations

Each cell uses 5,000 families, 20 hypotheses, seed 62173, alpha 0.05. The mixture
has five false-null p-values fixed at zero. All true-null p-values are independent
uniform draws. The acceptance upper bound is alpha plus six binomial standard
errors (approximately 0.0685). This is a reproducible diagnostic, not a proof.

| Method | Global null | Mixture | Quantity |
|---|---|---|---|
| Bonferroni | 0.04480 | 0.04340 | FWER |
| Holm | 0.04480 | 0.05400 | FWER |
| BH | 0.04640 | 0.03765 | FDR |
| BY | 0.01100 | 0.01160 | FDR |
| Two-stage BKY | 0.04440 | 0.04920 | FDR |

## Statistical verification and limits

Formulas were checked using primary publications/manuscripts or authoritative
software documentation, plus independent numerical tests. The original Holm,
Šidák, BH, BY and BKY sources and publisher abstracts for Hochberg and Hommel were
consulted; full publisher PDFs were not available for every method. The
[R documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html)
and independent exhaustive closure supply additional checks for Hommel. No claim
is made that inaccessible full papers were read. Original scientific authors and
verified historical PySATL contributors are separate fields on every method page.

Guarantees require valid single-test p-values. Holm/Bonferroni and BY permit
arbitrary dependence; Šidák methods and the BKY guarantee here assume independence.
BH permits PRDS on true nulls, while Hochberg/Hommel need valid intersection Simes
tests. Uncorrected two-stage BH is explicitly exploratory without nominal FDR
control asserted. Only two-stage BKY is implemented; its iterated/multistage
variants are deferred. Storey needs a specified and tested pi0 estimator.
Westfall–Young now provides the algorithms and independent permutation checks;
the experimental resampling design, calibration and subset pivotality remain the
caller's responsibility. Rom's recurrence was checked against the authors' academic
supplement and independently evaluated using 70-digit arithmetic; its full original
publisher paper was unavailable. Inverse critical values, not alpha-specific
multipliers, define its adjusted p-values.

## Final module layout and API

All production code lives in `src/pysatl_criterion/multiple_testing`.
The generic contract is in `base.py`, result types in `result.py`, p-value
contracts in `pvalue/base.py`, and joint-resampling contracts in `resampling/base.py`.
Each concrete procedure has its own named module in `pvalue/` or `resampling/`.
Legacy `abstract_multiple_testing.py`, `fdr.py`, `fwer.py` and the former directory
under `hypothesis_testing` have been removed. Root and `api.testing` exports use
the typed classes. The previously deleted audit page remains deleted.

## Added-method validation

Executed with the local `.venv` after adding the five procedures and their pages:

| Check | Result |
|---|---|
| `pytest tests/multiple_testing -q` | 544 passed, 5 opt-in simulations skipped |
| Opt-in Monte Carlo checks in `test_validation.py` | 5 passed |
| Public API tests, excluding the known Weibull metadata count failure | 33 passed, 1 deselected |
| Strict mypy on `multiple_testing` | Passed, 26 source files |
| Mypy on source and tests | Passed, 242 source files |
| Ruff lint and formatting on the module, exports and its tests | Passed |
| `mkdocs build --strict` | Passed |
| `git diff --check` | Passed |

The 544 tests include all five new documentation examples, 70-digit Rom reference
calculations, analytic inversion, independent Holland–Copenhaver comparisons,
Simes scalar results, exhaustive Westfall–Young closure and sign-flip FWER checks,
resampling modes and tails, input validation, registries and public exports.
The static negative test verifies that maxT and minP reject incompatible inputs
at type-check time. The `py.typed` marker exposes these contracts to callers.

The complete repository pytest suite was not rerun for this extension. Previously
observed unrelated failures are recorded above. The five opt-in Monte Carlo tests
cover the original procedures; new resampling methods instead have deterministic
exhaustive randomization checks in the normal unit suite.
