# Beta goodness-of-fit criteria

The package `pysatl_criterion.statistics.goodness_of_fit.continuous.beta` exports
**23 concrete criteria**: 13 EDF, 5 frequency, Neyman smooth and 4 fitted-Beta tests. Every criterion has a page below.
See the [mathematical verification report](mathematical-validation.md) for
formula checks, numerical regressions and the limits of the scientific attribution.

## Package layout

The layout follows `continuous/uniform`:

```text
continuous/beta/
├── __init__.py          # public exports and stable class module paths
├── __init__.pyi         # generated public type declarations
└── beta_statistics.py  # implementations and dynamic EDF/frequency catalogs
```

Existing imports from `continuous.beta` remain valid. The public module paths
are preserved for discovery and previously serialized class references.
Regenerate the stub with `python scripts/generate_beta_stub.py`; use `--check`
to detect drift. The generator reads the implementation without importing the
package and uses the project's Ruff formatter.

## Hypotheses and calibration

The support is fixed at $[0,1]$, with positive finite shapes $a,b$.
For the 19 specified-null criteria use `Beta.DEFAULT.parse({"a": 2, "b": 5})`.
Both shapes must be chosen independently of the tested sample; there are no
default shape values.
For Ebner–Liebenberg, Raschke, Riad–Abd El Mabood and Kullback–Leibler use `Beta.DEFAULT.parse({})`; both shapes are
estimated by maximum likelihood for every sample. Partially specified shapes
are not supported by these 23 classes.

All criteria return scalar statistics and reject large values. Executing a
statistic does not compute a p-value. Fixed-null EDF and smooth statistics use
the probability integral transform; fitted-Beta tests require a parametric
bootstrap with refitting.

## Complete catalog

Full identifiers append `_BETA_GOODNESS_OF_FIT` to the short codes below.
`CrammerVonMisesBetaGofStatistic` is the exported spelling; no alternative alias
is supplied. KS D+ and D− are separate classes. Frequency settings use `binning`,
not `lambda_`; Watson's short code is `WATSON`.

| Criterion                                                             | Class                                     | Short code                | Status and hypothesis                        |
|-----------------------------------------------------------------------|-------------------------------------------|---------------------------|----------------------------------------------|
| [Kolmogorov–Smirnov D+](kolmogorov_smirnov_d_plus_beta_criterion.md)  | `KolmogorovSmirnovDPlusBetaGofStatistic`  | `KS_PLUS`                 | EDF; fixed shapes                            |
| [Kolmogorov–Smirnov D−](kolmogorov_smirnov_d_minus_beta_criterion.md) | `KolmogorovSmirnovDMinusBetaGofStatistic` | `KS_MINUS`                | EDF; fixed shapes                            |
| [Kolmogorov–Smirnov](kolmogorov_smirnov_beta_criterion.md)            | `KolmogorovSmirnovBetaGofStatistic`       | `KS`                      | EDF; fixed shapes                            |
| [Anderson–Darling](anderson_darling_beta_criterion.md)                | `AndersonDarlingBetaGofStatistic`         | `AD`                      | EDF; fixed shapes                            |
| [Cramér–von Mises](cvm_beta_criterion.md)                             | `CrammerVonMisesBetaGofStatistic`         | `CVM`                     | EDF; fixed shapes                            |
| [Liao–Shimokawa](liao_shimokawa_beta_criterion.md)                    | `LiaoShimokawaBetaGofStatistic`           | `LS`                      | EDF; fixed shapes                            |
| [Watson](watson_beta_criterion.md)                                    | `WatsonBetaGofStatistic`                  | `WATSON`                  | EDF; fixed shapes                            |
| [Kuiper](kuiper_beta_criterion.md)                                    | `KuiperBetaGofStatistic`                  | `KUIPER`                  | EDF; fixed shapes                            |
| [Scaled Kuiper](kuiper_scaled_beta_criterion.md)                      | `KuiperScaledBetaGofStatistic`            | `KUIPER_SCALED`           | EDF; fixed shapes                            |
| [Corrected Kuiper](kuiper_corrected_beta_criterion.md)                | `KuiperCorrectedBetaGofStatistic`         | `KUIPER_CORRECTED`        | EDF; fixed shapes                            |
| [Zhang Z_K](zhang_k_beta_criterion.md)                                | `ZhangKBetaGofStatistic`                  | `ZHANG_ZK`                | EDF; fixed shapes                            |
| [Zhang Z_A](zhang_a_beta_criterion.md)                                | `ZhangABetaGofStatistic`                  | `ZHANG_ZA`                | EDF; fixed shapes                            |
| [Zhang Z_C](zhang_c_beta_criterion.md)                                | `ZhangCBetaGofStatistic`                  | `ZHANG_ZC`                | EDF; fixed shapes                            |
| [Pearson chi-squared](chi2_pearson_beta_criterion.md)                 | `Chi2PearsonBetaGofStatistic`             | `CHI2_PEARSON`            | Frequency; fixed shapes                      |
| [Freeman–Tukey](freeman_tukey_beta_criterion.md)                      | `FreemanTukeyBetaGofStatistic`            | `FREEMAN_TUKEY`           | Frequency; fixed shapes                      |
| [Modified log-likelihood](modified_log_likelihood_beta_criterion.md)  | `ModifiedLogLikelihoodBetaGofStatistic`   | `MODIFIED_LOG_LIKELIHOOD` | Frequency; fixed shapes                      |
| [Neyman chi-squared](neyman_chi2_beta_criterion.md)                   | `NeymanChi2BetaGofStatistic`              | `CHI2_NEYMAN`             | Frequency; fixed shapes                      |
| [Cressie–Read](cressie_read_beta_criterion.md)                        | `CressieReadBetaGofStatistic`             | `CRESSIE_READ`            | Frequency; fixed shapes                      |
| [Neyman smooth](neyman_smooth_beta_criterion.md)                      | `NeymanSmoothBetaGofStatistic`            | `NEYMAN`                  | Published functional; fixed shapes           |
| [Ebner–Liebenberg](ebner_liebenberg_beta_criterion.md)                | `EbnerLiebenbergBetaGofStatistic`         | `EL`                      | Published Beta test; both shapes fitted      |
| [Raschke](raschke_beta_criterion.md)                                  | `RaschkeBetaGofStatistic`                 | `RASCHKE`                 | Published transformation; both shapes fitted |
| [Riad–Abd El Mabood](riad_abd_el_mabood_beta_criterion.md)            | `RiadAbdElMaboodBetaGofStatistic`         | `RF`                      | Published functional; both shapes fitted     |
| [Kullback–Leibler (Vasicek)](kullback_leibler_beta_criterion.md)      | `KullbackLeiblerBetaGofStatistic`         | `KL`                      | Published functional; both shapes fitted     |

The two new pages specify exact formulas and document the limits of full-text
source verification; numerical equivalence to historical tables is not claimed.

## Examples

```python
from pysatl_criterion.distribution import BetaDistributionDescriptor as Beta
from pysatl_criterion.statistics.goodness_of_fit.continuous.beta import (
    AndersonDarlingBetaGofStatistic,
    EbnerLiebenbergBetaGofStatistic,
)

sample = [0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57]
fixed = AndersonDarlingBetaGofStatistic(Beta.DEFAULT.parse({"a": 2, "b": 5}))
fitted = EbnerLiebenbergBetaGofStatistic(Beta.DEFAULT.parse({}))
print(fixed.execute_statistic(sample))
print(fitted.execute_statistic(sample))
```

For a fitted test, call `parametric_bootstrap_beta` as illustrated on its
individual page. Never reuse a fixed-null EDF table after fitting shapes.
For frequency tests, preserve the [binning strategy](../binning/index.md)
and its count rule in calibration. Existing calibrations from the old
Pearson equal-width default or altered formulas are not interchangeable.
