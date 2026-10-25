# Pareto I goodness-of-fit criteria

All **21 criteria** use the zero-location Pareto I model

$$
F_{a,s}(x)=1-(s/x)^a,\qquad x\ge s>0,\quad a>0.
$$

`execute_statistic` returns a statistic, not a p-value. Every sample must be
finite, real, one-dimensional and unmasked. The table distinguishes fixed and
unknown parameters; omitted parameters are unknown and are never filled with defaults.
See [mathematical verification](mathematical-validation.md) for formulas,
scientific sources, independent checks and limitations.

## Fully specified EDF criteria

Specify both `shape` and `scale`. These 13 criteria do not fit parameters.
All use the right critical tail, including D+ and D−. Values must be at least
scale. Minimum computational size is one; Watson and Kuiper are constant at
that size and cannot provide useful discrimination.

| Criterion | Public class | Short code |
|---|---|---|
| [Kolmogorov–Smirnov](kolmogorov_smirnov_pareto_criterion.md) | `KolmogorovSmirnovParetoGofStatistic` | `KS` |
| [Kolmogorov–Smirnov D+](kolmogorov_smirnov_d_plus_pareto_criterion.md) | `KolmogorovSmirnovDPlusParetoGofStatistic` | `KS_PLUS` |
| [Kolmogorov–Smirnov D−](kolmogorov_smirnov_d_minus_pareto_criterion.md) | `KolmogorovSmirnovDMinusParetoGofStatistic` | `KS_MINUS` |
| [Anderson–Darling](anderson_darling_pareto_criterion.md) | `AndersonDarlingParetoGofStatistic` | `AD` |
| [Cramér–von Mises](cramer_von_mises_pareto_criterion.md) | `CramerVonMisesParetoGofStatistic` | `CVM` |
| [Liao–Shimokawa functional (legacy MinToshiyuki name)](min_toshiyuki_pareto_criterion.md) | `MinToshiyukiParetoGofStatistic` | `MT` |
| [Watson](watson_pareto_criterion.md) | `WatsonParetoGofStatistic` | `WATSON` |
| [Kuiper](kuiper_pareto_criterion.md) | `KuiperParetoGofStatistic` | `KUIPER` |
| [Scaled Kuiper](kuiper_scaled_pareto_criterion.md) | `KuiperScaledParetoGofStatistic` | `KUIPER_SCALED` |
| [Corrected Kuiper](kuiper_corrected_pareto_criterion.md) | `KuiperCorrectedParetoGofStatistic` | `KUIPER_CORRECTED` |
| [Zhang Z_K](zhang_k_pareto_criterion.md) | `ZhangKParetoGofStatistic` | `ZHANG_ZK` |
| [Zhang Z_A](zhang_a_pareto_criterion.md) | `ZhangAParetoGofStatistic` | `ZHANG_ZA` |
| [Zhang Z_C](zhang_c_pareto_criterion.md) | `ZhangCParetoGofStatistic` | `ZHANG_ZC` |

## Fully specified frequency criteria

Specify both parameters. Each class accepts keyword-only `binning`;
the default is `EqualProbabilityBins(PowerCount(2.0, 0.4), min_expected=None, max_bins=1000)`.
See [binning policies](../binning/index.md) and
[calibration storage](../binning/calibration.md). Empty observed cells can produce
positive infinity for modified log-likelihood and Neyman; that is their mathematical limit.

| Criterion | Public class | Short code |
|---|---|---|
| [Pearson chi-square](chi2_pearson_pareto_criterion.md) | `Chi2PearsonParetoGofStatistic` | `CHI2_PEARSON` |
| [Freeman–Tukey](freeman_tukey_pareto_criterion.md) | `FreemanTukeyParetoGofStatistic` | `FREEMAN_TUKEY` |
| [Modified log-likelihood](modified_log_likelihood_pareto_criterion.md) | `ModifiedLogLikelihoodParetoGofStatistic` | `MODIFIED_LOG_LIKELIHOOD` |
| [Neyman modified chi-square](neyman_chi2_pareto_criterion.md) | `NeymanChi2ParetoGofStatistic` | `CHI2_NEYMAN` |
| [Cressie–Read](cressie_read_pareto_criterion.md) | `CressieReadParetoGofStatistic` | `CRESSIE_READ` |

## Composite hypotheses

| Criterion | Public class | Short code | Hypothesis |
|---|---|---|---|
| [Obradović–Jovanović–Milošević](obradovic_pareto_criterion.md) | `ObradovicParetoGofStatistic` | `OJM_Integral` | Known scale; unknown shape |
| [Greenwood logarithmic residual ratio](greenwood_pareto_criterion.md) | `GreenwoodParetoGofStatistic` | `Greenwood` | Unknown shape and scale |
| [Vasicek–Song fixed-window KL](lequesne_kl_pareto_criterion.md) | `LequesneKlParetoGofStatistic` | `Lequesne_KL` | Unknown shape and scale |

Obradović uses the upper tail and needs at least two observations above or at the
known scale. Greenwood uses both tails and needs at least three positive,
nonconstant observations. KL uses the upper tail and needs at least four positive,
nonconstant observations and an integer window `1 <= m < n/2`.

## Examples

```python
from pysatl_criterion.distribution import ParetoDistributionDescriptor as Pareto
from pysatl_criterion.statistics.goodness_of_fit.continuous.pareto import (
    AndersonDarlingParetoGofStatistic,
    GreenwoodParetoGofStatistic,
    LequesneKlParetoGofStatistic,
    ObradovicParetoGofStatistic,
)

x = [1.1, 1.4, 1.9, 2.7, 4.2, 8.0]
fixed = AndersonDarlingParetoGofStatistic(Pareto.DEFAULT.parse({"shape": 2, "scale": 1}))
known_scale = ObradovicParetoGofStatistic(Pareto.DEFAULT.parse({"scale": 1}))
greenwood = GreenwoodParetoGofStatistic(Pareto.DEFAULT.parse({}))
kl = LequesneKlParetoGofStatistic(Pareto.DEFAULT.parse({}))
print(fixed.execute_statistic(x), known_scale.execute_statistic(x))
print(greenwood.execute_statistic(x), kl.execute_statistic(x, m=2))
```

## Calibration and interpretation

The full code appends `_PARETO_GOODNESS_OF_FIT` to each short code.
Built-in Pareto simulation is unavailable. Fixed-null EDF statistics can be
calibrated through independent uniforms; frequency calibration must preserve
bin probabilities and binning policy. Composite criteria have different null laws.
Greenwood and Obradović eliminate the nuisance shape under continuous IID sampling;
KL retains finite-sample shape dependence and requires refitting with the same window
on each replicate. Plug-in parametric bootstrap is approximate.

The Greenwood Pareto formula is a library adaptation of the Greenwood concentration
functional. The legacy `MinToshiyuki` name represents the Liao–Shimokawa functional
on a fixed CDF. Neither is asserted to reproduce a published fitted-Pareto test
with supplied critical values. KL implements the raw fixed-window estimate,
not the adaptive, bias-corrected procedure of `vsgoftest`.
