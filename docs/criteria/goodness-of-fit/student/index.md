# Student goodness-of-fit criteria

The module `pysatl_criterion.statistics.goodness_of_fit.continuous.student`
exports **21 concrete criteria**: 13 EDF, 5 frequency and 3 specialized classes.
Each has its own page, runnable example and scientific references below.
See [mathematical verification](mathematical-validation.md) for formulas,
source qualifications, calibration restrictions and numerical evidence.

## Choosing a hypothesis

| Family | Fixed parameters | Estimated parameters | Domain |
|---|---|---|---|
| EDF and frequency | `df`, `loc`, `scale` | None | `df > 0`, `scale > 0` |
| Meintanis | `df` | Location and scale, population moments (`ddof=0`) | `df > 2` |
| Çadirci | `df` | Sample variance (`ddof=1`) | `df > 2` |
| Lequesne T1 | `df` | Sample variance (`ddof=1`) | `df > 2` |
| Lequesne T2 | None | `df`, location and scale by MLE | Successful fit with `df > 2` |

Use `Student.DEFAULT.parse({'df': 5, 'loc': 0, 'scale': 1})` for a fully
specified null, `Student.DEFAULT.parse({'df': 5})` for a fixed-df composite
null, and `Student.DEFAULT.parse({})` only for Lequesne T2. Parameters omitted
from `parse` remain unknown. No criterion silently fills them with defaults.
All criteria reject for large values. Entropy estimates may be negative;
Neyman and modified log-likelihood may be infinite for empty observed bins.

## EDF criteria

Class names below have the suffix `StudentGofStatistic`. Full codes append
`_STUDENT_GOODNESS_OF_FIT` to the short code.

| Criterion | Class prefix | Short code |
|---|---|---|
| [Kolmogorov–Smirnov D+](kolmogorov_smirnov_d_plus_student_criterion.md) | `KolmogorovSmirnovDPlus` | `KS_PLUS` |
| [Kolmogorov–Smirnov D−](kolmogorov_smirnov_d_minus_student_criterion.md) | `KolmogorovSmirnovDMinus` | `KS_MINUS` |
| [Kolmogorov–Smirnov](kolmogorov_smirnov_student_criterion.md) | `KolmogorovSmirnov` | `KS` |
| [Anderson–Darling](anderson_darling_student_criterion.md) | `AndersonDarling` | `AD` |
| [Cramér–von Mises](cvm_student_criterion.md) | `CramerVonMises` | `CVM` |
| [Liao–Shimokawa](liao_shimokawa_student_criterion.md) | `LiaoShimokawa` | `LS` |
| [Watson](watson_student_criterion.md) | `Watson` | `WATSON` |
| [Kuiper](kuiper_student_criterion.md) | `Kuiper` | `KUIPER` |
| [Scaled Kuiper](kuiper_scaled_student_criterion.md) | `KuiperScaled` | `KUIPER_SCALED` |
| [Corrected Kuiper](kuiper_corrected_student_criterion.md) | `KuiperCorrected` | `KUIPER_CORRECTED` |
| [Zhang Z_K](zhang_k_student_criterion.md) | `ZhangK` | `ZHANG_ZK` |
| [Zhang Z_A](zhang_za_student_criterion.md) | `ZhangZa` | `ZHANG_ZA` |
| [Zhang Z_C](zhang_zc_student_criterion.md) | `ZhangZc` | `ZHANG_ZC` |

## Frequency criteria

Class names below have the suffix `StudentGofStatistic`. Full codes append
`_STUDENT_GOODNESS_OF_FIT` to the short code.

| Criterion | Class prefix | Short code |
|---|---|---|
| [Pearson chi-squared](chi_square_student_criterion.md) | `ChiSquare` | `CHI2_PEARSON` |
| [Freeman–Tukey](freeman_tukey_student_criterion.md) | `FreemanTukey` | `FREEMAN_TUKEY` |
| [Modified log-likelihood](modified_log_likelihood_student_criterion.md) | `ModifiedLogLikelihood` | `MODIFIED_LOG_LIKELIHOOD` |
| [Neyman chi-squared](neyman_chi2_student_criterion.md) | `NeymanChi2` | `CHI2_NEYMAN` |
| [Cressie–Read](cressie_read_student_criterion.md) | `CressieRead` | `CRESSIE_READ` |

## Specialized criteria

Class names below have the suffix `StudentGofStatistic`. Full codes append
`_STUDENT_GOODNESS_OF_FIT` to the short code.

| Criterion | Class prefix | Short code |
|---|---|---|
| [Meintanis ECF](meintanis_student_criterion.md) | `Meintanis` | `MEINTANIS_ECF` |
| [Çadirci Rényi kNN](cadirci_renyi_student_criterion.md) | `CadirciRenyi` | `CADIRCI_RENYI` |
| [Lequesne Rényi](lequesne_renyi_student_criterion.md) | `LequesneRenyi` | `LEQUESNE_RENYI` |

## Examples and calibration

```python
from pysatl_criterion.distribution import StudentDistributionDescriptor as Student
from pysatl_criterion.statistics.goodness_of_fit.continuous.student import (
    KolmogorovSmirnovStudentGofStatistic,
    LequesneRenyiStudentGofStatistic,
)

sample = [-3.2, -1.1, -0.3, 0.2, 0.7, 1.8, 4.1]
fixed = KolmogorovSmirnovStudentGofStatistic(
    Student.DEFAULT.parse({'df': 5, 'loc': 0, 'scale': 1})
)
composite = LequesneRenyiStudentGofStatistic(Student.DEFAULT.parse({'df': 5}), k=3)
print(fixed.execute_statistic(sample), composite.execute_statistic(sample))
```

The APIs return statistics, not p-values. Fixed-df composite calibration must
recompute the moments/neighbours on every Monte Carlo replicate, keeping `df`
and algorithm settings. Frequency calibration preserves the binning strategy.
Lequesne T2 requires external parametric bootstrap at explicitly fitted
simulation parameters and refits all parameters on every replicate; the
built-in generator and storage lookup reject an unspecified simulation `df`.

The implementations here are one-dimensional and assume independent observations.
The cited multivariate and GARCH research does not imply those models are
implemented by this module. Theorem assumptions can be stronger than the
computational domain; consult the individual pages before choosing calibration.
