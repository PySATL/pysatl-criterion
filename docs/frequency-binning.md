# Frequency statistics and binning

For the public user guide, comparison table and method examples, see
[Binning for frequency goodness-of-fit tests](criteria/goodness-of-fit/binning/index.md).

Frequency kernels in `continuous/frequency.py` take `observed` and `expected`
arrays. They do not choose bins, estimate distribution parameters, or calculate
p-values. The power-divergence family underlying these kernels is described by
[Cressie and Read (1984)](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x).
Five frequency criteria are available for uniformity:

| Class | Code |
| --- | --- |
| `Chi2PearsonUniformGofStatistic` | `CHI2_PEARSON` |
| `FreemanTukeyUniformGofStatistic` | `FREEMAN_TUKEY` |
| `ModifiedLogLikelihoodUniformGofStatistic` | `MODIFIED_LOG_LIKELIHOOD` |
| `NeymanChi2UniformGofStatistic` | `CHI2_NEYMAN` |
| `CressieReadUniformGofStatistic` | `CRESSIE_READ` |

The old `bins` and `lambda_` constructor arguments have been removed. Select
an algorithm by class and a grouping policy with `binning`:

```python
from pysatl_criterion.distribution import UniformDistributionDescriptor as Uniform
from pysatl_criterion.statistics.goodness_of_fit.binning import (
    EqualProbabilityBins, SturgesCount,
)
from pysatl_criterion.statistics.goodness_of_fit.continuous.uniform import (
    CressieReadUniformGofStatistic,
)

criterion = CressieReadUniformGofStatistic(
    Uniform.DEFAULT.parse({"a": 0, "b": 1}),
    binning=EqualProbabilityBins(count=SturgesCount(), min_expected=5),
)
```

Both Uniform bounds must be fixed. The criterion validates and binds the
null distribution on each execution, builds a frequency table, then calls the
kernel. Strategies are retained as configuration; tables are never cached across
samples. This also applies to Monte Carlo calibration.

## Number of bins

All count rules implement `resolve(sample_size)`:

| Rule | Requested count | Source / status |
| --- | --- | --- |
| Integer or `FixedCount(k)` | `k`, an integer >= 2 | User-selected configuration. |
| `SturgesCount()` | `max(2, ceil(1 + log2(n)))` | [Sturges (1926)](https://doi.org/10.1080/01621459.1926.10502161); rounding and the minimum of two are implementation choices. |
| `SqrtCount()` | `max(2, ceil(sqrt(n)))` | Convenience heuristic; no claim of an optimal GoF rule or attribution to a specific paper. |
| `PowerCount(coefficient=2, exponent=0.4)` | `max(2, ceil(coefficient * n**exponent))` | Default heuristic documented in [NIST §7.2.1.1](https://www.itl.nist.gov/div898/handbook/prc/section2/prc211.htm); NIST is a handbook, not a research paper. |

The default is `EqualProbabilityBins(PowerCount())`, with `max_bins=1000` and
`min_expected=None`. PowerCount is an explicit heuristic, not a claim of optimal
power or an implementation of a specific Mann–Wald formula. Counts are recomputed
for each sample size. `max_bins` caps the requested count, including fixed counts.
For the historical problem of choosing the number of chi-square classes, see
[Mann and Wald (1942)](https://doi.org/10.1214/aoms/1177731569). This reference is
background, not the source of the exact default coefficient or rounding policy.

`min_expected=None` permits sparse tables for simulation-based calibration. A
positive value, such as 5, is an explicit minimum expected count policy; it does
not by itself establish the validity of a chi-square approximation. Insufficient
sample sizes raise an error instead of silently reducing the table to one cell.
For a discussion of expected counts and chi-square calibration, see
[Rolke and Gutierrez Gongora, preprint §2](https://arxiv.org/html/2005.02793v1#S2).

## Strategies

Equal-probability and equal-width binning are discussed in
[Rolke and Gutierrez Gongora, preprint §5.1](https://arxiv.org/html/2005.02793v1#S5.SS1).
Their equation (3) also gives the interpolation of the two sets of boundaries.
Our tail handling, endpoint delegation and numerical CDF inversion are explicit
implementation policies described below, rather than a verbatim implementation
of the paper.

- `EqualProbabilityBins(count=..., min_expected=None, max_bins=1000)` transforms
  the observations with the continuous null CDF and uses equal-width probability
  cells. Expected counts are `n/k`. With `min_expected`, k is capped by
  `floor(n/min_expected)` before building the table.
- `ExplicitBins(edges=(...), min_expected=None)` uses full-support boundaries.
  Continuous cells follow NumPy's `[left, right)` convention, with the last right
  endpoint included. Bounds must include all null probability and all sample
  values. Use infinite endpoints for unbounded distributions.
- `EqualWidthBins(lower, upper, count=..., min_expected=None, max_bins=1000)` uses
  fixed finite central bounds and adds separate lower/upper tail categories when
  they have positive null probability. Count and max_bins apply to central cells;
  there may be two additional tail cells. Bounds must be specified independently
  of the tested sample.
- `InterpolatedBins(lower, upper, count=..., kappa=0.5, ...)` interpolates interior
  boundaries between null quantiles and equal-width boundaries, keeping full
  coverage. Null quantiles are obtained by numerical CDF inversion because the
  distribution evaluation protocol has no PPF. Failure to bracket a quantile
  raises a clear error. For `0 < kappa < 1`, equal-probability quantiles use the
  total count including tails. At exactly 0 it delegates to EqualProbabilityBins;
  at exactly 1 it delegates to EqualWidthBins. Thus the endpoint strategies can
  have different cell counts when tails are present.
- `DiscreteCategoryBins(cut_points=(...), min_expected=None)` uses right-closed
  groups `(-inf, c1]`, `(c1, c2]`, ..., `(cm, inf)`. Null probabilities are actual
  CDF differences, not `1/k`. Integer counts, real-valued atoms, and grouped
  discrete support all use the same convention. Uniform rejects this strategy.

For explicit, width-based and discrete strategies, `min_expected` merges the
smallest expected cell with its right neighbor (or left neighbor at the end)
until the requirement is met. Selection depends only on null expectations,
not the observed discrepancies. Zero-probability cells with zero observations
are removed. An observed value in a zero-probability category raises an error.
The result must contain at least two categories. This particular adjacent-cell
merging algorithm is a library policy; it is not attributed to the cited papers.

Interval probabilities use survival-function differences in the upper tail to
avoid cancellation. `FrequencyTable` validates positive expectations, aligned
one-dimensional arrays, finite nonnegative observations and equal totals; it owns
read-only copies of both arrays.

## Selection for a specified alternative

`AlternativeOptimizedBins.select(...)` performs a reproducible Monte Carlo search
across explicit candidates. Supply a bound null distribution, null and alternative
samplers `(n, rng) -> sample`, a frequency kernel, sample size, seed, repetition
count and significance level. Each candidate is scored using synthetic null
critical values and its estimated rejection rate on alternative samples.

The returned strategy freezes the winning candidate and sample size. It never
selects bins from the tested observations. Ties keep the first candidate. Rebuild
it for another sample size, null model, or kernel; the caller must use the same
model and kernel as during selection. Power scores are selection estimates, not
unbiased performance estimates. Calibrate the selected test on fresh null
simulations. The idea of selecting bins for a specified alternative is discussed
by [Rolke and Gutierrez Gongora (2021)](https://doi.org/10.1007/s00180-020-00997-x),
with an [open preprint, §5.3](https://arxiv.org/html/2005.02793v1#S5.SS3).
Our Monte Carlo candidate search is not an implementation of their exact selection
algorithm.

## Calibration

`code()` identifies the algorithm, `hypothesis()` describes the null hypothesis,
and `configuration()` describes computation settings. Storage keys include the
canonical JSON configuration hash alongside the code hash, hypothesis parameter
hash and sample size. Both limit distributions and critical values use this key.
Different binning policies therefore have independent calibrations.

Create new calibration records with
`LimitDistributionModel.from_statistic(criterion, sample_size, values)` after
simulation. `StorageLimitDistributionResolver` constructs the matching key
through `DistributionKey.from_statistic`. Direct model construction without a
configuration denotes legacy/unspecified data, not the current defaults.

Built-in binning policies expose `to_config()`, including defaults and policy
versions. Integer counts and `FixedCount` serialize identically. JSON keys are
sorted, equivalent numeric values are normalized, and infinite explicit edges
use `"-inf"` and `"+inf"`. Update the corresponding version whenever mathematical
behavior changes. Custom policies and count rules need a stable `to_config()`
dictionary for storage; Monte Carlo alone does not require it.
`AlternativeOptimizedBins` records its selected policy and sample size; estimated
powers and search provenance do not participate in calibration identity.

Use a new database for the configuration-aware schema.
`AlchemyDatabase.init()` rejects incompatible existing tables; it does not alter
their schema or transfer old calibration records.

See the [uniformity catalog](criteria/goodness-of-fit/uniformity/index.md)
for criterion descriptions, formulas, and usage examples.

## References

1. Pearson, K. (1900). *On the criterion that a given system of deviations from
   the probable in the case of a correlated system of variables is such that it
   can be reasonably supposed to have arisen from random sampling.*
   Philosophical Magazine, 50, 157–175.
   [DOI](https://doi.org/10.1080/14786440009463897).
   Original Pearson chi-square statistic.
2. Cressie, N., and Read, T. R. C. (1984). *Multinomial Goodness-of-Fit Tests.*
   Journal of the Royal Statistical Society, Series B, 46(3), 440–464.
   [DOI](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x),
   [publisher page](https://academic.oup.com/jrsssb/article/46/3/440/7028099).
   Power-divergence family used by the frequency kernels.
3. Sturges, H. A. (1926). *The Choice of a Class Interval.*
   Journal of the American Statistical Association, 21(153), 65–66.
   [DOI](https://doi.org/10.1080/01621459.1926.10502161),
   [article scan](https://www.stat.rice.edu/~scottdw/stat550/DEMOS/09-14-hist/Sturges1926.pdf).
   Basis for the logarithmic count rule, not a universal GoF power guarantee.
4. Mann, H. B., and Wald, A. (1942). *On the Choice of the Number of Class
   Intervals in the Application of the Chi Square Test.*
   The Annals of Mathematical Statistics, 13(3), 306–317.
   [DOI](https://doi.org/10.1214/aoms/1177731569).
   Historical reference on choosing chi-square class intervals.
5. Rolke, W., and Gutierrez Gongora, C. (2021). *A chi-square goodness-of-fit test
   for continuous distributions against a known alternative.*
   Computational Statistics, 36, 1885–1900.
   [DOI](https://doi.org/10.1007/s00180-020-00997-x),
   [open 2020 preprint](https://arxiv.org/abs/2005.02793).
   Binning types, interpolation and selection against a specified alternative.
   Section and equation links above refer to preprint version 1.

Additional practical guidance:
[NIST/SEMATECH e-Handbook, §7.2.1.1, Chi-square goodness-of-fit test](https://www.itl.nist.gov/div898/handbook/prc/section2/prc211.htm).
This handbook supplies the `2 * n**0.4` starting heuristic used by the default
count rule; it is listed separately from the research articles.
