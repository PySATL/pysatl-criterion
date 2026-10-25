# Binning for frequency goodness-of-fit tests

Binning converts a sample and a fully specified null distribution into observed
and expected category counts. A frequency statistic then measures the discrepancy
between those counts. Changing bins changes the test, even when its statistic
formula stays the same.

## How it works

```text
sample + fixed null parameters
              |
              v
count rule -> binning strategy -> observed / expected -> frequency statistic
```

For categories $C_1,\ldots,C_k$ and a sample of size $n$:

$$
O_i = \sum_{j=1}^{n}\mathbf{1}\{x_j\in C_i\},\qquad
E_i = n\,P_{H_0}(X\in C_i).
$$

A strategy's `build(sample, distribution)` returns a `FrequencyTable` with
`observed` and `expected` arrays. The arrays are read-only, have at least two
entries, and have matching totals. Observations must be a nonempty, finite,
one-dimensional real sample. Expected counts must be strictly positive.

The criterion validates its family-specific sample support, binds the
null parameters, builds a fresh table, and calls the frequency kernel. Binning
does not fit missing parameters or calculate p-values.

## All available methods

Choose one **count rule** and one compatible **binning strategy**. Explicit and
discrete strategies take boundaries directly and do not need a count rule.

| Method | Role | Rule or mechanism | Applicable input |
| --- | --- | --- | --- |
| [FixedCount](fixed-count.md) | Count | A specified integer $k\ge2$ | Strategies accepting a count |
| [SturgesCount](sturges-count.md) | Count | $\max(2,\lceil1+\log_2 n\rceil)$ | Strategies accepting a count |
| [SqrtCount](sqrt-count.md) | Count | $\max(2,\lceil\sqrt n\rceil)$ | Strategies accepting a count |
| [PowerCount](power-count.md) | Count | $\max(2,\lceil c n^p\rceil)$; default $c=2$, $p=0.4$ | Strategies accepting a count |
| [EqualProbabilityBins](equal-probability.md) | Strategy | Equal null probability; $E_i=n/k$ | Continuous null |
| [ExplicitBins](explicit.md) | Strategy | User-specified full-support edges | Continuous null |
| [EqualWidthBins](equal-width.md) | Strategy | Equal central widths plus nonempty null tails | Continuous null |
| [InterpolatedBins](interpolated.md) | Strategy | Interpolation of null-quantile and equal-width boundaries | Continuous null |
| [DiscreteCategoryBins](discrete.md) | Strategy | Right-closed categories with actual null masses | Discrete null |
| [AlternativeOptimizedBins](alternative-optimized.md) | Selection | Monte Carlo search over candidate strategies before testing | Compatible candidates and a specified alternative |

`FixedCount` and `SqrtCount` are configuration/heuristic choices, not claims of
optimal GoF power. Statistical sources and differences from published methods
are given on the individual pages.

## Quick start

```python
from pysatl_criterion.distribution import UniformDistributionDescriptor as Uniform
from pysatl_criterion.statistics.goodness_of_fit.binning import (
    EqualProbabilityBins, SturgesCount,
)
from pysatl_criterion.statistics.goodness_of_fit.continuous.uniform import (
    Chi2PearsonUniformGofStatistic,
)

parameters = Uniform.DEFAULT.parse({"a": 0, "b": 1})
criterion = Chi2PearsonUniformGofStatistic(
    parameters,
    binning=EqualProbabilityBins(count=SturgesCount()),
)
value = criterion.execute_statistic([0.1, 0.15, 0.4, 0.6, 0.8, 0.9])
assert value >= 0
```

The same `binning` argument is available on `FreemanTukeyUniformGofStatistic`,
`ModifiedLogLikelihoodUniformGofStatistic`, `NeymanChi2UniformGofStatistic`, and
`CressieReadUniformGofStatistic`. Choose the class for the desired formula;
the former `bins` and `lambda_` constructor arguments have been removed.

All strategies and count rules are imported from
`pysatl_criterion.statistics.goodness_of_fit.binning`. Standalone examples on
method pages use SciPy frozen distributions, which provide the required CDF/SF
methods; a descriptor's `bind(parameters)` result can be used as well.

## Defaults and small expected counts

The default is `EqualProbabilityBins(count=PowerCount(), max_bins=1000,
min_expected=None)`. It permits sparse tables for simulation-based calibration.
Counts are resolved again for every sample size. `max_bins` caps even an explicit
fixed count; width-based methods can add up to two extra tail categories.

Set `min_expected=5`, for example, to request a minimum expected-count policy.
For equal-probability bins, the count is capped by $\lfloor n/5\rfloor$.
Other strategies repeatedly merge the smallest expected category with its right
neighbor, or its left neighbor when last. Merging uses null expectations only.
If fewer than two valid categories remain, an error is raised.

A zero-probability category with no observations is removed. Observations in a
zero-probability category raise an error. No sample observations or tail mass
may be silently dropped. Minimum expected counts do not by themselves guarantee
an accurate asymptotic calibration.

## Calibration

The Uniform criteria support fully fixed bounds. Parameters are never
estimated implicitly. Continuous equal-probability binning must not be applied
to a discrete CDF; use discrete category probabilities instead.

Use `MonteCarloLimitDistributionResolver` to calibrate the configured criterion.
Every simulation rebuilds its bins. `StorageLimitDistributionResolver` retrieves
only an exact configuration match, including binning parameters. Legacy records
with unknown configuration do not match implicitly. See
[saving and retrieving calibrations](calibration.md) for examples and schema requirements.
A statistic value alone is not a p-value.

Selection for a specified alternative uses synthetic data before observing the
tested sample. Calibrate the selected strategy using fresh null simulations;
its reported selection power is not an unbiased performance estimate.

## Sources and related documentation

The frequency-kernel family is described in [Cressie and Read (1984), *Multinomial Goodness-of-Fit Tests*](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x).
Binning choices and alternative-specific selection are discussed in [Rolke and Gutierrez Gongora (2021), *A chi-square goodness-of-fit test for continuous distributions against a known alternative*](https://doi.org/10.1007/s00180-020-00997-x)
([open preprint, §5.1](https://arxiv.org/html/2005.02793v1#S5.SS1)). These sources motivate the methods; count caps, tail policies,
merging rules and the Monte Carlo candidate search are library choices.

See [frequency binning guide](../../../frequency-binning.md) and
[universal goodness-of-fit statistics](../universal.md).
