# Graph clique number for Gamma

`GraphCliqueNumberGammaGofStatistic`

Use fixed positive shape `alfa` and **rate** `beta` (SciPy scale is `1/beta`), with origin zero. Both parameters must be supplied; no fitting occurs. Samples must be finite, real, nonempty, one-dimensional and nonnegative. Zero is accepted as a support boundary. Ties are allowed unless they make the statistic infinite or degenerate.

## Definition

Vertices are the transformed observations $u_i$. Join distinct vertices exactly when $|u_i-u_j|<h$, where $h=(\max u-\min u)/10$. Let $d_i$ be the degree. Ties remain separate vertices. If h=0 the graph is edgeless: edges, maximum degree and average degree are 0; components and independence number are n; clique number is 1.

$$
\omega(G)=\max\{|S|:S\text{ is a clique}\}
$$

The cited paper provides related graph-based uniformity methodology, not this exact range/10 construction or its calibration. These are library-defined two-tail discrepancies. No claim of consistency against every Gamma alternative is made. The graph is invariant under positive affine changes of the transformed positions, so compressed uniform positions are a blind spot. At fixed n, average degree and edge count contain identical information.

Both tails are used; calibrate the exact graph construction. `execute_statistic` returns a scalar, not a p-value. The sample is not modified.

## Source

[Ebner, Nestmann and Schulte, Testing multivariate uniformity based on random geometric graphs](https://arxiv.org/abs/1812.08638). Background only; the exact implementation is a library adaptation.

## Usage

```python
from pysatl_criterion.distribution import GammaDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit.continuous.gamma import GraphCliqueNumberGammaGofStatistic

parameters = GammaDistributionDescriptor.DEFAULT.parse({"alfa": 2.5, "beta": 1.5})
statistic = GraphCliqueNumberGammaGofStatistic(parameters)
value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
```

See the [Gamma catalog](index.md) and [mathematical verification](mathematical-validation.md)
for calibration, numerical limits and the scope of scientific attribution.
