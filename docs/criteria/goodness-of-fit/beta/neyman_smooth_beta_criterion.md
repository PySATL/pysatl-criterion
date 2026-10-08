# Neyman smooth test for the Beta distribution

## Description

Tests the simple hypothesis of iid observations from $\mathrm{Beta}(a,b)$
on $[0,1]$, with specified $a,b>0$. Both shapes must be fixed independently
of the observations. This is Neyman's uniform smooth test after the probability
integral transform, not a new Beta-specific polynomial construction.

## Usage

```python
from pysatl_criterion.distribution.distributions import BetaDistributionDescriptor as Beta
from pysatl_criterion.statistics.goodness_of_fit import NeymanSmoothBetaGofStatistic

statistic = NeymanSmoothBetaGofStatistic(Beta.DEFAULT.parse({"a": 2, "b": 5}), k=4)
value = statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
```

## Arguments and statistic

`parameters` specifies both positive finite shapes; partial hypotheses are
rejected. `k` is a positive integer (default 4), chosen before examining the
sample. `rvs` is a nonempty one-dimensional finite real sample in $[0,1]$;
masked observations are rejected. A single observation and endpoints are valid.
Invalid inputs raise `ValueError` (invalid parameter types may raise `TypeError`).
The result is a scalar; larger values are significant. Identifiers are `NEYMAN`
and `NEYMAN_BETA_GOODNESS_OF_FIT`.

Let $n$ be sample size, $u_i=F_{a,b}(x_i)$, and $P_j$ the degree-$j$
Legendre polynomial, normalized by $P_j(1)=1$. Then

$$
N_k=\sum_{j=1}^{k}\left[\frac1{\sqrt n}\sum_{i=1}^{n}
\sqrt{2j+1}\,P_j(2u_i-1)\right]^2.
$$

## Calibration and limitations

For fixed $k$, the null limit is $\chi^2_k$. The finite-sample distribution
depends on $n,k$ but not on $a,b$, because the $u_i$ are iid uniform under
the simple null. Monte Carlo from independent uniforms gives finite-sample
critical values; the chi-square limit can be inaccurate for small $n$.
For fixed $k$ the test is consistent against alternatives with a nonzero
selected component, but **not omnibus**: other distributions can match all
$k$ component means. No adaptive order selection is implemented.

Plugging sample estimates into this class does not produce a calibrated
composite test. Such a procedure requires refitting in every bootstrap sample
and must not use the stated simple-null chi-square or uniform calibration.

## Implementation

[NeymanSmoothBetaGofStatistic](https://github.com/PySATL/pysatl-criterion/blob/main/src/pysatl_criterion/statistics/goodness_of_fit/beta.py)
delegates to `NeymanSmoothTestUniformGofStatistic`; it does not duplicate the
Legendre algorithm. Unit references integrate the Beta density and use explicit
polynomials independently of the implementation.

## References

Neyman, J. (1937). *Smooth test for goodness of fit.* Skandinavisk
Aktuarietidskrift, 20, 149–199.
DOI: [10.1080/03461238.1937.10404821](https://doi.org/10.1080/03461238.1937.10404821).

## Stored calibration

Storage keys omit algorithm options. Stored calibration is allowed only for
`k=4`. Other settings raise `ValueError` before storage lookup; use
`MonteCarloLimitDistributionResolver` instead. Existing stored results generated
with other options under the same key must be regenerated.
