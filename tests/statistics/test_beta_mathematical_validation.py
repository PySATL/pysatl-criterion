"""Independent Beta EDF references and numerical audit regressions."""

import numpy as np
import pytest
from scipy import integrate, stats

from pysatl_criterion.distribution import BetaDistributionDescriptor as Beta
from pysatl_criterion.statistics.goodness_of_fit.continuous import beta
from pysatl_criterion.statistics.goodness_of_fit.continuous.beta import beta_statistics


@pytest.mark.parametrize("a,b", [(0.3, 0.7), (1, 1), (2, 5), (8, 2), (80, 120)])
def test_all_edf_against_independent_formulas(a, b):
    # Quantiles chosen away from unrepresentable endpoints for this shape grid.
    x = stats.beta.ppf([0.08, 0.19, 0.32, 0.51, 0.68, 0.87], a, b)
    u = np.array([integrate.quad(lambda t: stats.beta.pdf(t, a, b), 0, z)[0] for z in x])
    n = len(x)
    i = np.arange(1, n + 1)
    dp = stats.kstest(x, stats.beta(a, b).cdf, alternative="greater").statistic
    dm = stats.kstest(x, stats.beta(a, b).cdf, alternative="less").statistic

    def discrepancy(t):
        return np.count_nonzero(u <= t) / n - t

    def integral(f):
        return integrate.quad(f, 0, 1, points=u, epsabs=1e-10)[0]

    mean = integral(discrepancy)
    r = i - 0.5
    references = {
        "KolmogorovSmirnovDPlus": dp,
        "KolmogorovSmirnovDMinus": dm,
        "KolmogorovSmirnov": max(dp, dm),
        "AndersonDarling": n * integral(lambda t: discrepancy(t) ** 2 / (t * (1 - t))),
        "CrammerVonMises": n * integral(lambda t: discrepancy(t) ** 2),
        "Watson": n * integral(lambda t: (discrepancy(t) - mean) ** 2),
        "Kuiper": dp + dm,
        "KuiperScaled": np.sqrt(n) * (dp + dm),
        "KuiperCorrected": (np.sqrt(n) + 0.155 + 0.24 / np.sqrt(n)) * (dp + dm),
        "LiaoShimokawa": sum(
            max(j / n - v, v - (j - 1) / n) / np.sqrt(v * (1 - v))
            for j, v in zip(i, u, strict=True)
        )
        / np.sqrt(n),
        "ZhangK": max(r * np.log(r / (n * u)) + (n - r) * np.log((n - r) / (n * (1 - u)))),
        "ZhangA": -sum(np.log(u) / (n - r) + np.log1p(-u) / r),
        "ZhangC": sum((np.log((1 - u) / u) - np.log((n - i + 0.25) / (i - 0.75))) ** 2),
    }
    assert {name + "BetaGofStatistic" for name in references} == set(
        beta_statistics._EDF_STATISTICS
    )
    for prefix, reference in references.items():
        criterion = getattr(beta, prefix + "BetaGofStatistic")(Beta.DEFAULT.parse({"a": a, "b": b}))
        assert criterion.execute_statistic(x[::-1]) == pytest.approx(reference, rel=2e-8, abs=1e-9)
