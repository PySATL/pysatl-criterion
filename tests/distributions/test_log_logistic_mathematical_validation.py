"""Independent integrals, multinomial references, tails and documentation coverage."""

import re
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate, stats
from scipy.special import expit

from pysatl_criterion.distribution import LogLogisticDistributionDescriptor as Distribution
from pysatl_criterion.statistics.goodness_of_fit.binning import EqualProbabilityBins, ExplicitBins
from pysatl_criterion.statistics.goodness_of_fit.continuous.log_logistic import (
    log_logistic_statistics as module,
)


@pytest.mark.parametrize("alpha,beta", [(1, 1), (3, 2.5), (1e-80, 0.3), (1e80, 10)])
@pytest.mark.parametrize(
    "probabilities", [[0.08, 0.19, 0.32, 0.51, 0.68, 0.87], [0.2, 0.2, 0.7], [0.4]]
)
def test_all_edf_against_independent_references(alpha, beta, probabilities):
    x = stats.fisk.ppf(probabilities, beta, scale=alpha)
    null = stats.fisk(beta, scale=alpha)
    u = null.cdf(x)
    n = len(x)
    i = np.arange(1, n + 1)
    dp = stats.kstest(x, null.cdf, alternative="greater").statistic
    dm = stats.kstest(x, null.cdf, alternative="less").statistic

    def discrepancy(t):
        return np.count_nonzero(u <= t) / n - t

    def integral(f):
        return integrate.quad(f, 0, 1, points=u, epsabs=1e-11)[0]

    mean = integral(discrepancy)
    r = i - 0.5
    references = {
        "KolmogorovSmirnovDPlus": dp,
        "KolmogorovSmirnovDMinus": dm,
        "KolmogorovSmirnov": max(dp, dm),
        "AndersonDarling": n * integral(lambda t: discrepancy(t) ** 2 / (t * (1 - t))),
        "CramerVonMises": n * integral(lambda t: discrepancy(t) ** 2),
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
    assert {name + "LogLogisticGofStatistic" for name in references} == set(module._EDF_STATISTICS)
    for prefix, expected in references.items():
        criterion = getattr(module, prefix + "LogLogisticGofStatistic")(
            Distribution.DEFAULT.parse({"alpha": alpha, "beta": beta})
        )
        assert criterion.execute_statistic(x[::-1]) == pytest.approx(expected, rel=2e-10, abs=1e-10)


_FREQUENCY = [
    ("Chi2Pearson", 1),
    ("FreemanTukey", -0.5),
    ("ModifiedLogLikelihood", -1),
    ("NeymanChi2", -2),
    ("CressieRead", 2 / 3),
]


@pytest.mark.parametrize("prefix,lambda_", _FREQUENCY)
@pytest.mark.parametrize("explicit", [False, True])
def test_frequency_against_scipy_power_divergence(prefix, lambda_, explicit):
    alpha, beta = 2, 1.5
    probabilities = np.array([0.02, 0.07, 0.11, 0.24, 0.35, 0.43, 0.62, 0.79, 0.88, 0.96])
    edges = np.array([0, 0.15, 0.5, 0.85, 1]) if explicit else np.linspace(0, 1, 5)
    observed = np.histogram(probabilities, edges)[0]
    expected = len(probabilities) * np.diff(edges)
    x = stats.fisk.ppf(probabilities, beta, scale=alpha)
    binning = (
        ExplicitBins(tuple(stats.fisk.ppf(edges, beta, scale=alpha)))
        if explicit
        else EqualProbabilityBins(4)
    )
    criterion = getattr(module, prefix + "LogLogisticGofStatistic")(
        Distribution.DEFAULT.parse({"alpha": alpha, "beta": beta}), binning=binning
    )
    assert criterion.execute_statistic(x) == pytest.approx(
        stats.power_divergence(observed, expected, lambda_=lambda_).statistic
    )


@pytest.mark.parametrize(
    "prefix,expected",
    [
        ("Chi2Pearson", 6),
        ("FreemanTukey", 8),
        ("ModifiedLogLikelihood", np.inf),
        ("NeymanChi2", np.inf),
        ("CressieRead", 3.6 * (4 ** (2 / 3) - 1)),
    ],
)
def test_empty_cells(prefix, expected):
    criterion = getattr(module, prefix + "LogLogisticGofStatistic")(
        Distribution.DEFAULT.parse({"alpha": 1, "beta": 1}), binning=EqualProbabilityBins(4)
    )
    assert criterion.execute_statistic([0.1, 0.2]) == pytest.approx(expected)


@pytest.mark.parametrize("alpha", [1e-250, 1e250])
@pytest.mark.parametrize("prefix,lambda_", _FREQUENCY)
def test_frequency_extreme_scale_uses_correct_cells(alpha, prefix, lambda_):
    # x/alpha overflows or underflows; the true probabilities remain in the interior.
    x = np.array([1e40, 1e60, 1e100, 1e250])
    if alpha > 1:
        x = 1 / x
    u = expit(0.001 * (np.log(x) - np.log(alpha)))
    counts = np.histogram(u, np.linspace(0, 1, 11))[0]
    expected = np.full(10, 0.4)
    if lambda_ in (-1, -2):
        reference = np.inf
    elif lambda_ == -0.5:
        reference = 4 * sum((np.sqrt(counts) - np.sqrt(expected)) ** 2)
    else:
        reference = stats.power_divergence(counts, expected, lambda_=lambda_).statistic
    criterion = getattr(module, prefix + "LogLogisticGofStatistic")(
        Distribution.DEFAULT.parse({"alpha": alpha, "beta": 0.001}),
        binning=EqualProbabilityBins(10),
    )
    assert criterion.execute_statistic(x) == pytest.approx(reference)


@pytest.mark.parametrize(
    "prefix", ["AndersonDarling", "LiaoShimokawa", "ZhangK", "ZhangA", "ZhangC"]
)
def test_extreme_tail_against_decimal(prefix):
    sample = [1e-100, 1.0, 1e100]
    with localcontext() as context:
        context.prec = 250
        n = Decimal(3)
        u = [Decimal.from_float(x) ** 2 / (1 + Decimal.from_float(x) ** 2) for x in sample]
        lc, ls = [v.ln() for v in u], [(1 - v).ln() for v in u]
        ranks = [Decimal(i) - Decimal("0.5") for i in range(1, 4)]
        references = {
            "AndersonDarling": -n
            - sum((2 * Decimal(i) - 1) * (lc[i - 1] + ls[-i]) for i in range(1, 4)) / n,
            "LiaoShimokawa": sum(
                max(Decimal(i) / n - v, v - Decimal(i - 1) / n) / (v * (1 - v)).sqrt()
                for i, v in enumerate(u, 1)
            )
            / n.sqrt(),
            "ZhangK": max(
                r * ((r / n).ln() - a) + (n - r) * (((n - r) / n).ln() - b)
                for r, a, b in zip(ranks, lc, ls, strict=True)
            ),
            "ZhangA": -sum(a / (n - r) + b / r for r, a, b in zip(ranks, lc, ls, strict=True)),
            "ZhangC": sum(
                (b - a - ((n - Decimal(i) + Decimal(".25")) / (Decimal(i) - Decimal(".75"))).ln())
                ** 2
                for i, (a, b) in enumerate(zip(lc, ls, strict=True), 1)
            ),
        }
    criterion = getattr(module, prefix + "LogLogisticGofStatistic")(
        Distribution.DEFAULT.parse({"alpha": 1, "beta": 2})
    )
    assert criterion.execute_statistic(sample) == pytest.approx(
        float(references[prefix]), rel=1e-12
    )


@pytest.mark.parametrize("method", ["cdf", "sf", "logcdf", "logsf"])
def test_bound_probabilities_support_and_scipy_agreement(method):
    bound = Distribution.bind(Distribution.DEFAULT.parse({"alpha": 3, "beta": 2}))
    x = np.array([-np.inf, -1, 0, 0.1, 1, 5, np.inf, np.nan])
    np.testing.assert_allclose(
        getattr(bound, method)(x), getattr(stats.fisk(2, scale=3), method)(x), equal_nan=True
    )
    assert isinstance(getattr(bound, method)(1.0), float)


def test_bound_probabilities_preserve_extreme_logs():
    bound = Distribution.bind(Distribution.DEFAULT.parse({"alpha": 1, "beta": 2}))
    assert bound.logcdf(1e-250) == pytest.approx(2 * np.log(1e-250))
    assert bound.logsf(1e250) == pytest.approx(-2 * np.log(1e250))


def test_all_criteria_documented_and_examples_execute():
    root = Path(__file__).resolve().parents[2]
    folder = root / "docs/criteria/goodness-of-fit/log-logistic"
    index = (folder / "index.md").read_text()
    nav = (root / "mkdocs.yml").read_text()
    for name in module.__all__:
        if name.startswith("Abstract"):
            continue
        pages = [p for p in folder.glob("*.md") if p.read_text().startswith("# " + name + "\n")]
        assert len(pages) == 1, name
        page = pages[0]
        content = page.read_text()
        assert page.name in index
        assert "log-logistic/" + page.name in nav
        assert "https://" in content
        examples = re.findall(r"```python\n(.*?)```", content, re.DOTALL)
        assert examples, name
        for example in examples:
            exec(compile(example, str(page), "exec"), {})  # noqa: S102 - local documentation examples
