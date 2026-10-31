"""Independent formula, boundary and documentation checks for every public criterion."""

import inspect
import re
from decimal import Decimal, localcontext
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate, stats

from pysatl_criterion.distribution import InverseGammaDistributionDescriptor as Distribution
from pysatl_criterion.statistics.goodness_of_fit.binning import EqualProbabilityBins
from pysatl_criterion.statistics.goodness_of_fit.continuous import inverse_gamma as module
from pysatl_criterion.statistics.goodness_of_fit.continuous.inverse_gamma import (
    inverse_gamma_statistics as implementation,
)


CLASSES = {
    name: getattr(module, name)
    for name in module.__all__
    if not inspect.isabstract(getattr(module, name))
}
EDF = {
    name: cls for name, cls in CLASSES.items() if name not in implementation._FREQUENCY_STATISTICS
}
DOCS = Path(__file__).resolve().parents[2] / "docs/criteria/goodness-of-fit/inverse-gamma"


def _decimal_references(values):
    """Evaluate printed scalar formulas at 70 digits, independently of EDF kernels."""
    with localcontext() as ctx:
        ctx.prec = 70
        u = sorted(Decimal(str(v)) for v in values)
        n = Decimal(len(u))
        half, quarter = Decimal("0.5"), Decimal("0.25")
        plus = max(Decimal(i) / n - v for i, v in enumerate(u, 1))
        minus = max(v - Decimal(i - 1) / n for i, v in enumerate(u, 1))
        cvm = 1 / (12 * n) + sum((v - (Decimal(i) - half) / n) ** 2 for i, v in enumerate(u, 1))
        liao = (
            sum(
                max(Decimal(i) / n - v, v - Decimal(i - 1) / n) / (v * (1 - v)).sqrt()
                for i, v in enumerate(u, 1)
            )
            / n.sqrt()
        )
        endpoints = [Decimal(0), *u, Decimal(1)]
        result = {
            "KolmogorovSmirnov": max(plus, minus),
            "KolmogorovSmirnovDPlus": plus,
            "KolmogorovSmirnovDMinus": minus,
            "AndersonDarling": -n
            - sum((2 * Decimal(i) - 1) * (v.ln() + (1 - u[-i]).ln()) for i, v in enumerate(u, 1))
            / n,
            "CramerVonMises": cvm,
            "Watson": cvm - n * (sum(u) / n - half) ** 2,
            "Kuiper": plus + minus,
            "KuiperScaled": n.sqrt() * (plus + minus),
            "KuiperCorrected": (n.sqrt() + Decimal(".155") + Decimal(".24") / n.sqrt())
            * (plus + minus),
            "LiaoShimokawa": liao,
            "Greenwood": sum((b - a) ** 2 for a, b in pairwise(endpoints)),
            "ZhangA": -sum(
                v.ln() / (n - Decimal(i) + half) + (1 - v).ln() / (Decimal(i) - half)
                for i, v in enumerate(u, 1)
            ),
            "ZhangC": sum(
                (
                    ((1 - v) / v).ln()
                    - ((n - Decimal(i) + quarter) / (Decimal(i) - 3 * quarter)).ln()
                )
                ** 2
                for i, v in enumerate(u, 1)
            ),
            "ZhangK": max(
                (Decimal(i) - half) * ((Decimal(i) - half) / (n * v)).ln()
                + (n - Decimal(i) + half) * ((n - Decimal(i) + half) / (n * (1 - v))).ln()
                for i, v in enumerate(u, 1)
            ),
        }
        return {key + "InverseGammaGofStatistic": float(value) for key, value in result.items()}


@pytest.mark.parametrize("probabilities", [[0.2], [0.3] * 4, [0.01, 0.19, 0.52, 0.8, 0.98]])
@pytest.mark.parametrize("alpha,beta", [(0.4, 0.25), (1, 2), (3, 7), (12, 0.8)])
def test_all_edf_and_spacing_formulas(probabilities, alpha, beta):
    # Reciprocal Gamma quantiles avoid the inverse-gamma CDF used by production.
    sample = beta / stats.gamma.isf(probabilities, a=alpha)
    references = _decimal_references(probabilities)
    assert references.keys() == EDF.keys()
    parameters = Distribution.DEFAULT.parse({"alpha": alpha, "beta": beta})
    for name, cls in EDF.items():
        assert cls(parameters).execute_statistic(sample[::-1]) == pytest.approx(
            references[name], rel=2e-11, abs=2e-12
        ), name


@pytest.mark.parametrize("probabilities", [[0.2], [0.3] * 4, [0.1, 0.3, 0.7, 0.85]])
def test_integral_definitions(probabilities):
    u = np.sort(probabilities)
    n = len(u)
    cuts = np.r_[0, u, 1]
    cvm = ad = mean_difference = 0.0
    for j, (left, right) in enumerate(pairwise(cuts)):
        if left == right:
            continue
        cvm += n * integrate.quad(lambda t, j=j: (j / n - t) ** 2, left, right)[0]
        ad += n * integrate.quad(lambda t, j=j: (j / n - t) ** 2 / (t * (1 - t)), left, right)[0]
        mean_difference += integrate.quad(lambda t, j=j: j / n - t, left, right)[0]
    parameters = Distribution.DEFAULT.parse({"alpha": 1, "beta": 2})
    sample = -2 / np.log(u)  # Exact inverse of F(x)=exp(-2/x).
    for prefix, expected in [
        ("AndersonDarling", ad),
        ("CramerVonMises", cvm),
        ("Watson", cvm - n * mean_difference**2),
    ]:
        cls = getattr(module, prefix + "InverseGammaGofStatistic")
        assert cls(parameters).execute_statistic(sample) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("alpha,beta", [(0.4, 0.25), (1, 2), (3, 7)])
@pytest.mark.parametrize("counts", [[2, 3, 1, 4], [0, 3, 0, 7], [0, 0, 0, 1]])
def test_frequency_formulas_and_zero_limits(alpha, beta, counts):
    observed = np.asarray(counts, dtype=float)
    expected = observed.sum() / 4
    reference = {
        "Chi2Pearson": sum((observed - expected) ** 2 / expected),
        "FreemanTukey": 4 * sum((np.sqrt(observed) - np.sqrt(expected)) ** 2),
        "ModifiedLogLikelihood": np.inf
        if 0 in counts
        else 2 * sum(expected * np.log(expected / observed)),
        "NeymanChi2": np.inf if 0 in counts else sum((observed - expected) ** 2 / observed),
        "CressieRead": 1.8 * sum(o * ((o / expected) ** (2 / 3) - 1) for o in observed if o > 0),
    }
    u = np.repeat([0.125, 0.375, 0.625, 0.875], counts)
    sample = beta / stats.gamma.isf(u, a=alpha)
    parameters = Distribution.DEFAULT.parse({"alpha": alpha, "beta": beta})
    assert {
        name + "InverseGammaGofStatistic" for name in reference
    } == implementation._FREQUENCY_STATISTICS.keys()
    for prefix, value in reference.items():
        cls = getattr(module, prefix + "InverseGammaGofStatistic")
        assert cls(parameters, binning=EqualProbabilityBins(4)).execute_statistic(
            sample
        ) == pytest.approx(value)


@pytest.mark.parametrize("cls", CLASSES.values())
@pytest.mark.parametrize("fixed", [{}, {"alpha": 2}, {"beta": 3}])
def test_parameters_must_be_fixed(cls, fixed):
    with pytest.raises(ValueError):
        cls(Distribution.DEFAULT.parse(fixed))


def test_greenwood_null_mean_by_quadrature():
    # For n=1, the uniform spacing integral is E[U²+(1-U)²]=2/3.
    cls = module.GreenwoodInverseGammaGofStatistic
    statistic = cls(Distribution.DEFAULT.parse({"alpha": 1, "beta": 2}))
    mean = integrate.quad(lambda u: statistic.execute_statistic([-2 / np.log(u)]), 0, 1)[0]
    assert mean == pytest.approx(2 / 3)


@pytest.mark.parametrize("name", CLASSES)
def test_every_criterion_has_documentation_and_working_example(name):
    pages = [
        path for path in DOCS.glob("*_criterion.md") if f"`{name}` has code" in path.read_text()
    ]
    assert len(pages) == 1, name
    page = pages[0]
    text = page.read_text()
    assert "## Formula" in text and "## Scientific references" in text
    assert "https://doi.org/" in text or "https://ir.cwi.nl/" in text
    assert page.name in (DOCS / "index.md").read_text()
    assert f"inverse-gamma/{page.name}" in (DOCS.parents[3] / "mkdocs.yml").read_text()
    snippets = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
    assert snippets
    for snippet in snippets:
        namespace = {}
        exec(compile(snippet, str(page), "exec"), namespace)  # noqa: S102
        assert not np.isnan(namespace["value"])


@pytest.mark.parametrize(
    "prefix", ["AndersonDarling", "ZhangA", "ZhangC", "ZhangK", "LiaoShimokawa"]
)
def test_logarithmic_statistics_in_rounded_upper_tail(prefix):
    with localcontext() as ctx:
        ctx.prec = 70
        probabilities = [(-Decimal(2) / x).exp() for x in [Decimal(1), Decimal("1e20")]]
    name = prefix + "InverseGammaGofStatistic"
    expected = _decimal_references(probabilities)[name]
    statistic = CLASSES[name](Distribution.DEFAULT.parse({"alpha": 1, "beta": 2}))
    assert statistic.execute_statistic([1.0, 1e20]) == pytest.approx(expected, rel=2e-13)
