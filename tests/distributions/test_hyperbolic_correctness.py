"""Regression checks for the fixed-parameter hyperbolic EDF criteria."""

import doctest

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import k1

from pysatl_criterion.statistics.alternative import RightAlternative
from pysatl_criterion.statistics.goodness_of_fit.continuous import hyperbolic
from tests.parameter_cases import parameters_for


CLASSES = [
    hyperbolic.KolmogorovSmirnovHyperbolicGofStatistic,
    hyperbolic.CramerVonMisesHyperbolicGofStatistic,
    hyperbolic.AndersonDarlingHyperbolicGofStatistic,
    hyperbolic.KuiperHyperbolicGofStatistic,
    hyperbolic.WatsonHyperbolicGofStatistic,
]


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("sample", [[], [[1, 2]], [np.nan], [np.inf], [-np.inf], [1j], [10**1000]])
def test_invalid_samples(cls, sample):
    with pytest.raises(ValueError):
        cls(parameters_for(cls)).execute_statistic(sample)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("parameter", ["alpha", "beta", "delta", "mu"])
@pytest.mark.parametrize("value", [[1], np.array([1, 2]), 1j, "1", None, True])
def test_parameters_must_be_real_scalars(cls, parameter, value):
    with pytest.raises(ValueError):
        cls(parameters_for(cls, **{parameter: value}))


@pytest.mark.parametrize(
    "params", [{"alpha": 1e308, "delta": 2}, {"alpha": 1e-300, "delta": 1e-300}]
)
def test_unrepresentable_scaled_shape(params):
    with pytest.raises(ValueError, match="Scaled parameters"):
        CLASSES[1](parameters_for(CLASSES[1], **params))


@pytest.mark.parametrize("cls", CLASSES)
def test_independent_calls_and_fixed_hypothesis(cls):
    statistic = cls(parameters_for(cls, alpha=2, beta=0.3, delta=1.5, mu=-0.5))
    sample = np.array([2.0, -1.0, 0.0, 0.0])
    original = sample.copy()
    first = statistic.execute_statistic(sample, unused=True)
    assert isinstance(first, float)
    assert isinstance(statistic.alternative(), RightAlternative)
    statistic.execute_statistic([3.0, 4.0])
    assert statistic.execute_statistic(sample) == first
    np.testing.assert_array_equal(sample, original)
    assert statistic.hypothesis().parameters() == {
        "alpha": 2,
        "beta": 0.3,
        "delta": 1.5,
        "mu": -0.5,
    }


@pytest.mark.parametrize("sample", [[0.0], [0.0, 0.0, 0.0], [-2.0, 0.2, 1.0, 3.0]])
@pytest.mark.parametrize(
    "parameters",
    [(2.0, 0.3, 1.5, -0.5), (1.0, 0.0, 1.0, 0.0), (1.8, -1.2, 0.7, 0.8), (1.0, 0.9, 0.4, -1.0)],
)
def test_against_integrated_classical_density(sample, parameters):
    # Independent classical density, not SciPy's GH parameter conversion.
    alpha, beta, delta, mu = parameters
    gamma = np.sqrt(alpha**2 - beta**2)
    normalizer = gamma / (2 * alpha * delta * k1(delta * gamma))

    def density(x):
        y = x - mu
        return normalizer * np.exp(-alpha * np.hypot(delta, y) + beta * y)

    assert quad(density, -np.inf, np.inf)[0] == pytest.approx(1.0, abs=1e-10)
    x = np.sort(sample)
    u = np.array([quad(density, -np.inf, value, epsabs=1e-12)[0] for value in x])
    sf = np.array([quad(density, value, np.inf, epsabs=1e-12)[0] for value in x])
    n = len(x)
    i = np.arange(1, n + 1)
    plus, minus = max(i / n - u), max(u - (i - 1) / n)
    w = 1 / (12 * n) + sum((u - (i - 0.5) / n) ** 2)
    expected = [
        max(plus, minus),
        w,
        -n - sum((2 * i - 1) / n * (np.log(u) + np.log(sf[::-1]))),
        plus + minus,
        w - n * (np.mean(u) - 0.5) ** 2,
    ]
    for cls, reference in zip(CLASSES, expected, strict=True):
        result = cls(
            parameters_for(cls, alpha=alpha, beta=beta, delta=delta, mu=mu)
        ).execute_statistic(sample)
        assert result == pytest.approx(reference, rel=1e-9, abs=1e-11)


@pytest.mark.parametrize("options", [{"alternative_type": "right"}, {"mode": "auto"}])
def test_ks_uses_shared_constructor(options):
    with pytest.raises(TypeError):
        CLASSES[0](parameters_for(CLASSES[0]), **options)


@pytest.mark.parametrize("cls", [CLASSES[i] for i in [0, 1, 3, 4]])
@pytest.mark.parametrize("values", [[np.nan, 0.5], [-0.1, 0.5], [0.5, 1.1], [0.9, 0.1]])
def test_cdf_numerical_failures_are_explicit(monkeypatch, cls, values):
    monkeypatch.setattr(
        hyperbolic.hyperbolic_statistics.scipy_stats.genhyperbolic, "cdf", lambda *a, **k: values
    )
    with pytest.raises(FloatingPointError, match="CDF"):
        cls(parameters_for(cls)).execute_statistic([0.0, 1.0])


def test_ad_tail_underflow_is_not_mathematical_infinity(monkeypatch):
    monkeypatch.setattr(
        hyperbolic.hyperbolic_statistics.scipy_stats.genhyperbolic,
        "logcdf",
        lambda *a, **k: np.array([-np.inf]),
    )
    with pytest.raises(FloatingPointError, match="underflowed"):
        CLASSES[2](parameters_for(CLASSES[2])).execute_statistic([-1000.0])


def test_ad_does_not_subtract_cdf_from_one():
    # CDF rounds to one here, but direct survival integration is still nonzero.
    result = CLASSES[2](parameters_for(CLASSES[2])).execute_statistic([40.0])
    assert np.isfinite(result) and result > 30


def test_watson_centering_avoids_cancellation():
    n = 10000
    # Equally spaced probabilities translated within (0, 1): all d_i coincide.
    u = (np.arange(n) + 0.5) / n + 0.00004
    result = CLASSES[4](parameters_for(CLASSES[4])).do_execute_statistic(u)
    assert result == pytest.approx(1 / (12 * n), abs=1e-18)


def test_docstring_examples():
    assert (
        doctest.testmod(
            hyperbolic, extraglobs={"Distribution": hyperbolic.hyperbolic_statistics.Distribution}
        ).failed
        == 0
    )


@pytest.mark.parametrize("cls", CLASSES)
def test_masked_observations_rejected(cls):
    with pytest.raises(ValueError):
        cls(parameters_for(cls)).execute_statistic(np.ma.array([0.0, 1.0], mask=[0, 1]))


@pytest.mark.parametrize("cls", CLASSES)
def test_partial_hypothesis_rejected(cls):
    from pysatl_criterion.distribution import HyperbolicDistributionDescriptor

    with pytest.raises(ValueError):
        cls(HyperbolicDistributionDescriptor.DEFAULT.parse({"alpha": 1.5}))


@pytest.mark.parametrize("cls", CLASSES)
def test_exports_pickle_and_explicit_hypotheses(cls):
    import pickle

    assert "supported_hypotheses" in cls.__dict__
    assert getattr(hyperbolic, cls.__name__) is cls
    legacy_global = (
        "cpysatl_criterion.statistics.goodness_of_fit.continuous.hyperbolic\n"
        + cls.__name__
        + "\n."
    ).encode()
    assert pickle.loads(legacy_global) is cls  # noqa: S301 - locally constructed reference
    instance = cls(parameters_for(cls))
    restored = pickle.loads(pickle.dumps(instance))  # noqa: S301 - own roundtrip
    assert type(restored) is cls
    assert restored.execute_statistic([-1.0, 0.0, 1.0]) == pytest.approx(
        instance.execute_statistic([-1.0, 0.0, 1.0])
    )


def test_public_execution_calls_nopython_kernels(monkeypatch):
    from pysatl_criterion.statistics.goodness_of_fit.continuous import edf

    kernels = [
        edf.AbstractKSStatistic,
        edf.AbstractCramerVonMisesStatistic,
        edf.AbstractAndersonDarlingStatistic,
        edf.AbstractKuiperStatistic,
        edf.AbstractWatsonStatistic,
    ]
    for cls, algorithm in zip(CLASSES, kernels, strict=True):
        method = "do_execute_log_statistic" if cls is CLASSES[2] else "do_execute_statistic"
        kernel = getattr(algorithm, method)
        calls = []

        def traced(*args, _kernel=kernel, _calls=calls):
            _calls.append(True)
            return _kernel(*args)

        with monkeypatch.context() as patch:
            patch.setattr(algorithm, method, staticmethod(traced))
            cls(parameters_for(cls)).execute_statistic([-1.0, 0.0, 1.0])
        assert calls == [True]
        assert kernel.nopython_signatures
        assert kernel.targetoptions["nopython"]


def test_family_base_does_not_declare_hypotheses():
    assert "supported_hypotheses" not in hyperbolic.AbstractHyperbolicGofStatistic.__dict__


def test_documentation_examples_and_catalogue():
    import re
    from pathlib import Path

    all_classes = [
        getattr(hyperbolic, name) for name in hyperbolic.__all__ if not name.startswith("Abstract")
    ]
    root = Path(__file__).resolve().parents[2]
    folder = root / "docs/criteria/goodness-of-fit/hyperbolic"
    pages = sorted(folder.glob("*.md"))
    index = (folder / "index.md").read_text()
    navigation = (root / "mkdocs.yml").read_text()
    for cls in all_classes:
        assert cls.__name__ in index
        assert cls.code() in index
    example_count = 0
    for page in pages:
        source = page.read_text()
        assert str(page.relative_to(root / "docs")) in navigation
        for example in re.findall(r"```python\n(.*?)```", source, re.DOTALL):
            exec(compile(example, str(page), "exec"), {})  # noqa: S102 - repository examples
            example_count += 1
        for link in re.findall(r"\]\(([^)]+)\)", source):
            if "://" not in link and not link.startswith("#"):
                assert (page.parent / link.split("#")[0]).exists(), link
    concrete = {name for name in hyperbolic.__all__ if not name.startswith("Abstract")}
    assert concrete == {cls.__name__ for cls in all_classes}
    criterion_pages = [page for page in pages if page.name.endswith("_criterion.md")]
    assert len(criterion_pages) == len(concrete)
    for cls in all_classes:
        matched = [page for page in criterion_pages if cls.__name__ in page.read_text()]
        assert len(matched) == 1
        contents = matched[0].read_text()
        assert "```python" in contents
        assert "https://doi.org/" in contents
    assert example_count >= len(concrete)


@pytest.mark.parametrize("sample", [[0.2], [-1.0, -1.0, 0.5], [-2.0, -0.3, 0.2, 1.1, 3.0]])
def test_public_statistics_against_edf_integrals(sample):
    from scipy import stats

    params = {"alpha": 1.5, "beta": -0.2, "delta": 0.8, "mu": 0.3}
    distribution = stats.genhyperbolic(1, 1.2, -0.16, loc=0.3, scale=0.8)
    u = np.sort(distribution.cdf(sample))
    n = len(u)

    def discrepancy(t):
        return np.count_nonzero(u <= t) / n - t

    def integral(function):
        return quad(function, 0, 1, points=u, epsabs=1e-11)[0]

    center = integral(discrepancy)
    plus = stats.ks_1samp(sample, distribution.cdf, alternative="greater").statistic
    minus = stats.ks_1samp(sample, distribution.cdf, alternative="less").statistic
    expected = [
        max(plus, minus),
        n * integral(lambda t: discrepancy(t) ** 2),
        n * integral(lambda t: discrepancy(t) ** 2 / (t * (1 - t))),
        plus + minus,
        n * integral(lambda t: (discrepancy(t) - center) ** 2),
    ]
    for cls, reference in zip(CLASSES, expected, strict=True):
        value = cls(parameters_for(cls, **params)).execute_statistic(sample)
        assert value == pytest.approx(reference, rel=1e-9, abs=1e-11)
    if n >= 2:  # SciPy requires at least two observations; the EDF integral also covers n=1.
        assert expected[1] == pytest.approx(
            stats.cramervonmises(sample, distribution.cdf).statistic
        )


@pytest.mark.parametrize("cls", CLASSES)
def test_affine_and_reflection_invariance(cls):
    sample = np.array([-2.0, -0.7, 0.1, 0.1, 1.3, 2.0])
    params = {"alpha": 1.7, "beta": 0.4, "delta": 0.9, "mu": -0.2}
    original = cls(parameters_for(cls, **params)).execute_statistic(sample)
    for scale, shift in [(0.25, -3.0), (4.0, 10.0), (-2.0, 0.5)]:
        transformed = {
            "alpha": params["alpha"] / abs(scale),
            "beta": params["beta"] / scale,
            "delta": params["delta"] * abs(scale),
            "mu": scale * params["mu"] + shift,
        }
        value = cls(parameters_for(cls, **transformed)).execute_statistic(scale * sample + shift)
        assert value == pytest.approx(original, rel=1e-9, abs=1e-11)


@pytest.mark.parametrize("value", [-3.0, 0.0, 3.0])
def test_singleton_circular_statistics_are_constant(value):
    for cls, expected in [(CLASSES[3], 1.0), (CLASSES[4], 1 / 12)]:
        assert cls(parameters_for(cls)).execute_statistic([value]) == pytest.approx(expected)
