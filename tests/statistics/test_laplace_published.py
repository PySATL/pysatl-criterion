"""Independent defining-integral, spacing and moment checks for fitted Laplace tests."""

import inspect
import pickle

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import eval_legendre
from scipy.stats import laplace as reference

from pysatl_criterion.distribution import LaplaceDistributionDescriptor as Distribution
from pysatl_criterion.statistics.alternative import AlternativeType
from pysatl_criterion.statistics.goodness_of_fit.continuous.laplace import (
    laplace_statistics as laplace,
)
from pysatl_criterion.utils.generator import get_hypothesis_generator


P = Distribution.DEFAULT.parse({})
X = np.array([-3.2, -1.4, -0.8, -0.1, 0.05, 0.5, 0.9, 1.7, 4.1])
CLASSES = [
    getattr(laplace, name)
    for name in laplace.__all__
    if not name.startswith("Abstract")
    and issubclass(getattr(laplace, name), laplace.AbstractFittedLaplaceGofStatistic)
]


def standardized(x):
    location, scale = reference.fit(x)
    return (np.sort(x) - location) / scale


@pytest.mark.parametrize("cls", CLASSES)
def test_contracts(cls):
    obj = cls(P)
    before = X.copy()
    value = obj.execute_statistic(X)
    assert np.isfinite(value)
    assert obj.execute_statistic(X[::-1]) == pytest.approx(value)
    assert obj.execute_statistic(3 * X + 7) == pytest.approx(value, abs=1e-11)
    assert np.array_equal(before, X)
    restored = pickle.loads(pickle.dumps(obj))  # noqa: S301
    assert restored.execute_statistic(X) == pytest.approx(value)
    assert restored.configuration() == obj.configuration()
    assert obj.hypothesis().parameters() == {}
    assert get_hypothesis_generator(obj) is not None
    assert "http" in inspect.getdoc(cls)
    with pytest.raises(ValueError, match="Unsupported"):
        cls(Distribution.DEFAULT.parse({"t": 0, "s": 1}))


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize(
    "sample",
    [[], [1, 2], [1, 1, 1, 1], [1, np.nan, 2], [1, np.inf, 2], [[1, 2, 3]], [1 + 2j, 2, 3]],
)
def test_invalid(cls, sample):
    with pytest.raises(ValueError):
        cls(P).execute_statistic(sample)


@pytest.mark.parametrize("j", [1, 2])
@pytest.mark.parametrize("a", [0.5, 2, 12])
@pytest.mark.parametrize("estimator", ["mle", "mom"])
def test_meintanis_defining_integral(j, a, estimator):
    z = standardized(X)
    if estimator == "mom":
        z = (z - z.mean()) / np.sqrt(z.var() / 2)

    def integrand(t):
        delta = (1 + t * t) * np.exp(1j * t * z).mean() - 1
        return abs(delta) ** 2 * np.exp(-a * (t if j == 1 else t * t))

    expected = 2 * len(z) * quad(integrand, 0, np.inf, epsabs=1e-10, limit=500)[0]
    obj = getattr(laplace, f"Meintanis{j}LaplaceGofStatistic")(P, a=a, estimator=estimator)
    assert obj.execute_statistic(X) == pytest.approx(expected, rel=2e-8, abs=1e-10)


@pytest.mark.parametrize("m", [1, 2, 4])
@pytest.mark.parametrize("kind", ["V", "C", "E"])
def test_choi_kim(m, kind):
    z = standardized(X)
    n = len(z)

    def at(j):
        return z[min(max(j, 0), n - 1)]

    if kind == "V":
        h = np.mean([np.log(n * (at(i + m) - at(i - m)) / (2 * m)) for i in range(n)])
    elif kind == "E":
        h = np.mean(np.log(np.diff(np.column_stack((z[:-m], z[m:])), axis=1))) + sum(
            1 / k for k in range(m, n + 1)
        )
    else:
        terms = []
        for i in range(n):
            xx = np.array([at(j) for j in range(i - m, i + m + 1)])
            # Slope of ranks on observations, including replicated boundary observations.
            slope = np.polyfit(xx, np.arange(i - m, i + m + 1) / n, 1)[0]
            terms.append(-np.log(slope))
        h = np.mean(terms)
    obj = getattr(laplace, f"ChoiKim{kind}LaplaceGofStatistic")(P, m=m)
    assert obj.execute_statistic(X) == pytest.approx(np.exp(h))
    assert obj.alternative().type() == AlternativeType.LEFT


@pytest.mark.parametrize("kind", ["V", "E", "Y", "A", "Z"])
@pytest.mark.parametrize("m", [1, 2, 4])
def test_ap_corrected_quantile_integral(kind, m):
    z = standardized(X)
    n = len(z)

    def at(j):
        return z[min(max(j - 1, 0), n - 1)]

    xi = np.array([sum(at(k) for k in range(i - m, i + m)) / (2 * m) for i in range(1, n + 2)])
    beta = xi.copy()
    lo = np.maximum(np.arange(n) - m, 0)
    hi = np.minimum(np.arange(n) + m, n - 1)
    spans = z[hi] - z[lo]
    if kind == "Y":
        f = np.empty(n)
        f[0] = (n - 1) / (n * (n + 1)) * (n / (n - 1) + n / (2 * n - 1))
        f[-1] = (n - 1) / (n * (n + 1)) * (((n - 1) ** 2 + 1) / (n - 1) + 1 + (n - 1) / (2 * n - 1))
        for i in range(1, n - 1):
            f[i] = (
                (n - 1)
                / (n * (n + 1))
                * (((i + 1) * (n - 1) + 1) / (n - 1) + (z[i] - z[i - 1]) / (z[i + 1] - z[i - 1]))
            )
        mass = f[hi] - f[lo]
        beta = 2 * m * xi / mass.sum()
        h = np.dot(mass / mass.sum(), np.log(spans / mass))
    else:
        divisors = np.full(n, 2.0)
        for i in range(1, n + 1):
            if kind == "E":
                divisors[i - 1] = 1 + min(i - 1, m, n - i) / m
            if kind == "A" and (i <= m or i >= n - m + 1):
                divisors[i - 1] = 1
            if kind == "Z":
                divisors[i - 1] = i / m if i <= m else ((n - i + 1) / m if i > n - m else 2)
        h = np.log(n * spans / (m * divisors)).mean()
        if kind != "V":
            for i in range(1, m + 1):
                beta[i - 1] = xi[m] - sum(
                    (at(m + k) - at(1)) / (m + k - 1 if kind == "E" else m if kind == "A" else k)
                    for k in range(i, m + 1)
                )
            for i in range(n - m + 2, n + 2):
                beta[i - 1] = xi[n - m] + sum(
                    (at(n) - at(k - m - 1))
                    / (n + m - k + 1 if kind == "E" else m if kind == "A" else n - k + 2)
                    for k in range(n - m + 2, i + 1)
                )
    grid = np.linspace(0, 1, n + 1)
    center = np.interp(0.5, grid, beta)
    theta = quad(
        lambda u: abs(np.interp(u, grid, beta) - center), 0, 1, points=sorted({0.5, *grid})
    )[0]
    cls = getattr(laplace, f"AlizadehNoughabiPark{kind}LaplaceGofStatistic")
    assert cls(P, m=m).execute_statistic(X) == pytest.approx(np.log(2 * theta) + 1 - h)
    assert cls(P, m=m, scale_estimator="mle").execute_statistic(X) == pytest.approx(
        np.log(2) + 1 - h
    )


def test_moments_and_directions():
    n = len(X)
    mu = X.mean()
    s = np.std(X)
    mad = np.mean(abs(X - np.median(X)))
    y = (X - mu) / s
    expected = {
        "BestRaynerThasV3": np.sqrt(n / 63) * np.mean(y**3),
        "BestRaynerThasV4": np.sqrt(n / 1188) * (np.mean(y**4) - 6),
        "Gel": n
        * (
            np.mean(((X - mu) / (np.sqrt(2) * mad)) ** 3) ** 2 / 60
            + (np.mean(((X - mu) / (np.sqrt(2) * mad)) ** 4) - 6) ** 2 / 1200
        ),
        "GonzalezEstradaVillasenor": 2 * np.sqrt(n) * (s / (np.sqrt(2) * np.mean(abs(X - mu))) - 1),
    }
    for name, value in expected.items():
        obj = getattr(laplace, name + "LaplaceGofStatistic")(P)
        assert obj.execute_statistic(X) == pytest.approx(value)
        assert obj.alternative().type() == (
            AlternativeType.RIGHT if name == "Gel" else AlternativeType.TWO_TAILED
        )


@pytest.mark.parametrize("n", [4, 5, 20, 21])
def test_dlo_even_odd(n):
    x = np.random.default_rng(n).normal(size=n)
    z = standardized(x)
    skew = z.mean()
    nonzero = abs(z[z != 0])
    kn = max(0, np.sum(nonzero * np.log(nonzero)) / n - skew**2 / 2)
    c = 1 - np.euler_gamma
    if n % 2:
        s = 1 - 0.281 / n**1.03
        bias = 1 - 0.198 / n**0.86
        var = 1 - 3.827 / n**1.04
    else:
        s = 1 - 1.856 / n**1.06
        bias = 1 - 0.422 / n**1.01
        var = 1 - 1.950 / n**0.92 + 39.349 / n**2.3
    zk = np.sqrt(n) * (kn**0.25 - c**0.25 * bias) / np.sqrt(c**-1.5 * (np.pi**2 / 3 - 3) * var / 16)
    assert laplace.DesgagneLafayeDeMicheauxOuimetZLaplaceGofStatistic(P).execute_statistic(
        x
    ) == pytest.approx(zk)
    assert laplace.DesgagneLafayeDeMicheauxOuimetXLaplaceGofStatistic(P).execute_statistic(
        x
    ) == pytest.approx(n * skew**2 / s + zk**2)


def test_likelihood_product_and_energy():
    z = standardized(X)
    n = len(z)
    products = []
    for m in range(1, int(np.ceil(min(n**0.5, n / 2)))):
        lo = np.maximum(np.arange(n) - m, 0)
        hi = np.minimum(np.arange(n) + m, n - 1)
        products.append(np.prod(2 * m / (n * (z[hi] - z[lo]) * reference.pdf(z))))
    obj = laplace.AlizadehNoughabiLikelihoodLaplaceGofStatistic(P)
    assert obj.execute_statistic(X) == pytest.approx(min(products))
    assert laplace.AlizadehNoughabiLikelihoodLaplaceGofStatistic(
        P, log_scale=True
    ).execute_statistic(X) == pytest.approx(np.log(min(products)))
    for m in [1, 2, 4]:
        lo = np.maximum(np.arange(n) - m, 0)
        hi = np.minimum(np.arange(n) + m, n - 1)
        expected = np.mean(2 * m / (n * (reference.cdf(z[hi]) - reference.cdf(z[lo]))))
        assert laplace.AlizadehNoughabiJarrahiferizLaplaceGofStatistic(P, m=m).execute_statistic(
            X
        ) == pytest.approx(expected)


def test_data_driven_population_covariance_and_selection():
    obj = laplace.DataDrivenSmoothLaplaceGofStatistic(P, max_components=4)
    z = standardized(X)
    phi = np.array(
        [np.sqrt(2 * j + 1) * eval_legendre(j, 2 * reference.cdf(z) - 1) for j in range(1, 5)]
    )
    score = np.array([np.sign(z).mean(), np.mean(abs(z)) - 1])
    projection = np.empty((4, 2))
    for i in range(4):
        for j in range(2):

            def integrand(x, i=i, j=j):
                return (
                    np.sqrt(2 * i + 3)
                    * eval_legendre(i + 1, 2 * reference.cdf(x) - 1)
                    * (np.sign(x) if j == 0 else abs(x) - 1)
                    * reference.pdf(x)
                )

            projection[i, j] = quad(integrand, -np.inf, 0)[0] + quad(integrand, 0, np.inf)[0]
    v = phi.mean(axis=1) - projection @ score
    w = [
        len(z) * v[:k] @ np.linalg.solve(np.eye(k) - projection[:k] @ projection[:k].T, v[:k])
        for k in range(1, 5)
    ]
    selected = np.argmax(np.array(w) - np.arange(1, 5) * np.log(len(z)))
    assert obj.execute_statistic(X) == pytest.approx(w[selected])


class TestLaplaceNumbaOptimization:
    @pytest.mark.parametrize("n", [5, 6, 101, 102])
    @pytest.mark.parametrize("scale", [1e-250, 1.0, 1e250])
    def test_fitted_standardization_extreme_scales(self, n, scale):
        x = np.random.default_rng(n).normal(size=n) * scale
        before = x.copy()
        expected = standardized(x / scale)
        actual = laplace.AbstractFittedLaplaceGofStatistic._fit(x)
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)
        np.testing.assert_array_equal(x, before)
        assert laplace.AbstractFittedLaplaceGofStatistic._fit_sorted.nopython_signatures

    @pytest.mark.parametrize("n", [101, 102])
    @pytest.mark.parametrize("kind", ["v", "e", "y", "a", "z"])
    def test_wide_corrected_window_against_direct_sum(self, n, kind):
        z = standardized(np.random.default_rng(n).laplace(size=n))
        m = n // 2 - 1
        # Literal knot sums independently check every rolling-window endpoint.
        beta = np.array(
            [
                sum(z[min(max(k, 0), n - 1)] / (2 * m) for k in range(i - m, i + m))
                for i in range(n + 1)
            ]
        )
        c = 0.7
        if kind == "y":
            beta *= 2 * m / c
        elif kind != "v":
            for i in range(m):
                beta[i] = beta[m] - sum(
                    (z[m + k - 1] - z[0]) / (m + k - 1 if kind == "e" else m if kind == "a" else k)
                    for k in range(i + 1, m + 1)
                )
            for i in range(n - m + 1, n + 1):
                beta[i] = beta[n - m] + sum(
                    (z[-1] - z[k - m - 2])
                    / (n + m - k + 1 if kind == "e" else m if kind == "a" else n - k + 2)
                    for k in range(n - m + 2, i + 2)
                )
        grid = np.arange(n + 1) / n
        median = np.interp(0.5, grid, beta)
        expected = quad(
            lambda u: abs(np.interp(u, grid, beta) - median),
            0,
            1,
            points=sorted({0.5, *grid}),
            limit=n + 10,
        )[0]
        actual = laplace.AbstractAlizadehNoughabiParkLaplaceGofStatistic._corrected_scale(
            z, m, kind, c
        )
        assert actual == pytest.approx(expected, rel=1e-12)

    def test_meintanis_fallback_compiles_numeric_callback(self):
        for family in [1, 2]:
            cls = getattr(laplace, f"Meintanis{family}LaplaceGofStatistic")
            assert np.isfinite(cls(P, a=12, estimator="mom").execute_statistic(X))
        base = laplace.AbstractMeintanisLaplaceGofStatistic
        assert base._ecf_integrand.nopython_signatures
        assert base._moment_standardize.nopython_signatures
