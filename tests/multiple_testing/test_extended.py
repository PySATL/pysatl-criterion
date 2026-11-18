"""Independent scalar, high-precision and exhaustive closure checks for new methods."""

from decimal import Decimal, localcontext
from itertools import combinations, product
from math import comb

import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests

import pysatl_criterion as public_api
from pysatl_criterion.api import testing as testing_api
from pysatl_criterion.multiple_testing import (
    GLOBAL_TESTS,
    RESAMPLING_PVALUE_PROCEDURES,
    RESAMPLING_STATISTIC_PROCEDURES,
    GlobalTestResult,
    HollandCopenhaver,
    JointResamplingAssumptions,
    PValues,
    ResamplingPValues,
    ResamplingStatistics,
    Rom,
    Simes,
    WestfallYoungMaxT,
    WestfallYoungMinP,
)


ASSUMPTIONS = JointResamplingAssumptions(exchangeable=True, subset_pivotal=True)


@pytest.mark.parametrize(
    "p,expected",
    [
        ([], 1.0),
        ([0.0], 0.0),
        ([1.0], 1.0),
        ([0.04, 0.001, 0.7, 0.02, 0.02], 0.005),
        ([0.04, 0.04, 0.04], 0.04),
        ([0.8, 0.9], 0.9),
    ],
)
def test_simes_global(p, expected):
    result = Simes().test(PValues(np.array(p)))
    assert isinstance(result, GlobalTestResult)
    assert result.p_value == pytest.approx(expected)
    assert result.rejected == (expected <= 0.05)
    assert not hasattr(result, "adjusted_p_values")
    assert not hasattr(Simes(), "adjust")


@pytest.mark.parametrize("value", [-0.1, 1.1, np.nan, np.inf])
def test_global_result_validation(value):
    with pytest.raises(ValueError):
        GlobalTestResult(value, 0.05)
    with pytest.raises(ValueError):
        Simes().combine(np.array([value]))


def test_holland_independent_software_and_logical_bounds():
    rng = np.random.default_rng(8721)
    for m in (1, 3, 10, 50):
        p = rng.uniform(size=m)
        expected = multipletests(p, method="holm-sidak")[1]
        np.testing.assert_allclose(HollandCopenhaver().adjust(p), expected)
    # Three pairwise equality nulls: once one is false, at most one can be true.
    p = np.array([0.8, 0.01, 0.04])
    adjusted = HollandCopenhaver((3, 1, 1)).adjust(p)
    np.testing.assert_allclose(adjusted, [0.8, 0.029701, 0.04])
    assert HollandCopenhaver((3, 1, 1)).test(PValues(p)).rejected.tolist() == [False, True, True]


@pytest.mark.parametrize("bounds", [(0,), (2,), (3, 1, 2), (1, 2, 1), (True,), (1.5,)])
def test_invalid_holland_bounds(bounds):
    with pytest.raises(ValueError):
        HollandCopenhaver(bounds)


def test_holland_bound_shape():
    with pytest.raises(TypeError):
        HollandCopenhaver([3, 2, 1])
    with pytest.raises(ValueError):
        HollandCopenhaver((2, 1)).adjust(np.array([0.1]))
    assert HollandCopenhaver(()).adjust(np.array([])).size == 0


def decimal_rom_thresholds(m, alpha):
    """Independent Decimal/polynomial oracle; no log-binomial or inversion.

    Recurrence documented by Garcia et al. (2010), authors' supplement
    https://sci2s.ugr.es/sicidm and scmamp adjustRom (R/post_hoc.R).
    """
    with localcontext() as context:
        context.prec = 70
        q = Decimal(str(alpha))
        c = [q, q / 2]
        for k in range(3, m + 1):
            a = sum(q**j for j in range(1, k))
            b = sum(Decimal(comb(k, j)) * c[j] ** (k - j) for j in range(1, k - 1))
            c.append((a - b) / k)
        return np.array([float(x) for x in c[:m][::-1]])


@pytest.mark.parametrize("m", [0, 1, 2, 3, 5, 10, 30])
@pytest.mark.parametrize("alpha", [0.001, 0.05, 0.2, 0.8])
def test_rom_critical_values_independent_precision(m, alpha):
    np.testing.assert_allclose(
        Rom().critical_values(m, alpha), decimal_rom_thresholds(m, alpha), rtol=2e-13
    )


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.2, 0.8])
def test_rom_decisions_match_independent_step_up(alpha):
    rng = np.random.default_rng(1221)
    for m in (1, 2, 3, 7):
        critical = decimal_rom_thresholds(m, alpha)
        for _ in range(8):
            p = rng.uniform(0, min(1, 2 * alpha), size=m)
            order = np.argsort(p)
            passes = np.flatnonzero(p[order] <= critical)
            expected = np.zeros(m, dtype=bool)
            if passes.size:
                expected[order[: passes[-1] + 1]] = True
            np.testing.assert_array_equal(Rom().test(PValues(p), alpha).rejected, expected)


def test_rom_analytic_three_hypothesis_adjustment():
    p = np.array([0.8, 0.01, 0.03])
    # c3(q) = q/3 + q**2/12; invert analytically with stable quadratic root.
    expected = [0.8, 6 * 0.01 / (np.sqrt(1 + 3 * 0.01) + 1), 0.06]
    np.testing.assert_allclose(Rom().adjust(p), expected, rtol=1e-14)
    np.testing.assert_allclose(Rom().critical_values(3), [0.016875, 0.025, 0.05])


@pytest.mark.parametrize("m", [-1, 1.5, True])
def test_rom_invalid_size(m):
    with pytest.raises(ValueError):
        Rom().critical_values(m)


def closure_oracle(observed, resampled, monte_carlo):
    """Enumerate EVERY intersection, then enforce closed testing for each null."""
    result = np.zeros(len(observed))
    correction = int(monte_carlo)
    for size in range(1, len(observed) + 1):
        for subset in combinations(range(len(observed)), size):
            indices = list(subset)
            local = (
                sum(max(row[indices]) >= max(observed[indices]) for row in resampled) + correction
            ) / (len(resampled) + correction)
            result[indices] = np.maximum(result[indices], local)
    return result


@pytest.mark.parametrize("step_down,expected", [(True, [0.6, 0.4, 0.6]), (False, [1.0, 0.4, 1.0])])
@pytest.mark.parametrize("sampling", ["monte_carlo", "exact"])
@pytest.mark.parametrize("method", [WestfallYoungMaxT, WestfallYoungMinP])
def test_westfall_young_hand_counts(method, step_down, expected, sampling):
    observed = np.array([3.0, 4.0, 2.0])
    draws = np.array([[3.0, 1.0, 0.0], [0.0, 5.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 1.0]])
    if sampling == "exact":
        draws = np.vstack((draws, observed))
    if method is WestfallYoungMinP:
        data = ResamplingPValues((6 - observed) / 6, (6 - draws) / 6)
    else:
        data = ResamplingStatistics(observed, draws)
    result = method(ASSUMPTIONS, step_down=step_down, sampling=sampling).test(data, 0.5)
    np.testing.assert_allclose(result.adjusted_p_values, expected)
    np.testing.assert_array_equal(result.rejected, np.array(expected) <= 0.5)
    assert not result.alpha_dependent


@pytest.mark.parametrize("tail", ["greater", "less", "two-sided"])
@pytest.mark.parametrize("sampling", ["monte_carlo", "exact"])
def test_max_t_exhaustive_closure_and_column_order(tail, sampling):
    rng = np.random.default_rng(54391)
    observed = np.array([1.0, -2.0, 1.0, 0.2])
    draws = rng.normal(size=(29, 4))
    if sampling == "exact":
        draws = np.vstack((draws, observed))
    transform = {"greater": lambda x: x, "less": np.negative, "two-sided": np.abs}[tail]
    expected = closure_oracle(transform(observed), transform(draws), sampling == "monte_carlo")
    method = WestfallYoungMaxT(ASSUMPTIONS, tail=tail, sampling=sampling)
    for order in ([0, 1, 2, 3], [3, 2, 0, 1]):
        data = ResamplingStatistics(observed[order], draws[:, order])
        np.testing.assert_allclose(method.test(data).adjusted_p_values, expected[order])


def test_min_p_exact_sign_flip_orbit():
    # A real joint randomization scheme: all 2**4 sign flips, shared across columns.
    values = np.array([[1.0, 2.0, -1.0], [2.0, 1.0, 3.0], [0.5, 2.0, 1.0], [3.0, 1.0, 2.0]])
    orbit = np.array(
        [np.mean(values * np.array(sign)[:, None], axis=0) for sign in product([-1, 1], repeat=4)]
    )
    p_orbit = np.mean(orbit[:, None, :] >= orbit[None, :, :], axis=0)
    for row in range(len(orbit)):
        data = ResamplingPValues(p_orbit[row], p_orbit)
        expected = closure_oracle(-p_orbit[row], -p_orbit, False)
        actual = WestfallYoungMinP(ASSUMPTIONS, sampling="exact").test(data)
        np.testing.assert_allclose(actual.adjusted_p_values, expected)


@pytest.mark.parametrize("method", [WestfallYoungMaxT, WestfallYoungMinP])
def test_joint_sign_flip_fwer_global_and_partial_null(method):
    # Exhaustive probability over 32 equally likely null signs, with a fixed
    # false-null column. The null subvector's orbit is unaffected by that column.
    values = np.array([[1.0, 2.0], [2.0, 1.0], [0.5, 2.0], [3.0, 1.0], [1.0, 0.5]])
    signs = np.array(list(product([-1, 1], repeat=5)))
    for false_null in (False, True):
        errors = 0
        for sign in signs:
            sample = values * sign[:, None]
            if false_null:
                sample = np.column_stack((sample, np.full(5, 100.0)))
            orbit = np.mean(signs[:, :, None] * sample[None, :, :], axis=1)
            observed = np.mean(sample, axis=0)
            if method is WestfallYoungMaxT:
                data = ResamplingStatistics(observed, orbit)
            else:
                p_orbit = np.mean(orbit[:, None, :] >= orbit[None, :, :], axis=0)
                p_obs = np.mean(orbit >= observed, axis=0)
                data = ResamplingPValues(p_obs, p_orbit)
            result = method(ASSUMPTIONS, sampling="exact").test(data, alpha=0.1)
            errors += bool(np.any(result.rejected[:2]))
        assert errors / len(signs) <= 0.1


@pytest.mark.parametrize(
    "method,container",
    [(WestfallYoungMaxT, ResamplingStatistics), (WestfallYoungMinP, ResamplingPValues)],
)
def test_resampling_boundaries_and_validation(method, container):
    empty = container(np.empty(0), np.empty((1, 0)))
    assert method(ASSUMPTIONS).test(empty).rejected.size == 0
    observed, draws = (np.array([1.0]), np.array([[0.0]]))
    if method is WestfallYoungMinP:
        observed, draws = 1 - observed, 1 - draws
    data = container(observed, draws)
    np.testing.assert_allclose(method(ASSUMPTIONS).test(data).adjusted_p_values, [0.5])
    with pytest.raises(ValueError, match="identity"):
        method(ASSUMPTIONS, sampling="exact").test(data)
    for kwargs, error in [
        ({"assumptions": None}, TypeError),
        ({"step_down": 1}, TypeError),
        ({"sampling": "bad"}, ValueError),
    ]:
        with pytest.raises(error):
            method(**({"assumptions": ASSUMPTIONS} | kwargs))
    with pytest.raises(TypeError):
        method()


@pytest.mark.parametrize("exchangeable,pivotal", [(False, True), (True, False), (1, True)])
def test_resampling_assumptions_required(exchangeable, pivotal):
    with pytest.raises(ValueError):
        JointResamplingAssumptions(exchangeable, pivotal)


def test_invalid_tail():
    with pytest.raises(ValueError):
        WestfallYoungMaxT(ASSUMPTIONS, tail="invalid")


def test_new_public_exports_and_registries():
    for cls in (
        Simes,
        Rom,
        HollandCopenhaver,
        WestfallYoungMaxT,
        WestfallYoungMinP,
        JointResamplingAssumptions,
        GlobalTestResult,
    ):
        assert getattr(public_api, cls.__name__) is cls
        assert getattr(testing_api, cls.__name__) is cls
    for registry, cls in (
        (GLOBAL_TESTS, Simes),
        (RESAMPLING_STATISTIC_PROCEDURES, WestfallYoungMaxT),
        (RESAMPLING_PVALUE_PROCEDURES, WestfallYoungMinP),
    ):
        assert registry[cls.short_code()] is cls
        with pytest.raises(TypeError):
            registry["other"] = cls
