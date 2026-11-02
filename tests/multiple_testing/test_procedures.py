"""Independent reference values and properties of all ten procedures."""

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from pysatl_criterion.multiple_testing import (
    P_VALUE_PROCEDURES,
    AbstractPValueAdjustment,
    BenjaminiKriegerYekutieli,
    Bonferroni,
    Holm,
    Hommel,
    MultipleTestingResult,
    PValues,
    ResamplingPValues,
    ResamplingStatistics,
    Sidak,
    TwoStageBenjaminiHochberg,
)


REFERENCE = json.loads((Path(__file__).parent / "fixtures/statsmodels.json").read_text())
METHODS = tuple(P_VALUE_PROCEDURES.values())


@pytest.mark.parametrize("record", REFERENCE["records"])
def test_independent_fixtures(record):
    """Generated ONLY by statsmodels 0.14.6; generator records provenance."""
    procedure = P_VALUE_PROCEDURES[record["method"]]()
    values = np.array(record["p"])
    result = procedure.test(PValues(values), record["alpha"])
    np.testing.assert_allclose(result.adjusted_p_values, record["adjusted"], atol=1e-15)
    np.testing.assert_array_equal(result.rejected, record["rejected"])
    np.testing.assert_array_equal(result.rejected, result.adjusted_p_values <= result.alpha)


@pytest.mark.parametrize(
    "method", [m for m in METHODS if m.short_code() not in ("rom", "holland-copenhaver")]
)
def test_live_reference(method):
    """Cross-library check over seeded random families, beyond committed fixtures."""
    multitest = pytest.importorskip("statsmodels.stats.multitest")
    rng = np.random.default_rng(2348)
    for m in (1, 2, 7, 30):
        for alpha in (0.001, 0.05, 0.3):
            p = rng.uniform(size=m) ** 3
            expected_r, expected_p, _, _ = multitest.multipletests(
                p,
                alpha=alpha,
                method=method.short_code(),
                maxiter=1,
            )
            result = method().test(PValues(p), alpha)
            np.testing.assert_allclose(result.adjusted_p_values, expected_p, atol=1e-15)
            np.testing.assert_array_equal(result.rejected, expected_r)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("p", [[], [0], [1], [0.1] * 8, [0.9, 0.0, 1.0, 0.02, 0.02]])
def test_order_bounds_mutation_and_empty(method, p):
    values = np.array(p, dtype=float)
    original = values.copy()
    data = PValues(values)
    result = method().test(data)
    np.testing.assert_array_equal(values, original)
    np.testing.assert_array_equal(data.values, original)
    assert result.rejected.dtype == np.bool_
    assert result.rejected.shape == values.shape
    assert np.all((0 <= result.adjusted_p_values) & (result.adjusted_p_values <= 1))
    assert np.all(np.diff(result.adjusted_p_values[np.argsort(values)]) >= -1e-15)
    reverse = method().test(PValues(values[::-1]))
    np.testing.assert_allclose(reverse.adjusted_p_values[::-1], result.adjusted_p_values)
    assert not result.adjusted_p_values.flags.writeable
    assert not result.rejected.flags.writeable
    if isinstance(method(), AbstractPValueAdjustment):
        np.testing.assert_array_equal(method().adjust(values), result.adjusted_p_values)


@pytest.mark.parametrize(
    "bad", [[-0.1], [1.1], [np.nan], [np.inf], [[0.1]], 0.2, [1j], ["0.1"], [True]]
)
def test_invalid_inputs(bad):
    for method in METHODS:
        with pytest.raises(ValueError):
            method().test(PValues(np.asarray(bad)))
        if isinstance(method(), AbstractPValueAdjustment):
            with pytest.raises(ValueError):
                method().adjust(np.asarray(bad))


@pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.1, np.nan, np.inf, [0.05], 1j, "0.05"])
@pytest.mark.parametrize("method", METHODS)
def test_invalid_alpha(method, alpha):
    with pytest.raises((ValueError, TypeError)):
        method().test(PValues(np.array([0.1])), alpha)


def test_copies_boundary_and_tiny_p():
    p = np.array([0.025, 0.9])
    data = PValues(p)
    p[0] = 0.9
    assert data.values[0] == 0.025
    assert Bonferroni().test(data).rejected[0]  # <=, not legacy <
    with pytest.raises(ValueError):
        data.values[0] = 0.5
    tiny = Sidak().adjust(np.array([1e-30, 1.0]))
    assert tiny[0] == pytest.approx(2e-30, rel=1e-14, abs=0)
    assert tiny[1] == 1


@pytest.mark.parametrize("method", [TwoStageBenjaminiHochberg, BenjaminiKriegerYekutieli])
def test_adaptive_reuse_and_alpha_dependence(method):
    data = PValues(np.array([0.001, 0.02, 0.06, 0.2, 0.8]))
    low = method().test(data, 0.01)
    high = method().test(data, 0.1)
    assert not np.array_equal(low.adjusted_p_values, high.adjusted_p_values)
    with pytest.raises(ValueError, match="again"):
        low.decisions_at(0.1)
    np.testing.assert_array_equal(low.decisions_at(0.01), low.rejected)
    assert not hasattr(method(), "adjust")
    assert low.alpha_dependent


def test_hommel_against_exhaustive_closed_simes():
    """Independent mathematical oracle: enumerate all intersection hypotheses.

    Hommel (1988): adjusted p_i is the maximum local Simes value over
    intersections containing i. This deliberately avoids the production shortcut.
    """
    rng = np.random.default_rng(149)
    for p in [np.array([0.01, 0.03, 0.04]), rng.uniform(size=7), np.array([0, 1, 0.2, 0.2])]:
        expected = np.zeros(p.size)
        for size in range(1, p.size + 1):
            for subset in itertools.combinations(range(p.size), size):
                local = min(1, min(size * np.sort(p[list(subset)]) / np.arange(1, size + 1)))
                for index in subset:
                    expected[index] = max(expected[index], local)
        np.testing.assert_allclose(Hommel().adjust(p), expected)


def test_independent_two_stage_decisions():
    """BKY authors' Definition 6: direct critical-value step-up at each stage."""

    def step_up(p, level):
        order = np.argsort(p)
        hits = np.flatnonzero(p[order] <= level * np.arange(1, p.size + 1) / p.size)
        rejected = np.zeros(p.size, dtype=bool)
        if hits.size:
            rejected[order[: hits[-1] + 1]] = True
        return rejected

    rng = np.random.default_rng(936)
    for _ in range(100):
        p = rng.uniform(size=9) ** 3
        alpha = 0.07
        first = step_up(p, alpha / (1 + alpha))
        r1 = np.count_nonzero(first)
        expected = (
            first if r1 in (0, p.size) else step_up(p, alpha / (1 + alpha) * p.size / (p.size - r1))
        )
        np.testing.assert_array_equal(
            BenjaminiKriegerYekutieli().test(PValues(p), alpha).rejected, expected
        )


def test_result_validation_and_recomparison():
    with pytest.raises(ValueError):
        MultipleTestingResult(np.array([True]), 0.05, np.array([0.8]))
    with pytest.raises(ValueError):
        MultipleTestingResult(np.array([True]), 0.05, np.array([]))
    with pytest.raises(ValueError):
        MultipleTestingResult(np.array([1]), 0.05)
    with pytest.raises(ValueError):
        MultipleTestingResult(np.array([False]), 0.05).decisions_at(0.1)
    result = Holm().test(PValues(np.array([0.01, 0.1])))
    np.testing.assert_array_equal(result.decisions_at(0.2), [True, True])


@pytest.mark.parametrize("container", [ResamplingStatistics, ResamplingPValues])
def test_joint_containers(container):
    observed = np.array([0.1, 0.2])
    resampled = np.array([[0.4, 0.3], [0.2, 0.1]])
    data = container(observed, resampled)
    observed[:] = 0
    resampled[:] = 0
    assert data.observed[0] == 0.1
    assert data.resampled[0, 0] == 0.4
    assert not data.resampled.flags.writeable
    for invalid in (np.empty((0, 2)), np.ones((3, 1)), np.ones(2), np.full((2, 2), np.nan)):
        with pytest.raises(ValueError):
            container(data.observed, invalid)
    with pytest.raises(ValueError):
        container(np.ones((1, 2)), np.ones((2, 2)))


def test_resampling_input_semantics_and_discovery():
    ResamplingStatistics(np.array([-3.0]), np.array([[4.0]]))
    with pytest.raises(ValueError):
        ResamplingPValues(np.array([-3.0]), np.array([[4.0]]))
    assert len(P_VALUE_PROCEDURES) == 12
    for code, method in P_VALUE_PROCEDURES.items():
        assert code == method.short_code()
        assert method.code().startswith("MULTIPLE_TESTING_")
        assert method.metadata.references
