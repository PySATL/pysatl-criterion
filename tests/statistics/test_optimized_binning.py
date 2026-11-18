"""Deterministic selection references, independent of Monte Carlo noise."""

from dataclasses import FrozenInstanceError, dataclass

import numpy as np
import pytest

from pysatl_criterion import DistributionKind
from pysatl_criterion.statistics.goodness_of_fit.binning import (
    AlternativeOptimizedBins,
    EqualProbabilityBins,
    FrequencyTable,
)


@dataclass(frozen=True)
class ScoredBins:
    """Encode a known score in a valid frequency table; -1 denotes the alternative."""

    alternative_score: float

    def supports(self, kind):
        return kind is DistributionKind.CONTINUOUS

    def build(self, sample, distribution):
        score = self.alternative_score if sample[0] == -1 else sample[0]
        return FrequencyTable([score, len(sample) - score], [len(sample) / 2] * 2)


@pytest.fixture
def selection_kwargs():
    # Unsorted null scores catch a missing sort as well as incorrect quantile ranks.
    scores = iter(np.roll(np.arange(20), 7))
    return {
        "candidates": [ScoredBins(10), ScoredBins(20)],
        "distribution": object(),
        "null_sampler": lambda n, rng: np.full(n, next(scores)),
        "alternative_sampler": lambda n, rng: np.full(n, -1),
        "kernel": lambda observed, expected: float(observed[0]),
        "sample_size": 100,
        "repetitions": 20,
    }


def test_selection_chooses_greatest_power_in_candidate_order(selection_kwargs):
    result = AlternativeOptimizedBins.select(**selection_kwargs)
    assert result.selected is selection_kwargs["candidates"][1]
    assert result.estimated_powers == (0.0, 1.0)
    assert result.sample_size == 100


def test_selection_ties_keep_first_candidate(selection_kwargs):
    selection_kwargs["candidates"] = [ScoredBins(20), ScoredBins(21)]
    result = AlternativeOptimizedBins.select(**selection_kwargs)
    assert result.estimated_powers == (1.0, 1.0)
    assert result.selected is selection_kwargs["candidates"][0]


@pytest.mark.parametrize(
    "alpha,score,power",
    [
        (0.05, 18, 0),
        (0.05, 19, 0),
        (0.05, 20, 1),
        (0.5, 10, 0),
        (0.5, 11, 1),
        (0.95, 1, 0),
        (0.95, 2, 1),
    ],
)
def test_order_statistic_and_strict_rejection_boundary(selection_kwargs, alpha, score, power):
    # For null scores 0..19, the critical scores are 19, 10 and 1 respectively.
    selection_kwargs.update(candidates=[ScoredBins(score)], alpha=alpha)
    assert AlternativeOptimizedBins.select(**selection_kwargs).estimated_powers == (power,)


def test_estimated_power_counts_only_rejections(selection_kwargs):
    scores = iter([18, 19, 20, 21] * 5)
    selection_kwargs.update(
        candidates=[ScoredBins(0)],
        alternative_sampler=lambda n, rng: np.full(n, next(scores)),
    )
    assert AlternativeOptimizedBins.select(**selection_kwargs).estimated_powers == (0.5,)


@pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.1, np.nan, np.inf])
def test_invalid_alpha_is_rejected_before_sampling(selection_kwargs, alpha, mocker):
    sampler = mocker.Mock(side_effect=AssertionError("must not sample"))
    selection_kwargs.update(alpha=alpha, null_sampler=sampler, alternative_sampler=sampler)
    with pytest.raises(ValueError, match="alpha"):
        AlternativeOptimizedBins.select(**selection_kwargs)
    sampler.assert_not_called()


def test_empty_candidates_are_rejected(selection_kwargs):
    selection_kwargs["candidates"] = []
    with pytest.raises(ValueError, match="Candidates"):
        AlternativeOptimizedBins.select(**selection_kwargs)


@pytest.mark.parametrize(
    "name,values", [("sample_size", [0, -1, True, 1.5]), ("repetitions", [0, 19, True, 20.5])]
)
def test_invalid_integer_settings(selection_kwargs, name, values):
    for value in values:
        with pytest.raises(ValueError, match=name):
            AlternativeOptimizedBins.select(**{**selection_kwargs, name: value})


@pytest.mark.parametrize("sampler", ["null_sampler", "alternative_sampler"])
@pytest.mark.parametrize("sample", [np.zeros(99), np.zeros(101), [], [[0]], [np.nan], [np.inf]])
def test_invalid_sampler_output(selection_kwargs, sampler, sample):
    selection_kwargs[sampler] = lambda n, rng: sample
    with pytest.raises(ValueError, match="[Ss]ample"):
        AlternativeOptimizedBins.select(**selection_kwargs)


@pytest.mark.parametrize("nan_position", [0, 1])
def test_nan_from_either_population_is_rejected(selection_kwargs, nan_position):
    values = iter([np.nan, 0] if nan_position == 0 else [0, np.nan])
    selection_kwargs.update(
        candidates=[ScoredBins(0)], kernel=lambda observed, expected: next(values, 0)
    )
    with pytest.raises(ValueError, match="Kernel produced NaN"):
        AlternativeOptimizedBins.select(**selection_kwargs)


@pytest.mark.parametrize(
    "null_score,alternative_score,power", [(np.inf, np.inf, 0), (1, np.inf, 1), (np.inf, 1, 0)]
)
def test_infinite_kernel_values(selection_kwargs, null_score, alternative_score, power):
    selection_kwargs.update(
        candidates=[ScoredBins(0)],
        kernel=lambda observed, expected: alternative_score if observed[0] == 0 else null_score,
        null_sampler=lambda n, rng: np.full(n, 1),
    )
    assert AlternativeOptimizedBins.select(**selection_kwargs).estimated_powers == (power,)


def test_candidate_failure_is_not_silently_excluded(selection_kwargs, mocker):
    broken = mocker.Mock(spec=ScoredBins)
    broken.build.side_effect = RuntimeError("candidate failed")
    selection_kwargs["candidates"].append(broken)
    with pytest.raises(RuntimeError, match="candidate failed"):
        AlternativeOptimizedBins.select(**selection_kwargs)


def test_candidates_share_simulated_samples_and_rng(selection_kwargs, mocker):
    candidates = [mocker.Mock(wraps=c) for c in selection_kwargs["candidates"]]
    null = mocker.Mock(wraps=selection_kwargs["null_sampler"])
    alternative = mocker.Mock(wraps=selection_kwargs["alternative_sampler"])
    selection_kwargs.update(
        candidates=candidates, null_sampler=null, alternative_sampler=alternative
    )
    AlternativeOptimizedBins.select(**selection_kwargs)
    assert null.call_count == alternative.call_count == 20
    rng = null.call_args.args[1]
    assert isinstance(rng, np.random.Generator)
    assert all(call.args[1] is rng for call in null.call_args_list + alternative.call_args_list)
    for left, right in zip(
        candidates[0].build.call_args_list, candidates[1].build.call_args_list, strict=True
    ):
        assert left.args[0] is right.args[0]
        assert left.args[1] is right.args[1] is selection_kwargs["distribution"]
    assert candidates[0].build.call_count == candidates[1].build.call_count == 40


def test_frozen_result_delegates_and_checks_sample_before_build(mocker):
    selected = mocker.Mock(spec=EqualProbabilityBins)
    selected.supports.return_value = False
    result = AlternativeOptimizedBins(selected, 3, (0.5,))
    assert result.supports(DistributionKind.DISCRETE) is False
    selected.supports.assert_called_once_with(DistributionKind.DISCRETE)
    distribution = object()
    assert result.build([0.1, 0.2, 0.3], distribution) is selected.build.return_value
    np.testing.assert_array_equal(selected.build.call_args.args[0], [0.1, 0.2, 0.3])
    assert selected.build.call_args.args[1] is distribution
    selected.build.reset_mock()
    for sample in ([0.1], [], [0.1, np.nan, 0.3]):
        with pytest.raises(ValueError, match="[Ss]ample"):
            result.build(sample, distribution)
    selected.build.assert_not_called()
    with pytest.raises(FrozenInstanceError):
        result.sample_size = 4


@pytest.mark.parametrize("sample_size", [0, -1, True, 1.5])
def test_result_rejects_invalid_sample_size(sample_size):
    with pytest.raises(ValueError, match="sample_size"):
        AlternativeOptimizedBins(EqualProbabilityBins(2), sample_size, ())
