from pysatl_criterion.hypothesis_testing.limit_distribution.base import (
    StorageLimitDistributionResolver,
)
from pysatl_criterion.persistence.models.limit_distribution import (
    LimitDistributionModel,
    LimitDistributionQuery,
)


class FakeHypothesis:
    def __init__(self, parameters):
        self._parameters = parameters

    def parameters(self):
        return self._parameters


class FakeStatistic:
    def __init__(self, code, parameters):
        self._code = code
        self._hypothesis = FakeHypothesis(parameters)

    def code(self):
        return self._code

    def hypothesis(self):
        return self._hypothesis


class FakeStorage:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def get_data(self, query):
        self.calls.append(query)
        return self.result


def test_resolve_returns_storage_results_as_list():
    storage_model = LimitDistributionModel(
        experiment_id=1,
        criterion_code="ks",
        criterion_parameters=[0.5, 1.5],
        sample_size=25,
        monte_carlo_count=1000,
        results_statistics=(1.0, 2.0, 3.0),
    )
    storage = FakeStorage(storage_model)
    statistic = FakeStatistic(code="ks", parameters=[0.5, 1.5])
    resolver = StorageLimitDistributionResolver(storage)

    result = resolver.resolve(statistic, sample_size=25)

    assert result == [1.0, 2.0, 3.0]
    assert storage.calls == [
        LimitDistributionQuery(
            criterion_code="ks",
            criterion_parameters=[0.5, 1.5],
            sample_size=25,
            monte_carlo_count=1,
        )
    ]


def test_resolve_returns_none_when_storage_has_no_distribution():
    storage = FakeStorage(None)
    statistic = FakeStatistic(code="ad", parameters=[])
    resolver = StorageLimitDistributionResolver(storage)

    result = resolver.resolve(statistic, sample_size=50)

    assert result is None
    assert storage.calls == [
        LimitDistributionQuery(
            criterion_code="ad",
            criterion_parameters=[],
            sample_size=50,
            monte_carlo_count=1,
        )
    ]
