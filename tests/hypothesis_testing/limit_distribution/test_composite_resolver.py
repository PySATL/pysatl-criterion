from pysatl_criterion.hypothesis_testing.limit_distribution.base import (
    CompositeLimitDistributionResolver,
)


class FakeResolver:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def resolve(self, statistic, sample_size):
        self.calls.append((statistic, sample_size))
        return self.result


def test_resolve_returns_local_results_when_available():
    statistic = object()
    local_resolver = FakeResolver([1.0, 2.0, 3.0])
    monte_carlo_resolver = FakeResolver([4.0, 5.0, 6.0])
    resolver = CompositeLimitDistributionResolver(local_resolver, monte_carlo_resolver)

    result = resolver.resolve(statistic, sample_size=10)

    assert result == [1.0, 2.0, 3.0]
    assert local_resolver.calls == [(statistic, 10)]
    assert monte_carlo_resolver.calls == []


def test_resolve_uses_monte_carlo_results_when_local_results_are_missing():
    statistic = object()
    local_resolver = FakeResolver(None)
    monte_carlo_resolver = FakeResolver([4.0, 5.0, 6.0])
    resolver = CompositeLimitDistributionResolver(local_resolver, monte_carlo_resolver)

    result = resolver.resolve(statistic, sample_size=10)

    assert result == [4.0, 5.0, 6.0]
    assert local_resolver.calls == [(statistic, 10)]
    assert monte_carlo_resolver.calls == [(statistic, 10)]
