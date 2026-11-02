import pytest

from pysatl_criterion.hypothesis_testing.limit_distribution.base import (
    AbstractLimitDistributionResolver,
    CompositeLimitDistributionResolver,
)


class FakeResolver(AbstractLimitDistributionResolver):
    def __init__(self, result, calls):
        self.result = result
        self.calls = calls

    def resolve(self, statistic, sample_size):
        self.calls.append((self, statistic, sample_size))
        return self.result


@pytest.mark.parametrize(
    ("results", "expected", "call_count"),
    [
        ([[1.0], [2.0], [3.0]], [1.0], 1),
        ([None, [2.0], [3.0]], [2.0], 2),
        ([None, None, [3.0], [4.0]], [3.0], 3),
        ([None, None, None], None, 3),
        ([None, [], [3.0]], [], 2),
        ([], None, 0),
    ],
)
def test_resolve_tries_resolvers_in_order_until_result(results, expected, call_count):
    statistic = object()
    calls = []
    resolvers = [FakeResolver(result, calls) for result in results]
    composite = CompositeLimitDistributionResolver(resolvers)

    assert composite.resolve(statistic, sample_size=10) == expected
    assert calls == [(resolver, statistic, 10) for resolver in resolvers[:call_count]]


def test_resolve_propagates_errors_without_trying_next_resolver(mocker):
    statistic = object()
    first = mocker.Mock(spec=AbstractLimitDistributionResolver)
    first.resolve.side_effect = ValueError("Invalid sample")
    second = mocker.Mock(spec=AbstractLimitDistributionResolver)
    composite = CompositeLimitDistributionResolver([first, second])

    with pytest.raises(ValueError, match="Invalid sample"):
        composite.resolve(statistic, sample_size=10)

    second.resolve.assert_not_called()
