import pytest

from pysatl_criterion import DistributionType
from pysatl_criterion.utils.generator import get_available_generator


@pytest.mark.parametrize(
    ("distribution", "params", "generator_type_name"),
    [
        (DistributionType.BETA, {"a": 2, "b": 3}, "BetaRVSGenerator"),
        (DistributionType.NORMAL, {"mean": 1, "var": 4}, "NormalGenerator"),
    ],
)
def test_get_available_generator_returns_matching_generator(
    distribution,
    params,
    generator_type_name,
):
    generator = get_available_generator(distribution, params)

    assert type(generator).__name__ == generator_type_name
    assert generator.distribution_type() == distribution
    assert generator.parameters() == params


def test_get_available_generator_raises_for_missing_generator():
    with pytest.raises(StopIteration):
        get_available_generator(object(), {})
