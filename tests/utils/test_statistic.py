import inspect
from collections import Counter
from importlib import import_module

import pytest

from pysatl_criterion import DistributionType
from pysatl_criterion.statistics import AbstractGoodnessOfFitStatistic
from pysatl_criterion.statistics.goodness_of_fit.continuous.beta import (
    KolmogorovSmirnovBetaGofStatistic,
)
from pysatl_criterion.utils.statistic import (
    get_available_criteria,
    get_available_criteria_codes,
    get_criterion_by_code,
)


@pytest.mark.parametrize(
    "family,module_name",
    [
        (DistributionType.NORMAL, "normal"),
        (DistributionType.EXPONENTIAL, "exponent"),
        (DistributionType.WEIBULL, "weibull"),
        (DistributionType.EXPONENTIATED_WEIBULL, "exponentiated_weibull"),
        (DistributionType.UNIFORM, "uniform"),
        (DistributionType.STUDENT, "student"),
        (DistributionType.GAMMA, "gamma"),
    ],
)
def test_get_available_criteria_matches_public_exports(family, module_name):
    module = import_module(f"pysatl_criterion.statistics.goodness_of_fit.continuous.{module_name}")
    expected = {
        getattr(module, name)
        for name in module.__all__
        if inspect.isclass(getattr(module, name))
        and issubclass(getattr(module, name), AbstractGoodnessOfFitStatistic)
        and not inspect.isabstract(getattr(module, name))
    }
    actual = get_available_criteria(family)
    assert expected
    assert set(actual) == expected
    assert len(actual) == len(expected)
    assert Counter(get_available_criteria_codes(family)) == Counter(
        c.short_code() for c in expected
    )


@pytest.mark.parametrize(
    "distribution",
    ["normal", "exponential", "weibull", "uniform", "student", "gamma"],
)
def test_get_available_criteria_all_criteria_classes(distribution):
    criteria = get_available_criteria(DistributionType(distribution))

    assert criteria
    assert all(issubclass(criterion, AbstractGoodnessOfFitStatistic) for criterion in criteria)
    assert all(not inspect.isabstract(criterion) for criterion in criteria)
    assert all(
        criterion.distribution().type() == DistributionType(distribution) for criterion in criteria
    )


def test_get_criterion_by_code_returns_criterion_class():
    criterion = get_criterion_by_code(KolmogorovSmirnovBetaGofStatistic.code())

    assert criterion is KolmogorovSmirnovBetaGofStatistic


def test_get_criterion_by_code_raises_for_unknown_code():
    with pytest.raises(ValueError, match="UNKNOWN"):
        get_criterion_by_code("UNKNOWN")
