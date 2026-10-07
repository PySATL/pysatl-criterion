import numpy as np
import pytest
from scipy import stats

from pysatl_criterion.hypothesis_testing.limit_distribution.base import (
    MonteCarloLimitDistributionResolver,
)
from pysatl_criterion.statistics.alternative import AlternativeType
from pysatl_criterion.statistics.goodness_of_fit.normal import (
    CramerVonMiseNormalityGofStatistic,
    KolmogorovSmirnovNormalityGofStatistic,
)
from tests.parameter_cases import parameters_for


STATISTIC_CLASSES = [KolmogorovSmirnovNormalityGofStatistic, CramerVonMiseNormalityGofStatistic]
PARAMETER_CASES = [
    pytest.param({}, id="defaults"),
    pytest.param({"mean": 3}, id="mean-only"),
    pytest.param({"var": 4}, id="variance-only"),
    pytest.param({"mean": -2, "var": 9}, id="mean-and-variance"),
    pytest.param({"mean": 1.5, "var": 0.25}, id="variance-below-one"),
]


@pytest.fixture(params=PARAMETER_CASES)
def normal_case(request):
    parameters = {"mean": 0, "var": 1, **request.param}
    # Keep the observations unsorted, including a repeated value.
    data = parameters["mean"] + np.sqrt(parameters["var"]) * np.array(
        [1.6, -0.8, 0.25, 2.2, -1.4, 0.7, 0.25, -0.05]
    )
    reference = stats.norm(loc=parameters["mean"], scale=np.sqrt(parameters["var"]))
    return request.param, parameters, data, reference


@pytest.mark.parametrize(
    ("alternative", "scipy_alternative"),
    [
        (AlternativeType.TWO_TAILED, "two-sided"),
        (AlternativeType.RIGHT, "greater"),
        (AlternativeType.LEFT, "less"),
    ],
)
def test_ks_respects_normal_parameters(normal_case, alternative, scipy_alternative):
    kwargs, parameters, data, reference = normal_case
    statistic = KolmogorovSmirnovNormalityGofStatistic(
        parameters_for(KolmogorovSmirnovNormalityGofStatistic, **kwargs),
        alternative_type=alternative,
    )
    expected = stats.kstest(data, reference.cdf, alternative=scipy_alternative).statistic

    assert statistic.execute_statistic(data) == pytest.approx(expected, rel=1e-12)
    assert statistic.hypothesis().parameters() == parameters


def test_cvm_respects_normal_parameters(normal_case):
    kwargs, parameters, data, reference = normal_case
    statistic = CramerVonMiseNormalityGofStatistic(
        parameters_for(CramerVonMiseNormalityGofStatistic, **kwargs)
    )
    expected = stats.cramervonmises(data, reference.cdf).statistic

    assert statistic.execute_statistic(data) == pytest.approx(expected, rel=1e-12)
    assert statistic.hypothesis().parameters() == parameters


@pytest.mark.parametrize("statistic_class", STATISTIC_CLASSES)
@pytest.mark.parametrize("mean", [np.nan, np.inf, -np.inf])
def test_normal_parameters_reject_nonfinite_mean(statistic_class, mean):
    with pytest.raises(ValueError, match="Invalid value for mean"):
        statistic_class(parameters_for(statistic_class, mean=mean))


@pytest.mark.parametrize("statistic_class", STATISTIC_CLASSES)
@pytest.mark.parametrize("var", [0, -1, np.nan, np.inf, -np.inf])
def test_normal_parameters_reject_invalid_variance(statistic_class, var):
    with pytest.raises(ValueError, match="Invalid value for var"):
        statistic_class(parameters_for(statistic_class, var=var))


@pytest.mark.parametrize(
    ("statistic_class", "reference_test"),
    [
        (KolmogorovSmirnovNormalityGofStatistic, stats.kstest),
        (CramerVonMiseNormalityGofStatistic, stats.cramervonmises),
    ],
)
def test_monte_carlo_samples_and_evaluates_the_same_normal_hypothesis(
    mocker, statistic_class, reference_test
):
    samples = np.array([4.5, 1.0, 3.5, -0.5, 6.0])
    sampler = mocker.patch(
        "pysatl_criterion.generator.generators.generate_norm", return_value=samples
    )
    statistic = statistic_class(parameters_for(statistic_class, mean=3, var=4))
    expected = reference_test(samples, stats.norm(loc=3, scale=2).cdf).statistic

    actual = MonteCarloLimitDistributionResolver(2).resolve(statistic, sample_size=5)

    assert actual == pytest.approx([expected, expected], rel=1e-12)
    assert sampler.call_args_list == [mocker.call(size=5, mean=3, var=4, random_state=None)] * 2
