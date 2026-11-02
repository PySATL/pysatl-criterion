"""Cross-family regressions for the shared ParameterValues constructor."""

import inspect
from dataclasses import replace
from itertools import combinations

import numpy as np
import pytest

from pysatl_criterion import DistributionType
from pysatl_criterion.distribution import BetaDistributionDescriptor as Beta
from pysatl_criterion.distribution import NormalDistributionDescriptor as Normal
from pysatl_criterion.statistics import AbstractGoodnessOfFitStatistic
from pysatl_criterion.statistics.hypothesis import GoodnessOfFitHypothesis
from pysatl_criterion.utils.statistic import get_available_criteria
from tests.parameter_cases import parameters_for


CRITERIA = sorted(
    {criterion for family in DistributionType for criterion in get_available_criteria(family)},
    key=lambda criterion: criterion.__name__,
)
FITTED = {
    "EbnerLiebenbergBetaGofStatistic",
    "RaschkeBetaGofStatistic",
    "GreenwoodParetoGofStatistic",
    "LequesneKlParetoGofStatistic",
    "NikulinLogLogisticGofStatistic",
    "MirvalievLogLogisticGofStatistic",
    "KullbackLeiblerBetaGofStatistic",
    "RiadAbdElMaboodBetaGofStatistic",
    "AlizadehNoughabiJarrahiferizLaplaceGofStatistic",
    "AlizadehNoughabiLikelihoodLaplaceGofStatistic",
    "AlizadehNoughabiParkALaplaceGofStatistic",
    "AlizadehNoughabiParkELaplaceGofStatistic",
    "AlizadehNoughabiParkVLaplaceGofStatistic",
    "AlizadehNoughabiParkYLaplaceGofStatistic",
    "AlizadehNoughabiParkZLaplaceGofStatistic",
    "BestRaynerThasV3LaplaceGofStatistic",
    "BestRaynerThasV4LaplaceGofStatistic",
    "ChoiKimCLaplaceGofStatistic",
    "ChoiKimELaplaceGofStatistic",
    "ChoiKimVLaplaceGofStatistic",
    "DataDrivenSmoothLaplaceGofStatistic",
    "DesgagneLafayeDeMicheauxOuimetXLaplaceGofStatistic",
    "DesgagneLafayeDeMicheauxOuimetZLaplaceGofStatistic",
    "GelLaplaceGofStatistic",
    "GonzalezEstradaVillasenorLaplaceGofStatistic",
    "Meintanis1LaplaceGofStatistic",
    "Meintanis2LaplaceGofStatistic",
}


EXPONENTIAL_BOTH_MODES = {
    "AndersonDarlingExponentialityGofStatistic",
    "CramerVonMisesExponentialityGofStatistic",
    "WatsonExponentialityGofStatistic",
    "KuiperExponentialityGofStatistic",
    "Chi2PearsonExponentialityGofStatistic",
    "CressieReadExponentialityGofStatistic",
    "FreemanTukeyExponentialityGofStatistic",
    "KuiperCorrectedExponentialityGofStatistic",
    "KuiperScaledExponentialityGofStatistic",
    "LiaoShimokawaExponentialityGofStatistic",
    "ModifiedLogLikelihoodExponentialityGofStatistic",
    "NeymanChi2ExponentialityGofStatistic",
    "ZhangAExponentialityGofStatistic",
    "ZhangCExponentialityGofStatistic",
    "ZhangKExponentialityGofStatistic",
}


LOG_NORMAL_BOTH_MODES = {
    "BonettSeier",
    "BontempsMeddahi1",
    "BontempsMeddahi2",
    "CabanaCabana1",
    "CabanaCabana2",
    "ChenShapiro",
    "Coin",
    "DAP",
    "Dagostino",
    "DesgagneLafaye",
    "DoornikHansen",
    "EppsPulley",
    "Filli",
    "GMG",
    "GlenLeemisBarr",
    "Hosking1",
    "Hosking2",
    "Hosking3",
    "Hosking4",
    "JB",
    "KLIntegral",
    "KLSupremum",
    "Kurtosis",
    "LooneyGulledge",
    "MartinezIglewicz",
    "RobustJarqueBera",
    "RyanJoiner",
    "SF",
    "SWRG",
    "ShapiroWilk",
    "Skew",
    "Spiegelhalter",
    "ZhangQ",
    "ZhangQStar",
}


def expected_fixed(criterion):
    """Independent inventory of the implemented null hypotheses."""
    name = criterion.__name__
    family = criterion.distribution().type()
    if family is DistributionType.WEIBULL or name in FITTED:
        return set()
    if family is DistributionType.NORMAL:
        if name.startswith("Graph"):
            return {"var"}
        if name not in {
            "KolmogorovSmirnovNormalityGofStatistic",
            "CramerVonMiseNormalityGofStatistic",
        }:
            return set()
    if family is DistributionType.EXPONENTIAL and name not in {
        "KolmogorovSmirnovExponentialityGofStatistic",
        "KolmogorovSmirnovDMinusExponentialityGofStatistic",
        "KolmogorovSmirnovDPlusExponentialityGofStatistic",
        "CramerVonMisesExponentialityGofStatistic",
    }:
        return set()
    if name in {
        "CadirciRenyiStudentGofStatistic",
        "LequesneRenyiStudentGofStatistic",
        "MeintanisStudentGofStatistic",
    }:
        return {"df"}
    if name == "ObradovicParetoGofStatistic":
        return {"scale"}
    return {parameter.name for parameter in criterion.distribution().DEFAULT.parameters}


@pytest.mark.parametrize("criterion", CRITERIA, ids=lambda cls: cls.__name__)
def test_shared_constructor_checks_every_fixed_parameter_subset(criterion):
    descriptor = criterion.distribution()
    schema = descriptor.DEFAULT
    names = [parameter.name for parameter in schema.parameters]
    defaults = {parameter.name: parameter.default for parameter in schema.parameters}
    if descriptor is Beta:
        defaults.update(a=2, b=3)  # Mode requires both shapes to exceed one.
    if descriptor.type() is DistributionType.STUDENT:
        defaults["df"] = 5
    expected = expected_fixed(criterion)
    for count in range(len(names) + 1):
        for subset in combinations(names, count):
            values = schema.parse({name: defaults[name] for name in subset})
            supported = set(subset) == expected
            if criterion.__name__ in EXPONENTIAL_BOTH_MODES:
                supported = set(subset) in (set(), {"lam"})
            if criterion.__name__ == "LequesneRenyiStudentGofStatistic":
                supported = set(subset) in (set(), {"df"})
            if criterion.distribution().type() is DistributionType.LOG_NORMAL:
                prefix = criterion.__name__.replace("LogNormalGofStatistic", "").replace(
                    "LogNormalGoFStatistic", ""
                )
                if prefix in LOG_NORMAL_BOTH_MODES:
                    supported = set(subset) in (set(), {"mu", "s"})
            assert criterion.supports_hypothesis(GoodnessOfFitHypothesis(values)) is supported
            if supported:
                statistic = criterion(values)
                assert statistic.hypothesis().parameter_values is values
                assert statistic.hypothesis().parameters() == values.as_dict()
                assert statistic.hypothesis.__func__ is AbstractGoodnessOfFitStatistic.hypothesis
            else:
                with pytest.raises(ValueError, match="Unsupported"):
                    criterion(values)


@pytest.mark.parametrize("criterion", CRITERIA, ids=lambda cls: cls.__name__)
def test_constructor_requires_values_and_initializes_them_once(criterion, mocker):
    descriptor = criterion.distribution()
    values = parameters_for(
        criterion, **({"a": 2, "b": 3} if descriptor is Beta and expected_fixed(criterion) else {})
    )
    spy = mocker.spy(descriptor, "convert_parameters")
    criterion(values)
    assert spy.call_count == 1
    assert not hasattr(criterion, "from_parameters")
    signature = inspect.signature(criterion)
    assert signature.parameters["parameters"].default is inspect.Parameter.empty
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for name, parameter in signature.parameters.items()
        if name != "parameters"
    )
    with pytest.raises(TypeError, match="ParameterValues"):
        criterion(values.as_dict())
    with pytest.raises(TypeError):
        criterion()
    foreign = replace(descriptor.DEFAULT, id="unsupported.coordinates").parse(values.as_dict())
    with pytest.raises(ValueError, match="Unsupported"):
        criterion(foreign)


@pytest.mark.parametrize("value", [True, np.bool_(True), [1], np.array(1), np.array([1]), "1", 1j])
def test_schema_rejects_non_real_scalar_values_even_without_a_custom_validator(value):
    with pytest.raises(ValueError, match="Invalid value for mean"):
        Normal.DEFAULT.parse({"mean": value})
