"""Regression checks for the Beta statistics audit, including independent references."""

import numpy as np
import pytest
from scipy import stats

from pysatl_criterion.distribution import BetaDistributionDescriptor as Beta
from pysatl_criterion.statistics.alternative import RightAlternative
from pysatl_criterion.statistics.goodness_of_fit.continuous import beta
from pysatl_criterion.statistics.goodness_of_fit.continuous.beta import beta_statistics


STATISTICS = [
    beta.KolmogorovSmirnovBetaGofStatistic,
    beta.AndersonDarlingBetaGofStatistic,
    beta.CrammerVonMisesBetaGofStatistic,
    beta.EbnerLiebenbergBetaGofStatistic,
    beta.WatsonBetaGofStatistic,
    beta.KuiperBetaGofStatistic,
    beta.Chi2PearsonBetaGofStatistic,
]


@pytest.mark.parametrize("cls", STATISTICS)
@pytest.mark.parametrize("sample", [[], [[0.2, 0.3]], [np.nan] * 8, [np.inf], [-0.1], [1.1]])
def test_invalid_samples(cls, sample):
    with pytest.raises(ValueError):
        fixed = {} if cls is beta.EbnerLiebenbergBetaGofStatistic else {"a": 2, "b": 5}
        statistic = cls(Beta.DEFAULT.parse(fixed))
        statistic.execute_statistic(sample)


@pytest.mark.parametrize(
    "cls", [cls for cls in STATISTICS if cls is not beta.EbnerLiebenbergBetaGofStatistic]
)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, [2, 3]])
@pytest.mark.parametrize("parameter", ["alpha", "beta"])
def test_invalid_parameters(cls, value, parameter):
    with pytest.raises(ValueError):
        cls(Beta.DEFAULT.parse({"a": 2, "b": 5, {"alpha": "a", "beta": "b"}[parameter]: value}))


@pytest.mark.parametrize(
    "cls,scipy_direction",
    [
        (beta.KolmogorovSmirnovBetaGofStatistic, "two-sided"),
        (beta.KolmogorovSmirnovDPlusBetaGofStatistic, "greater"),
        (beta.KolmogorovSmirnovDMinusBetaGofStatistic, "less"),
    ],
)
def test_ks_direction_and_rejection_tail(cls, scipy_direction):
    sample = [0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57]
    statistic = cls(Beta.DEFAULT.parse({"a": 2, "b": 5}))
    assert isinstance(statistic.alternative(), RightAlternative)
    expected = stats.kstest(sample, stats.beta(2, 5).cdf, alternative=scipy_direction).statistic
    assert statistic.execute_statistic(sample) == pytest.approx(expected)


def test_ad_boundary_infinity_is_preserved():
    assert np.isposinf(
        beta.AndersonDarlingBetaGofStatistic(
            Beta.DEFAULT.parse({"a": 2, "b": 5})
        ).execute_statistic([0, 0.5, 1])
    )


def test_pearson_uses_survival_probabilities_in_right_tail():
    # The last CDF difference rounds to zero, while the survival probability does not.
    sample = stats.beta(2, 50).ppf((np.arange(100) + 0.5) / 100)
    statistic = beta.Chi2PearsonBetaGofStatistic(
        Beta.DEFAULT.parse({"a": 2, "b": 50})
    ).execute_statistic(sample)
    assert np.isfinite(statistic)
    assert statistic >= 0


@pytest.mark.parametrize("cls", STATISTICS)
def test_all_beta_statistics_support_uniform_scalar_execution(cls):
    from pysatl_criterion.statistics import AbstractGoodnessOfFitStatistic

    statistic: AbstractGoodnessOfFitStatistic = cls(
        Beta.DEFAULT.parse({} if cls is beta.EbnerLiebenbergBetaGofStatistic else {"a": 1, "b": 1})
    )
    sample = np.array([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
    original = sample.copy()
    result = statistic.execute_statistic(sample)
    assert isinstance(result, (float, np.float64))
    assert np.isfinite(result)
    assert result >= 0
    np.testing.assert_array_equal(sample, original)
    assert statistic.execute_statistic(sample) == result


def test_composite_beta_cannot_use_default_uniform_calibration():
    from pysatl_criterion.utils.generator import get_hypothesis_generator

    with pytest.raises(ValueError, match="simulation shapes externally"):
        get_hypothesis_generator(beta.EbnerLiebenbergBetaGofStatistic(Beta.DEFAULT.parse({})))


@pytest.mark.parametrize("a,b", [(1, 1), (2, 5), (5, 2), (0.5, 0.5)])
def test_fixed_edf_statistics_match_uniform_transform(a, b):
    u = (np.arange(41) + 0.3) / 41
    sample = stats.beta(a, b).ppf(u)
    for cls in [
        beta.KolmogorovSmirnovBetaGofStatistic,
        beta.AndersonDarlingBetaGofStatistic,
        beta.CrammerVonMisesBetaGofStatistic,
        beta.WatsonBetaGofStatistic,
        beta.KuiperBetaGofStatistic,
    ]:
        assert cls(Beta.DEFAULT.parse({"a": a, "b": b})).execute_statistic(sample) == pytest.approx(
            cls(
                Beta.DEFAULT.parse(
                    {} if cls is beta.EbnerLiebenbergBetaGofStatistic else {"a": 1, "b": 1}
                )
            ).execute_statistic(u),
            abs=2e-13,
        )


def test_beta_docstring_examples():
    import doctest

    # Examples use the implementation namespace (Beta, NumPy and SciPy).
    result = doctest.testmod(beta, extraglobs=vars(beta_statistics), raise_on_error=True)
    assert result.attempted >= 3


@pytest.mark.parametrize("cls", STATISTICS)
def test_each_statistic_documents_formula_hypothesis_and_source_status(cls):
    import inspect

    doc = inspect.getdoc(cls)
    if cls.__name__ in {*beta_statistics._EDF_STATISTICS, *beta_statistics._FREQUENCY_STATISTICS}:
        assert "Beta hypothesis" in doc
        assert "https://doi.org/" in doc
        return
    assert "Notes\n-----" in doc
    assert "Examples\n--------" in doc
    assert "Returns\n-------" in inspect.getdoc(cls.execute_statistic)
    assert "References\n----------" in doc
    assert "https://doi.org/" in doc


@pytest.mark.parametrize("cls", STATISTICS)
@pytest.mark.parametrize(
    "sample", [np.array([0.1 + 0.2j, 0.3 + 0.4j]), np.ma.array([0.2, 0.4], mask=[0, 1])]
)
def test_no_silent_loss_of_sample_information(cls, sample):
    with pytest.raises(ValueError, match="real|masked"):
        cls(
            Beta.DEFAULT.parse(
                {} if cls is beta.EbnerLiebenbergBetaGofStatistic else {"a": 1, "b": 1}
            )
        ).execute_statistic(sample)


@pytest.mark.parametrize("parameter", ["alpha", "beta"])
def test_complex_shape_is_rejected(parameter):
    with pytest.raises(ValueError):
        beta.KolmogorovSmirnovBetaGofStatistic(
            Beta.DEFAULT.parse({"a": 2, "b": 5, {"alpha": "a", "beta": "b"}[parameter]: 2 + 1j})
        )


@pytest.mark.parametrize("direction", [None, "left", 42])
def test_invalid_ks_direction_is_not_silently_two_sided(direction):
    with pytest.raises(TypeError, match="alternative_type"):
        beta.KolmogorovSmirnovBetaGofStatistic(
            Beta.DEFAULT.parse({"a": 1, "b": 1}), alternative_type=direction
        )


@pytest.mark.parametrize("power", [np.nan, np.inf, [1, 2], 1 + 2j])
def test_power_parameter_validated_at_construction(power):
    with pytest.raises(TypeError, match="lambda_"):
        beta.Chi2PearsonBetaGofStatistic(Beta.DEFAULT.parse({"a": 1, "b": 1}), lambda_=power)


@pytest.mark.parametrize("cls", STATISTICS)
def test_schema_construction_and_supported_fixed_parameters(cls):
    from dataclasses import replace

    from pysatl_criterion.distribution import BetaDistributionDescriptor as Beta
    from pysatl_criterion.statistics.hypothesis import GoodnessOfFitHypothesis

    fitted = cls is beta.EbnerLiebenbergBetaGofStatistic
    values = Beta.DEFAULT.parse({} if fitted else {Beta.ALPHA: 2, Beta.BETA: 5})
    statistic = cls(values)
    assert statistic.supports_hypothesis(statistic.hypothesis())
    assert statistic.hypothesis().parameter_values == values
    if not fitted:
        assert (statistic.alpha, statistic.beta) == (2, 5)
    for fixed in ({}, {Beta.ALPHA: 2}, {Beta.BETA: 5}, {Beta.ALPHA: 2, Beta.BETA: 5}):
        candidate = Beta.DEFAULT.parse(fixed)
        supported = len(fixed) == (0 if fitted else 2)
        assert cls.supports_hypothesis(GoodnessOfFitHypothesis(candidate)) is supported
        if not supported:
            with pytest.raises(ValueError, match="Unsupported"):
                cls(candidate)
    foreign = replace(Beta.DEFAULT, id="beta.other").parse(values.as_dict())
    with pytest.raises(ValueError, match="Unsupported"):
        cls(foreign)


def test_schema_construction_forwards_algorithm_options():
    from pysatl_criterion.distribution import BetaDistributionDescriptor as Beta

    values = Beta.DEFAULT.parse({"a": 2, "b": 5})
    statistic = beta.KolmogorovSmirnovDPlusBetaGofStatistic(values)
    expected = beta.KolmogorovSmirnovDPlusBetaGofStatistic(Beta.DEFAULT.parse({"a": 2, "b": 5}))
    sample = [0.1, 0.2, 0.3, 0.5]
    assert statistic.execute_statistic(sample) == expected.execute_statistic(sample)
    with pytest.raises(TypeError):
        beta.KolmogorovSmirnovBetaGofStatistic(values, alpha=3)


def test_schema_hypothesis_generator_preserves_beta_shapes():
    from pysatl_criterion.distribution import BetaDistributionDescriptor as Beta
    from pysatl_criterion.utils.generator import get_hypothesis_generator

    values = Beta.DEFAULT.parse({"a": 2, "b": 5})
    statistic = beta.KolmogorovSmirnovBetaGofStatistic(values)
    generator = get_hypothesis_generator(statistic)
    assert generator.parameters() == values.as_dict()
    actual = generator.generate(32, random_state=np.random.default_rng(42))
    expected = stats.beta.rvs(2, 5, size=32, random_state=np.random.default_rng(42))
    np.testing.assert_allclose(actual, expected)


def test_beta_family_does_not_require_fixed_shapes():
    from pysatl_criterion.distribution import BetaDistributionDescriptor as Beta
    from pysatl_criterion.distribution.parameters import HypothesisSupport

    class FixedAlphaStatistic(beta.AbstractBetaGofStatistic):
        """Test double for a criterion declaring only alpha as fixed."""

        @classmethod
        def supported_hypotheses(cls):
            return (HypothesisSupport(Beta.DEFAULT, frozenset({Beta.ALPHA})),)

        @staticmethod
        def short_code():
            return "FIXED_ALPHA_TEST_DOUBLE"

        def alternative(self):
            return RightAlternative()

        def execute_statistic(self, rvs):
            raise NotImplementedError("This test double checks only hypothesis construction")

    values = Beta.DEFAULT.parse({Beta.ALPHA: 2})
    statistic = FixedAlphaStatistic(values)
    assert statistic.hypothesis().parameter_values == values
    assert statistic.hypothesis().parameter_values[Beta.ALPHA] == 2
    for fixed in ({}, {Beta.BETA: 5}, {Beta.ALPHA: 2, Beta.BETA: 5}):
        with pytest.raises(ValueError, match="Unsupported"):
            FixedAlphaStatistic(Beta.DEFAULT.parse(fixed))


@pytest.mark.parametrize("cls", STATISTICS)
def test_beta_hierarchy_reflects_fixed_parameter_contract(cls):
    assert issubclass(cls, beta.AbstractBetaGofStatistic)
    fitted = cls is beta.EbnerLiebenbergBetaGofStatistic
    assert issubclass(cls, beta.AbstractSpecifiedBetaGofStatistic) is not fitted
    if fitted:
        assert dict(cls(Beta.DEFAULT.parse({})).hypothesis().parameter_values) == {}
        with pytest.raises(TypeError):
            cls(alpha=2, beta=5)
        with pytest.raises(TypeError):
            cls(2, 5)


@pytest.mark.parametrize("cls", STATISTICS)
def test_direct_constructor_enforces_the_schema_contract(cls):
    from dataclasses import replace

    fitted = cls is beta.EbnerLiebenbergBetaGofStatistic
    values = Beta.DEFAULT.parse({} if fitted else {"a": 2, "b": 5})
    statistic = cls(values)
    assert statistic.hypothesis().parameter_values is values
    for invalid in (
        Beta.DEFAULT.parse({"a": 2}),
        replace(Beta.DEFAULT, id="beta.other").parse(values.as_dict()),
    ):
        with pytest.raises(ValueError, match="Unsupported"):
            cls(invalid)
    with pytest.raises(TypeError):
        cls()
    with pytest.raises(TypeError, match="ParameterValues"):
        cls(values.as_dict())


def test_beta_schema_renaming_does_not_change_formulas(monkeypatch):
    from dataclasses import replace

    values = Beta.DEFAULT.parse({"a": 2, "b": 5})
    sample = [0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57]
    expected = beta.KolmogorovSmirnovBetaGofStatistic(values).execute_statistic(sample)
    alpha = replace(Beta.ALPHA, name="alpha", aliases=("a",))
    second = replace(Beta.BETA, name="beta", aliases=("b",))
    schema = replace(Beta.DEFAULT, parameters=(alpha, second))
    monkeypatch.setattr(Beta, "ALPHA", alpha)
    monkeypatch.setattr(Beta, "BETA", second)
    monkeypatch.setattr(Beta, "DEFAULT", schema)
    renamed = schema.parse({"alpha": 2, "beta": 5})
    statistic = beta.KolmogorovSmirnovBetaGofStatistic(renamed)
    assert renamed.storage_identity() == values.storage_identity()
    assert statistic.execute_statistic(sample) == expected
    with pytest.raises(AttributeError):
        statistic.alpha = 3
