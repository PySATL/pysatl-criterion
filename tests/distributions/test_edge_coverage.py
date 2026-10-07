from types import SimpleNamespace

import numpy as np
import pytest

from pysatl_criterion.distribution.distributions import BetaDistributionDescriptor as Beta
from pysatl_criterion.distribution.distributions import (
    InverseGammaDistributionDescriptor,
    LogLogisticDistributionDescriptor,
    LogNormalDistributionDescriptor,
    NormalDistributionDescriptor,
    ParetoDistributionDescriptor,
    UniformDistributionDescriptor,
)
from pysatl_criterion.distribution.distributions import WeibullDistributionDescriptor as Weibull
from pysatl_criterion.statistics.goodness_of_fit import (
    beta,
    inverse_gamma,
    log_logistic,
    log_normal,
    normal,
    pareto,
    uniform,
    weibull,
)
from tests.parameter_cases import parameters_for


def test_distribution_hypotheses_and_parameter_validation():
    statistics = [
        beta.KolmogorovSmirnovBetaGofStatistic(Beta.DEFAULT.parse({"a": 1, "b": 1})),
        inverse_gamma.KolmogorovSmirnovInverseGammaGofStatistic(
            InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": 1, "beta": 1})
        ),
        log_logistic.KolmogorovSmirnovLogLogisticGofStatistic(
            LogLogisticDistributionDescriptor.DEFAULT.parse({"alpha": 1, "beta": 1})
        ),
        log_normal.KolmogorovSmirnovLogNormalGofStatistic(
            LogNormalDistributionDescriptor.SHAPE_SCALE.parse({"s": 1, "scale": 1})
        ),
        pareto.KolmogorovSmirnovParetoGofStatistic(
            ParetoDistributionDescriptor.DEFAULT.parse({"shape": 1, "scale": 1})
        ),
        uniform.KolmogorovSmirnovUniformGofStatistic(
            UniformDistributionDescriptor.DEFAULT.parse({"a": 0, "b": 1})
        ),
        weibull.LillieforsWeibullGofStatistic(Weibull.DEFAULT.parse({})),
    ]
    for statistic in statistics:
        assert statistic.supports_hypothesis(statistic.hypothesis())

    for kwargs in ({"alpha": 0}, {"beta": 0}):
        with pytest.raises(ValueError):
            beta.KolmogorovSmirnovBetaGofStatistic(
                Beta.DEFAULT.parse(
                    {{"alpha": "a", "beta": "b"}[key]: value for key, value in kwargs.items()}
                )
            )
        with pytest.raises(ValueError):
            log_logistic.KolmogorovSmirnovLogLogisticGofStatistic(
                LogLogisticDistributionDescriptor.DEFAULT.parse({"alpha": 1, "beta": 1, **kwargs})
            )
    for kwargs in ({"s": 0}, {"scale": 0}):
        with pytest.raises(ValueError):
            log_normal.KolmogorovSmirnovLogNormalGofStatistic(
                LogNormalDistributionDescriptor.SHAPE_SCALE.parse({"s": 1, "scale": 1, **kwargs})
            )
    with pytest.raises(ValueError, match="observations are required"):
        inverse_gamma.MinToshiyukiInverseGammaGofStatistic(
            InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": 1, "beta": 1})
        ).execute_statistic([])


def test_beta_ratio_zero_mean_is_rejected():
    with pytest.raises(ValueError, match="Arithmetic mean"):
        beta.RatioBetaGofStatistic(Beta.DEFAULT.parse({"a": 1, "b": 1})).execute_statistic(
            [0.0, 0.0]
        )


def test_log_logistic_censoring_and_failed_optimization(monkeypatch):
    statistic = log_logistic.NikulinLogLogisticGofStatistic(
        LogLogisticDistributionDescriptor.DEFAULT.parse({})
    )
    times, events = log_logistic._survival_data([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(events, np.ones_like(times))
    with pytest.raises(ValueError, match="three events"):
        statistic.execute_statistic((times, [1, 0, 1]))
    monkeypatch.setattr(
        log_logistic,
        "minimize",
        lambda *_args, **_kwargs: SimpleNamespace(fun=np.inf, x=[0, 0], jac=[np.inf, np.inf]),
    )
    with pytest.raises(RuntimeError, match="did not converge"):
        statistic.execute_statistic(times)
    with pytest.raises(ValueError, match="Singular"):
        log_logistic._solve_positive(np.zeros((2, 2)), np.ones(2))


@pytest.mark.parametrize("roundoff", [1e-12, 1e-3])
def test_mirvaliev_only_clamps_numerical_roundoff(monkeypatch, roundoff):
    statistic = log_logistic.MirvalievLogLogisticGofStatistic(
        LogLogisticDistributionDescriptor.DEFAULT.parse({}), n_intervals=3
    )
    sample = np.exp(np.random.default_rng(123).logistic(size=100))
    baseline = statistic.execute_statistic(sample)
    solve = log_logistic._solve_positive
    calls = 0

    def perturbed_solve(matrix, rhs):
        nonlocal calls
        calls += 1
        result = solve(matrix, rhs)
        if calls == 2:
            # Perturb the final correction so the quadratic form is -roundoff.
            result = result + (baseline + roundoff) * rhs / (rhs @ rhs)
        return result

    monkeypatch.setattr(log_logistic, "_solve_positive", perturbed_solve)
    if roundoff < 1e-10:
        assert statistic.execute_statistic(sample) == 0
    else:
        with pytest.raises(RuntimeError, match="negative Mirvaliev"):
            statistic.execute_statistic(sample)


def test_log_normal_critical_value_edges():
    supremum = log_normal.KLSupremumLogNormalGoFStatistic(
        LogNormalDistributionDescriptor.SHAPE_SCALE.parse({"s": 1, "scale": 1})
    )
    integral = log_normal.KLIntegralLogNormalGoFStatistic(
        LogNormalDistributionDescriptor.SHAPE_SCALE.parse({"s": 1, "scale": 1})
    )
    assert np.isfinite(supremum.calculate_critical_value(20, 0.01))
    assert np.isfinite(supremum.calculate_critical_value(20, 0.02))
    assert np.isfinite(integral.calculate_critical_value(20, 0.01))
    assert np.isfinite(integral.calculate_critical_value(20, 0.02))
    with pytest.raises(ValueError):
        supremum.calculate_critical_value(20, 0.5)
    with pytest.raises(ValueError):
        integral.calculate_critical_value(20, 0.5)


def test_log_normal_small_r_and_empty_grid_fallbacks(monkeypatch):
    statistic = log_normal.KLSupremumLogNormalGoFStatistic(
        LogNormalDistributionDescriptor.SHAPE_SCALE.parse({"s": 1, "scale": 1})
    )
    original_linspace = log_normal.np.linspace
    monkeypatch.setattr(log_normal.np, "linspace", lambda *_args, **_kwargs: np.array([0.0, 1e-7]))
    assert np.isfinite(statistic.execute_statistic([1.0, 1.2, 1.5, 2.0]))
    monkeypatch.setattr(log_normal.np, "linspace", lambda *_args, **_kwargs: np.array([]))
    assert statistic.execute_statistic([1.0, 1.2, 1.5, 2.0]) == np.inf
    monkeypatch.setattr(log_normal.np, "linspace", original_linspace)


def test_normality_validation_and_helper_branches():
    assert normal.KolmogorovSmirnovNormalityGofStatistic(
        NormalDistributionDescriptor.DEFAULT.parse({"mean": 0, "var": 1})
    ).hypothesis().parameters() == {
        "mean": 0,
        "var": 1,
    }
    with pytest.raises(ValueError, match="observation"):
        normal.JBNormalityGofStatistic(
            NormalDistributionDescriptor.DEFAULT.parse({})
        ).execute_statistic([])
    with pytest.raises(ValueError, match="at least 8"):
        normal.SkewNormalityGofStatistic(
            NormalDistributionDescriptor.DEFAULT.parse({})
        ).execute_statistic([1, 2, 3])
    with pytest.raises(ValueError, match="at least 5"):
        normal.KurtosisNormalityGofStatistic(
            NormalDistributionDescriptor.DEFAULT.parse({})
        ).execute_statistic([1, 2, 3])

    repeated = [1.0, 1.0, 2.0, 3.0, 4.0]
    ordered = normal.LooneyGulledgeNormalityGofStatistic._normal_order_statistic(
        repeated, weighted=True
    )
    assert len(ordered) == len(repeated)
    result = normal.RyanJoinerNormalityGofStatistic(
        NormalDistributionDescriptor.DEFAULT.parse({}), weighted=True
    ).execute_statistic(repeated)
    assert np.isfinite(result)
    normal.RyanJoinerNormalityGofStatistic._order_statistic(5, "1/2")
    normal.RyanJoinerNormalityGofStatistic._order_statistic(5, "0")

    coin = normal.CoinNormalityGofStatistic(NormalDistributionDescriptor.DEFAULT.parse({}))
    assert coin.correct(1, 4) == pytest.approx(1.9e-5)
    assert coin.correct(0, 10) == 0
    assert coin.correct(1, 21) == 0
    assert coin.correct(4, 41) == 0
    with pytest.raises(ValueError, match="n2"):
        coin.nscor2([0.0] * 5, 4, [3])
    with pytest.raises(ValueError, match="n<=1"):
        coin.nscor2([0.0] * 5, 1, [0])
    coin.nscor2([0.0] * 5, 2, [1])
    with pytest.raises(ValueError, match="n <= 2000"):
        coin.nscor2([0.0] * 5, 2001, [1])

    for size in (30, 60):
        values = np.random.default_rng(size).normal(size=size)
        for statistic_class in (
            normal.Hosking1NormalityGofStatistic,
            normal.Hosking2NormalityGofStatistic,
            normal.Hosking3NormalityGofStatistic,
            normal.Hosking4NormalityGofStatistic,
        ):
            assert np.isfinite(
                statistic_class(parameters_for(statistic_class)).execute_statistic(values)
            )
    with pytest.raises(ValueError, match="at least 6"):
        normal.Hosking2NormalityGofStatistic(
            NormalDistributionDescriptor.DEFAULT.parse({})
        ).execute_statistic([1, 2, 3])

    assert np.isfinite(
        normal.SpiegelhalterNormalityGofStatistic(
            NormalDistributionDescriptor.DEFAULT.parse({})
        ).execute_statistic(np.random.default_rng(150).normal(size=150))
    )


def test_bhs_small_sample_and_helper_paths(monkeypatch):
    statistic = normal.BHSNormalityGofStatistic(NormalDistributionDescriptor.DEFAULT.parse({}))
    monkeypatch.setattr(statistic, "mc_c_d", lambda *_args: 0.0)
    assert np.isfinite(statistic.stat16([1, 2, 3, 4, 5]))
    monkeypatch.undo()

    # Exact medcouple values, including the antisymmetric median-tie convention.
    eps = [2.220446e-16, 2.225074e-308]
    assert statistic.mc_c_d([1.0, 1.0, 2.0], eps, [10, 0]) == 0.5
    assert statistic.mc_c_d([1.0, 2.0, 2.0], eps, [10, 0]) == -0.5
    assert statistic.mc_c_d([1.0, 2.0, 3.0], eps, [10, 4]) == 0
    assert statistic.mc_c_d([1.0, 2.0, 4.0], eps, [10, 3]) == pytest.approx(1 / 6)

    assert statistic.h_kern(1, 2, 1, 1, 3, 0) == 1
    assert np.isfinite(statistic.h_kern(2, -1, 1, 1, 3, 0))
    assert statistic.whi_med_i([1.0, 2.0, 3.0], [10, 1, 1], 3, [0.0] * 3, [0.0] * 3, [0] * 3) == 1


def test_empty_graph_independence_statistic():
    with pytest.raises(ValueError, match="at least 2"):
        normal.GraphIndependenceNumberNormalityGofStatistic(
            NormalDistributionDescriptor.DEFAULT.parse({"var": 1})
        ).execute_statistic([])


def test_pareto_kl_spacing_fallbacks():
    statistic = pareto.LequesneKlParetoGofStatistic(ParetoDistributionDescriptor.DEFAULT.parse({}))
    with pytest.raises(ValueError, match="At least 4"):
        statistic.execute_statistic([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="nonconstant"):
        statistic.execute_statistic([1.0, 1.0, 1.0, 1.0], m=10)


def test_weibull_family_base_paths():
    values = np.linspace(0.5, 3.0, 10)
    assert weibull.WPPWeibullGofStatistic.alternative(object())
    assert weibull.LaplaceTransformWeibullGofStatistic.alternative(object())
    with pytest.raises(ValueError, match="positive sample"):
        weibull.WPPWeibullGofStatistic.MLEst(np.array([0.0, 1.0]))
    estimates = weibull.WPPWeibullGofStatistic.MLEst(values)
    assert estimates["beta"] > 0
    assert np.all(np.diff(estimates["y"]) > 0)
    # Family helpers remain abstract: concrete classes implement the common API.
    with pytest.raises(TypeError):
        weibull.WPPWeibullGofStatistic(Weibull.DEFAULT.parse({}))


def test_uniform_constant_sample_bins_and_fully_censored_sample():
    chi = uniform.Chi2PearsonUniformGofStatistic(
        UniformDistributionDescriptor.DEFAULT.parse({"a": 0, "b": 1}), bins="auto"
    )
    assert np.isfinite(chi.execute_statistic(np.full(10, 0.5)))
    statistic = uniform.CensoredSteinUniformGofStatistic(
        UniformDistributionDescriptor.DEFAULT.parse({"a": 1, "b": 2})
    )
    assert statistic.execute_statistic(np.linspace(1.1, 1.9, 5), np.ones(5)) == 0
