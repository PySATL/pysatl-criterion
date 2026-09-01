from types import SimpleNamespace

import numpy as np
import pytest

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


def test_distribution_hypotheses_and_parameter_validation():
    statistics = [
        beta.KolmogorovSmirnovBetaGofStatistic(),
        inverse_gamma.KolmogorovSmirnovInverseGammaGofStatistic(),
        log_logistic.KolmogorovSmirnovLogLogisticGofStatistic(),
        log_normal.KolmogorovSmirnovLogNormalGofStatistic(),
        pareto.KolmogorovSmirnovParetoGofStatistic(),
        uniform.KolmogorovSmirnovUniformGofStatistic(),
        weibull.KolmogorovSmirnovWeibullGofStatistic(),
    ]
    for statistic in statistics:
        assert statistic.hypothesis().parameters()

    for kwargs in ({"alpha": 0}, {"beta": 0}):
        with pytest.raises(ValueError):
            beta.KolmogorovSmirnovBetaGofStatistic(**kwargs)
        with pytest.raises(ValueError):
            log_logistic.KolmogorovSmirnovLogLogisticGofStatistic(**kwargs)
    for kwargs in ({"s": 0}, {"scale": 0}):
        with pytest.raises(ValueError):
            log_normal.KolmogorovSmirnovLogNormalGofStatistic(**kwargs)
    with pytest.raises(ValueError, match="At least one observation"):
        inverse_gamma.MinToshiyukiInverseGammaGofStatistic().execute_statistic([])


def test_beta_ratio_zero_mean_is_rejected():
    with pytest.raises(ValueError, match="Arithmetic mean"):
        beta.RatioBetaGofStatistic().execute_statistic([0.0, 0.0])


def test_log_logistic_censoring_and_optimization_fallbacks(monkeypatch):
    statistic = log_logistic.NikulinLogLogisticGofStatistic()
    times, censored = statistic._extract_data([1.0, 2.0])
    assert np.array_equal(censored, np.ones_like(times))
    assert statistic._neg_log_likelihood(np.array([-1.0, 1.0]), times, censored) == 1e10
    assert statistic._get_initial_guesses(times, np.zeros_like(times))[0][1] == 1

    monkeypatch.setattr(
        log_logistic,
        "minimize",
        lambda *_args, **_kwargs: SimpleNamespace(success=False, message="failed"),
    )
    with pytest.raises(RuntimeError, match="failed"):
        statistic._fit_mle(times, censored)

    def raise_singular(_matrix):
        raise np.linalg.LinAlgError

    monkeypatch.setattr(log_logistic.np.linalg, "inv", raise_singular)
    assert np.isfinite(
        statistic._compute_qf(
            np.array([1.0, 2.0]),
            np.array([1, 0]),
            np.ones(5),
            np.ones(5),
            np.array([0.0, 0.5, 1.0, 1.5, 2.0, np.inf]),
            1.0,
            1.0,
        )
    )
    matrices = log_logistic.MirvalievLogLogisticGofStatistic()._compute_mirvaliev_matrices(
        np.array([1.0, 2.0]),
        np.array([1, 0]),
        np.ones(8),
        np.ones(8),
        np.linspace(0.1, 2.0, 9),
        1.0,
        1.0,
    )
    assert len(matrices) == 5


def test_mirvaliev_roundoff_is_clamped(monkeypatch):
    statistic = log_logistic.MirvalievLogLogisticGofStatistic(n_intervals=1)
    monkeypatch.setattr(statistic, "_extract_data", lambda _rvs: (np.array([1.0]), np.array([1])))
    monkeypatch.setattr(statistic, "_fit_mle", lambda *_args: (1.0, 1.0))
    monkeypatch.setattr(statistic, "_build_intervals", lambda *_args: np.array([0.0, np.inf]))
    monkeypatch.setattr(statistic, "_compute_frequencies", lambda *_args: (np.ones(1), np.ones(1)))
    c_value = np.sqrt(1.0 + 1e-9)
    monkeypatch.setattr(
        statistic,
        "_compute_mirvaliev_matrices",
        lambda *_args: (
            np.ones(1),
            np.ones((1, 1)),
            np.array([[c_value]]),
            np.ones((1, 1)),
            np.ones((1, 1)),
        ),
    )
    assert statistic.execute_statistic([1.0]) == pytest.approx(1.0 + 1e-9)


def test_log_normal_critical_value_edges():
    supremum = log_normal.KLSupremumLogNormalGoFStatistic()
    integral = log_normal.KLIntegralLogNormalGoFStatistic()
    assert np.isfinite(supremum.calculate_critical_value(20, 0.01))
    assert np.isfinite(supremum.calculate_critical_value(20, 0.02))
    assert np.isfinite(integral.calculate_critical_value(20, 0.01))
    assert np.isfinite(integral.calculate_critical_value(20, 0.02))
    with pytest.raises(ValueError):
        supremum.calculate_critical_value(20, 0.5)
    with pytest.raises(ValueError):
        integral.calculate_critical_value(20, 0.5)


def test_log_normal_small_r_and_empty_grid_fallbacks(monkeypatch):
    statistic = log_normal.KLSupremumLogNormalGoFStatistic()
    original_linspace = log_normal.np.linspace
    monkeypatch.setattr(log_normal.np, "linspace", lambda *_args, **_kwargs: np.array([0.0, 1e-7]))
    assert np.isfinite(statistic.execute_statistic([1.0, 1.2, 1.5, 2.0]))
    monkeypatch.setattr(log_normal.np, "linspace", lambda *_args, **_kwargs: np.array([]))
    assert statistic.execute_statistic([1.0, 1.2, 1.5, 2.0]) == np.inf
    monkeypatch.setattr(log_normal.np, "linspace", original_linspace)


def test_normality_validation_and_helper_branches(capsys):
    assert normal.KolmogorovSmirnovNormalityGofStatistic().hypothesis().parameters() == [0, 1]
    with pytest.raises(ValueError, match="observation"):
        normal.JBNormalityGofStatistic().execute_statistic([])
    with pytest.raises(ValueError, match="less than 8"):
        normal.SkewNormalityGofStatistic().execute_statistic([1, 2, 3])
    with pytest.raises(ValueError, match="at least 5"):
        normal.KurtosisNormalityGofStatistic().execute_statistic([1, 2, 3])

    repeated = [1.0, 1.0, 2.0, 3.0, 4.0]
    ordered = normal.LooneyGulledgeNormalityGofStatistic._normal_order_statistic(
        repeated, weighted=True
    )
    assert len(ordered) == len(repeated)
    result = normal.RyanJoinerNormalityGofStatistic(weighted=True).execute_statistic(repeated)
    assert np.isfinite(result)
    normal.RyanJoinerNormalityGofStatistic._order_statistic(5, "1/2")
    normal.RyanJoinerNormalityGofStatistic._order_statistic(5, "0")

    coin = normal.CoinNormalityGofStatistic()
    assert coin.correct(1, 4) == pytest.approx(1.9e-5)
    assert coin.correct(0, 10) == 0
    assert coin.correct(1, 21) == 0
    assert coin.correct(4, 41) == 0
    with pytest.raises(ValueError, match="n2"):
        coin.nscor2([0.0] * 5, 4, [3])
    with pytest.raises(ValueError, match="n<=1"):
        coin.nscor2([0.0] * 5, 1, [0])
    coin.nscor2([0.0] * 5, 2, [1])
    coin.nscor2([0.0] * 5, 2001, [1])
    assert "inaccurate" in capsys.readouterr().out

    for size in (30, 60):
        values = np.random.default_rng(size).normal(size=size)
        for statistic_class in (
            normal.Hosking1NormalityGofStatistic,
            normal.Hosking2NormalityGofStatistic,
            normal.Hosking3NormalityGofStatistic,
            normal.Hosking4NormalityGofStatistic,
        ):
            assert np.isfinite(statistic_class().execute_statistic(values))
    assert normal.Hosking2NormalityGofStatistic().execute_statistic([1, 2, 3]) == 0

    assert np.isfinite(
        normal.SpiegelhalterNormalityGofStatistic().execute_statistic(
            np.random.default_rng(150).normal(size=150)
        )
    )


def test_bhs_small_sample_and_helper_paths(capsys, monkeypatch):
    statistic = normal.BHSNormalityGofStatistic()
    monkeypatch.setattr(statistic, "mc_c_d", lambda *_args: 0.0)
    assert np.isfinite(statistic.stat16([1, 2, 3, 4, 5]))
    monkeypatch.undo()

    # NOTE: ``mc_c_d``/``whi_med_i`` are an incomplete port of R robustbase's
    # medcouple and do not terminate for many multi-point samples (see the
    # skipped ``test_bhs_normality_criterion``). Only the small inputs that
    # provably reach an early return / iteration cap are exercised here.
    eps = [2.220446e-16, 2.225074e-308]
    assert statistic.mc_c_d([1.0, 1.0, 2.0], eps, [10, 0]) == 1
    assert statistic.mc_c_d([1.0, 2.0, 2.0], eps, [10, 0]) == -1
    assert np.isfinite(statistic.mc_c_d([1.0, 2.0, 3.0], eps, [10, 4]))
    assert np.isfinite(statistic.mc_c_d([1.0, 2.0, 4.0], eps, [10, 3]))
    assert np.isfinite(statistic.mc_c_d([-np.inf, 0.0, np.inf], eps, [0, 1]))
    assert capsys.readouterr().out

    assert statistic.h_kern(1, 2, 1, 1, 3, 0) == 1
    assert np.isfinite(statistic.h_kern(2, -1, 1, 1, 3, 0))
    assert statistic.whi_med_i([1.0, 2.0, 3.0], [10, 1, 1], 3, [0.0] * 3, [0.0] * 3, [0] * 3) == 1


def test_empty_graph_independence_statistic():
    assert normal.GraphIndependenceNumberNormalityGofStatistic().execute_statistic([]) == 0


def test_pareto_kl_spacing_fallbacks():
    with pytest.raises(ValueError, match="n >= 4"):
        pareto.LequesneKlParetoGofStatistic.execute_statistic(object(), [1.0, 2.0, 3.0])
    assert np.isscalar(
        pareto.LequesneKlParetoGofStatistic.execute_statistic(object(), [1.0, 1.0, 1.0, 1.0], m=10)
    )


def test_weibull_family_base_paths(monkeypatch):
    values = np.linspace(0.5, 3.0, 10)
    assert weibull.WPPWeibullGofStatistic.alternative(object())
    assert weibull.LaplaceTransformWeibullGofStatistic.alternative(object())
    with pytest.raises(ValueError, match="positive sample"):
        weibull.WPPWeibullGofStatistic.MLEst(np.array([0.0, 1.0]))

    original_minimize = weibull.minimize_scalar

    def call_zero(function, *args, **kwargs):
        function(0, *kwargs.get("args", ()))
        return original_minimize(function, *args, **kwargs)

    monkeypatch.setattr(weibull, "minimize_scalar", call_zero)
    assert weibull.WPPWeibullGofStatistic.MLEst(values)["beta"] > 0

    for statistic_type in (
        weibull.RSBWeibullGofStatistic.code(),
        weibull.ST1WeibullGofStatistic.code(),
        weibull.ST2WeibullGofStatistic.code(),
    ):
        assert np.isfinite(weibull.WPPWeibullGofStatistic.execute_statistic(values, statistic_type))

    with pytest.raises(ValueError, match="type must"):
        weibull.LaplaceTransformWeibullGofStatistic.execute_statistic(object(), values, _type="bad")


def test_uniform_fallback_bins_and_fully_censored_sample():
    chi = uniform.Chi2PearsonUniformGofStatistic(bins="unknown")
    assert np.isfinite(chi.execute_statistic(np.linspace(0.1, 0.9, 10)))
    statistic = uniform.CensoredSteinUniformGofStatistic(a=1, b=2)
    assert statistic.execute_statistic(np.linspace(1.1, 1.9, 5), np.ones(5)) == 0
