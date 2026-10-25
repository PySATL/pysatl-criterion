"""Seeded statistical validation; opt-in via PYSATL_STATISTICAL=1.

Also executable: python -m tests.statistical.test_exponential_validation
Writes docs/exponential-monte-carlo.json. Simulations use production statistics;
null laws/critical values are independent theoretical or published references.
Calibration and validation samples are disjoint. Tolerances allow MC error and
finite-n asymptotic error; power is a diagnostic, never a proof of consistency.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from scipy import special, stats

from pysatl_criterion.distribution import ExponentialDistributionDescriptor as E
from pysatl_criterion.statistics.alternative import AlternativeType
from pysatl_criterion.statistics.goodness_of_fit.continuous import exponent as exp


pytestmark = pytest.mark.skipif(
    os.environ.get("PYSATL_STATISTICAL") != "1",
    reason="Set PYSATL_STATISTICAL=1 for statistical validation",
)


def make(cls, **kwargs):
    return cls(E.DEFAULT.parse({}), **kwargs)


def validate_null():
    rng = np.random.default_rng(20261008)
    count, n = 5000, 500
    specs = [
        ("EP", make(exp.EppsPulleyExponentialityGofStatistic), lambda t: t),
        ("CO", make(exp.CoxOakesExponentialityGofStatistic), lambda t: t * np.sqrt(6 / n) / np.pi),
        (
            "MN",
            make(exp.MoranExponentialityGofStatistic),
            lambda t: (
                (t - (special.digamma(1) - special.digamma(n) + np.log(n) + np.euler_gamma))
                / np.sqrt(special.polygamma(1, 1) / n - special.polygamma(1, n))
            ),
        ),
        (
            "GREENWOOD",
            make(exp.GreenwoodExponentialityGofStatistic),
            lambda t: (t - 2 / (n + 1)) / np.sqrt(4 * (n - 1) / ((n + 1) ** 2 * (n + 2) * (n + 3))),
        ),
        (
            "GINI",
            make(exp.GiniExponentialityGofStatistic),
            lambda t: (t - 0.5) * np.sqrt(12 * (n - 1)),
        ),
    ]
    report = {}
    for name, obj, transform in specs:
        values = np.array(
            [transform(obj.execute_statistic(rng.exponential(size=n))) for _ in range(count)]
        )
        level = float(np.mean(abs(values) > stats.norm.ppf(0.975)))
        report[name] = {
            "n": n,
            "replications": count,
            "mean": float(values.mean()),
            "variance": float(values.var()),
            "type1": level,
        }
        assert abs(values.mean()) < 0.12, report[name]
        assert 0.80 < values.var() < 1.20, report[name]
        assert 0.03 < level < 0.075, report[name]
    # Exact F laws follow from independent normalized exponential order spacings.
    for cls, df in [
        (exp.GnedenkoExponentialityGofStatistic, (8, 32)),
        (exp.HarrisExponentialityGofStatistic, (16, 24)),
    ]:
        obj = make(cls, r=4)
        values = [obj.execute_statistic(rng.exponential(size=20)) for _ in range(count)]
        level = float(np.mean(np.array(values) > stats.f.ppf(0.95, *df)))
        assert 0.035 < level < 0.065
        report[obj.short_code()] = {"n": 20, "replications": count, "type1": level, "law": f"F{df}"}
    # BH (1991), Table 2: n=20, a=1, 95th percentile .303.
    # Ebner (2023), Table 1: a=1, limiting 95th percentile approximately .463.
    for cls, size, critical in [
        (exp.BaringhausHenzeELTExponentialityGofStatistic, 20, 0.303),
        (exp.BaringhausHenzeBHCExponentialityGofStatistic, 300, 0.463),
    ]:
        obj = make(cls)
        values = np.array([obj.execute_statistic(rng.exponential(size=size)) for _ in range(3000)])
        level = float(np.mean(values > critical))
        report[obj.short_code()] = {
            "n": size,
            "replications": 3000,
            "critical": critical,
            "type1": level,
            "mean": float(values.mean()),
        }
        assert 0.03 < level < 0.075, report[obj.short_code()]
        if cls is exp.BaringhausHenzeBHCExponentialityGofStatistic:
            assert abs(values.mean() - 1 / 6) < 0.015
    # Fully specified Pearson (k=5) -> chi-square_4, KS -> exact kstwo.
    for cls, critical in [
        (exp.Chi2PearsonExponentialityGofStatistic, stats.chi2.ppf(0.95, 4)),
        (exp.KolmogorovSmirnovExponentialityGofStatistic, stats.kstwo.ppf(0.95, 200)),
    ]:
        from pysatl_criterion.statistics.goodness_of_fit.binning import EqualProbabilityBins

        settings = (
            {"binning": EqualProbabilityBins(count=5)}
            if cls is exp.Chi2PearsonExponentialityGofStatistic
            else {}
        )
        obj = cls(E.DEFAULT.parse({"lam": 3}), **settings)
        values = [obj.execute_statistic(rng.exponential(1 / 3, 200)) for _ in range(count)]
        level = float(np.mean(np.array(values) > critical))
        assert 0.035 < level < 0.065
        report[obj.short_code() + "_fixed"] = {"n": 200, "replications": count, "type1": level}
    return report


def validate_power():
    rng = np.random.default_rng(19461991)
    classes = [
        exp.LillieforsExponentialityGofStatistic,
        exp.CramerVonMisesExponentialityGofStatistic,
        exp.CoxOakesExponentialityGofStatistic,
        exp.EppsPulleyExponentialityGofStatistic,
        exp.MoranExponentialityGofStatistic,
        exp.GreenwoodExponentialityGofStatistic,
        exp.AndersonDarlingExponentialityGofStatistic,
        exp.WatsonExponentialityGofStatistic,
        exp.KuiperExponentialityGofStatistic,
        exp.Chi2PearsonExponentialityGofStatistic,
        exp.BaringhausHenzeELTExponentialityGofStatistic,
        exp.BaringhausHenzeBHCExponentialityGofStatistic,
    ]
    report = {}
    for cls in classes:
        obj = make(cls)
        two = obj.alternative().type() == AlternativeType.TWO_TAILED
        by_size = {}
        for n in (20, 80):
            null = np.array([obj.execute_statistic(rng.exponential(size=n)) for _ in range(1800)])
            lo, hi = np.quantile(null, [0.025, 0.975] if two else [0, 0.95])
            samplers = {
                "null": lambda n=n: rng.exponential(size=n),
                "weibull_0.6": lambda n=n: rng.weibull(0.6, n),
                "weibull_1.8": lambda n=n: rng.weibull(1.8, n),
                "gamma_2": lambda n=n: rng.gamma(2, size=n),
                "lognormal": lambda n=n: rng.lognormal(sigma=0.6, size=n),
                "uniform": lambda n=n: rng.uniform(size=n),
            }
            rates = {}
            for name, sampler in samplers.items():
                values = np.array([obj.execute_statistic(sampler()) for _ in range(600)])
                rates[name] = float(np.mean((values > hi) | ((values < lo) if two else False)))
            # Two independent MC experiments; allow error in both critical value and validation.
            assert 0.015 < rates["null"] < 0.095, (obj.short_code(), n, rates)
            by_size[n] = rates
        for name in ("weibull_0.6", "weibull_1.8", "gamma_2", "lognormal", "uniform"):
            assert by_size[80][name] > by_size[20][name] - 0.06, (obj.short_code(), name, by_size)
        assert by_size[80]["weibull_1.8"] > 0.5, (obj.short_code(), by_size)
        report[obj.short_code()] = by_size
    return report


def validate_legacy_calibration():
    """Level diagnostic for every preexisting class, not proof of its formula."""
    import inspect

    rng = np.random.default_rng(31082026)
    report = {}
    classes = sorted(
        (
            cls
            for cls in vars(exp).values()
            if inspect.isclass(cls)
            and cls.__module__ == exp.__name__
            and not inspect.isabstract(cls)
            and not issubclass(
                cls,
                (
                    exp.AbstractExponentialEDFStatistic,
                    exp.AbstractExponentialFrequencyStatistic,
                    exp.AbstractTunedExponentialStatistic,
                ),
            )
            and cls is not exp.GreenwoodExponentialityGofStatistic
        ),
        key=lambda cls: cls.__name__,
    )
    for cls in classes:
        parameters = E.DEFAULT.parse(
            {"lam": 1} if cls is exp.KolmogorovSmirnovExponentialityGofStatistic else {}
        )
        obj = cls(parameters)
        null = np.array([obj.execute_statistic(rng.exponential(size=12)) for _ in range(1200)])
        values = np.array([obj.execute_statistic(rng.exponential(size=12)) for _ in range(600)])
        two = obj.alternative().type() == AlternativeType.TWO_TAILED
        lo, hi = np.quantile(null, [0.025, 0.975] if two else [0, 0.95])
        level = float(np.mean((values > hi) | ((values < lo) if two else False)))
        # Strict quantile decisions can be conservative for discrete graph statistics.
        assert level < 0.10, (obj.short_code(), level)
        report[obj.short_code()] = {
            "n": 12,
            "calibration_replications": 1200,
            "validation_replications": 600,
            "type1": level,
            "unique_null_statistics": len(np.unique(null)),
        }
    return report


def test_legacy_calibration():
    validate_legacy_calibration()


def test_null_laws():
    validate_null()


def test_level_and_power():
    validate_power()


if __name__ == "__main__":
    results = {
        "seed_null": 20261008,
        "seed_power": 19461991,
        "null": validate_null(),
        "power": validate_power(),
        "legacy_calibration": validate_legacy_calibration(),
        "calibration_replications": 1800,
        "power_replications": 600,
    }
    Path("docs/exponential-monte-carlo.json").write_text(json.dumps(results, indent=2) + "\n")
    print("Statistical validation passed; docs/exponential-monte-carlo.json written.")
