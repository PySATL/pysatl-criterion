"""Documentation, static typing, public exports and opt-in simulation checks."""

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import pysatl_criterion as public_api
from pysatl_criterion import multiple_testing as typed_api
from pysatl_criterion.api import testing as testing_api
from pysatl_criterion.multiple_testing import (
    BenjaminiHochberg,
    BenjaminiKriegerYekutieli,
    BenjaminiYekutieli,
    Bonferroni,
    Holm,
    PValues,
)


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("method", tuple(typed_api.P_VALUE_PROCEDURES.values()))
def test_public_exports(method):
    assert getattr(public_api, method.__name__) is method
    assert getattr(testing_api, method.__name__) is method
    result = getattr(public_api, method.__name__)().test(PValues(np.array([0.0, 1.0])))
    assert isinstance(result, public_api.MultipleTestingResult)
    assert result.rejected.tolist() == [True, False]


def test_static_input_contracts(tmp_path):
    """Mypy must reject incorrect inputs even through generic annotations."""
    source = """
import numpy as np
from pysatl_criterion.multiple_testing import (
    AbstractMultipleTesting, AbstractResamplingMultipleTesting, Holm,
    PValues, ResamplingStatistics, BenjaminiKriegerYekutieli,
    WestfallYoungMaxT, WestfallYoungMinP, JointResamplingAssumptions,
)
p = PValues(np.array([0.1]))
r = ResamplingStatistics(np.array([1.]), np.array([[2.]]))
procedure: AbstractMultipleTesting[PValues] = Holm()
procedure.test(p)
procedure.test(r)
a = JointResamplingAssumptions(True, True)
WestfallYoungMaxT(a).test(p)
WestfallYoungMinP(a).test(r)
Holm().test(np.array([0.1]))
BenjaminiKriegerYekutieli().adjust(np.array([0.1]))
def resampling(procedure: AbstractResamplingMultipleTesting) -> None:
    procedure.test(r)
    procedure.test(p)
"""
    path = tmp_path / "typing_contract.py"
    path.write_text(source)
    check = subprocess.run(
        [sys.executable, "-m", "mypy", "--follow-imports=silent", str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert check.returncode == 1, check.stdout + check.stderr
    assert check.stdout.count("[arg-type]") == 5, check.stdout
    assert check.stdout.count("[attr-defined]") == 1, check.stdout


@pytest.mark.parametrize("path", sorted((ROOT / "docs/multiple_testing").glob("*.md")))
def test_documentation_examples_and_links(path):
    text = path.read_text()
    for snippet in re.findall(r"```python\n(.*?)```", text, re.DOTALL):
        exec(compile(snippet, str(path), "exec"), {})  # noqa: S102 - repository examples
    for target in re.findall(r"\]\(([^)]+)\)", text):
        if "://" not in target and not target.startswith("#"):
            assert (path.parent / target.split("#")[0]).exists(), (path, target)


@pytest.mark.skipif(
    os.environ.get("PYSATL_MULTIPLE_TESTING_MONTE_CARLO") != "1",
    reason="Opt-in Monte Carlo validation; see docs/multiple_testing/index.md",
)
@pytest.mark.parametrize(
    "method", [Bonferroni, Holm, BenjaminiHochberg, BenjaminiYekutieli, BenjaminiKriegerYekutieli]
)
def test_monte_carlo_global_null_and_mixture(method):
    """Independent U(0,1) nulls: FDR=FWER under global null.

    5,000 repetitions, 20 hypotheses, fixed seed. A six-SE upper tolerance
    avoids fragile simulation tests and is validation, not proof of control.
    Mixture adds five false-null p-values fixed at zero and measures FDP.
    """
    rng = np.random.default_rng(62173)
    repetitions, m, alpha = 5000, 20, 0.05
    tolerance = 6 * np.sqrt(alpha * (1 - alpha) / repetitions)
    for false_nulls in (0, 5):
        p = rng.uniform(size=(repetitions, m))
        p[:, :false_nulls] = 0
        errors = []
        for family in p:
            rejected = method().test(PValues(family), alpha).rejected
            v = np.count_nonzero(rejected[false_nulls:])
            errors.append(
                float(v > 0)
                if method.metadata.error_criterion == "FWER"
                else v / max(1, np.count_nonzero(rejected))
            )
        rate = float(np.mean(errors))
        print(f"{method.__name__} false_nulls={false_nulls}: {rate:.5f}")
        assert rate <= alpha + tolerance
