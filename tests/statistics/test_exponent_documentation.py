"""Keep the public exponential catalog, pages and executable examples complete."""

import contextlib
import inspect
import io
import re
from pathlib import Path

import pytest

from pysatl_criterion.statistics.goodness_of_fit.continuous import exponent


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/criteria/goodness-of-fit/exponentiality"
CLASSES = {name for name in exponent.__all__ if not inspect.isabstract(getattr(exponent, name))}
PAGES = list(DOCS.glob("*_exp_criterion.md"))


def test_one_page_for_every_concrete_criterion():
    documented = []
    for path in PAGES:
        text = path.read_text()
        names = re.findall(r"^`(\w+ExponentialityGofStatistic)`", text, re.MULTILINE)
        assert len(names) == 1, path
        documented.extend(names)
        cls = getattr(exponent, names[0])
        assert f"`{cls.short_code()}`" in text
        assert "## Scientific references" in text
        assert re.search(r"\]\(https://", text)
        assert "## Hypothesis and formula" in text
        assert "## Runnable example" in text
    assert len(documented) == len(set(documented)) == 50
    assert set(documented) == CLASSES


def test_catalog_navigation_and_local_links():
    catalog = (DOCS / "index.md").read_text()
    report = (DOCS / "mathematical-validation.md").read_text()
    navigation = (ROOT / "mkdocs.yml").read_text()
    for path in PAGES:
        assert f"]({path.name})" in catalog
        assert f"]({path.name})" in report
        assert f"exponentiality/{path.name}" in navigation
    for path in [*PAGES, DOCS / "index.md", DOCS / "mathematical-validation.md"]:
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if "://" not in target and not target.startswith("#"):
                assert (path.parent / target.split("#")[0]).exists(), (path, target)


@pytest.mark.parametrize("path", [*PAGES, DOCS / "index.md"], ids=lambda p: p.stem)
def test_documented_examples_execute(path):
    blocks = re.findall(r"```python\n(.*?)```", path.read_text(), re.DOTALL)
    assert blocks, path
    for block in blocks:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(block, str(path), "exec"), {})  # noqa: S102 - repository examples
