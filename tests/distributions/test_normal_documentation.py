"""Keep the normal catalog, individual pages, navigation and examples aligned."""

import inspect
import re
from pathlib import Path

import pytest

from pysatl_criterion.statistics.goodness_of_fit.continuous import normal
from tests.parameter_cases import parameters_for


ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "docs/criteria/goodness-of-fit/normality"
CLASSES = {
    name: getattr(normal, name)
    for name in normal.__all__
    if not inspect.isabstract(getattr(normal, name))
}


def test_catalog_exactly_matches_runtime_exports():
    index = (FOLDER / "index.md").read_text()
    documented = re.findall(r"\| `([A-Za-z0-9]+NormalityGofStatistic)` \|", index)
    assert len(documented) == len(set(documented)) == len(CLASSES) == 60
    assert set(documented) == set(CLASSES)
    for name, cls in CLASSES.items():
        row = next(line for line in index.splitlines() if f"| `{name}` |" in line)
        assert f"`{cls.code()}`" in row
        assert cls.short_code() in (FOLDER / "mathematical-validation.md").read_text()


@pytest.mark.parametrize("name,cls", CLASSES.items())
def test_each_statistic_has_one_page_with_example_references_and_identity(name, cls):
    pages = [p for p in FOLDER.glob("*_criterion.md") if f"**Class:** `{name}`" in p.read_text()]
    assert len(pages) == 1
    page = pages[0]
    text = page.read_text()
    assert f"**Code:** `{cls.code()}`" in text
    assert "## References" in text
    assert re.search(r"https://(?:doi.org|arxiv.org|www.additive-net.de)", text)
    assert "```python\n" in text
    assert name in text.split("```python\n", 1)[1].split("```", 1)[0]
    assert page.name in (FOLDER / "index.md").read_text()
    assert f"criteria/goodness-of-fit/normality/{page.name}" in (ROOT / "mkdocs.yml").read_text()
    doc = inspect.getdoc(cls)
    assert "References" in doc and "https://" in doc and "Examples" in doc
    assert cls(parameters_for(cls)).hypothesis().parameters() is not None


@pytest.mark.parametrize("page", sorted(FOLDER.glob("*.md")), ids=lambda p: p.name)
def test_documentation_local_links_and_examples(page):
    text = page.read_text()
    for target in re.findall(r"(?<!!)\[[^\]]*\]\(([^)]+)\)", text):
        if "://" in target or target.startswith("#"):
            continue
        path = target.split("#", 1)[0]
        assert (page.parent / path).resolve().exists(), (page.name, target)
    for block in re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL):
        exec(compile(block, str(page), "exec"), {})  # noqa: S102 - local documentation examples
