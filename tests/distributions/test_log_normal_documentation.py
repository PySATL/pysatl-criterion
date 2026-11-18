"""The runtime, catalog, scientific references, examples and navigation stay aligned."""

import doctest
import inspect
import re
from pathlib import Path

import pytest

from pysatl_criterion.statistics.goodness_of_fit.continuous import log_normal


ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "docs/criteria/goodness-of-fit/log-normal"
CLASSES = {
    name: getattr(log_normal, name)
    for name in log_normal.__all__
    if not inspect.isabstract(getattr(log_normal, name))
}


def test_catalog_exactly_matches_runtime_exports():
    index = (FOLDER / "index.md").read_text()
    documented = re.findall(r"\| `([A-Za-z0-9]+LogNormalGo[fF]Statistic)` \|", index)
    assert len(documented) == len(set(documented)) == len(CLASSES) == 55
    assert set(documented) == set(CLASSES)
    report = (FOLDER / "mathematical-validation.md").read_text()
    for name, cls in CLASSES.items():
        row = next(line for line in index.splitlines() if f"| `{name}` |" in line)
        assert f"`{cls.code()}`" in row
        assert f"`{cls.short_code()}`" in report


@pytest.mark.parametrize("name,cls", CLASSES.items())
def test_each_statistic_has_exactly_one_page_with_sources_and_examples(name, cls):
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
    assert f"criteria/goodness-of-fit/log-normal/{page.name}" in (ROOT / "mkdocs.yml").read_text()
    doc = inspect.getdoc(cls)
    assert "References" in doc and "https://" in doc and "Examples" in doc
    runner = doctest.DocTestRunner()
    test = doctest.DocTestParser().get_doctest(doc, {}, name, str(page), 0)
    runner.run(test)
    assert runner.failures == 0


@pytest.mark.parametrize("page", sorted(FOLDER.glob("*.md")), ids=lambda p: p.name)
def test_local_links_and_python_examples(page):
    text = page.read_text()
    for target in re.findall(r"(?<!!)\[[^\]]*\]\(([^)]+)\)", text):
        if "://" in target or target.startswith("#"):
            continue
        assert (page.parent / target.split("#", 1)[0]).resolve().exists(), (page.name, target)
    for block in re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL):
        exec(compile(block, str(page), "exec"), {})  # noqa: S102 - local documentation examples
