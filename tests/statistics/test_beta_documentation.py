"""Keep the Beta documentation aligned with the runtime public catalog."""

import inspect
import re
from pathlib import Path

import pytest

from pysatl_criterion.statistics.goodness_of_fit.continuous import beta


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/criteria/goodness-of-fit/beta"
PAGES = sorted(DOCS.glob("*_criterion.md"))
CLASSES = {
    name: getattr(beta, name)
    for name in beta.__all__
    if not inspect.isabstract(getattr(beta, name))
}


def test_catalog_pages_and_navigation_are_complete():
    overview = (DOCS / "index.md").read_text()
    navigation = (ROOT / "mkdocs.yml").read_text()
    assert len(CLASSES) == len(PAGES) == 23
    for name in CLASSES:
        matching = [page for page in PAGES if name in page.read_text()]
        assert len(matching) == 1, name
        page = matching[0]
        assert name in overview
        assert f"]({page.name})" in overview
        assert f"criteria/goodness-of-fit/beta/{page.name}" in navigation
        text = page.read_text()
        assert "```python" in text
        assert re.search(r"https://(?:doi.org|arxiv.org)/", text)
    assert "criteria/goodness-of-fit/beta/index.md" in navigation
    assert "criteria/goodness-of-fit/beta/mathematical-validation.md" in navigation


@pytest.mark.parametrize("page", PAGES + [DOCS / "index.md"])
def test_examples_execute(page):
    blocks = re.findall(r"```python\n(.*?)```", page.read_text(), re.DOTALL)
    assert blocks
    for index, code in enumerate(blocks):
        exec(compile(code, f"{page}:{index}", "exec"), {})  # noqa: S102


@pytest.mark.parametrize("page", sorted(DOCS.glob("*.md")))
def test_local_links_resolve(page):
    for target in re.findall(r"\]\(([^)]+)\)", page.read_text()):
        if "://" not in target and not target.startswith("#"):
            assert (page.parent / target.split("#")[0]).exists(), (page, target)
