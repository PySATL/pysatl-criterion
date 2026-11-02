"""All continuous distribution packages expose statically declared runtime exports."""

import ast
import importlib
import inspect
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
CONTINUOUS = ROOT / "src/pysatl_criterion/statistics/goodness_of_fit/continuous"
DISTRIBUTIONS = sorted(path.parent.name for path in CONTINUOUS.glob("*/*_statistics.py"))


def stub_exports(tree):
    return next(
        set(ast.literal_eval(node.value))
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
    )


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_module_declarations_and_package_reexports_match_runtime(distribution):
    package = importlib.import_module(
        f"pysatl_criterion.statistics.goodness_of_fit.continuous.{distribution}"
    )
    module_name = f"{distribution}_statistics"
    implementation = importlib.import_module(f"{package.__name__}.{module_name}")
    folder = CONTINUOUS / distribution
    module_tree = ast.parse((folder / f"{module_name}.pyi").read_text())
    package_tree = ast.parse((folder / "__init__.pyi").read_text())
    exports = set(package.__all__)
    assert exports == set(implementation.__all__)
    assert stub_exports(module_tree) == exports
    assert stub_exports(package_tree) == exports
    declarations = {node.name for node in module_tree.body if isinstance(node, ast.ClassDef)}
    declarations.update(
        target.id
        for node in module_tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and isinstance(node.value, ast.Name)
    )
    assert exports <= declarations
    imports = [node for node in package_tree.body if isinstance(node, ast.ImportFrom)]
    assert imports
    assert all(node.level == 1 and node.module == module_name for node in imports)
    aliases = [alias for node in imports for alias in node.names]
    assert {alias.name for alias in aliases} == exports
    assert all(alias.asname in {None, alias.name} for alias in aliases)
    for name in exports:
        assert getattr(package, name) is getattr(implementation, name)


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_declared_method_signatures_match_runtime(distribution):
    """Validate shipped stubs directly, without a dependency on removed generators."""
    package = importlib.import_module(
        f"pysatl_criterion.statistics.goodness_of_fit.continuous.{distribution}"
    )
    path = CONTINUOUS / distribution / f"{distribution}_statistics.pyi"
    for declaration in ast.parse(path.read_text()).body:
        if not isinstance(declaration, ast.ClassDef):
            continue
        cls = getattr(package, declaration.name)
        assert [ast.unparse(base).split(".")[-1] for base in declaration.bases] == [
            base.__name__ for base in cls.__bases__
        ], declaration.name
        for method in declaration.body:
            if not isinstance(method, ast.FunctionDef):
                continue
            member = inspect.getattr_static(cls, method.name)
            if isinstance(member, (staticmethod, classmethod)):
                member = member.__func__
            elif isinstance(member, property):
                member = member.fget
            signature = inspect.signature(member)
            args = method.args
            names = [arg.arg for arg in (*args.posonlyargs, *args.args)]
            if args.vararg:
                names.append(args.vararg.arg)
            names.extend(arg.arg for arg in args.kwonlyargs)
            if args.kwarg:
                names.append(args.kwarg.arg)
            assert names == list(signature.parameters), (declaration.name, method.name)
            for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
                parameter = signature.parameters[arg.arg]
                assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
                if default is None:
                    assert parameter.default is inspect.Parameter.empty
                elif isinstance(default, ast.Constant):
                    assert parameter.default == default.value
