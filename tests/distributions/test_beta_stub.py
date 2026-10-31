"""Keep the Beta stub aligned with runtime exports and constructor options."""

import ast
import inspect
import pickle
from pathlib import Path

import pytest

from pysatl_criterion.statistics.goodness_of_fit.continuous import beta
from pysatl_criterion.statistics.goodness_of_fit.continuous.beta import beta_statistics


class TestBetaStub:
    def test_exports_and_signatures_match_runtime(self):
        tree = ast.parse(Path(beta_statistics.__file__).with_suffix(".pyi").read_text())
        classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
        assert set(classes) == set(beta.__all__)
        exports = next(
            node.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
            )
        )
        assert set(ast.literal_eval(exports)) == set(beta.__all__)
        for name, node in classes.items():
            runtime = getattr(beta, name)
            assert [ast.unparse(base).split(".")[-1] for base in node.bases] == [
                base.__name__ for base in runtime.__bases__
            ]
            for method in node.body:
                if not isinstance(method, ast.FunctionDef):
                    continue
                member = inspect.getattr_static(runtime, method.name)
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
                assert names == list(signature.parameters), (name, method.name)
                for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
                    if default is not None:
                        assert ast.literal_eval(default) == signature.parameters[arg.arg].default

    def test_package_stub_reexports_module_classes(self):
        tree = ast.parse(Path(beta.__file__).with_suffix(".pyi").read_text())
        imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)]
        assert imports
        assert all(node.level == 1 and node.module == "beta_statistics" for node in imports)
        aliases = [alias for node in imports for alias in node.names]
        assert {alias.name for alias in aliases} == set(beta.__all__)
        assert all(alias.asname == alias.name for alias in aliases)
        exports = next(node.value for node in tree.body if isinstance(node, ast.Assign))
        assert set(ast.literal_eval(exports)) == set(beta.__all__)

    @pytest.mark.parametrize("name", beta.__all__)
    def test_package_preserves_identity_pickle_and_discovery(self, name):
        cls = getattr(beta, name)
        assert cls is getattr(beta_statistics, name)
        assert cls.__module__ == beta.__name__
        old_reference = f"c{beta.__name__}\n{name}\n.".encode()
        assert pickle.loads(old_reference) is cls  # noqa: S301
        assert pickle.loads(pickle.dumps(cls)) is cls  # noqa: S301
        if not inspect.isabstract(cls):
            from pysatl_criterion.utils.statistic import (
                get_available_criteria,
                get_criterion_by_code,
            )

            assert get_criterion_by_code(cls.code()) is cls
            assert get_available_criteria(cls.distribution().type()).count(cls) == 1
