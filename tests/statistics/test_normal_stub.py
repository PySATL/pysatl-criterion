"""The normality stub must expose the runtime classes and constructor options."""

import ast
import inspect
import pickle
from pathlib import Path

import pytest

from pysatl_criterion.statistics.goodness_of_fit.continuous import normal
from pysatl_criterion.statistics.goodness_of_fit.continuous.normal import normal_statistics


class TestNormalStub:
    def test_exports_bases_and_method_parameters_match_runtime(self):
        tree = ast.parse(Path(normal.__file__).with_name("normal_statistics.pyi").read_text())
        classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
        assert set(classes) == set(normal.__all__)
        exports = next(
            n.value
            for n in tree.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "__all__" for t in n.targets)
        )
        assert set(ast.literal_eval(exports)) == set(normal.__all__)
        for name, cls in classes.items():
            runtime = getattr(normal, name)
            assert [ast.unparse(b).split(".")[-1] for b in cls.bases] == [
                b.__name__ for b in runtime.__bases__
            ]
            for method in cls.body:
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
                    if default is not None and isinstance(default, ast.Constant):
                        assert default.value == signature.parameters[arg.arg].default

    @pytest.mark.parametrize("name", normal.__all__)
    def test_package_preserves_identity_pickle_and_discovery(self, name):
        cls = getattr(normal, name)
        assert cls is getattr(normal_statistics, name)
        assert cls.__module__ == normal.__name__
        old_reference = f"c{normal.__name__}\n{name}\n.".encode()
        assert pickle.loads(old_reference) is cls  # noqa: S301
        assert pickle.loads(pickle.dumps(cls)) is cls  # noqa: S301
        if not inspect.isabstract(cls):
            from pysatl_criterion.utils.statistic import (
                get_available_criteria,
                get_criterion_by_code,
            )

            assert get_criterion_by_code(cls.code()) is cls
            assert get_available_criteria(cls.distribution().type()).count(cls) == 1

    def test_package_exports_match_implementation(self):
        assert set(normal.__all__) == set(normal_statistics.__all__)
