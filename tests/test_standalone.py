"""Dependency-surface tests: the package imports only what pyproject declares."""
import ast
import pathlib

import pytest

PKG = pathlib.Path(__file__).resolve().parent.parent / "timingtask"


def _imported_roots(path: pathlib.Path):
    """Top-level module names imported anywhere in ``path``, including
    imports nested inside functions."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:      # absolute import only
                roots.add(node.module.split(".")[0])
    return roots


def test_package_imports_without_optional_extras():
    """``import timingtask`` requires only numpy, torch and h5py."""
    import timingtask
    assert timingtask.TrialGenerator is not None
    assert timingtask.VanillaRNN is not None
    assert timingtask.records_to_trajectory is not None


def test_dependency_surface_is_what_the_pyproject_declares():
    """No module imports a third-party package that is neither a core
    dependency nor a declared extra."""
    allowed = {
        # core
        "numpy", "torch", "h5py",
        # declared extras
        "matplotlib", "sklearn", "gymnasium",
        # stdlib used by the package
        "__future__", "argparse", "ast", "collections", "copy", "dataclasses",
        "datetime", "json", "math", "os", "pathlib", "time", "typing",
        "warnings", "itertools", "functools", "random", "sys", "csv", "re",
        # itself
        "timingtask",
    }
    seen = set()
    for path in PKG.glob("*.py"):
        seen |= _imported_roots(path)
    unexpected = seen - allowed
    assert not unexpected, (
        f"undeclared dependencies: {sorted(unexpected)}. Add them to "
        f"pyproject.toml, or to the allow-list here if they are stdlib.")
