"""The split is the point: this package must not depend on the analysis one.

These tests are the guard rail on that. They fail loudly if an import creeps
back in, because the failure mode otherwise is silent -- both packages are
installed in the same conda environment, so an accidental
``from neuralgeom...`` works perfectly on this machine and only breaks for
someone who installed timingtask alone.
"""
import ast
import pathlib

import pytest

PKG = pathlib.Path(__file__).resolve().parent.parent / "timingtask"


def _imported_roots(path: pathlib.Path):
    """Every top-level module name imported anywhere in ``path``, including
    imports nested inside functions (which is where a lazy dependency hides)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:      # absolute import only
                roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("path", sorted(PKG.glob("*.py")), ids=lambda p: p.name)
def test_no_neuralgeom_import(path):
    assert "neuralgeom" not in _imported_roots(path), (
        f"{path.name} imports neuralgeom. The two repositories meet at the "
        f"Trajectory HDF5 file written by timingtask.export, not at an import.")


def test_package_imports_without_optional_extras():
    """``import timingtask`` must work with only numpy/torch/h5py present.

    gymnasium, matplotlib and scikit-learn are extras; the package-level
    ``__getattr__`` keeps ``TimingTaskEnv`` out of the import path and
    ``plots.py`` imports matplotlib inside its functions.
    """
    import timingtask
    assert timingtask.TrialGenerator is not None
    assert timingtask.VanillaRNN is not None
    assert timingtask.records_to_trajectory is not None


def test_dependency_surface_is_what_the_pyproject_declares():
    """Nothing in the package may import a third-party package that is neither
    a core dependency nor a declared extra."""
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
