"""Plugin-side boundary rules, enforced mechanically (review round 54, minor #3).

The kernel-side rule (harness imports nothing) lives in test_kernel.py; these
cover the other direction: plugins never import each other, and profiles stay
declarative.
"""

import ast
import pathlib
import sys

from harness.manifest import discover

#: Transitional L0 allowance: plugins may import the legacy governor library
#: (ARCHITECTURE.md migration ladder); it goes away at L2. Any stdlib module
#: plus numpy is fine; what matters is harness/governor/plugins discipline.
_PLUGIN_ALLOWED = ("harness", "governor", "numpy")

#: Third-party dependencies each plugin DECLARES -- now READ FROM the manifests
#: (each card's ``third_party`` field), not a hand-kept base table. The simulator
#: belongs to the embodiment plugin and nowhere else -- an rsi or policies module
#: importing robosuite is an embodiment leak, and this is what catches it. A card
#: that needs a new third-party root declares it in its own manifest.
_PLUGIN_THIRD_PARTY = discover().third_party


def _allowed(root: str, plugin: str = "") -> bool:
    if root in _PLUGIN_THIRD_PARTY.get(plugin, ()):
        return True
    return root in _PLUGIN_ALLOWED or root in sys.stdlib_module_names


def _imports(path: pathlib.Path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            yield node.lineno, node.module or ""


#: The card tree these boundary rules govern: everything under plugins/ EXCEPT
#: plugins/candidates/, which an evolve round writes at runtime (a model-written
#: candidate is a run artefact, gitignored, and never mounted without the doctor).
#: A candidate that breaks a boundary must fail its doctor, not this suite.
def _cards():
    return [p for p in pathlib.Path("plugins").rglob("*.py")
            if "candidates" not in p.relative_to("plugins").parts[:1]]


def _plugin_package(path: pathlib.Path) -> str:
    rel = path.relative_to("plugins")
    return rel.parts[0].removesuffix(".py")


def test_plugins_never_import_each_other():
    for path in _cards():
        own = _plugin_package(path)
        for lineno, name in _imports(path):
            if name.startswith("plugins"):
                parts = name.split(".")
                imported = parts[1] if len(parts) > 1 else ""
                assert imported in ("", own), \
                    f"{path}:{lineno} imports sibling plugin {name}"


def test_plugins_import_only_the_allowed_surfaces():
    for path in _cards():
        for lineno, name in _imports(path):
            root = name.split(".")[0]
            assert _allowed(root, _plugin_package(path)) or root.startswith("plugins"), \
                f"{path}:{lineno} imports {name}, outside the allowed plugin surface"


def test_profiles_stay_declarative():
    """profiles/ holds Mounts and prereg overrides; provider code stays out."""
    for path in pathlib.Path("profiles").rglob("*.py"):
        for lineno, name in _imports(path):
            root = name.split(".")[0]
            assert root == "harness" or root in sys.stdlib_module_names, \
                f"{path}:{lineno} imports {name}; profiles reference providers " \
                "by string, never by import"
