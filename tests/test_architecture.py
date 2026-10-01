"""Dependency rules between the parts of the iot_radar package.

Each module is parsed (not imported) and its imports of ``iot_radar.*`` are
collected; imports under ``if TYPE_CHECKING:`` are ignored because they are
only used by type checkers.  The rules are those of REFACTOR_PLAN.md:

* ``dsp`` is pure signal processing: it does not know where samples come
  from, how they are displayed, nor the machine-learning code;
* ``ui`` never touches the acquisition hardware, even indirectly;
* ``ml`` does not depend on the acquisition;
* ``config`` and ``physics`` are leaves.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "iot_radar"

FORBIDDEN: dict[str, set[str]] = {
    "dsp": {"acquisition", "ui", "ml", "pipeline"},
    "ml": {"acquisition"},
    "pipeline": {"ui", "ml"},
    "config": {"acquisition", "dsp", "ml", "pipeline", "ui", "physics"},
    "physics": {"acquisition", "dsp", "ml", "pipeline", "ui", "config"},
}
FORBIDDEN_TRANSITIVELY: dict[str, set[str]] = {
    "ui": {"acquisition", "ml"},
}


def _module_name(path: Path) -> str:
    parts = path.relative_to(PACKAGE_DIR.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _part(module: str) -> str:
    """``iot_radar.dsp.decimation`` → ``dsp``; ``iot_radar.config`` → ``config``."""
    parts = module.split(".")
    return parts[1] if len(parts) > 1 else ""


def _is_type_checking_block(node: ast.AST) -> bool:
    return isinstance(node, ast.If) and (
        (isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING")
        or (isinstance(node.test, ast.Attribute) and node.test.attr == "TYPE_CHECKING")
    )


def _runtime_imports(path: Path) -> set[str]:
    """``iot_radar`` modules imported by *path* at run time."""
    found: set[str] = set()

    def visit(node: ast.AST) -> None:
        if _is_type_checking_block(node):
            return
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names if a.name.startswith("iot_radar"))
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("iot_radar"):
            found.add(node.module)
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(ast.parse(path.read_text(encoding="utf-8")))
    return found


GRAPH: dict[str, set[str]] = {
    _module_name(path): _runtime_imports(path) for path in PACKAGE_DIR.rglob("*.py")
}


def _reachable(module: str) -> set[str]:
    seen: set[str] = set()
    stack = [module]
    while stack:
        for imported in GRAPH.get(stack.pop(), set()):
            if imported not in seen:
                seen.add(imported)
                stack.append(imported)
    return seen


@pytest.mark.parametrize("module", sorted(GRAPH))
def test_direct_dependencies(module: str) -> None:
    forbidden = FORBIDDEN.get(_part(module), set())
    offending = sorted(m for m in GRAPH[module] if _part(m) in forbidden)
    assert not offending, f"{module} must not import {offending}"


@pytest.mark.parametrize("module", sorted(m for m in GRAPH if _part(m) in FORBIDDEN_TRANSITIVELY))
def test_transitive_dependencies(module: str) -> None:
    forbidden = FORBIDDEN_TRANSITIVELY[_part(module)]
    offending = sorted(m for m in _reachable(module) if _part(m) in forbidden)
    assert not offending, f"{module} must not depend (even indirectly) on {offending}"
