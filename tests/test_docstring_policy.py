"""Policy tests for module and function docstrings in source and scripts."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _iter_policy_files() -> list[Path]:
    """Collect files covered by the docstring policy test."""
    return sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "scripts").rglob("*.py"))


def test_module_docstrings_present_for_src_and_scripts() -> None:
    missing: list[str] = []
    for path in _iter_policy_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if ast.get_docstring(tree) is None:
            missing.append(str(path.relative_to(ROOT)))
    assert not missing, f"Missing module docstrings:\n" + "\n".join(missing)


def test_function_docstrings_present_for_src_and_scripts() -> None:
    missing: list[str] = []
    for path in _iter_policy_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if ast.get_docstring(node) is None:
                    rel = path.relative_to(ROOT)
                    missing.append(f"{rel}:{node.lineno}:{node.name}")
    assert not missing, "Missing function docstrings:\n" + "\n".join(missing)
