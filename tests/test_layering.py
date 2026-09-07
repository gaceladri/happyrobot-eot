"""The package is organised by pipeline stage; a stage may only import from stages before it."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "eot"
# Modules that must import without torch (serving and benchmark adapters run on CPU-only images).
TORCH_FREE_MODULES = [
    "eot.serving.service",
    "eot.eval.eotbench",
    "eot.eval.policy",
    "eot.eval.krisp",
]
# Stage-neutral utilities may be imported from anywhere.
NEUTRAL = {"io", "context", "metrics", "onnx"}
ORDER = ["audio", "labeling", "data", "modeling", "eval", "serving"]


def _stage(module: str) -> str | None:
    parts = module.split(".")
    if len(parts) < 2 or parts[0] != "eot":
        return None
    return parts[1]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


def test_stages_only_import_earlier_stages():
    violations = []
    for path in SRC.rglob("*.py"):
        stage = path.relative_to(SRC).parts[0].removesuffix(".py")
        if stage in NEUTRAL or stage == "__init__":
            continue
        for module in _imports(path):
            target = _stage(module)
            if target is None or target in NEUTRAL or target == stage:
                continue
            if ORDER.index(target) > ORDER.index(stage):
                violations.append(f"{path.relative_to(SRC)} imports {module}")
    assert not violations, "\n".join(violations)


def test_neutral_helpers_do_not_import_stages():
    for name in NEUTRAL:
        for module in _imports(SRC / f"{name}.py"):
            assert _stage(module) in (None, *NEUTRAL), f"{name}.py imports {module}"


@pytest.mark.parametrize("module", TORCH_FREE_MODULES)
def test_import_is_torch_free(module: str):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    code = f"import {module}, sys; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], env=env, check=True, capture_output=True, text=True)
