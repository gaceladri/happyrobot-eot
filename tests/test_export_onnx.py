"""ONNX export guards."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import tiny_model

from eot.modeling.export_onnx import _parity_inputs


def test_export_rejects_missing_or_empty_validation_manifest(tmp_path: Path):
    rng = np.random.default_rng(0)
    with pytest.raises(FileNotFoundError):
        _parity_inputs(tiny_model(), tmp_path / "missing.jsonl", rng)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(ValueError, match="empty"):
        _parity_inputs(tiny_model(), empty, rng)
