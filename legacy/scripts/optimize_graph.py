#!/usr/bin/env python
"""Fuse Whisper attention/linear graph operations, then quantize supported MatMul operators.
Accuracy acceptance remains the responsibility of the real-audio comparison, never file size.
"""
import argparse
from pathlib import Path
from onnxruntime.transformers.optimizer import optimize_model
p=argparse.ArgumentParser();p.add_argument('model',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
m=optimize_model(str(a.model),model_type='bart',num_heads=6,hidden_size=384,opt_level=1)
print(m.get_fused_operator_statistics())
m.save_model_to_file(str(a.out))
