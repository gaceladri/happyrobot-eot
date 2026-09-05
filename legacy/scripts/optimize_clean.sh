#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
.venv/bin/eot-train --samples data/optimization/mixed.jsonl --out runs/opt_clean \
 --init-checkpoint models/smart-turn-public/model.pt --no-fvad --epochs 8 --max-steps 800 --lr 0.00001 \
 --batch-size 32 --workers 8 --require-cuda --seed 17 --compile-model --checkpoint-every 400 > logs/opt_clean.log 2>&1
EOT_CHECKPOINT="$PWD/runs/opt_clean/model.pt" EOT_DISPLAY_NAME='OPT clean mixed (CPU-matched FP32)' PYTHONPATH="$PWD/third_party/eot-bench" \
 HF_HUB_OFFLINE=1 .venv/bin/python -m eot_harness predict --path livekit/eot-bench-data --name en --split validation \
 --revision ca9d98a9686b920a2d8c9eb984224ba9be74e4dd --adapter eot.eotbench_adapter:EOTAdapter --batch-size 64 --output-dir eval/eotbench \
 > logs/opt_clean_eval.log 2>&1
.venv/bin/eot-export --checkpoint runs/opt_clean/model.pt --out exports/opt_clean/eot.onnx \
 --validation-samples data/optimization/mixed.jsonl > logs/export_opt_clean.log 2>&1
