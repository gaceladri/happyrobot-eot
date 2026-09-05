#!/usr/bin/env bash
# Run after the performance grid to avoid contaminating CPU timing.
cd "$(dirname "$0")/.."
set -e
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
.venv/bin/eot-train --samples data/optimization/mixed.jsonl --out runs/opt_distilled \
 --init-checkpoint models/smart-turn-public/model.pt --encoder-layers 2 --prune-strategy uniform \
 --teacher-checkpoint runs/opt_mixed/model.pt --distill-weight 0.7 --no-fvad --epochs 8 --max-steps 1600 \
 --lr 0.00003 --batch-size 32 --workers 8 --require-cuda --seed 17 --compile-model --checkpoint-every 800 \
 > logs/opt_distilled.log 2>&1
EOT_CHECKPOINT="$PWD/runs/opt_distilled/model.pt" EOT_DISPLAY_NAME='OPT distilled student' PYTHONPATH="$PWD/third_party/eot-bench" \
 HF_HUB_OFFLINE=1 .venv/bin/python -m eot_harness predict --path livekit/eot-bench-data --name en --split validation \
 --revision ca9d98a9686b920a2d8c9eb984224ba9be74e4dd --adapter eot.eotbench_adapter:EOTAdapter --batch-size 64 --output-dir eval/eotbench \
 > logs/opt_distilled_eval.log 2>&1
.venv/bin/eot-export --checkpoint runs/opt_distilled/model.pt --out exports/opt_distilled/eot.onnx \
 --validation-samples data/optimization/mixed.jsonl > logs/export_opt_distilled.log 2>&1
