#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
.venv/bin/eot-krisp score --clips data/raw/krisp/clips/clips.jsonl --checkpoint runs/opt_clean/model.pt --out eval/krisp/opt_clean.jsonl > logs/krisp_opt_clean.log 2>&1
for name in opt_clean opt_mixed r0_vad r1_smart_turn_public r2 r2_legacy r3 r4; do
 .venv/bin/eot-krisp metrics --predictions "eval/krisp/$name.jsonl" --out "eval/krisp/$name.metrics.json" --bootstrap-reps 500 > "logs/krisp_metrics_$name.log" 2>&1
done
.venv/bin/eot-report heldout --samples data/mined/apptek-oracle/heldout/samples.jsonl --checkpoint runs/opt_clean/model.pt --out eval/heldout/opt_clean.json > logs/heldout_opt_clean.log 2>&1
