#!/usr/bin/env bash
cd "$(dirname "$0")/.."
set -e
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
for r in opt_mixed; do
 .venv/bin/eot-krisp score --clips data/raw/krisp/clips/clips.jsonl --checkpoint "runs/$r/model.pt" --out "eval/krisp/$r.jsonl" > "logs/krisp_$r.log" 2>&1
 echo "Krisp scored $r"
done
for p in eval/krisp/*.jsonl; do
 .venv/bin/eot-krisp metrics --predictions "$p" --out "${p%.jsonl}.metrics.json" --bootstrap-reps 500 > "logs/krisp_metrics_$(basename "$p").log" 2>&1
 echo "Krisp metrics $p"
done
mkdir -p eval/heldout
.venv/bin/eot-report heldout --samples data/mined/apptek-oracle/heldout/samples.jsonl --checkpoint runs/opt_mixed/model.pt --out eval/heldout/opt_mixed.json > logs/heldout_opt_mixed.log 2>&1
