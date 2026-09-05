#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
for name in whole mined mixed student; do
  samples=$name
  extra=()
  lr=0.00001
  if [[ "$name" == student ]]; then samples=mixed; extra=(--encoder-layers 2); lr=0.00005; fi
  .venv/bin/eot-train --samples "data/optimization/$samples.jsonl" --out "runs/opt_$name" \
    --init-checkpoint models/smart-turn-public/model.pt --no-fvad --epochs 8 --max-steps 800 \
    --lr "$lr" --batch-size 32 --workers 8 --require-cuda --seed 17 --compile-model \
    --checkpoint-every 400 "${extra[@]}" > "logs/opt_$name.log" 2>&1
  echo "finished $name"
done
