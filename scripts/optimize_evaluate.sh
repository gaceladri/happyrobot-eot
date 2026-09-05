#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/third_party/eot-bench"
for name in whole mined mixed student; do
  [[ -f "runs/opt_$name/history.json" ]] || { echo "training incomplete: $name"; exit 1; }
  EOT_CHECKPOINT="$PWD/runs/opt_$name/model.pt" EOT_DISPLAY_NAME="OPT $name (public initialization)" \
    .venv/bin/python -m eot_harness predict --path livekit/eot-bench-data --name en --split validation \
    --revision ca9d98a9686b920a2d8c9eb984224ba9be74e4dd --adapter eot.eotbench_adapter:EOTAdapter \
    --batch-size 64 --output-dir eval/eotbench
  echo "evaluated $name"
done
third_party/eot-bench/.venv/bin/python scripts/eotbench_collect.py --root eval/eotbench \
  --reference third_party/eot-bench/output/livekit__eot-bench-data__validation__min_silence_100ms/en --out eval/eotbench_comparison/en
third_party/eot-bench/.venv/bin/eot-harness compare-models eval/eotbench_comparison/en
