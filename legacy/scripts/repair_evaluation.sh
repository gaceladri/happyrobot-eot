#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
HV=third_party/eot-bench/.venv/bin
D=eval/eotbench_comparison/en
"$HV/eot-harness" compare-models "$D"
for d in "$D"/*; do
  [[ -f "$d/metrics/summary.json" ]] || continue
  "$HV/python" scripts/eotbench_bootstrap.py --run-dir "$d" --reps 1000
  if [[ "$d" == *eot_adapter__* ]]; then
    "$HV/python" scripts/eotbench_bootstrap.py --run-dir "$d" --paired-with "$D/smart_turn_audio_adapter__cb4f57229a" --reps 1000
    "$HV/python" scripts/eotbench_bootstrap.py --run-dir "$d" --calibration-fraction 0.5 --reps 1000 --out "$d/metrics/calibrated_test.json"
  fi
done
