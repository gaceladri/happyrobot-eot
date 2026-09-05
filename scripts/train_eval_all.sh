#!/usr/bin/env bash
# Pre-registered runs R2, R2', R3, R4 on the local GPU, then every evaluation.
# Usage: scripts/train_eval_all.sh [train|export|eotbench|krisp|heldout|summarize|all]
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
set -a; [[ -f .env ]] && . ./.env; set +a
export HF_TOKEN="${HF_TOKEN:-${huggingface_token:-}}"

EPOCHS="${EOT_EPOCHS:-3}"
BATCH="${EOT_BATCH_SIZE:-32}"
WORKERS="${EOT_WORKERS:-8}"
SEED="${EOT_SEED:-0}"
BENCH_REVISION="ca9d98a9686b920a2d8c9eb984224ba9be74e4dd"
HARNESS=third_party/eot-bench
HVENV="$HARNESS/.venv/bin"
SPAN_SET=livekit__eot-bench-data__validation__min_silence_100ms
EVAL_EN="eval/eotbench_comparison/en"

declare -A SAMPLES=(
  [r2]=data/mined/smart-turn-ensemble/samples.jsonl
  [r2_legacy]=data/mined/smart-turn-legacy/samples.jsonl
  [r3]=data/mined/r3-ensemble-plus-apptek/samples.jsonl
  [r4]=data/mined/r4-apptek-only/samples.jsonl
)
declare -A NAMES=(
  [r2]="R2 ours: Smart Turn mined, ensemble detector"
  [r2_legacy]="R2' ours: Smart Turn mined, legacy detector"
  [r3]="R3 ours: R2 + AppTek oracle (capped)"
  [r4]="R4 ours: AppTek oracle only"
)
RUNS=(r2 r2_legacy r3 r4)

manifests() {
  local n_ens
  n_ens=$(wc -l < data/mined/smart-turn-ensemble/samples.jsonl)
  echo "== combined manifests (AppTek capped at the Smart Turn ensemble size: $n_ens)"
  uv run eot-report combine --inputs data/mined/smart-turn-ensemble/samples.jsonl data/mined/apptek-oracle/train/samples.jsonl \
      --caps 0 "$n_ens" --out data/mined/r3-ensemble-plus-apptek/samples.jsonl --seed "$SEED"
  uv run eot-report combine --inputs data/mined/apptek-oracle/train/samples.jsonl \
      --caps "$n_ens" --out data/mined/r4-apptek-only/samples.jsonl --seed "$SEED"
}

train() {
  for r in "${RUNS[@]}"; do
    if [[ -f "runs/$r/history.jsonl" ]] && (( $(wc -l < "runs/$r/history.jsonl") >= EPOCHS )); then echo "== $r already trained"; continue; fi
    echo "== train $r ($(wc -l < "${SAMPLES[$r]}") samples)"
    args=(--require-cuda --samples "${SAMPLES[$r]}" --out "runs/$r" --epochs "$EPOCHS" --batch-size "$BATCH"
          --workers "$WORKERS" --seed "$SEED" --pin-memory --checkpoint-every 200)
    [[ -f "runs/$r/checkpoints/last.pt" ]] && args+=(--resume)
    [[ "${EOT_COMPILE:-1}" == "1" ]] && args+=(--compile-model)
    uv run eot-train "${args[@]}" > "logs/train_$r.log" 2>&1
    grep -E '^\{"acc' "logs/train_$r.log" || tail -n 5 "logs/train_$r.log"
    [[ -f "runs/$r/model.pt" ]]
  done
}

export_all() {
  for r in "${RUNS[@]}"; do
    if python3 - "$r" <<'PY_CHECK'
import json, hashlib, sys
from pathlib import Path
r=sys.argv[1]; p=Path(f'exports/{r}/eot.onnx'); meta=p.with_suffix('.json')
try:
 m=json.loads(meta.read_text())
 ok=m.get('accepted') and m['sha256']==hashlib.sha256(p.read_bytes()).hexdigest() and m['checkpoint_sha256']==hashlib.sha256(Path(f'runs/{r}/model.pt').read_bytes()).hexdigest()
except (OSError,KeyError,ValueError): ok=False
sys.exit(0 if ok else 1)
PY_CHECK
    then continue; fi
    echo "== export $r"
    uv run eot-export --checkpoint "runs/$r/model.pt" --out "exports/$r/eot.onnx"
  done
}

eotbench() {
  mkdir -p "eval/eotbench/$SPAN_SET"
  ours() { grep -l "\"display_name\": \"${NAMES[$1]}\"" "$EVAL_EN"/*/manifest.json 2>/dev/null | head -1 | xargs -r dirname; }
  for r in "${RUNS[@]}"; do
    [[ -n "$(ours "$r")" ]] && { echo "== $r already predicted"; continue; }
    echo "== eot-harness predict $r"
    EOT_ONNX="$PWD/exports/$r/eot.onnx" EOT_DISPLAY_NAME="${NAMES[$r]}" "$HVENV/eot-harness" predict \
        --path livekit/eot-bench-data --name en --split validation --revision "$BENCH_REVISION" \
        --adapter eot.eotbench_adapter:EOTAdapter --batch-size 64 --output-dir eval/eotbench
  done
  # Collect revision-qualified runs after verifying complete prediction-key alignment.
  "$HVENV/python" scripts/eotbench_collect.py --root eval/eotbench \
      --reference "$HARNESS/output/$SPAN_SET/en" --out "$EVAL_EN"
  while IFS= read -r pred; do
    d="$(dirname "$pred")"
    [[ -f "$d/metrics/summary.json" ]] || "$HVENV/eot-harness" compute-metrics --predictions "$pred" --score-point 0.2 --output-dir "$d/metrics"
  done < <(find "$EVAL_EN" -name predictions.parquet)
  "$HVENV/eot-harness" compare-models "$EVAL_EN"
  echo "== turn-level bootstrap (fixed policies)"
  for d in "$EVAL_EN"/*/; do
    [[ -f "$d/metrics/summary.json" && ! -f "$d/metrics/bootstrap.json" ]] || continue
    "$HVENV/python" scripts/eotbench_bootstrap.py --run-dir "$d" --reps "${EOT_BOOTSTRAP_REPS:-500}"
  done
  # paired: each of our runs vs published Smart Turn v3.2, and R2 vs R2', R3 vs R2
  st=$(ls -d "$EVAL_EN"/smart_turn_audio_adapter__* | head -1)
  for r in "${RUNS[@]}"; do
    d=$(ours "$r"); [[ -n "$d" ]] || continue
    [[ -f "$d/metrics/bootstrap_paired.json" ]] || "$HVENV/python" scripts/eotbench_bootstrap.py --run-dir "$d" --paired-with "$st" --reps "${EOT_BOOTSTRAP_REPS:-500}"
  done
  d2=$(ours r2); d2l=$(ours r2_legacy); d3=$(ours r3)
  [[ -n "$d2" && -n "$d2l" ]] && "$HVENV/python" scripts/eotbench_bootstrap.py --run-dir "$d2" --paired-with "$d2l" --reps "${EOT_BOOTSTRAP_REPS:-500}" --out "$d2/metrics/bootstrap_paired_vs_legacy.json"
  [[ -n "$d3" && -n "$d2" ]] && "$HVENV/python" scripts/eotbench_bootstrap.py --run-dir "$d3" --paired-with "$d2" --reps "${EOT_BOOTSTRAP_REPS:-500}" --out "$d3/metrics/bootstrap_paired_vs_r2.json"
  true
}

krisp() {
  mkdir -p eval/krisp
  [[ -f eval/krisp/r0_vad.jsonl ]] || uv run eot-krisp score --clips data/raw/krisp/clips/clips.jsonl --vad-baseline --out eval/krisp/r0_vad.jsonl
  [[ -f eval/krisp/r1_smart_turn_public.jsonl ]] || uv run eot-krisp score --clips data/raw/krisp/clips/clips.jsonl --smart-turn-public --out eval/krisp/r1_smart_turn_public.jsonl
  for r in "${RUNS[@]}"; do
    [[ -f "eval/krisp/$r.jsonl" ]] || uv run eot-krisp score --clips data/raw/krisp/clips/clips.jsonl --checkpoint "runs/$r/model.pt" --out "eval/krisp/$r.jsonl"
  done
  for p in eval/krisp/*.jsonl; do
    m="${p%.jsonl}.metrics.json"
    [[ -f "$m" ]] || uv run eot-krisp metrics --predictions "$p" --out "$m" --bootstrap-reps "${EOT_BOOTSTRAP_REPS:-500}"
  done
}

heldout() {
  mkdir -p eval/heldout
  for r in "${RUNS[@]}"; do
    [[ -f "eval/heldout/$r.json" ]] || uv run eot-report heldout --samples data/mined/apptek-oracle/heldout/samples.jsonl --checkpoint "runs/$r/model.pt" --out "eval/heldout/$r.json"
  done
}

summarize() {
  python3 - <<'PY'
import json
names = {"r2": "R2 ours: Smart Turn mined, ensemble detector", "r2_legacy": "R2' ours: Smart Turn mined, legacy detector",
         "r3": "R3 ours: R2 + AppTek oracle (capped)", "r4": "R4 ours: AppTek oracle only",
         "r0_vad": "R0 VAD baseline", "r1_smart_turn_public": "R1 Smart Turn v3.2 (public)"}
json.dump(names, open("eval/names.json", "w"), indent=2)
PY
  uv run eot-report summarize --eotbench-dir "$EVAL_EN" --krisp-dir eval/krisp --heldout-dir eval/heldout --names eval/names.json --out eval/summary
}

case "${1:-all}" in
  manifests) manifests ;;
  train) train ;;
  export) export_all ;;
  eotbench) eotbench ;;
  krisp) krisp ;;
  heldout) heldout ;;
  summarize) summarize ;;
  all) manifests; train; export_all; eotbench; krisp; heldout; summarize ;;
  *) echo "usage: $0 [manifests|train|export|eotbench|krisp|heldout|summarize|all]"; exit 2 ;;
esac
