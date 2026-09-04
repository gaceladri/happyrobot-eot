#!/usr/bin/env bash
set -Eeuo pipefail

# Reproducible RunPod driver. All generated state is below the mounted volume.
EOT_ROOT="${EOT_ROOT:-/workspace/eot}"
EOT_RUN_ID="${EOT_RUN_ID:-base}"
EOT_LANGUAGE="${EOT_LANGUAGE:-eng}"
EOT_DATASET="${EOT_DATASET:-pipecat-ai/smart-turn-data-v3.2-train}"
# Immutable Smart Turn v3.2 dataset revision verified on 2026-09-04.
EOT_DATASET_REVISION="${EOT_DATASET_REVISION:-e564e2ac567f774d1880aa1db6ce97afb8c519b7}"
EOT_BENCH_REVISION="${EOT_BENCH_REVISION:-ca9d98a9686b920a2d8c9eb984224ba9be74e4dd}"
EOT_PER_LABEL="${EOT_PER_LABEL:-8000}"
EOT_SEED="${EOT_SEED:-0}"
EOT_BATCH_SIZE="${EOT_BATCH_SIZE:-32}"
EOT_WORKERS="${EOT_WORKERS:-4}"
EOT_EPOCHS="${EOT_EPOCHS:-3}"
EOT_CHECKPOINT_EVERY="${EOT_CHECKPOINT_EVERY:-200}"
EOT_MIN_FREE_GB="${EOT_MIN_FREE_GB:-30}"

RAW_DIR="${EOT_RAW_DIR:-${EOT_ROOT}/data/raw/smart-turn-v3.2-${EOT_LANGUAGE}}"
MINED_DIR="${EOT_MINED_DIR:-${EOT_ROOT}/data/mined/prefix-v1-${EOT_LANGUAGE}}"
RUN_DIR="${EOT_RUN_DIR:-${EOT_ROOT}/runs/${EOT_RUN_ID}}"
EXPORT_DIR="${EOT_EXPORT_DIR:-${EOT_ROOT}/exports/${EOT_RUN_ID}}"
EVAL_DIR="${EOT_EVAL_DIR:-${EOT_ROOT}/eval/${EOT_RUN_ID}}"

export EOT_ROOT
export HF_HOME="${HF_HOME:-${EOT_ROOT}/cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TORCH_HOME="${TORCH_HOME:-${EOT_ROOT}/cache/torch}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

usage() {
    cat <<'EOF'
Usage: eot-runpod <preflight|smoke|acquire|mine|audit|train|export|evaluate|all>

Important environment variables:
  EOT_ROOT=/workspace/eot      Persistent root
  EOT_RUN_ID=base              Run/export/evaluation name
  EOT_PER_LABEL=8000           Smart Turn examples per class
  EOT_BATCH_SIZE=32            GPU batch size
  EOT_WORKERS=4                Audio preprocessing workers
  EOT_EPOCHS=3                 Total epochs (also when resuming)
  EOT_CHECKPOINT_EVERY=200     Optimizer steps between atomic recovery saves
  EOT_DATASET_REVISION=<sha>   Immutable Hub revision (a verified default is pinned)
  EOT_BENCH_REVISION=<sha>     Immutable EoT Bench data revision (default pinned)

The main Smart Turn run is intentionally audio-only. Do not add --use-context:
that corpus does not contain paired previous-agent text.
EOF
}

preflight() {
    local available_kb available_gb
    available_kb="$(df -Pk "$EOT_ROOT" | awk 'NR==2 {print $4}')"
    available_gb="$((available_kb / 1024 / 1024))"
    if (( available_gb < EOT_MIN_FREE_GB )); then
        echo "ERROR: ${available_gb} GiB free under $EOT_ROOT; need at least ${EOT_MIN_FREE_GB} GiB." >&2
        return 1
    fi

    python - <<'PY'
import json
import os
import platform
import torch

info = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "runpod_pod_id": os.environ.get("RUNPOD_POD_ID"),
}
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    info.update(gpu=props.name, gpu_vram_gib=round(props.total_memory / 2**30, 2))
print(json.dumps(info, indent=2))
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; refusing to spend GPU-Pod time on CPU training")
if torch.version.cuda != "12.8":
    raise SystemExit(f"expected the image CUDA 12.8 build, got {torch.version.cuda!r}")
PY

    command -v eot-acquire >/dev/null
    command -v eot-mine >/dev/null
    command -v eot-train >/dev/null
    command -v eot-export >/dev/null
    command -v eot-harness >/dev/null
    echo "Preflight passed; persistent root has ${available_gb} GiB free."
}

acquire() {
    local args=(
        --dataset "$EOT_DATASET"
        --language "$EOT_LANGUAGE"
        --per-label "$EOT_PER_LABEL"
        --seed "$EOT_SEED"
        --out "$RAW_DIR"
        --resume
    )
    args+=(--revision "$EOT_DATASET_REVISION")
    if [[ -n "${EOT_MAX_ROWS:-}" ]]; then
        args+=(--max-rows "$EOT_MAX_ROWS")
    fi
    eot-acquire "${args[@]}"
}

mine() {
    eot-mine \
        --manifest "$RAW_DIR/clips.jsonl" \
        --out "$MINED_DIR" \
        --grid 0.2 0.4 0.6 \
        --min-pause 0.2 \
        --resume
}

audit_samples() {
    local require_internal="${1:-1}"
    python - "$MINED_DIR/samples.jsonl" "$RUN_DIR/mining_audit.json" "$require_internal" <<'PY'
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

manifest = Path(sys.argv[1])
destination = Path(sys.argv[2])
require_internal = bool(int(sys.argv[3]))
rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
labels = Counter(int(row["label"]) for row in rows)
kinds = Counter(str(row.get("kind", "unknown")) for row in rows)
sources = Counter(str(row.get("source", "unknown")) for row in rows)
internal = [row for row in rows if row.get("kind") == "internal"]
errors = []
if set(labels) != {0, 1}:
    errors.append(f"expected both labels, found {sorted(labels)}")
if require_internal and not internal:
    errors.append("no internal-pause HOLD examples were mined")
if any(int(row["label"]) != 0 for row in internal):
    errors.append("an internal-pause sample is not HOLD")
if any(float(row.get("time_to_onset", 0.0)) < -1e-8 for row in internal):
    errors.append("an internal-pause sample has negative time_to_onset")
if any(any(int(v) for v in row.get("fvad", [])) for row in rows if int(row["label"]) == 1):
    errors.append("an EOT sample has a positive future-speech target")
if any(not row.get("sha256") for row in rows):
    errors.append("one or more mined rows lack an audio hash")

representatives = {}
for kind in ("internal", "final", "whole"):
    selected = sorted((row for row in rows if row.get("kind") == kind), key=lambda row: str(row["id"]))[:3]
    representatives[kind] = [str((manifest.parent / row["path"]).resolve()) for row in selected]
tto = [float(row["time_to_onset"]) for row in internal]
report = {
    "manifest": str(manifest.resolve()),
    "n_samples": len(rows),
    "labels": dict(sorted(labels.items())),
    "kinds": dict(sorted(kinds.items())),
    "sources": dict(sorted(sources.items())),
    "internal_time_to_onset_s": {
        "min": min(tto) if tto else None,
        "median": statistics.median(tto) if tto else None,
        "max": max(tto) if tto else None,
    },
    "listen_to": representatives,
    "errors": errors,
    "passed": not errors,
}
destination.parent.mkdir(parents=True, exist_ok=True)
tmp = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
os.replace(tmp, destination)
print(json.dumps(report, indent=2, sort_keys=True))
if errors:
    raise SystemExit("mining audit failed: " + "; ".join(errors))
PY
}

train_model() {
    local args=(
        --require-cuda
        --samples "$MINED_DIR/samples.jsonl"
        --out "$RUN_DIR"
        --epochs "$EOT_EPOCHS"
        --batch-size "$EOT_BATCH_SIZE"
        --workers "$EOT_WORKERS"
        --seed "$EOT_SEED"
        --pin-memory
        --checkpoint-every "$EOT_CHECKPOINT_EVERY"
    )
    if [[ -f "$RUN_DIR/checkpoints/last.pt" ]]; then
        args+=(--resume "$RUN_DIR/checkpoints/last.pt")
    fi
    eot-train "${args[@]}"
}

export_model() {
    local checkpoint="$RUN_DIR/model.pt"
    [[ -f "$checkpoint" ]] || { echo "ERROR: missing $checkpoint" >&2; return 1; }
    eot-export --checkpoint "$checkpoint" --out "$EXPORT_DIR/eot.onnx"
}

evaluate_model() {
    local checkpoint="$RUN_DIR/model.pt"
    local predictions
    local found=0
    [[ -f "$checkpoint" ]] || { echo "ERROR: missing $checkpoint" >&2; return 1; }
    export EOT_CHECKPOINT="$checkpoint"
    unset EOT_ONNX
    eot-harness predict \
        --path livekit/eot-bench-data \
        --name en \
        --split validation \
        --revision "$EOT_BENCH_REVISION" \
        --adapter eot.eotbench_adapter:EOTAdapter \
        --overwrite \
        --output-dir "$EVAL_DIR"
    while IFS= read -r predictions; do
        found=1
        eot-harness compute-metrics \
            --predictions "$predictions" \
            --score-point 0.2 \
            --output-dir "$(dirname "$predictions")/metrics"
    done < <(find "$EVAL_DIR" -type f -name predictions.parquet -print)
    if (( found == 0 )); then
        echo "ERROR: EoT Bench produced no predictions.parquet under $EVAL_DIR" >&2
        return 1
    fi
}

smoke() {
    local saved_root="$EOT_ROOT"
    EOT_ROOT="${saved_root}/smoke"
    RAW_DIR="$EOT_ROOT/data/raw"
    MINED_DIR="$EOT_ROOT/data/mined"
    RUN_DIR="$EOT_ROOT/runs/smoke"
    EXPORT_DIR="$EOT_ROOT/exports/smoke"
    EVAL_DIR="$EOT_ROOT/eval/smoke"
    EOT_PER_LABEL=32
    EOT_EPOCHS=1
    EOT_BATCH_SIZE=4
    EOT_WORKERS=2
    EOT_MAX_ROWS="${EOT_SMOKE_MAX_ROWS:-5000}"
    mkdir -p "$RAW_DIR" "$MINED_DIR" "$RUN_DIR" "$EXPORT_DIR" "$EVAL_DIR"
    preflight
    acquire
    mine
    audit_samples 0
    local args=(
        --require-cuda --samples "$MINED_DIR/samples.jsonl" --out "$RUN_DIR"
        --epochs 1 --batch-size 4 --workers 2 --seed "$EOT_SEED"
        --pin-memory --max-steps 2
        --checkpoint-every 1
    )
    if [[ -f "$RUN_DIR/checkpoints/last.pt" ]]; then
        args+=(--resume "$RUN_DIR/checkpoints/last.pt")
    fi
    eot-train "${args[@]}"
    export_model
    echo "Smoke pipeline passed: $EXPORT_DIR/eot.onnx"
}

if [[ $# -ne 1 ]]; then
    usage
    exit 2
fi

if [[ ! "$EOT_RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "ERROR: EOT_RUN_ID must use only letters, numbers, dots, underscores, and hyphens." >&2
    exit 2
fi

mkdir -p "$RAW_DIR" "$MINED_DIR" "$RUN_DIR" "$EXPORT_DIR" "$EVAL_DIR" \
    "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME"

case "${1:-}" in
    preflight) preflight ;;
    smoke) smoke ;;
    acquire) acquire ;;
    mine) mine ;;
    audit) audit_samples ;;
    train) preflight; train_model ;;
    export) export_model ;;
    evaluate) evaluate_model ;;
    all) preflight; acquire; mine; audit_samples; train_model; export_model; evaluate_model ;;
    *) usage; exit 2 ;;
esac
