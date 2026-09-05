#!/usr/bin/env bash
# Sequential labeling stage (runs after eot-acquire and the AppTek/Krisp snapshots are on disk).
# Sequential on purpose: three multi-worker writers at once saturated the local NVMe.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
WORKERS="${EOT_WORKERS:-8}"

rm -rf data/mined/apptek-oracle data/mined/smart-turn-ensemble data/mined/smart-turn-legacy

echo "== AppTek oracle labeling"
uv run eot-apptek --root data/raw/apptek/hf --out data/mined/apptek-oracle --workers "$WORKERS"

echo "== Smart Turn mining: ensemble detector"
uv run eot-mine --manifest data/raw/smart-turn-v3.2-eng/clips.jsonl --out data/mined/smart-turn-ensemble \
    --grid 0.2 0.4 0.6 --detector ensemble --workers "$WORKERS"

echo "== Smart Turn mining: legacy detector (R2')"
uv run eot-mine --manifest data/raw/smart-turn-v3.2-eng/clips.jsonl --out data/mined/smart-turn-legacy \
    --grid 0.2 0.4 0.6 --detector legacy --workers "$WORKERS"

echo "== labeling QA"
uv run eot-qa run \
    --smart-turn-clips data/raw/smart-turn-v3.2-eng/clips.jsonl \
    --mined data/mined/smart-turn-ensemble/samples.jsonl data/mined/smart-turn-legacy/samples.jsonl \
    --apptek data/mined/apptek-oracle --apptek-root data/raw/apptek/hf \
    --krisp-clips data/raw/krisp/clips/clips.jsonl \
    --listen-manifests data/mined/smart-turn-ensemble/samples.jsonl data/mined/apptek-oracle/train/samples.jsonl \
    --out eval/labeling_qa
echo "labeling stage done"
