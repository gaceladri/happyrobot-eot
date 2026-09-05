#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
.venv/bin/python - <<'PY'
from dataclasses import asdict
from pathlib import Path
import torch
from eot.model import load_checkpoint,EOTConfig,EOTModel,save_checkpoint
torch.manual_seed(17)
base=load_checkpoint('models/smart-turn-public/model.pt')
cfg=EOTConfig(**{**asdict(base.cfg),'use_fvad':True})
model=EOTModel(cfg,pretrained=False)
missing,unexpected=model.load_state_dict(base.state_dict(),strict=False)
assert not unexpected and all(k.startswith('fvad_head.') for k in missing)
p=Path('models/smart-turn-fvad');p.mkdir(exist_ok=True)
save_checkpoint(model,str(p/'model.pt'))
PY
.venv/bin/eot-train --samples data/optimization/mined.jsonl --out runs/opt_fvad \
 --init-checkpoint models/smart-turn-fvad/model.pt --epochs 8 --max-steps 800 --lr 0.00001 \
 --batch-size 32 --workers 8 --require-cuda --seed 17 --compile-model --checkpoint-every 400 > logs/opt_fvad.log 2>&1
EOT_CHECKPOINT="$PWD/runs/opt_fvad/model.pt" EOT_DISPLAY_NAME='OPT mined with corrected FVAD' PYTHONPATH="$PWD/third_party/eot-bench" \
 HF_HUB_OFFLINE=1 .venv/bin/python -m eot_harness predict --path livekit/eot-bench-data --name en --split validation \
 --revision ca9d98a9686b920a2d8c9eb984224ba9be74e4dd --adapter eot.eotbench_adapter:EOTAdapter --batch-size 64 --output-dir eval/eotbench \
 > logs/opt_fvad_eval.log 2>&1
