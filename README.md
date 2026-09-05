# eot — end-of-turn detection for voice agents

Unified, stage-oriented source for the HappyRobot EoT exercise. It merges the baseline package
that produced the shipped model with the reusable pieces of the research worktrees and drops
the experiments that were rejected (encoder pruning, distillation, auxiliary branches).

```
src/eot/
  io.py          durable JSONL/WAV helpers                      (torch-free)
  context.py     hashed agent-text features                    (torch-free)
  metrics.py     AUC and Pareto helpers                        (torch-free)
  onnx.py        onnxruntime session helpers                   (torch-free)
  audio/         front-end, pause detectors, augmentation     (torch-free)
  labeling/      prefix mining, dual-channel oracle, event targets, labeler QA
  data/          Smart Turn acquisition, grouped splits, torch dataset
  modeling/      Whisper-Tiny model, training, weight soups, ONNX export
  eval/          endpointing policy, EoT Bench adapter, Krisp test set, reports, challenge set
  serving/       FastAPI + onnxruntime service and load test  (torch-free)
```

Each stage imports only from the stages above it (`tests/test_layering.py` checks the import graph).
`eot.serving`, `eot.eval.policy` and the ONNX path of `eot.eval.eotbench` never import torch, which
the tests also enforce.

## What came from where

| Piece | Origin | Why it is in |
|---|---|---|
| `audio.frontend.resample_antialiased`, `causal_prefix` | D019 | polyphase resampling with >40 dB stopband; prefix truncated before filtering so training audio is provably causal |
| `labeling.apptek --rule-version v2` | D012 | abstains on duration-only "backchannel" HOLDs instead of asserting intent; refuses to overwrite a labeling run |
| `labeling.turn_events` | D015/D016 | conservative targets from dual-channel conversational event annotations |
| `modeling.train --exclude-train-ids/--split-seed` | E013/E017 | train-only exclusions after the frozen split; seed replication with a fixed split |
| `modeling.model` internal context crop | P010 | shorter encoder context behind the same 8 s interface (whisper-base feasibility) |
| `modeling.soup` | E008/E020 | fixed-alpha weight soups scored on the frozen dev split |
| `io.write_wav`/`atomic_write_wav`, `audio.load_wav`/`decode_payload`/`to_16k`, `metrics`, `onnx` | unification | one loader, one writer (returning the sha256), one AUC, one Pareto front, one onnxruntime session helper instead of many copies |

Dropped: two-layer pruning and soft-label distillation (`OPT student/distilled`, worse frontier),
the MFU throughput benchmark (research tooling), and the research registry scripts (live in the
main repository under `research/`).

## Use

```bash
uv sync --extra data --extra dev
uv run pytest -q
uv run eot-mine --manifest data/raw/clips.jsonl --out data/mined --workers 8
uv run eot-apptek --root data/raw/apptek/hf --out data/mined/apptek --rule-version v2
uv run eot-train --samples data/mined/samples.jsonl --out runs/base --max-steps 800
uv run eot-soup --left runs/a/model.pt --right runs/b/model.pt --alpha 0.5 --samples ... --out runs/soup
uv run eot-export --checkpoint runs/base/model.pt --out exports/eot.onnx --validation-samples ...
docker build -t happyrobot-eot . && docker run --rm -p 8000:8000 -v "$PWD/exports:/models:ro" \
  -e EOT_ONNX=/models/eot.onnx happyrobot-eot eot-serve --host 0.0.0.0 --threads 4 --max-inflight 8
uv run eot-loadtest --url http://localhost:8000 --wav clip.wav --concurrency 1 4 8 --requests 400
```

The shipped artifact (`exports/opt_clean/eot.onnx` in the main repository) was trained and
exported with the baseline package; the serving front-end (`log_mel`, linear `resample`) is
unchanged here so that artifact remains valid.
