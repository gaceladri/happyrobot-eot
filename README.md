# HappyRobot end-of-turn detector

A compact end-of-turn (EoT) system for voice agents: Smart Turn data acquisition,
pause-level prefix mining, Whisper-Tiny fine-tuning, LiveKit EoT Bench evaluation,
ONNX export, CPU FastAPI serving, and load testing.

The main experimental claim is deliberately narrow: aligning training examples
with the causal pause decisions made in production should reduce false cutoffs
without adding a large model to the latency-critical path.


## Measured optimized configuration

Use the onset-validated full model in `exports/opt_clean/eot.onnx`. The optimization report
[output/OPTIMIZATION_RESULTS.md](output/OPTIMIZATION_RESULTS.md) supersedes the earlier smoke-test
numbers and records both request latency and the remaining cutoff/delay tradeoff. The release
check in `eval/release_check.json` distinguishes a runnable exercise demo from production readiness.

```bash
docker build -t happyrobot-eot:optimized .
docker run --rm -p 8000:8000 \
  -v "$PWD/exports/opt_clean:/models:ro" -e EOT_ONNX=/models/eot.onnx \
  happyrobot-eot:optimized eot-serve --host 0.0.0.0 --port 8000 --threads 4 --max-inflight 8
# In a second terminal, use an actual WAV from your local data:
uv run eot-loadtest --url http://localhost:8000 --wav /path/to/clip.wav \
  --concurrency 1 4 8 --requests 400 --out eval/loadtest.json
```

`POST /v1/eot?sr=16000&deadline_ms=100` accepts raw PCM16. Its `decision` is a classification,
not a stateful commitment to speak. Integrate `eot.policy.Endpointer` with the live VAD:
feed `on_vad`, reserve work with `request_score`, and return its generation with `on_score`.
Call `on_tick` each frame for action-delay and timeout decisions, and `on_error` when inference
fails. `reset` starts a new user turn. The default scores once at 200 ms of silence and latches
that result until the policy's action delay, matching the primary evaluation protocol.
Use `score_mode="max"` only with a separately evaluated repeated-score policy. Actual conversational
latency additionally includes VAD boundary detection, inference and downstream LLM/TTS work.

Training and external evaluation remain separate. New runs have source/sample/checkpoint hashes;
Krisp is test-only. The human listening audit remains pending and AppTek labels are heuristic.
Neither multilingual quality nor a production-safe cutoff rate is implied by CPU timing.

## Project map

- `src/eot/acquire.py`: resumable Smart Turn acquisition.
- `src/eot/prefix_mining.py`: causal HOLD/EOT examples and future-speech targets.
- `src/eot/train.py`: full Whisper-Tiny fine-tune and recoverable checkpoints.
- `src/eot/eotbench_adapter.py`: official-harness adapter.
- `src/eot/export_onnx.py`: ONNX export with enforced parity.
- `src/eot/serve.py`: Torch-free CPU service.
- `src/eot/load_test.py`: concurrency/latency stress test.
- `Dockerfile.train`: GPU Pod image; preserves RunPod's Jupyter/SSH lifecycle.
- `Dockerfile`: small CPU inference image.
- `requirements/*.lock`: Python 3.12/linux-amd64 dependency snapshots used by the images.
- `RUNPOD.md`: build, Pod, training, retrieval, and cleanup runbook.

## Local development

```bash
uv sync --extra data --extra dev --extra tracking
uv run pytest -q
```

Typical local pipeline after a manifest exists:

```bash
uv run eot-mine --manifest data/raw/clips.jsonl --out data/mined --resume
uv run eot-train --samples data/mined/samples.jsonl --out runs/base --epochs 3
uv run eot-export --checkpoint runs/base/model.pt --out artifacts/eot.onnx
```

On an NVIDIA GPU, the measured local fast path keeps the original batch size
while enabling four audio workers, fused AdamW, and a compiled training forward:

```bash
uv run eot-train \
  --samples data/mined/samples.jsonl \
  --out runs/base \
  --epochs 3 \
  --batch-size 32 \
  --workers 4 \
  --compile-model \
  --wandb-project happyrobot-eot
```

Fused AdamW is selected automatically on CUDA. W&B logging is opt-in and stores
one stable run id under the output directory so checkpoint resumes update the
same remote run. Use `eot-benchmark-mfu` to repeat the local MFU/throughput grid.

The Smart Turn corpus does not contain paired previous-agent text. The primary
training run must therefore remain audio-only; `--use-context` is an ablation for
a future genuinely paired dataset, not a switch to enable on Smart Turn.

For the complete RunPod workflow, see [RUNPOD.md](RUNPOD.md).

## Clone on the GPU machine

```bash
git clone https://github.com/gaceladri/happyrobot-eot.git
cd happyrobot-eot
```

The repository intentionally excludes datasets, checkpoints, ONNX exports,
caches, and local secrets. The pinned acquisition and training workflow rebuilds
those under the persistent `/workspace/eot` volume; follow `RUNPOD.md` from the
new machine.
