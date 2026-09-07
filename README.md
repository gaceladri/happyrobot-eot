# Knowing when to answer

End-of-turn detection for a voice agent: decide, from the audio alone, whether the caller has finished their turn, so the agent answers without cutting the caller off and without waiting longer than necessary.

This repository is the submission for the HappyRobot AI/ML engineering exercise. It contains the training and evaluation code, two inference services with their Dockerfiles, the measurements behind the model selection, the short solution document and the presentation. Model weights are distributed separately.

## Results

Two models were selected, one per deployment target. Both are scored on the public EoT Bench English validation set with the benchmark's own policy grid at a 5 % false-cutoff budget; the historical HTTP measurements below cover the full request (decode, features, inference, response) at concurrency 1 / 4 / 8, 400 warm requests per level, on the local machine.

| Model | Runtime | Waiting after the caller stops (mean, 5 % cutoffs) | HTTP p95 c1 / c4 / c8 |
|---|---|---|---|
| Whisper-base, distilled from Cohere teachers | ONNX Runtime FP32, 8 threads, CPU (i9-10900X) | 756 ms | 20.1 / 46.2 / 87.4 ms |
| Cohere Transcribe encoder, LoRA fine-tuned | TensorRT mixed FP16/FP32, GPU (RTX 3090) | 615 ms | 18.3 / 41.2 / 55.7 ms |

The [post-review verification](evidence/deployment/post-review/README.md) qualifies the current source and rebuilt images, with frozen-probe parity and new HTTP measurements. The earlier [integration recheck](evidence/deployment/release/README.md), at 99.9 ms CPU / 53.6 ms GPU at concurrency 8, predates the cleanup and is retained as historical evidence. Waiting time is an endpointing-quality metric and HTTP latency is an execution metric; they are separate measurements. The public benchmark was inspected repeatedly and its policies were fitted on the same data, so these are local comparisons, not a blind test. For reference, the project's first Whisper-tiny baseline waits 1,031 ms under the same rule and the best public model we compared against (LiveKit v1) waits 543 ms.

## Start here

1. [Solution brief](docs/solution-brief.md) (four pages; also as [PDF](docs/solution-brief.pdf)): approach, data and labeling, model selection, serving, monitoring plan and the discussion questions.
2. [Presentation](docs/presentation/eot-presentation.html): open locally, 12 slides plus a technical appendix. The video follows [this script](docs/video-script.txt).
3. [Reproduce inference and evaluation](docs/reproduce-inference.md): install the model bundle, run both services, repeat the latency and quality measurements.
4. [Reproduce data and training](docs/reproduce-training.md): labeling decisions, teacher fine-tuning, distillation, and where the archived experiment runner lives.
5. [Serving integration and monitoring](docs/operations.md): API contract, failure behaviour, what to observe in production and how to feed calls back into training.
6. [Discussion questions](docs/discussion.md): limits of the solution, audio versus text versus VAD, using a transcriber for this task, integration in a voice agent.
7. [Evidence](evidence/REPORT.md): the final experiment report, the [deployment receipts](evidence/deployment/DEPLOYMENT_REPORT.md) and the measurements behind every number above.

## Requirements

- Python 3.13 and [uv](https://docs.astral.sh/uv/) for development; Docker with Compose 2.30+ to run the services.
- The model bundle (5.3 GB: ONNX model, TensorRT plan, sidecars), downloaded by `scripts/download_weights.py` from a public read-only link; every hash is pinned in [configs/inference/selected.json](configs/inference/selected.json).
- GPU service only: an NVIDIA driver with the container toolkit and an RTX 3090-class GPU. The TensorRT plan is specific to that hardware and runtime; another GPU needs a rebuild and requalification ([how](docs/reproduce-inference.md#4-rebuild-and-qualify-exports)).

## Get the model weights

The weights are one 5.3 GB tar on Google Drive, readable by anyone with the link, no account needed:
[happyrobot-eot-weights-316125d.tar](https://drive.google.com/file/d/1Knd2B29FpMuOiUNmvyOV1Nf_uDj2JgI9/view?usp=sharing)
(SHA-256 `8c6ca046bde26f9a3860c5fe7e6ec0100cf6c6a70166beda95090814f1eb47b5`). It contains exactly the five
files listed in [configs/inference/selected.json](configs/inference/selected.json): the Whisper ONNX model and
sidecar, and the Cohere TensorRT plan, sidecar and front-end.

```bash
uv sync --locked                                   # or any Python 3.11+ with the package installed
uv run python scripts/download_weights.py          # downloads (resumable), verifies the tar and every file, installs to artifacts/deployment
uv run python scripts/download_weights.py --model whisper-cpu   # CPU model only
```

The script pins the link and the hashes, so a tampered or truncated download is refused. Downloaded by
hand instead (browser, `gdown`, `curl -L`)? Extract the tar and install it with
`uv run eot-artifacts install --source model-bundle`; `uv run eot-artifacts verify` re-checks the installed files.

## Run the services

```bash
docker compose -f compose.inference.yaml up -d --build whisper-cpu
curl -fsS http://127.0.0.1:8891/healthz
# On a compatible GPU host:
docker compose -f compose.inference.yaml up -d --build cohere-gpu
curl -fsS http://127.0.0.1:8892/healthz
```

For a CPU-only bundle add `--model whisper-cpu` to the artifact commands; `EOT_MODEL_ROOT` points Compose at another bundle location. Both services accept a WAV body, or little-endian PCM16 with `?sr=`:

```bash
curl -fsS --data-binary @clip.wav 'http://127.0.0.1:8891/v1/eot?threshold=0.26&deadline_ms=100'
```

```json
{"p_eot": 0.91, "decision": "eot", "threshold": 0.26, "p_fvad": {}, "audio_seconds": 3.2,
 "timings_ms": {"decode": 0.4, "features": 2.1, "inference": 18.7, "total": 21.9}, "model_sha": "b35854bfa525"}
```

The API is a stateless scorer over the last 8 s of the caller's audio. The selected benchmark protocol scores once per eligible pause, at 200 ms of silence (`score_mode="score_point"`). The streaming caller applies the action delay and timeout, and invalidates pending scores when speech resumes, as described in [operations.md](docs/operations.md). Scoring repeatedly every 100 ms is a different policy and needs its own evaluation; a threshold response alone does not reproduce the benchmark's endpointing behaviour.

## Develop and verify

```bash
uv sync --locked --extra dev --extra data
uv run pytest -q          # synthetic fixtures only: no checkpoints or audio are downloaded
uv tool run ruff@0.16.6 check src tests scripts
```

CI runs the same suite in a hash-locked CPU environment (`requirements/test-cpu.lock`). The inference images have their own locked dependencies; the CPU image does not install PyTorch. Training and export additionally need `--extra backbone` (Cohere teacher), `--extra gpu` (TensorRT build) and a CUDA machine.

## Repository map

```text
src/eot/          the package, ordered by pipeline stage (each stage imports only earlier ones; tests enforce it)
  audio/          waveform conventions, Whisper log-mel front-end, pause detectors, augmentation (room tone, telephony, tail texture)
  labeling/       prefix mining around pauses, dual-channel oracle labels, event targets, label-quality audits
  data/           Smart Turn acquisition, source-grouped splits, torch datasets, cached teacher logits
  modeling/       Whisper model and trainer, LoRA, Cohere backbone and trainer, distillation, weight soups, ONNX export
  eval/           causal endpointing policy, EoT Bench adapters, Krisp test set, held-out reports
  serving/        CPU (ONNX Runtime) and GPU (TensorRT) FastAPI services, artifact verification, load test
tests/            synthetic contract and regression tests
scripts/          download the weights, export models, build and verify the TensorRT plan, measure HTTP, recompute public quality, release parity
configs/          selected inference artifacts (hashes) and the frozen final training protocol
docs/             solution brief, presentation, discussion answers, reproduction guides, operations design, video script
evidence/         final experiment report and deployment receipts (JSON, CSV, figures)
requirements/     locked environments: CPU serving, GPU serving, CI tests
```

This public repository contains a single delivery snapshot. The complete research history (41 registered experiments, protocols, reports and runners, plus the earlier flat package) is retained in a separate recoverable archive and can be supplied on request. See [training reproduction](docs/reproduce-training.md) for the archived campaign and its data requirements.

## What the work established

- **Labels come from real pauses.** Training rows are causal audio prefixes cut inside pauses of real conversations, labelled by what the speaker did next; heuristics were calibrated against existing human annotations rather than by annotating a new corpus.
- **Each ingredient earned its place in a controlled experiment** before the final combination: a larger encoder only helps when it is fine-tuned (Cohere LoRA, 789 to 636 ms); rewriting the pause tail with a label-independent texture does nothing alone but adds 48 ms of improvement on top of distillation; distillation from the Cohere teacher moves Whisper-base from 897 to 802 ms.
- **The final recipe combines expanded data, continued teachers, an ensemble of teacher logits, dense pause supervision and multiple passes.** Its gains cannot be attributed to single components; the controlled results above are the attributable evidence.
- **Quality and compute were decided together.** FP16 TensorRT beat the investigated INT8 engine on both quality and latency; the accepted numerical deviations and the rejected variants are recorded, not hidden.

## Limitations

The benchmark was reused across many decisions and its policies were tuned on it, so the numbers are optimistic relative to a blind test. The CPU service sits near the 100 ms target at concurrency 8. The TensorRT plan is host-specific. Monitoring, live-call validation and rollout controls are proposed in the operations document; the repository implements inference and evaluation, not a production monitoring stack.

Author: Adrián Brunetto (adrianbrunetto1@gmail.com). Submitted for the HappyRobot AI/ML engineering exercise, September 2026.
