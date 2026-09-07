# Serving integration and monitoring

The repository implements the stateless end-of-turn scorer and offline evaluation. The monitoring and rollout below are a proposed production integration, not an already installed observability stack.

## API contract

`POST /v1/eot` accepts a WAV body, or PCM16 little-endian audio with `sr=16000` (supported declared rates: 8–192 kHz). `threshold` is in [0,1]; `deadline_ms` is optional and positive. Both selected models ignore `agent_text` and report `context_used=false`.

Both services share one implementation of the contract (`eot.serving.http`). A successful response contains `p_eot`, `decision`, `threshold`, `model_sha`, `version`, audio duration, `timings_ms` and `batch_size` (how many requests shared the GPU forward; always 1 on CPU). `GET /healthz` exposes model/runtime identity, the admission settings in force and `source_sha256` (the image's code digest, `null` outside the containers); GPU health includes observed batch-size counts. These diagnostics are not a Prometheus endpoint or persisted metrics store.

The application maintains the streaming policy: VAD/pause eligibility, causal audio windows, threshold crossing, action delay, maximum timeout, and cancellation when new speech arrives. A late response must not trigger a reply after the caller has resumed. The offline policy reference is `src/eot/eval/policy.py`; inference alone does not implement the full voice interaction.

Both selected models use `score_mode="score_point"`: score once at 200 ms of an eligible pause, then advance the action-delay and timeout timers without rescoring. Use `Endpointer(..., score_mode="score_point")`; its generation identifier invalidates a result when speech resumes. Repeated scoring every 100 ms (`score_mode="max"`) changes the policy and must be evaluated and calibrated separately.

The selected benchmark policies use threshold 0.26, action delay 0.5 s and timeout 3.0 s for Whisper; 0.29, 0.5 s and 2.5 s for Cohere. Those are benchmark-fitted operating points, not production defaults validated on real calls.

## Configuration

Both entry points (`eot-serve`, `eot-serve-gpu`) take `--host`, `--port`, `--threads`, `--max-inflight` and `--max-body-bytes`; the GPU service adds `--wait-ms` (batching window, 0-10 ms) and `--no-graphs`. Environment variables supply the defaults the containers use:

| Variable | Service | Default | Meaning |
| --- | --- | --- | --- |
| `EOT_ONNX` | CPU | required (or `--onnx`) | Path to `eot.onnx`; its `eot.json` sidecar must exist and record `accepted: true` and the file's sha256. |
| `EOT_TRT_ENGINE` | GPU | required (or `--engine`) | Path to `eot.plan`; `eot.json` and `frontend.pt` must sit beside it with matching hashes. |
| `EOT_THREADS` | both | `2` | onnxruntime intra-op threads (CPU) / torch front-end threads (GPU); overridden by `--threads`. |
| `EOT_MAX_BODY_BYTES` | both | `2000000` | Request body limit; overridden by `--max-body-bytes`. |

`uv run eot-artifacts verify` checks the mounted bundle against `configs/inference/selected.json` before a rollout; `install --source ... [--model NAME]` copies a bundle without replacing different files. Model names are validated against the manifest.

## Failure behavior

- `400` / `422`: malformed audio or invalid parameters. Fix the request; do not retry it unchanged.
- `408`: the body did not arrive within 2 s. Slow or stalled uploads are rejected before they take an admission slot.
- `413`: body exceeds the limit (2 MB by default). Keep a bounded rolling audio window.
- `409`: the request deadline expired. Discard the result and retain the caller's conservative timeout fallback.
- `503`: all admission slots (`--max-inflight`, default eight) are occupied, or the GPU worker has stopped. Back off or use the fallback; do not create an unbounded retry queue.
- `5xx`, unhealthy service or model identity mismatch: exclude the instance and revert to the last qualified configuration.

Native work cannot be safely cancelled mid-inference. A timed-out GPU operation retains its admission slot until completion. The service drains outstanding work before freeing CUDA resources on shutdown. These behaviors are covered by tests.

## What to observe and what to do

**Service health:** request rate, error/deadline/rejection rates, in-flight work, effective GPU batch sizes, warm p50/p95/p99 HTTP time, decode/frontend/inference components, startup time, CPU/GPU utilization and memory. Export response/health fields through the hosting application's metrics collector and correlate them with the model hash. If deadline misses or saturation rise, reduce admission, scale capacity, or fall back; lowering an EoT threshold does not fix compute saturation.

**Conversation outcomes:** false cutoffs, waiting after true turn ends, fallback-timeout rate, caller restarts/overlap after the agent begins, and abandoned turns. Service probability distributions alone cannot establish cutoff quality. Estimate outcomes from consented, access-controlled call samples with an explicit annotation protocol, and track uncertainty and sample size.

**Input and quality drift:** language/accent, channel and sample rate, noise/SNR, pause duration, input length and score distributions. Compare cohorts against a versioned reference and review the associated call outcomes before retraining. A shifted score histogram is a diagnostic signal, not proof that calibration worsened.

**Release decisions:** shadow the candidate first, compare it on paired traffic, then canary with an explicit rollback budget. Define the quality and latency budgets before inspecting the canary. Log deployment version, policy version and model identity together so an incident can be reproduced.

## Feedback into training

Review representative failures and uncertain pauses, preserve source-group splits, and keep a fresh time-separated holdout. Human review here is a proposed next step; the completed research calibrated heuristics against already annotated public data and did not create a new manually labeled corpus.

Validate label rules before expanding the corpus, retrain against a versioned manifest, regenerate teacher targets for the exact student views, and repeat numerical fidelity, cutoff/waiting quality, batch consistency and representative-load HTTP checks. Keep rejected candidates and reasons. Add new evidence without rewriting earlier failures or presenting benchmark reuse as independent validation.

Protect audio and transcripts through consent, minimal retention and restricted access in the production application. The inference response and aggregate metrics do not require storing raw audio. The exercise service binds to localhost; authentication, TLS, centralized metrics and call orchestration belong to the deployment integration.
