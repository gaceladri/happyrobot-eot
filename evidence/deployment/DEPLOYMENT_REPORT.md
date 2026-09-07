# Two inference deployments - local qualification

Date: 7 September 2026. This is an additive L041 deployment follow-up. Historical experiment verdicts and original measurements are unchanged.

## Selection

CPU: expanded-data distilled Whisper-base, optimized ONNX FP32. GPU: Cohere with **mixed FP16 / FP32**. Both quality and HTTP readouts below use the same artifact. Select the lowest observed batch-1 waiting at a 5% cutoff budget among accepted exports whose batch-1 policy remains within 5% at batches 1/4/8; use HTTP latency to break ties. This is a descriptive selection on the inspected benchmark.

| Actual runtime | Waiting at 5% (ms) | False cutoffs | HTTP p95 c1 (ms) | c4 (ms) | c8 (ms) |
|---|---:|---:|---:|---:|---:|
| Whisper CPU | 756.25 | 4.96% | 20.11 | 46.16 | 87.45 |
| Cohere TensorRT FP16 | 615.00 | 4.82% | 18.33 | 41.18 | 55.66 |
| Cohere TensorRT INT8 | 644.00 | 4.96% | 28.68 | 57.84 | 80.57 |

Whisper ONNX thread screen selected 8 intra-op threads using 160 requests at c8; the table uses a subsequent 400-request sweep at each load. CPU public quality used two threads; all 97 frozen feature probes were bit-identical at two and eight threads. GPU frontend uses two CPU threads, one GPU executor, maximum batch/admission 8 and a 2 ms batch wait.

Waiting is mean delay after actual speech completion, excluding inference. HTTP p95 is full local request response including decoding, frontend, execution, queue and transport. Their sum is not a measured end-to-end percentile. One warm desktop sweep does not establish sustained-arrival capacity, concurrent phone-call capacity or a production SLA. No Docker CPU quota was applied. The desktop and existing background services remained running.

## INT8 investigation and explicit acceptance

The historical 612 ms result belongs to native CUDA INT8, not TensorRT. That original batch-1 policy exceeded 5% when reused at batch 8. No original result has been overwritten.

1. TensorRT rejected the original unsigned activation quantizers: the GPU path requires signed symmetric INT8.
2. Recentring the frozen quantizers made the graph importable, but the intervening affine offsets prevented integer matrix fusion. That first engine used floating GEMMs and is not an INT8 throughput result.
3. Moving each constant offset correction after its linear product exposed integer work. The engine inspector verifies **191 INT8 GEMM layers** (remaining work includes FP32). Scales, weights and clipping ranges were not recalibrated on the benchmark.
4. The measured maximum probability deviation on 34 frozen training/synthetic probes is **0.036724210**. The original 0.02 gate failed. The user explicitly accepted this observed deviation for engine `b3841feace2c17c15d10c1383a16fdcd907e7f9ae7ffdc53bebcafc9e3fb3be6`. `int8-user-acceptance.json` records that exception; `fused-int8-original-numerical-verdict.json` preserves the original verdict. This is an explicit engineering acceptance, not a claim that the original tolerance passed.
5. The actual converted INT8 engine gives **644.00 ms** at **4.9645%** in batch 1. Its shared-policy waiting is **644.00 / 652.00 / 644.00 ms** for batches 1/4/8. The public batch-4 scores can differ from batch 1 by 0.09816; the measured cutoff rate under the shared policy stays within 5%. The 34-probe error is not a uniform bound over arbitrary audio.

Both INT8 and FP16 are retained as measured candidates. The selected runtime is chosen from their measured quality and response cost, not from the precision label or inherited 612 ms score.

## Runtime optimization

The CPU export fuses six attention blocks, twelve residual layer normalizations and six bias/GELU operations. It preserves input normalization and the four-second internal window. Maximum absolute output error was 0.000005662 over 97 frozen probes at batches 1/4/8 (291 comparisons), below 0.0001.

The floating TensorRT candidate merges 384 adapter projections, runs suitable operations in FP16, and protects normalizations, reductions, softmax, pooling and classification in FP32. TF32 is disabled. Builder optimization level 4, a 4 GiB workspace, zero auxiliary streams, and profiles 1/1/1, 2/4/4 and 5/8/8 target the serving batches. Each batch size 1 through 8 owns an execution context, fixed addresses, pinned transfer buffers and a captured CUDA graph. The original frontend is frozen and runs in FP32 with CPU-only PyTorch; the encoder and head execute in TensorRT.

The tuned floating engine's maximum probe difference is 0.00220412. Kernel-plus-transfer p95 is 10.23 / 16.77 / 28.26 ms for batches 1/4/8; these are not HTTP latencies. The first, less-tuned mixed engine measured 15.21 / 19.20 / 36.07 ms on the same probe timing protocol.

## Run the two inference images

From the main repository:

```bash
docker compose -f compose.inference.yaml up -d --build
curl -fsS http://127.0.0.1:8891/healthz
curl -fsS http://127.0.0.1:8892/healthz
curl --data-binary @clip.wav 'http://127.0.0.1:8891/v1/eot'
curl --data-binary @clip.wav 'http://127.0.0.1:8892/v1/eot'
```

Both APIs accept WAV or raw PCM16 (`?sr=16000`) and return `p_eot`, a threshold flag, model identity and timings. The threshold flag is not a complete end-of-turn policy. Apply the benchmark's action delay/timeout in the agent, and cancel stale decisions if speech resumes. The GPU service holds bounded admission until native work finishes even after a deadline expires.

Models are external read-only volumes. CPU needs `eot.onnx` plus `eot.json`; GPU needs `eot.plan`, `eot.json` and `frontend.pt`. The document ZIP includes code and receipts, not multi-gigabyte model weights. The local files are identified in `deployment.json`. TensorRT plans must be rebuilt and requalified on a different runtime/GPU; startup verifies engine and frontend hashes and refuses an unaccepted sidecar.

## Rebuild and reproduce

The selected source checkpoints and frozen validation panel were restored from the user's archive. That archive (`happyrobot-archive-2026-09-07`) is kept outside Git; its README and `SHA256SUMS` describe recovery (see `docs/reproduce-training.md`). The selected Whisper checkpoint hash is `d82bb9172b83251179b2dd759585532da4db72e38321160dd2b19542df406d4a`; the floating Cohere checkpoint hash is `98720ce3ee27a4298d47ae532dd06dcfc3893cd7928fff99c1213eff9ebe5ef4`.

```bash
uv run python scripts/export_models.py whisper --out artifacts/deployment/whisper-cpu
uv run python scripts/export_models.py cohere --out artifacts/deployment/cohere-gpu
python3 scripts/build_trt.py --onnx artifacts/deployment/cohere-gpu/eot.onnx --out artifacts/deployment/cohere-gpu/tuned.plan --precision fp16 --tuned
python3 scripts/verify_trt.py --engine artifacts/deployment/cohere-gpu/tuned.plan --source-dir artifacts/deployment/cohere-gpu --tolerance .02
```

The plan was built as `cohere-gpu/tuned.plan` and installed in the delivered bundle as `cohere-gpu/eot.plan`; the SHA-256 is unchanged.

The local build environment uses TensorRT 10.16.1.11, CUDA runtime 13.0.96, CUDA Python bindings 13.3.1, driver 595.84 and RTX 3090. Image IDs, engine SHA-256, frontend SHA-256 and full payload hashes are in `deployment.json`; image dependency locks are under `requirements/`. A build recipe does not promise identical TensorRT tactics across hardware or an independently rebuilt image.

`public_quality.py pack` freezes the original 400 English turns into 11,191 causal scoring prefixes. `infer` ran inside the actual respective Docker images. `metrics` uses the unmodified harness commit `6594d8b3b8af385b15f116dde310ce45af92d646`, dataset revision `ca9d98a9686b920a2d8c9eb984224ba9be74e4dd`, score point 200 ms, original HOLD filtering and original policy grid. The displayed curve is batch 1; the same 5% policy is checked at GPU batches 4/8. Policies were selected on this repeatedly inspected benchmark. There is no new blind test, manual-label campaign, benchmark-driven weight calibration or production promotion.

HTTP used the same frozen training WAV payload as the earlier Cohere measurement: PCM SHA-256 `a0579bc7560032f6137237177f09da25b7cb0d28c0f7ccff9f2c1b4bc25908cc`. `measure_http.py` records all attempts, errors, service identity, container image and GPU telemetry. Historical Whisper hardware measurements used a different audio payload and remain identified separately.

## Primary optimization references

- [NVIDIA performance and batching](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/performance/optimization.html)
- [NVIDIA precision and accuracy](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/accuracy-considerations.html)
- [NVIDIA builder optimization](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/performance/builder-performance.html)
- [NVIDIA quantized types and Q/DQ](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html)
- [ONNX Runtime graph optimization](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html)
