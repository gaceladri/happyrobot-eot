# Post-review deployment verification — 7 September 2026

## Scope and identity

This verification follows the shared HTTP refactor and the eight pre-submission review fixes. It qualifies the runtime source identified by [source.json](source.json): base commit `316125d7a581ac5982e2233bf21e473863ed296b`, the exact [source patch](source.patch), SHA-256 for every tracked source/build input, and the resulting Docker image/source digests. Both running image source digests were independently recomputed from local source and matched. Model weights and the selected-artifact manifest were not changed.

The earlier [integration receipts](../release/README.md) predate this refactor. Their numbers and the original presentation figures remain historical observations, not measurements of these images.

## Completed checks

- 229 package tests passed in the development environment and in the separately installed, hash-locked CPU environment. The last change translated report text only; both report tests passed again afterward. Lint, format and `uv lock --check --offline` passed.
- Both Docker images rebuilt and started through Compose with the selected read-only model mounts. CPU: Python 3.13.15, NumPy 2.5.2 and ONNX Runtime 1.29.0, no PyTorch. GPU: the selected TensorRT mixed FP16/FP32 plan on the local RTX 3090, two CPU frontend threads, one GPU execution worker, CUDA graphs and a 2 ms batching window.
- [Frozen HTTP parity](http-after.json) used the same 34 probes, panel hash, PCM payload hashes and model identities as `../release/http-after.json`. The predefined probability tolerance was `1e-5`. Maximum absolute difference: Whisper **5.662441253662109e-7**; Cohere **0.0**. Both passed. [http-before.json](http-before.json) establishes that the services running before the rebuild matched the historical reference exactly.
- This is a release regression check, not a new public cutoff sweep. The historical 756.25 / 615.00 ms waiting results remain attributed to their original benchmark receipts. Probe agreement does not establish equality on every possible input.

## Final-image HTTP latency

- **Whisper CPU**, full HTTP p95 at concurrency 1/4/8: **20.24 / 44.83 / 92.12 ms**; c8 p99: **107.43 ms**. Zero errors in 1,200 requests.
- **Cohere GPU**, full HTTP p95 at concurrency 1/4/8: **17.96 / 40.96 / 62.94 ms**; c8 p99: **68.04 ms**. Zero errors in 1,200 requests.

The original frozen 8 s PCM16 training payload was used (SHA-256 `a0579bc7560032f6137237177f09da25b7cb0d28c0f7ccff9f2c1b4bc25908cc`). Each service ran sequentially through concurrency 1, 4 and 8, with 400 warm requests per level: 2,400 requests and zero errors across the final images. Every p95 was below 100 ms; this is not a guarantee for every request. CPU c8 p99 exceeds 100 ms, so a per-request 100 ms deadline can still cause fallbacks.

No CPU quota was imposed. The GPU process inventory contained only the selected service during measurement. Other desktop applications were running. These loopback measurements do not establish production capacity or an end-to-end voice latency SLO.

Receipts: [Whisper HTTP](whisper-http.json), [Cohere HTTP](cohere-http.json). They include container identity, health fields, all-attempt latency, errors and GPU telemetry.

## Retained first pass

The [first-pass receipts](first-pass/source.json) were taken before the final report-language cleanup (no serving logic changed afterward): CPU p95 **20.10 / 45.05 / 84.37 ms**, GPU **18.00 / 40.49 / 64.19 ms**. Both had zero errors and passed parity. They are retained to show run-to-run variation; the results above belong to the final image identities, not the most favorable run.

## Reproduce

Build and start both services with `docker compose -f compose.inference.yaml up -d --build`. Restore the frozen probe panel and measurement WAV as described in [inference reproduction](../../../docs/reproduce-inference.md). Run `scripts/verify_http_parity.py` with the original panel, `--reference evidence/deployment/post-review/http-after.json`, `--tolerance 0.00001`, and a fresh output path. Run `scripts/measure_http.py` for each service sequentially with 400 requests at concurrency 1/4/8, also writing new output paths.

The probe panel and historical WAV are separate research artifacts, not included in the five-file inference bundle. Preserve old receipts and compare panel/payload/model/source identities when evaluating a subsequent release. [inventory.json](inventory.json) records SHA-256 hashes of this verification's files.
