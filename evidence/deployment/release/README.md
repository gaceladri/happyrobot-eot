# Historical integration verification — 7 September 2026

This directory records validation of the integration at `c22e84a`, before the repository cleanup and shared HTTP refactor. It does not validate the later Python/NumPy environment or service source. Current post-review verification is recorded separately in [../post-review/README.md](../post-review/README.md).
The original deployment receipts and presentation figures remain unchanged.

- The five selected inference files match `configs/inference/selected.json`.
- Both images were rebuilt from the integrated source and pinned base images, then started through Compose with read-only model mounts.
- The CPU image has no PyTorch installation. The GPU image uses CPU-only PyTorch 2.14.0 for the original frontend and TensorRT 10.16.1.11 for the encoder/head.
- All 227 package and selected research-contract tests passed in a newly created, hash-locked CPU environment. The same 227 passed in the existing development environment.
- Before/after HTTP verification used 34 frozen probes, sent sequentially with identical PCM payloads. Both models had maximum probability difference **0.0**, within the predefined 1e-5 tolerance. This is a release-regression check, not a new cutoff-quality benchmark.
- New sequential warm HTTP sweeps used 400 requests at each concurrency and the same frozen training payload. Both services returned zero errors.

## Rebuilt-image latency

- **Whisper**, HTTP p95 at concurrency 1/4/8: **23.40 / 51.78 / 99.90 ms**. Image `sha256:14390c495526c7937e39bd4213f6179b509ba97975331c3fcd57b78bf1603026`.
- **Cohere**, HTTP p95 at concurrency 1/4/8: **18.09 / 39.90 / 53.63 ms**. Image `sha256:6686c3c60410f627d3b2937ece36fedbab80f8a07a7b23fead8ce809954d462a`.

The CPU c8 result is close to 100 ms. The original reported 87.45 ms and the new 99.90 ms are observations from different runs/images of the same model artifact, not a guaranteed latency bound. Do not replace the original receipt with a more favorable rerun. The GPU c8 recheck is 53.63 ms versus the original 55.66 ms. These measurements do not establish concurrent-call capacity or production SLO compliance.

## Repeat the checks

Run the package tests from the main README. Use `scripts/verify_http_parity.py --inputs ... --out ...` to capture a reference on frozen training probes; after a release, pass `--reference` and a new `--out` to compare. The panel hash, every payload hash and model identity must match. The original panel is a separate artifact, not a training dataset embedded in Git.

`scripts/measure_http.py` repeats the fixed-payload latency sweep. The release receipts identify their rebuilt Docker images; earlier receipts identify the original measured images. No training, calibration, model-weight change, or new full public-quality sweep occurred during this integration.
