# Optimization results — 5 September 2026

**Selected CPU configuration: OPT clean mixed, fused FP32, four inference threads, maximum eight in-flight requests.** Final HTTP p95: 23.6 ms at concurrency 1, 49.0 ms at concurrency 4, 86.7 ms at concurrency 8, with zero errors across 1,200 requests. The exercise request-latency target is met on this host. This model remains an exercise/demo candidate; production conversational quality is not established.

The exercise asks for ideally **<100 ms per API request**. Endpointing delay is a separate quality metric. CPU timings below are full HTTP round trips against the Torch-free Docker image, with real eight-second PCM16 audio on an Intel i9-10900X. They exclude container startup, which is recorded separately. No inference response cache is used.

## Quality

English EoT Bench: pinned dataset revision, 400 turns. The primary local-model comparison uses a score at 200 ms, matching published Smart Turn. Cutoff is per eligible HOLD span, not per call. All runs must match the published prediction IDs, spans, timestamps, silence durations and labels before aggregation.

| Model | Cutoff @300 ms | Cutoff @600 ms | Delay @5% cutoff | Delay @10% cutoff |
|---|---:|---:|---:|---:|
| OPT mined (public initialization) | 31.9% | 14.2% | 1042 ms | 742 ms |
| OPT clean mixed (CPU-matched FP32) | 30.5% | 14.5% | 1031 ms | 732 ms |
| R2 ours: Smart Turn mined, ensemble detector | 55.6% | 15.7% | 1190 ms | 825 ms |
| OPT mixed (public initialization) | 29.8% | 14.0% | 1031 ms | 722 ms |
| OPT whole (public initialization) | 55.6% | 15.5% | 1080 ms | 782 ms |
| OPT mined with corrected FVAD | 33.6% | 14.2% | 1058 ms | 734 ms |
| R3 ours: R2 + AppTek oracle (capped) | 29.9% | 14.3% | 1184 ms | 798 ms |
| OPT student (public initialization) | 41.7% | 15.9% | 1253 ms | 812 ms |
| OPT clean mixed (onset-validated) | 30.5% | 14.5% | 1031 ms | 732 ms |
| R2' ours: Smart Turn mined, legacy detector | 55.6% | 14.8% | 1175 ms | 840 ms |
| OPT distilled student | 42.3% | 17.4% | 1275 ms | 834 ms |
| R4 ours: AppTek oracle only | 28.8% | 13.2% | 1122 ms | 746 ms |
| SmartTurn v3.2 | 35.2% | 14.8% | 1051 ms | 739 ms |

Intervals in `metrics/bootstrap*.json` resample whole turns. Paired comparisons use each model’s own fixed policy. Full-benchmark policy intervals are conditional and descriptive. `calibrated_test.json` selects policies on separate calibration turns; the overall benchmark has already been inspected, so those are exploratory splits, not a fresh blind test. For OPT mixed, a policy targeting 5% on calibration turns exceeded that target on the remaining turns; do not promise the optimized frontier as a production guarantee. The selected CPU-matched model’s calibration-selected 5% policy achieved 6.45% cutoff on test turns (95% interval 2.72–10.59%) at 1,034 ms mean delay.

## Independent Krisp evaluation

Krisp is never used for training. Metric version 3 counts timeout-triggered interruptions and treats firing exactly at speech resumption consistently. These policies use repeated causal scores, and intervals resample speakers. Policy sweeps on Krisp itself remain descriptive.

| Model | AUC at 200 ms | Cutoff @300 ms | Delay @5% cutoff | Metric version |
|---|---:|---:|---:|---:|
| opt_clean | 0.917 | 31.4% | 1035 ms | 3 |
| opt_mixed | 0.914 | 31.3% | 1109 ms | 3 |
| r0_vad | 0.500 | 99.9% | — | 3 |
| r1_smart_turn_public | 0.896 | 36.7% | 1438 ms | 3 |
| r2 | 0.855 | — | 1789 ms | 3 |
| r2_legacy | 0.849 | — | 1805 ms | 3 |
| r3 | 0.875 | — | 1597 ms | 3 |
| r4 | 0.904 | 34.3% | 1307 ms | 3 |

## HTTP serving measurements

| Artifact | Intra-op threads | Concurrency | Client p95 | Client p99 | Successful requests/s | Errors |
|---|---:|---:|---:|---:|---:|---:|
| exports/opt_clean/eot.onnx | 4 | 1 | 23.6 ms | 27.5 ms | 47.8 | 0 |
| exports/opt_clean/eot.onnx | 4 | 4 | 49.0 ms | 54.7 ms | 93.8 | 0 |
| exports/opt_clean/eot.onnx | 4 | 8 | 86.7 ms | 92.4 ms | 108.5 | 0 |
| exports/r4/eot.onnx | 1 | 1 | 79.7 ms | 81.6 ms | 15.7 | 0 |
| exports/r4/eot.onnx | 1 | 4 | 104.3 ms | 121.4 ms | 50.2 | 0 |
| exports/r4/eot.onnx | 1 | 8 | 102.1 ms | 107.9 ms | 90.7 | 0 |
| exports/r4/eot.onnx | 2 | 1 | 48.6 ms | 55.9 ms | 25.5 | 0 |
| exports/r4/eot.onnx | 2 | 4 | 75.8 ms | 81.0 ms | 61.3 | 0 |
| exports/r4/eot.onnx | 2 | 8 | 167.5 ms | 198.2 ms | 80.2 | 0 |
| exports/r4/eot.onnx | 4 | 1 | 36.6 ms | 49.3 ms | 39.5 | 0 |
| exports/r4/eot.onnx | 4 | 4 | 82.9 ms | 160.2 ms | 69.9 | 0 |
| exports/r4/eot.onnx | 4 | 8 | 143.0 ms | 172.2 ms | 82.6 | 0 |
| exports/opt_student/eot.onnx | 1 | 1 | 46.7 ms | 48.5 ms | 26.5 | 0 |
| exports/opt_student/eot.onnx | 1 | 4 | 67.2 ms | 75.4 ms | 89.6 | 0 |
| exports/opt_student/eot.onnx | 1 | 8 | 91.5 ms | 149.8 ms | 134.0 | 0 |
| exports/opt_student/eot.onnx | 2 | 1 | 21.8 ms | 26.5 ms | 49.4 | 0 |
| exports/opt_student/eot.onnx | 2 | 4 | 41.5 ms | 45.6 ms | 116.9 | 0 |
| exports/opt_student/eot.onnx | 2 | 8 | 57.9 ms | 61.4 ms | 165.0 | 0 |
| exports/opt_student/eot.onnx | 4 | 1 | 15.3 ms | 21.3 ms | 69.0 | 0 |
| exports/opt_student/eot.onnx | 4 | 4 | 32.0 ms | 34.7 ms | 141.8 | 0 |
| exports/opt_student/eot.onnx | 4 | 8 | 56.1 ms | 59.8 ms | 171.5 | 0 |
| exports/opt_mixed/eot.onnx | 1 | 1 | 57.0 ms | 60.1 ms | 18.4 | 0 |
| exports/opt_mixed/eot.onnx | 1 | 4 | 68.1 ms | 70.8 ms | 65.1 | 0 |
| exports/opt_mixed/eot.onnx | 1 | 8 | 84.4 ms | 87.4 ms | 102.5 | 0 |
| exports/opt_mixed/eot.onnx | 2 | 1 | 36.1 ms | 37.7 ms | 31.8 | 0 |
| exports/opt_mixed/eot.onnx | 2 | 4 | 58.3 ms | 64.3 ms | 74.0 | 0 |
| exports/opt_mixed/eot.onnx | 2 | 8 | 86.8 ms | 91.3 ms | 102.9 | 0 |
| exports/opt_mixed/eot.onnx | 4 | 1 | 25.8 ms | 30.3 ms | 47.1 | 0 |
| exports/opt_mixed/eot.onnx | 4 | 4 | 47.8 ms | 55.6 ms | 92.7 | 0 |
| exports/opt_mixed/eot.onnx | 4 | 8 | 86.5 ms | 90.2 ms | 109.3 | 0 |

## What changed and why

- Fixed revision-qualified benchmark discovery and preserved each published model’s scoring mode. Added prediction-key alignment checks and both-mode bootstrap parity tests.
- Fixed Krisp timeout accounting; incomplete scoring writes a `.partial` file and publishes only on completion.
- Future-speech masks now operate per horizon. Clip completion no longer labels unobserved future silence. AppTek can use its observed next onset independently of EOT. Old unobserved EOT targets are defensively masked.
- Pause cuts must lie after VAD agreement and before the earliest speech onset. Old negative-time-to-onset examples are excluded during training and counted in provenance. Historical artifacts are retained.
- Follow-up training caps each Smart Turn source at 1,500 high-confidence samples. The mixed manifest’s largest source is 12.4%; human auditing remains pending. Source grouping prevents prefix leakage.
- Imported pinned public Smart Turn weights with maximum probability drift below 1e-7. Fine-tuning, whole-clip controls, two-layer pruning, distillation and corrected-FVAD experiments are recorded separately. The first two-layer and distilled models lost quality and were rejected despite good CPU timing.
- Vectorized the Whisper FFT frontend; feature parity is tested on silence and multiple audio lengths, with/without normalization. ONNX attention/LayerNorm/GELU fusion retains FP32 probabilities. INT8 candidates that exceeded the 0.02 probability-drift gate were rejected, not shipped.
- Export validation covers batch sizes 1/2/8, real clean and telephony audio, and decision flips. Artifacts carry checksums, preprocessing and head configuration; the service verifies metadata at startup.
- Bounded admission and native jobs retain capacity after HTTP cancellation. Deadlines include upload, decode and inference. Non-finite audio is rejected. Shutdown closes the executor.
- The streaming policy now works without model responses, ignores stale generations and idle silence, latches a score until its action delay, and prevents duplicate in-flight inference. Use score-point mode for the primary calibrated policy; repeated scoring is a distinct evaluated mode.

## Limits and release status

The request-latency requirement is demonstrated locally; this is not evidence that the model safely handles production turn taking. The low-cutoff endpointing-delay tradeoff remains unsatisfactory for a fast production voice agent. AppTek labels describe observed role-play behavior, not certain intent. The 175-clip blind human audit has zero completed judgments; confidence filters cannot replace it. AppTek’s training-source suitability also requires resolution before a production model. Context and Spanish benefits are not claimed. `eval/release_check.json` separates demo readiness from these production limits.

Whisper generic or public Smart Turn initialization may already have seen the Smart Turn training sources used in local dev; near-perfect whole-clip dev AUC is not an independent generalization result. External tests and the stated distribution limits carry more weight.

## Reproduction

```bash
.venv/bin/python scripts/import_smart_turn.py
.venv/bin/python scripts/prepare_optimization.py
bash scripts/optimize_train.sh
bash scripts/optimize_evaluate.sh
bash scripts/optimize_distill.sh
bash scripts/optimize_fvad.sh
bash scripts/optimize_clean.sh
bash scripts/repair_evaluation.sh
bash scripts/optimize_external.sh
bash scripts/final_validate.sh
```

Training commands use new run directories and refuse to silently overwrite completed runs. Use a fresh checkout/output location to repeat the full sequence; keep the pinned data acquisitions. Artifact filenames, SHA-256 values, sample-manifest hashes, source-code hashes and software versions are retained with each run. Benchmark parameters and rejected exports remain in logs.

Primary implementation references: [Smart Turn model](https://huggingface.co/pipecat-ai/smart-turn-v3), [ONNX Runtime threading](https://onnxruntime.ai/docs/performance/tune-performance/threading.html), [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).

## Held-out AppTek diagnostic

Selected model: 13086 cuts at 200 ms; room-tone AUC 0.691, raw-audio AUC 0.705. At threshold 0.5, HOLD false positives are 16.2% and EOT detection is 34.3%. These GPU traces predate the final TF32 precision correction. These heuristic role-play labels and fixed-score detection rates are not conversational latency metrics.

## Selected artifact and validation

The selected artifact is `exports/opt_clean/eot.onnx`: four encoder layers, FP32 fused attention, audio normalization enabled, context and FVAD disabled. Its SHA-256 is `1ee1c00dca9a9447d6fa531e325d985aa0e8973acff5d9791b9ab3c0e5bf679e`. The final benchmark is labelled **CPU-matched FP32**; the earlier onset-validated trace used CUDA default TF32 convolutions. TF32 caused up to 0.000873 probability drift across 1,105 score points despite no flips at 0.5. This strict parity failure is retained in `eval/deployment_point_parity_tf32_rejected.json`. Evaluation now disables TF32 and has a distinct adapter ID. Final score parity passed on all 1,105 score points: maximum drift 5.55e-6, zero flips at 0.5; see `eval/deployment_point_parity.json`. Export validation independently passed 108 examples across clean/telephony/synthetic audio with maximum drift 2.78e-6. The feature and request lifecycle regression suite passed 56 tests; the harness bootstrap test also passed separately in its pinned environment.

## Monitoring and rollout design

Record request p50/p95/p99, decode/features/inference timings, 409/503 rates and in-flight work by artifact SHA and thread setting. Alert on p95 above 100 ms or admission failures, and verify capacity on the target deployment hardware. The local load test is closed-loop, uses one real eight-second clip, excludes cold start and does not demonstrate cross-region network latency or burst capacity.

For turn-taking, log the VAD boundary, score, policy threshold/delay/timeout, speech-resumption event and model version. Use reviewed call samples to estimate cutoffs, missed completions and task-completion impact. Caller protests and post-interruption speech are biased proxies; absence of a protest is not a correct-label signal. A small randomized longer-timeout comparison can help measure intervention bias. Recalibrate on representative calls and complete the human audit before any production rollout; do not train automatically on the current policy’s own decisions.
