# Bitácora

## 2026-09-05 — Program established

The accepted demo incumbent is OPT clean mixed (onset-validated public Smart Turn fine-tune),
fused FP32, no FVAD/context. It passes local request latency and 1,105-point CPU/GPU parity,
but has not met the SOTA or production quality objective. Existing experiments are imported as
historical evidence, not newly preregistered trials. Human label judgments remain unfilled.

The strongest published same-protocol reference is LiveKit v1: 543 ms delay at 5% cutoff.
Both reference and candidate score at 200 ms. Commercial streaming models use distinct modes
and remain contextual comparisons, not silently pooled protocol controls.

E001 is motivated by dev AUC still improving at 800 steps (0.8632 -> 0.8710 -> 0.8724).
Extending training is an untried full-model budget/schedule test; failed student distillation
is not evidence against this hypothesis. No external benchmark is needed to reject it if
source-held-out dev does not improve.

## E001 — rejected at the development gate

3,200 updates completed in 440 s of measured training time. Best source-held-out dev AUC
was 0.87044 at step 954, below the incumbent’s 0.87239; final AUC fell to 0.84079 while
training loss fell. This longer-budget/cosine-schedule experiment overfit. No LiveKit benchmark
was accessed and the incumbent remains unchanged. Do not repeat a longer unregularized schedule
on this manifest without new evidence. W&B: https://wandb.ai/adrianbrunetto1-reshape/happyrobot-eot/runs/13e9f0d0

E002 now tests training-only selection at 200 ms with all original dev rows preserved. E003
will test full-model public-teacher regularization; E004 is a separate zero-initialized local
acoustic residual branch. All have separate code/config IDs and the same 800-update budget.

## Development-set limitation

The frozen source split has 1,818 dev rows across four Smart Turn sources, of which 1,478
come from liva_1. It has no AppTek development examples. We retain this split for matched
initial ablations and report its limitation: AUC gains there are a screening signal, not evidence
of real-call generalization. Future data expansions must preserve these dev sources and use a
separately registered conversational development slice, never repurpose benchmark labels.

## E002 — rejected at the development gate

Filtering training examples to 200 ms after the unchanged source split retained 5,157 examples.
At the same 800-update budget, best dev AUC was 0.85212 (incumbent 0.87239), and the
200 ms-only dev diagnostic was also weaker. Final dev AUC was 0.84709; measured training
time 162 s. No external benchmark was accessed. Later pause examples appear useful as
training diversity under this recipe; timing alignment alone is not enough. This result does
not rule out a separately controlled mix or sampling policy.
W&B: https://wandb.ai/adrianbrunetto1-reshape/happyrobot-eot/runs/46004da7

## Runtime observation

E003 emitted nonfatal PyTorch DataLoader worker-shutdown diagnostics between completed epochs.
Training continued to its declared 800 updates; verify final dev row counts/checkpoints before
accepting its terminal result. This is recorded separately from model quality. No suppression of
errors or changes to the candidate/evaluation code were made during the run.

## E003 — rejected at the development gate

Full-model public-teacher regularization (30% teacher BCE, 70% hard-label loss) reached
best dev AUC 0.86603 at 800 steps, below 0.87239. All 1,818 dev examples were evaluated
and the final checkpoint was written; DataLoader shutdown diagnostics did not truncate
the run. No external benchmark was accessed. This specific teacher/weight/budget does not
solve overfitting. W&B: https://wandb.ai/adrianbrunetto1-reshape/happyrobot-eot/runs/bf9b543a

## Confirmation-data inventory

The full AppTek training manifest contains 82,488 cuts from 677 clip groups. Only 5 groups
(168 cuts) are absent from the incumbent’s mixed training manifest. This does not support a
large fresh conversational holdout simply by repartitioning the existing data; previous
models may also have seen those groups. Keep the independently uninspected Spanish benchmark
reserved for a later transfer check and disclose the absence of a fresh English confirmation set.

## E004 — inconsequential branch at the original learning rate

Dev AUC 0.872429 versus incumbent 0.872389: no meaningful gain, rejected without external
evaluation. A diagnostic on 16 training examples found the new branch contributes just 0.23%
of pooled feature norm and changes probability by at most 0.00095. E006 therefore tests a
100x learning-rate multiplier for the newly initialized branch while retaining the encoder/head
rate and all other E004 settings. This is an evidence-based follow-up, not a repeated blind trial.
W&B E004: https://wandb.ai/adrianbrunetto1-reshape/happyrobot-eot/runs/153b1875

## E005 — rejected at the development gate

Equal source/label sampling achieved best dev AUC 0.87302, only +0.00063 over incumbent,
below the preregistered +0.005 screen. Final AUC 0.87169. No external benchmark was
accessed. W&B: https://wandb.ai/adrianbrunetto1-reshape/happyrobot-eot/runs/c13d5a6d

## Prospective E007 screening refinement

The ranking-loss hypothesis explicitly targets low false-positive rates, so its preregistered
dev screen additionally allows >=5 percentage-point gain in 200ms-dev TPR at empirical 5%
FPR, with global AUC no worse than incumbent minus 0.005. This does not change any official
evaluation rule or retroactively rescue E001–E005. Benchmark promotion gates stay unchanged.

## E006/E007/E008 — completed, incumbent unchanged

E006 (100x acoustic-branch LR) reached dev AUC 0.87213; the higher branch rate did not
produce a meaningful improvement. E007 (hard-negative rank loss) reached AUC 0.86996 and
reduced 200ms-dev TPR at 5% empirical FPR from 41.03% to 32.31%, failing both screens.
E008 (fixed 50/50 incumbent/source-balanced weight soup) reached AUC 0.87547, a modest
+0.00308 improvement that falls short of the +0.005 gate. None accessed the external benchmark.
The weight soup is retained as a development lead, not promoted or called SOTA.

## E009 — data-filter hypothesis

The global high-confidence filter conflated VAD boundary confidence with the original clip’s
semantic label. It excluded 5,997 medium-confidence final/whole examples from existing training
sources (3,336 EOT, 2,661 HOLD). E009 restores those examples while retaining every original
training sample and all original dev IDs. It does not add medium-confidence internal cuts.
This differs from the old failed whole-only control, which removed pause supervision. The
new manifest, overlap guards and an additional exact-audio leakage audit are recorded separately.

## Additional reference diagnostic

The unmodified public Smart Turn checkpoint scores dev AUC 0.82667 and 200ms-dev TPR
22.56% at empirical 5% FPR, versus incumbent 0.87239 and 41.03%. Fine-tuning has therefore
helped the frozen prefix development task; the recent trials have plateaued around that gain.
This diagnostic uses no benchmark examples.

The leakage checker initially used a 1e-9 tolerance for negative time-to-onset, thereby
checking a slightly larger training superset than the actual trainer. Its earlier no-collision
result remains conservative. The checker now uses the trainer’s exact >=0 filter so counts
and source diagnostics match; both original and expanded audits are regenerated.

## E009 — rejected; first search batch complete

Restoring medium-confidence semantic-label samples reached best dev AUC 0.86893, below
the incumbent. The matched 800-update budget may not be enough for the expanded dataset,
but this experiment does not justify promotion. The additional exact-audio/source/clip
leakage audit passed (16,173 training rows, 1,818 unchanged dev rows). No external benchmark
was accessed. W&B: https://wandb.ai/adrianbrunetto1-reshape/happyrobot-eot/runs/74e0c694

Nine isolated trials have completed, all below their prospective screening gates. The frozen
incumbent and submission bundle remain unchanged. There is no new SOTA result. The best
new development lead is E008’s weight soup (+0.00308 AUC), retained with its rejection reason.

Next phase should investigate a meaningfully different representation/backbone with CPU
feasibility measured before long training, or independently sourced conversational supervision.
Do not spend the next batch repeating these optimizer/loss/branch variants at nearly identical
settings. Consider a prospectively registered low-FPR development screen for new architecture
trials, because global AUC is only a proxy for the official objective. Never change official metrics.

P010: Base 2s/4s random-weight export and CPU HTTP feasibility passed. 4s p95 c1/4/8=27.0/52.8/85.7ms, no errors; no quality claim. E010 preregistered public-Tiny context ablation; E011 preregistered larger ASR-pretrained representation exploration with explicit recipe confounding.

User steered search to data/labeler quality. E010 finished rejected (dev AUC0.85205); E011 deferred before training. otoSpeech104h has two channels and17 event classes but metadata/SRT access returned GatedRepoError. Pinned documentation/license revision. D012 begins with existing-labeler ambiguity audit; no frozen data changes or new benchmark use.

D012 complete as a code/audit checkpoint: 1,321 ambiguous training pauses, 258 affected incumbent-mixture rows;100 blank independent-review cases.21 tests pass. V2 abstains; no accuracy gain claimed, no data/model promotion. DATA_STRATEGY.md records the otoSpeech pilot and scale gates.

E013/E014 preregistered: ambiguity filtering vs matched random HOLD removal,258examples each, same source counts.9918train/1818unchangeddev, public initialization and800-step recipe fixed. Subset leakage proofs reuse frozen parent audit. No label-accuracy claim from this training comparison.

E013 passed official quality and serving (delay5%979.5ms, p95c8=84.96ms), but seed42 replication failed: filtered AUC0.874875, matched control0.876204. Preregistered replication guard prevents promotion; keep documented alternative. Bootstrap on benchmark turns does not establish robustness to training seed.

otoSpeech access verified with existing configured credential after user accepted gate. Frozen305/29/30conversation train/dev/confirmation splits,56cross-speaker-partition conversations quarantined;206opaque actor IDs remain local. D016 strict train parse98852events, twelve-conversation2.94569h pilot2950samples. D019 fixes48to16aliasing and truncates raw prefixes before symmetric filtering;2956samples validated against source.200-case blinded review packs prepared,0human judgments; no otoSpeech model trained.

E020/E021 fixed two-seed parameter averages test variance without inference ensembles. Filtered AUC0.880112 vs control0.877517 passes preregistered0.002margin; official filtered delay5%992.5ms fails minimum improvement and pairedCI gates. Rejected; incumbent unchanged.


## D022 — independent audio teacher, registered and prepared

User requested a strong teacher that helps us learn from failures. Pinned Qwen3-Omni Thinking Q4_K_M (20.76GB with bf16 audio projector), local runtime only. Frozen 24 train-only cases across ten conversations:8 random,8 strong student/silver disagreements,8 uncertainty. All8 strong disagreements are silver EOT/student HOLD; this is a targeted failure hypothesis, not proof the student is wrong. Labels/scores/IDs hidden from teacher; prefix and hindsight views separate. Seven correctness tests passed. A causal parity assertion caught a one-sample floating-point crop discrepancy before pilot freezing; fixed with integer sample indices. Runtime build and audio wiring smoke precede any label generation. No incumbent change.


## D022 interruption and P023 CPU feasibility

CUDA build completed, but the resumed session denies GPU and sockets. D022 is runtime-unavailable, with all cases/weights/code preserved. Registered P023 separately: CPU console, two audio wiring controls,240seconds each maximum, constrained boolean output. No SOTA or label-precision claim; W&B synchronization pending under the restricted network.


## P023 passed; D024 CPU pilot registered

Speech and silence controls both passed (20.16s/20.47s including model load). Audio-encoder calls are visible in local stderr. D024 reuses the24 frozen cases with CPU CLI, JSON schema, fresh process per call,120s deadline and45min total allowance. The original D022 GPU HTTP protocol remains unexecuted. All teacher outputs remain proposals, and W&B sync is pending network availability.


### D024 early failure pattern, post-hoc diagnostic

Case002 is EOT in both teacher views despite own-speaker acoustic resumption20ms after the decision. The same annotated Normal Turn spans the cut and continues through the last word; another Normal Turn follows, with an acknowledgement backchannel from the listener. This suggests semantic sentence closure can dominate exact continuation evidence in the teacher. It is a prioritized failure, not a human-adjudicated label. Added a transparent post-hoc audit for teacher EOT with own continuation<=300ms; it withholds automatic teacher overrides and queues review, never changes labels. Do not score this selected diagnostic as an untouched validation result.


### D024 random control completed (8 cases; provisional silver comparison)

Current silver labels contain1EOT/7HOLD. Teacher prefix proposes6EOT/2HOLD, disagreeing on5/8; hindsight proposes7EOT/1HOLD, disagreeing on6/8. No teacher abstentions. These small, clustered, unreviewed counts are not gold accuracy, but they argue strongly against scaling direct teacher relabeling. Continue the preregistered targeted strata to characterize the failure pattern; do not cherry-pick teacher agreements from the EOT-heavy disagreement stratum.


### LiveKit teacher question

Rechecked current official model license and Cloud terms. Distinguish full audio v1, v1-mini and the older text detector. See livekit_teacher_review.json for exact sections and source URLs. No LiveKit inference outputs were used for labeling/training; published benchmark results remain the comparison reference.


## Revisión estratégica solicitada por el usuario

D024 interrumpido mediante la sesión de ejecución (salida130),34/48 llamadas guardadas,17casos completos; la última llamada incompleta no cuenta como resultado. El usuario pide reconsiderar la dirección por falta de avances. No se reanuda automáticamente. El incumbente, pesos y ONNX conservan sus hashes y los820archivos congelados pasan integridad. STRATEGY_REVIEW.md distingue el objetivo del ejercicio del objetivo SOTA, documenta la falta de progreso reproducible y recomienda una auditoría explicativa antes de nuevos datos, profesores o arquitecturas. W&B queda pendiente de red.
