# Optimization follow-up — 5 September 2026

The exercise's <100 ms requirement is per API request. Endpointing delay is a separate quality
tradeoff. A successful CPU timing does not establish acceptable interruption behavior.

Original R2–R4 are retained. Their full English benchmark has already been inspected; subsequent
English results are exploratory. Krisp and held-out AppTek remain outside training. Human audit
labels cannot be manufactured: the pending blind audit stays an explicit release limitation.
AppTek supervision is a behavioral heuristic, not ground truth about user intent.

Follow-up candidates, fixed before evaluating them:
- public Smart Turn v3.2 weights imported with enforced parity, frozen revision;
- public weights fine-tuned on source-capped high-confidence prefix examples;
- same weights and per-source sample counts using whole-clip labels/audio (control);
- source-capped prefixes plus high-confidence AppTek, at the same optimizer-step budget;
- two encoder layers on that same mixed dataset, to test CPU cost versus quality.

All: seed 17, batch 32, maximum 800 optimizer updates, up to 8 epochs, AdamW, no FVAD,
30% telephony augmentation. Full models lr 1e-5; the pruned student lr 5e-5 (architecture
experiment, not an isolated depth ablation). Context remains disabled without a verified paired
training/evaluation protocol. No claim of Spanish quality without Spanish measurements.
Future-speech heads on old checkpoints are experimental; unknown clip-end futures are masked.

Selection: compare quality on source-grouped dev and exploratory EoT Bench, then confirm the
selected artifact on the independent Krisp set. Benchmark-selected policies get conditional
intervals only; separate calibration-turn policies are also reported. Preserve both scoring modes,
with 200 ms as the primary comparison against public Smart Turn. Cutoff is per HOLD span.

Serving: measure FP32 versus INT8 on real audio, batch 1/2/8 export parity, CPU thread counts,
and HTTP concurrency 1/4/8. Reject quantization if real-audio probability drift exceeds 0.02;
report threshold decision flips. Enforce deadlines after decode and inference, reject stale scores,
and bound admission to limit overloaded latency. Report warm startup separately from requests.

Production remains conditional on human label audit and real call rollout metrics, not merely
passing a synthetic unit test or timing threshold.

Follow-up after the first student result: direct two-layer pruning lost too much quality
(41.7% FC@300). Test one distilled student initialized from uniformly spaced public encoder
layers (0 and 3), 70% soft-teacher BCE from OPT mixed + 30% labels, 1,600 updates at 3e-5,
seed 17 on the same mixed training sources. This is an explicitly exploratory architecture
experiment; it will only replace the full model if its quality and CPU measurements justify it.

Final controlled auxiliary-loss ablation: initialize the same public primary weights with only
new FVAD-head parameters (seed 17); use the exact OPT mined samples, lr 1e-5, 800-step budget,
seed and source split. Only observed internal future activity is supervised; clip-end futures
are masked. Compare against OPT mined without FVAD. This closes the auxiliary-supervision
question with corrected labels rather than interpreting the confounded original runs.

Final data-integrity check found 130/12,124 legacy mixed examples with negative time-to-onset:
the energy pause ended slightly after the VAD onset. Mining now bounds cuts by the earlier
onset; training excludes these records and logs the count without changing old manifests.
Train a clean full-model candidate with the same mixed recipe and 800-update budget. Historical
OPT mixed performance remains a historical measurement, not relabeled as clean training.
