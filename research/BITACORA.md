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
