# Ranked hypotheses

1. **E001: undertraining** — same public initialization, data, augmentation, LR and seed; extend
   800 to 3,200 optimizer updates. Original best dev AUC was still improving at the last update.
   This changes the cosine schedule duration too; interpret as a training-budget/schedule test.
2. **E002: boundary-aligned supervision** — train on exact 200 ms pause prefixes and balance
   HOLD/EOT within each source to reduce source and pause-duration shortcuts. First inspect
   existing cut distributions and dev per-source errors. Preserve the frozen source split.
3. **E003: preserve pretrained semantics** — layerwise LR or public-teacher regularization on
   the full model, unlike the failed two-layer distillation. Trigger only if fine-tuning degrades
   clean public-source dev while helping conversational dev.
4. **E004: acoustic branch** — lightweight causal prosody encoder fused with pretrained Whisper
   features, inspired by LiveKit's published architecture. Requires a matched branch-off ablation,
   additional input/export parity, and CPU budget before benchmark access.
5. **E005: richer observed-future supervision** — joint future voice activity, not the failed
   independent censored FVAD heads; requires enough trustworthy observed dual-channel data.

## Previously tried; do not repeat without new evidence

- Two-layer first-layer pruning: fast but FC@300 41.7%, worse than full model.
- Uniform two-layer pruning plus soft-label distillation: FC@300 42.3%, rejected.
- Corrected independent FVAD auxiliary loss: worse than matched no-FVAD control.
- INT8 MatMul variants: exceeded probability parity gate; rejected exports removed.
- Generic Whisper initialization on mined data: weaker than public Smart Turn initialization.
- Whole-clip control: substantially worse early cutoff than pause-prefix training.
- Context head on unpaired metadata: no supported primary context claim.
- CUDA default TF32 convolution traces: ~8.7e-4 deployment drift; final evaluation disables TF32.

## Research references (hypotheses, not proof of local gains)

- LiveKit v1 architecture: semantic encoder/LLM plus recurrent acoustic branch.
  https://livekit.com/blog/solving-end-of-turn-detection
- Ekstedt & Skantze (2022), Voice Activity Projection: joint future-activity dependency modeling.
  https://arxiv.org/abs/2205.09812
- Official immutable comparison protocol and published reference artifacts:
  https://github.com/livekit/eot-bench
