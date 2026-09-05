# Ranked hypotheses and decisions

> Estado actual: búsqueda experimental detenida a petición del usuario. D024 quedó interrumpido con 34/48 llamadas guardadas. Leer [STRATEGY_REVIEW.md](STRATEGY_REVIEW.md) antes de cualquier nueva prueba; las secciones de ejecución anteriores conservan el historial, no una orden de reanudación.

The accepted incumbent remains H-1d657c44f3. User priority: supervision quality and learning
from failures. E011 architecture training is deferred. Read DATA_STRATEGY.md and the ledger.

1. **D024 active: independent audio teacher on CPU.** D022 GPU/socket access is unavailable; P023 audio wiring passed. Blind training-only prefix/hindsight judgments,
   random controls and targeted disagreements. Establish useful evidence and abstention before
   trusting generated labels. Preserve independent review; never call teacher agreement gold.
2. **Test explicit temporal grounding for teacher judgments.** D024 already has teacher EOT despite immediate own continuation in the hindsight view. Do not blindly relabel with a general audio model. Compare a temporally explicit prompt or coherent-context input on fresh training audit cases; preserve the current pilot as discovery evidence.
3. **Adjudicate D019 labels and revise the event-to-pause mapper.** All 2,956 anti-aliased
   causal waveforms passed numerical checks; semantic precision remains unknown. The 200-case
   blind pack exists. Translate confirmed teacher/human failure patterns into narrow rule tests.
4. **Matched data ablation after review.** Compare old data, revised labels and new otoSpeech
   examples with fixed seeds/splits/update counts. Separate label corrections from added quantity.
5. **Pause/future-activity supervision.** Use observed two-channel outcomes with explicit
   uncertainty and censoring. Do not repeat the failed independent-FVAD-head experiment.
6. **Representation change, deferred.** CPU feasibility P010 passed, but data evidence takes
   precedence. Any resumed E011 must read its deferred state and preregister a fresh compute budget.

E013 improved one seed but E017/E018 replication failed; retain it unpromoted. E020 equal
weight averaging improved dev versus matched controls but failed the official improvement and
uncertainty gates. Do not retry exclusion/averaging without new evidence or tune mixture weights
against the benchmark. E001–E010 also failed their development gates; exact entries preserve
hypotheses and outcomes. Teacher E003 was a different experiment: public Smart Turn soft-label
regularization, not independent audio-language label adjudication.

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

- JAL-Turn (2026): frozen shared ASR features plus an acoustic representation branch and cross-attention; a candidate mechanism, not a benchmark result for this project. https://arxiv.org/abs/2603.26515
- Russell et al. (2026): controlled prosody/lexical perturbations suggest both cues can support turn-taking. This motivates branch-specific ablations rather than assuming semantics alone suffice. https://arxiv.org/abs/2601.13835

- E001: longer unregularized full-model training (3,200 updates) overfit; best dev AUC 0.87044, final 0.84079. No external benchmark run.

## Teacher licensing boundary

Do not use LiveKit model weights or outputs to train an independent standalone student;
its published license restricts that use. No restricted teacher model was downloaded.
https://github.com/livekit/agents/blob/main/MODEL_LICENSE

Additional research used for hypotheses:
- Targeting a low-FPR ROC region can differ from optimizing global AUC:
  https://proceedings.mlr.press/v162/zhu22g.html
  E007's simple ranking surrogate is not an implementation of that paper's optimizer.
- Averaging nearby fine-tuned models can improve generalization without inference ensembles:
  https://proceedings.mlr.press/v162/wortsman22a.html

- E009: rejected at 0.86893 dev AUC. Expanded data at the same 800-step budget did not help.
- Backbone exploration remains deferred pending the data-quality pilot.

Additional screened implementations: UltraVAD uses a Llama-8B backbone (model card), making direct CPU serving implausible without a separate measured compression program: https://huggingface.co/fixie-ai/ultraVAD . Vogent uses Whisper Tiny plus a 12-layer SmolLM and optional current transcript; its gated weights were not downloaded, and internal accuracy is not a comparable SOTA result: https://huggingface.co/vogent/Vogent-Turn-80M . These motivate semantic fusion hypotheses, not unsupported performance claims.
