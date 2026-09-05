# Data-first research checkpoint

> Estado actual: búsqueda experimental detenida a petición del usuario. D024 quedó interrumpido con 34/48 llamadas guardadas. Leer [STRATEGY_REVIEW.md](STRATEGY_REVIEW.md) antes de cualquier nueva prueba; las secciones de ejecución anteriores conservan el historial, no una orden de reanudación.

The user redirected the search to data quality. E011 larger-model training is deferred;
E010 completed and failed its development gate. The incumbent and frozen manifests remain unchanged.

## Evidence motivating the change

- Existing boundary QA recovered 67.25% of annotated gaps within 100 ms and 79.94% within
  200 ms. Annotation padding means these are agreement measurements, not exact acoustic truth.
- The original 175-case listening audit has no completed human judgments.
- E009 added 5,997 existing source-label examples and did not improve development AUC.
- D012 found 1,321 training pause decisions called HOLD solely because the listener spoke
  briefly. Short responses are not necessarily backchannels. These affect 258 of the 6,033
  valid AppTek rows in the incumbent mixture. A duration/word-count rule cannot resolve intent.

## D012 implemented checkpoint

Isolated branch `research/d012`, worktree `../happyrobot-eot-experiments/d012`.
`eot.apptek --rule-version v2` abstains on those short-listener cases, records label uncertainty
separately from acoustic confidence, prevents crops extending past resumed customer speech,
and refuses to overwrite an existing labeling run. The default v1 remains available for controls.
No existing training manifest was regenerated and no model was trained on the proposed change.

The metadata audit contains 100 deterministically sampled review cases, with blank judgments.
Its 138 multi-event-chain flags use manual segment timing only to prioritize review; they are
not 138 proven labeling errors. V2 changes coverage; improved label accuracy remains unmeasured.

Reproduce the audit from the repository root:

```bash
PYTHONPATH=../happyrobot-eot-experiments/d012/src .venv/bin/python ../happyrobot-eot-experiments/d012/scripts/audit_label_ambiguity.py \
  --root data/raw/apptek/hf --decisions data/mined/apptek-oracle/conversations.jsonl \
  --mixed data/optimization/mixed.jsonl --out ../happyrobot-eot-experiments/d012/eval/label_audit
```

## otoSpeech pilot: access verified and training-only data prepared

Reference: [otoSpeech full-duplex turn 104h](https://huggingface.co/datasets/otoearth/otoSpeech-full-duplex-turn-104h),
pinned revision `46f520297f434edf804389f82f9075a59d2f8268`.
Its card describes 420 English conversations, aligned separate channels and 17 conversational
event categories. This makes event-aware supervision a plausible improvement over timing alone.
Access was verified after explicitly loading the existing Hugging Face credential. Metadata and SRT files are pinned; only twelve training conversations were downloaded as separate audio channels.
The [license](https://huggingface.co/datasets/otoearth/otoSpeech-full-duplex-turn-104h/blob/46f520297f434edf804389f82f9075a59d2f8268/LICENSE.md)
restricts corporate research and commercial deployment. The pilot is isolated research data; acceptance of the repository gate does not establish commercial production rights. No submission model uses these data.

1. Pin annotations, metadata and audio hashes. Parse all metadata before selecting a pilot;
   create splits by conversations and, where feasible, connected components of provided actor
   IDs. Do not identify speakers from voice or link them to external datasets. Check component
   sizes before promising a speaker-disjoint split. Reserve confirmation data before inspection.
2. Select 12 eligible training conversations across categories using seed17 and no model scores.
   Download only the separate channels. Avoid duplicating the stereo mix.
3. Locate acoustic pauses on the current-speaker channel. Use annotations and both channels
   only to derive training targets and uncertainty. Model inputs end at the scored cut and
   contain neither future audio nor annotation text/event labels.
4. Distinguish own floor holds, listener backchannels, cooperative/competitive interruptions,
   real floor transfer, channel bleed, overlap and right-censored tails. A `Normal Turn` segment
   ending is not automatically EOT. Ambiguous events abstain; they are not forced negatives.
5. Review a blind stratified sample independently of the rule outputs: at least 100 proposed
   HOLD and 100 EOT cases, plus ambiguous/overlap cases. Report cut validity, label precision,
   Wilson intervals, inter-reviewer disagreement and retained coverage by event/category.
   Prospective scale gate: >=95% valid cuts and per-class label-precision lower 95% bound >=90%.
   Automated agreement alone cannot satisfy the independent-review gate.
6. Compare baseline data, ambiguity-filtered existing data, and new event-supervised data under
   matched initialization/optimizer/updates and frozen old development IDs. Control source balance
   separately. Perform exact-audio leakage checks before training; disclose their limitations.
7. Only meaningful dev-qualified models reach the immutable official LiveKit protocol and
   existing deployment gates. Keep endpointing delay and HTTP performance separate.

No claim of SOTA improvement or production readiness follows from this data audit.

## Completed data experiments and current decision

E013 removed 258 ambiguous training rows and reached 979.5 ms delay at 5% cutoff,
compared with the incumbent's 1031.5 ms. CPU HTTP p95 at concurrency 8 was 84.96 ms.
However, its seed42 replication E017 (dev AUC 0.87487) lost to the matched random-removal
control E018 (0.87620). E013 remains an unpromoted alternative; its replication gate failed.
E020 averaged the two filtered seeds, passing the matched dev comparison, but its official
992.5 ms result failed the predeclared improvement/uncertainty gates. No incumbent changed.

D016 froze actor-disjoint groups: 305 train, 29 dev, 30 confirmation conversations;
56 cross-partition conversations were quarantined. Only train annotation content was inspected.
D019 repeats the same twelve-conversation pilot with anti-aliased resampling, cropping raw
signals before filtering to prevent future-audio leakage. All 2,956 causal waveforms passed
validation (2,627 HOLD, 329 EOT). The local blind review pack has 200 cases and zero human
judgments: `data/research/D019/review/index.html`. Numerical signal correctness does not prove
semantic label correctness. No otoSpeech student has been trained.

## Active D022: independent audio teacher and error-driven labeling

Use a pinned Qwen3-Omni-30B-A3B-Thinking Q4_K_M model locally as a proposed-label reviewer.
The published general audio results motivate testing; they do not establish EoT annotation
quality, especially after quantization. Source: https://arxiv.org/abs/2509.17765 .
Runtime conversion: https://huggingface.co/ggml-org/Qwen3-Omni-30B-A3B-Thinking-GGUF .

Freeze 24 training-only pauses: 8 random controls, 8 strong student/silver disagreements,
and 8 uncertain student predictions. These are candidate failures, not proven mistakes.
Hide silver labels, IDs and student scores from the teacher. Obtain separate prefix-only
judgments and retrospective judgments using both speakers and six seconds of later context.
Future context is target-construction evidence only; student inference remains causal.
Store structured label, cut validity, abstention and short audible evidence locally. Do not
request numerical confidence as a substitute for calibration. Invalid responses stay failed.

Compare views and silver labels by sampling stratum; build a review queue for backchannels,
incomplete utterances, interruption, bad cuts and insufficient context. Teacher disagreements
must be adjudicated before changing rules or creating high-trust training labels. Model
outputs never count as human review or satisfy the existing independent-review scale gate.
Use newly frozen train-only audit cases when revising prompts from discovered failures;
do not optimize prompts repeatedly on the same final audit set. Preserve actor confirmation
annotations and the official LiveKit harness untouched. Log aggregate metrics/plots in W&B;
keep weights, raw audio, transcripts and per-case responses local.

Worktree: `../happyrobot-eot-experiments/d022`. Exact run instructions and runtime hashes
will be recorded in `research/experiments/D022.json` and `data/research/D022/runtime.json`.


D022 GPU execution became unavailable when the session permissions changed. Its CUDA build completed and all assets/cases are preserved. P023 CPU speech/silence controls passed. D024 now performs the same training-only24-case study through a CPU console with explicitly constrained JSON decoding; source/runtime/configuration hashes are separate. The latest local review will be `data/research/D024/report/review.html`. W&B synchronization is pending restricted network access; no new model is promoted.
