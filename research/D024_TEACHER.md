# D024 CPU teacher checkpoint

> Estado actual: búsqueda experimental detenida a petición del usuario. D024 quedó interrumpido con 34/48 llamadas guardadas. Leer [STRATEGY_REVIEW.md](STRATEGY_REVIEW.md) antes de cualquier nueva prueba; las secciones de ejecución anteriores conservan el historial, no una orden de reanudación.

This is an offline annotation diagnostic, not a student training run or an official
benchmark result. The accepted incumbent remains H-1d657c44f3. D022's GPU/HTTP
configuration was unavailable in this session; P023 verified the CPU console with
speech and silence before D024 started.

Exact inputs and runtime:
- Frozen cases: `data/research/D022/pilot.json`, 24 training cases, 10 conversations.
- Runtime/model/command identity: `data/research/D024/runtime.json`.
- Source teacher instruction: D022 worktree commit
  `a4c7fc5feb0da3cd85e11fddf5ee60379a132a92`.
- CPU runner: `scripts/research_teacher_cpu.py`, six threads, fresh process per call,
  4,096 context tokens, 256 generated tokens, seed 17, greedy JSON-constrained decoding.
- Two views per case: causal target prefix; retrospective context with both channels
  and future audio. Future audio is exclusively for offline annotation research.
- 120 seconds per call, 45 minutes per invocation; three consecutive errors stop work.
  Existing completed/failed calls are retained on resume, never silently retried.

Run from `/home/ad/Desktop/happyrobot-eot`. Do not start a duplicate while the current
runner is active. Resumption validates the model, binary, instruction, code and input
identities before reusing results:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/research_teacher_cpu.py
MPLCONFIGDIR=/tmp/eot-matplotlib PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=../happyrobot-eot-experiments/d022/src .venv/bin/python \
  scripts/research_teacher_report.py --pilot data/research/D022/pilot.json \
  --results data/research/D024/teacher --out data/research/D024/report
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/audit_teacher_temporal_consistency.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/research.py verify
```

`teacher/status.json` records execution state. `report/summary.json` separates strata
and views. Teacher/silver disagreement is not accuracy: silver labels have not been
independently reviewed. Selected disagreement and uncertainty strata cannot estimate
population precision. The displayed Wilson intervals also ignore conversation clustering.

`report/review.html` provides local two-channel audio and requires a blind judgment
before revealing proposals. Its exported judgments are separate from D019's 200-case
review gate. No machine response counts as a human review.

The post-hoc temporal audit flags EOT proposals where the source evidence indicates
own-speaker continuation within 300 ms. That threshold was selected after observing
case 002, which resumes just 20 ms after the cut. This prioritizes inspection; it does
not establish gold labels, change the official metric, or automatically relabel data.

Provisional decision: do not scale direct teacher relabeling. The random stratum
already shows EOT-heavy proposals, and hindsight does not reliably resolve temporal
contradictions. The next hypothesis is explicit grounding in continuation and floor
transfer on fresh, frozen training audit cases, with independent adjudication.

W&B synchronization is pending because shell network access is unavailable. Stable
run IDs and aggregate payloads remain local. When connectivity returns, synchronize
the same IDs, not duplicate runs. Never upload weights, checkpoints, private audio,
transcripts, or raw per-case teacher responses.

## Specialized teacher research

LiveKit's strongest published audio model is v1, served in LiveKit Cloud, distinct
from v1-mini and the older Hugging Face text detector. Its published model license
section 3.b(ii) restricts using outputs to improve independent models. Cloud terms
have additional restrictions in section 3.2(c); API access alone does not establish
permission for this labeling program. No LiveKit outputs were generated for training.
Published benchmark artifacts remain the comparison reference.

- https://github.com/livekit/agents/blob/main/MODEL_LICENSE
- https://livekit.com/legal/terms-of-service
- https://livekit.com/blog/solving-end-of-turn-detection
- https://github.com/livekit/eot-bench

UltraVAD is another specialized candidate (published English delay at 5% cutoff:
899 ms). Its EoT model card has no explicit license, and the configured text backbone
`fixie-ai/turntaking-pretraining-it-multilingual-3c` has no model card. The MIT license
on the general Ultravox code/model is not assumed to grant rights to these distinct
EoT weights. License status is unresolved; no weights downloaded or inference run.
Its audio encoder is Whisper large-v3-turbo and the text backbone has 8B parameters;
the 0.7B tensor count on the audio package does not describe the complete model.

- https://huggingface.co/fixie-ai/ultraVAD
- https://huggingface.co/fixie-ai/ultraVAD/blob/main/config.json
- https://huggingface.co/fixie-ai/turntaking-pretraining-it-multilingual-3c
- https://github.com/fixie-ai/ultravox/blob/main/LICENSE

Sources inspected on 2026-09-05. Public benchmark performance does not establish
annotation accuracy on otoSpeech's human-to-human conversations.
