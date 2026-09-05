# EoT autonomous research

The submission incumbent remains `exports/opt_clean/eot.onnx` until all promotion gates pass.
The main working tree includes pre-existing user work. A separate Git snapshot preserves that
state without altering its branch/index; experiment worktrees derive from that snapshot.

## Frozen objective

Primary: official English EoT Bench mean endpointing delay at <=5% false cutoffs, fixed score
at 200 ms, unchanged harness defaults. Baseline incumbent 1.0315 s; strongest published
same-protocol trace LiveKit v1 0.543 s. This reference is reproducible from published scores;
we have not independently rerun the cloud model. English has already been inspected and is
an adaptive development benchmark, not a blind confirmation set. Krisp and held-out AppTek
were also inspected. Spanish remains uninspected and reserved for a later transfer check;
it cannot establish an independent English SOTA claim.

Promotion: >=50 ms and >=5% primary improvement over incumbent; paired turn-bootstrap upper
95% endpointing difference <0 (conditional on selected policies, not a correction for adaptive
search); FC@300/600 regression <=2 percentage points; delay@10% regression <=50 ms. Secondary
wins without these gates remain Pareto alternatives. Report delay@2/5/10 and FC@300/600.
SOTA claims additionally require beating the frozen strongest baseline and explicit uncertainty,
scoring mode and adaptive-selection disclosure. No claim of universal or production superiority.

Serving: Intel i9-10900X, CPU-only Docker, 4 intra-op threads, max 8 in flight, 400 requests each
at concurrency 1/4/8. Full HTTP p95 <100 ms and zero errors at every level. Real 8-second PCM16
clip frozen in `protocol.json`; no response caching. Warm serving and startup recorded separately.
Promotion also requires tests, preprocessing/ONNX parity (max probability drift <1e-4, report
flips at operating thresholds), frozen evaluation integrity, and training/evaluation leakage checks.
No evaluation harness, labels, splits, scoring grids, or harness preprocessing edits permitted.

## State and resumption

- `protocol.json`: frozen hashes, workload and references.
- `incumbent.json`: authoritative selected artifact and source snapshot.
- `ledger.jsonl`: append-only experiment events; incomplete lines are never a completed result.
- `experiments/<id>.json`: atomic latest status/specification for each experiment.
- `BITACORA.md`: narrative discoveries and decisions.
- `BACKLOG.md`: ranked hypotheses, prior failures, next tests.
- `scripts/research.py`: integrity, registration, recording and W&B synchronization helpers.

Before resuming: run `.venv/bin/python scripts/research.py verify`, read the ledger/bitácora,
inspect running processes and W&B status, and use the exact experiment resume command.
Never restart a completed experiment under the same ID. Worktree source changes are committed
on experiment branches; training output stays in dedicated ignored run directories. Interruptions
retain atomic training checkpoints and mark partial results explicitly. The incumbent remains usable.

W&B project: existing `happyrobot-eot`, group `autoresearch-20260905`. Only scalar metrics,
small tables, plots, config and provenance; never checkpoint/model artifacts, audio or datasets.
Local JSON is authoritative; W&B failures are recorded as pending synchronization.

The clean baseline submission bundle is at `artifacts/research/incumbent-20260905/` (source tar,
ONNX, metadata, load test, parity and checksums). It remains independent of running experiments.
The source snapshot branch is `research/baseline-20260905`; the main branch/index were not changed.

On unexpected process termination, `python scripts/research.py reconcile` marks abandoned
PID-tracked runs interrupted; it never starts duplicate work. Then run
`.venv/bin/python scripts/research_train.py E00N --resume` for the selected interrupted trial.
A quality-qualified candidate still needs deployment evidence and
`research_promote.py`; that command rejects incomplete or stale evidence and atomically updates
the incumbent manifest, leaving old releases intact.
