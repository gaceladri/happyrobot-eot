# Reproduce the data and training decisions

The selected CPU model is a Whisper-base encoder with an end-of-turn classification head, distilled from three fine-tuned Cohere teachers. The GPU service uses the best single Cohere checkpoint at the 5% cutoff operating point. Every baseline also has a trained head: “frozen encoder” means that the head learns while the pretrained encoder stays fixed.

The reusable implementation is in `src/eot`. The exact expanded-data campaign runner (`research/labeling/experiments/L041/`) and the complete research program are retained in a separate research-code archive, together with every experiment's protocol, report and results. Request that archive for exact campaign reproduction; the public repository intentionally contains only the delivery snapshot. That runner preserves the original audio-view RNG, epoch cache identity, update counts and selection rules; the general `eot-train` CLI reproduces the earlier controlled recipe but is not an interchangeable reproduction of the final campaign.

## Evidence and frozen configuration

- [Final recipe protocol](../configs/training/final-protocol.json): byte-for-byte snapshot from Git commit `d44278e`, SHA256 `946652a6a190aa9a5fa31bd615642693c9b5c774cf817997ead0dcc39050f0c2`. This is the protocol used by the recorded training. Later deployment follow-ups in the active registry do not change it.
- [Final experiment report](../evidence/REPORT.md): selected checkpoints, rejected variants, development metrics, hardware and limitations; its evidence inventory (separate research archive: `output/L041/evidence_inventory.json`) binds every receipt by hash.
- Research program (separate research archive: `research/labeling/PROGRAM.md`) (separate research-code archive): chronological decisions and links to experiments. `L001`–`L041` are experiment identifiers, not model names.
- [Final deployment selection](../configs/inference/selected.json): the exact exports used in the presentation's CPU/GPU comparisons.

The historical report predates the later local TensorRT deployment. Read it together with [the deployment follow-up](../evidence/deployment/DEPLOYMENT_REPORT.md); its earlier “not measured” statements describe the original closeout.

## Labeling: what we changed and why

No new manual annotation campaign took place. Publicly available annotations served as reference evidence for evaluating and calibrating heuristics. We then used those rules to mine additional training prefixes. Existing annotations are not equivalent to a new independent human review of every generated training row.

The code paths are:

1. `eot.data.smart_turn` acquires the public Smart Turn material. `eot.labeling.prefix_mining` proposes causal prefixes around pauses and applies confidence filters.
2. `eot.labeling.apptek` and `eot.labeling.turn_events` derive targets from dual-channel conversational events. The revised rules abstain on ambiguous duration-only backchannel cases instead of assigning intent from pause length alone.
3. `eot.labeling.qa` and the recorded labeling studies assess coverage, disagreement and exclusions against existing annotated references.
4. `eot.data.splits` groups related audio before splitting. `L041/data_expansion/frozen.py` verifies the expanded training set against the unchanged development split, including IDs, source groups, hashes and excluded corpora.
5. `L041/teacher/dense.py` and `materialize.py` derive additional causal pause prefixes from training groups only. These points carry soft teacher targets, not invented hard labels.

Example entry points for a new, separately recorded data preparation run:

```bash
uv run eot-acquire --help
uv run eot-mine --manifest data/raw/clips.jsonl --out data/mined/new-run \
  --detector ensemble --min-confidence medium --workers 8
uv run eot-apptek --root data/raw/apptek/hf --out data/mined/apptek-new \
  --rule-version v2
uv run eot-qa --help
```

These commands alone do not recreate the final frozen mixture. To reproduce that experiment, restore the recorded manifests, audio and derivation receipts. The exact expansion logic is in `L041/data_expansion/prepare.py`; it was a protocol-registration step, not a command to rerun against the current registry. Preserve the recorded protocol and generate new derivations in separate directories.

The final training mixture contains 144,197 rows: 93,549 prior training rows plus 24,571 additional AppTek cuts and 26,077 cuts from 8,000 additional English Smart Turn clips. The 16,390-row development set stays fixed. AMI, ICSI, Krisp, otoSpeech and TurnBench do not enter this training set. otoSpeech was an exploratory pilot.

The audit found no exact train/dev or expanded-train/benchmark overlap. It did not prove complete absence of leakage: two brief perceptual similarities remain unresolved, 1,463 short additions could not be checked by the fingerprint method, and pretraining overlap is not fully auditable.

## Controlled progression before the final combination

Read the original reports for matched controls, seed counts and failures:

- Cohere encoder adaptation (separate research archive: `research/labeling/experiments/L031/REPORT.md`): frozen encoder plus trained head, followed by LoRA adaptation of the encoder. The mean waiting improved from about 789 to 636 ms across three runs.
- Pause augmentation (separate research archive: `research/labeling/experiments/L037/REPORT.md`): replace only the pause tail with digital silence or low-level noise, independently of the label. Speech is preserved.
- Distillation × augmentation (separate research archive: `research/labeling/experiments/L040/REPORT.md`): Whisper-base hard-label control 896.8 ms; distillation 850.0 ms; distillation with pause augmentation 801.8 ms, each a three-run mean. This is the controlled evidence for the combination, rather than an attribution of every later improvement.

The package backbone trainer now writes `eot-backbone-resume-v2` checkpoints. It rejects v1 resumes because the permutation seed and loader RNG behavior changed. Resume a historical run with its exact original code and environment; changing the format field cannot make it compatible. Full model and delta checkpoint formats are unchanged.

The general package exposes `eot-train-backbone`, `eot-distill-cache`, and `eot-train --teacher-cache`. Each teacher logit is tied to a hash of the exact augmented waveform consumed by the student. Temperature is applied in the Bernoulli KL loss. A distilled student does not require a teacher at inference time and has the same architecture as its corresponding hard-label control.

## Exact final teacher continuation

The commands in this section and the next require the separate research-code archive and run its campaign runner, not the package CLIs. The model/data recovery archive described below is a different artifact.
Use a CUDA-capable training environment and the separately supplied pretrained snapshots, three original Cohere LoRA parent deltas, and frozen expanded manifests. The training used a rented H200; RTX 3090 serving measurements are a different stage. Three resident teachers and cached-feature batch 1024 are not promised to fit on a 24 GB GPU.

```bash
# In a clone of the separately supplied research-code archive:
# git clone /path/to/full-history.bundle research-history
# cd research-history && git checkout archive/research
uv sync --locked --extra dev --extra data --extra backbone
export PYTHONPATH="$PWD/src:$PWD:$PWD/research/labeling/experiments/L041"

# Example for one teacher. Repeat with each seed's registered parent and SHA.
uv run python -m teacher.continue \
  --samples data/research/L041/data_v2/inputs/train.jsonl \
  --dev data/research/L041/data_v2/inputs/dev.jsonl \
  --protocol configs/training/final-protocol.json \
  --parent data/research/L031/runs/C2_lora/seed222/delta.pt \
  --parent-sha256 5873671a34acd23353acf685f37d4534dad7fe4c0a84b6b72178013ce5e9d7b0 \
  --out runs/reproduction/teachers/seed222 --seed 222 --require-cuda \
  --batch-size 128 --micro-batch 128 --max-steps 1127 \
  --workers 8 --eval-batch 128
```

The runner enforces LoRA rank 16, alpha 32, dropout 0.05, encoder/head learning rates 2e-5/1e-3, weight decay 0.01, 20% warmup, BF16 autocast with FP32 master weights, 30% telephony augmentation, and the original pause-tail augmentation. There is one additional pass per teacher. Loading the parent delta starts a new optimizer; it is a weight-only continuation, not an exact resume of the parent's optimizer.

Use `--smoke --max-steps 5 --max-dev-rows 32` with a separate output for a short feasibility check. Such an output is marked non-scientific and cannot be consumed as a completed teacher. `--resume` requires the run's own checkpoint, code/runtime and identity to match. Do not bypass those checks to resume an old checkpoint after a refactor.

## Ensemble targets and Whisper distillation

The ensemble averages the three teachers' **raw logits**, in FP32, on the same realized causal audio view. It does not average LoRA factors or introduce future audio. Dense supervision uses eligible training pauses on a 100 ms grid, capped at 16,384 extra points per epoch.

For each restored frozen epoch manifest, build a cache with the three completed teacher paths:

```bash
uv run python -m teacher.ensemble_cache \
  --rows data/research/L041/data_v2/dense/epoch1.jsonl \
  --teachers runs/reproduction/teachers/seed111/delta.pt \
    runs/reproduction/teachers/seed222/delta.pt \
    runs/reproduction/teachers/seed333/delta.pt \
  --protocol configs/training/final-protocol.json --epoch 1 --seed 111 \
  --out data/reproduction/caches/epoch1 --batch-size 256 \
  --workers 8 --student-features
```

Repeat for epoch 2. The actual validated cache batch should be chosen by the runner's training-only pilot for the available hardware. Cache metadata records it. The final campaign cached 353,930 targets including the later 32,768-row continuation panel; the selected student itself used two 160,581-row epochs. These are repeated/augmented target rows, not 353,930 distinct manually labeled examples.

Build an immutable student plan from the restored input receipt, current cache receipts, and frozen dense metadata. Obtain the dense metadata SHA with `sha256sum` and pass it explicitly:

```bash
uv run python -m student.prepare plans --mode trajectory \
  --protocol configs/training/final-protocol.json \
  --inputs data/research/L041/data_v2/inputs \
  --dense-metadata data/research/L041/data_v3/dense_metadata.json \
  --dense-metadata-sha256 <SHA256-of-that-file> \
  --cache-root data/reproduction/caches --root "$PWD" \
  --learning-rate 8e-5 --out data/reproduction/trajectory.json

uv run python -m student.train \
  --plan data/reproduction/trajectory.json --plan-sha256 <SHA256-of-plan> \
  --out runs/reproduction/whisper --workers 8 --cpu-threads 4
```

Angle-bracket values are explicit placeholders for receipts produced by the preceding step. Relocated manifests retain source and logical hashes; do not edit old plan paths in place and pretend the original byte hash still applies.

The selected trajectory used batch 1024, 314 updates, temperature 2, pure soft-target KL, cosine learning-rate schedule, 10% warmup and weight decay 0.01. Three learning rates were tried: 2e-5, 8e-5 and 3.2e-4. The middle trajectory's final checkpoint won on the fixed development AUC, with weighted BCE as a tiebreaker. Its snapshot soup did not satisfy the selection rule. `infra/campaign_stage.py` on the archived branch retains the complete LR-selection and soup-selection logic.

The final bundle combines data expansion, continued teachers, ensemble supervision, denser targets and multiple passes. It does not isolate their separate effects. The best single Cohere at the public 5% cutoff point is used for GPU delivery; that public operating-point selection is distinct from fixed-dev student selection.

## Artifacts, archive and reproducibility limits

The original owner archive is named `happyrobot-archive-2026-09-07`. Its `README.md` documents recovery and `SHA256SUMS` identifies complete `.tar.zst` files. Archive members contain paths relative to the owner's Desktop. Extract only required archives into a separate recovery root, verify the checksums, then supply the restored data/checkpoint paths. Do not extract `.partial` files or overwrite active work.

For example, from the archive directory:

```bash
# Verify the specific archive against its entry in SHA256SUMS first.
mkdir -p /path/to/recovery
tar --zstd -tf selected-archive.tar.zst
# Extract only after inspecting the archive's member paths.
tar --zstd -xf selected-archive.tar.zst -C /path/to/recovery
```

The archive contains model/data state, not the current Python environments or the final TensorRT export. Pretrained snapshots may require acceptance of the original provider's license and local credentials; no credentials belong in this repository. Dataset/model licenses remain those of their respective sources.

A fresh clone can run unit tests and inspect all code/receipts. Full training requires the separate data/checkpoint bundle and suitable hardware. Code, configuration, seeds, split identities, cache pairing and selection rules are reproducible; bitwise equality across different GPU/runtime stacks is not promised. New runs and rebuilt exports receive new identities and measurements.
