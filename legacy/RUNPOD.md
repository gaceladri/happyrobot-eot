# RunPod training runbook

This runbook packages the EoT take-home as two intentionally different images:

- `Dockerfile.train` is an interactive CUDA image for acquisition, mining,
  training, official evaluation, and export.
- `Dockerfile` is the graded production surface: a small CPU ONNX service.

The split prevents the serving image from carrying Torch/CUDA and prevents a
dependency sync from replacing the training image's tested CUDA stack.

## 1. Storage and directory contract

All generated state is rooted at `/workspace/eot`:

```text
/opt/eot/                              immutable package in the image
/workspace/eot/
  cache/{huggingface,torch}/
  data/raw/smart-turn-v3.2-eng/
  data/mined/prefix-v1-eng/
  runs/<run-id>/{model.pt,history.json,history.jsonl,provenance.json}
  runs/<run-id>/checkpoints/{last.pt,best.pt}
  exports/<run-id>/{eot.onnx,eot.json}
  eval/<run-id>/
```

For one short take-home run, use a **100 GB volume disk** mounted at
`/workspace`: it survives Pod stop/restart and provides local I/O for many small
WAV files. It is deleted when the Pod is terminated, so retrieve artifacts
first. Use a network volume instead only when the data must survive Pod deletion
or move between Pods; RunPod documents that network volumes are Secure Cloud
only, attach at creation, replace the volume disk at `/workspace`, and constrain
GPU availability to their datacenter. See [storage
types](https://docs.runpod.io/pods/storage/types) and [network
volumes](https://docs.runpod.io/storage/network-volumes).

## 2. Build and publish from this Apple Silicon Mac

Use a private registry if the take-home is not public. Replace the example name
and keep versioned tags; do not use `latest`.

```bash
docker buildx build --platform linux/amd64 \
  -f Dockerfile.train \
  -t ghcr.io/OWNER/happyrobot-eot-train:0.1.0 \
  --push .

docker buildx imagetools inspect ghcr.io/OWNER/happyrobot-eot-train:0.1.0

docker buildx build --platform linux/amd64 \
  -f Dockerfile \
  -t ghcr.io/OWNER/happyrobot-eot-serve:0.1.0 \
  --push .
```

For a private GHCR image, save a registry authentication in RunPod (use the
GitHub username and a read-only package token), then select those **Registry
credentials** on the Pod template. Do not paste the token into the image, start
command, or ordinary environment variables. RunPod documents both the template
field in [template management](https://docs.runpod.io/pods/templates/manage-templates)
and the equivalent [`runpodctl registry`](https://docs.runpod.io/runpodctl/reference/runpodctl-registry)
operation.

RunPod explicitly requires `linux/amd64` for images built on ARM Macs and
recommends immutable versioned tags in its [custom-template
guide](https://docs.runpod.io/pods/templates/create-custom-template).

`Dockerfile.train` pins the documented RunPod PyTorch image by its amd64 digest:
Torch 2.8.0, CUDA 12.8.1, Python 3.12. It installs the fully resolved
Python 3.12/linux-amd64 non-Torch dependency snapshot in
`requirements/runpod-train.lock`, then installs this package with `--no-deps`.
**Do not run `uv sync` inside
the Pod**: `uv.lock` was generated on another platform and currently resolves a
different Torch/CUDA tuple.

The initial `openai/whisper-tiny` weights are also pinned at revision
`169d4a4341b33bc18d8881c4b69c2e104e1cc0af`; that revision is persisted in the
training checkpoint together with the full Whisper configuration.

## 3. Create the Pod template

Recommended first-run settings:

| Field | Value |
|---|---|
| Cloud / rental | Secure or Community, On-Demand |
| GPU | One available 24 GB card (RTX 4090/3090, A5000, or L4) |
| Image | `ghcr.io/OWNER/happyrobot-eot-train:0.1.0` |
| Container disk | 30-40 GB |
| Volume disk | 100 GB mounted at `/workspace` |
| TCP ports | `22/tcp` for full SSH/rsync |
| HTTP ports | `8888/http` only if Jupyter is wanted |
| Container start command | **leave blank** |

Leaving the start command blank preserves the base image's `/start.sh` and its
Jupyter/SSH services. A template command would override the Docker `CMD`; see
[template management](https://docs.runpod.io/pods/templates/manage-templates).

Set these environment variables once when creating the Pod:

```text
EOT_ROOT=/workspace/eot
EOT_IMAGE_REF=ghcr.io/OWNER/happyrobot-eot-train:0.1.0
HF_HOME=/workspace/eot/cache/huggingface
HF_DATASETS_CACHE=/workspace/eot/cache/huggingface/datasets
TORCH_HOME=/workspace/eot/cache/torch
TOKENIZERS_PARALLELISM=false
```

The launcher pins Smart Turn v3.2 revision
`e564e2ac567f774d1880aa1db6ce97afb8c519b7` by default. Override
`EOT_DATASET_REVISION` only when deliberately starting a new, separately named
experiment. Acquisition requires a non-empty revision, records it in provenance,
and refuses to resume if it changes; the supplied default is the verified commit
pin. Official evaluation likewise pins EoT Bench data at
`ca9d98a9686b920a2d8c9eb984224ba9be74e4dd` through
`EOT_BENCH_REVISION`.

Smart Turn and EoT Bench are public today, so a Hugging Face credential is not
normally required. If one is needed, create a RunPod secret and reference it as:

```text
HF_TOKEN={{ RUNPOD_SECRET_huggingface_token }}
```

Never bake tokens into the image or print the full environment. RunPod explains
secret references and the restart implications of environment edits in its
[environment-variable documentation](https://docs.runpod.io/pods/templates/environment-variables).

## 4. Prove the path before the full download

Connect with SSH or Jupyter terminal. Use `tmux` so a dropped client connection
does not kill the process; checkpoints, rather than tmux, handle a Pod restart.

```bash
tmux new -s eot
set -o pipefail
mkdir -p /workspace/eot
eot-runpod smoke 2>&1 | tee /workspace/eot/smoke.log
```

The smoke command refuses CPU fallback, verifies disk and tools, acquires 32
examples per label, mines them, runs two optimization steps, writes a checkpoint,
and exports ONNX with a parity gate. Do not start the full acquisition until it
passes.

## 5. Run or resume the full pipeline

```bash
export EOT_RUN_ID=base
export EOT_PER_LABEL=8000
export EOT_BATCH_SIZE=32
export EOT_WORKERS=4
export EOT_EPOCHS=3

set -o pipefail
mkdir -p "/workspace/eot/runs/$EOT_RUN_ID"

# Recommended first full-size pass: stop for the human audio-quality gate.
eot-runpod acquire 2>&1 | tee "/workspace/eot/runs/$EOT_RUN_ID/acquire.log"
eot-runpod mine 2>&1 | tee "/workspace/eot/runs/$EOT_RUN_ID/mine.log"
eot-runpod audit 2>&1 | tee "/workspace/eot/runs/$EOT_RUN_ID/audit.log"
# Inspect mining_audit.json and listen to its representative WAVs, then continue:
eot-runpod all 2>&1 | tee "/workspace/eot/runs/$EOT_RUN_ID/pipeline.log"
```

`eot-runpod all` runs: CUDA/disk preflight -> resumable acquisition -> resumable
prefix mining -> structural mining audit -> audio-only training -> ONNX export ->
official English EoT Bench.
If interrupted, run the same command again. Existing acquisition/mining rows are
skipped and training resumes from `runs/<run-id>/checkpoints/last.pt`. The
launcher updates that checkpoint atomically every 200 optimizer steps and at
each epoch boundary; set `EOT_CHECKPOINT_EVERY` to tune the recovery window.

Useful individual stages:

```bash
eot-runpod preflight
eot-runpod acquire
eot-runpod mine
eot-runpod audit
eot-runpod train
eot-runpod export
eot-runpod evaluate
```

Before committing to the full GPU run, inspect `runs/<run-id>/mining_audit.json`:
it checks label/kind/FVAD invariants and lists representative internal, final,
and incomplete WAVs. Listen to those examples in Jupyter or after downloading
them; the energy-based offline pause miner is intentionally simple and an
automated distribution check cannot replace a short audio spot-check.

Hugging Face validates its cached downloads, while acquisition records a local
SHA-256 for every resulting WAV and verifies those hashes on resume. Those local
hashes detect later corruption; they are not presented as a separately
published upstream checksum manifest.

Whisper-Tiny is small; a 24 GB GPU is ample at the conservative batch size 32.
Increase only after observing utilization. If memory is unexpectedly tight,
start a new `EOT_RUN_ID` with `EOT_BATCH_SIZE=16`; a recovery run deliberately
rejects batch-size and other semantic changes. The [RunPod GPU
catalog](https://docs.runpod.io/references/gpu-types) lists current VRAM; current
prices and availability should be checked in the deployment console rather than
hard-coded here.

## 6. Retrieve and verify artifacts

For repeated transfers, RunPod recommends rsync over full SSH. Use the public IP
and mapped external SSH port shown by the Pod's Connect panel:

```bash
# On the Pod: make the source-side manifest before transfer.
cd /workspace/eot
find runs/base exports/base eval/base -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum > SHA256SUMS

# On the local machine:
mkdir -p artifacts/{runs,exports,eval}/base

rsync -avz --progress -e "ssh -p EXTERNAL_PORT -i ~/.ssh/id_ed25519" \
  root@PUBLIC_IP:/workspace/eot/runs/base/ ./artifacts/runs/base/

rsync -avz --progress -e "ssh -p EXTERNAL_PORT -i ~/.ssh/id_ed25519" \
  root@PUBLIC_IP:/workspace/eot/exports/base/ ./artifacts/exports/base/

rsync -avz --progress -e "ssh -p EXTERNAL_PORT -i ~/.ssh/id_ed25519" \
  root@PUBLIC_IP:/workspace/eot/eval/base/ ./artifacts/eval/base/

rsync -avz -e "ssh -p EXTERNAL_PORT -i ~/.ssh/id_ed25519" \
  root@PUBLIC_IP:/workspace/eot/SHA256SUMS ./artifacts/SHA256SUMS

(cd artifacts && shasum -a 256 -c SHA256SUMS)
```

See RunPod's [SSH](https://docs.runpod.io/pods/configuration/use-ssh) and [file
transfer](https://docs.runpod.io/pods/storage/transfer-files) documentation.
`runpodctl send/receive` is convenient for a single small file; rsync is better
for these repeated directory transfers.

## 7. Validate CPU serving locally

```bash
docker build -t happyrobot-eot-serve:0.1.0 .
docker run --rm -p 8000:8000 \
  -v "$PWD/artifacts/exports/base:/models:ro" \
  -e EOT_ONNX=/models/eot.onnx \
  happyrobot-eot-serve:0.1.0

uv run eot-loadtest --url http://localhost:8000 \
  --concurrency 1 4 8 --requests 200 \
  --out artifacts/loadtest.json
```

The load-test report labels its first request `warmed_first_request_ms` because
the service performs three startup warmups. Measure cold container
startup-to-healthy separately, then report that value plus throughput and
p50/p95/p99 request latency. The
EoT task asks for inference below 100 ms; EoT Bench latency is a different metric
(conversational wait versus compute time) and should be reported separately.

## 8. Cleanup without losing the result

After verifying the downloaded hashes, stop the Pod to end compute billing. Do
not terminate it until you have confirmed the artifacts locally: volume disk is
deleted on termination. Network-volume and storage billing continue separately;
see RunPod's current [Pod pricing](https://docs.runpod.io/pods/pricing).
