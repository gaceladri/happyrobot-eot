# Reproduce inference and evaluation

Run commands from the repository root. There are three distinct levels of reproducibility: run the exact delivered files; rebuild a model export and validate it; retrain the model. Reusing an artifact can reproduce its bytes. A new TensorRT build or training run may produce different bytes and must establish its own measurements.

## 1. Install the exact inference files

The versioned `configs/inference/selected.json` enumerates five files: the Whisper ONNX and metadata; the Cohere plan, metadata, and original audio frontend. The bundle is about 5.33 GB. It contains no training data. Model files are not embedded in Git or the presentation.

```bash
uv run python scripts/download_weights.py          # public Google Drive link pinned in the script, tar SHA-256 verified
uv run eot-artifacts verify

# The same files received another way (disk, another mirror):
uv run eot-artifacts install --source /path/to/model-bundle --root artifacts/deployment
```

The source bundle must contain `whisper-cpu/eot.onnx`, `whisper-cpu/eot.json`, `cohere-gpu/eot.plan`, `cohere-gpu/eot.json`, and `cohere-gpu/frontend.pt`. Paths can be relocated; hashes cannot. The installer reads only these enumerated files. Use `--model whisper-cpu` for CPU only.

`scripts/download_weights.py` holds the public download link ([Google Drive, anyone with the link](https://drive.google.com/file/d/1Knd2B29FpMuOiUNmvyOV1Nf_uDj2JgI9/view?usp=sharing), SHA-256 `8c6ca046bde26f9a3860c5fe7e6ec0100cf6c6a70166beda95090814f1eb47b5`) and resumes interrupted downloads; `--model whisper-cpu` downloads and installs the CPU model only. The bundle can also be rebuilt from the separately supplied checkpoint archive described in [training reproduction](reproduce-training.md). The archive predates the final TensorRT build, so recovering a checkpoint is not the same as recovering the selected plan.

## 2. Build and run

```bash
docker compose -f compose.inference.yaml up -d --build whisper-cpu
# Requires NVIDIA Container Toolkit and a compatible GPU/driver:
docker compose -f compose.inference.yaml up -d --build cohere-gpu
```

Images pin their Python base digest and Python package versions. Models are mounted read-only; services run as a non-root user, with a read-only root filesystem and no added capabilities. Ports bind to localhost. Do not expose the unauthenticated scorer directly on the public Internet.

The measured CPU configuration uses eight ONNX Runtime threads and eight admitted requests. Cohere uses two CPU frontend threads, one GPU execution thread, a maximum batch of eight, and a 2 ms batching window. The selected plan was built with TensorRT 10.16.1.11 for the local RTX 3090, with CUDA 13 runtime libraries. CPU-only PyTorch computes the original Cohere frontend; TensorRT runs the entire encoder and head.

Cohere uses pinned I/O buffers, separate fixed contexts for batches 1–8, and CUDA graphs captured after warmup. Batch profiles are 1/1/1, 2/4/4, and 5/8/8. Builder optimization level is 4, workspace is 4 GiB, and TF32 is disabled. FP32 is retained for sensitive normalization, reduction, softmax, pooling/head, and sigmoid operations. The engine is therefore mixed precision, not uniformly FP16.

Health is available at ports 8891/8892. Both services return model hashes and runtime configuration. A body over 2 MB is rejected, admission is bounded at eight requests, and a timed-out native operation retains its capacity slot until it finishes. See [the API and failure policy](operations.md).

## 3. Repeat HTTP measurements

```bash
uv run eot-loadtest --url http://127.0.0.1:8891 --wav /path/to/clip.wav \
  --concurrency 1 4 8 --requests 400
uv run eot-loadtest --url http://127.0.0.1:8892 --wav /path/to/clip.wav \
  --concurrency 1 4 8 --requests 400
```

For the exact historical payload, restore the recorded training WAV and use `scripts/measure_http.py`. It checks the normalized PCM payload SHA before measuring, requires a fresh output path, and records the image, health configuration, GPU state, and request results. `--root` relocates the data tree.

```bash
uv run python scripts/measure_http.py \
  --url http://127.0.0.1:8892 --container eot-delivery-cohere-gpu-1 \
  --out evidence/reproduction/cohere-http.json
```

Run configurations sequentially on an otherwise quiet host. The reported 400-request sweeps are warm, fixed-payload, loopback HTTP measurements with no CPU quota. They include decoding, frontend, inference, queueing and response handling; they exclude startup and geographic network latency. Requests/s do not establish simultaneous-call capacity. Keep new receipts separate from the original measurements.

## 4. Rebuild and qualify exports

Install training/export dependencies with `uv sync --locked --extra dev --extra data --extra backbone`. TensorRT compilation additionally needs the pinned packages in `requirements/gpu-serve.txt` and a CUDA-capable PyTorch environment. The GPU serving image deliberately uses CPU PyTorch and is not the training/export environment. Do not install its CPU Torch wheel over a CUDA training environment.

Restore the selected source checkpoints, pretrained snapshots, and frozen probes to the recorded `data/` layout. `export_models.py --root` supports another data root. It rejects a changed selected checkpoint or probe panel, and refuses nonempty output directories.

```bash
uv run python scripts/export_models.py \
  whisper --root "$PWD" --out artifacts/rebuild/whisper-cpu
uv run python scripts/export_models.py \
  cohere --root "$PWD" --out artifacts/rebuild/cohere-gpu

# Run these with a CUDA PyTorch + TensorRT 10.16.1.11 build environment:
python3 scripts/build_trt.py \
  --onnx artifacts/rebuild/cohere-gpu/eot.onnx \
  --out artifacts/rebuild/cohere-gpu/eot.plan --precision fp16 --tuned
python3 scripts/verify_trt.py \
  --engine artifacts/rebuild/cohere-gpu/eot.plan \
  --source-dir artifacts/rebuild/cohere-gpu --tolerance 0.02
```

The selected Whisper export passed a 1e-4 probability tolerance over 97 frozen training/synthetic probes, exercised at batch sizes 1/4/8. Cohere passed 0.02 on 34 frozen probes at the same batch sizes; its maximum observed probability difference was 0.002204. Those panels do not bound every possible input. Numerical acceptance is followed by full public quality evaluation. Never attach the source checkpoint's score to an unmeasured converted engine.

A rebuilt engine must have its own sidecar, public result, HTTP receipt and selected-artifact manifest. Do not replace hashes in the historical manifest to make a different engine appear identical. The default compose volume can be overridden with `EOT_MODEL_ROOT` to exercise a separately qualified build.

## 5. Recompute public quality with the original harness

Use `livekit/eot-bench` commit `6594d8b3b8af385b15f116dde310ce45af92d646` and dataset `livekit/eot-bench-data`, English validation revision `ca9d98a9686b920a2d8c9eb984224ba9be74e4dd`. Install the pinned harness as described in its README; the Docker images do not require the harness for scoring.

```bash
git clone https://github.com/livekit/eot-bench.git third_party/eot-bench
git -C third_party/eot-bench checkout 6594d8b3b8af385b15f116dde310ce45af92d646
uv pip install -e third_party/eot-bench

uv run python scripts/public_quality.py \
  pack --out eval/reproduction/inputs
```

The pack produces 11,191 aligned score points from 400 English turns and approximately 5.7 GB of normalized waveforms. Mount that directory and the evaluator script into the relevant inference container; `infer` loads no labels. For example:

```bash
mkdir -p eval/reproduction/cohere
# The container's uid 10001 needs write access to this output directory.
sudo chown 10001:10001 eval/reproduction/cohere

docker compose -f compose.inference.yaml run --rm --no-deps \
  -v "$PWD/eval/reproduction/inputs:/inputs:ro" \
  -v "$PWD/eval/reproduction/cohere:/results" \
  -v "$PWD/scripts:/evaluation:ro" \
  cohere-gpu python /evaluation/public_quality.py infer --kind gpu \
  --model /models/eot.plan --inputs /inputs --out /results --batches 1 4 8

uv run python scripts/public_quality.py \
  metrics --inputs eval/reproduction/inputs --out eval/reproduction/cohere
```

Stop the GPU service before running another engine on a memory-constrained GPU. Use the analogous CPU service, `--kind cpu`, and `/models/eot.onnx` for Whisper. The quality adapter uses two CPU threads, matching the original quality measurement; a separate frozen-panel check established identical outputs at the selected eight threads.

The harness reports mean waiting and false cutoffs using its original threshold/action-delay/timeout grid, a 0.2 s score point, and a 5% cutoff budget. Verify batch-specific operating points and apply the batch-1 policy unchanged across batches. Preserve the prediction hashes, row identity, summaries and policy grid. Repeated benchmark inspection and same-benchmark policy tuning remain limitations even when every computation is reproducible.
