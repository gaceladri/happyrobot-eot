# D022: local audio teacher pilot

Status and decisions: `experiments/D022.json`. Source branch `research/d022`; the incumbent
and official benchmark are unchanged. This is offline label research, not serving inference.

The 24 cases are frozen in `data/research/D022/pilot.json`: eight random controls, eight
student/silver disagreements and eight uncertain student predictions, covering ten training
conversations. The student scored 1,607 unique training pauses. Every selected causal prefix
was checked against its previously validated D019 waveform. Gold accuracy is unknown.

Qwen3-Omni-30B-A3B-Thinking Q4_K_M and bf16 multimodal projector are local only, revision
`6807f126832efa5bce2969fec85a80594e21df9d` from ggml-org. Exact weight hashes, compiler,
runtime revision and settings are in `data/research/D022/runtime.json`. Quantization is a
feasibility choice, not demonstrated parity with the original model. No weights go to W&B.

Run all commands from `/home/ad/Desktop/happyrobot-eot`. Build tools have an isolated venv.
The C++ source is pinned at `4d9176092d00586775af140581bb0b558ddc4389`.

```bash
artifacts/research/tools/build-env-py313/bin/cmake \
  -S artifacts/research/tools/llama.cpp -B artifacts/research/tools/llama.cpp/build-cuda129 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_OPENSSL=OFF
artifacts/research/tools/build-env-py313/bin/cmake \
  --build artifacts/research/tools/llama.cpp/build-cuda129 --parallel 6 --target llama-server
```

Start the teacher on localhost in its own terminal. Do not start a second instance if it is
already running. The serving candidate uses no part of this server.

```bash
artifacts/research/tools/llama.cpp/build-cuda129/bin/llama-server \
  -m models/teacher-qwen3-omni/Qwen3-Omni-30B-A3B-Thinking-Q4_K_M.gguf \
  --mmproj models/teacher-qwen3-omni/mmproj-Qwen3-Omni-30B-A3B-Thinking-bf16.gguf \
  --host 127.0.0.1 --port 8097 -c 8192 -np 1 -ngl 99 -t 4 -fa on \
  --reasoning-format deepseek --reasoning-budget 512
```

Check that the server consumes audio before the pilot:

```bash
PYTHONPATH=../happyrobot-eot-experiments/d022/src .venv/bin/python \
  ../happyrobot-eot-experiments/d022/scripts/smoke_audio_teacher.py \
  --pilot data/research/D022/pilot.json --out data/research/D022/smoke
```

First run one case with `--limit 1`, inspect its two outputs and the audio encoder server log.
Then omit `--limit` to complete the same frozen pilot. Completed and failed calls are retained,
not silently retried. Model/prompt/code/audio identity mismatches reject resumption. Each call
has a 120-second timeout; three consecutive failures stop the run. Stop the client and its
teacher server to release the GPU; no student or incumbent files need cleanup.

```bash
PYTHONPATH=../happyrobot-eot-experiments/d022/src .venv/bin/python \
  ../happyrobot-eot-experiments/d022/scripts/run_audio_teacher.py \
  --pilot data/research/D022/pilot.json --runtime data/research/D022/runtime.json \
  --splits data/research/D016/splits.json --smoke data/research/D022/smoke/smoke.json \
  --out data/research/D022/teacher
PYTHONPATH=../happyrobot-eot-experiments/d022/src .venv/bin/python \
  ../happyrobot-eot-experiments/d022/scripts/report_audio_teacher.py \
  --pilot data/research/D022/pilot.json --results data/research/D022/teacher \
  --out data/research/D022/report
```

Prefix view: target channel only, ending at the same cut as the student input. Hindsight view:
the same target prefix, aligned other-speaker prefix, and six seconds of future audio on both
channels. Later context can inform offline target construction, never student inference inputs.
Teacher prompts contain no IDs, annotations, transcripts, student scores or proposed labels.
The teacher returns EOT/HOLD/ABSTAIN, acoustic cut validity, a reason category and brief audible
evidence. Confidence scores are not requested. Abstentions are not converted to HOLD.

`report/adjudication_queue.jsonl` is the local source for learning from disagreements. A
student/silver or teacher/silver disagreement is not itself a confirmed error. Only independently
adjudicated cases may justify rule changes or high-trust relabeling. Do not enter model answers
as human reviews. The existing D019 human review scale gate remains in force. Freeze fresh
train-only audit cases before validating revisions learned from this pilot; confirmation data
and the official LiveKit evaluation remain untouched.

W&B receives only aggregate counts/timings/plots and configuration, never case audio, transcripts,
raw responses or checkpoints. Runtime and plot paths are recorded in the local experiment ledger.

References: [technical report](https://arxiv.org/abs/2509.17765),
[original model](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking),
[runtime conversion](https://huggingface.co/ggml-org/Qwen3-Omni-30B-A3B-Thinking-GGUF).
General audio benchmark leadership motivates this pilot; EoT annotator superiority is unproven.


Research follow-ups, not established local gains:
- Automatic offline annotation of turn-taking transitions in task-oriented dialogue studies
  whole-conversation annotation with surrounding speech, distinct from online endpointing:
  https://www.sciencedirect.com/science/article/pii/S0885230822000857 .
- Fonseca et al. identify likely missing audio labels with a teacher and mask suspect losses
  rather than unconditionally replacing labels: https://arxiv.org/abs/2005.00878 .
  A possible later hypothesis is instance-specific abstention/weighting after review, not a
  repeat of E013's blanket heuristic exclusion. That prior seed replication already failed.
- Qwen3.5-Omni is a newer hosted family in the official catalog. This pilot uses a reproducible
  open-weight local candidate, not a claim that Qwen3-Omni is the strongest current hosted model:
  https://docs.qwencloud.com/developer-guides/getting-started/vision-models .


## Resume after restricted-session interruption

The CUDA server and console binaries both built successfully. On the next turn the session
changed to restricted permissions: CUDA initialization and socket creation are unavailable.
This is a runtime-access failure, not evidence against the teacher. D022 GPU calls remain
unexecuted; no labels were generated. Its original settings and frozen cases are preserved.
P023 separately probes CPU console audio with the same assets and grammar-constrained boolean
output, without sockets. See `experiments/P023.json`; do not mix it into D022 quality results.
W&B synchronization is pending until network access is available. No download or main build
needs repeating. The local source/weights/benchmark remain intact.


## D024: CPU console continuation

P023 passed both speech/silence controls in20.16/20.47seconds, including model startup.
D024 uses the same frozen24 cases and model assets, with CPU-only fresh processes and
JSON-schema-constrained output. This avoids persistent conversation or sampler state leaking
between cases. It is a distinct registered runtime, not an unreported replacement of D022.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/research_teacher_cpu.py --limit 1
# After inspecting both first-case judgments, continue cached results:
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/research_teacher_cpu.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../happyrobot-eot-experiments/d022/src .venv/bin/python \
  ../happyrobot-eot-experiments/d022/scripts/report_audio_teacher.py \
  --pilot data/research/D022/pilot.json --results data/research/D024/teacher \
  --out data/research/D024/report
```

Runtime/source hashes and exact arguments are in `data/research/D024/runtime.json`; completed
case files are immutable under resumption. A120second per-call timeout,45minute per-invocation
ceiling and three-consecutive-failure stop limit CPU cost. Per-case timings include model reload,
and are unrelated to production HTTP or endpointing delay. No student has been trained.

Selection/reporting uses0.5 as a diagnostic student classification threshold. This is not the incumbent official operating policy (which also includes thresholding and a waiting action). In particular, a mismatch at0.5 does not by itself establish a production cutoff or latency error. The strongest eight selected silver EOT disagreements have student scores0.012–0.028, while the random first case has0.382.
