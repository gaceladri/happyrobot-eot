# Knowing when to answer

Short solution document - 7 September 2026.

This Markdown and the four-page PDF are built from the same source. Figures are observed results, not production qualification.

## A stronger teacher, a practical student

Train on pauses, adapt Cohere to turn completion, then distil its decisions into a small Whisper encoder.

**Two inference deployments:** distilled Whisper-base on CPU with optimized ONNX, and fine-tuned Cohere on RTX 3090 with TensorRT. At a 5% false-cutoff budget, the deployed models wait **756 ms** and **615 ms**, respectively. Both are re-evaluated after export.

### Improve the decision people actually experience

The historical Whisper Tiny baseline waits 1,032 ms. The selected student reduces this by 27%. At a fixed 300 ms waiting budget, its false cutoffs fall from 30.5% to 13.8%. A false cutoff is measured per eligible mid-sentence pause, not per call.

### Put quality and compute in the same decision

Whisper provides a compact CPU path at **87.4 ms HTTP p95** with eight requests in flight. Cohere delivers lower waiting on GPU at **55.7 ms HTTP p95** under the same load. Its quality curve and response times refer to the same TensorRT artifact. Each service has its own Docker image.

### Keep the two clocks separate

**Benchmark waiting** excludes inference. **HTTP response p95** includes audio processing, model execution and local queuing. Adding a service p95 to mean waiting does not give a measured end-to-end response time. LLM generation and first TTS audio still need integration measurement.

### Compare under the same evaluation rule

The public English overlay uses the same spans, labels, 200 ms score point, HOLD filtering and policy grid as the pinned LiveKit harness. LiveKit v1 reports 543 ms; the deployed Cohere is 72 ms slower. These are local descriptive comparisons, not official leaderboard entries or a blind test. Repeated benchmark inspection limits generalization claims.

*Evidence: Deployment quality, HTTP and artifact receipts; expanded-data report. Public harness 6594d8b3, dataset ca9d98a9; 400 English turns. Full paths, hashes and policy sweeps are included in the companion evidence.*

## Calibrate automatic labels with existing annotations

We did not manually label a new corpus. We reused public annotations to diagnose and tune an automatic labeling pipeline.

### 1. Construct the training target at real pauses

A caller who resumes speaking supplies a HOLD candidate. A substantive agent reply followed by caller silence supplies an EOT candidate. Rules handle timing, word counts, overlap and backchannels. Future behavior is used offline to label a prefix; the model receives only audio available before that cut. Agent behavior is a proxy for intent.

### 2. Derive a calibration reference from TurnBench

TurnBench dev already contains 38 conversations and three annotator tracks per speaker. We map their event annotations into HOLD/EOT references and combine the tracks. The binary reference is therefore derived by our mapper, not an independently supplied gold label for every pause. Strict and overlap-tolerant mappings answer different questions.

### 3. Tune transferable heuristics, then test candidate recovery

Revised rules increase reference-EOT coverage from **12.4% to 20.4%** with **97.4% precision** against the strict derived reference. Rules use timing and word counts, so they transfer to AppTek without TurnBench event labels. Of 275 recovered reference EOTs, only **81** are eligible at the 200 ms decision point; most others occur after the agent starts.

The headline result uses the calibration conversations. A two-fold check found high global precision, but one marginal rule was unstable; it is not an untouched label-quality test. In AppTek, combined rules recover 1,202 eligible candidates, including 826 training candidates. Subsequent curation experiments did not establish a model-quality gain.

### 4. Keep pilots and trained data distinct

The otoSpeech pilot produced 2,956 causal examples from 12 calls. Its review UI contained **0 completed reviews out of 200**; no otoSpeech-trained model or downstream gain is claimed. The selected expanded-data recipe adds AppTek and SmartTurn cuts. It does not inherit every labeling correction simply because that correction was studied.

*Evidence: Historical audit: original labeling commits plus calibration, curation and final training contracts (L001/L008/L009/L020/L040/L041; D019). Pre-existing human annotations do not imply a completed listening audit by us.*

## Make each improvement earn its place

Use controlled comparisons to establish transfer; keep later bundled improvements and unresolved effects explicit.

### Match the task, then adapt the representation

Pause-prefix rather than whole-clip training reduces cutoffs at 300 ms waiting from **55.6% to 31.9%**. With fixed data and budget, frozen Cohere, partial fine-tuning and all-layer low-rank adapters give **789 / 710 / 636 ms** mean waiting over three runs. Adapter fine-tuning saves 152 ms [paired 95% interval: -191, -113].

### Separate distillation from pause augmentation

Training condition | EoT Bench | Krisp
--- | --- | ---
Hard-label control | 896.8 ms | 1,199.9 ms
Texture only | 900.0 ms | 1,202.1 ms
Distillation | 850.0 ms | 1,091.9 ms
Distillation + texture | 801.8 ms | 1,089.4 ms

Three runs per condition; means, not ensembles. Distillation improves both evaluations: **-46.8 ms** on EoT Bench and **-108.0 ms** on Krisp, with paired intervals below zero. Texture adds another public gain; its extra Krisp effect is unresolved. Teacher and student receive identical realized causal audio; only Whisper runs at inference.

### Distinguish research progress from component attribution

Expanded-data research uses 144,197 training examples, a fixed 16,390-row validation set and 353,930 teacher targets. Its selected student reaches 756 ms. Data, teacher continuation and denser targets were changed together, so their separate gains are not identified. This recipe has no new Krisp readout.

### Treat failed ideas as decisions, not hidden successes

Curation failed replicated controls; longer training did not repair transfer. Teacher averaging slightly improved AUC yet worsened waiting. An evaluation audit attributed 157.6 ms of an apparent dataset gap to a different firing rule. These findings prevent false attribution and repeated mistakes.

*Evidence: Matched training study; adaptation, scoring audit, controlled distillation and expanded-data studies. Paired intervals condition on policies selected on the inspected benchmarks. The 801.8 ms recipe is the historical control; the selected expanded-data student reaches 756.25 ms in its CPU Docker.*

## Deliver the service; qualify the voice interaction

The model API is implemented. The live turn-taking policy, feedback loop and production capacity still need qualification.

### Measure the actual inference containers

Full HTTP p95 | 1 request | 4 requests | 8 requests
--- | --- | --- | ---
Whisper / CPU ONNX | 20.1 ms | 46.2 ms | 87.4 ms
Cohere / GPU TensorRT | 18.3 ms | 41.2 ms | 55.7 ms

Intel i9-10900X (8 ONNX threads); RTX 3090 (TensorRT mixed FP16 / FP32, 2 frontend threads). One fixed 8 s PCM16 payload, 400 warm requests per load level and service; 2,400 total with zero errors. No CPU quota. These closed-loop sweeps do not establish concurrent-call capacity.

### Separate the scorer from the response policy

VAD identifies a pause. The API accepts up to 8 s of audio, uses 4 s internal context, and returns a score, flag and timings. It implements warm-up, bounded admission, deadlines and model identity. The agent must still manage commitment, resumed speech and LLM/TTS cancellation.

### Qualify the runtime, then recheck the policy

The CPU export uses fused attention and normalization. TensorRT uses merged adapters, shape-specific profiles, stable CUDA graphs and protected FP32 reductions. Numerical probes precede a full public quality replay. Batch-dependent scores are checked under a shared policy. INT8 conversion diagnostics remain separate from the selected artifact; a historical 612 ms result is not a TensorRT result.

### Next investment: representative labels and live behavior

Log decisions, model/policy versions, latency, timeouts and resumed speech. Track interruptions, waiting, corrections, number capture and task completion. Sample uncertain and consequential cases for explicit review; no complaint is weak evidence. Minimize retained audio.

Build an independently reviewed call set by accent, channel and overlap. Freeze selection rules; test, shadow, canary and retain rollback. Limits include benchmark reuse, unresolved short-audio similarities and unknown pretraining overlap. Historical Tiny remains the formal research incumbent; the two deployments are selected for this exercise, not promoted to production.

*Evidence: Deployment quality and HTTP receipts; CPU and GPU service source; reproduction manifest. Checkpoints restored from the archive, exports and engine hashed. Model weights are distributed separately from the document bundle.*
