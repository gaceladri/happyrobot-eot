# L041: Whisper integer-QAT native CPU addendum

Snapshot: 2026-09-07T04:34:15.049766+00:00. **Experimental native CPU backend verified; LiveKit remains ahead.**

This is an additive serving experiment using the already trained final Whisper pair. It does not repeat training, change the quantization-aware training recipe, relax the frozen tolerance, or replace the original failed QDQ ONNX export. The ONNX contract remains unresolved. No model is promoted.

## Numerical repair and real integer execution

Layer diagnostics verified that all 36 trained INT8 weight matrices were copied exactly. Small differences between the floating-point operators in PyTorch and ONNX Runtime can change activation rounding codes and propagate through later layers. Keeping the PyTorch floating-point remainder while executing the projections with signed INT8 products and INT32 accumulation reduced the discrepancy.

The saved and reloaded QAT artifact passes all 97 frozen training-derived probes at batch 8 and batch 1: maximum probability error **0.01287657 <0.02**. Serialization adds zero observed error; batch dependence is below 1.2e-7. Its graph contains 36 `aten::_int_mm` nodes and its loaded execution trace has 36 integer calls. An exact accumulator of 33,032,065 matches an INT64 reference and is not exactly representable in FP32. The matched FP artifact has zero observed reference discrepancy and no integer nodes.

Artifacts are TorchScript CPU files tied to PyTorch 2.14.0. The convolution, attention and other remaining operations retain floating-point execution; this is not an all-integer network. No GPU INT8 speedup is claimed. The numerical pass applies to the registered panel, not to all possible audio.

## Actual public quality of the saved artifacts

Both files scored the same 11,191 frozen public English points using the unchanged audio frontend and harness. Matched FP: **774.00 ms** mean EoT delay at the 5% cutoff budget; QAT: **774.00 ms**. LiveKit v1 remains at **543 ms**. These delays are different from inference/HTTP latency.

At each model’s benchmark-selected operating point, QAT minus FP is **+0.00 ms**, conditional paired 95% interval **[+0.00, +0.00] ms**. With the FP policy frozen, the QAT delay difference is **-6.00 ms** and cutoff is **5.248%**. A point above 5% does not satisfy the fixed budget.

Equal EoT delay does not mean identical probabilities or all quality measures: AUC FP/QAT is **0.950996/0.950848**, and AP is **0.900987/0.899924**. The selected FP and QAT thresholds are 0.17 and 0.18, respectively.

Policies are selected on this repeatedly inspected benchmark; 1,000 turn-cluster bootstrap replicates with seed 0 are descriptive and conditional, not independent blind validation. The runtime changes were motivated by training-derived numerical checks, not selected to improve public scores.

## Full HTTP on the local CPU

Intel i9-10900X, 20 logical CPUs; loopback HTTP with the unchanged frontend/API, one fixed training-audio payload, 400 requests per concurrency and maximum eight in flight. One/two-thread extensions each passed all 97 training-derived probes against four threads at tolerance 1e-6 before timing. CPU work was serialized with the shared host lock; there was no local GPU use. Timing is one observed run per setting, not a deployment capacity confidence interval.

| Artifact | CPU threads | p95 c1 / c4 / c8 (ms) | HTTP errors |
|---|---:|---:|---:|
| FP | 1 | 74.45 / 81.77 / 103.73 | 0 |
| FP | 2 | 48.59 / 62.80 / 111.10 | 0 |
| FP | 4 | 34.55 / 52.26 / 111.52 | 0 |
| QAT | 1 | 51.06 / 58.94 / 84.23 | 0 |
| QAT | 2 | 35.69 / 46.03 / 82.87 | 0 |
| QAT | 4 | 22.18 / 41.49 / 101.34 | 0 |

![Native CPU quality and latency](figures/native_cpu.png)

Runtime thread selection uses latency only; no checkpoint or data is selected on public quality. The 100 ms line is a reference: the exercise does not specify hardware, percentile or concurrency. Original ONNX measurements remain in the main report and are separate engines.

## Files and limits

- Evidence inventory with hashes (separate research archive: `output/L041/native_cpu_inventory.json`) (archived with the research tree).
- [Main L041 report](REPORT.md) preserves original rejected ONNX artifacts, all remote results, rental closure and data-audit limitations.
- Native files: `data/research/L041/native_cpu/{fp,qat}/eot.ts`, with numerical receipts, profiling traces, registered scoring, paired statistics and HTTP results beside them.
- Krisp and local Cohere CPU evaluation remain deferred. The H200 rental is deleted. This additive CPU work incurs no new rental cost.
- W&B receives aggregate metrics only. The protected production ONNX and source remain unchanged.
- The goal of beating LiveKit is unmet; this repair establishes a measured experimental QAT serving path, not production readiness.
