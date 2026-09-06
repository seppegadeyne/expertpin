# GPU-request-routing — actual outcome 2026-09-06

Code: 2d3a6f7fb46d5089e590eacec74e7ce4d79f15d3 (origin/main read back).
Raw run: `evidence/expert-offset-trace/run-20260906T142541/`.

## Completed
- Same prompt, seed42, 68 input +32 generated reasoning tokens, target+MTP nmax4, NCMOE36, CTX8192 capacity. 21/22 draft tokens accepted.
- GPU trace: 15534 records (7350 prefill/8184 decode), layers0–11,36 tensors. CPU trace:45015 (20940/24075), layers12–47,108 tensors. Both one identical target request/sequence; both footer dropped0/error0. All records joined to fresh validated GGUF headers. All144 expert tensors observed across backends; this does not prove global per-token ordering or every alternative kernel path.
- Real decode6.629663476210178 tok/s (32000/4826.791ms); goal20 NOT reached. Trace forces IDs readback/synchronization and CUDA graphs off. Not comparable as an optimization A/B with prior16.2689; overhead attribution not measured independently.
- Highest read cgroup memory.peak32.40370178222656GiB /40; MemoryMax36GiB. VRAM device sample peak22.580078125GiB /28,39 samples. Swap sample peak1.6844139099121094GiB separately; final journal32.4G RAM/1.6G swap. Last max/oom/oom_kill events0; high212865. Sampled VRAM is not a continuous maximum.
- Host TierA44.35684585571289GiB/GPU0%; DRY guard green; GGML_CUDA_NO_PINNED1. Qli stop14:25:43.894517/start14:26:52.346518CEST (68.452001s); scope stopped verified; cleanup_errors[]. Qli + Chromium active, qli journal `[qli-tune] OK` independently read.
- Device-wide dmon13 numeric samples14:26:38–50 overlap request. Raw MB/s: rx maximum95,tx maximum2. NOT per-process attribution, pageable staging, link-capacity calibration, expert-transfer bytes, or summed physical traffic. Rounded0 is not zero traffic.
- `candidate-accounting.json`: exact tensor-slice inventory prices joined with decode IDs, per-layer static frequency/byte greedy cap classes1/2/4/8GiB. CPU unique11766 slices8066508800bytes; GPU4371 slices3068441600bytes. These are workload logical working sets, NOT allocated/resident cache or future hit-rate predictions. Ordering is heuristic, not optimal placement/temporal replay. All selected cap byte sums computed from actual rows.

## Still open — no manufactured recommendation
- Matched pageable staging/PCIe calibration and attributable physical traffic, conservative non-cache reserves, routed candidate compute costs. dmon and prior synthetic hot-GEMM microseconds cannot fill these slots.
- Advisor truly rerun: status blocked / recommendation null; now three explicitly proposed cap combinations, missing fields null. Versus analytical baseline15335f1e: more real routing/accounting provenance, but still no justified final recommendation or performance comparison.
- Model-shadow-pressure gate and >=20tok/s at depth remain OPEN.

## Quality
- Independent delegate review APPROVE, with sample-validation and monitor-test caveats. Parent verified scope boundary server-context.cpp4681–4719 and GPU hooks/graph gate in ggml-cuda.cu, bounded writer header. New monitor termination/log cleanup regression8-method harness suite green. Targeted CTest16/16; main34/36 with same BERT tokenizer/chat-template failures; full CPU build + CUDA server build green.
- No test-first behavioral RED claim. Failed patch validation changed no files; temporary test indentation error fixed before tests/commit. Implementation child nested CLI attempts are not counted as review; no live children or its known CLI process before GPU.
- X leads, no external metrics: https://x.com/SamSchieds/status/2096508476499562784 and https://x.com/DoingFedTime/status/2096079245437001737. Research completed before tests/review/GPU.
