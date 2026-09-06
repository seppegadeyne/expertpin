# Phase-2 input audit

The requested final recommendation cannot honestly be forced from existing measurements.

| Advisor fields | Available evidence | Why not a calibrated candidate input |
|---|---|---|
| ram_bytes_per_s / nvme_bytes_per_s | trace-matched-bandwidth/bandwidth-cpu.json, exact offset sample with mincore and process read_bytes | Effective buffered preadv includes copies/syscalls/Python; warm preadv is not additional staging and cold payload timing isn't isolated hardware throughput. Existing measurement contract explicitly rejects this conversion. |
| pcie_bytes_per_s | prior run dmon; new backend transfer API instrumentation | Device samples aren't per-process pageable calibration. API enqueue wall time isn't completed DMA throughput. |
| cpu_compute_seconds_per_token | expert-gemm/cpu-*-threads*.json | Synthetic repeated hot two-expert MUL_MAT_ID, threads1/8, not actual candidate workload at launcher threads16. |
| gpu_compute_seconds_per_token | gpu-idle-readiness/run/gpu.json | Synthetic hot single-expert MUL_MAT with dispatch/sync; not runtime MUL_MAT_ID/candidate routing. |
| *_bytes_per_token | gpu-request-routing/candidate-accounting.json and matched routing | Static frequency-greedy logical demand has no global temporal ordering or actual allocator misses; cannot become physical traffic by dividing by generated tokens. |
| host_other_bytes / gpu_other_bytes | model cgroup peak + device samples | Total footprint includes existing mapped experts and caches; subtracting proposed caps would manufacture a non-cache decomposition. Different placements haven't been executed. |
| host_cache_bytes / gpu_cache_bytes | explicit proposed 4/1,8/2,12/4 GiB | Valid hypothetical caps only. Runtime GPU expert-cache allocator does not exist in this slice. |

Baseline 15335f1e remains an analytical baseline, not a measured optimization result. Final advisor must run and preserve null recommendation if these inputs remain missing. A valid schema is not evidence. Next cycle should choose a runtime placement/allocator or concrete matched A/B, rather than repeatedly collecting routing or treating unrelated microbench timings as per-token costs.
