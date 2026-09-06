# Synthetic CUDA single-expert calibration

`tests/bench-expert-gpu.cpp` reuses the deterministic quantization and scalar
reference idioms of `bench-expert-cpu.cpp`, but executes a real CUDA-backend
`GGML_OP_MUL_MAT`. There is no CPU graph fallback, model reader, expert routing,
full checkpoint load, service control, or host-preparation code.

## Build and guarded execution

Compile only in the existing CUDA tree (tests enabled):

```sh
cmake --build build-sm120 --target bench-expert-gpu -j 2
```

The target exists only for CUDA builds (not HIP/MUSA). It is deliberately **not
registered with CTest**: neither a build nor ordinary tests should launch a GPU
workload. Implementation verification is compile/link only; runtime correctness
and timing remain unverified until a separately authorized, guarded run.

Only the parent/operator may launch it after applying the GPU-run protocol:
stop the miner with guaranteed restart on success/failure, perform the required
host-prep/dry-run and Tier-A checks (TierA >=34 GiB), launch in a cgroup with
MemoryMax <=40 GiB, and monitor/report system-RAM and VRAM peaks (VRAM <28 GiB).
Those guards are external; the executable does not enforce a cgroup or measure
peak footprint. Its small tensors do not waive the protocol.

The physical harness `evidence/trace-matched-bandwidth/run-physical.py` now waits
up to 60 seconds after stopping qli, observing utilization every two seconds.
Only busy utilization is retried: <5% is still required; low RAM, excessive
VRAM, malformed output or command failure abort immediately. Each observation
logs `nvidia-smi`, `free -g`, and structured values. Command time counts against
the readiness deadline. The pre-GEMM check remains an immediate strict guard.
Set `EXPERTPIN_PHYSICAL_OUT` to an existing empty directory under repo `evidence/`
to preserve earlier results. `EXPERTPIN_GPU_ONLY=1` skips the already measured CPU
offset-read probe; it does not bypass readiness, budgets or guaranteed restore.
Invalid/reused output destinations are rejected before any lifecycle actions or
writes. Cleanup observation/log failures cannot skip stop/SIGKILL attempts; lack
of verified termination still fails the run even though miner restart is mandatory.
The wrapper uses MemoryMax=40G, MemorySwapMax=0, 120s GPU timeout and 300s scope
timeout. It loads no model; DRAFT/launcher DRY are not applicable to this bench.

The payload command **inside that guarded launcher**, not a standalone approval
to run GPU work, is:

```sh
GGML_CUDA_NO_PINNED=1 CUDA_VISIBLE_DEVICES=0 build-sm120/bin/bench-expert-gpu 11
# For the longer calibration, replace 11 with 101.
```

The single mandatory argument is exactly `11` or `101`; no shapes, model paths,
large batches, or unbounded repetition counts are accepted. Select the intended
GPU with `CUDA_VISIBLE_DEVICES`; exactly one device must be visible. The parent
must set `GGML_CUDA_NO_PINNED=1` **before launching**. Missing/other values and
invalid arguments return exit 2 before any explicit CUDA API call. CUDA remains
a linked dependency, so even invalid-argument checks are not CPU-only tests.
Caught allocation, unsupported-op, timing or correctness errors return exit 1
without a partial JSON result. Fatal ggml/CUDA assertions and allocation errors
can abort instead. Backend diagnostics go to stderr.

## Cases and validation

All cases are one expert, one token (`tokens: 1`), with F32 activations and F32
output. Fixed shapes and byte counts match the recorded inventory in
`evidence/bounded-cache-design/local-header-inventory.json`; the benchmark does
not read that inventory or any model at runtime. It checks the quantized tensor's
`nb[2]` and `ggml_nbytes` against the slice size.

| GGML type ID | Quant | Input columns | Output rows | Bytes per expert |
|---|---|---:|---:|---:|
| 20 | IQ4_NL | 640 | 2560 | 921600 |
| 21 | IQ3_S | 2560 | 640 | 704000 |
| 22 | IQ2_S | 2560 | 640 | 524800 |

Seed 17 drives the same integer PRNG as the CPU bench. Production
`ggml_quantize_chunk` uses uniform importance weights, and
`ggml_validate_row_data` rejects invalid quantized payloads. The reference is an
independent scalar double-accumulation dot product of dequantized weights and the
original F32 activations, **not** a comparison against unquantized source weights.
All CUDA outputs must be finite, and relative L2 error must be <=3%, allowing
internal activation quantization as in the CPU bench. Validation runs before
warmups and after timed repetitions. JSON reports maximum relative L2 and maximum
absolute error across those checks; this is execution validation, not model quality.

The graph uses the direct CUDA backend and device tensor allocator; it rejects
unsupported `MUL_MAT` and host buffers. In the inspected backend these three types
have quant-specific MMVQ implementations; a one-token contiguous matrix dispatch
reaches that supported path. The benchmark does not claim to have profiled a
particular kernel. CUDA graph capture/replay is explicitly disabled by backend
parameter `graphs=0` (builds without graph support may log that parameter as ignored).

Cases allocate/release tensors sequentially. Each source F32 matrix contains
`2560 * 640` floats, released before timing, and only one sub-MiB quantized expert
is uploaded at a time. Device tensor allocation is checked against a 2 MiB bound
before allocation. The reported `device_tensor_buffer_bytes` includes backend
padding/alignment, **not** CUDA context, driver/library, quantizer lookup tables,
or temporary backend workspaces. It is not measured peak VRAM/RAM.

## Timing and claim boundary

One JSON document is emitted only after all cases and cleanup succeed. Each case
contains raw `samples_us`, `median_us`, `minimum_us`, `maximum_us`, quant ID/name,
layout, tensor-buffer bytes and correctness results. Repetitions are odd, so the
median is the middle sorted sample. Timing uses a monotonic host clock, drains
the CUDA stream before each sample, then measures one async ggml graph dispatch
**plus stream synchronization**. This includes host dispatch, activation
conversion, device computation, completion wait and any backend per-call costs.
It is **not a CUDA-event, kernel-only duration**. An initial validated dispatch
and three additional warmups are untimed. Context setup, allocation, quantization,
H2D upload, scalar reference, D2H validation and JSON output are excluded.

**Hot synthetic single-expert matvec timing is not routed model GEMM timing.**
The same tiny weights/input are repeatedly reused; cache residency is not measured.
`MUL_MAT` here is not the CPU bench's `MUL_MAT_ID`, a real 512-expert routing trace,
a fused gate/up/down block, a cold transfer, multi-token GEMM, or model decode.
Different quant cases also have different matrix orientations, so this is not
an isolated quant-format speed comparison. Do not infer model tokens/s, physical
VRAM bandwidth (`slice_bytes / time`), PCIe bandwidth, routed-layer compute cost,
or linear scaling with active expert count. Record build flags, contention and
the external protocol's measured budget peaks beside any future results.
