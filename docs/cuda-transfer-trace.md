# Bounded request-scoped CUDA transfer trace

Set `GGML_CUDA_TRANSFER_TRACE_FILE=/path/to/new/cuda-transfers.csv` before
starting the server. Unset/empty is disabled (default). The file is created
exclusively: existing files are not overwritten; open failure warns and disables
tracing. Configuration is sampled once per process. The guarded 32-token launcher
`evidence/expert-offset-trace/run-guarded.py` now sets this variable alongside
CPU routing, GPU routing and device-wide PCIe observations. This change alone
is not evidence that the next run was rebuilt or that transfers occurred.

## Scope and coverage

Shares `ggml-moe-stats.h`'s process-global target request/batch scope. Enabling
this trace alone activates the existing server scope guards, including their
boundary synchronizations. It needs neither a CPU cache shadow nor GPU routing
tracing. Rows snapshot request ID, sequence, prefill/decode/mixed phase and
position range at entry. Warmup, initialization, unscoped draft work and invalid
or multi-sequence scopes do not consume the cap. Concurrent contexts are not
supported; use the existing single-server bounded diagnostic protocol.

Exactly four ordinary-buffer entrypoints in `ggml-cuda.cu` are instrumented:

- `ggml_backend_cuda_buffer_set_tensor` / `get_tensor` (synchronous contract)
- `ggml_backend_cuda_set_tensor_async` / `get_tensor_async`

Excluded: split buffers, internal CUDA copies, direct routing IDs readback,
peer/device-to-device copies, host-buffer memcpy, memset, graph replay internals,
unified-memory migration and allocation/registration. Empty evidence does not
mean there was no PCIe traffic. No expert ID or expert-transfer attribution is
inferred from a tensor name. GPU expert routing is logical access to possibly
already-resident weights, **not H2D**. dmon is device-wide, not this coverage.
HIP/MUSA entrypoints retain original operations without transfer instrumentation.

## CSV interpretation

Comment header declares schema and coverage. Each row contains an admission ID,
`context=target`, request/sequence/phase/position scope, backend API, H2D/D2H,
quoted tensor name, tensor-relative byte offset, requested byte size, device,
host pointer kind/query status, monotonic start time, separate copy and existing
synchronize API wall times (nanoseconds), and numeric CUDA return statuses.
Tensor names are bounded to `GGML_MAX_NAME`, quotes escaped, controls replaced
with `?`. No raw pointer addresses or payload data are emitted.

`copy_api_wall_ns` surrounds **only cudaMemcpyAsync**. `sync_api_wall_ns`
surrounds the pre-existing cudaStreamSynchronize for synchronous entrypoints;
it can include earlier stream work. Async rows have sync status -1 and zero
sync time. Device selection, pointer classification, writer locking and file
I/O are excluded from these intervals. These are **host API wall durations,
not DMA durations, completed-transfer byte counts, or DMA bandwidth**. Do not
turn bytes/API time into a pageable staging rate or advisor calibration. A warm
preadv measurement is not RAM-to-device staging either. The trace provides
matched call evidence, not a measured pageable staging bandwidth candidate.

After successful calls, CUDA >=11 `cudaPointerGetAttributes` classifies the
host-side endpoint as `pageable` (unregistered), `pinned` (host), `device`,
`managed`, or `unknown`. This classifies only the supplied pointer, not all pages
in its range, residency, driver bounce buffers or time spent staging.
Unsupported/older runtime: status -1; pre-existing CUDA error: -2; skipped
(null/zero size/failed copy or wait): -3; otherwise CUDA's query status (0 is
success). Query failure is unknown, never guessed pageable. Prior error state
is inspected non-destructively; no cudaGetLastError clearing is introduced.
Unexpected query failures can leave CUDA error state set and may surface prior
async failures: this opt-in diagnostic is not performance-neutral.

## Bounds and completeness

Hard cap: 100,000 admitted calls across the process, including in-flight calls;
each serialized row is less than 1024 bytes. Constant-size row/name storage,
no payload allocation or retained event queue. After the cap or I/O failure,
scoped calls increment drops without clocks or pointer queries. All disabled
calls take the original copy/wait path: no pointer query, clock read, device
event, or extra synchronization. Enabled async copies never gain a wait.
Writer locking, flushing per row, classification, and enabled server boundary
waits can perturb inference. Routing tracing may add its own overhead.

Normal shutdown writes a comment footer with `admitted`, `written`, `dropped`,
`errors`, and API/pointer/I/O error counts. Require a footer, admitted=written,
zero drops and zero errors for complete evidence **within this narrow scope**.
Unknown/unsupported classification must be checked per row even with zero
errors. Abort/SIGKILL can omit the footer; output-device failure can prevent
writing the footer at all (stderr warning). Neither case is complete evidence.
Counters are process-wide, not per-request. Row order is completion/write order;
admission IDs and scope snapshots disambiguate overlapping calls.

## Reproducible CPU-only tests

Use a CPU-only build tree (no model or GPU initialization):

```sh
cmake -S . -B build-cpu-shadow -DGGML_CUDA=OFF -DGGML_METAL=OFF -DLLAMA_BUILD_TESTS=ON
cmake --build build-cpu-shadow --target test-cuda-transfer-trace test-moe-gpu-trace test-expert-manifest -j2
ctest --test-dir build-cpu-shadow -R 'test-(cuda-transfer-trace|moe-gpu-trace|expert-manifest)' --output-on-failure
```

Writer tests cover escaping, bounded names, zero/exact caps including in-flight
reservations, scope snapshots/filtering, classes/directions and footer errors.
Linux tests additionally exercise exclusive-create failure and `/dev/full`, and
use an explicitly synthetic CUDA API double around the production wrapper to
check disabled/unscoped query avoidance, retained sync counts, endpoint
selection and error propagation. Synthetic durations are not observations and
are not saved as GPU evidence. Real CUDA compilation and a matched guarded GPU
run remain separate validation; these CPU tests cannot establish DMA behavior.
