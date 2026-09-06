# Request-scoped CUDA expert routing trace

Set `GGML_MOE_GPU_TRACE_FILE=/absolute/new/gpu-routing.csv` **before process start**.
The path is exclusively created; an existing path logs an error and disables
this trace. This is separate from `GGML_MOE_TRACE_FILE`: no CPU shadow ownership,
capacity, or `GGML_MOE_TRACE_REQUEST_ONLY` flag is required. The environment is
read once. No tracing/readback occurs by default.

The existing server target-batch scope is enabled by either trace. Only a valid
active target request (`request_id`, `seq_id`, phase 1/2/3) records IDs. Init,
warmup and MTP draft computations outside this scope do not even read back IDs.
The caller must synchronize target compute before clearing/changing the scope;
this is process-global single-request instrumentation, **not concurrent-context
attribution**. Phase `decode` includes multi-token target verification; positions
are batch min/max, not an assignment of each expert to an individual token.
A 32-output-token MTP run can consequently contain more than 32 target positions.

## What is recorded

Actual I32 IDs consumed by CUDA `MUL_MAT_ID` and `MOE_FUSED_UP_GATE`, including:

* ordinary vector, MMQ and fallback expert matmul paths;
* the second `MUL_MAT_ID` folded into a vector dispatch;
* both up/gate tensors (or the combined tensor when gate is null);
* down projections folded into fused up/gate: vector, MMQ and fallback paths.

Fused children use the parent's actual IDs, as the CUDA kernels do, rather than
assuming the child's nominal IDs were consumed. ID readback happens on the
consuming stream before launching the corresponding expert operation. It packs
only selected columns from strided top-k ID views; no weight payload is copied
or read by the tracer. IDs are deduplicated per weight entry over the whole ID
batch; repeated weights in later operations/batches remain separate entries.
`-1` is the pruned-expert sentinel and is omitted; other invalid IDs fail closed.

CUDA graph capture **and replay are disabled whenever this trace is enabled**,
including init, to avoid missing subsequent requests or capturing host readback.
Per-entry stream synchronization changes performance. This is routing evidence,
not a throughput benchmark, GPU memory-bandwidth measurement or physical I/O.

## Schema, bounds and completion

The CSV uses exactly `join-expert-trace.py`'s base + complete scope-tag columns.
`context=target`; `epoch` is this GPU tracer's active scope serial (not CPU epoch),
and `weight_entry` is its own per-weight serial. Do not numerically join these
serials to CPU serials. Join identity uses the request tags and tensor names.

**`shadow_hit=0,shadow_bypass=1` always means NOT SIMULATED.** It does not mean a
cache miss, capacity bypass, residency result, read or transfer. Keep the GPU
CSV separately named and do not pass it to CPU shadow hit/miss accounting.
The joiner can validate ordinary supported IQ4_NL/IQ3_S/IQ2_S tensor layouts,
not every runtime type; runtime repacking or different names/layouts may cause
it to reject the trace. A successful join proves offsets only, not GPU/file
payload identity. Biases, routing activations, shared/dense experts, non-CUDA
backends and per-token expert frequencies are outside this trace's coverage.

Hard limits per process: 100000 CSV records, 100000 observed dispatch entries,
64 KiB packed IDs per readback, 64 MiB cumulative IDs readback. Host ID selection
storage is bounded by that 64 KiB; no model-sized bitmap or weight copy is made.
Invalid/unsupported layouts, read failures and exhausted budgets stop further
readback and mark incomplete evidence (`dropped`/`error`). Limits do not silently
sample. CUDA API failures still follow the backend's normal fatal error path.

A normal process exit writes `# end written=N dropped=0 error=0` only for a
complete trace; missing footer or any nonzero dropped/error is not complete.
The footer is process-lifetime, not per-request; shut down cleanly before joining.
CPU tests exercise metadata/selection, padding, sentinel/dedup, scope and caps;
CUDA compilation does not prove runtime stream ordering or fused kernel coverage.
A guarded real request is required for that verification.
