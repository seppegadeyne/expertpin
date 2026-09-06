# Bounded CPU expert observation trace

Linux opt-in: set `GGML_MOE_TRACE_FILE` to a **new** path and enable
`--expert-cache-sim-mib` (launcher: `EXPERT_CACHE_SIM_MIB`). Exclusive creation
refuses overwrite. Trace belongs to the single owning shadow context and closes
when that context releases its shadow. No file is opened when unset.

CSV columns describe the production shadow observer's serialized access order:
`seq`, `weight_entry`, scheduler `epoch`, `ids_rows`, tensor name/type, expert ID,
runtime stride, **tensor-relative** byte offset, runtime slice bytes, shadow hit
and shadow bypass. Duplicate and invalid IDs are excluded by the existing hook;
within one weight entry IDs are ascending, NOT token top-k order. A fused up/gate
with two weight tensors produces two weight entries. `ids_rows` is a shape, NOT
an authenticated prefill/decode/MTP phase tag. Epoch is the global prefetch epoch;
without an active prefetch pool it can remain constant. It is not a graph boundary,
token index, context ID, or timestamp.

Hard output bound: first 100,000 observations, then count dropped records. No
payload buffer; stdio buffer only. CSV quotes names per RFC 4180. The final
non-CSV footer `# end written=N dropped=N error=N` is emitted on release. Missing
footer, write error or dropped>0 means incomplete evidence. Output I/O happens
synchronously under the shadow mutex and can perturb timing; do NOT report its
throughput as an uninstrumented benchmark. Open/write errors warn on stderr;
inference remains enabled. Use a local regular file, not a special device.

## Deliberate limitations

- Logical CPU kernel-distinct accesses only. GPU-resident expert kernels are not
  observed. Shadow misses do not prove page faults, storage reads or I/O latency.
- Runtime tensor-relative offsets are not GGUF absolute file offsets. Repacking,
  fused layouts and model/draft aliases must be matched before a file-offset join.
- No request/context/phase attribution yet. In MTP, multi-row means neither
  automatically prefill nor automatically decode. Do not infer those labels.
- Counters reset mid-ownership does not restart the trace: this diagnostic is an
  ownership-lifetime prefix, not automatically a per-request capture.
- This is a prerequisite slice, NOT completion of issue #2 physical calibration.

Tests exercise byte-exact CSV/cap/footer behavior and the real CPU shadow hook:
thread-zero filtering, distinct/invalid IDs, offsets/hits, ownership, no overwrite.
No full model load is needed for the unit tests.
