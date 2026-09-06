# Request-scoped logical CPU expert trace

Set `GGML_MOE_TRACE_REQUEST_ONLY=1` alongside `GGML_MOE_TRACE_FILE` and
`--expert-cache-sim-mib`. The existing unscoped format remains the default.

The server scopes target `process_batch_tokens` calls using actual prompt-batch
ranges, the internal task ID and llama sequence ID (slot ID). Columns appended:
`context,request_id,seq_id,phase,pos_min,pos_max`. `context=target`; `phase` is
`prefill`, `decode` (including multi-token speculative target verification), or
`mixed` if a batch straddles a prompt boundary. Positions bound the whole batch,
not the exact token that routed each expert. Kernel-distinct experts are still
ordered by ID, not token-level routing order.

Synchronization before/after each scoped target decode prevents async work from
inheriting stale scope tags; this diagnostic can alter timings. The request ID
is server-internal, not the OpenAI response ID. Initialisation/warmup, draft model
execution, multi-sequence batches and other server decode paths are excluded,
not assigned invented phases. Excluded records cannot fill the 100000-record cap.
The cap remains per trace file/server lifetime, not reset for every request.
Only use the scope with a single server/context computing at once; the shadow
and scope are process-global. CPU observations of target expert kernels only:
this is NOT a complete CPU+GPU routing trace, physical I/O trace or model-content
identity proof. Scope tags do not change aggregate shadow stats, which continue
to include initialisation and may include other contexts.

`evidence/expert-offset-trace/run-guarded.py` enables this option for the existing
single 32-token chat probe, with draft MTP, cgroup and RAM/VRAM guards unchanged.
The probe is not a long-context performance/quality benchmark.
