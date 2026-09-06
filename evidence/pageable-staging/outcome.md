# Matched pageable endpoint evidence — phase-2 closeout, 2026-09-06

## Result
Code `123de34449a9a9ca0977fa0f36515d6ad603e974` committed and exact origin/main read back before host prep. One guarded model request executed: `evidence/expert-offset-trace/run-20260906T144945/`. CPU/GPU routing + four ordinary backend CUDA set/get entrypoints observed in the same target request25/seq0, 68 prompt/32 generated reasoning tokens, MTP accepted21/drafted22. DRAFT1/nmax4, NCMOE36, CTX8192, cache512MiB, shadow8GiB (not allocation), no pinned host allocations.

**19.464803073979027 tok/s**, goal20 not reached (gap0.5351969260209728). Short, heavily instrumented diagnostic, no matched performance A/B. Cannot attribute difference from previous16.2689/6.6297 to this instrumentation or call it an optimization; state/cache/swap conditions differ. No long-context or generation-quality result.

## What the transfers prove
1427 calls, all host endpoints CUDA-classified pageable. Normal shutdown footer admitted=written=1427, dropped/errors/API/pointer/IO errors all0; parent parsed every row and footer after process exit. This also verifies the production shutdown boundary missing from the standalone singleton unittest (not a new regression test).

| Phase | Direction | Calls | Requested bytes | Copy API wall sum ns | Existing sync wall sum ns |
|---|---|---:|---:|---:|---:|
| prefill | H2D |90|252111152|13192055|1432459|
| prefill | D2H |148|100264768|23510545|52993|
| decode | H2D |449|118703672|6479490|4297260|
| decode | D2H |740|46528576|10693914|199595|

These are actual in-process API-call requested byte counts and host durations, NOT completed DMA bytes/rate or measured driver staging. Async calls gain no explicit sync; existing sync can include earlier stream work. Four API coverage excludes internal copies, routing-tracer readbacks, split/peer copies and UM migrations. Pageable classification is endpoint-only, not an assertion about internal driver bounce buffers or range residency.

CPU45015/GPU15534 routing rows independently fresh-GGUF-header joined to exact offsets, all144 expert tensors covered, zero drops/errors. Same request/phase scope as transfers. No exact expert tensor-name match among transfer rows; this is NOT a proof of zero global expert transfer and not an offset-to-DMA mapping. Transfer names are activation/result/routing tensors: parent checked `src/llama-build-context.cpp:1528,1649-1650` and `src/graphs/build_qwen4exp.cpp:52-55`. Example decode H2D MoE outputs:360 calls/117964800 requested bytes (112.5MiB), not expert weights. Tensor-relative offsets in this trace are not GGUF file offsets. Device dmon has9 numeric samples overlapping request; device-wide/readahead/driver/other-client noise, no per-process calibration claim.

## Budgets and lifecycle
- TierA44.56033706665039GiB MemAvailable, GPU0%; free/nvidia before load and launcher DRY guards OK.
- Cgroup MemoryMax36G <=40G; highest read memory.peak32.40388107299805GiB <40. Final scope journal32.4G memory/811.4M swap peak.
- VRAM device sampled peak22.580078125GiB <28,35 budget samples; not a continuous VRAM maximum.
- Swap sampled peak0.791534423828125GiB reported separately. memory.events max/oom/oom_kill0; high207855.
- Qli stop14:49:47.301215/start14:50:52.946844CEST,65.645629s. Own scope inactive, cleanup_errors[], qli+Chromium active, journal[qli-tune]OK. Parent reissued qli start and verified both active at closeout; no llama-server.
- BinarySHA2569f9a8c35ae22bb8699ed586294c6c718631b8a6e9f1e47d5bb47bdb10dc1f783. Built before code commit; build-info old hash isn't source evidence; actual CUDA compile/link log and hash supplied.

## Advisor and phase 2
Advisor truly rerun, exit2/statusblocked/recommendationnull,24 missing fields across3 proposed cap pairs. No proxies inserted. `input-provenance.md` identifies every missing field: additional RAM staging/NVMe/pageable PCIe rates; per-candidate physical traffic, matched CPU/GPU compute, non-cache reserve decomposition. Existing cold/warm preadv and synthetic hot GEMM observations remain valid in their narrow scopes, not these candidate fields. Total run footprint cannot be subtracted from proposed cache caps to invent reserves.

Analytical baseline15335f1e also had statusblocked/recommendationnull (only3 missing global rates and no candidates). Its advisor code is unchanged. Now routing, endpoint evidence and proposed-cap accounting exist, but no honest cap ranking/baseline speedup comparison is possible. **Measurement sub-slices are complete; integral phase2 and final cap recommendation remain OPEN.** This interim series closes here, not by falsely marking calibration complete.

Next regular03:00 cycle: prioritize a bounded repeatable decode A/B without routing/transfer tracing, enough tokens to measure variance, then one placement change under40/28 guards. Do not spend another cycle remeasuring only routing or populating advisor with synthetic proxies. Actual GPU cache allocator, candidate-specific physical replay/compute/reserve calibration and model-shadow-pressure gate stay explicitly open.

## Quality and research
Independent reviewdeleg_26e0a19b gpt-6-astra APPROVE, one nonblocking singleton-footer regression gap; parent checks in review.md and actual footer above. Full CPU and CUDAserver builds green; targeted18/18; main37/39 with historical BERT/chat failures; full40/43 additionally missing stories260K.gguf/libcurl eval-callback prerequisite. No all-green or TDD-RED claim.

X-research completed before tests/review/GPU; no browser spawned, no external posts. Leads only: https://x.com/inovelloE/status/2096577582741238212 and https://x.com/xueyu1125/status/2096574113943089359. No external performance/model-variant claims adopted. Implementation child initial420s call timeout but completed; independent review finished and no live children before GPU. No further interim runs scheduled.
