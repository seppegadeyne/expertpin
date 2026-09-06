# Bounded host expert cache: sizing and evidence limits

Date: 2026-09-06. CPU-only design slice; **no real eviction implementation, no new model run**.
Goal remains >=20 tok/s within 40 GiB run RAM and 28 GiB absolute VRAM.

## Reproducible artifact

```sh
python3 scripts/analyze-expert-cache.py \
  --stats evidence/cache-shadow-ab/20260905T032156+0200/shadow-32g/expert-stats.json \
  --assert-stable-cold-start
python3 tests/test-expert-cache-analysis.py
jq -f evidence/bounded-cache-design/summarize-layout.jq \
  evidence/bounded-cache-design/local-header-inventory.json
```

The analyzer defaults to **unknown** unless the caller explicitly asserts a fresh
cache and immutable tensor identity/stride/size over the observation. It rejects
inconsistent counters, non-finite/disagreeing hit rates, duplicate JSON keys and
invalid capacities. Its synthetic order counterexample is not model routing.

The captured inventory is sufficient to reproduce sizing without model files.
Optional local refresh (not portable tooling; uses an existing host helper):

```sh
python3 evidence/bounded-cache-design/inventory-local-headers.py \
  --header-reader /home/seppe/.hermes/profiles/expertpin/skills/devops/gpu-ram-prep/scripts/gguf_tensor_sizes.py \
  --model-dir /home/seppe/Models/qwen3.8-flash-next/AD-4.27bpw-Q4_K_M-M64
```

This helper reads at most a 64-MiB prefix per shard sequentially, including some
payload beyond the header; it does not instantiate or infer with the model.
Offset spans can include padding. The checked-in layout summary independently
uses quant-block formulas and finds payload bytes equal to spans for all 144
expert tensors in these 33 shards. It is NOT a general GGUF validator.

## Actual expert layout, not a uniform-layer estimate

| Projection | Quant | Tensors | Bytes / contiguous expert slice |
|---|---|---:|---:|
| down | IQ4_NL | 48 | 921600 |
| gate/up | IQ3_S | 24 | 704000 |
| gate/up | IQ2_S | 72 | 524800 |

Each tensor has 512 experts. Shapes in GGUF order: down [640,2560,512],
gate/up [2560,640,512]. IQ3_S gate/up occurs in layers 0–3 and 40–47;
the other gate/up layers use IQ2_S. Whole expert payload: 50646220800 bytes
(47.16796875 GiB). This is why multiplying a global average by NCMOE is not exact.
Quant type/block mapping: `ggml/include/ggml.h:412-414` and
`gguf-py/gguf/constants.py:2206-2208`; C structs also assert block byte sizes in
`ggml/src/ggml-common.h:474,510,590`.

For the single-device branch, `src/llama-load-tensors.cpp:274-287` selects CPU
expert overrides from the last layer backwards (the CLI help says first N).
Layers 18–47 contain 46080 projection slices and 31745638400 bytes:
**29.5654296875 GiB**, exactly equal to the captured shadow misses and resident
bytes. That match is consistent with full CPU-bank coverage, not proof of a
small persistent hot subset. It does not expose which phase first touched it.
The old 29.480-GiB average-layer projection is superseded for this NCMOE=30 set;
historical measured cgroup/VRAM peaks remain unchanged.

## What the captured access pattern actually means

`ggml/src/ggml-moe-prefetch.cpp:709-731` deduplicates expert IDs across a kernel
entry, then accesses selected IDs in ascending order once per weight tensor.
Thus requests are **distinct projection-slice observations per kernel entry**,
not token activations, original router ordering, or all physical memory reads.
JSON stores totals only (`src/llama-expert-stats.cpp:182-194`). Warmup, prompt
processing and measured requests share the observer in `run-ab.sh`; no
decode-only phase markers or chronological expert IDs were persisted.

Fresh acquire resets policy (`ggml-moe-prefetch.cpp:555-562`). With fixed keys
and sizes, no bypasses and no evictions, each miss inserts a unique slice.
But the policy invalidates changed sizes without counting them as evictions
(`ggml-moe-cache-lru.cpp:38-43`): zero evictions alone is not enough. The analyzer
therefore keeps the stable-layout assumption explicit, not inferred from JSON.

Observed: 470463 requests, 424383 hits, 46080 misses (90.2054% count hit rate),
no bypasses/evictions. Under those assumptions, for this identical stream:

| Logical capacity | What can be concluded |
|---|---|
| 28 GiB | Working set exceeds cap by 1680867328 bytes; misses in [46080,470463]; exact hits/evictions unknown |
| 30 / 32 / 34 GiB | Entire observed logical set fits: same 424383 hits and 46080 compulsory misses, no evictions |

These are conditional replay deductions, **not new cap experiments**. The
synthetic sequences AABBCCDD and ABCDABCD have identical frequencies and
four-entry-cache aggregates but respectively four and zero hits with two entries.
Therefore no exact 28-GiB LRU curve can honestly be reconstructed from totals.
Mean unique slice bytes is not request-weighted bytes; count hit rate cannot
be converted to byte hit rate. Logical excess is not a disk-read lower bound.

## Proposed real bounded-cache contract (not implemented here)

1. Preserve immutable GGUF/NVMe source and manifest; index keys by model
   generation, tensor identity/layout and expert ID. File offsets must survive
   graph copies; raw runtime pointers must never be serialized as durable IDs.
2. Use a byte-bounded payload arena, initially 28 GiB, with separate bounded
   metadata and I/O staging. Variable-size or size-class slots must account for
   internal fragmentation; do not charge only the nominal quant payload.
3. Acquire a **lease for the full selected kernel working set** before compute;
   release only after all CPU readers / asynchronous device transfers finish.
   Never evict a leased/loading entry. If selected bytes exceed the arena,
   chunk the operation under an explicit correctness-tested contract or fail;
   never allocate an unaccounted bypass buffer.
4. Existing kernels assume base + expert-ID * stride. An arena needs a tested
   indirect expert-address table or a gather/chunk path; a policy-only LRU
   cannot redirect those reads. Keep this ABI change separate and CPU-test it.
5. Plain pread into an arena can duplicate model bytes in page cache. Candidate
   cold reads use aligned O_DIRECT staging, where supported, with explicit
   offset/EOF handling and bounded buffers; do not silently fall back to
   unbounded buffered I/O. No disk-wide cache dropping or pinned host bank.
6. Merely issuing MADV_DONTNEED on the source is not a hard expert-cache cap:
   file page cache and concurrent readers have separate lifetimes. Treat any
   mmap advice prototype as soft residency management, never as the arena.
7. 28–34 GiB is a design range, not an approved run footprint. Start at 28 GiB;
   charge arena + metadata + staging + KV/backbone/PLE + scratch + page cache
   against the separate 40-GiB cgroup. Measure MemoryPeak and memory.stat,
   real read bytes/latency and absolute VRAM <=28 GiB. Host guards still apply.
8. GPU LRU and bandwidth-adaptive placement follow only after CPU arena
   correctness and actual miss-cost evidence; do not load a new checkpoint.

## Follow-up gates, in order

- Keep the existing **model-shadow-pressure gate OPEN**: smaller cap, both
  arm orders, positive evictions, consistent counters, output identity and
  measured RAM/VRAM. Do not reuse the 32-GiB 90% hit threshold there.
- Add bounded, phase-tagged slice observation records (sequence, stable tensor
  key, expert ID, bytes, kernel/phase boundary, explicit truncation/drop count).
  Label it kernel-distinct order, not original token-routing order. An incomplete
  trace must fail closed for whole-run replay claims.
- Replay that trace with production LRU at 28/30/32/34 GiB; capture byte hits,
  reuse distances and per-kernel leased peak. This supplies arena sizing evidence.
- Then test arena leases/indirection on CPU fixtures and a sparse file before
  real model eviction/I/O tests. Synthetic evictions still prove no NVMe costs.

## X intake (completed before effective tests)

- https://x.com/JonathanLeaders/status/2095715911072207016 — x_search describes
  a 5090/64GB AD-3.84bpw MTP setup claiming 24 tok/s. Not a measured 40-GiB
  footprint and not verbatim verified in this run; no speed claim adopted.
- https://x.com/Brjen/status/2095872991502475358 — x_search describes IQ3,
  3090/64GB and 34 CPU MoE layers. Only a placement lead; no comparable budget
  proof. Search mixed EXL3/AD and model variants: those equivalences rejected.
- No browser opened, no external posts, no download. Keep bounded cold-expert
  residency rather than switching stacks based on incomparable measurements.

## Verification at initial review

- New Python suite: 7 methods pass; initial RED was a missing-module scaffold,
  not a demonstrated pre-existing behavioral regression.
- Full CPU build succeeds (`GGML_CUDA=OFF`), targeted CTest 7/7 passes.
- Broad main suite: 25/27 passes; same known BERT tokenizer and chat-template
  failures as recorded before this slice. No C++ runtime behavior changed.
- No new tok/s, run RAM peak or VRAM peak measured. Historical 8K-only result:
  54–56 tok/s, 33.83 GiB cgroup MemoryPeak, 26.0 GiB VRAM; >=20 tok/s at depth
  remains unproven. Review and final closeout are recorded in the cron state.

## Independent review and final verification

Review `deleg_88e270ff` (gpt-6-astra): GO for this CPU-only slice, one minor
validation gap. Parent verified its core claim against production LRU eviction
condition (`ggml-moe-cache-lru.cpp:51-64`) and reproduced three failing subcases:
evicted bytes smaller than evicted count; stable cold-start eviction even though
resident + evicted bytes did not exceed capacity (including equality).
Both invariants now reject bad input; the pressure fixture now has reachable
sizes (30,10,10,20; cap64; evict30, resident40). No second review pass.

Final Python suite: 11 methods pass, including captured-evidence regression.
Targeted CTest remains 7/7; broad main 25/27, same two known baseline failures.
Both JSON outputs were regenerated by the parent and cmp-identical. Parent
also checked last30 count/byte equality against the original stats with jq,
quant constants/structs, backwards placement selection, acquire reset and
kernel deduplication directly in the cited source. No new runtime cap or speed
claim follows from the reviewer self-report.
