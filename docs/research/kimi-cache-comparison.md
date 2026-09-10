# Kimi's streaming cache versus expertpin's advisor

Source audit, 2026-09-10. **Decision: reuse the measurement discipline and bounded
load/publish design, not the cache-size numbers or a new placement default.**
This fulfills Seppe's requested comparison; no engine, launcher or policy change.

## Scope and provenance

Read-only reference: [FareedKhan-dev/kimi-k3-in-c at
117e9d29bde14db9742f54fb66a191fd0bf03903](https://github.com/FareedKhan-dev/kimi-k3-in-c/tree/117e9d29bde14db9742f54fb66a191fd0bf03903),
local `third-party/kimi-k3-in-c`, clean at inspection. Below, `K:` means a path
inside that pinned tree; `E:` means expertpin at `3ad35b98` before this note.
No reference code executed, weights downloaded, contributions or external posts.
Reference benchmark statements are author reports, not independently reproduced.
X intake and local test artifacts: `evidence/cache-reference-20260910/`.

## What actually differs

| Topic | Kimi reference | expertpin today / transferable idea |
|---|---|---|
| Cache unit | Whole packed expert, pointers to three packed/scale pairs; fixed padded slots sized from a checkpoint probe (`K:src/cache/k3_cache.c:23–33,280–312`). | Variable tensor slices keyed by data/stride/expert, byte-bounded metadata LRU (`E:ggml/src/ggml-moe-cache-lru.h:8–35`, `.cpp:24–66`). Do not import a constant expert size or slot-count budget. |
| Physical storage | Allocated arena; direct load into a slot, publish only after exact-length success (`K:src/cache/k3_cache.c:182–217,315–332`). | Current shadow records accesses; it does not populate an arena or evict physical pages (`E:ggml/src/ggml-moe-prefetch.cpp:756–773`). Logical resident bytes are not RAM consumption or avoided storage reads. |
| Batch safety | Reserve serially with a distinct INFLIGHT state; read in parallel; publish serially (`K:src/cache/k3_cache.c:35–59,131–217`). | Adopt this state machine for a future payload arena, **plus leases across all live kernel operands**. Our LRU has no leases and can evict multiple small slices for one large slice. A reservation is not a lifetime guarantee for GPU work. |
| Sizing | `K:tools/sim_cache.py:36–110,147–177` replays LRU, equal-size offline Belady, and hindsight hot-set pinning. | `E:scripts/advise-expert-cache.py:64–114` ranks supplied byte-budget candidates by serial modeled cost; it does not generate a temporal cache curve. `analyze-expert-cache.py:45–98` deliberately keeps smaller-cap answers unknown from aggregates. These tools are complementary, not interchangeable. |
| Trace meaning | `cache_get` records layer/expert demand before serving it (`K:src/cache/k3_cache.c:235–260`). | Our CPU shadow deduplicates within each kernel and visits IDs in ascending order under its mutex (`E:ggml/src/ggml-moe-prefetch.cpp:746–773`). This is an ordered **CPU shadow stream**, not original per-token router order or a globally ordered CPU+GPU demand stream. |

The reference is useful, not a drop-in correctness oracle. Its trace buffer grows
with `realloc`; allocation failure can omit records without a completeness footer
(`K:src/cache/k3_cache.c:247–255`). Our joined traces already require a complete
zero-drop/error footer (`E:scripts/join-expert-trace.py:19–29,47–53`). Preserve that
fail-closed contract. Its simulator also uses decimal GB and an approximate
`n // 1472` token count (`K:tools/sim_cache.py:129–160`); expertpin must use exact
phase/request counters, byte prices, and GiB = 2^30 bytes.

## The two lessons that change our interpretation

**1. Demand served from a cache is not necessarily a retained hit.**
The reference reports apparent 100% arena hits after batch prefetch while reading
all expert bytes anew (`K:README.md:3074–3102`). The code supports the distinction:
`getmany` increments `prefetch_reads`/`bytes_read`; the subsequent `admit` counts
hits. Our shadow hits, residency hits (including prefetched pages), and physical
I/O are likewise separate populations. Do not subtract counters from different
windows. Do not adopt `1 - evictions/requests` as a general hit-rate formula:
cold fills, bypasses, size changes and variable-size multi-evictions invalidate it.

Our existing production-LRU fixture was rerun, not newly invented:

| Synthetic logical cap | Requests | Hits | Evictions |
|---|---:|---:|---:|
| 8 GiB | 147456 | 73728 | 65536 |
| 16 GiB | 147456 | 73728 | 57344 |
| 32 GiB | 147456 | 122880 | 0 |

All enabled arms agree within each cap, in both on/off/on and off/on/off order.
This proves that more capacity can leave hits flat while changing eviction
counts, and zero evictions need not mean all hits. It proves **no model hit rate,
physical eviction or NVMe cost**. Payloads are never allocated in the fixture
(`E:tests/test-expert-cache-pressure.cpp:9–15,30–90,95–105`).

**2. An attractive trace curve can describe the wrong execution.**
The reference explicitly says its sizing trace came from repeated full-prefix
recompute, not incremental decode, and later reports a different measured curve
(`K:README.md:2663–2668,3061–3066`). Its trunk/expert split results show why total
latency matters more than expert-hit rate (`K:README.md:3111–3168`), but do **not**
prove "trunk first" is optimal for Qwen on our mixed CPU/GPU setup.

Replay is conditional on an unchanged demand stream and numerical execution.
The Kimi simulator is also **demand-only**, not exact replay of its default
runtime: `K:src/core/k3_ops.c:600–615` prefetches the batch before demand gets;
`K:src/cache/k3_cache.c:143–163` skips resident hits without promoting recency
while reserving victims for misses. Runtime padded slot capacity (`:301–304`)
also differs from the simulator's payload-only division (`tools/sim_cache.py:152`).
Unchanged routing alone therefore does not prove policy-equivalent hit counts.
Even the reference exposes cache-only draft routing through `cache_resident`
(`K:src/cache/k3_cache.c:220–232`). Cache-independence must not be generalized to
all speculative modes. Our n_max-dependent greedy divergence already prevents
claiming identical workload across depths. The reference's "three invariants"
are Kimi architectural rules, not a universal floating-point contract
(`K:README.md:831–857`); its separate FP contract is in `K:src/core/k3_ops.c:1–20`.
Neither resolves our batch/recurrent-state diagnosis.

## Concrete adoption sequence, without changing today's priority

1. **Correctness first:** identical-prefix replay around the known divergent
   positions with controlled batch/checkpoint state remains the next runtime
   diagnostic. No depth/default/placement change on the strength of this note.
2. For cache sizing, replay a complete **single-backend** ordered trace through
   the production byte-LRU. **A zero-drop CSV footer is not complete cache
   history.** The shadow mutates before the request-only filter
   (`E:ggml/src/ggml-moe-prefetch.cpp:769–772`, `ggml-moe-trace.h:31`), so unscoped
   warmup can populate it without a CSV record (`docs/request-scoped-expert-trace.md:16–24`).
   Concrete existing witness: `evidence/expert-offset-trace/run-20260906T131516/joined.csv:2–5`
   starts at weight_entry 216 and reports a hit for expert 20 on its first recorded
   appearance (seq 3), despite the clean footer at line 45017. This scoped artifact
   is **not directly replay-ready from an empty cache**. Require all shadow
   mutations from a known reset boundary, including warmup/reset events, or
   justified capacity-specific initial-state assumptions; a source-cap snapshot
   alone cannot establish the correct warm state for smaller capacities.
   Keep prefill in the replay to warm state, report prefill/decode separately,
   and label an intentionally empty decode-only replay as a different experiment.
   Preserve request, epoch, kernel-entry and model identity; reproducing recorded
   source-cap shadow decisions is a mandatory gate. Preserve repetitions: the
   bandwidth probe's deduplicated sample is not a replay.
3. Report count hits, hit/miss **bytes**, compulsory misses, re-reference misses,
   peak simultaneous leased payload and metadata/staging reserves. Test unequal
   sizes, oversized bypasses, partial tails, truncated traces, duplicate keys,
   read failures and live-lease eviction attempts. Do not call ordinary
   equal-size Belady an optimal bound for variable-size slices; train any pinned
   hot set on separate prompts, not the evaluated trace.
4. Only then evaluate one bounded physical arena: reserve → load → publish,
   no readable failed/inflight slots, rollback on short reads, leases until the
   last CPU/GPU consumer finishes. Price packed quant slices directly; avoid a
   full-model dequant allocation. Direct-I/O alignment/extra read bytes and
   allocations need their own validation; no direct-I/O/hugepage flag is adopted.
5. Compare candidate end-to-end cost at **fixed total memory**, including
   backbone/PLE, KV, staging, page cache and swap separately. The advisor still
   needs matched traffic/compute/reserves; use null for unknowns. Cgroup charges
   alone do not account for all pre-existing shared model page cache. Never
   sum mapped RSS blindly (shared pages can double-count). Require a defensible
   whole-model physical-accounting method before accepting <=40 GiB RAM, plus
   <=28 GiB device VRAM and >=20 decode tok/s under the normal GPU guards.

## Execution and acceptance status

Reproduced with the existing CPU build, no Qwen/Kimi model payload load or GPU
run. Tests with GPU/CUDA in their names in this CPU build exercise host-side
trace code; they do not execute CUDA inference.

```sh
cmake --build build-cpu-shadow -j 8
ctest --test-dir build-cpu-shadow -R 'test-(expert|moe-prefetch|cache-shadow|trace-bandwidth)' --output-on-failure
ctest --test-dir build-cpu-shadow -L main --output-on-failure
build-cpu-shadow/bin/test-expert-cache-pressure
python3 scripts/advise-expert-cache.py evidence/pageable-staging/local-incomplete.json
```

Build succeeded; focused CTest **12/12**, main **39/41** (the existing BERT-tokenizer
and chat-template failures). Advisor returned **exit 2, blocked, recommendation
null** on the existing incomplete physical profile; no rates were filled from
the reference. Raw logs/JSONL and advisor JSON accompany this note.
No new tok/s, model-run RAM or VRAM peak. No long-context acceptance test: the
64K minimum and literal 264K goal, exact token budgeting, API streaming/tool-call
contracts and Hermes end-to-end integration remain unvalidated by this slice.
Configured context or historical short-prompt throughput is not proof of them.
