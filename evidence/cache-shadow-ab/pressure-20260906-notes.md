# Cache-pressure CPU experiment — 2026-09-06

X-research gate completed before tests; no browser was opened.
- https://x.com/lefu777/status/2096059297046437921 — x_search reports pure-LRU/cache changes; hardware and budget evidence absent. Treat as a lead, not a verified local benchmark.
- https://x.com/Oluwaphilemon1/status/2096411031664787833 — x_search reports long-context slowdown on two 3090s; not equivalent to our single-GPU budget.

Scope: execute the production LRU with synthetic logical expert slices at 8/16/32 GiB caps, and replay enabled/disabled/enabled plus the inverse. No model access, payload allocation, GPU, real mmap eviction or NVMe latency measurement. Aggregate historical stats cannot reconstruct an exact routing trace. This is a policy regression experiment, not a model-performance benchmark.

## Reproduction and actual CPU results

```sh
cmake -S . -B build-cpu-shadow -DGGML_CUDA=OFF
cmake --build build-cpu-shadow -j 4
build-cpu-shadow/bin/test-expert-cache-pressure > evidence/cache-shadow-ab/pressure-20260906.jsonl
ctest --test-dir build-cpu-shadow -R "test-(expert|moe-prefetch|cache-shadow)" --output-on-failure
```

The production ggml-moe-cache-lru.cpp is compiled directly into a CPU-only test (no llama/common/CUDA linkage). Three sweeps visit 24,576 logical 1-MiB slices, each twice immediately. Each arm has a fresh cache. Both on/off/on and off/on/off are executed at each cap; the disabled arm skips the observer, not a zero-capacity cache. All counters must match across repeats. This is not a runtime-toggle test or a correction for model/page-cache timing confounds.

| Logical cap | Observed requests per enabled arm | Hits | Misses | Simulated evictions | Misses on later sweeps |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 GiB | 147456 | 73728 | 73728 | 65536 | 49152 |
| 16 GiB | 147456 | 73728 | 73728 | 57344 | 49152 |
| 32 GiB | 147456 | 122880 | 24576 | 0 | 0 |

The 18 machine-readable records are in pressure-20260906.jsonl. Actual NVMe bytes, reload time and tok/s are null. Logical caps and eviction bytes are NOT physical RAM allocations or I/O. The oracle checks every hit, capacity and byte conservation per access, then exact final totals. There is no 90% hit-rate gate for this deliberately pressure-inducing fixture.

Full CPU build succeeded; targeted CTest 6/6 passed. Broad main suite: 24/26 passed; test-tokenizer-0-bert-bge and test-chat-template reproduce the two previously recorded failures. No production inference code changed. Initial missing-target build failed before implementation; this was scaffolding verification, not a behavioral RED test.

## Independent review and own verification

gpt-6-astra review (deleg_9defdc64) confirmed the policy/oracle and claim boundaries, but withheld unconditional approval because unconditional GiB fixture registration overflowed size_t on 32-bit targets. Own jq calculation confirmed each cap product modulo 2^32 is zero; tests/test-expert-cache-pressure.cpp:31 is the multiplication, and ggml/src/ggml-moe-cache-lru.cpp:45-48 bypasses zero capacity. Fixed by gating this target on CMAKE_SIZEOF_VOID_P >= 8 (tests/CMakeLists.txt:243-251); reconfigured, rebuilt and reran tests. No actual 32-bit build was performed, and there was no second reviewer pass.

Own readback of test-expert-cache-pressure.cpp:39-54,60-90 confirms disabled skipping, per-access assertions, exact counts and null NVMe/tok_s fields. Existing test-expert-cache-lru covers hit promotion and victim identity; the new cyclic fixture alone would not distinguish LRU from FIFO. Fresh-arm replay tests determinism, not real warm-cache behavior.

## Remaining step 0

CPU policy pressure is proven; real model shadow pressure is NOT. Next implement a separate pressure-experiment gate (positive evictions and coherent counters, not the 32-GiB 90% qualification gate) and wire guarded runtime on/off/on plus inverse at 8192/16384 MiB. Historical aggregate stats cannot be replayed as an exact routing trace. Before real mmap eviction, capture/measure actual page residency and reload costs; a simulated eviction is not evidence of an NVMe miss. No GPU run in this slice: the step-1 baseline already exists. No change to model tok/s, RAM or VRAM peak has been established.
