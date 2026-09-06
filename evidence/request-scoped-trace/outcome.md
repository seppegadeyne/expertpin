# Scoped request trace — measured 2026-09-06 13:15–13:16 CEST

Code: `d26a2b23ee94335f2d8710a943a8ea2c62802a31`.
Raw run: `evidence/expert-offset-trace/run-20260906T131516/`.
Validation/calculations: `outcome.json` in this directory.

## Actual request and budgets

- Server target + MTP ready; chat request with **68 input tokens, 32 output
  reasoning tokens**, length stop, nonempty text.
- Decode **16.268900522282546 tok/s** = 32000 / 1966.943 ms. Goal 20 NOT met;
  gap 3.7310994777174535 tok/s. MTP accepted **21/22** (0.9545454545454546).
- Highest read cgroup `memory.peak`: **32.40419006347656 GiB < 40**;
  `MemoryMax=36 GiB`, not a claim of 40-GiB allocated RAM. Final scope journal
  agrees at rounded 32.4G. Highest sampled swap.current 1.4808616638183594 GiB;
  RAM+swap.current sample maximum 33.87876892089844 GiB.
- Device-wide VRAM sample maximum **22.65234375 GiB < 28**, 43 samples.
  This is not a continuous VRAM-peak guarantee. Last memory.events max/oom/
  oom_kill all zero; high pressure events occurred, no guard bypass/OOM.
- Preflight MemAvailable **44.332725524902344 GiB**, GPU 0%, Tier A and DRY
  headroom green; DRAFT1/nmax4, NCMOE36, no pinned CUDA, CTX8192 capacity.
- Extra synchronization is diagnostic. No matched AB speedup claim from the
  earlier 12.23 tok/s probe. This is NOT 8K input, quality or deep-context proof.

## Trace and join validation

- **45015** logical CPU target expert observations, **20940 prefill + 24075
  decode**, internal request **27**, sequence **0**, footer drop0/error0.
- Prefill bounds 0–62 and 63–67; decode target-verification bounds 68–98.
  Multi-token speculative verification is correctly decode, not inferred
  prefill from ids_rows. Revisited positions can reflect rejection/replay.
- All rows joined to absolute GGUF offsets using fresh header validation of
  all shards against the inventory, production quant layout/stride checks,
  expert range/relative offset/slice byte/file bounds and complete CSV footer.
- Six independent spot-checks (first/last row per type 20/21/22) use the existing
  raw header reader separately; detailed data_start/tensor/expert offsets in
  outcome.json. All match. Header prefixes only, no full checkpoint load.
- This is **CPU-only target observation coverage**, NOT GPU or draft routing,
  physical I/O, runtime/file payload identity or a complete per-token trace.
  Unscoped init/draft/multisequence calls are excluded; cap is per-file.

## Tests/review/restore

Full CPU build + CUDA server build green. Targeted CTest **13/13**, including
manifest and trace; standalone join unittest **26/26** independently executed.
Main **31/33**, unchanged BERT tokenizer/chat-template failures. No TDD RED claim.
Independent review deleg_a96f15c6 PASS, no findings; parent verified key scope
and header-layout premises (review.md). All work/review/codepush before GPU stop.

Qli stop requested **13:15:18.446093**, restart **13:16:33.214336** CEST, interval
74.768243s; exact scope stopped/reaped. Qli+headless Chromium **active**,
`[qli-tune] OK`, cleanup_errors empty. Parent independently checked active states
and absence of llama-server (pgrep exit1 means absent, not a failed restore).

Next: GPU-side routed observations, matched cold/warm physical BW on these
joined offsets, GPU expert-GEMM and advisor. Model-shadow-pressure AB gate OPEN.
X leads and uncertainty in research.md. No upstream sync or external posts.
