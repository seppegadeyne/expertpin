# Whole-checkpoint cold-cache verification — 2026-09-14

## Defect and bounded fix

The existing cold-cache harness evicted and measured only `env['MODEL']`.
For UD-Q4_K_XL that is shard 1 of 4, so its old cold label did not establish
whole-checkpoint coldness. Historical single-file IQ2_XXS observations are not
invalidated by this finding. Historical split-model runs are not re-labelled.

The helper now enumerates the conventional split, validates every nonempty
regular file before eviction, drops every shard before sampling any of them,
and retains per-file observations plus aggregate totals. Every shard must
individually meet the cold threshold (default <=10% resident pages); a small
warm shard cannot hide in a low aggregate ratio. UNKNOWN/NOT_COLD blocks launch.
Missing shards are refused before host preparation. The legacy `file` field
identifies the entrypoint; `files` and `scope=whole_checkpoint` define the totals.

This is filename validation, NOT a GGUF metadata/integrity validator. Mincore
snapshots are sequential, not atomic; concurrent readers can repopulate files.
The helper covers the target checkpoint, not its separate drafter.
No inference kernels, quantization, placement or serving defaults changed.

## Self-verification (not an independent approval)

The higher-priority profile's Astra pause through approximately September 15
prevented delegate review. The pinned reviewer was not called or reconfigured.

Premises checked directly:
- `src/llama.cpp:13414-13417`: loader split names are `%s-%05d-of-%05d.gguf`.
- `src/llama-model-loader.cpp:361-389`: metadata controls split count, first
  shard is required, and remaining paths use the same naming convention.
- `evidence/coldcache-ab-20260911/run-guarded.py`, `model_files`,
  `verified_cold_report`, and `Run.execute`: complete file validation before
  `prepared=True`; report written and rejected before launcher invocation.
- `tests/test-coldcache-shards-20260914.py`: real local temporary files plus
  mocked residency distinguish an unseen warm final shard from a cold first
  shard, prove all-drops-before-all-snapshots, late-drop error propagation,
  per-shard thresholding, and fail-closed incomplete/error cases.
- `tests/test-coldcache-ab-20260911.py`: execute wiring refuses missing shards
  before host changes, and NOT_COLD/UNKNOWN before dry launch or process spawn.

Initial RED: 7 methods, 4 behavior failures and 3 missing-API error reports
(including subtests). GREEN: new shard suite 10/10, cold-cache suite 11/11,
existing residency suite 5/5. CPU configure/build succeeded with CUDA/Vulkan
OFF. Targeted CTest 8/8 including expert-manifest and E2E regressions. Full
CTest: 3 failures out of 57, the known bert-bge, chat-template and eval-callback
failures (last requires unavailable stories260K.gguf/libcurl). No all-green
full-suite claim. `git diff --check` clean.

## Physical validation plan

Use the existing harness with UD-Q4_K_XL, NCMOE=40, MTP n_max=4, 2x256 tokens,
CTX=8192, reasoning auto. Strict zero-residency checks on ALL target shards
and on the separate drafter. This is a throughput probe, not a Hermes code gate.
A local operational wrapper places harness/curl/diagnostics in a 4G scope;
the server remains in its existing 36G scope. Both kernel peaks are recorded
and their sum checked against 40 GiB; device VRAM is sampled against 28 GiB.
The wrapper refuses raw execution outside its client scope (actually tested).
The outer shell has a bounded timeout and independent service restoration.

No new GPU throughput, RAM or VRAM measurements are claimed in this pre-run note.
X research completed before testing; relevant unverified leads:
https://x.com/Hermesf42w/status/2099174134475366553 (PLE Q3/SSD, 5070/64GB),
https://x.com/m_sigepon/status/2099204634925834409 (5090/128GB).
Neither establishes performance under our 40 GiB RAM budget.
