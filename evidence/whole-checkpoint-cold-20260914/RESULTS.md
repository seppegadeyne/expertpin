# Physical UD validation — 2026-09-14

One run completed at 03:12–03:14 CEST on the existing CUDA binary. This validates
the shard-aware cold-cache path, not an inference speedup from the Python fix.

| Observation | Result |
|---|---:|
| First decode, from verified-cold files | 13.020092 tok/s |
| Second decode, same server | 27.942684 tok/s |
| Goal | >=20 tok/s: first FAIL, second PASS |
| First / second prefill | 6.066113 / 25.014595 tok/s |
| Server cgroup kernel peak | 32.404377 GiB |
| Client cgroup kernel peak | 0.055859 GiB |
| Sum of kernel peaks (conservative upper bound) | 32.460236 / 40 GiB |
| Device VRAM sampled peak | 23.156250 / 28 GiB |
| Server swap sampled peak, reported separately | 2.015995 GiB |
| Samples / largest adjacent gap | 78 / 2.058962 s |
| Run wall time including cleanup | 110.871188 s |

## Method and checks

UD-Q4_K_XL, NCMOE=40, CTX=8192, q8_0 KV, MTP n_max=4, reasoning auto,
seed42/temp0/cache_prompt=false, one fixed 68-token prompt, 2x256 generated
tokens. `run/environment.json` is the effective launch configuration, and the
scope journal independently records the actual model path. The launcher still
advertises its historical IQ2_XXS API alias: that is NOT checkpoint provenance.

Before load, whole-target mincore returned **0/27,181,314** pages across all four
shards, plus **0/465,614** pages for the separate drafter. The first target shard
is only 10,946,624 bytes of 111,334,654,784 bytes: measuring only it missed the
weight-bearing remainder. This physical run used a stricter zero-page threshold
than the helper's default 10%. Decompressed `run/server.log.gz:7,76` confirms three additional shards
and split.count=4, matching the filename-derived set.

Tier A was met: 43.318867 GiB MemAvailable, GPU utilization 0%, DRY guards OK,
GGML_CUDA_NO_PINNED=1. Harness/curl/diagnostic subprocesses ran inside a dedicated
4G scope (swap disabled), while the server used the existing separate 36G scope.
Both caps were checked during the run. `inclusive-ram.jsonl` records both kernel
peaks; client-final.json retains its later peak through cleanup. Sum of these
peaks upper-bounds simultaneously charged RAM rather than pretending they
occurred at the same instant. Server memory.events max/oom/kill remained zero;
client final events were also zero. Server journal independently reports rounded
32.4G RAM / 2G swap peaks. VRAM monitoring remains sampled, NOT continuous.

Jetski was stopped by the night-run harness and restored active. Chromium restore
completed. qli was never mutated: actual state inactive/not-found, so masking was
not independently verified. Both owned scopes were inactive/not-found afterward;
no llama-server or listener on port8102 remained. Redundant outer cleanup printed
'unit not loaded' for scopes the harness had already removed; cleanup_errors=[]
and the independent post-run checks confirmed teardown.

## Evidence and limits

`outcome.json` was calculated from raw responses and both sample streams using
the archived `summarize.py`; assertions passed for complete responses, exact
response/summary timing equality, cold per-file counts, caps and cleanup.
`artifact-manifest.json` hashes the archived bytes. Existing CUDA binary SHA256
was measured after the run; its embedded build commit is 60a14628, NOT the
harness source commit 5dfb2555. No CUDA rebuild or model hash recomputation.
The archived dated operational wrappers retain the exact local paths they used;
run-gpu.sh refuses reusing their existing run artifacts. They are evidence of
this single execution, not a new general E2E harness.

- One prompt, two short requests: no statistically established warm regression
  versus historical 35.44 tok/s, nor an isolated speedup/host-state comparison.
- Cold files were measured sequentially, not atomically. No continuous exclusion
  of unrelated readers; shared/outside-cgroup page charges remain an attribution
  caveat even though all measured files started with zero resident pages.
- Sampled VRAM does not prove the continuous peak stayed below 28 GiB.
- This reasoning-auto throughput probe is NOT the reasoning-off Hermes code gate.
  No new long-context, API-contract, toolcalling or Hermes multi-turn validation.
- Full CPU CTest remains 54/57 with the three documented historical failures;
  targeted 8/8, shard tests10/10, cold-cache11/11 and residency5/5 are green.
- No independent subagent approval: Astra was not called under the higher-priority
  profile pause. Self-review and actual execution are the available evidence.

Next: use the client-scope accounting approach for the pending bounded UD
code/multi-turn rerun; enforce <=1800s total and fix its misleading summary
checkpoint/task/status metadata before treating those reports as authoritative.
Cold-start decode still misses the 20 tok/s goal. No default/quant switch.
