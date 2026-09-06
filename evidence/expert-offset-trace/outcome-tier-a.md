# Tier A GPU attempt — 2026-09-06 12:24–12:25 CEST

Code: 053dbf2fad1af74c935668ed4275652468cd2462. Both guards passed:
MemAvailable 41.85551452636719 GiB, GPU 0%; launcher DRY guards OK with
RAM_BUDGET_GIB=36, MemoryHigh=33177 MiB. No guard override or OOM retry.

## Actual result: initialization failed, no decode benchmark

The target and MTP weights were loaded and target context created. Server never
became healthy; no completion request was sent. `model_loaded=false` in the raw
harness means *health not reached*, NOT that no model weights were loaded.

`server.log:614-618` explicitly reports the draft trying to acquire an already
owned expert-cache shadow, followed by draft-context and server initialization
failure. Parent source verification:

- common/speculative.cpp:2004 copies target params, :2025-2032 only clears MTP
  offload settings, :2057 converts the still-inherited cache shadow setting.
- common/common.cpp:4431 propagates expert_cache_sim_mib to context bytes.
- common/speculative.cpp:1335,1381-1384 creates the draft with those parameters.
- src/llama.cpp:9253-9257 rejects a second shadow owner.

This is a discovered real MTP+shadow integration blocker, not evidence that
normal MTP without shadow fails. Fix draft telemetry inheritance independently
while preserving target ownership; add regression and repeat capture next.

## Memory and trace evidence

28 periodic samples. Highest observed cgroup memory.peak: **32.40488052368164
GiB** (<40); MemoryMax consistently **36 GiB**. Device-wide sampled VRAM peak:
**21.576171875 GiB** (<28). Highest sampled swap.current: 0.8272514343261719 GiB;
combined RAM+swap.current sample maximum: 33.22274398803711 GiB. Final systemd
journal reports rounded 32.4G RAM and 899.3M swap peaks over 55.802 s. Sampled
high-water is not an exact final cgroup readback after scope removal; GPU
sampling does not rule out brief unobserved peaks. Last sampled events:
MemoryHigh=215756, max=0, oom=0, oom_kill=0. Pressure/reclaim occurred, no sampled
OOM event and no OOM termination reported in the exact scope journal.

trace.csv has **57132** sequential records, footer
`# end written=57132 dropped=0 error=0`. Parsed row count, sequence, relative
stride offsets and shadow hit totals match expert-stats.json. These are
**initialization-only observations**, not workload routing or prefill/decode.
No phase tags or physical GGUF offsets. The all-expert startup observations must
not feed the advisor as measured decode traffic. No new tok/s result and no
new conclusion about >=20 tok/s on depth. Physical phase 2 remains open.

## Shutdown

Qli stop requested 12:24:15.760355, restart 12:25:15.682613 (59.922258 s interval).
The server had already exited and its scope disappeared. Raw harness status is
`cleanup_failed` because stopping an unloaded scope returned rc5; preserve this
rather than silently rewriting success. Final exact-scope readback verified
inactive; qli and headless Chromium both active, journal `[qli-tune] OK`.
Parent independently rechecked both services and absence of llama-server.

Tests: harness six methods green, targeted CTest12/12; main30/32 with historical
BERT/tokenizer and chat-template failures. CPU and CUDA server builds green.
See review-tier-a.md for initial independent NO-GO and verified remediations;
no second review GO claimed. X sources in research-20260906-tier-a.md.
