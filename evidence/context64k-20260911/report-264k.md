# 264K context on ps-iq2xxs — 2026-09-11 (13:46–14:10 CEST)

Question: does ps-iq2xxs serve 264K context (the trained 262,144-token
window; Seppe's desired end goal) with needle recall, within 40/28?

## Verdict: PASS — 3/3 recall at 254,864 tokenizer-exact processed tokens

| position | recall | wall |
|---|---|---|
| end (90%) | 7391 | 524 s |
| middle (50%) | 7391 | 325 s (partial prefix reuse) |
| start (10%) | 7391 | 520 s |

Budgets: RAM peak 32.41/40 GiB; VRAM peak 19.94/28 GiB; memory.events
max/oom/oom_kill all zero. Honest detail: cgroup swap peaked at 12.45 GiB
during the 250K prefill (MemoryHigh throttling with sock_throttled 159 —
reclaim pressure, not an OOM; swap sits outside the 40/28 physical caps and
is reported separately per standing convention).

## Reading

- The full trained context window is served WITH correct recall at the
  extremes; VRAM headroom remains ~8 GiB (KV q8_0 scaling as projected
  from the ladder).
- Prefill dominates wall-time (~5-9 min per fresh position at ~500 tok/s):
  interactive 250K use needs prompt caching or long-lived sessions — the
  dev-serving profile (issue #3) should keep the server resident.
- 64K (product acceptance) was already PASS; this extends the proof to the
  full window.

## Limitations

- Single needle per position, n=1, greedy decoding.
- No generation-quality or tool-calling behavior tested AT 250K context.
- "264K" interpreted as the trained 262,144-token window (256K KiB
  convention); tokenizer-exact 254,864 processed here.

## Artifacts

- outcome-264k.json, run-20260911T134603 (1,368 budget samples over the
  24-minute run). Ladder harness: 468455ca.
