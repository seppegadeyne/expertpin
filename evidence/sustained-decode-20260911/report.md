# Sustained decode on ps-iq2xxs — 2026-09-11 (11:20–11:29 CEST)

Question: does the PS IQ2_XXS checkpoint sustain >=20 tok/s decode over
longer generations (2x512 and 2x1024 tokens) within the 40/28 budgets?

## Verdict: SUSTAINED — every request >=37 tok/s, all budgets green

| depth | request 1 (colder) | request 2 (warmer) | acceptance | RAM peak | VRAM peak | swap peak |
|---|---|---|---|---|---|---|
| 2x512 | 37.77 tok/s (13.6 s) | 51.67 tok/s (9.9 s) | 0.678 | 32.41/40 GiB | 18.64/28 GiB | 1.19 GiB |
| 2x1024 | 41.62 tok/s (24.6 s) | 48.51 tok/s (21.1 s) | 0.647 | 32.41/40 GiB | 18.83/28 GiB | 0.55 GiB |

Config: clean-MTP harness copy with SUSTAINED_TOKENS env override (commit
4d1039b9 — one line, diff-verified against the committed base), ps-iq2xxs
checkpoint, nmax4/DRAFT1, seed 42, temp 0, ignore_eos, cache_prompt false,
prompt 68 tokens. Binary SHA 3235529f... unchanged. MemoryMax 36G;
memory.events max/oom/oom_kill all zero in both runs.

## Reading

- Sustained decode holds 37-52 tok/s through 1024 tokens: no degradation
  trend within these depths on a warm cache; the first request of each pair
  (colder) is slower as in the 256-token A/B.
- Acceptance drifts slightly with depth (0.678 @512 vs 0.647 @1024) —
  within the range seen before; not an acceptance collapse.
- These runs measured throughput only: output text was length-forced
  (ignore_eos), quality not assessed (covered separately by the quality
  gates).

## Limitations

- Single fixed prompt, two requests per depth; no prompt diversity.
- Warm cache from earlier same-day runs; cold-cache first-token behavior is
  the separate open item (controlled fadvise DONTNEED A/B).
- Same-day host state; cross-day comparisons remain invalid without
  host-state control.

## Artifacts

- outcome.json (machine-readable), runs run-20260911T112056 (512),
  run-20260911T112455 (1024). Harness: 4d1039b9 (pushed before the GPU
  runs; origin/main read back exactly).
