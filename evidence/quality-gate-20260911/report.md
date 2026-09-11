# Tool-call JSON quality gate A/B — 2026-09-11 (10:00–10:08 CEST)

Question: does the PS IQ2_XXS checkpoint (75.2 GiB) still produce schema-exact
tool calls and a clean tool-result answer within the 40/28 budgets, compared to
the reference AD-4.27bpw Q4_K_M checkpoint (197 GiB)? This is state-file "next
step 2" (IQ2_XXS quality gate) — a precondition for any default-switch.

## Verdict: PASS on both checkpoints

| checkpoint | round 1 tool call | round 2 answer | RAM peak | VRAM peak |
|---|---|---|---|---|
| ps-iq2xxs | get_weather {city: Ghent, unit: celsius}, id present, 5.1 s | "The current weather in Ghent is **17.5°C** and cloudy." (2.1 s) | 32.41 / 40 GiB | 16.18 / 28 GiB |
| reference | get_weather {city: Ghent, unit: celsius}, id present, 13.2 s | "The current weather in Ghent is cloudy, with a temperature of **17.5°C**." (5.1 s) | 32.41 / 40 GiB | 20.09 / 28 GiB |

Both rounds used the identical scenario: forced single tool binding (enum
schema), tool_choice string "required", temp 0, seed 42, target-only (DRAFT=0)
so checkpoint quantization is the only model variable. Swap peaked at 0.61 /
0.91 GiB. Same binary SHA 3235529f... as the 03:00 checkpoint A/B; only
MODEL/MODEL_DIR differ between arms (environment.json diff proves one
variable).

## What this proves

- The PS IQ2_XXS checkpoint emits structurally correct OpenAI-style tool_calls
  JSON (exact name, both schema keys with enum-valid values, usable id) and
  answers cleanly after the tool round, reflecting the numeric result.
- Within budgets on this host, on the first post-load requests (warm files).

## What this does NOT prove

- n=1 scenario, greedy decoding: no multi-tool selection, parallel calls,
  streaming tool deltas, long trajectories, or sampling robustness.
- No needle-recall / long-context quality (separate slice), no 64K/264K.
- No Hermes end-to-end run against this endpoint.
- Wall-times are single-shot observations on a warm host, not a throughput
  benchmark (see the 03:00 A/B for throughput).
- The reference arm needed one retry: first attempt aborted on EADDRINUSE
  (port 8102 TIME-WAIT from the ps arm) before any host change; no model load
  happened in that attempt.

## Artifacts

- Runs: run-20260911T100044 (ps-iq2xxs), run-20260911T100516 (reference);
  the EADDRINUSE refusal is preserved in run-20260911T100231.
- outcome.json (machine-readable verdict + budgets), review.md (429-blocked
  review, parent self-verification), research.md (search-tool leads only).
- Code: fda74ed1 (harness + 14 CPU-only tests, pushed before the GPU run;
  origin/main read back exactly).

## Next steps

1. Needle-recall slice (second half of the quality gate) — begin/middle/end
   retrieval at growing context before any default-switch.
2. Sustained decode depth on ps-iq2xxs (2×512/1024) within 40/28.
3. Controlled cold-cache A/B (fadvise DONTNEED) for worst-case first tokens.
4. 64K context on ps-iq2xxs within 40/28.
