Progress update 2026-09-11 (afternoon slices) — 64K context PASS (9/9 recall) and Hermes E2E API contract PASS (5/5 steps), both within 40/28.

Per Seppe's extended GPU-duration mandate (~12:15: "use what you need"), the last two open items of the quality roadmap were completed today.

## 1. 64K context (evidence/context64k-20260911/)

Harness 468455ca: needle-gate lineage with SERVER_CTX / NEEDLE_TARGET_TOKENS / budget env overrides; stepped ladder on one host day, ps-iq2xxs, target-only, token counts tokenizer-exact:

| CTX | processed tokens | end / middle / start | RAM peak | VRAM peak |
|---|---|---|---|---|
| 16384 | 15,269 | 7391 / 7391 / 7391 | 32.42/40 GiB | 16.14/28 GiB |
| 32768 | 30,920 | 7391 / 7391 / 7391 | 32.41/40 GiB | 16.57/28 GiB |
| 65536 | 61,401 | 7391 / 7391 / 7391 | 32.41/40 GiB | 17.14/28 GiB |

9/9 recall across the ladder; memory.events max/oom/oom_kill zero at every step. Seppe's 64K product-acceptance minimum for Hermes context is met with ~11 GiB VRAM and ~7.6 GiB RAM headroom at the 64K step. Not tested: 264K (separate goal; KV scaling observed here is ~0.5 GiB VRAM per 16K tokens), multi-needle, generation quality at depth.

## 2. Hermes E2E OpenAI-API contract (evidence/hermes-e2e-20260911/)

Harness e84ddcd5 (--gate contract, 6 CPU-only tests; suites 23/23, targeted ctest 4/4): mandate items 5-6 contract tests against the live ps-iq2xxs endpoint — all PASS:

- GET /v1/models discovery (model id = GGUF path; Hermes config must use that exact string)
- non-streaming completion, finish_reason stop (2.0 s)
- SSE streaming plain, coherent chunks, single terminal finish stop (0.9 s)
- SSE streamed forced tool call: 10 indexed deltas, finish tool_calls (2.7 s)
- full tool round-trip: forced call -> tool message referencing the emitted call id -> clean stop answer (3.8 s)

Integration notes: tool_choice must use string forms (the OpenAI object form silently degrades to "auto" in this server build); streamed tool calls must be aggregated by delta index. Open next: error-path contract tests (context overflow, invalid schema) and the full Hermes-agent run (needs the issue #3 dev-serving profile).

Commits: 468455ca, ff77d267, 5f686ddd (64K) + e84ddcd5, 4f057fc1, 99118e11 (contract) — all pushed and read back exactly. Budgets on every run: RAM <=32.42/40, VRAM <=17.14/28, no OOM events.
