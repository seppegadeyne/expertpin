# Hermes agent E2E — 2026-09-11 (15:24 CEST)

Question: can a REAL Hermes agent run end-to-end (reasoning -> tool call ->
tool result -> correct final answer) against the guarded local ps-iq2xxs
endpoint?

## Verdict: PASS — the full agent loop runs on the local checkpoint

Session 20260911_152451_5f0772 (4 messages, 2 tool calls, 68.6 s wall):

1. Reasoning: the model decides to use the terminal tool for the printf task.
2. Tool call: `printf "hermes-e2e-ok"` executed (0.1 s).
3. Reasoning: processes the tool result.
4. Final answer: exactly `hermes-e2e-ok`.

Budgets during the whole session: RAM peak 32.41/40 GiB, VRAM 19.22/28 GiB,
no OOM events. Server at CTX=65536 (Hermes requires >=64K reported).

## Integration recipe (the issue #3 dev-serving core, now proven)

```yaml
# isolated HERMES_HOME config.yaml
model:
  provider: custom
  name: /home/seppe/Models/qwen3.8-flash-next-ps-iq2xxs/Qwen3.8-Flash-Next-IQ2_XXS.gguf
  base_url: http://127.0.0.1:8102/v1
  api_key: local-no-key-required   # server ignores auth; placeholder satisfies the client
  context_length: 262144           # server reports its CTX; Hermes demands >=64K
```

Gotchas found (each cost one iteration, all documented in e2e-outcome.json):
- The model_aliases direct-alias path does NOT feed the default-provider
  startup; configure the classic custom provider (model.provider/base_url)
  instead.
- Hermes hard-refuses servers reporting <64K context — serve CTX>=65536.
- The marker check must exclude the echoed query (false-positive guard).

## Limitations

- Single task/session; no multi-turn loops, error recovery, or long
  trajectories yet.
- The isolated home is a minimal profile (no skills/memory): this proves
  provider + tool-calling + result processing, not the full skill stack.
- Default agent sampling; ~69 s for a 2-turn tool task incl. client startup.

## Artifacts

- e2e-outcome.json; run-20260911T152411-e2e (hermes stdout/stderr/exit,
  server log, samples). Runner commits: 023a8259, 6b502480, ce1849e1,
  3af940c7. Failed iterations preserved (144126, 145440, 150049, 150707,
  151641).
