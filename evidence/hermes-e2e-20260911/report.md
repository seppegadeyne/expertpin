# Hermes E2E contract gate — 2026-09-11 (12:50 CEST)

Question: does the local ps-iq2xxs endpoint satisfy the OpenAI API contract
Hermes depends on (mandate 2026-09-09 items 5-6)?

## Verdict: PASS — all 5 contract steps green

| step | result | wall |
|---|---|---|
| 01-models | GET /v1/models → 1 model id (the GGUF path, llama.cpp convention) | <0.1 s |
| 02-plain | non-streaming completion, finish_reason stop | 2.0 s |
| 03-stream-plain | SSE chunks assemble, single terminal finish stop | 0.9 s |
| 04-stream-tool | streamed forced tool call: 10 indexed deltas, finish tool_calls | 2.7 s |
| 05-tool-round | forced call → tool message with emitted call id → clean stop answer | 3.8 s |

Budgets: RAM peak 32.41/40 GiB, VRAM 15.94/28 GiB, memory.events clean.
Target-only, temp 0, seed 42, binary SHA 3235529f... unchanged.

## Notes for the Hermes integration (issue #3)

- The model id is the GGUF file path — Hermes provider config must use that
  exact string as the model name.
- tool_choice must use the string form ("required"/"auto"/"none"); the
  OpenAI object form silently degrades to "auto" in this server build
  (json_value type_error fallback, documented in the tool-call gate review).
- Streamed tool calls arrive as multiple indexed deltas (10 in this run)
  that assemble name and arguments incrementally — a client must aggregate
  by index.

## Limitations

- Single scenario per step, greedy decoding.
- No error-path contract tests yet (context overflow, invalid schema —
  mandate item 5 asks for predictable error handling too): separate slice.
- This validates the API surface; a full Hermes-agent run executing real
  tools against this endpoint is the final E2E step (needs the dev-serving
  profile from issue #3).

## Artifacts

- outcome.json, run-20260911T125037 (incl. saved SSE frames, requests and
  responses for every step, and the tool-round trip with call-id linkage).
  Harness: e84ddcd5 (6 CPU-only tests; suites 23/23, targeted 4/4).
