# Review 2026-09-11 — Hermes E2E contract gate

Review attempt deleg_39849311 (12:50): FAILED — HTTP 429 usage limit after
3 retries, 9.7s. Fifth 429-blocked review attempt today (after deleg_1901d667,
deleg_e1f13ec2, deleg_a025517a, deleg_251bbce0); the review pin stays
unchanged — this is an upstream usage-limit condition, not a repo issue.

Parent self-verification instead (premises checked in the server sources):

1. SSE streaming endpoint exists: examples/server/server.cpp:1287 emits the
   `data: [DONE]` sentinel and :1301 sets `text/event-stream` with a chunked
   content provider — the contract_stream parser matches that exact framing
   (`data: ` prefix lines + [DONE]).
2. Streamed tool-call deltas with index/id/name are emitted by
   examples/server/server-context.cpp:776-820 (tool_call_delta.id/name on
   header deltas, per-index deltas afterwards); validators require an int
   index on every tool delta and aggregate by it.
3. finish_reason semantics: 'tool_calls' when parsed calls exist, else
   'stop'/'length' (examples/server/server-task.cpp:374, 416).
4. tool_choice must be the string form (common/chat.cpp:253-263; the object
   form silently degrades to 'auto' via json_value type_error fallback —
   documented since the tool-call gate); the contract payloads use strings.
5. The physical run validated every premise live: 10 indexed tool deltas
   observed with finish tool_calls; tool-round call-id linkage verified by
   contract_check_tool_round; models id equals the GGUF path.
6. Self-adversarial checks before the run: double finish_reason rejection,
   non-assistant delta-role rejection, missing-index tool-delta rejection,
   empty-stream rejection — all covered by the 6 CPU tests (6/6 green,
   inherited suites 23/23, targeted ctest 4/4).

Known open items (deliberately out of scope, listed in outcome.json):
error-path contract tests (context overflow, invalid schema) and the full
Hermes-agent run against the endpoint.
