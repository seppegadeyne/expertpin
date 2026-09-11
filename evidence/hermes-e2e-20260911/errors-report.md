# Error-path contract — 2026-09-11 (13:27–13:35 CEST)

Question: does the endpoint fail predictably on malformed/oversized requests
(mandate 2026-09-09 item 5)?

## Verdict: PASS — all three probes return predictable error envelopes

| probe | status | type | message |
|---|---|---|---|
| context overflow (>CTX prompt) | 500 | server_error | "the request exceeds the available context size, try increasing it" |
| invalid tool_choice ('bogus-choice') | 500 | server_error | "Invalid tool_choice: bogus-choice" |
| missing messages | 500 | server_error | "'messages' is required" |

Every response is a JSON {"error": {code, type, message}} envelope — the
shape Hermes can branch on. Note: this server classifies ALL three as
ERROR_TYPE_SERVER/500 (overflow at server-context.cpp:3983; the exceptions
from common/chat.cpp:263 and message validation surface the same way),
where OpenAI would use 400. Predictable, but the Hermes client must map
these by MESSAGE (context-size vs invalid-parameter), not by status class.

## Iteration history (fail-closed, preserved)

1. Run 132751: first probe (overflow) returned 500; the gate's initial
   4xx-only assumption aborted the run fail-closed. Preserved.
2. 8d3fcff3: assumption corrected after source verification (the
   classification is deliberate in this server build); gate now accepts
   4xx/5xx envelopes, still rejects 2xx/3xx.
3. Run 133326 (counted): all three probes validated.

## Limitations

- Three probes, one request each — not an exhaustive invalid-input matrix.
- No streaming error path yet (SSE mid-stream error frame) — future slice.
- Cosmetic: scope-stop needed >10 s twice (32 GiB mmap teardown); SIGKILL
  escalation verified the scope gone both times; harness widened to a 60 s
  stop deadline (tests still green).

## Artifacts

- errors-outcome.json, runs 132751 (aborted first attempt) and 133326
  (counted). Harness commits: 597c0da5, 8d3fcff3, 687b6a1b + the 60 s
  scope-stop fix.
