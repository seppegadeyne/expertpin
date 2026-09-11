# Review 2026-09-11 — tool-call quality gate slice

Two independent review attempts against the pinned reviewer
(openai-codex/gpt-6-astra, delegation.reasoning_effort=max):

- deleg_1901d667 (09:55): FAILED — HTTP 429 usage limit after 3 retries, 8.8s.
- deleg_e1f13ec2 (09:59): FAILED — HTTP 429 usage limit after 3 retries, 9.5s.
- deleg_a025517a (10:44, needle-gate addition): FAILED — HTTP 429 after 3
  retries, 10.9s.

Same failure mode as the 2026-09-11 03:00 run (deleg_31462b26, deleg_950dab5e).
Not treated as a blocker; per standing protocol the parent self-verified the
core premises in the server/launcher sources before the GPU run:

1. Launcher honors external MODEL/MODEL_DIR: scripts/run-qwen38-flash-next.sh
   line 37-38 reads ${MODEL:-...}; line 135 passes -m "$MODEL" verbatim; the
   harness clean_environment sets MODEL/MODEL_DIR via env.update (always wins).
2. Assistant message with content:null + tool_calls is accepted by the OAI
   chat parser: examples/server/server-common.cpp:704-714 (null content falls
   through to continue).
3. tool_choice string parsing exists only for "auto"/"none"/"required"
   (common/chat.cpp:253-263); the OpenAI object form {"type":"function",...}
   is read via json_value(..., std::string) and silently degrades to "auto"
   (examples/server/server-common.h:129-140 type_error fallback + server-common.cpp:651).
   => the gate sends the STRING "required"; with exactly one tool bound this
   is equivalent to forcing get_weather (grammar min_calls=1,
   common/chat.cpp:921 and 1063).
4. finish_reason "tool_calls" is emitted when parsed tool calls exist
   (examples/server/server-task.cpp:374, 416).
5. Thinking-model budget: reasoning before the tool call is expected, so
   max_tokens is 512 per round (validate fails closed on finish_reason
   'length').
6. Round-2 tool message format {"role":"tool","tool_call_id":...,"content":...}
   follows the same shape the server itself emits when echoing tool results
   (server-chat.cpp:385-389).

No second independent opinion this run; the review pin stays unchanged.

## Needle-gate premises (self-verified 10:45, third 429-blocked review)

7. /tokenize endpoint exists and returns {'tokens': [...]} — same shape the
   harness already uses in capture_tokenizations (POST tokenize with
   {'content', 'add_special'}); server.cpp registers POST /tokenize.
8. Question naming Aurora is intended single-needle NIAH design; filler
   paragraphs contain no 'Aurora' and no digits beyond their index numbers,
   so '7391' occurs exactly once per haystack (asserted by tests).
9. Prefix-cache claim corrected in report.md: end->middle->start shares the
   raw filler prefix, but the inserted needle (90% vs 50%) breaks the common
   prefix at the insertion point, so cache_prompt reuse is partial at best —
   an optimization note, not a correctness claim.
10. Budgets: 2048-token prefill at measured cold ~38 tok/s (ps) ≈ 54 s and
    ~11 tok/s (reference) ≈ 186 s worst-case first request; request budget
    300 s, work budget 960 s, alarm 960 s — all under the 30-min GPU cap.
    NEEDLE_TOKENS=256 (thinking model: reasoning must fit before the answer;
    64 risked spurious finish_reason='length').

