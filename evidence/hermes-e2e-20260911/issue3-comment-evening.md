Progress update 2026-09-11 (evening) — multi-turn agent E2E PASS, SSE error contract pinned, dev-serving recipe documented. The three remaining roadmap items are done.

1. Multi-turn agent E2E (evidence/hermes-e2e-20260911/multiturn-outcome.json): turn 1 creates a marker file via the shell tool (62.5 s); turn 2 resumes the session (`--resume latest`) and cats that exact artifact, answering with its content (4.4 s). Session 20260911_164157_a98121 totals 8 messages / 4 tool calls across both turns — session continuity and reuse of the agent's own artifact are proven, not just two independent queries. Gotcha found: `-c` is continue-by-TITLE, not continue-latest.

2. SSE mid-stream error contract (tests/test-sse-midstream-error-20260911.py): source-verified in examples/server/server.cpp — an error BEFORE the first chunk returns a normal HTTP error response (:1215); an error AFTER the first chunk is emitted as one `data: {"error": {...}}` SSE frame and the stream ends WITHOUT [DONE] (:1264-1267). A physical mid-stream failure cannot be triggered deterministically without engine fault injection, so the client-side expectation is pinned with 3 synthetic-frame tests (fail closed on error frames and truncated streams). Physically NOT TESTED — honestly labeled.

3. Dev-serving recipe (docs/dev-serving-recipe.md): the E2E-proven integration consolidated — guarded serving with CTX>=65536, the classic custom-provider config (the model_aliases path does NOT feed default-provider startup), the three gotchas that each cost an iteration, measured agent timings (63 s turn 1 / 4 s resumed turn 2), and the remaining follow-up (resident systemd unit with keep-alive).

Commits: 0d306ff4, e8147746, 6e3ea201, 56c6e288, 301aa742, b601b793 — all pushed and read back exactly. Targeted ctest 7/7. Reviewer gpt-6-astra untouched per the usage-limit hold.

Remaining in this issue: the resident systemd dev-serving unit (wrapping scripts/run-qwen38-flash-next.sh with the same cgroup caps) and longer agent trajectories.
